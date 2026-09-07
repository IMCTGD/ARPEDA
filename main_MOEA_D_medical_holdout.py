#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_MOEA_D_medical.py

ARPEDA 医学数据集运行脚本：支持 sMRI 和 EEG。

放置位置建议：
    /home/caohaonian/ARPEDA/main_MOEA_D_medical.py

依赖同原始项目，仍然复用：
    module/fitness_function3.py
    module/discretization_result_LM_v3.py
    module/compute_CCC.py 等

核心改动：
1) 新增 sMRI / EEG 数据加载函数；
2) sMRI 支持预划分 5-fold CSV，例如 norm-SZvsHC-ALLFeatures-5-1tra.csv / 1val.csv；
3) EEG 支持 Feature/*.npy + Label/label.npy，并按 subject-level StratifiedKFold 划分，避免 epoch-level leakage；
4) n_jobs 参数控制 MOEA/D fitness、分类器调参、分类器评估的 CPU 并行；
5) 输出每折结果、Pareto Front、CART/SVM 医学汇总指标 Accuracy/F1/AUC/Cutpoints。
"""

import argparse
import itertools
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from deap import base, creator, tools
from joblib import Parallel, delayed

from module.discretization_result_LM_v3 import discretization_result
from module.fitness_function3 import fitness_function


# -----------------------------
# 全局变量由 argparse 初始化
# -----------------------------
ARGS = None
CLASSIFIERS = {}
EEG_CACHE = {}


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "medical_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


# -----------------------------
# 数据读取：通用工具
# -----------------------------
def _read_csv_auto(path: Path) -> pd.DataFrame:
    """尽量兼容有表头/无表头 CSV。"""
    try:
        df = pd.read_csv(path)
        # 如果第一行被误当表头，且列名不是数值/常规特征名，一般仍不影响；
        # 若读出来空或列过少，再用 header=None。
        if df.shape[1] <= 1:
            df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=None)
    return df


def _encode_labels(y) -> np.ndarray:
    y = np.asarray(y).ravel()
    if not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y.astype(str))
    return y.astype(int)


def _clean_X(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _split_label_feature(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """label_col='first' 或 'last'。返回 X, y。"""
    if label_col == "first":
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
    elif label_col == "last":
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        raise ValueError("label_col 只能是 first 或 last")
    return _clean_X(X), _encode_labels(y)


def _find_smri_fold_files(smri_dir: Path, fold_idx: int) -> Tuple[Path, Path]:
    """寻找 sMRI 第 fold_idx 折的 train/val 文件。"""
    if not smri_dir.exists():
        raise FileNotFoundError(f"sMRI 目录不存在: {smri_dir}")

    all_csv = list(smri_dir.glob("*.csv")) + list(smri_dir.glob("*.CSV"))
    if not all_csv:
        raise FileNotFoundError(f"sMRI 目录中没有 CSV 文件: {smri_dir}")

    fold_tokens = [
        f"-{fold_idx}tra", f"_{fold_idx}tra", f"{fold_idx}tra",
        f"-{fold_idx}train", f"_{fold_idx}train", f"{fold_idx}train",
    ]
    val_tokens = [
        f"-{fold_idx}val", f"_{fold_idx}val", f"{fold_idx}val",
        f"-{fold_idx}tst", f"_{fold_idx}tst", f"{fold_idx}tst",
        f"-{fold_idx}test", f"_{fold_idx}test", f"{fold_idx}test",
    ]

    def match_any(path: Path, tokens: List[str]) -> bool:
        name = path.name.lower()
        return any(tok.lower() in name for tok in tokens)

    train_candidates = [p for p in all_csv if match_any(p, fold_tokens)]
    val_candidates = [p for p in all_csv if match_any(p, val_tokens)]

    if not train_candidates or not val_candidates:
        # 兼容 norm-SZvsHC-ALLFeatures-5-1tra.csv 这种形式
        train_candidates = [p for p in all_csv if f"{fold_idx}tra" in p.name.lower()]
        val_candidates = [p for p in all_csv if (f"{fold_idx}val" in p.name.lower() or f"{fold_idx}tst" in p.name.lower())]

    if not train_candidates or not val_candidates:
        raise FileNotFoundError(
            f"找不到 sMRI 第 {fold_idx} 折 train/val 文件。\n"
            f"目录: {smri_dir}\n"
            f"示例期望: *{fold_idx}tra.csv 和 *{fold_idx}val.csv"
        )

    return sorted(train_candidates)[0], sorted(val_candidates)[0]


def load_smri_fold(data_dir: Path, fold_idx: int, label_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """sMRI: 读取预先划分好的 5-fold CSV。默认标签在第一列。"""
    smri_dir = data_dir / "sMRI"
    train_file, val_file = _find_smri_fold_files(smri_dir, fold_idx)
    train_df = _read_csv_auto(train_file)
    val_df = _read_csv_auto(val_file)
    X_train, y_train = _split_label_feature(train_df, label_col=label_col)
    X_test, y_test = _split_label_feature(val_df, label_col=label_col)
    logging.info(f"sMRI fold {fold_idx}: train={train_file.name}, val={val_file.name}")
    return X_train, y_train, X_test, y_test


def _load_eeg_subjects(data_dir: Path) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """读取 EEG subject-level 特征。返回 subject_features, subject_labels, subject_ids。"""
    eeg_dir = data_dir / "EEG"
    feature_dir = eeg_dir / "Feature"
    label_dir = eeg_dir / "Label"
    label_file = label_dir / "label.npy"

    if not feature_dir.exists():
        raise FileNotFoundError(f"EEG Feature 目录不存在: {feature_dir}")
    if not label_file.exists():
        raise FileNotFoundError(f"EEG label.npy 不存在: {label_file}")

    feature_files = sorted(feature_dir.glob("*.npy"))
    if not feature_files:
        raise FileNotFoundError(f"EEG Feature 目录中没有 .npy 文件: {feature_dir}")

    labels = np.load(label_file, allow_pickle=True)
    labels = np.asarray(labels)

    if labels.ndim == 1:
        subject_labels = labels
        subject_ids = np.arange(len(labels))
    else:
        # 约定：第一列为类别，第二列为 subject id。如果你的 label.npy 不同，可在这里改。
        subject_labels = labels[:, 0]
        subject_ids = labels[:, 1] if labels.shape[1] >= 2 else np.arange(labels.shape[0])

    subject_labels = _encode_labels(subject_labels)
    subject_ids = np.asarray(subject_ids)

    if len(feature_files) != len(subject_labels):
        raise ValueError(
            f"EEG 特征文件数({len(feature_files)}) 与 label 数({len(subject_labels)}) 不一致。\n"
            f"请确认 Feature/*.npy 是否为每个 subject 一个文件，并且顺序与 label.npy 对应。"
        )

    subject_features = []
    for f in feature_files:
        arr = np.load(f, allow_pickle=True)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        arr = _clean_X(arr)
        subject_features.append(arr)

    logging.info(
        f"EEG loaded: subjects={len(subject_features)}, "
        f"feature_dim={subject_features[0].shape[1]}, labels={np.unique(subject_labels, return_counts=True)}"
    )
    return subject_features, subject_labels, subject_ids


def _get_eeg_cache(data_dir: Path):
    key = str(data_dir.resolve())
    if key not in EEG_CACHE:
        EEG_CACHE[key] = _load_eeg_subjects(data_dir)
    return EEG_CACHE[key]


def load_eeg_fold(data_dir: Path, fold_idx: int, n_folds: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EEG: subject-level StratifiedKFold；每个 subject 的所有 epoch 只进同一侧。"""
    subject_features, subject_labels, subject_ids = _get_eeg_cache(data_dir)

    min_class = np.min(np.bincount(subject_labels))
    n_splits = min(n_folds, int(min_class))
    if n_splits < 2:
        raise ValueError(f"EEG 每类 subject 数太少，无法做 StratifiedKFold: {np.bincount(subject_labels)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(subject_labels)), subject_labels))
    if fold_idx > len(splits):
        raise ValueError(f"EEG fold_idx={fold_idx} 超过实际 n_splits={len(splits)}")

    train_subject_idx, test_subject_idx = splits[fold_idx - 1]

    X_train = np.vstack([subject_features[i] for i in train_subject_idx])
    y_train = np.concatenate([np.full(subject_features[i].shape[0], subject_labels[i], dtype=int) for i in train_subject_idx])
    X_test = np.vstack([subject_features[i] for i in test_subject_idx])
    y_test = np.concatenate([np.full(subject_features[i].shape[0], subject_labels[i], dtype=int) for i in test_subject_idx])

    train_ids = subject_ids[train_subject_idx]
    test_ids = subject_ids[test_subject_idx]
    overlap = set(map(str, train_ids)).intersection(set(map(str, test_ids)))
    if overlap:
        raise RuntimeError(f"EEG subject leakage detected: {overlap}")

    logging.info(
        f"EEG fold {fold_idx}/{len(splits)}: train_subjects={len(train_subject_idx)}, "
        f"test_subjects={len(test_subject_idx)}, X_train={X_train.shape}, X_test={X_test.shape}"
    )
    return X_train, y_train, X_test, y_test


def _find_smri_test_file(smri_dir: Path) -> Path:
    """寻找 sMRI 独立测试集文件，例如 *tst.csv / *test.csv。"""
    if not smri_dir.exists():
        raise FileNotFoundError(f"sMRI 目录不存在: {smri_dir}")
    all_csv = list(smri_dir.glob("*.csv")) + list(smri_dir.glob("*.CSV"))
    candidates = [
        p for p in all_csv
        if any(tok in p.name.lower() for tok in ["tst", "test", "independent", "holdout"])
        and not any(f"-{i}tst" in p.name.lower() or f"_{i}tst" in p.name.lower() for i in range(1, 10))
    ]
    if not candidates:
        # 兼容你当前的 norm-SZvsHC-ALLFeatures-tst.csv
        candidates = [p for p in all_csv if "tst" in p.name.lower() or "test" in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"未找到 sMRI 独立测试集文件，请确认是否存在 *tst.csv 或 *test.csv: {smri_dir}")
    return sorted(candidates)[0]


def load_smri_holdout(data_dir: Path, label_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """sMRI 论文式独立测试协议：用某一折的 tra+val 合并为训练全集，使用 *tst.csv 作为独立测试集。"""
    smri_dir = data_dir / "sMRI"
    train_file, val_file = _find_smri_fold_files(smri_dir, 1)
    test_file = _find_smri_test_file(smri_dir)

    train_df = _read_csv_auto(train_file)
    val_df = _read_csv_auto(val_file)
    test_df = _read_csv_auto(test_file)

    X_tr, y_tr = _split_label_feature(train_df, label_col=label_col)
    X_val, y_val = _split_label_feature(val_df, label_col=label_col)
    X_test, y_test = _split_label_feature(test_df, label_col=label_col)

    X_train = np.vstack([X_tr, X_val])
    y_train = np.concatenate([y_tr, y_val])

    logging.info(
        f"sMRI holdout: train={train_file.name}+{val_file.name}, test={test_file.name}; "
        f"X_train={X_train.shape}, X_test={X_test.shape}"
    )
    return X_train, y_train, X_test, y_test


def load_eeg_holdout(data_dir: Path, seed: int, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EEG 论文式 subject-independent holdout：先按 subject 分出 20% 独立测试集。"""
    subject_features, subject_labels, subject_ids = _get_eeg_cache(data_dir)
    all_idx = np.arange(len(subject_labels))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=seed,
        stratify=subject_labels,
    )

    X_train = np.vstack([subject_features[i] for i in train_idx])
    y_train = np.concatenate([np.full(subject_features[i].shape[0], subject_labels[i], dtype=int) for i in train_idx])
    X_test = np.vstack([subject_features[i] for i in test_idx])
    y_test = np.concatenate([np.full(subject_features[i].shape[0], subject_labels[i], dtype=int) for i in test_idx])

    train_ids = subject_ids[train_idx]
    test_ids = subject_ids[test_idx]
    overlap = set(map(str, train_ids)).intersection(set(map(str, test_ids)))
    if overlap:
        raise RuntimeError(f"EEG holdout subject leakage detected: {overlap}")

    logging.info(
        f"EEG holdout: train_subjects={len(train_idx)}, test_subjects={len(test_idx)}, "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"train_classes={np.unique(y_train, return_counts=True)}, test_classes={np.unique(y_test, return_counts=True)}"
    )
    return X_train, y_train, X_test, y_test


def load_fold_data(dataset_name: str, fold_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """医学数据统一入口。"""
    try:
        data_dir = Path(ARGS.data_dir)
        ds = dataset_name.lower()
        if ds == "smri":
            X_train, y_train, X_test, y_test = load_smri_fold(data_dir, fold_idx, ARGS.smri_label_col)
        elif ds == "eeg":
            X_train, y_train, X_test, y_test = load_eeg_fold(data_dir, fold_idx, ARGS.n_folds, ARGS.seed)
        else:
            raise ValueError(f"未知医学数据集: {dataset_name}，只支持 sMRI / EEG")

        if ARGS.standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        X_train = np.copy(_clean_X(X_train))
        X_test = np.copy(_clean_X(X_test))
        y_train = np.copy(_encode_labels(y_train))
        y_test = np.copy(_encode_labels(y_test))

        X_train.setflags(write=True)
        X_test.setflags(write=True)
        y_train.setflags(write=True)
        y_test.setflags(write=True)

        logging.info(
            f"成功加载 {dataset_name} fold {fold_idx}: "
            f"X_train={X_train.shape}, X_test={X_test.shape}, "
            f"train_classes={np.unique(y_train, return_counts=True)}, "
            f"test_classes={np.unique(y_test, return_counts=True)}"
        )
        return X_train, y_train, X_test, y_test, True
    except Exception as e:
        logging.exception(f"加载数据集 {dataset_name} fold {fold_idx} 失败: {e}")
        return None, None, None, None, False


# -----------------------------
# MOEA/D 主体
# -----------------------------
def initialize_population(num_features: int, population_size: int, max_levels: int, diversity_factor: float = 0.5) -> list:
    population = []
    heuristic_count = int(population_size * diversity_factor)
    for i in range(population_size):
        if i < heuristic_count:
            upper = min(5, max_levels + 1)
            individual = np.random.randint(ARGS.l_min, upper, size=num_features)
        else:
            individual = np.random.randint(ARGS.l_min, max_levels + 1, size=num_features)
        population.append(individual.tolist())
    return population


def discretize_data(X: np.ndarray, levels: list) -> tuple:
    X = _clean_X(X)
    if X.ndim != 2:
        raise ValueError("X 必须是二维 NumPy 数组")
    if len(levels) != X.shape[1]:
        raise ValueError(f"levels 长度 {len(levels)} 必须等于特征数 {X.shape[1]}")

    X_discretized = np.zeros_like(X, dtype=float)
    total_cutpoints = 0
    for i in range(X.shape[1]):
        L = int(levels[i])
        if L <= 1:
            X_discretized[:, i] = 1.0
        else:
            try:
                X_discretized[:, i], _ = discretization_result(X[:, i], L)
                total_cutpoints += L - 1
            except Exception as e:
                raise ValueError(f"特征 {i} 离散化失败: {e}")
    return X_discretized, total_cutpoints


def setup_toolbox(num_features: int, max_levels: int) -> base.Toolbox:
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, ARGS.l_min, max_levels)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=ARGS.l_min, up=max_levels, indpb=ARGS.gene_mutation_prob)
    return toolbox


def generate_simplex_lattice(n_obj: int, p: int) -> np.ndarray:
    weights = []
    for comb in itertools.product(range(p + 1), repeat=n_obj):
        if sum(comb) == p:
            weights.append([c / p for c in comb])
    return np.array(weights, dtype=float)


def _evaluate_fitness_wrapper(individual, X, y, X_val, y_val):
    individual_array = np.array(individual, dtype=int)
    train_data = np.hstack((y[:, np.newaxis], X))
    val_data = np.hstack((y_val[:, np.newaxis], X_val))
    return tuple(fitness_function(individual_array, train_data, val_data))


def run_moead(X_train: np.ndarray, y_train: np.ndarray, max_levels: int, generations: int,
              crossover_rate: float, mutation_rate: float):
    # 内部验证集仅用于适应度函数，不接触外部测试 fold。
    stratify = y_train if len(np.unique(y_train)) > 1 and np.min(np.bincount(y_train)) >= 2 else None
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=ARGS.inner_val_size, random_state=ARGS.seed, stratify=stratify
    )

    num_features = X_train_sub.shape[1]
    toolbox = setup_toolbox(num_features, max_levels)

    weights = generate_simplex_lattice(n_obj=3, p=ARGS.p_division)
    population_size = len(weights)
    logging.info(f"MOEA/D subproblems/population size={population_size}, p_division={ARGS.p_division}")

    pop = initialize_population(num_features, population_size, max_levels)
    pop = [creator.Individual(ind) for ind in pop]

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = Parallel(n_jobs=ARGS.n_jobs, prefer=ARGS.parallel_backend)(
        delayed(_evaluate_fitness_wrapper)(ind, X_train_sub, y_train_sub, X_val, y_val)
        for ind in invalid_ind
    )
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    z = np.min([ind.fitness.values for ind in pop], axis=0)
    dists = cdist(weights, weights, "euclidean")
    T = min(ARGS.neighborhood_size, population_size - 1)
    neighbors = [np.argsort(d)[1:T + 1].tolist() for d in dists]
    logging.info(f"MOEA/D neighborhood size T={T}")

    delta = ARGS.delta
    for gen in range(generations):
        if gen % ARGS.log_every == 0:
            logging.info(f"MOEA/D generation {gen + 1}/{generations}")

        offspring = []
        for i in range(population_size):
            B = neighbors[i] if random.random() < delta else list(range(population_size))
            if len(B) < 2:
                continue
            k, l = random.sample(B, 2)
            child1 = toolbox.clone(pop[k])
            child2 = toolbox.clone(pop[l])
            if random.random() < crossover_rate:
                toolbox.mate(child1, child2)
            child = random.choice([child1, child2])
            if random.random() < mutation_rate:
                toolbox.mutate(child)
            if child.fitness.valid:
                del child.fitness.values
            offspring.append(child)

        fitnesses = Parallel(n_jobs=ARGS.n_jobs, prefer=ARGS.parallel_backend)(
            delayed(_evaluate_fitness_wrapper)(ind, X_train_sub, y_train_sub, X_val, y_val)
            for ind in offspring
        )
        for child, fit in zip(offspring, fitnesses):
            child.fitness.values = fit

        for i in range(min(population_size, len(offspring))):
            child = offspring[i]
            fit = np.array(child.fitness.values, dtype=float)
            z = np.minimum(z, fit)
            B = neighbors[i] if random.random() < delta else list(range(population_size))
            for j in B:
                g_new = np.max(weights[j] * np.abs(fit - z))
                g_old = np.max(weights[j] * np.abs(np.array(pop[j].fitness.values) - z))
                if g_new <= g_old:
                    pop[j] = toolbox.clone(child)

    pareto_front = tools.sortNondominated(pop, population_size, first_front_only=True)[0]
    return pop, pareto_front


# -----------------------------
# Pareto 解选择与评估
# -----------------------------
def select_solution_from_pareto(pareto_front, rule: str, eps: float) -> int:
    fits = np.array([ind.fitness.values for ind in pareto_front], dtype=float)
    fits = np.nan_to_num(fits, nan=1e6, posinf=1e6, neginf=1e6)

    if rule == "argmin_f2_only":
        return int(np.argmin(fits[:, 1]))

    if rule == "paper":
        # 分类错误率最小；并列时 CIS 更小；再并列 interval complexity 更小。
        order = np.lexsort((fits[:, 2], fits[:, 0], fits[:, 1]))
        return int(order[0])

    if rule == "epsilon_min_cutpoints":
        # 在接近最佳分类误差的候选中，优先选复杂度小的解。
        best_f2 = np.min(fits[:, 1])
        candidates = np.where(fits[:, 1] <= best_f2 + eps)[0]
        if candidates.size == 0:
            candidates = np.array([int(np.argmin(fits[:, 1]))])
        cand_fits = fits[candidates]
        order = np.lexsort((cand_fits[:, 0], cand_fits[:, 1], cand_fits[:, 2]))
        return int(candidates[order[0]])

    if rule == "knee":
        mins = fits.min(axis=0)
        maxs = fits.max(axis=0)
        norm = (fits - mins) / np.maximum(maxs - mins, 1e-12)
        return int(np.argmin(np.linalg.norm(norm, axis=1)))

    raise ValueError(f"未知 selection_rule: {rule}")


def _safe_cv(y_train, max_cv: int = 5) -> int:
    counts = np.bincount(_encode_labels(y_train))
    min_count = int(np.min(counts)) if counts.size > 0 else 0
    return max(2, min(max_cv, min_count))


def _make_classifier(name: str, params: Dict):
    if name == "SVM":
        p = dict(params)
        p.setdefault("probability", True)
        p.setdefault("random_state", ARGS.seed)
        return SVC(**p)
    if name == "CART":
        p = dict(params)
        p.setdefault("random_state", ARGS.seed)
        return DecisionTreeClassifier(**p)
    if name == "KNN":
        return KNeighborsClassifier(**params)
    raise ValueError(f"未知分类器: {name}")


def evaluate_individual(X_train_discretized: np.ndarray, y_train: np.ndarray,
                        X_test_discretized: np.ndarray, y_test: np.ndarray,
                        classifier_name: str) -> dict:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"Accuracy": 0.0, "F1_weighted": 0.0, "AUC": np.nan, "Best_Params": {}}

    param_grid = CLASSIFIERS[classifier_name]["params"]
    param_combinations = [dict(zip(param_grid.keys(), vals)) for vals in itertools.product(*param_grid.values())]
    cv = _safe_cv(y_train, max_cv=5)

    def score_param(param):
        try:
            clf = _make_classifier(classifier_name, param)
            scores = cross_val_score(clf, X_train_discretized, y_train, cv=cv, scoring="accuracy", n_jobs=1)
            return float(np.mean(scores)), param
        except Exception as e:
            logging.warning(f"{classifier_name} 参数 {param} CV失败: {e}")
            return -np.inf, param

    param_results = Parallel(n_jobs=ARGS.n_jobs, prefer=ARGS.parallel_backend)(
        delayed(score_param)(param) for param in param_combinations
    )
    best_score, best_params = max(param_results, key=lambda x: x[0])
    if best_params is None or not np.isfinite(best_score):
        return {"Accuracy": 0.0, "F1_weighted": 0.0, "AUC": np.nan, "Best_Params": {}}

    clf = _make_classifier(classifier_name, best_params)
    try:
        clf.fit(X_train_discretized, y_train)
        y_pred = clf.predict(X_test_discretized)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        auc = np.nan
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test_discretized)
            classes = np.unique(np.concatenate([y_train, y_test]))
            if len(classes) == 2 and proba.shape[1] >= 2:
                auc = float(roc_auc_score(y_test, proba[:, 1]))
            elif len(classes) > 2 and proba.shape[1] == len(clf.classes_):
                auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
    except Exception as e:
        logging.warning(f"{classifier_name} 测试评估失败: {e}")
        acc, f1, auc = 0.0, 0.0, np.nan

    return {"Accuracy": acc, "F1_weighted": f1, "AUC": auc, "Best_Params": best_params}


def _maybe_select_features(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, top_k: int):
    if top_k is None or top_k <= 0 or top_k >= X_train.shape[1]:
        return X_train, X_test, None
    try:
        scores, _ = f_classif(X_train, y_train)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        selected = np.argsort(scores)[-top_k:]
        selected = np.sort(selected)
        logging.warning(f"启用 feature_top_k={top_k}，仅用于调试/快速实验；正式论文结果需说明特征筛选。")
        return X_train[:, selected], X_test[:, selected], selected
    except Exception as e:
        logging.warning(f"feature_top_k 失败，使用全部特征: {e}")
        return X_train, X_test, None


# -----------------------------
# 交叉验证与结果保存
# -----------------------------
def perform_cross_validation(dataset_name: str, output_dir: Path, results_summary: dict) -> pd.DataFrame:
    results = []
    max_levels = ARGS.max_levels
    n_folds = 1 if ARGS.smoke else ARGS.n_folds
    if ARGS.max_folds is not None:
        n_folds = min(n_folds, ARGS.max_folds)
    generations = 1 if ARGS.smoke else ARGS.generations

    pareto_dir = output_dir / "pareto_fronts"
    pareto_dir.mkdir(parents=True, exist_ok=True)

    for fold_num in range(1, n_folds + 1):
        logging.info(f"\n数据集 {dataset_name} 第 {fold_num}/{n_folds} 折")
        X_train, y_train, X_test, y_test, success = load_fold_data(dataset_name, fold_num)
        fold_result = {"Fold": fold_num}
        if not success:
            logging.warning(f"{dataset_name} fold {fold_num} 加载失败，跳过")
            continue

        X_train, X_test, selected_features = _maybe_select_features(X_train, y_train, X_test, ARGS.feature_top_k)

        _, pareto_front = run_moead(
            X_train, y_train,
            max_levels=max_levels,
            generations=generations,
            crossover_rate=ARGS.crossover_rate,
            mutation_rate=ARGS.mutation_rate,
        )

        pareto_records = []
        for idx, ind in enumerate(pareto_front):
            f1, f2, f3 = ind.fitness.values
            pareto_records.append({
                "Solution": idx + 1,
                "NDCG_Loss": float(f1),
                "Classification_Error": float(f2),
                "Complexity": float(f3),
                "Genome": ",".join(map(str, ind)),
            })
        pareto_file = pareto_dir / f"{dataset_name}_fold{fold_num}_pareto.csv"
        pd.DataFrame(pareto_records).to_csv(pareto_file, index=False)
        logging.info(f"Pareto Front saved: {pareto_file}, Solutions={len(pareto_records)}")

        best_idx = select_solution_from_pareto(pareto_front, ARGS.selection_rule, ARGS.epsilon)
        best_individual = list(map(int, pareto_front[best_idx][:]))
        selected_fit = pareto_front[best_idx].fitness.values

        discretized_train, total_cutpoints = discretize_data(X_train, best_individual)
        discretized_test, _ = discretize_data(X_test, best_individual)

        # L=1 特征不参与后续分类。
        collapsed = np.where(np.array(best_individual) <= 1)[0]
        if collapsed.size > 0:
            discretized_train = np.delete(discretized_train, collapsed, axis=1)
            discretized_test = np.delete(discretized_test, collapsed, axis=1)
        if discretized_train.shape[1] == 0:
            logging.warning("所有特征都被折叠为 L=1，分类评估将返回无效值。")

        fold_result.update({
            "Best_Individual": str(best_individual),
            "Interval_Sum": int(sum(best_individual)),
            "Cutpoints_From_Individual": int(sum(max(x - 1, 0) for x in best_individual)),
            "Cutpoints": int(total_cutpoints),
            "Selected_F1_CISLoss": float(selected_fit[0]),
            "Selected_F2_ClsLoss": float(selected_fit[1]),
            "Selected_F3_Complexity": float(selected_fit[2]),
            "Pareto_Front_Size": len(pareto_front),
            "Selected_MAX_LEVELS": max_levels,
            "Generations_Used": generations,
            "Selection_Rule": ARGS.selection_rule,
            "Epsilon": ARGS.epsilon,
            "L_Min": ARGS.l_min,
            "Feature_Top_K": ARGS.feature_top_k if ARGS.feature_top_k else "all",
            "Selected_Feature_Count": discretized_train.shape[1],
        })

        def eval_one(clf_name):
            if discretized_train.shape[1] == 0:
                return clf_name, {"Accuracy": 0.0, "F1_weighted": 0.0, "AUC": np.nan, "Best_Params": {}}
            return clf_name, evaluate_individual(discretized_train, y_train, discretized_test, y_test, clf_name)

        clf_results = Parallel(n_jobs=min(ARGS.n_jobs, len(CLASSIFIERS)), prefer=ARGS.parallel_backend)(
            delayed(eval_one)(clf_name) for clf_name in CLASSIFIERS.keys()
        )
        for clf_name, metrics in clf_results:
            fold_result[f"{clf_name}_Accuracy"] = metrics["Accuracy"]
            fold_result[f"{clf_name}_F1_weighted"] = metrics["F1_weighted"]
            fold_result[f"{clf_name}_AUC"] = metrics["AUC"]
            fold_result[f"{clf_name}_Best_Params"] = str(metrics.get("Best_Params", {}))
            logging.info(
                f"fold {fold_num} {clf_name}: Acc={metrics['Accuracy']:.4f}, "
                f"F1={metrics['F1_weighted']:.4f}, AUC={metrics['AUC']}"
            )

        results.append(fold_result)

    results_df = pd.DataFrame(results)
    per_fold_file = output_dir / f"{dataset_name}_medical_moead_classifier_results.csv"
    results_df.to_csv(per_fold_file, index=False)
    logging.info(f"每折结果保存: {per_fold_file}")

    for clf_name in CLASSIFIERS.keys():
        if results_df.empty:
            continue
        rec = {
            "Dataset": dataset_name,
            "Classifier": clf_name,
            "Mean_Accuracy": results_df[f"{clf_name}_Accuracy"].mean(),
            "Std_Accuracy": results_df[f"{clf_name}_Accuracy"].std(),
            "Mean_F1_weighted": results_df[f"{clf_name}_F1_weighted"].mean(),
            "Std_F1_weighted": results_df[f"{clf_name}_F1_weighted"].std(),
            "Mean_AUC": results_df[f"{clf_name}_AUC"].mean(),
            "Std_AUC": results_df[f"{clf_name}_AUC"].std(),
            "Mean_Cutpoints": results_df["Cutpoints"].mean(),
            "Std_Cutpoints": results_df["Cutpoints"].std(),
            "Generations_Used": generations,
            "MOEAD_Subproblems": len(generate_simplex_lattice(3, ARGS.p_division)),
            "Selection_Rule": ARGS.selection_rule,
            "Epsilon": ARGS.epsilon,
            "L_Min": ARGS.l_min,
            "Max_Levels": ARGS.max_levels,
            "Feature_Top_K": ARGS.feature_top_k if ARGS.feature_top_k else "all",
        }
        results_summary.setdefault(clf_name, []).append(rec)

    return results_df


def perform_independent_test(dataset_name: str, output_dir: Path, results_summary: dict) -> pd.DataFrame:
    """论文式医学独立测试协议：训练/内部验证用于搜索离散化方案，最终只在独立测试集评估一次。"""
    logging.info(f"\n数据集 {dataset_name} 独立测试协议 holdout")
    data_dir = Path(ARGS.data_dir)
    ds = dataset_name.lower()

    if ds == "smri":
        X_train, y_train, X_test, y_test = load_smri_holdout(data_dir, ARGS.smri_label_col)
    elif ds == "eeg":
        X_train, y_train, X_test, y_test = load_eeg_holdout(data_dir, ARGS.seed, ARGS.independent_test_size)
    else:
        raise ValueError(f"未知医学数据集: {dataset_name}")

    if ARGS.standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train = np.copy(_clean_X(X_train))
    X_test = np.copy(_clean_X(X_test))
    y_train = np.copy(_encode_labels(y_train))
    y_test = np.copy(_encode_labels(y_test))

    if ARGS.feature_top_k and ARGS.feature_top_k > 0:
        X_train, X_test, selected_features = _maybe_select_features(X_train, y_train, X_test, ARGS.feature_top_k)
    else:
        selected_features = None

    generations = 1 if ARGS.smoke else ARGS.generations
    pareto_dir = output_dir / "pareto_fronts"
    pareto_dir.mkdir(parents=True, exist_ok=True)

    _, pareto_front = run_moead(
        X_train, y_train,
        max_levels=ARGS.max_levels,
        generations=generations,
        crossover_rate=ARGS.crossover_rate,
        mutation_rate=ARGS.mutation_rate,
    )

    pareto_records = []
    for idx, ind in enumerate(pareto_front):
        f1, f2, f3 = ind.fitness.values
        pareto_records.append({
            "Solution": idx + 1,
            "NDCG_Loss": float(f1),
            "Classification_Error": float(f2),
            "Complexity": float(f3),
            "Genome": ",".join(map(str, ind)),
        })
    pareto_file = pareto_dir / f"{dataset_name}_holdout_pareto.csv"
    pd.DataFrame(pareto_records).to_csv(pareto_file, index=False)
    logging.info(f"Pareto Front saved: {pareto_file}, Solutions={len(pareto_records)}")

    best_idx = select_solution_from_pareto(pareto_front, ARGS.selection_rule, ARGS.epsilon)
    best_individual = list(map(int, pareto_front[best_idx][:]))
    selected_fit = pareto_front[best_idx].fitness.values

    discretized_train, total_cutpoints = discretize_data(X_train, best_individual)
    discretized_test, _ = discretize_data(X_test, best_individual)

    collapsed = np.where(np.array(best_individual) <= 1)[0]
    if collapsed.size > 0:
        discretized_train = np.delete(discretized_train, collapsed, axis=1)
        discretized_test = np.delete(discretized_test, collapsed, axis=1)

    result = {
        "Dataset": dataset_name,
        "Protocol": "holdout",
        "Train_Samples": int(X_train.shape[0]),
        "Test_Samples": int(X_test.shape[0]),
        "Original_Feature_Count": int(X_train.shape[1]),
        "Selected_Feature_Count": int(discretized_train.shape[1]),
        "Best_Individual": str(best_individual),
        "Interval_Sum": int(sum(best_individual)),
        "Cutpoints_From_Individual": int(sum(max(x - 1, 0) for x in best_individual)),
        "Cutpoints": int(total_cutpoints),
        "Selected_F1_CISLoss": float(selected_fit[0]),
        "Selected_F2_ClsLoss": float(selected_fit[1]),
        "Selected_F3_Complexity": float(selected_fit[2]),
        "Pareto_Front_Size": len(pareto_front),
        "Generations_Used": generations,
        "MOEAD_Subproblems": len(generate_simplex_lattice(3, ARGS.p_division)),
        "Selection_Rule": ARGS.selection_rule,
        "Epsilon": ARGS.epsilon,
        "L_Min": ARGS.l_min,
        "Max_Levels": ARGS.max_levels,
        "Feature_Top_K": ARGS.feature_top_k if ARGS.feature_top_k else "all",
    }

    def eval_one(clf_name):
        if discretized_train.shape[1] == 0:
            return clf_name, {"Accuracy": 0.0, "F1_weighted": 0.0, "AUC": np.nan, "Best_Params": {}}
        return clf_name, evaluate_individual(discretized_train, y_train, discretized_test, y_test, clf_name)

    clf_results = Parallel(n_jobs=min(ARGS.n_jobs, len(CLASSIFIERS)), prefer=ARGS.parallel_backend)(
        delayed(eval_one)(clf_name) for clf_name in CLASSIFIERS.keys()
    )
    for clf_name, metrics in clf_results:
        result[f"{clf_name}_Accuracy"] = metrics["Accuracy"]
        result[f"{clf_name}_F1_weighted"] = metrics["F1_weighted"]
        result[f"{clf_name}_AUC"] = metrics["AUC"]
        result[f"{clf_name}_Best_Params"] = str(metrics.get("Best_Params", {}))
        logging.info(
            f"holdout {dataset_name} {clf_name}: Acc={metrics['Accuracy']:.4f}, "
            f"F1={metrics['F1_weighted']:.4f}, AUC={metrics['AUC']}"
        )

        results_summary.setdefault(clf_name, []).append({
            "Dataset": dataset_name,
            "Classifier": clf_name,
            "Protocol": "holdout",
            "Accuracy": metrics["Accuracy"],
            "F1_weighted": metrics["F1_weighted"],
            "AUC": metrics["AUC"],
            "Cutpoints": int(total_cutpoints),
            "Train_Samples": int(X_train.shape[0]),
            "Test_Samples": int(X_test.shape[0]),
            "Generations_Used": generations,
            "MOEAD_Subproblems": len(generate_simplex_lattice(3, ARGS.p_division)),
            "Selection_Rule": ARGS.selection_rule,
            "Epsilon": ARGS.epsilon,
            "L_Min": ARGS.l_min,
            "Max_Levels": ARGS.max_levels,
            "Feature_Top_K": ARGS.feature_top_k if ARGS.feature_top_k else "all",
        })

    df = pd.DataFrame([result])
    out = output_dir / f"{dataset_name}_medical_holdout_results.csv"
    df.to_csv(out, index=False)
    logging.info(f"独立测试结果保存: {out}")
    return df


def configure_classifiers(names: List[str]) -> Dict:
    all_cls = {
        "SVM": {
            "model": SVC,
            "params": {
                "kernel": ["rbf", "linear"],
                "C": [0.1, 1.0, 10.0],
                "probability": [True],
            },
        },
        "CART": {
            "model": DecisionTreeClassifier,
            "params": {
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier,
            "params": {"n_neighbors": [3, 5, 7]},
        },
    }
    selected = {}
    for n in names:
        key = n.strip()
        if not key:
            continue
        if key not in all_cls:
            raise ValueError(f"未知 classifier: {key}，可选 {list(all_cls)}")
        selected[key] = all_cls[key]
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Run ARPEDA on sMRI / EEG medical datasets with CPU parallelism.")
    parser.add_argument("--data_dir", type=str, required=True, help="数据根目录，下面应包含 sMRI/ 和 EEG/。")
    parser.add_argument("--output_dir", type=str, required=True, help="结果输出目录。")
    parser.add_argument("--datasets", type=str, default="sMRI,EEG", help="逗号分隔，例如 sMRI,EEG。")
    parser.add_argument("--n_folds", type=int, default=5, help="医学数据默认 5-fold。")
    parser.add_argument("--protocol", choices=["cv", "holdout"], default="cv", help="cv=原5折平均；holdout=论文式独立测试集评估。")
    parser.add_argument("--independent_test_size", type=float, default=0.2, help="EEG holdout 独立测试 subject 比例，论文为 0.2。")
    parser.add_argument("--max_folds", type=int, default=None, help="只跑前 N 折，调试用。")
    parser.add_argument("--smri_label_col", choices=["first", "last"], default="first", help="sMRI CSV 标签列位置，默认 first。")

    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--max_levels", type=int, default=15)
    parser.add_argument("--l_min", type=int, default=2, help="为对齐论文医学实验，默认 2；如需允许属性折叠可设 1。")
    parser.add_argument("--p_division", type=int, default=19, help="三目标 simplex-lattice 划分参数，19 对应 210 个子问题。")
    parser.add_argument("--neighborhood_size", type=int, default=20)
    parser.add_argument("--crossover_rate", type=float, default=0.8)
    parser.add_argument("--mutation_rate", type=float, default=0.2)
    parser.add_argument("--gene_mutation_prob", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.9)
    parser.add_argument("--inner_val_size", type=float, default=0.2)

    parser.add_argument("--selection_rule", choices=["argmin_f2_only", "paper", "epsilon_min_cutpoints", "knee"], default="paper")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--classifiers", type=str, default="SVM,CART", help="默认只跑论文主评估 SVM,CART；可加 KNN。")

    parser.add_argument("--n_jobs", type=int, default=max(1, os.cpu_count() // 2), help="CPU 并行数。")
    parser.add_argument("--parallel_backend", choices=["threads", "processes"], default="threads", help="高维数据建议 threads，避免进程复制大矩阵。")
    parser.add_argument("--standardize", action="store_true", help="是否对每折训练/测试做 StandardScaler。")
    parser.add_argument("--feature_top_k", type=int, default=0, help="快速调试用，只保留训练集 ANOVA 前 K 个特征；正式结果慎用。")
    parser.add_argument("--smoke", action="store_true", help="只跑每个数据集 1 折 1 代，用于检查数据读取和流程。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main():
    global ARGS, CLASSIFIERS
    ARGS = parse_args()
    output_dir = Path(ARGS.output_dir)
    setup_logging(output_dir)
    set_seed(ARGS.seed)
    CLASSIFIERS = configure_classifiers([x.strip() for x in ARGS.classifiers.split(",")])

    logging.info("========== ARPEDA Medical Run ==========")
    logging.info(f"data_dir={ARGS.data_dir}")
    logging.info(f"output_dir={ARGS.output_dir}")
    logging.info(f"datasets={ARGS.datasets}")
    logging.info(f"generations={ARGS.generations}, max_levels={ARGS.max_levels}, l_min={ARGS.l_min}")
    logging.info(f"selection_rule={ARGS.selection_rule}, epsilon={ARGS.epsilon}")
    logging.info(f"n_jobs={ARGS.n_jobs}, parallel_backend={ARGS.parallel_backend}")
    logging.info(f"classifiers={list(CLASSIFIERS.keys())}")

    results_summary = {clf_name: [] for clf_name in CLASSIFIERS.keys()}
    datasets = [d.strip() for d in ARGS.datasets.split(",") if d.strip()]

    for dataset_name in datasets:
        logging.info(f"\n开始处理医学数据集: {dataset_name}")
        perform_independent_test(dataset_name, output_dir, results_summary) if ARGS.protocol == "holdout" else perform_cross_validation(dataset_name, output_dir, results_summary)

    summary_dir = output_dir / "medical_result"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for clf_name, rows in results_summary.items():
        df = pd.DataFrame(rows)
        out = summary_dir / f"{clf_name}_medical_results.csv"
        df.to_csv(out, index=False)
        logging.info(f"{clf_name} 汇总结果保存: {out}")

    logging.info("全部完成。")


if __name__ == "__main__":
    main()

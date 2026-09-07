import numpy as np
import pandas as pd
import random
import os
# 限制 BLAS/NumPy/Sklearn 底层线程，避免 joblib 多进程时线程过度嵌套
for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")
import logging
import itertools
from scipy.io import loadmat
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from deap import base, creator, tools
from joblib import Parallel, delayed
import multiprocessing as mp
import time

from module.discretization_result_LM_v3 import discretization_result, lloyd_max_1d
from module.fitness_function3 import fitness_function

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 随机种子
np.random.seed(42)
random.seed(42)

# 数据目录和全局参数
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('ARPEDA_DATA_DIR', os.path.join(SCRIPT_DIR, 'data', 'UCI_data'))
OUTPUT_DIR = os.environ.get('ARPEDA_OUTPUT_DIR', os.path.join(SCRIPT_DIR, 'uci_result'))
RESULT_DIR = OUTPUT_DIR

POP_SIZE = 200
MAX_ITER = int(os.environ.get('ARPEDA_MAX_ITER', '300'))
MAX_LEVELS = int(os.environ.get('ARPEDA_MAX_LEVELS', '15'))
CXPB = 0.8
MUTPB = 0.2

CPU_COUNT = os.cpu_count() or 4
# 可在命令行通过环境变量覆盖，例如：ARPEDA_N_JOBS_FITNESS=16 python main_MOEA_D.py
N_JOBS_FITNESS = int(os.environ.get('ARPEDA_N_JOBS_FITNESS', max(1, min(8, CPU_COUNT - 2))))
N_JOBS_CLASSIFIER = int(os.environ.get('ARPEDA_N_JOBS_CLASSIFIER', max(1, min(4, N_JOBS_FITNESS))))

# 分类器
# 默认只跑论文主表需要的 SVM 和 CART，避免 CatBoost/KNN/NB 在某些折中拖住整个程序。
# 需要全部分类器时可用：ARPEDA_CLASSIFIERS=SVM,KNN,CART,NaiveBayes,CatBoost
ALL_CLASSIFIERS = {
    'SVM': {'model': SVC, 'params': {'kernel': ['rbf', 'linear'], 'C': [0.1, 1.0, 10.0]}},
    'KNN': {'model': KNeighborsClassifier, 'params': {'n_neighbors': [3, 5, 7]}},
    'CART': {'model': DecisionTreeClassifier, 'params': {'max_depth': [3, 5, 10]}},
    'NaiveBayes': {'model': GaussianNB, 'params': {'var_smoothing': [1e-9, 1e-8, 1e-7]}},
    'CatBoost': {'model': CatBoostClassifier,
                 'params': {'iterations': [100], 'depth': [4, 6], 'learning_rate': [0.01, 0.1],
                            'verbose': [False], 'thread_count': [1], 'allow_writing_files': [False]}}
}
_selected_classifiers = [s.strip() for s in os.environ.get('ARPEDA_CLASSIFIERS', 'SVM,CART').split(',') if s.strip()]
CLASSIFIERS = {name: ALL_CLASSIFIERS[name] for name in _selected_classifiers if name in ALL_CLASSIFIERS}
if not CLASSIFIERS:
    raise ValueError(f"未选择有效分类器：{_selected_classifiers}")

# 数据集列表；可用环境变量临时指定，例如：ARPEDA_DATASETS=appendicitis,balance,bupa
DATASETS_DEFAULT = [
 'abalone',
    'balance',
    'contraceptive',
    'haberman',
    'iris',
    'penbased',
    'phoneme',
    'pima',
    'saheart',
    'satimage',
    'sonar',
    'tae',
    'transfusion',
    'vehicle',
    'yeast',
]
DATASETS = [s.strip() for s in os.environ.get('ARPEDA_DATASETS', ','.join(DATASETS_DEFAULT)).split(',') if s.strip()]

# 加载数据函数
def load_fold_data(dataset_name, fold_idx):
    """加载指定折的训练和测试数据，标签为最后一列"""
    try:
        train_file = os.path.join(DATA_DIR, f"{dataset_name}-10dobscv-{fold_idx}tra.csv")
        test_file = os.path.join(DATA_DIR, f"{dataset_name}-10dobscv-{fold_idx}tst.csv")
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"文件 {train_file} 或 {test_file} 不存在")
        train_data = pd.read_csv(train_file, header=None)
        test_data = pd.read_csv(test_file, header=None)
        # logging.info(f"折 {fold_idx} 训练集列数: {train_data.shape[1]}, 前5行: \n{train_data.head()}")
        # logging.info(f"折 {fold_idx} 测试集列数: {test_data.shape[1]}, 前5行: \n{test_data.head()}")
        X_train = train_data.iloc[:, :-1].values.astype(float)
        y_train = train_data.iloc[:, -1].values.astype(int)
        X_test = test_data.iloc[:, :-1].values.astype(float)
        y_test = test_data.iloc[:, -1].values.astype(int)
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        logging.info(f"折 {fold_idx} 训练集类别: {train_classes}, 测试集类别: {test_classes}")
        if len(train_classes) < 2:
            logging.warning(f"折 {fold_idx} 训练集只有 {len(train_classes)} 个类别")
        if len(test_classes) < 2:
            logging.warning(f"折 {fold_idx} 测试集只有 {len(test_classes)} 个类别")
        X_train = np.copy(X_train)
        y_train = np.copy(y_train)
        X_test = np.copy(X_test)
        y_test = np.copy(y_test)
        X_train.setflags(write=True)
        y_train.setflags(write=True)
        X_test.setflags(write=True)
        y_test.setflags(write=True)
        logging.info(f"成功加载数据集 {dataset_name} 折 {fold_idx}")
        return X_train, y_train, X_test, y_test, True
    except Exception as e:
        logging.error(f"加载数据集 {dataset_name} 折 {fold_idx} 失败: {str(e)}")
        return None, None, None, None, False

# 初始化种群
def initialize_population(num_features: int, population_size: int, max_levels: int,
                          diversity_factor: float = 0.5) -> list:
    """初始化种群，结合启发式和随机初始化"""
    population = []
    heuristic_count = int(population_size * diversity_factor)
    for i in range(population_size):
        if i < heuristic_count:
            individual = np.random.randint(2, min(5, max_levels + 1), size=num_features)
        else:
            individual = np.random.randint(2, max_levels + 1, size=num_features)
        population.append(individual.tolist())
    return population

# 离散化函数，计算切点数
def discretize_data(X: np.ndarray, levels: list) -> tuple:
    """离散化数据并返回切点数"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X 必须是二维 NumPy 数组")
    if len(levels) != X.shape[1]:
        raise ValueError("levels 的长度必须等于特征数")
    X_discretized = np.zeros_like(X, dtype=float)
    total_cutpoints = 0  # 跟踪总切点数
    for i in range(X.shape[1]):
        L = int(levels[i])
        if L < 0:
            raise ValueError(f"特征 {i} 的离散化级别数必须非负")
        if L <= 1:
            X_discretized[:, i] = 1
        else:
            try:
                X_discretized[:, i], _ = discretization_result(X[:, i],L)
                total_cutpoints += (L - 1)  # 为该特征添加切点数
            except Exception as e:
                raise ValueError(f"特征 {i} 离散化失败: {str(e)}")
    return X_discretized, total_cutpoints


def fit_discretizer_on_train(X: np.ndarray, levels: list) -> tuple:
    """只在训练集上学习 Lloyd-Max thresholds/levels，并返回训练集离散化结果、规则和切点数。"""
    if len(levels) != X.shape[1]:
        raise ValueError("levels 的长度必须等于特征数")
    X_discretized = np.zeros_like(X, dtype=float)
    rules = []
    total_cutpoints = 0
    for i in range(X.shape[1]):
        L = int(levels[i])
        if L <= 1:
            X_discretized[:, i] = 1.0
            rules.append({'L': 1, 'levels': None, 'boundaries': None})
            continue
        levels_i, thresholds = lloyd_max_1d(X[:, i], L)
        boundaries = thresholds[1:-1]
        codes = np.digitize(X[:, i], boundaries)
        codes = np.clip(codes, 0, L - 1)
        X_discretized[:, i] = levels_i[codes]
        rules.append({'L': L, 'levels': levels_i, 'boundaries': boundaries})
        total_cutpoints += (L - 1)
    return X_discretized, rules, total_cutpoints


def transform_with_train_discretizer(X: np.ndarray, rules: list) -> np.ndarray:
    """使用训练集学到的 thresholds/levels 变换测试集，避免测试集分布泄漏。"""
    if len(rules) != X.shape[1]:
        raise ValueError("rules 的长度必须等于特征数")
    X_discretized = np.zeros_like(X, dtype=float)
    for i, rule in enumerate(rules):
        L = int(rule['L'])
        if L <= 1:
            X_discretized[:, i] = 1.0
            continue
        codes = np.digitize(X[:, i], rule['boundaries'])
        codes = np.clip(codes, 0, L - 1)
        X_discretized[:, i] = rule['levels'][codes]
    return X_discretized

# DEAP 配置
def setup_toolbox(num_features: int, max_levels: int) -> base.Toolbox:
    """配置 DEAP 工具箱"""
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 2, max_levels)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=2, up=max_levels, indpb=0.2)
    return toolbox

def binom(n, k):
    if k > n - k:
        k = n - k
    res = 1
    for i in range(1, k + 1):
        res = res * (n - i + 1) // i
    return res

def generate_simplex_lattice(n_obj, p):
    weights = []
    for comb in itertools.product(range(p + 1), repeat=n_obj):
        if sum(comb) == p:
            weights.append([c / p for c in comb])
    return np.array(weights)

def run_moead(X_train: np.ndarray, y_train: np.ndarray, population_size: int, max_levels: int,
              generations: int, crossover_rate: float, mutation_rate: float) -> tuple:
    # 小样本/多类别数据集中，某些类别在当前训练折可能只有 1 个样本，
    # 此时 stratify 会报错；这里仅在每类至少 2 个样本时使用分层划分。
    classes, counts = np.unique(y_train, return_counts=True)
    stratify_arg = y_train if counts.size > 0 and counts.min() >= 2 else None
    if stratify_arg is None:
        logging.warning("当前训练折存在样本数小于 2 的类别，内部验证集划分不使用 stratify。")
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_arg
    )
    num_features = X_train_sub.shape[1]
    toolbox = setup_toolbox(num_features, max_levels)
    def fitness_function_wrapper(individual: list, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
                                 y_val: np.ndarray) -> tuple:
        individual_array = np.array(individual, dtype=int)
        if individual_array.ndim != 1:
            raise ValueError(f"individual 必须是一维数组，收到 shape {individual_array.shape}")
        train_data = np.hstack((y[:, np.newaxis], X))
        val_data = np.hstack((y_val[:, np.newaxis], X_val))
        return fitness_function(individual_array, train_data, val_data)
    toolbox.register("evaluate", fitness_function_wrapper)
    n_obj = 3
    p = 19
    weights = generate_simplex_lattice(n_obj, p)
    population_size = len(weights)
    logging.info(f"Population size: {population_size}")
    pop = initialize_population(num_features, population_size, max_levels)
    pop = [creator.Individual(ind) for ind in pop]
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = Parallel(n_jobs=N_JOBS_FITNESS, backend="loky")(
        delayed(lambda ind, X, y, X_val, y_val: toolbox.evaluate(ind, X, y, X_val, y_val))(
            ind, X_train_sub, y_train_sub, X_val, y_val
        ) for ind in invalid_ind
    )
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    z = np.min([ind.fitness.values for ind in pop], axis=0)
    dists = cdist(weights, weights, 'euclidean')
    T = min(20, population_size - 1)  # 防止 T 过大
    neighbors = [np.argsort(d)[1:T+1].tolist() for d in dists]
    logging.info(f"MOEA/D neighborhood size T={T}, number of subproblems={population_size}")
    delta = 0.9
    moe_start_time = time.perf_counter()
    log_every = max(1, int(os.environ.get('ARPEDA_LOG_EVERY', '5')))
    for gen in range(generations):
        gen_start_time = time.perf_counter()
        if gen == 0 or (gen + 1) % log_every == 0:
            logging.info(f"MOEA/D generation {gen + 1}/{generations} started")
        offspring = []
        for i in range(population_size):
            if random.random() < delta:
                B = neighbors[i]
            else:
                B = list(range(population_size))
            if len(B) < 2:
                logging.error(f"Insufficient neighbors for index {i}: {B}")
                continue
            k, l = random.sample(B, 2)
            child1 = toolbox.clone(pop[k])
            child2 = toolbox.clone(pop[l])
            if random.random() < crossover_rate:
                toolbox.mate(child1, child2)
            child = random.choice([child1, child2])
            if random.random() < mutation_rate:
                toolbox.mutate(child)
            del child.fitness.values
            offspring.append(child)
        fitnesses = Parallel(n_jobs=N_JOBS_FITNESS, backend="loky")(
            delayed(lambda ind, X, y, X_val, y_val: toolbox.evaluate(ind, X, y, X_val, y_val))(
                ind, X_train_sub, y_train_sub, X_val, y_val
            ) for ind in offspring
        )
        for child, fit in zip(offspring, fitnesses):
            child.fitness.values = fit
        if gen == 0 or (gen + 1) % log_every == 0 or (gen + 1) == generations:
            elapsed_total = time.perf_counter() - moe_start_time
            elapsed_gen = time.perf_counter() - gen_start_time
            avg_per_gen = elapsed_total / max(1, gen + 1)
            remaining = avg_per_gen * max(0, generations - gen - 1)
            logging.info(
                f"MOEA/D generation {gen + 1}/{generations} finished; "
                f"this_gen={elapsed_gen:.1f}s, elapsed={elapsed_total/60:.1f}min, ETA={remaining/60:.1f}min"
            )
        for i in range(population_size):
            child = offspring[i]
            fit = np.array(child.fitness.values)
            z = np.min([z, fit], axis=0)
            if random.random() < delta:
                B = neighbors[i]
            else:
                B = list(range(population_size))
            if len(B) < 2:
                logging.error(f"Insufficient neighbors for index {i}: {B}")
                continue
            for j in B:
                g_new = np.max(weights[j] * np.abs(fit - z))
                g_old = np.max(weights[j] * np.abs(np.array(pop[j].fitness.values) - z))
                if g_new <= g_old:
                    pop[j] = toolbox.clone(child)
    pareto_front = tools.sortNondominated(pop, population_size, first_front_only=True)[0]
    return pop, pareto_front

# 评估个体（仅保留准确率）
def evaluate_individual(individual: list, X_train_discretized: np.ndarray, y_train: np.ndarray,
                        X_test_discretized: np.ndarray, y_test: np.ndarray, classifier_name: str) -> dict:
    """评估个体，使用交叉验证选择参数，仅返回准确率"""
    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)
    if len(train_classes) < 2:
        logging.warning(f"分类器 {classifier_name}: 训练集只有 {len(train_classes)} 个类别，返回无效指标")
        return {'Accuracy': 0.0, 'Best_Params': {}}
    if len(test_classes) < 2:
        logging.warning(f"分类器 {classifier_name}: 测试集只有 {len(test_classes)} 个类别，返回无效指标")
        return {'Accuracy': 0.0, 'Best_Params': {}}
    param_grid = CLASSIFIERS[classifier_name]['params']
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    # 与论文实验设置一致，对所有基准数据集使用五折参数搜索。
    cv_splits = 5

    logging.info(f"分类器 {classifier_name}: 开始参数搜索，共 {len(param_combinations)} 组参数，cv={cv_splits}")
    clf_start_time = time.time()

    def evaluate_params(param):
        clf_params = param.copy()
        if classifier_name in ['SVM', 'CART', 'CatBoost']:
            clf_params['random_state'] = 42
        clf = CLASSIFIERS[classifier_name]['model'](**clf_params)
        try:
            scores = cross_val_score(clf, X_train_discretized, y_train, cv=cv_splits, scoring='accuracy', n_jobs=1)
            return scores.mean(), param
        except Exception as e:
            logging.warning(f"分类器 {classifier_name} 参数 {param} 交叉验证失败: {str(e)}")
            return -np.inf, param
    param_results = Parallel(n_jobs=N_JOBS_CLASSIFIER, backend="loky")(
        delayed(evaluate_params)(param) for param in param_combinations
    )
    logging.info(f"分类器 {classifier_name}: 参数搜索完成，耗时 {(time.time() - clf_start_time):.1f}s")
    best_acc, best_params = -np.inf, None
    for acc, param in param_results:
        if acc > best_acc:
            best_acc = acc
            best_params = param
    if best_params is None:
        logging.error(f"分类器 {classifier_name} 无有效参数组合")
        return {'Accuracy': 0.0, 'Best_Params': {}}
    clf_params = best_params.copy()
    if classifier_name in ['SVM', 'CART', 'CatBoost']:
        clf_params['random_state'] = 42
    clf = CLASSIFIERS[classifier_name]['model'](**clf_params)
    logging.info(f"分类器 {classifier_name}: 最优参数 {best_params}，开始测试集拟合/预测")
    try:
        clf.fit(X_train_discretized, y_train)
        y_pred = clf.predict(X_test_discretized)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        logging.error(f"分类器 {classifier_name} 测试集评估失败: {str(e)}")
        accuracy = 0.0
    return {'Accuracy': accuracy, 'Best_Params': best_params}

# 测试集评估（仅保留准确率）
def evaluate_test_set(best_individual: list, X_train_discretized: np.ndarray, y_train: np.ndarray,
                      X_test_discretized: np.ndarray, y_test: np.ndarray, classifier_name: str,
                      best_params: dict) -> dict:
    """在测试集上评估最佳个体，仅返回准确率"""
    test_classes = np.unique(y_test)
    if len(test_classes) < 2:
        logging.warning(f"分类器 {classifier_name}: 测试集只有 {len(test_classes)} 个类别，返回无效指标")
        return {'Accuracy': 0.0}
    clf_params = best_params.copy()
    if classifier_name in ['SVM', 'CART', 'CatBoost']:
        clf_params['random_state'] = 42
    clf = CLASSIFIERS[classifier_name]['model'](**clf_params)
    try:
        clf.fit(X_train_discretized, y_train)
        y_pred = clf.predict(X_test_discretized)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        logging.error(f"测试集评估失败，分类器 {classifier_name}: {str(e)}")
        accuracy = 0.0
    return {'Accuracy': accuracy}

def perform_cross_validation(dataset_name: str, base_path: str, population_size: int = 50,
                             generations: int = 100, crossover_rate: float = 0.8,
                             mutation_rate: float = 0.2, results_summary: dict = None) -> pd.DataFrame:
    """执行十折交叉验证，动态选择最佳 MAX_LEVELS，记录准确率、标准差和切点数"""
    results = []
    # 定义 MAX_LEVELS 候选值
    # max_levels_candidates = [5, 10, 15, 20]
    max_levels_candidates = [MAX_LEVELS]  # 正式重跑先固定，避免用外层测试折选择超参数
    best_max_levels = None
    best_avg_accuracy = -np.inf
    # 循环评估每个 MAX_LEVELS
    for max_levels in max_levels_candidates:
        logging.info(f"\n评估数据集 {dataset_name} 的 MAX_LEVELS = {max_levels}")
        fold_results = []
        # 执行十折交叉验证
        for fold_num in range(1, 11):
            logging.info(f"\n数据集 {dataset_name} 第 {fold_num}/10 折 with MAX_LEVELS = {max_levels}")
            X_train, y_train, X_test, y_test, success = load_fold_data(dataset_name, fold_num)
            fold_result = {'Fold': fold_num}
            if not success:
                logging.warning(f"第 {fold_num} 折数据加载失败，跳过评估")
                for clf_name in CLASSIFIERS.keys():
                    fold_result.update({f'{clf_name}_Test_Accuracy': 0.0})
                fold_result['Cutpoints'] = 0
                fold_results.append(fold_result)
                checkpoint_path = os.path.join(base_path, f'{dataset_name}_moead_classifier_results_partial.csv')
                pd.DataFrame(fold_results).to_csv(checkpoint_path, index=False)
                logging.info(f"临时结果已保存至 {checkpoint_path}")
                continue
            train_mean = np.mean(X_train, axis=0)
            train_std = np.std(X_train, axis=0, ddof=1)
            train_std = np.where(train_std < 1e-10, 1.0, train_std)
            best_population, pareto_front = run_moead(X_train, y_train, population_size,
                                                      max_levels, generations, crossover_rate, mutation_rate)
            fitness_values = [ind.fitness.values for ind in pareto_front]
            best_idx = min(range(len(fitness_values)), key=lambda idx: (fitness_values[idx][1], fitness_values[idx][0], fitness_values[idx][2]))
            best_individual = pareto_front[best_idx][:]
            discretized_data_train, discretizer_rules, total_cutpoints = fit_discretizer_on_train(X_train, best_individual)
            discretized_data_test = transform_with_train_discretizer(X_test, discretizer_rules)
            discretized_data_train.setflags(write=True)
            discretized_data_test.setflags(write=True)
            fold_result['Cutpoints'] = total_cutpoints
            logging.info(f"第 {fold_num} 折 切点数: {total_cutpoints}")
            seq_std_zero = np.where(np.array(best_individual) == 1)[0]
            if seq_std_zero.size > 0:
                discretized_data_train = np.delete(discretized_data_train, seq_std_zero, axis=1)
                discretized_data_test = np.delete(discretized_data_test, seq_std_zero, axis=1)
            def evaluate_classifier(clf_name):
                return clf_name, evaluate_individual(
                    best_individual, discretized_data_train, y_train, discretized_data_test, y_test, clf_name
                )
            # 分类器顺序执行，并且每完成一个就立即写日志，避免某个分类器卡住时看不到进度。
            for clf_name in CLASSIFIERS.keys():
                one_clf_start = time.time()
                logging.info(f"第 {fold_num} 折 开始评估分类器 {clf_name}")
                try:
                    clf_name, metrics = evaluate_classifier(clf_name)
                except Exception as e:
                    logging.exception(f"第 {fold_num} 折 分类器 {clf_name} 评估异常，记为 0.0: {e}")
                    metrics = {'Accuracy': 0.0}
                fold_result.update({f'{clf_name}_Test_Accuracy': metrics['Accuracy']})
                logging.info(
                    f"第 {fold_num} 折 测试集 {clf_name} - 准确率: {metrics['Accuracy']:.4f}; "
                    f"耗时 {(time.time() - one_clf_start):.1f}s"
                )
            fold_results.append(fold_result)
            checkpoint_path = os.path.join(base_path, f'{dataset_name}_moead_classifier_results_partial.csv')
            pd.DataFrame(fold_results).to_csv(checkpoint_path, index=False)
            logging.info(f"第 {fold_num} 折临时结果已保存至 {checkpoint_path}")
        # 计算当前 MAX_LEVELS 的平均准确率
        fold_results_df = pd.DataFrame(fold_results)
        if fold_results_df.empty or not CLASSIFIERS:
            logging.warning(f"数据集 {dataset_name}，MAX_LEVELS = {max_levels} 无有效数据或分类器，跳过准确率计算")
            continue
        avg_accuracy = fold_results_df[[f'{clf_name}_Test_Accuracy' for clf_name in CLASSIFIERS.keys()]].mean().mean()
        clf_avg_accuracies = fold_results_df[[f'{clf_name}_Test_Accuracy' for clf_name in CLASSIFIERS.keys()]].mean()
        clf_std_accuracies = fold_results_df[[f'{clf_name}_Test_Accuracy' for clf_name in CLASSIFIERS.keys()]].std()
        # 检查是否有加载失败的折
        if fold_results_df[[f'{clf_name}_Test_Accuracy' for clf_name in CLASSIFIERS.keys()]].eq(0).all().any():
            logging.warning(f"数据集 {dataset_name}，MAX_LEVELS = {max_levels} 包含加载失败的折，平均准确率可能受影响")
        # 输出详细日志
        logging.info(f"数据集 {dataset_name}，MAX_LEVELS = {max_levels} 的平均准确率: {avg_accuracy:.4f}")
        logging.info("每个分类器的平均准确率和标准差：")
        for clf_name in CLASSIFIERS.keys():
            logging.info(f"  {clf_name}: {clf_avg_accuracies[f'{clf_name}_Test_Accuracy']:.4f} (±{clf_std_accuracies[f'{clf_name}_Test_Accuracy']:.4f})")
        # 更新最佳 MAX_LEVELS
        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            best_max_levels = max_levels
            results = fold_results
    logging.info(f"\n数据集 {dataset_name} 的最佳 MAX_LEVELS: {best_max_levels}")
    results_df = pd.DataFrame(results)
    output_path = os.path.join(base_path, f'{dataset_name}_moead_classifier_results.csv')
    results_df.to_csv(output_path, index=False)
    logging.info(f"\n结果已保存至 {output_path}")
    logging.info("\n10折交叉验证（测试集）总结：")
    cv_results = results_df[results_df['Fold'].apply(lambda x: isinstance(x, int))]
    for clf_name in CLASSIFIERS.keys():
        mean_acc = cv_results[f'{clf_name}_Test_Accuracy'].mean()
        std_acc = cv_results[f'{clf_name}_Test_Accuracy'].std()
        logging.info(f"\n分类器: {clf_name}")
        logging.info(f"测试集平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    if results_summary is not None:
        for clf_name in CLASSIFIERS.keys():
            mean_acc = cv_results[f'{clf_name}_Test_Accuracy'].mean()
            std_acc = cv_results[f'{clf_name}_Test_Accuracy'].std()
            mean_cutpoints = cv_results['Cutpoints'].mean()
            if clf_name not in results_summary:
                results_summary[clf_name] = []
            results_summary[clf_name].append({
                'Dataset': dataset_name,
                'Mean_Accuracy': mean_acc,
                'Std_Accuracy': std_acc,
                'Mean_Cutpoints': mean_cutpoints,
                'Best_MAX_LEVELS': best_max_levels
            })
    return results_df

# 主程序
if __name__ == "__main__":

    logging.info(f"DATA_DIR={DATA_DIR}")
    logging.info(f"OUTPUT_DIR={OUTPUT_DIR}")
    logging.info(f"N_JOBS_FITNESS={N_JOBS_FITNESS}, N_JOBS_CLASSIFIER={N_JOBS_CLASSIFIER}")
    logging.info(f"MAX_ITER={MAX_ITER}, MAX_LEVELS={MAX_LEVELS}")
    logging.info(f"DATASETS={DATASETS}")
    logging.info(f"CLASSIFIERS={list(CLASSIFIERS.keys())}")
    results_summary = {clf_name: [] for clf_name in CLASSIFIERS.keys()}

    result_dir = RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    for dataset_name in DATASETS:
        logging.info(f"\n处理数据集: {dataset_name}")
        results_df = perform_cross_validation(
            dataset_name=dataset_name,
            base_path=result_dir,
            population_size=POP_SIZE,
            generations=MAX_ITER,
            crossover_rate=CXPB,
            mutation_rate=MUTPB,
            results_summary=results_summary
        )

    for clf_name in CLASSIFIERS.keys():
        clf_results = pd.DataFrame(results_summary[clf_name])
        output_path = os.path.join(result_dir, f'{clf_name}_results.csv')
        clf_results.to_csv(output_path, index=False)
        logging.info(f"分类器 {clf_name} 的结果已保存至 {output_path}")

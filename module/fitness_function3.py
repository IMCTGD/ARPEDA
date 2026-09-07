import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from .compute_CCC import compute_CCC
from .compute_CCC_include_zeros import compute_CCC_include_zeros
from .discounted_CG_vector import discounted_CG_vector
from .delete_replication_function import delete_replication_function
from .discretization_result_LM_v2 import discretization_result

VARIANT = "cis"
_EPS = 1e-12


def _finite(x, fill=0.0):
    return np.nan_to_num(np.asarray(x, dtype=float), nan=fill, posinf=fill, neginf=fill)


def _safe_std(x, ddof=1, fallback=1e-2):
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size <= ddof:
        return fallback
    s = float(np.std(x, ddof=ddof))
    if not np.isfinite(s) or s <= 0:
        return fallback
    return max(s, fallback)


def _safe_mean(x, fallback=0.0):
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return fallback
    m = float(np.mean(x))
    return m if np.isfinite(m) else fallback


def _safe_compute_ccc(A, y, include_zeros=False):
    A = _finite(A)
    if A.ndim != 2 or A.shape[1] == 0:
        return np.zeros((2, 0), dtype=float)
    try:
        if include_zeros:
            out = compute_CCC_include_zeros(A, y)
        else:
            out = compute_CCC(A, y)
        return _finite(out)
    except Exception:
        # Do not let a degenerate individual kill the whole evolutionary run.
        return np.zeros((2, A.shape[1]), dtype=float)


def _discretize_matrix(X, means, stds, L_vector):
    """Discretize X and remove features whose L <= 1 from the effective matrix."""
    X = _finite(X)
    full = np.zeros_like(X, dtype=float)
    keep = []

    for i in range(X.shape[1]):
        L = int(L_vector[i])
        if L <= 1:
            full[:, i] = 1.0
            continue
        col_mean = float(means[i]) if np.isfinite(means[i]) else _safe_mean(X[:, i])
        col_std = float(stds[i]) if np.isfinite(stds[i]) and stds[i] > 0 else _safe_std(X[:, i])
        try:
            full[:, i], _ = discretization_result(X[:, i], col_mean, col_std, L)
            full[:, i] = _finite(full[:, i])
            keep.append(i)
        except Exception:
            # Collapse failed Lloyd-Max columns rather than crashing.
            full[:, i] = 1.0

    if keep:
        effective = full[:, keep]
    else:
        effective = np.empty((X.shape[0], 0), dtype=float)
    return full, effective, np.asarray(keep, dtype=int)


def _quantize_relevance(values, max_l=15):
    """Quantize continuous CCC values into ordinal relevance scores."""
    values = _finite(values).ravel()
    n = values.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1 or np.unique(np.round(values, 12)).size <= 1:
        return np.ones(n, dtype=float)

    ccc_mean = _safe_mean(values)
    ccc_std = _safe_std(values, ddof=1, fallback=1e-2)
    ccc_l = min(max_l, max(2, int(np.ceil(ccc_std / 1e-3))))

    try:
        q, _ = discretization_result(values, ccc_mean, ccc_std, ccc_l)
        q = _finite(q).reshape(-1, 1)
        uniq = delete_replication_function(q.T)
        uniq = np.sort(_finite(uniq).flatten())
        if uniq.size == 0:
            return np.ones(n, dtype=float)
        scores = np.ones(n, dtype=float)
        for i in range(n):
            matched = np.where(np.isclose(q[i, 0], uniq, rtol=1e-5, atol=1e-12))[0]
            scores[i] = float(matched[0] + 1) if matched.size > 0 else 1.0
        return scores
    except Exception:
        order = np.argsort(values)
        scores = np.empty(n, dtype=float)
        scores[order] = np.arange(1, n + 1, dtype=float)
        return scores


def _cis_loss(A, y, A_disc_effective, keep_indices, fundus=2):
    """CCC/NDCG-based CIS loss: 1 - mean(NDCG). Lower is better."""
    if keep_indices.size == 0 or A_disc_effective.shape[1] == 0:
        return 1.0

    before = _safe_compute_ccc(A, y, include_zeros=False)
    after = _safe_compute_ccc(A_disc_effective, y, include_zeros=True)
    if before.shape[1] == 0 or after.shape[1] == 0:
        return 1.0

    # compute_CCC usually returns [ccc_values; feature_ids(1-based)].
    before_by_feature = {}
    for value, fid in zip(before[0, :], before[1, :].astype(int)):
        before_by_feature[int(fid) - 1] = float(value)

    before_values_kept = np.array([before_by_feature.get(int(idx), 0.0) for idx in keep_indices], dtype=float)
    rel_scores_kept = _quantize_relevance(before_values_kept, max_l=15)
    rel_by_feature = {int(idx): float(score) for idx, score in zip(keep_indices, rel_scores_kept)}

    ideal_order_local = np.argsort(-before_values_kept)
    ideal_scores = rel_scores_kept[ideal_order_local]

    actual_original_features = []
    for fid in after[1, :].astype(int):
        local_idx = int(fid) - 1
        if 0 <= local_idx < keep_indices.size:
            actual_original_features.append(int(keep_indices[local_idx]))

    if not actual_original_features:
        return 1.0

    actual_scores = np.array([rel_by_feature.get(idx, 0.0) for idx in actual_original_features], dtype=float)
    k = min(ideal_scores.size, actual_scores.size)
    if k == 0:
        return 1.0

    ideal_scores = ideal_scores[:k]
    actual_scores = actual_scores[:k]

    idcg = _finite(discounted_CG_vector(ideal_scores, fundus))
    dcg = _finite(discounted_CG_vector(actual_scores, fundus))
    ndcg = dcg / np.where(np.abs(idcg) < _EPS, 1.0, idcg)
    ndcg = np.clip(_finite(ndcg), 0.0, 1.0)
    return float(1.0 - np.mean(ndcg))


def _class_entropy(y):
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts.astype(float) / max(np.sum(counts), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _conditional_entropy_loss(X_disc, y):
    """Average normalized H(C|X_j)/H(C). Lower is better."""
    X_disc = _finite(X_disc)
    y = np.asarray(y)
    if X_disc.ndim != 2 or X_disc.shape[1] == 0 or y.size == 0:
        return 1.0

    h_c = _class_entropy(y)
    if h_c <= _EPS:
        return 0.0

    losses = []
    n = len(y)
    for j in range(X_disc.shape[1]):
        xj = X_disc[:, j]
        h_cond = 0.0
        for v in np.unique(xj):
            mask = (xj == v)
            if not np.any(mask):
                continue
            h_cond += (np.sum(mask) / n) * _class_entropy(y[mask])
        losses.append(h_cond / h_c)

    if not losses:
        return 1.0
    return float(np.mean(losses))


def _information_objective(A, y, A_disc_effective, keep_indices):
    if VARIANT == "cis":
        return _cis_loss(A, y, A_disc_effective, keep_indices, fundus=2)
    if VARIANT == "none":
        return 0.0
    if VARIANT == "entropy":
        return _conditional_entropy_loss(A_disc_effective, y)
    raise ValueError(f"Unknown VARIANT: {VARIANT}")


def fitness_function(L_vector, train_data, val_data):
    """Return [information-loss objective, classifier-error objective, complexity objective]."""
    if not isinstance(L_vector, np.ndarray) or L_vector.ndim != 1:
        raise ValueError("L_vector 必须是一维 NumPy 数组")
    if not isinstance(train_data, np.ndarray) or not isinstance(val_data, np.ndarray):
        raise ValueError("train_data 和 val_data 必须是 NumPy 数组")
    if train_data.ndim != 2 or val_data.ndim != 2:
        raise ValueError("train_data 和 val_data 必须是二维数组")
    if train_data.shape[1] != val_data.shape[1]:
        raise ValueError("train_data 和 val_data 的特征数必须一致")
    if train_data.shape[1] < 2:
        raise ValueError("数据至少需要包含 1 列标签和 1 列特征")

    train_data = _finite(train_data)
    val_data = _finite(val_data)

    n_features = train_data.shape[1] - 1
    L_vector = np.floor(L_vector).astype(int)
    if L_vector.size != n_features:
        raise ValueError(f"L_vector 长度 {L_vector.size} 与特征数 {n_features} 不一致")
    L_vector = np.clip(L_vector, 1, None)

    A = train_data[:, 1:]
    y = train_data[:, 0]
    val_A = val_data[:, 1:]
    val_y = val_data[:, 0]

    means = _finite(np.mean(A, axis=0))
    stds = _finite(np.std(A, axis=0, ddof=1), fill=1.0)
    stds = np.where(stds <= 0, 1.0, stds)

    _, A_disc, keep = _discretize_matrix(A, means, stds, L_vector)
    _, val_disc, keep_val = _discretize_matrix(val_A, means, stds, L_vector)

    if not np.array_equal(keep, keep_val):
        common = np.intersect1d(keep, keep_val)
        train_pos = [int(np.where(keep == idx)[0][0]) for idx in common]
        val_pos = [int(np.where(keep_val == idx)[0][0]) for idx in common]
        A_disc = A_disc[:, train_pos] if train_pos else np.empty((A.shape[0], 0), dtype=float)
        val_disc = val_disc[:, val_pos] if val_pos else np.empty((val_A.shape[0], 0), dtype=float)
        keep = common.astype(int)

    f1 = _information_objective(A, y, A_disc, keep)

    if A_disc.shape[1] > 0:
        try:
            tree = DecisionTreeClassifier(random_state=0)
            tree.fit(A_disc, y)
            acc_tree = float(np.mean(tree.predict(val_disc) == val_y))
        except Exception:
            acc_tree = 0.0

        try:
            k = max(1, min(10, A_disc.shape[0]))
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(A_disc, y)
            acc_knn = float(np.mean(knn.predict(val_disc) == val_y))
        except Exception:
            acc_knn = 0.0

        mean_acc = (acc_tree + acc_knn) / 2.0
    else:
        mean_acc = 0.0

    # Paper's third objective: total interval count for attributes with L_i > 1.
    complexity = float(np.sum(L_vector[L_vector != 1]))

    fit_value = np.array([f1, 1.0 - mean_acc, complexity], dtype=float)
    return np.nan_to_num(fit_value, nan=1.0, posinf=1e6, neginf=1e6)

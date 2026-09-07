import numpy as np
from .find_zero_variance import find_zero_variance

def compute_CCC_include_zeros(A, B=None):
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.size == 0:
        raise ValueError("A 必须是非空的二维 NumPy 数组")

    attr_number = A.shape[1]
    seq = np.arange(1, attr_number + 1)
    seq_std_zero = find_zero_variance(A)

    if seq_std_zero.size > 0:
        A_reduced = np.delete(A, seq_std_zero - 1, axis=1)
        seq_reduced = np.delete(seq, seq_std_zero - 1)
    else:
        A_reduced = A
        seq_reduced = seq

    # print(f"seq_reduced: {seq_reduced}, size: {seq_reduced.size}")
    if A_reduced.shape[1] <= 1:
        sort_mspc_weigh = np.zeros((2, attr_number))
        sort_mspc_weigh[1, :] = seq
        if A_reduced.shape[1] == 1:
            sort_mspc_weigh[0, np.where(seq == seq_reduced)[0]] = 1
        return sort_mspc_weigh

    # 标准化
    A_mean = np.mean(A_reduced, axis=0)
    A_std = np.std(A_reduced, axis=0, ddof=1)
    A_std = np.where(A_std == 0, 1, A_std)
    A_standard = (A_reduced - A_mean) / A_std

    # 协方差和特征值分解
    covariance_matrix_A = np.cov(A_standard.T, bias=False)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix_A)
    eigen_values = np.maximum(eigen_values, 0)

    # 排序特征值
    sort_seq = np.argsort(-eigen_values)
    sort_eigen_values = eigen_values[sort_seq]
    sort_eigen_vectors = eigen_vectors[:, sort_seq]

    # 贡献率
    sum_eigen_values = np.sum(sort_eigen_values)
    eigen_values_rate = sort_eigen_values / sum_eigen_values if sum_eigen_values > 0 else np.zeros_like(sort_eigen_values)
    eigen_values_rate = eigen_values_rate.reshape(-1, 1)

    # CCC 计算
    sqrt_eigen_values = np.sqrt(sort_eigen_values).reshape(-1, 1)
    weighted_eigen_values = eigen_values_rate * sqrt_eigen_values
    mspc_weigh = np.abs(sort_eigen_vectors) @ weighted_eigen_values
    mspc_weigh = mspc_weigh.flatten()

    # print(f"mspc_weigh: {mspc_weigh}, shape: {mspc_weigh.shape}")
    sort_mspc_weigh_seq = np.argsort(-mspc_weigh)
    # print(f"sort_mspc_weigh_seq: {sort_mspc_weigh_seq}, max: {np.max(sort_mspc_weigh_seq)}")

    # 验证索引
    if np.max(sort_mspc_weigh_seq) >= seq_reduced.size:
        raise ValueError(f"sort_mspc_weigh_seq 中存在无效索引: {sort_mspc_weigh_seq}, seq_reduced 大小: {seq_reduced.size}")

    mspc_weigh = mspc_weigh[sort_mspc_weigh_seq]
    sort_mspc_weigh_seq = seq_reduced[sort_mspc_weigh_seq]

    # 处理零方差特征
    if seq_std_zero.size > 0:
        part_mspc_weigh = np.zeros(seq_std_zero.size)
        sort_mspc_weigh_seq = np.concatenate((seq_std_zero, sort_mspc_weigh_seq))
        mspc_weigh = np.concatenate((part_mspc_weigh, mspc_weigh))

    sort_mspc_weigh = np.vstack((mspc_weigh, sort_mspc_weigh_seq))
    return sort_mspc_weigh
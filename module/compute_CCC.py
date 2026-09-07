import numpy as np
from .find_zero_variance import find_zero_variance


def compute_CCC(A, B=None):
    """
    计算特征矩阵 A 的 CCC 值（特征重要性），基于主成分分析。

    参数:
        A: 特征矩阵，每行是一条记录，每列是一个特征。
        B: 标签向量（未使用，保留以兼容接口）。

    返回:
        sort_mspc_weigh: 2×n 矩阵，第一行是降序排列的 CCC 值，第二行是特征编号。

    抛出:
        ValueError: 如果 A 为空、维度不正确或内部数组形状不匹配。
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.size == 0:
        raise ValueError("A 必须是非空的二维 NumPy 数组")

    # print("A shape:", A.shape)

    # 检测零方差特征
    seq_std_zero = find_zero_variance(A)
    attr_number = A.shape[1]
    seq = np.arange(1, attr_number + 1)

    # print("seq_std_zero:", seq_std_zero)
    # print("seq:", seq)

    if seq_std_zero.size > 0:
        # 移除零方差特征
        A_reduced = np.delete(A, seq_std_zero - 1, axis=1)
        seq_reduced = np.delete(seq, seq_std_zero - 1)
    else:
        A_reduced = A
        seq_reduced = seq

    # print("A_reduced shape:", A_reduced.shape)
    # print("seq_reduced:", seq_reduced)

    if A_reduced.shape[1] == 0:
        # 所有特征方差为 0
        sort_mspc_weigh = np.zeros((2, attr_number))
        sort_mspc_weigh[1, :] = seq
        # print("所有特征方差为 0，返回零 CCC 值")
        return sort_mspc_weigh
    elif A_reduced.shape[1] == 1:
        # 只有一个非零方差特征
        sort_mspc_weigh = np.zeros((2, attr_number))
        sort_mspc_weigh[1, :] = seq
        sort_mspc_weigh[0, np.where(seq == seq_reduced)[0]] = 1
        # print("单一非零方差特征，CCC 值设为 1")
        return sort_mspc_weigh

    # 标准化
    A_mean = np.mean(A_reduced, axis=0)
    A_std = np.std(A_reduced, axis=0, ddof=1)
    # print("A_std:", A_std)
    A_std = np.where(A_std == 0, 1, A_std)  # 避免除零
    A_standard = (A_reduced - A_mean) / A_std

    # print("A_standard shape:", A_standard.shape)

    # 计算协方差矩阵
    covariance_matrix_A = np.cov(A_standard.T, bias=False)

    # print("covariance_matrix_A shape:", covariance_matrix_A.shape)

    # 特征值分解
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix_A)
    eigen_values = np.maximum(eigen_values, 0)  # 确保非负

    # print("eigen_values shape:", eigen_values.shape)
    # print("eigen_values:", eigen_values)
    # print("eigen_vectors shape:", eigen_vectors.shape)

    # 降序排序
    sort_seq = np.argsort(-eigen_values)
    sort_eigen_values = eigen_values[sort_seq]
    sort_eigen_vectors = eigen_vectors[:, sort_seq]

    # print("sort_seq (eigen_values):", sort_seq)
    # print("sort_eigen_values:", sort_eigen_values)

    # 计算特征贡献率
    sum_eigen_values = np.sum(sort_eigen_values)
    # print("sum_eigen_values:", sum_eigen_values)
    if sum_eigen_values == 0:
        eigen_values_rate = np.zeros_like(sort_eigen_values)
    else:
        eigen_values_rate = sort_eigen_values / sum_eigen_values
    eigen_values_rate = eigen_values_rate.reshape(-1, 1)

    # print("eigen_values_rate shape:", eigen_values_rate.shape)
    # print("eigen_values_rate:", eigen_values_rate)

    # 计算 CCC 值
    sqrt_eigen_values = np.sqrt(sort_eigen_values).reshape(-1, 1)  # 确保 (4, 1)
    weighted_eigen_values = eigen_values_rate * sqrt_eigen_values
    # print("sqrt_eigen_values shape:", sqrt_eigen_values.shape)
    # print("weighted_eigen_values shape:", weighted_eigen_values.shape)

    if weighted_eigen_values.shape != (A_reduced.shape[1], 1):
        raise ValueError(
            f"weighted_eigen_values 形状 {weighted_eigen_values.shape} 不匹配预期 ({A_reduced.shape[1]}, 1)")

    mspc_weigh = np.abs(sort_eigen_vectors) @ weighted_eigen_values
    mspc_weigh = mspc_weigh.flatten()

    # print("mspc_weigh shape:", mspc_weigh.shape)
    # print("mspc_weigh:", mspc_weigh)

    if mspc_weigh.shape[0] != A_reduced.shape[1]:
        raise ValueError(f"mspc_weigh 长度 {mspc_weigh.shape[0]} 不匹配 A_reduced 特征数 {A_reduced.shape[1]}")

    sort_mspc_weigh_seq = np.argsort(-mspc_weigh)

    # print("sort_mspc_weigh_seq (before mapping):", sort_mspc_weigh_seq)

    if np.any(sort_mspc_weigh_seq >= A_reduced.shape[1]):
        raise ValueError(f"sort_mspc_weigh_seq 包含无效索引: {sort_mspc_weigh_seq}")

    mspc_weigh = mspc_weigh[sort_mspc_weigh_seq]
    sort_mspc_weigh_seq = seq_reduced[sort_mspc_weigh_seq]

    # print("mspc_weigh (sorted):", mspc_weigh)
    # print("sort_mspc_weigh_seq (mapped):", sort_mspc_weigh_seq)

    # 合并零方差特征
    if seq_std_zero.size > 0:
        part_mspc_weigh = np.zeros(seq_std_zero.size)
        sort_mspc_weigh_seq = np.concatenate((seq_std_zero, sort_mspc_weigh_seq))
        mspc_weigh = np.concatenate((part_mspc_weigh, mspc_weigh))

    # print("final mspc_weigh:", mspc_weigh)
    # print("final sort_mspc_weigh_seq:", sort_mspc_weigh_seq)

    sort_mspc_weigh = np.vstack((mspc_weigh, sort_mspc_weigh_seq))

    # print("sort_mspc_weigh shape:", sort_mspc_weigh.shape)

    return sort_mspc_weigh
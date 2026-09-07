import numpy as np

def discounted_CG_vector(directed_gain, fundus):
    """
    计算基于排序的 DCG（折扣累积增益），用于 NDCG 计算。

    参数:
        directed_gain: 行向量，表示排序后的值（如 CCC 等级）。
        fundus: 折扣底数（通常为 2）。

    返回:
        DCG_vector: DCG 值向量，前 i 项的 DCG。

    抛出:
        ValueError: 如果 directed_gain 为空或维度不正确。
    """
    if not isinstance(directed_gain, np.ndarray) or directed_gain.ndim != 1:
        raise ValueError("directed_gain 必须是一维 NumPy 数组")
    if directed_gain.size == 0:
        return np.array([])

    # 计算 CG
    CG_vector = np.cumsum(directed_gain)

    # 计算 DCG
    DCG_vector = np.zeros_like(CG_vector, dtype=float)
    DCG_vector[:fundus-1] = CG_vector[:fundus-1]
    for i in range(fundus-1, directed_gain.size):
        DCG_vector[i] = DCG_vector[i-1] + directed_gain[i] / (np.log10(i+2) / np.log10(fundus))

    return DCG_vector
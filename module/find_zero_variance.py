import numpy as np


def find_zero_variance(A):
    """
    查找方差为 0 的特征索引。

    参数:
        A: 输入特征矩阵，形状 (n_samples, n_features)。

    返回:
        seq_std_zero: 方差为 0 的特征索引（从 1 开始，MATLAB 风格）。
    """
    std_A = np.std(A, axis=0, ddof=1)
    seq_std_zero = np.where(std_A < 1e-5)[0] + 1  # 放宽阈值
    return seq_std_zero
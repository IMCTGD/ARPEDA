import numpy as np

import numpy as np


def lloyd_max_1d(data, L, max_iter=100, tol=1e-5):
    """
    一维 Lloyd-Max 量化器，优化量化阈值和重构值。

    参数：
    - data: 输入数据，shape=(n_samples,)
    - L: 量化级别
    - max_iter: 最大迭代次数
    - tol: 收敛容差（动态调整）

    返回：
    - levels: 重构值（电平），shape=(L,)
    - thresholds: 量化阈值，shape=(L+1,)（包含 -inf 和 inf）
    """
    min_val, max_val = np.min(data), np.max(data)

    # 处理单值或稀疏数据
    if max_val - min_val < tol or np.sum(np.abs(data - data.mean()) > 1e-10) < L:
        return np.full(L, data.mean()), np.full(L + 1, data.mean())

    # 自适应初始化：分位数 + 微小扰动确保单调
    quantiles = np.linspace(0, 1, L + 2)[1:-1]
    levels = np.quantile(data, quantiles)
    levels = np.sort(levels + np.arange(L) * 1e-10)  # 避免重复电平

    # 动态容差
    tol = tol * (max_val - min_val)

    for _ in range(max_iter):
        old_levels = levels.copy()

        # 计算阈值
        boundaries = (levels[:-1] + levels[1:]) / 2.0
        thresholds = np.concatenate(([-np.inf], boundaries, [np.inf]))

        # 向量化分区
        codes = np.digitize(data, boundaries)
        codes = np.clip(codes, 0, L - 1)

        # 更新电平
        new_levels = np.zeros(L)
        for i in range(L):
            mask = (codes == i)
            if np.any(mask):
                new_levels[i] = data[mask].mean()
            else:
                # 空区间插值
                new_levels[i] = levels[i - 1] if i == 0 else (levels[i - 1] + levels[i + 1]) / 2 if i < L - 1 else \
                levels[i - 1]

        # 强制单调递增
        levels = np.sort(new_levels)

        # 检查收敛
        if np.max(np.abs(levels - old_levels)) < tol:
            break

    # 最终阈值
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    thresholds = np.concatenate(([-np.inf], boundaries, [np.inf]))

    return levels, thresholds


def discretization_result(A, A_mean, A_std, L, standard_result=None):
    """
    对输入矩阵A的每列独立执行Lloyd-Max量化，返回离散化结果B和阈值-电平对矩阵L_standard_result。

    参数:
        A: numpy数组，形状(n_samples, n_features) 或 (n_samples,)，待量化的数据矩阵。
        A_mean: 每列的均值，标量或 shape=(n_features,)，保留接口，未使用。
        A_std: 每列的标准差，标量或 shape=(n_features,)，保留接口，未使用。
        L: int，每列量化级数。
        standard_result: 保留接口，未使用。

    返回:
        B: numpy数组，形状与A相同，为量化后的输出（使用重构电平值表示）。
        L_standard_result: numpy数组，形状(L, 2*n_features)，每列对应的阈值-电平对：
                           对于第j个特征，阈值存在列2*j，电平存在列2*j+1，每行对应一个电平/阈值对。
    """
    # 转换为 NumPy 数组并验证输入
    A = np.asarray(A, dtype=float)

    # 检查 NaN、inf、空输入
    if np.any(np.isnan(A)) or np.any(np.isinf(A)):
        raise ValueError("输入 A 包含 NaN 或 inf 值")
    if A.size == 0:
        raise ValueError("输入 A 不能为空")

    # 检查 L
    if L < 2:
        raise ValueError("量化级别 L 必须大于等于 2")
    if A.ndim > 1 and L > A.shape[0]:
        raise ValueError("量化级别 L 不能大于样本数")

    # 处理一维输入
    is_1d = A.ndim == 1
    if is_1d:
        A = A.reshape(-1, 1)
    n_samples, n_features = A.shape

    # 验证 A_mean 和 A_std（尽管未使用）
    A_mean = np.asarray(A_mean, dtype=float).flatten()
    A_std = np.asarray(A_std, dtype=float).flatten()
    if A_mean.size == 1:
        A_mean = np.full(n_features, A_mean)
    if A_std.size == 1:
        A_std = np.full(n_features, A_std)
    if A_mean.shape[0] != n_features or A_std.shape[0] != n_features:
        raise ValueError("A_mean 和 A_std 的长度必须与特征数一致")

    # 初始化输出
    B = np.zeros_like(A, dtype=float)
    L_standard_result = np.zeros((L, 2 * n_features), dtype=float)

    # 对每一列特征独立处理
    for j in range(n_features):
        col = A[:, j]

        # 运行 Lloyd-Max 量化
        try:
            levels, thresholds = lloyd_max_1d(col, L)
        except Exception as e:
            raise ValueError(f"特征 {j} 量化失败: {str(e)}")

        # 填充 L_standard_result
        L_standard_result[:, 2 * j] = thresholds[1:L + 1]  # 存储 L 个有效阈值
        L_standard_result[:, 2 * j + 1] = levels

        # 离散化
        boundaries = thresholds[1:-1]
        codes = np.digitize(col, boundaries)
        codes = np.clip(codes, 0, L - 1)
        B[:, j] = levels[codes]

    # 如果输入是一维，返回一维 B
    if is_1d:
        B = B.flatten()

    return B, L_standard_result

# import numpy as np
#
#
# def discretization_result(A, A_mean, A_std, L, standard_result=None):
#     """
#     对输入矩阵A的每列独立执行Lloyd-Max量化，返回离散化结果B和阈值-电平对矩阵L_standard_result。
#
#     参数:
#         A: numpy数组，形状(n_samples, n_features)，待量化的数据矩阵。
#         A_mean, A_std: 保留原函数接口，未在本实现中使用。
#         L: int，每列量化级数。
#         standard_result: 保留接口，未使用。
#
#     返回:
#         B: numpy数组，形状与A相同，为量化后的输出（使用重构电平值表示）。
#         L_standard_result: numpy数组，形状(L, 2*n_features)，每列对应的阈值-电平对：
#                            对于第j个特征，阈值存在列2*j，电平存在列2*j+1，每行对应一个电平/阈值对。
#     """
#     A = np.asarray(A, dtype=float)
#     n_samples, n_features = A.shape
#     # 初始化输出矩阵
#     B = np.zeros_like(A, dtype=float)
#     L_standard_result = np.zeros((L, 2 * n_features), dtype=float)
#
#     max_iter = 100  # 最大迭代次数
#     tol = 1e-5  # 收敛阈值
#
#     # 对每一列特征独立处理
#     for j in range(n_features):
#         col = A[:, j]
#         # 如果该列所有值相同，则直接赋值
#         min_val = col.min()
#         max_val = col.max()
#         if max_val - min_val < tol:
#             # 量化级数内全使用相同电平
#             levels = np.full(L, min_val)
#             thresholds = np.full(L, min_val)
#             # 离散化结果直接是常数列
#             B[:, j] = min_val
#         else:
#             # 1) 初始化L个重构值（在数据范围内等距）
#             levels = np.linspace(min_val, max_val, L)
#
#             # 2) 迭代更新
#             for _ in range(max_iter):
#                 # 2.1 计算相邻重构值的中点作为判决阈值
#                 boundaries = (levels[:-1] + levels[1:]) / 2.0
#                 # 2.2 根据阈值对样本进行分区，np.digitize返回区间索引[0..L-1]
#                 codes = np.digitize(col, boundaries)
#
#                 # 2.3 更新每个区间的电平为该区间内样本的均值
#                 new_levels = levels.copy()
#                 for i in range(L):
#                     if np.any(codes == i):
#                         new_levels[i] = col[codes == i].mean()
#                 # 2.4 检查收敛性：若电平变化都小于tol，则停止迭代
#                 if np.max(np.abs(new_levels - levels)) < tol:
#                     levels = new_levels
#                     break
#                 levels = new_levels
#
#             # 迭代结束后，计算最终判决阈值
#             boundaries = (levels[:-1] + levels[1:]) / 2.0
#             # 构造长度为L的阈值向量：前L-1项为boundaries，最后一项重复最后一个阈值
#             thresholds = np.zeros(L)
#             thresholds[:-1] = boundaries
#             thresholds[-1] = boundaries[-1]
#
#             # 3) 使用最终阈值和电平离散化原始数据
#             codes = np.digitize(col, boundaries)
#             B[:, j] = levels[codes]
#
#         # 将该列的阈值-电平对写入输出矩阵
#         L_standard_result[:, 2 * j] = thresholds
#         L_standard_result[:, 2 * j + 1] = levels
#
#     return B, L_standard_result

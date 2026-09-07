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


def discretization_result(A,  L, standard_result=None):
    """
    对输入矩阵A的每列独立执行Lloyd-Max量化，返回离散化结果B和阈值-电平对矩阵L_standard_result。

    参数:
        A: numpy数组，形状(n_samples, n_features) 或 (n_samples,)，待量化的数据矩阵。
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



"""
BTCH/TCH 实现（含双语注释）
BTCH/TCH implementation with bilingual comments.
"""
import numpy as np

def build_J(N):
    """
    构造差分矩阵 J ∈ R^{(N-1)×N}，将每个产品与第 N 个产品做差。
    Build differencing matrix J (N-1,N) that subtracts the N-th reference.

    Args:
        N (int): number of products
    Returns:
        np.ndarray: J
    """
    J = np.zeros((N-1, N))
    for i in range(N-1):
        J[i, i] = 1.0
        J[i, -1] = -1.0
    return J

def compute_S(et_products):
    """
    计算差分矩阵 Y 的协方差 S = cov(Y)。
    Compute covariance of differenced products S = cov(Y).

    Args:
        et_products (np.ndarray): shape (N, T)
    Returns:
        S (np.ndarray), J (np.ndarray)
    """
    N, T = et_products.shape
    J = build_J(N)
    Y = J @ et_products  # (N-1, T)
    # unbiased covariance with rows as variables -> need row-wise covariance
    # np.cov expects variables in rows -> correct, bias corrected by default (ddof=1)
    S = np.cov(Y)
    return S, J

def tch_uncorrelated_variances(et_products):
    """
    经典三角帽（无相关）估计各产品误差方差 σ_i^2。
    Classic TCH (uncorrelated errors) to estimate variances σ_i^2.

    Uses LS on Var(ET_i - ET_j) ≈ σ_i^2 + σ_j^2.

    Args:
        et_products (np.ndarray): shape (N,T)

    Returns:
        np.ndarray: variances (N,)
    """
    N, T = et_products.shape
    # compute pairwise diff variances
    pairs = []
    b = []  # target vars
    for i in range(N):
        for j in range(i+1, N):
            diff = et_products[i] - et_products[j]
            v = np.var(diff, ddof=1)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            pairs.append(row)
            b.append(v)
    A = np.vstack(pairs)  # shape (P, N)
    b = np.array(b)
    # Nonnegative LS (clip at 0) using normal equation + clip
    # To stabilize, add small ridge
    ridge = 1e-8 * np.eye(N)
    x = np.linalg.lstsq(A.T @ A + ridge, A.T @ b, rcond=None)[0]
    x = np.maximum(x, 1e-12)  # no negative variances
    return x

def project_psd(S):
    """
    将对称矩阵投影到最邻近的半正定矩阵（裁剪负特征值）。
    Project a symmetric matrix to the nearest PSD by clipping negative eigenvalues.
    """
    S = 0.5*(S+S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(w) @ V.T

def tch_correlated_regularized(et_products, alpha=1e-2, lr=1e-2, iters=500, seed=0):
    """
    正则化相关 TCH 近似（可运行版本）
    Regularized correlated TCH approximation (runnable).

    Solve:  min_R  ||J R J^T - S||_F^2 + alpha ||offdiag(R)||_F^2    s.t. R ≽ 0

    We perform simple gradient descent with PSD projection and symmetry enforcement.

    Args:
        et_products (np.ndarray): (N,T)
        alpha (float): off-diagonal penalty
        lr (float): learning rate
        iters (int): iterations

    Returns:
        R (np.ndarray): estimated covariance (N,N)
    """
    rng = np.random.default_rng(seed)
    S, J = compute_S(et_products)
    N, T = et_products.shape

    # init with diagonal from uncorrelated solution
    diag_vars = tch_uncorrelated_variances(et_products)
    R = np.diag(diag_vars)

    for k in range(iters):
        JRJ = J @ R @ J.T
        G_core = J.T @ (JRJ - S) @ J  # gradient of ||JRJ - S||_F^2 wrt R
        # offdiag penalty gradient
        off = R - np.diag(np.diag(R))
        G = 2.0 * G_core + 2.0 * alpha * off

        R = R - lr * G
        # ensure symmetry and PSD
        R = 0.5*(R + R.T)
        R = project_psd(R)

    return R

def btch_weights_from_sigmas(sigmas2):
    """
    根据论文公式（基于最大似然代价函数的解析解）计算 BTCH 权重。
    Compute BTCH weights from error variances using the product-of-others formula:

        w_k = prod_{i != k} σ_i^2  /  sum_{j} prod_{i != j} σ_i^2

    Implemented in log domain for numerical stability.
    """
    sigmas2 = np.asarray(sigmas2).ravel()
    if np.any(sigmas2 <= 0):
        raise ValueError("All variances must be positive.")
    log_s2 = np.log(sigmas2)
    N = sigmas2.size
    # precompute total sum of logs
    total = np.sum(log_s2)
    # each weight uses total - log_s2[k]
    logs = np.array([total - log_s2[k] for k in range(N)])
    # stabilize
    m = np.max(logs)
    w_unnorm = np.exp(logs - m)
    w = w_unnorm / np.sum(w_unnorm)
    return w

def integrate_btch(et_products, R):
    """
    使用协方差 R 的对角线（误差方差）来计算 BTCH 权重并融合。
    Use diag(R) to compute BTCH weights and fuse.

    Args:
        et_products (np.ndarray): (N,T)
        R (np.ndarray): (N,N)

    Returns:
        fused (np.ndarray): (T,)
        weights (np.ndarray): (N,)
        sigmas2 (np.ndarray): (N,)
    """
    sigmas2 = np.diag(R).copy()
    weights = btch_weights_from_sigmas(sigmas2)
    fused = weights @ et_products
    return fused, weights, sigmas2

def integrate_em(et_products):
    """
    简单算术平均 / Ensemble Mean
    """
    return np.mean(et_products, axis=0)

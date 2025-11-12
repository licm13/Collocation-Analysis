
"""
通用工具函数 / General utilities
"""
import numpy as np

def corrcoef(x, y):
    """
    计算皮尔逊相关系数（单变量）
    Compute Pearson correlation coefficient (scalar).

    Args:
        x (np.ndarray): shape (T,)
        y (np.ndarray): shape (T,)
    Returns:
        float
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom == 0:
        return np.nan
    return float((xm*ym).sum()/denom)

def rmse(x, y):
    """
    均方根误差 / Root Mean Square Error
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    return float(np.sqrt(np.mean((x-y)**2)))

def bias(x, y):
    """
    偏差（均值差） / Bias
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    return float(np.mean(x-y))

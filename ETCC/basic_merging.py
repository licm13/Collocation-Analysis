"""
Basic example demonstrating TC and ETCC merging methods.

This example:
1. Generates synthetic precipitation data
2. Applies both TC and ETCC merging
3. Evaluates and compares results
4. Visualizes the comparison
"""
# ETCC/basic_merging.py
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collocation import etcc  # 修正：使用绝对导入
from collocation import tc    # 增补：导入 'tc' 模块
import xarray as xr
import os
import logging
from typing import Dict, List, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_paths: List[str], var_name: str) -> Dict[str, np.ndarray]:
    """
    从 NetCDF 文件加载数据。
    """
    data = {}
    for path in file_paths:
        try:
            with xr.open_dataset(path) as ds:
                # 假设数据是 (time, lat, lon) 或 (time, y, x)
                # 为简单起见，我们将其展平为 1D 时间序列（或在空间上取平均）
                # 这里我们假设数据已经是时序，或者我们只关心一个点
                # 为了演示，我们先简单地取空间平均
                if len(ds[var_name].dims) > 1:
                    # 假设 'time' 是第一个维度
                    spatial_dims = [dim for dim in ds[var_name].dims if dim != 'time']
                    data[os.path.basename(path)] = ds[var_name].mean(dim=spatial_dims).values
                else:
                    data[os.path.basename(path)] = ds[var_name].values
            logging.info(f"Loaded data from {path}")
        except Exception as e:
            logging.error(f"Failed to load data from {path}: {e}")
    return data

def calculate_merged_product(data: Dict[str, np.ndarray], 
                             reference_product_name: str, 
                             output_filename: str,
                             reference_index: Optional[int] = None) -> Optional[xr.Dataset]:
    """
    使用 ETCC 方法计算融合产品。
    
    Args:
        data (Dict[str, np.ndarray]): 包含数据集的字典，键为产品名，值为数据数组。
        reference_product_name (str): 用于 TC 权重的参考产品的名称。
        output_filename (str): 输出 NetCDF 文件的路径。
        reference_index (Optional[int]): (已弃用，但为了兼容) ETCC merge 函数的参考索引。
                                         在基于 'tc' 的权重计算中，参考索引不影响权重。

    Returns:
        Optional[xr.Dataset]: 包含融合产品的 xarray Dataset，如果失败则返回 None。
    """
    if not data:
        logging.error("Data dictionary is empty. Cannot perform merging.")
        return None

    product_names = list(data.keys())
    
    # 确保所有数组长度一致
    try:
        data_array = np.array([data[name] for name in product_names])
    except ValueError as e:
        logging.error(f"Data arrays have inconsistent lengths: {e}")
        return None
        
    logging.info(f"Products to be merged: {product_names}")
    
    # 1. 计算 TC 权重
    # tc.weights_calc 接受一个 (n_products, n_observations) 的数组
    # 它返回的权重不依赖于参考数据集
    try:
        # BUG 修复：使用导入的 'tc' 模块
        weights = tc.weights_calc(data_array)
        logging.info(f"Calculated weights: {weights}")
    except Exception as e:
        logging.error(f"Failed to calculate weights using tc.weights_calc: {e}")
        return None

    # 2. 使用 ETCC（加权平均）
    # etcc.etcc_merge 实际上也做了加权平均，但它内部再次调用 tc.weights_calc
    # 我们可以直接使用 etcc_merge，或者手动进行加权平均
    
    # 方法一：使用 etcc_merge (它会自己计算权重)
    # 注意：etcc_merge 里的 reference_index 也不影响权重计算
    try:
        merged_data_etcc = etcc.etcc_merge(data_array, reference_index=0)
        logging.info("Merged data calculated using etcc.etcc_merge.")
    except Exception as e:
        logging.error(f"Error during etcc.etcc_merge: {e}")
        return None

    # 方法二：手动使用我们计算的权重 (与方法一等效)
    # merged_data_manual = np.average(data_array, axis=0, weights=weights)
    # logging.info("Merged data calculated manually using np.average and TC weights.")

    # 3. 保存结果到 NetCDF
    # 假设我们有一个时间坐标（如果原始数据有的话）
    # 为简单起见，我们只创建一个简单的 Dataset
    try:
        # 假设所有数据都有相同的时间长度
        time_len = data_array.shape[1]
        time_coords = np.arange(time_len)
        
        ds_out = xr.Dataset(
            {
                "merged_product": (["time"], merged_data_etcc),
            },
            coords={
                "time": time_coords,
            },
            attrs={
                "title": "ETCC Merged Product",
                "reference_product_for_weights": reference_product_name, # 尽管TC权重是独立的
                "source_products": ", ".join(product_names),
                "weights": str(dict(zip(product_names, weights)))
            }
        )
        
        # 添加原始数据以便比较
        for i, name in enumerate(product_names):
            ds_out[f"source_{name}"] = (["time"], data_array[i])
            
        ds_out.to_netcdf(output_filename)
        logging.info(f"Successfully saved merged product to {output_filename}")
        return ds_out
        
    except Exception as e:
        logging.error(f"Failed to create or save NetCDF file: {e}")
        return None

if __name__ == '__main__':
    # 这是一个示例用法
    # 你需要提供你的 NetCDF 文件路径
    
    # 示例：创建一些模拟数据
    N = 1000
    t = np.linspace(0, 10, N)
    true_signal = np.sin(t)
    
    data_dict = {
        "ProductA": true_signal + np.random.normal(0, 0.5, N), # 噪声
        "ProductB": 0.8 * true_signal + 0.2 + np.random.normal(0, 0.3, N), # 带偏差和噪声
        "ProductC": 1.1 * true_signal - 0.1 + np.random.normal(0, 0.4, N)  # 带偏差和噪声
    }
    
    # 假设我们将模拟数据保存到临时文件
    temp_files = []
    for name, arr in data_dict.items():
        temp_fn = f"{name}.nc"
        temp_ds = xr.Dataset({
            "et": (["time"], arr),
            "time": (["time"], np.arange(N))
        })
        temp_ds.to_netcdf(temp_fn)
        temp_files.append(temp_fn)

    print(f"Created temporary files: {temp_files}")

    # --- 运行融合 ---
    output_file = "etcc_merged_output.nc"
    # 在这个实现中，'ProductA' 作为参考产品名称被记录，但 TC 权重计算实际上不依赖于参考产品
    calculate_merged_product(
        data=data_dict, 
        reference_product_name="ProductA", 
        output_filename=output_file
    )

    # 清理临时文件
    for f in temp_files:
        os.remove(f)
    if os.path.exists(output_file):
        print(f"Merging complete. Output saved to {output_file}")
        # os.remove(output_file) # 可选：清理输出文件
    else:
        print("Merging failed.")

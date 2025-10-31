#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ET产品数据集元数据分析工具
遍历指定路径下的NetCDF文件，提取并整理关键信息
"""

import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 尝试导入tqdm进度条，如果没有则使用简单的进度显示
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: 安装tqdm可以显示更好的进度条 (pip install tqdm)")
    print()

def get_temporal_resolution(time_coord):
    """
    计算时间分辨率
    """
    if len(time_coord) < 2:
        return "单时刻"
    
    # 计算时间差
    time_diff = pd.to_datetime(time_coord[1:].values) - pd.to_datetime(time_coord[:-1].values)
    median_diff = pd.Series(time_diff).median()
    
    days = median_diff.days
    
    if days == 0:
        return "hourly"
    elif days == 1:
        return "daily"
    elif 7 <= days <= 10:
        return "8-day"
    elif 10 <= days <= 16:
        return "16-day"
    elif 28 <= days <= 31:
        return "monthly"
    elif 365 <= days <= 366:
        return "yearly"
    else:
        return f"{days}天"

def get_spatial_resolution(lat, lon):
    """
    计算空间分辨率（度）
    """
    if len(lat) < 2 or len(lon) < 2:
        return "未知"
    
    lat_res = abs(float(lat[1] - lat[0]))
    lon_res = abs(float(lon[1] - lon[0]))
    
    # 转换为公里（粗略估计）
    lat_km = lat_res * 111  # 1度纬度约111km
    lon_km = lon_res * 111 * np.cos(np.radians(np.mean(lat)))  # 考虑纬度
    
    return f"{lat_res:.4f}° × {lon_res:.4f}° (~{lat_km:.1f}km × {lon_km:.1f}km)"

def get_spatial_extent(lat, lon):
    """
    获取空间范围
    """
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    
    return f"[{lat_min:.2f}°N, {lat_max:.2f}°N] × [{lon_min:.2f}°E, {lon_max:.2f}°E]"

def identify_et_type(variables, dataset_name):
    """
    识别ET数据类型
    """
    var_lower = [v.lower() for v in variables]
    dataset_lower = dataset_name.lower()
    
    et_types = []
    
    # 检查是否包含总ET
    if any(v in var_lower for v in ['et', 'evapotranspiration', 'evaporation', 'le', 'et_total']):
        et_types.append("总ET")
    
    # 检查是否包含PET
    if any(v in var_lower for v in ['pet', 'potential_evapotranspiration', 'eto', 'etp']) or 'pet' in dataset_lower:
        et_types.append("PET")
    
    # 检查是否包含分量
    if any(v in var_lower for v in ['evaporation_from_canopy', 'ec', 'transpiration', 'et_c', 'et_i']):
        et_types.append("包含分量")
    
    if any(v in var_lower for v in ['transpiration', 'et_t', 'et_veg']):
        et_types.append("蒸腾")
    
    if any(v in var_lower for v in ['soil_evaporation', 'es', 'et_s', 'evap_bare']):
        et_types.append("土壤蒸发")
    
    if any(v in var_lower for v in ['interception', 'ei', 'et_i', 'evap_canopy']):
        et_types.append("冠层截留")
    
    if not et_types:
        et_types.append("未分类")
    
    return " + ".join(et_types)

def analyze_netcdf_file(file_path):
    """
    分析单个NetCDF文件
    """
    try:
        ds = xr.open_dataset(file_path, decode_times=True)
        
        # 识别坐标变量
        time_coord = None
        lat_coord = None
        lon_coord = None
        
        for coord_name in ['time', 't', 'Time']:
            if coord_name in ds.coords or coord_name in ds.dims:
                time_coord = ds.coords[coord_name] if coord_name in ds.coords else ds[coord_name]
                break
        
        for lat_name in ['lat', 'latitude', 'y', 'Latitude']:
            if lat_name in ds.coords or lat_name in ds.dims:
                lat_coord = ds.coords[lat_name] if lat_name in ds.coords else ds[lat_name]
                break
        
        for lon_name in ['lon', 'longitude', 'x', 'Longitude']:
            if lon_name in ds.coords or lon_name in ds.dims:
                lon_coord = ds.coords[lon_name] if lon_name in ds.coords else ds[lon_name]
                break
        
        # 提取数据变量（排除坐标）
        data_vars = [v for v in ds.data_vars]
        
        info = {
            'file_path': file_path,
            'variables': ', '.join(data_vars[:10]) + ('...' if len(data_vars) > 10 else ''),
            'num_variables': len(data_vars),
            'temporal_resolution': 'N/A',
            'time_range': 'N/A',
            'time_steps': 0,
            'spatial_resolution': 'N/A',
            'spatial_extent': 'N/A',
            'grid_shape': 'N/A',
            'et_type': 'N/A'
        }
        
        # 时间信息
        if time_coord is not None:
            info['time_steps'] = len(time_coord)
            info['temporal_resolution'] = get_temporal_resolution(time_coord)
            
            time_start = pd.to_datetime(time_coord[0].values)
            time_end = pd.to_datetime(time_coord[-1].values)
            info['time_range'] = f"{time_start.strftime('%Y-%m-%d')} 至 {time_end.strftime('%Y-%m-%d')}"
        
        # 空间信息
        if lat_coord is not None and lon_coord is not None:
            info['spatial_resolution'] = get_spatial_resolution(lat_coord.values, lon_coord.values)
            info['spatial_extent'] = get_spatial_extent(lat_coord.values, lon_coord.values)
            info['grid_shape'] = f"{len(lat_coord)} × {len(lon_coord)}"
        
        # ET类型识别
        dataset_name = os.path.basename(os.path.dirname(file_path))
        info['et_type'] = identify_et_type(data_vars, dataset_name)
        
        ds.close()
        return info
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e)
        }

def scan_directory(root_path):
    """
    扫描目录下的所有NetCDF文件
    """
    results = []
    root_path = Path(root_path)
    
    # 查找所有nc文件
    print("正在搜索NetCDF文件...")
    nc_files = list(root_path.rglob('*.nc'))
    nc4_files = list(root_path.rglob('*.nc4'))
    all_files = nc_files + nc4_files
    
    print(f"找到 {len(all_files)} 个NetCDF文件")
    print("开始分析...\n")
    
    # 使用进度条
    if TQDM_AVAILABLE:
        file_iterator = tqdm(all_files, desc="分析进度", unit="文件", 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        file_iterator = all_files
    
    for idx, file_path in enumerate(file_iterator, 1):
        if not TQDM_AVAILABLE:
            # 简单的进度显示
            print(f"[{idx}/{len(all_files)}] ({idx*100//len(all_files):3d}%) 正在分析: {file_path.name}")
        else:
            # tqdm会自动更新进度条，设置描述显示当前文件
            file_iterator.set_postfix_str(f"当前: {file_path.name[:40]}")
        
        dataset_name = file_path.parent.name
        file_info = analyze_netcdf_file(str(file_path))
        file_info['dataset'] = dataset_name
        file_info['filename'] = file_path.name
        
        results.append(file_info)
    
    if TQDM_AVAILABLE:
        print()  # 进度条结束后换行
    
    return results

def generate_summary(results):
    """
    生成数据集级别的汇总
    """
    df = pd.DataFrame(results)
    
    # 按数据集分组汇总
    summary = []
    
    for dataset in df['dataset'].unique():
        dataset_files = df[df['dataset'] == dataset]
        
        if 'error' not in dataset_files.columns or dataset_files['error'].isna().all():
            summary_info = {
                '数据集名称': dataset,
                '文件数量': len(dataset_files),
                'ET类型': dataset_files['et_type'].mode()[0] if len(dataset_files['et_type'].mode()) > 0 else 'N/A',
                '时间分辨率': dataset_files['temporal_resolution'].mode()[0] if 'temporal_resolution' in dataset_files else 'N/A',
                '空间分辨率': dataset_files['spatial_resolution'].mode()[0] if 'spatial_resolution' in dataset_files else 'N/A',
                '空间范围': dataset_files['spatial_extent'].mode()[0] if 'spatial_extent' in dataset_files else 'N/A',
                '主要变量': dataset_files['variables'].iloc[0] if 'variables' in dataset_files else 'N/A',
                '时间跨度': dataset_files['time_range'].iloc[0] if 'time_range' in dataset_files else 'N/A',
                '修改日期': 'N/A'  # 这个可以从文件系统获取
            }
        else:
            summary_info = {
                '数据集名称': dataset,
                '文件数量': len(dataset_files),
                '状态': '读取错误'
            }
        
        summary.append(summary_info)
    
    return pd.DataFrame(summary)

def main():
    """
    主函数
    """
    # 设置数据路径
    data_path = r"Z:\Evaporation_Flux"
    
    # 获取输出目录（改为数据目录）
    output_dir = Path(data_path)
    
    print("=" * 80)
    print("ET产品数据集元数据分析工具")
    print("=" * 80)
    print(f"\n分析路径: {data_path}")
    print(f"输出路径: {output_dir}\n")
    
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"错误: 路径 {data_path} 不存在!")
        print("请修改脚本中的 data_path 变量为正确的路径")
        return
    
    # 扫描并分析
    results = scan_directory(data_path)
    
    # 生成详细报告
    print("\n" + "=" * 80)
    print("生成报告...")
    print("=" * 80 + "\n")
    
    df_detail = pd.DataFrame(results)
    df_summary = generate_summary(results)
    
    # 保存结果到脚本所在目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细信息
    detail_file = output_dir / f"ET_products_detail_{timestamp}.csv"
    df_detail.to_csv(detail_file, index=False, encoding='utf-8-sig')
    print(f"✓ 详细信息已保存至: {detail_file}")
    
    # 保存汇总信息
    summary_file = output_dir / f"ET_products_summary_{timestamp}.csv"
    df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✓ 汇总信息已保存至: {summary_file}")
    
    # 保存为Excel（如果安装了openpyxl）
    try:
        excel_file = output_dir / f"ET_products_report_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='数据集汇总', index=False)
            df_detail.to_excel(writer, sheet_name='详细信息', index=False)
        print(f"✓ Excel报告已保存至: {excel_file}")
    except ImportError:
        print("  (跳过Excel输出，需要安装 openpyxl: pip install openpyxl)")
    
    # 打印汇总统计
    print("\n" + "=" * 80)
    print("数据集汇总统计")
    print("=" * 80)
    print(f"\n总数据集数量: {len(df_summary)}")
    print(f"总文件数量: {len(df_detail)}")
    
    if 'et_type' in df_summary.columns:
        print("\nET类型分布:")
        print(df_summary['ET类型'].value_counts().to_string())
    
    if 'temporal_resolution' in df_detail.columns:
        print("\n时间分辨率分布:")
        print(df_detail['temporal_resolution'].value_counts().to_string())
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()

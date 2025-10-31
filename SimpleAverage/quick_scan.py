#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ET产品数据集快速扫描工具
快速查看文件夹结构和文件统计，不深入读取NetCDF内容
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import defaultdict

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: 安装tqdm可以显示更好的进度条 (pip install tqdm)")
    print()

def scan_directory_structure(root_path):
    """
    快速扫描目录结构
    """
    results = []
    root_path = Path(root_path)
    
    # 获取所有子文件夹（数据集）
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    print(f"找到 {len(subdirs)} 个数据集文件夹\n")
    
    # 使用进度条
    if TQDM_AVAILABLE:
        subdir_iterator = tqdm(subdirs, desc="扫描进度", unit="数据集")
    else:
        subdir_iterator = subdirs
    
    for idx, subdir in enumerate(subdir_iterator, 1):
        dataset_name = subdir.name
        
        # 统计文件
        nc_files = list(subdir.glob('*.nc'))
        nc4_files = list(subdir.glob('*.nc4'))
        all_nc_files = nc_files + nc4_files
        
        # 统计子文件夹
        sub_folders = [d for d in subdir.iterdir() if d.is_dir()]
        
        # 获取修改日期（最新文件的修改时间）
        if all_nc_files:
            latest_file = max(all_nc_files, key=lambda x: x.stat().st_mtime)
            mod_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
            mod_date = mod_time.strftime('%Y-%m-%d %H:%M')
        else:
            mod_date = 'N/A'
        
        # 计算总大小
        total_size = sum(f.stat().st_size for f in all_nc_files)
        size_gb = total_size / (1024**3)
        
        info = {
            '数据集名称': dataset_name,
            'NC文件数量': len(all_nc_files),
            '子文件夹数量': len(sub_folders),
            '总大小(GB)': f"{size_gb:.2f}",
            '最新修改日期': mod_date,
            '文件路径': str(subdir)
        }
        
        results.append(info)
        
        if not TQDM_AVAILABLE:
            print(f"[{idx}/{len(subdirs)}] {dataset_name:40s} - {len(all_nc_files):3d} 个文件, {size_gb:6.2f} GB")
        else:
            subdir_iterator.set_postfix_str(f"{len(all_nc_files)} 文件, {size_gb:.2f}GB")
    
    if TQDM_AVAILABLE:
        print()  # 进度条结束后换行
    
    return results

def identify_potential_type(dataset_name):
    """
    根据数据集名称推测类型
    """
    name_lower = dataset_name.lower()
    
    types = []
    
    # 知名数据集识别
    dataset_mapping = {
        'era5': 'ERA5再分析',
        'gleam': 'GLEAM',
        'modis': 'MODIS',
        'glass': 'GLASS',
        'gldas': 'GLDAS',
        'fluxcom': 'FLUXCOM',
        'bess': 'BESS',
        'camele': 'CAMELE',
        'pet': 'PET数据',
        'flux': '通量数据',
        'synthesis': '集成产品'
    }
    
    for key, value in dataset_mapping.items():
        if key in name_lower:
            types.append(value)
    
    # ET类型识别
    if 'pet' in name_lower or 'potential' in name_lower:
        types.append('PET')
    elif 'et' in name_lower:
        types.append('总ET')
    
    return ' / '.join(types) if types else '未识别'

def main():
    """
    主函数
    """
    # 设置数据路径
    data_path = r"Z:\Evaporation_Flux"
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.absolute()
    
    print("=" * 80)
    print("ET产品数据集快速扫描工具")
    print("=" * 80)
    print(f"\n扫描路径: {data_path}")
    print(f"输出路径: {script_dir}\n")
    
    # 检查路径
    if not os.path.exists(data_path):
        print(f"错误: 路径 {data_path} 不存在!")
        print("请修改脚本中的 data_path 变量")
        return
    
    # 扫描
    results = scan_directory_structure(data_path)
    
    # 添加类型推测
    for r in results:
        r['推测类型'] = identify_potential_type(r['数据集名称'])
    
    # 生成报告
    print("\n" + "=" * 80)
    print("生成快速扫描报告...")
    print("=" * 80 + "\n")
    
    df = pd.DataFrame(results)
    
    # 保存结果到脚本所在目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = script_dir / f"ET_quick_scan_{timestamp}.csv"
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 扫描报告已保存至: {output_file}")
    
    # 尝试保存Excel
    try:
        excel_file = script_dir / f"ET_quick_scan_{timestamp}.xlsx"
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✓ Excel报告已保存至: {excel_file}")
    except ImportError:
        print("  (跳过Excel输出，需要安装 openpyxl: pip install openpyxl)")
    
    # 打印统计
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    print(f"\n总数据集数量: {len(df)}")
    print(f"总文件数量: {df['NC文件数量'].sum()}")
    print(f"总数据大小: {df['总大小(GB)'].astype(float).sum():.2f} GB")
    
    print("\n推测类型分布:")
    print(df['推测类型'].value_counts().to_string())
    
    print("\n文件数量最多的前10个数据集:")
    top10 = df.nlargest(10, 'NC文件数量')[['数据集名称', 'NC文件数量', '总大小(GB)']]
    print(top10.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("快速扫描完成!")
    print("=" * 80)
    print("\n提示: 运行 analyze_et_products.py 可以获取详细的NetCDF元数据分析")

if __name__ == "__main__":
    main()

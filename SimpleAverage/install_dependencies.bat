@echo off
REM ET产品分析工具 - 依赖安装脚本
REM 自动安装所有必需和推荐的Python包

echo ========================================
echo ET产品分析工具 - 依赖安装
echo ========================================
echo.

echo 正在安装必需的包...
pip install xarray netCDF4 pandas numpy openpyxl

echo.
echo 正在安装推荐的包（进度条）...
pip install tqdm

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 现在你可以运行分析脚本了：
echo   python analyze_et_products.py
echo   或
echo   python quick_scan.py
echo.
pause

@echo off

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 激活 Conda 环境
call C:\Fluids\miniconda3\condabin\conda.bat activate hkc

REM 启动程序
python main.py

pause
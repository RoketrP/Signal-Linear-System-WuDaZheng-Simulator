@echo off
chcp 65001 >nul
cd /d %~dp0
echo 正在安装依赖并启动 Signal-Linear-System-WuDazheng...
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause

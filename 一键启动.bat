@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在准备独立环境并启动 Signal-Linear-System-WuDazheng...
if not exist ".venv\Scripts\python.exe" (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
    if errorlevel 1 goto :python_error
    python -m venv .venv
    if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :failed
pause
exit /b 0

:python_error
echo 请先安装 Python 3.10 或更高版本，并勾选 Add Python to PATH。
pause
exit /b 1

:failed
echo 启动失败，请查看上方错误信息。依赖安装失败时不会继续启动。
pause
exit /b 1

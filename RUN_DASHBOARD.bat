@echo off
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m pip install -r dashboard_requirements.txt
    "venv\Scripts\python.exe" -m streamlit run dashboard_app.py
) else (
    python -m pip install -r dashboard_requirements.txt
    python -m streamlit run dashboard_app.py
)
pause

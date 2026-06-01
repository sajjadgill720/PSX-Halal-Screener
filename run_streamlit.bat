@echo off
REM Run Streamlit using project virtual environment
REM Double-click or run from this repo root
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0dashboard\app.py"
pause

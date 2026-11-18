@echo off
echo Starting AgriAI Local Environment...
echo ===================================

cd /d "%~dp0"

:: Start backend in a new command window
echo Starting FastAPI Backend...
start "AgriAI Backend (FastAPI)" cmd /k ".\venv\Scripts\activate.bat && set PYTHONPATH=backend && python -m uvicorn api:app --host 127.0.0.1 --port 8000"

:: Wait 3 seconds for backend to initialize
ping 127.0.0.1 -n 4 > nul

:: Start frontend in a new command window
echo Starting Streamlit Frontend...
start "AgriAI Frontend (Streamlit)" cmd /k ".\venv\Scripts\activate.bat && streamlit run frontend/app.py"

echo ===================================
echo Both Backend (port 8000) and Frontend (port 8501) are starting!
echo You can close this window now.
pause

@echo off
echo ============================================================
echo Starting Backend Job Application Engine
echo ============================================================

:: 1. Start FastAPI app server in the background
echo [1/2] Launching FastAPI App Server on http://127.0.0.1:8000...
start /b uvicorn app.main:app --host 127.0.0.1 --port 8000

:: Wait 5 seconds for server startup
timeout /t 5 /nobreak >nul

:: 2. Run job discovery and auto-applier in backend mode
echo [2/2] Running Nationwide Job Search & Automation...
python run_job_search.py

echo ============================================================
echo Process finished!
echo ============================================================
pause

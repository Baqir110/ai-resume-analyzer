@echo off
echo Stopping any running Edge processes...
taskkill /F /IM msedge.exe /T 2>nul
timeout /t 3 /nobreak >nul

REM Find the Edge binary
set "EDGE="
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    set "EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    set "EDGE=C:\Program Files\Microsoft\Edge\Application\msedge.exe"
)

if "%EDGE%"=="" (
    echo ERROR: Microsoft Edge not found. Install it or edit this script.
    pause
    exit /b 1
)

echo Launching Edge with remote debugging on port 9222...
echo   Binary: %EDGE%
start "" "%EDGE%" --remote-debugging-port=9222

echo.
echo Edge is now running with remote debugging enabled.
echo Log into any ATS portals you need, then leave Edge open.
echo You can now start the dashboard with: python run.py
timeout /t 5 /nobreak >nul

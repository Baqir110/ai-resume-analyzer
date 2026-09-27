@echo off
setlocal
if /I not "%ALLOW_BROWSER_PROFILE_SETUP%"=="true" (
  echo Browser profile setup is disabled. Set ALLOW_BROWSER_PROFILE_SETUP=true explicitly.
  exit /b 1
)

set "BRAVE="
if exist "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" (
  set "BRAVE=C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
) else if exist "%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe" (
  set "BRAVE=%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"
)
if "%BRAVE%"=="" (
  echo ERROR: Brave browser not found.
  exit /b 1
)

set "PROFILE=%CD%\data\browser_profiles\manual-brave"
mkdir "%PROFILE%" 2>nul
echo Launching an isolated Brave profile with loopback-only CDP...
start "" "%BRAVE%" --user-data-dir="%PROFILE%" --remote-debugging-address=127.0.0.1 --remote-debugging-port=9222 --no-first-run --no-default-browser-check
echo Do not expose port 9222 beyond this machine.
endlocal

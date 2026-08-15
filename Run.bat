@echo off
:: Change to the directory where the batch file is located
cd /d "%~dp0"

:: Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process 'cmd.exe' -ArgumentList '/c %~f0' -Verb RunAs"
    exit /b
)

echo Starting UniClaudeProxy with administrator privileges...
set "HOST=127.0.0.1"
set "PORT=9223"

if exist global.json (
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$cfg = Get-Content -Raw -Path 'global.json' | ConvertFrom-Json; if ($cfg.server.host) { $cfg.server.host } else { '127.0.0.1' }"`) do set "HOST=%%i"
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$cfg = Get-Content -Raw -Path 'global.json' | ConvertFrom-Json; if ($cfg.server.port) { $cfg.server.port } else { 9223 }"`) do set "PORT=%%i"
)

echo Using host=%HOST% port=%PORT%
python -m uvicorn app.main:app --host %HOST% --port %PORT% --reload
pause

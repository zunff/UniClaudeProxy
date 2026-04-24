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
set "PORT=8000"

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$cfg = Get-Content -Raw -Path 'config.json' | ConvertFrom-Json; $cfg.server.host"`) do set "HOST=%%i"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$cfg = Get-Content -Raw -Path 'config.json' | ConvertFrom-Json; $cfg.server.port"`) do set "PORT=%%i"

echo Using host=%HOST% port=%PORT%
python -m uvicorn app.main:app --host %HOST% --port %PORT% --reload
pause

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
python -m uvicorn app.main:app --host 127.0.0.1 --port 9223 --reload
pause

@echo off
REM Start the UniClaudeProxy admin dashboard on Windows.
setlocal

set SCRIPT_DIR=%~dp0
set ADMIN_DIR=%SCRIPT_DIR%admin-dashboard
cd /d "%ADMIN_DIR%"

if not exist "node_modules" (
  echo [admin] Installing dependencies...
  where pnpm >nul 2>nul
  if %ERRORLEVEL%==0 (
    call pnpm install
  ) else (
    call npm install
  )
  if errorlevel 1 (
    echo Error: dependency install failed.
    exit /b 1
  )
)

echo.
echo ===========================================
echo   UniClaudeProxy Admin Dashboard
echo   Frontend: http://127.0.0.1:5173
echo   Proxy API (must be running): 127.0.0.1:10388
echo ===========================================
echo.

where pnpm >nul 2>nul
if %ERRORLEVEL%==0 (
  call pnpm dev
) else (
  call npm run dev
)

endlocal

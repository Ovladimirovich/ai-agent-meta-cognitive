@echo off

:: Set console to UTF-8
chcp 65001 >nul

echo ====================
echo Starting System...
echo ====================

:: Check Node.js
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed. Please install Node.js
    pause
    exit /b 1
)

:: Stop any running services
echo.
echo [1/3] Stopping any running services...
call "%~dp0stop_simple.bat" silent

:: Start backend
echo.
echo [2/3] Starting backend...
start "Backend" cmd /k "@echo off && cd /d "%~dp0" && call venv\Scripts\activate.bat && echo Starting backend... && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload && pause"

:: Start frontend
echo.
echo [3/3] Starting frontend...
if exist "%~dp0frontend\package.json" (
    start "Frontend" cmd /k "@echo off && cd /d "%~dp0frontend" && echo Installing dependencies... && call npm install && echo Starting frontend... && npm run dev -- --host localhost --port 3000 && pause"
) else (
    echo [ERROR] Frontend directory not found or package.json is missing
    pause
    exit /b 1
)

echo.
echo ====================
echo System started!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo ====================

timeout /t 5 >nul
start "" "http://localhost:3000"

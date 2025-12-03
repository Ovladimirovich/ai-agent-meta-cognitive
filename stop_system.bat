@echo off
echo Stopping AI Agent Meta Cognitive System...

echo Checking and killing processes on ports...

REM Освобождение порта 8000 (main application - uvicorn)
echo Checking port 8000 (AI Agent API)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    echo Found process %%a listening on port 8000 - killing...
    taskkill /PID %%a /F >nul 2>&1
    if %errorlevel%==0 (
        echo Successfully killed process %%a on port 8000
    ) else (
        echo Failed to kill process %%a on port 8000
    )
)
if %errorlevel% neq 0 echo No processes found on port 8000

REM Освобождение порта 5432 (PostgreSQL - if running locally)
echo Checking port 5432 (PostgreSQL)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5432') do (
    echo Found process %%a listening on port 5432 - killing...
    taskkill /PID %%a /F >nul 2>&1
    if %errorlevel%==0 (
        echo Successfully killed process %%a on port 5432
    ) else (
        echo Failed to kill process %%a on port 5432
    )
)
if %errorlevel% neq 0 echo No processes found on port 5432

REM Освобождение порта 6379 (Redis - if running locally)
echo Checking port 6379 (Redis)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :6379') do (
    echo Found process %%a listening on port 6379 - killing...
    taskkill /PID %%a /F >nul 2>&1
    if %errorlevel%==0 (
        echo Successfully killed process %%a on port 6379
    ) else (
        echo Failed to kill process %%a on port 6379
    )
)
if %errorlevel% neq 0 echo No processes found on port 6379

REM Освобождение порта 8001 (ChromaDB - if running locally)
echo Checking port 8001 (ChromaDB)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001') do (
    echo Found process %%a listening on port 8001 - killing...
    taskkill /PID %%a /F >nul 2>&1
    if %errorlevel%==0 (
        echo Successfully killed process %%a on port 8001
    ) else (
        echo Failed to kill process %%a on port 8001
    )
)
if %errorlevel% neq 0 echo No processes found on port 8001

REM Освобождение порта 3000 (Frontend - Vite development server)
echo Checking port 3000 (Frontend)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :300') do (
    echo Found process %%a listening on port 3000 - killing...
    taskkill /PID %%a /F >nul 2>&1
    if %errorlevel%==0 (
        echo Successfully killed process %%a on port 3000
    ) else (
        echo Failed to kill process %%a on port 3000
    )
)
if %errorlevel% neq 0 echo No processes found on port 3000

echo.
echo System stopped and ports released successfully!
echo Ports checked: 8000 (API), 3000 (Frontend), 5432 (PostgreSQL), 6379 (Redis), 8001 (ChromaDB)
pause

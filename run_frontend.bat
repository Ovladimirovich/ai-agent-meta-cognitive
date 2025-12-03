@echo off
REM Простой скрипт запуска фронтенда

echo Проверка наличия Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Node.js не найден
    pause
    exit /b 1
)

echo Проверка наличия директории frontend...
if not exist "frontend" (
    echo [ОШИБКА] Директория frontend не найдена
    pause
    exit /b 1
)

echo Переход в директорию frontend и запуск сервера...
cd frontend

echo Запуск фронтенд-сервера...
start "Frontend Server" cmd /k "npm run dev -- --host localhost --port 3000"

echo.
echo Frontend сервер запущен в отдельном окне.
echo Откройте браузер по адресу http://localhost:3000 для доступа к приложению
echo Нажмите любую клавишу для закрытия этого окна...
pause >nul
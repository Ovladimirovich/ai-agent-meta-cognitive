@echo off
setlocal enabledelayedexpansion

echo ====================================
echo Запуск AI Agent Meta Cognitive System
echo ====================================

:: Получаем текущую директорию
set "CURRENT_DIR=%~dp0"
cd /d "%CURRENT_DIR%"

:: Проверка наличия Node.js
echo.
echo [1/5] Проверка Node.js...
where node >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Node.js не найден. Пожалуйста, установите Node.js
    pause
    exit /b 1
)

:: Проверка и освобождение портов
echo.
echo [2/5] Проверка и освобождение портов...
if exist "%CURRENT_DIR%stop_all.bat" (
    call "%CURRENT_DIR%stop_all.bat" silent
) else (
    echo [ПРЕДУПРЕЖДЕНИЕ] Файл stop_all.bat не найден, пропускаем освобождение портов
)

:: Проверка виртуального окружения
echo.
echo [3/5] Проверка виртуального окружения...
if not exist "%CURRENT_DIR%venv\Scripts\python.exe" (
    echo [ОШИБКА] Виртуальное окружение не найдено. Запустите activate_and_install.bat
    pause
    exit /b 1
)

:: Запуск бэкенда
echo.
echo [4/5] Запуск бэкенда...
start "Backend Server" cmd /k "@echo off && cd /d "%CURRENT_DIR%" && echo Запуск бэкенда... && call venv\Scripts\activate.bat && echo Виртуальное окружение активировано && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload && pause"

:: Запуск фронтенда
echo.
echo [5/5] Запуск фронтенда...
if exist "%CURRENT_DIR%frontend" (
    start "Frontend Server" cmd /k "@echo off && cd /d "%CURRENT_DIR%frontend" && echo Установка зависимостей... && call npm install && echo Запуск фронтенда... && npm run dev -- --host localhost --port 3000 && pause"
) else (
    echo [ОШИБКА] Директория frontend не найдена
    pause
    exit /b 1
)

:: Ожидание запуска серверов
echo.
echo Ожидание запуска серверов...
timeout /t 10 /nobreak >nul

:: Открытие фронтенда в браузере
echo.
echo Открытие приложения в браузере...
start "" "http://localhost:3000"

echo.
echo ===================================
echo Система успешно запущена!
echo Бэкенд: http://localhost:8000
echo Фронтенд: http://localhost:3000
echo ===================================

endlocal

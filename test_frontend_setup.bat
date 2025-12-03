@echo off
echo Тестирование компонентов для запуска фронтенда...
echo.

REM Проверка Node.js
echo 1. Проверка Node.js...
node --version
if errorlevel 1 (
    echo [ОШИБКА] Node.js не найден
    goto error
) else (
    echo [ОК] Node.js найден
)

REM Проверка npm
echo.
echo 2. Проверка npm...
npm --version
if errorlevel 1 (
    echo [ОШИБКА] npm не найден
    goto error
) else (
    echo [ОК] npm найден
)

REM Проверка директории frontend
echo.
echo 3. Проверка директории frontend...
if not exist "frontend" (
    echo [ОШИБКА] Директория frontend не найдена
    goto error
) else (
    echo [ОК] Директория frontend найдена
)

REM Проверка package.json в директории frontend
echo.
echo 4. Проверка package.json в директории frontend...
cd frontend
if not exist "package.json" (
    echo [ОШИБКА] package.json не найден в директории frontend
    cd ..
    goto error
) else (
    echo [ОК] package.json найден
)
cd ..

echo.
echo Все проверки пройдены успешно!
echo.
echo Для запуска фронтенда выполните: cd frontend && npm run dev
echo.

:error
echo.
echo Нажмите любую клавишу для закрытия окна...
pause >nul

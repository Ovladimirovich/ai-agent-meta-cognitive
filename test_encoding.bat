@echo off
chcp 1251 >nul

echo ===================================
echo Тест кодировки и запуска скриптов
echo ===================================

echo.
echo 1. Проверка кодировки...
echo Текст на русском: Привет, мир!

echo.
echo 2. Запуск stop_all.bat...
call "%~dp0stop_all.bat"

echo.
echo 3. Запуск start_all.bat...
call "%~dp0start_all.bat"

echo.
echo Тестирование завершено.
pause

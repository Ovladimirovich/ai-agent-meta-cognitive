# PowerShell скрипт для запуска фронтенда AI-агента
# Этот скрипт устанавливает зависимости и запускает фронтенд на порту 300

Write-Host "Запуск фронтенда AI-агента..."

# Проверяем наличие Node.js
try {
    $nodeVersion = & node --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ошибка: Node.js не найден. Пожалуйста, установите Node.js перед запуском."
        Read-Host "Нажмите Enter для выхода"
        exit 1
    }
} catch {
    Write-Host "Ошибка: Node.js не найден. Пожалуйста, установите Node.js перед запуском."
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

# Проверяем наличие npm
try {
    $npmVersion = & npm --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ошибка: npm не найден. Пожалуйста, установите npm перед запуском."
        Read-Host "Нажмите Enter для выхода"
        exit 1
    }
} catch {
    Write-Host "Ошибка: npm не найден. Пожалуйста, установите npm перед запуском."
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

# Переходим в папку frontend
Set-Location frontend

# Проверяем и устанавливаем зависимости, если нужно
if (!(Test-Path "node_modules")) {
    Write-Host "Установка зависимостей..."
    & npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ошибка при установке зависимостей"
        Read-Host "Нажмите Enter для выхода"
        exit 1
    }
} else {
    Write-Host "Зависимости уже установлены, пропускаем npm install"
}
} else {
    Write-Host "Зависимости уже установлены, пропускаем npm install"
}

# Добавляем паузу перед запуском фронтенда
Start-Sleep -Seconds 6

# Запускаем фронтенд
Write-Host "Запуск фронтенда на http://localhost:300"
Write-Host "ПРИМЕЧАНИЕ: Убедитесь, что бэкенд запущен на http://localhost:8000 для полной функциональности"

# Запускаем npm run dev в новом окне
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "npm run dev" -WindowStyle Normal

# Возвращаемся в исходную директорию
Set-Location ..

Write-Host "Фронтенд запущен в отдельном окне."
Read-Host "Нажмите Enter для выхода"

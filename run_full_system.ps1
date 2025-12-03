# PowerShell скрипт для запуска полной системы AI-агента (бэкенд + фронтенд)
# Этот скрипт запускает бэкенд на порту 8000 и фронтенд на порту 3000

Write-Host "Запуск полной системы AI-агента..."

# Проверяем наличие Node.js
$nodeMissing = $false
try {
    $nodeVersion = & node --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Предупреждение: Node.js не найден. Фронтенд не будет запущен."
        $nodeMissing = $true
    }
} catch {
    Write-Host "Предупреждение: Node.js не найден. Фронтенд не будет запущен."
    $nodeMissing = $true
}

# Проверяем наличие npm
$npmMissing = $false
try {
    $npmVersion = & npm --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Предупреждение: npm не найден. Фронтенд не будет запущен."
        $npmMissing = $true
    }
} catch {
    Write-Host "Предупреждение: npm не найден. Фронтенд не будет запущен."
    $npmMissing = $true
}

# Проверяем наличие Python
try {
    $pythonVersion = & python --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ошибка: Python не найден. Пожалуйста, установите Python перед запуском."
        Read-Host "Нажмите Enter для выхода"
        exit 1
    }
} catch {
    Write-Host "Ошибка: Python не найден. Пожалуйста, установите Python перед запуском."
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

# Проверяем наличие uvicorn
try {
    $uvicornCheck = & pip show uvicorn 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Установка uvicorn..."
        & pip install uvicorn
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Ошибка при установке uvicorn"
            Read-Host "Нажмите Enter для выхода"
            exit 1
        }
    }
} catch {
    Write-Host "Установка uvicorn..."
    & pip install uvicorn
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ошибка при установке uvicorn"
        Read-Host "Нажмите Enter для выхода"
        exit 1
    }
}

# Запускаем бэкенд в отдельном окне
Write-Host "Запуск бэкенда на http://localhost:8000"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload" -WindowStyle Normal

# Если Node.js и npm доступны, запускаем фронтенд
if (!$nodeMissing -and !$npmMissing) {
    Write-Host "Запуск фронтенда на http://localhost:3000"
    Set-Location frontend
    if (!(Test-Path "node_modules")) {
        Write-Host "Установка зависимостей фронтенда..."
        & npm install
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Ошибка при установке зависимостей фронтенда"
            Set-Location ..
            goto end
        }
    } else {
        Write-Host "Зависимости фронтенда уже установлены"
    }
    Set-Location ..
    Start-Sleep -Seconds 6
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "cd frontend && npm run dev" -WindowStyle Normal
} else {
    if ($nodeMissing) {
        Write-Host "Node.js не найден, фронтенд не будет запущен"
        Write-Host "Для запуска фронтенда установите Node.js и npm"
    }
    if ($npmMissing) {
        Write-Host "npm не найден, фронтенд не будет запущен"
        Write-Host "Для запуска фронтенда установите Node.js и npm"
    }
}

:end
Write-Host ""
Write-Host "Система запущена!"
Write-Host "Бэкенд: http://localhost:8000"
if (!$nodeMissing -and !$npmMissing) {
    Write-Host "Фронтенд: http://localhost:3000"
}
Write-Host ""
Read-Host "Нажмите Enter для завершения"

# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt requirements-dev.txt requirements-render.txt ./

# Устанавливаем зависимости - используем разные файлы в зависимости от среды
# По умолчанию используем основной requirements.txt
ARG REQUIREMENTS_FILE=requirements.txt
ENV REQUIREMENTS_FILE=${REQUIREMENTS_FILE}

# Устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

# Копируем исходный код
COPY . .

# Устанавливаем права на выполнение для скриптов запуска
RUN chmod +x /app/run_frontend_simple.bat /app/run_frontend.bat /app/run_frontend.ps1

# Экспортируем порт, который будет использоваться
EXPOSE $PORT 8000

# Команда для запуска приложения
CMD ["sh", "-c", "uvicorn api.main:app --host=0.0.0.0 --port=${PORT:-8000} --workers 4"]

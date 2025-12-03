# Multi-stage Dockerfile для frontend части AI Агента
# Оптимизирован для минимального размера и быстрой сборки

# ================================
# Stage 1: Build dependencies
# ================================
FROM node:18-alpine as dependencies

# Установка зависимостей
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production

# ================================
# Stage 2: Build frontend
# ================================
FROM node:18-alpine as build

WORKDIR /app

# Копирование исходного кода
COPY . .

# Установка зависимостей для сборки
RUN npm ci

# Сборка приложения
RUN npm run build

# ================================
# Stage 3: Runtime (production)
# ================================
FROM nginx:alpine as runtime

# Установка системных утилит для отладки
RUN apk add --no-cache curl

# Копирование собранного приложения из stage build
COPY --from=build /app/dist /usr/share/nginx/html

# Копирование конфигурации nginx
COPY nginx.conf /etc/nginx/nginx.conf

# Экспорт порта
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Запуск nginx
CMD ["nginx", "-g", "daemon off;"]

# ================================
# Stage 4: Development
# ================================
FROM node:18-alpine as development

WORKDIR /app

# Установка зависимостей
COPY package.json package-lock.json ./
RUN npm ci

# Копирование остального исходного кода
COPY . .

# Установка утилит для разработки
RUN apk add --no-cache git bash

# Экспорт порта для разработки
EXPOSE 5173

# Команда для запуска в режиме разработки
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
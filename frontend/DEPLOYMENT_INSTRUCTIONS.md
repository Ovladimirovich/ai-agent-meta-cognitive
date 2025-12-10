# Инструкции по деплою фронтенд-приложения

## Обновленная конфигурация

Для корректной работы приложения после обновления были внесены изменения в конфигурацию:

### 1. Конфигурация nginx (`nginx.conf`)

Обновлены прокси-настройки для корректной маршрутизации API-запросов:

```nginx
# API proxy to backend
location /api/ {
    proxy_pass http://ai-agent:8000/api/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_cache_bypass $http_upgrade;

    # CORS headers
    add_header Access-Control-Allow-Origin * always;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
    add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
    add_header Access-Control-Expose-Headers "Content-Length,Content-Range" always;

    # Handle preflight requests
    if ($request_method = 'OPTIONS') {
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
        add_header Content-Length 0;
        add_header Content-Type text/plain;
        return 204;
    }
}

# Additional proxy configuration for API endpoints without trailing slash
location /api {
    proxy_pass http://ai-agent:8000/api;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_cache_bypass $http_upgrade;

    # CORS headers
    add_header Access-Control-Allow-Origin * always;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
    add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
    add_header Access-Control-Expose-Headers "Content-Length,Content-Range" always;

    # Handle preflight requests
    if ($request_method = 'OPTIONS') {
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
        add_header Content-Length 0;
        add_header Content-Type text/plain;
        return 204;
    }
}
```

### 2. Переменные окружения

Обновлены переменные окружения для корректной работы API-клиента:

**`.env.production`**:
```
VITE_API_BASE_URL=/api
```

**`.env.example`**:
```
VITE_API_BASE_URL=/api
```

### 3. Конфигурация Vite (`vite.config.ts`)

Обновлена прокси-настройка для разработки:

```typescript
proxy: {
 '/api': {
    target: 'http://ai-agent:8000/api',
    changeOrigin: true,
    secure: false,
  }
}
```

### 4. API клиент (`src/shared/lib/apiClient.ts`)

Обновлена логика определения базового URL:

```typescript
const apiClient = new ApiClient({
  baseUrl: import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '/api' : 'http://localhost:8000/api')
});
```

## Процесс деплоя

1. Убедитесь, что все изменения внесены в репозиторий
2. Обновите переменные окружения в GitHub Pages или другом хостинге
3. Пересоберите приложение:
   ```bash
   cd frontend
   npm run build
   ```
4. Загрузите содержимое папки `dist` на хостинг

## Проверка работоспособности

После деплоя проверьте:

1. Загрузку главной страницы
2. Работу API-запросов (например, `/api/health`)
3. Отсутствие ошибок в консоли браузера
4. Корректность отображения данных от бэкенда

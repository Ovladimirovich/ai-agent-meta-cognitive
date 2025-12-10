# Система Health Check

## Обзор

Система health check предоставляет информацию о состоянии AI агента и его компонентов. Она включает в себя как базовые проверки системы, так и мета-когнитивные метрики агента.

## API эндпоинты

### Доступные эндпоинты

1. **GET** `/health` - базовая проверка здоровья системы
2. **GET** `/api/health` - проверка здоровья системы с совместимостью с фронтендом

Оба эндпоинта возвращают одинаковую структуру данных в формате JSON:
```json
{
  "status": "healthy|warning|error|unhealthy",
  "health_score": 0.85,
  "issues_count": 2,
  "last_check": "2023-12-10T06:30:46.548Z",
  "details": {},
  "cognitive_metrics": {
    "cognitiveLoad": 0.7,
    "confidenceLevel": 0.85,
    "processingSpeed": 0.9,
    "memoryUtilization": 0.65,
    "attentionSpan": 0.8,
    "decisionAccuracy": 0.92
  }
}
```

## Фронтенд интеграция

### API клиент

В файле `frontend/src/shared/lib/apiClient.ts` определен метод `getHealthStatus()` который автоматически определяет правильный URL в зависимости от окружения:
- В продакшене: использует `VITE_API_BASE_URL` или `https://ai-agent-meta-cognitive.onrender.com/api` по умолчанию
- В разработке: использует `http://localhost:8000/api`

### Компоненты

1. **SystemHealthMonitor** (`frontend/src/widgets/SystemHealthMonitor/SystemHealthMonitor.tsx`) - отображает общее состояние системы
2. **CognitiveHealthMonitor** (`frontend/src/widgets/CognitiveHealthMonitor/CognitiveHealthMonitor.tsx`) - отображает когнитивные метрики агента

## Типы данных

Типы данных определены в `frontend/src/shared/types/api.ts`:
- `HealthStatus` - основной тип для статуса здоровья
- `CognitiveHealthData` - тип для когнитивных метрик

## Устранение неполадок

### Ошибка 404 для /api/health

Если вы получаете ошибку 404 при обращении к `/api/health`, проверьте следующее:
1. Убедитесь, что бэкенд запущен и доступен по указанному URL
2. Проверьте настройки CORS в бэкенде
3. Убедитесь, что переменная `VITE_API_BASE_URL` правильно установлена в файле `.env`

### Проблемы с фронтендом

Если компоненты не отображают данные:
1. Проверьте, что API возвращает данные в правильном формате
2. Убедитесь, что типы данных соответствуют ожидаемым структурам
3. Проверьте консоль браузера на наличие ошибок

## Настройка для продакшена

Для корректной работы в продакшене:
1. Установите переменную `VITE_API_BASE_URL` в `.env.production` файле
2. Убедитесь, что ваш бэкенд доступен по указанному URL
3. Проверьте, что CORS настроен на бэкенде для разрешения запросов с вашего домена

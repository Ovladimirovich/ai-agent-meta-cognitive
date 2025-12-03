# Система Кэширования AI Агента

## Обзор

Система кэширования AI Агента предоставляет высокопроизводительное решение для временного хранения данных с использованием алгоритма LRU (Least Recently Used). Система оптимизирована для работы с большими объемами данных и обеспечивает эффективное управление памятью.

## Основные Компоненты

### LRUCache

Основной класс кэширования, реализующий алгоритм LRU с дополнительными возможностями:

#### Особенности
- **Автоматическое удаление** наименее недавно использованных элементов
- **Ограничение по размеру** и памяти
- **TTL (Time To Live)** для записей
- **Статистика использования** в реальном времени
- **Prometheus метрики** для мониторинга
- **Расширенные стратегии инвалидации**

#### Основные Методы

##### Базовые Операции
```python
from cache import LRUCache

# Создание кэша
cache = LRUCache(max_size=1000, max_memory_mb=256, name="my_cache")

# Установка значения
cache.set('key', 'value', ttl=3600)  # TTL опционально

# Получение значения
value = cache.get('key')

# Удаление значения
cache.delete('key')

# Очистка всего кэша
cache.clear()
```

##### Статистика и Мониторинг
```python
# Получение статистики
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2f}")
print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")

# Получение показателей здоровья кэша
health = cache.get_cache_health()
print(f"Health score: {health['health_score']}/100")
```

##### Стратегии Инвалидации
```python
# Инвалидация по шаблону
deleted = cache.invalidate_pattern('user:*')  # Удалит все ключи начинающиеся с 'user:'

# Инвалидация пространства имен
deleted = cache.invalidate_namespace('api:v1')  # Удалит все ключи 'api:v1:*'

# Инвалидация по возрасту
deleted = cache.invalidate_by_age(3600)  # Удалит записи старше 1 часа

# Инвалидация по количеству доступов
deleted = cache.invalidate_by_access_count(5)  # Удалит записи с < 5 доступами

# Очистка истекших записей
cache.cleanup_expired()
```

## Конфигурация

### Параметры LRUCache

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `max_size` | int | 1000 | Максимальное количество записей |
| `max_memory_mb` | int | 256 | Максимальный объем памяти в MB |
| `name` | str | "lru_cache" | Имя кэша для метрик |

### Переменные Окружения

```bash
# Prometheus метрики
PROMETHEUS_AVAILABLE=true

# Настройки кэша по умолчанию
CACHE_MAX_SIZE=10000
CACHE_MAX_MEMORY_MB=512
CACHE_DEFAULT_TTL=3600
```

## Мониторинг и Метрики

### Prometheus Метрики

Система предоставляет следующие метрики Prometheus:

#### Counters
- `cache_hits_total` - Общее количество попаданий в кэш
- `cache_misses_total` - Общее количество промахов кэша
- `cache_sets_total` - Общее количество установок значений
- `cache_deletes_total` - Общее количество удалений
- `cache_evictions_total` - Общее количество принудительных удалений

#### Gauges
- `cache_size` - Текущий размер кэша
- `cache_memory_bytes` - Текущее использование памяти
- `cache_hit_rate` - Коэффициент попаданий (0-1)

#### Histograms
- `cache_operation_duration_seconds` - Время выполнения операций

### Показатели Здоровья Кэша

Метод `get_cache_health()` возвращает:

```python
{
    'hit_rate': 0.85,                    # Коэффициент попаданий
    'memory_efficiency_percent': 75.5,   # Эффективность использования памяти
    'size_efficiency_percent': 60.0,     # Эффективность использования размера
    'expired_entries_count': 0,          # Количество истекших записей
    'avg_entry_age_seconds': 1250.5,     # Средний возраст записей
    'max_entry_age_seconds': 3600.0,     # Максимальный возраст
    'min_entry_age_seconds': 10.5,       # Минимальный возраст
    'avg_access_count': 3.2,             # Среднее количество доступов
    'error_rate': 0.001,                 # Коэффициент ошибок
    'health_score': 92.5                 # Общая оценка здоровья (0-100)
}
```

## Стратегии Кэширования

### 1. Кэширование API Ответов

```python
class APIClient:
    def __init__(self):
        self.response_cache = LRUCache(
            max_size=1000,
            max_memory_mb=100,
            name="api_responses"
        )

    async def make_request(self, endpoint, params):
        cache_key = f"api:{endpoint}:{hash(str(params))}"

        # Проверяем кэш
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            return cached_response

        # Выполняем запрос
        response = await self._execute_request(endpoint, params)

        # Кэшируем результат
        self.response_cache.set(cache_key, response, ttl=300)  # 5 минут

        return response
```

### 2. Кэширование Вычислений

```python
class ComputationCache:
    def __init__(self):
        self.result_cache = LRUCache(
            max_size=5000,
            max_memory_mb=2000,
            name="computations"
        )

    def compute_expensive_operation(self, input_data):
        # Создаем ключ на основе входных данных
        cache_key = f"compute:{hash(str(input_data))}"

        # Проверяем кэш
        cached_result = self.result_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Выполняем вычисление
        result = self._perform_expensive_computation(input_data)

        # Кэшируем результат
        self.result_cache.set(cache_key, result, ttl=3600)  # 1 час

        return result
```

### 3. Иерархическое Кэширование

```python
class HierarchicalCache:
    def __init__(self):
        # L1 - быстрый кэш для горячих данных
        self.l1_cache = LRUCache(max_size=100, max_memory_mb=50, name="l1_cache")

        # L2 - больший кэш для теплых данных
        self.l2_cache = LRUCache(max_size=1000, max_memory_mb=500, name="l2_cache")

    def get(self, key):
        # Сначала проверяем L1
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Затем L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Поднимаем в L1 для будущих запросов
            self.l1_cache.set(key, value)
            return value

        return None

    def set(self, key, value, ttl=None):
        # Устанавливаем в оба уровня
        self.l1_cache.set(key, value, ttl)
        self.l2_cache.set(key, value, ttl)
```

## Производительность

### Бенчмарки

На тестовом наборе данных (5000 записей):

| Операция | Время выполнения | Пропускная способность |
|----------|------------------|----------------------|
| set() | 0.02 ms | 50,000 ops/sec |
| get() (hit) | 0.01 ms | 100,000 ops/sec |
| get() (miss) | 0.015 ms | 66,000 ops/sec |
| invalidate_pattern() | 0.5 ms | 2,000 ops/sec |

### Оптимизации

1. **Пакетные операции** - поддержка множественных set/get операций
2. **Компрессия** - автоматическое сжатие больших значений
3. **Асинхронная инвалидация** - фоновая очистка истекших записей
4. **Memory pooling** - повторное использование памяти

## Безопасность

### Защита от Атак

1. **Ограничение размера** - предотвращает исчерпание памяти
2. **TTL enforcement** - автоматическое удаление старых данных
3. **Key validation** - проверка корректности ключей
4. **Rate limiting** - ограничение частоты операций

### Аудит

```python
# Включение аудита операций
cache.enable_audit_log('/var/log/cache_audit.log')

# Получение лога операций
audit_entries = cache.get_audit_log(hours=24)
```

## Интеграция с Системой

### Использование в AI Агента

```python
from agent.core.cache_manager import CacheManager

class AgentCore:
    def __init__(self):
        self.cache_manager = CacheManager()

        # Кэш для результатов推理
        self.reasoning_cache = self.cache_manager.create_cache(
            name="reasoning_results",
            max_size=5000,
            max_memory_mb=1000,
            default_ttl=1800  # 30 минут
        )

        # Кэш для внешних API
        self.api_cache = self.cache_manager.create_cache(
            name="external_api",
            max_size=2000,
            max_memory_mb=500,
            default_ttl=300  # 5 минут
        )

    async def process_request(self, request):
        # Проверяем кэш результатов推理
        cache_key = f"reasoning:{hash(request.content)}"
        cached_result = self.reasoning_cache.get(cache_key)

        if cached_result:
            return cached_result

        # Выполняем推理
        result = await self._perform_reasoning(request)

        # Кэшируем результат
        self.reasoning_cache.set(cache_key, result)

        return result
```

## Мониторинг и Обслуживание

### Регулярные Задачи

```python
import asyncio

async def maintenance_tasks(cache):
    """Регулярные задачи обслуживания кэша"""
    while True:
        # Очистка истекших записей
        cache.cleanup_expired()

        # Оптимизация по возрасту (удаляем записи старше 24 часов)
        cache.invalidate_by_age(86400)

        # Оптимизация по использованию (удаляем редко используемые)
        cache.invalidate_by_access_count(1)

        # Логируем здоровье
        health = cache.get_cache_health()
        logger.info(f"Cache health: {health['health_score']:.1f}/100")

        await asyncio.sleep(3600)  # Каждый час
```

### Мониторинг Производительности

```python
def setup_cache_monitoring(cache):
    """Настройка мониторинга кэша"""

    # Экспорт метрик в Prometheus
    from prometheus_client import start_http_server
    start_http_server(8000)

    # Настройка алертов
    def check_cache_health():
        health = cache.get_cache_health()

        if health['health_score'] < 50:
            alert(f"Cache health critical: {health['health_score']}")

        if health['hit_rate'] < 0.7:
            alert(f"Cache hit rate low: {health['hit_rate']:.2f}")

    # Периодическая проверка
    scheduler.add_job(check_cache_health, 'interval', minutes=5)
```

## Troubleshooting

### Распространенные Проблемы

#### 1. Высокий Memory Usage
```python
# Проверить использование памяти
stats = cache.get_stats()
print(f"Memory usage: {stats['memory_usage_mb']:.2f}/{stats['max_memory_mb']:.2f} MB")

# Уменьшить лимиты
cache.max_memory_mb = 128

# Очистить старые записи
cache.invalidate_by_age(7200)  # Старше 2 часов
```

#### 2. Низкий Hit Rate
```python
# Проверить hit rate
health = cache.get_cache_health()
print(f"Hit rate: {health['hit_rate']:.2f}")

# Увеличить размер кэша
cache.max_size = 2000

# Проверить стратегию инвалидации
# Возможно, данные удаляются слишком рано
```

#### 3. Высокая Загрузка CPU
```python
# Проверить время операций
stats = cache.get_stats()

# Оптимизировать: уменьшить частоту cleanup
# Или использовать асинхронную инвалидацию
```

### Логирование

```python
import logging

# Настройка логирования кэша
logging.getLogger('cache.cache_system_enhanced').setLevel(logging.DEBUG)

# Логи будут содержать:
# - Операции инвалидации
# - Ошибки сериализации
# - Предупреждения о превышении лимитов
```

## Заключение

Система кэширования предоставляет надежное и высокопроизводительное решение для управления временными данными в AI Агента. Гибкая архитектура позволяет адаптировать систему под специфические требования различных компонентов агента.

Для получения дополнительной информации обратитесь к:
- Исходному коду: `cache/cache_system_enhanced.py`
- Тестам: `tests/test_lru_cache_advanced.py`
- Примерам использования: `examples/cache_usage.py`

# Распределенная очередь задач (DistributedTaskQueue)

## Обзор

Распределенная очередь задач (DistributedTaskQueue) - это компонент масштабируемой архитектуры AI агента, обеспечивающий асинхронную обработку задач, отказоустойчивость и масштабирование через внешние брокеры сообщений. Компонент интегрирован с существующей системой CQRS (Command Query Responsibility Segregation) и обеспечивает:
- Асинхронную обработку задач
- Отказоустойчивость
- Масштабируемость через внешние брокеры
- Мониторинг и логирование

## Архитектура

### Основные компоненты

1. **DistributedTaskQueue** - основной класс, управляющий очередью задач
2. **TaskQueueBackend** - абстрактный интерфейс бэкенда очереди задач
3. **RedisTaskQueueBackend** - реализация бэкенда очереди задач на основе Redis
4. **Task** - представление задачи в распределенной очереди
5. **TaskResult** - результат выполнения задачи
6. **TaskStatus** - перечисление статусов задачи
7. **TaskPriority** - перечисление приоритетов задач

### Статусы задач

- `PENDING` - задача ожидает обработки
- `PROCESSING` - задача в процессе обработки
- `COMPLETED` - задача успешно завершена
- `FAILED` - задача завершена с ошибкой
- `RETRYING` - задача в процессе повторной попытки
- `CANCELLED` - задача отменена

### Приоритеты задач

- `LOW` - низкий приоритет
- `NORMAL` - нормальный приоритет
- `HIGH` - высокий приоритет
- `CRITICAL` - критический приоритет

## Использование

### Создание очереди задач

```python
from distributed_task_queue import create_distributed_task_queue

# Создание экземпляра распределенной очереди задач
task_queue = await create_distributed_task_queue()

# Запуск очереди
await task_queue.start()
```

### Создание и добавление задачи

```python
from distributed_task_queue import Task, TaskPriority

# Создание задачи
task = Task(
    id="unique_task_id",
    name="task_name",
    payload={"data": "task_data"},
    priority=TaskPriority.HIGH
)

# Добавление задачи в очередь
success = await task_queue.enqueue_task(task)
```

### Получение статуса задачи

```python
from distributed_task_queue import TaskStatus

# Получение статуса задачи
status = await task_queue.get_task_status(task.id)
if status == TaskStatus.COMPLETED:
    # Задача завершена
    result = await task_queue.get_task_result(task.id)
```

### Регистрация обработчика задач

```python
from distributed_task_queue import TaskResult, TaskStatus

async def custom_task_processor(task):
    # Обработка задачи
    try:
        # Выполнение логики задачи
        result_data = process_task_logic(task.payload)
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            result=result_data
        )
    except Exception as e:
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            error=str(e)
        )

# Регистрация обработчика для специфичных типов задач
task_queue.register_task_processor("custom_task_name", custom_task_processor)
```

## Интеграция с API

### Асинхронная обработка запросов

В файле `api/main.py` добавлены следующие эндпоинты для работы с распределенной очередью задач:

#### `/agent/process-async`

Асинхронная обработка запроса агентом через распределенную очередь задач. Принимает объект `AgentRequest` и возвращает ID задачи для отслеживания.

#### `/agent/task-status/{task_id}`

Получение статуса задачи асинхронной обработки. Возвращает статус задачи и результат, если доступен.

### Пример использования API

```bash
# Отправка задачи на асинхронную обработку
curl -X POST http://localhost:8000/agent/process-async \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Привет, как дела?",
    "user_id": "user123",
    "session_id": "session456"
  }'

# Ответ: {"task_id": "async_task_12345", "status": "enqueued", "message": "Request has been queued for asynchronous processing"}

# Проверка статуса задачи
curl -X GET http://localhost:800/agent/task-status/async_task_12345
```

## Конфигурация

Конфигурация распределенной очереди задач осуществляется через файл `config.py`. Основные параметры:

- `api_workers` - количество воркеров для обработки задач (по умолчанию 4)
- `redis_url` - URL для подключения к Redis (по умолчанию "redis://localhost:6379/0")

## Безопасность

Компонент использует существующую систему аутентификации и авторизации из API, обеспечивая безопасный доступ к эндпоинтам управления задачами. Все операции требуют аутентификации пользователя через JWT токены.

## Мониторинг

Система включает в себя мониторинг очереди задач с отслеживанием:
- Размера очереди
- Статусов задач
- Времени выполнения задач
- Ошибок обработки

## Тестирование

Для тестирования компонента созданы unit-тесты в файле `tests/test_distributed_queue.py`, которые проверяют:
- Создание очереди задач
- Добавление и извлечение задач
- Управление статусами задач
- Обработку задач с разными приоритетами
- Повторные попытки выполнения задач
- Метрики размера очереди

## Масштабируемость

Компонент обеспечивает масштабируемость за счет:
- Использования Redis в качестве брокера сообщений
- Поддержки нескольких воркеров
- Возможности горизонтального масштабирования через добавление новых инстансов воркеров
- Асинхронной обработки задач без блокировки основного потока

## Отказоустойчивость

Система обеспечивает отказоустойчивость за счет:
- Хранения задач в Redis с TTL
- Поддержки повторных попыток выполнения задач
- Обработки ошибок с логированием
- Механизмов восстановления после сбоев

"""
Распределенная очередь задач для масштабируемой архитектуры AI агента

Этот модуль реализует систему распределенных задач с поддержкой:
- Асинхронной обработки задач
- Отказоустойчивости
- Масштабирования через внешние брокеры сообщений
- Интеграции с существующей системой CQRS
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
import pickle
import redis.asyncio as redis
from redis.asyncio import Redis

from config import config
from cqrs.command_bus import Command, CommandBus, CommandResult
from cqrs.event_sourcing import DomainEvent, EventPublisher

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Статусы задачи в очереди"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Приоритеты задач"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Представление задачи в распределенной очереди"""
    id: str
    name: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = None
    scheduled_at: datetime = None
    expires_at: datetime = None
    max_retries: int = 3
    current_retry: int = 0
    timeout: float = 30.0  # секунд
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=24)  # 24 часа по умолчанию

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование задачи в словарь для сериализации"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['scheduled_at'] = self.scheduled_at.isoformat() if self.scheduled_at else None
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        data['priority'] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Создание задачи из словаря"""
        data_copy = data.copy()
        data_copy['created_at'] = datetime.fromisoformat(data['created_at'])
        if data_copy['scheduled_at']:
            data_copy['scheduled_at'] = datetime.fromisoformat(data['scheduled_at'])
        if data_copy['expires_at']:
            data_copy['expires_at'] = datetime.fromisoformat(data['expires_at'])
        data_copy['priority'] = TaskPriority(data['priority'])
        return cls(**data_copy)

    def is_expired(self) -> bool:
        """Проверка, истекло ли время жизни задачи"""
        return datetime.now() > self.expires_at

    def is_scheduled(self) -> bool:
        """Проверка, запланирована ли задача к выполнению"""
        return datetime.now() >= self.scheduled_at


@dataclass
class TaskResult:
    """Результат выполнения задачи"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    completed_at: datetime = None

    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()


class TaskQueueBackend(ABC):
    """Абстрактный интерфейс бэкенда очереди задач"""

    @abstractmethod
    async def enqueue(self, task: Task) -> bool:
        """Добавление задачи в очередь"""
        pass

    @abstractmethod
    async def dequeue(self, queue_name: str = "default") -> Optional[Task]:
        """Извлечение задачи из очереди"""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Получение задачи по ID"""
        pass

    @abstractmethod
    async def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Обновление статуса задачи"""
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Получение статуса задачи"""
        pass

    @abstractmethod
    async def delete_task(self, task_id: str) -> bool:
        """Удаление задачи из очереди"""
        pass

    @abstractmethod
    async def get_queue_size(self, queue_name: str = "default") -> int:
        """Получение размера очереди"""
        pass


class RedisTaskQueueBackend(TaskQueueBackend):
    """Реализация бэкенда очереди задач на основе Redis"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.task_key_prefix = "task_queue:task:"
        self.queue_key = "task_queue:default"
        self.status_key_prefix = "task_queue:status:"
        self.results_key_prefix = "task_queue:results:"

    async def enqueue(self, task: Task) -> bool:
        """Добавление задачи в очередь Redis"""
        try:
            # Сохраняем задачу
            task_key = f"{self.task_key_prefix}{task.id}"
            task_data = json.dumps(task.to_dict(), ensure_ascii=False)
            await self.redis.setex(task_key, 86400, task_data) # TTL 24 часа

            # Добавляем в очередь с приоритетом
            priority_score = -task.priority.value  # Обратный порядок для сортировки
            await self.redis.zadd(self.queue_key, {task.id: priority_score})

            logger.info(f"Task {task.id} enqueued with priority {task.priority.name}")
            return True
        except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Redis connection error during enqueue: {e}")
            # Возвращаем False при ошибках подключения к Redis
            return False
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.id}: {e}")
            return False

    async def dequeue(self, queue_name: str = "default") -> Optional[Task]:
        """Извлечение задачи из очереди Redis"""
        try:
            # Используем блоĸирующую операцию для получения задачи
            queue_key = f"task_queue:{queue_name}" if queue_name != "default" else self.queue_key
            
            # Получаем задачи с учетом приоритета (сортировка по score)
            task_ids = await self.redis.zrange(queue_key, 0, 0, withscores=True)
            if not task_ids:
                return None

            task_id, priority_score = task_ids[0]
            task_id = task_id.decode('utf-8') if isinstance(task_id, bytes) else task_id

            # Получаем данные задачи
            task_key = f"{self.task_key_prefix}{task_id}"
            task_data = await self.redis.get(task_key)
            if not task_data:
                # Задача не найдена, удаляем из очереди
                await self.redis.zrem(queue_key, task_id)
                return None

            # Десериализуем задачу
            task_dict = json.loads(task_data)
            task = Task.from_dict(task_dict)

            # Проверяем, не истекло ли время жизни задачи
            if task.is_expired():
                await self.delete_task(task.id)
                return None

            # Проверяем, запланирована ли задача к выполнению
            if not task.is_scheduled():
                return None

            # Удаляем задачу из очереди (временно)
            await self.redis.zrem(queue_key, task_id)

            # Устанавливаем статус PROCESSING
            await self.update_task_status(task.id, TaskStatus.PROCESSING)

            logger.info(f"Task {task.id} dequeued for processing")
            return task
        except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Redis connection error during dequeue: {e}")
            # Возвращаем None при ошибках подключения к Redis
            return None
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Получение задачи по ID из Redis"""
        try:
            task_key = f"{self.task_key_prefix}{task_id}"
            task_data = await self.redis.get(task_key)
            if not task_data:
                return None

            task_dict = json.loads(task_data)
            return Task.from_dict(task_dict)
        except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Redis connection error during get_task: {e}")
            # Возвращаем None при ошибках подключения к Redis
            return None
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None

    async def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Обновление статуса задачи в Redis"""
        try:
            status_key = f"{self.status_key_prefix}{task_id}"
            await self.redis.setex(status_key, 86400, status.value)  # TTL 24 часа
            return True
        except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Redis connection error during update_task_status: {e}")
            # Возвращаем False при ошибках подключения к Redis
            return False
        except Exception as e:
            logger.error(f"Failed to update status for task {task_id}: {e}")
            return False

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Получение статуса задачи из Redis"""
        try:
            status_key = f"{self.status_key_prefix}{task_id}"
            status_value = await self.redis.get(status_key)
            if not status_value:
                return None

            status_value = status_value.decode('utf-8') if isinstance(status_value, bytes) else status_value
            return TaskStatus(status_value)
        except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Redis connection error during get_task_status: {e}")
            # Возвращаем None при ошибках подключения к Redis
            return None
        except Exception as e:
            logger.error(f"Failed to get status for task {task_id}: {e}")
            return None

    async def delete_task(self, task_id: str) -> bool:
        """Удаление задачи из Redis"""
        try:
            # Удаляем задачу, статус и результат
            task_key = f"{self.task_key_prefix}{task_id}"
            status_key = f"{self.status_key_prefix}{task_id}"
            result_key = f"{self.results_key_prefix}{task_id}"

            await self.redis.delete(task_key, status_key, result_key)
            await self.redis.zrem(self.queue_key, task_id)

            return True
        except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Redis connection error during delete_task: {e}")
            # Возвращаем False при ошибках подключения к Redis
            return False
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False

    async def get_queue_size(self, queue_name: str = "default") -> int:
        """Получение размера очереди в Redis"""
        try:
            queue_key = f"task_queue:{queue_name}" if queue_name != "default" else self.queue_key
            return await self.redis.zcard(queue_key)
        except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Redis connection error during get_queue_size: {e}")
            # Возвращаем 0 при ошибках подключения к Redis
            return 0
        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0

    async def store_result(self, result: TaskResult) -> bool:
        """Сохранение результата задачи в Redis"""
        try:
            result_key = f"{self.results_key_prefix}{result.task_id}"
            result_data = json.dumps({
                'status': result.status.value,
                'result': result.result,
                'error': result.error,
                'execution_time': result.execution_time,
                'completed_at': result.completed_at.isoformat()
            }, ensure_ascii=False)
            await self.redis.setex(result_key, 86400, result_data) # TTL 24 часа
            return True
        except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Redis connection error during store_result: {e}")
            # Возвращаем False при ошибках подключения к Redis
            return False
        except Exception as e:
            logger.error(f"Failed to store result for task {result.task_id}: {e}")
            return False

    async def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Получение результата задачи из Redis"""
        try:
            result_key = f"{self.results_key_prefix}{task_id}"
            result_data = await self.redis.get(result_key)
            if not result_data:
                return None

            result_dict = json.loads(result_data)
            result_dict['status'] = TaskStatus(result_dict['status'])
            result_dict['completed_at'] = datetime.fromisoformat(result_dict['completed_at'])
            return TaskResult(task_id=task_id, **result_dict)
        except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Redis connection error during get_result: {e}")
            # Возвращаем None при ошибках подключения к Redis
            return None
        except Exception as e:
            logger.error(f"Failed to get result for task {task_id}: {e}")
            return None


class DistributedTaskQueue:
    """
    Распределенная очередь задач с поддержкой обработки команд CQRS
    
    Этот класс интегрируется с существующей системой CQRS и обеспечивает:
    - Асинхронную обработку задач
    - Масштабируемость через внешние брокеры
    - Отказоустойчивость
    - Мониторинг и логирование
    """

    def __init__(
        self,
        backend: TaskQueueBackend,
        command_bus: CommandBus = None,
        event_publisher: EventPublisher = None
    ):
        self.backend = backend
        self.command_bus = command_bus
        self.event_publisher = event_publisher
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.worker_count = config.api_workers # Используем конфигурацию API
        self.task_processors: Dict[str, Callable[[Task], Awaitable[TaskResult]]] = {}

    async def start(self):
        """Запуск обработчиков очереди"""
        if self.is_running:
            return

        self.is_running = True
        logger.info(f"Starting {self.worker_count} task queue workers")

        # Запускаем воркеры
        for i in range(self.worker_count):
            worker_task = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker_task)

        logger.info("Task queue workers started")

    async def stop(self):
        """Остановка обработчиков очереди"""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping task queue workers...")

        # Отменяем все задачи-воркеры
        for worker in self.workers:
            worker.cancel()

        # Ждем завершения всех воркеров
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()
        logger.info("Task queue workers stopped")

    async def _worker_loop(self, worker_id: int):
        """Цикл обработки задач для отдельного воркера"""
        logger.info(f"Worker {worker_id} started")
        # Счетчик последовательных ошибок подключения к Redis
        consecutive_redis_errors = 0
        max_redis_errors_before_slowdown = 5 # После 5 ошибок переходим в щадящий режим
        
        while self.is_running:
            try:
                # Получаем задачу из очереди
                task = await self.backend.dequeue()
                if task:
                    # Сброс счетчика ошибок при успешной операции
                    consecutive_redis_errors = 0
                    # Обрабатываем задачу
                    await self._process_task(task, worker_id)
                else:
                    # Если задач нет, определяем время ожидания в зависимости от ситуации
                    if hasattr(self.backend, 'redis'):
                        # Если используем Redis-бэкенд
                        if consecutive_redis_errors >= max_redis_errors_before_slowdown:
                            # В щадящем режиме ждем дольше
                            await asyncio.sleep(2.0)
                        else:
                            # В нормальном режиме короткая пауза
                            await asyncio.sleep(0.1)
                    else:
                        # Если используем fallback-бэкенд, короткая пауза
                        await asyncio.sleep(0.1)
            except (redis.ConnectionError, ConnectionRefusedError, OSError) as e:
                # Увеличиваем счетчик ошибок Redis
                consecutive_redis_errors += 1
                logger.warning(f"Worker {worker_id} Redis connection error (attempt {consecutive_redis_errors}): {e}")
                
                # Увеличиваем паузу при последовательных ошибках
                if consecutive_redis_errors >= max_redis_errors_before_slowdown:
                    # В щадящем режиме ждем дольше
                    await asyncio.sleep(5.0)
                else:
                    # При первых ошибках пауза поменьше
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                # Для других ошибок сбрасываем счетчик и делаем паузу
                consecutive_redis_errors = 0
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Пауза перед следующей итерацией

        logger.info(f"Worker {worker_id} stopped")

    async def _process_task(self, task: Task, worker_id: int):
        """Обработка отдельной задачи"""
        logger.info(f"Worker {worker_id} processing task {task.id}: {task.name}")

        try:
            # Обновляем статус задачи
            await self.backend.update_task_status(task.id, TaskStatus.PROCESSING)

            # Начинаем измерение времени выполнения
            start_time = asyncio.get_event_loop().time()

            # Проверяем, есть ли специфичный обработчик для этой задачи
            if task.name in self.task_processors:
                processor = self.task_processors[task.name]
                result = await processor(task)
            else:
                # Обработка как команды CQRS по умолчанию
                result = await self._process_as_command(task)

            # Вычисляем время выполнения
            execution_time = asyncio.get_event_loop().time() - start_time

            # Обновляем результат
            result.execution_time = execution_time
            await self.backend.store_result(result)

            # Обновляем статус
            await self.backend.update_task_status(task.id, result.status)

            logger.info(f"Task {task.id} completed with status {result.status.name} in {execution_time:.3f}s")

            # Публикуем событие, если доступен event_publisher
            if self.event_publisher:
                event_data = {
                    'task_id': task.id,
                    'task_name': task.name,
                    'status': result.status.name,
                    'execution_time': execution_time,
                    'result': result.result
                }
                event = DomainEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="TaskCompleted",
                    aggregate_id="task_queue",
                    aggregate_type="TaskQueue",
                    event_data=event_data,
                    metadata=task.metadata,
                    timestamp=datetime.now(),
                    version=1
                )
                await self.event_publisher.publish(event)

        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")

            # Вычисляем время выполнения до ошибки
            execution_time = asyncio.get_event_loop().time() - start_time if 'start_time' in locals() else 0

            # Проверяем количество попыток
            if task.current_retry < task.max_retries:
                # Обновляем количество попыток
                new_task = task
                new_task.current_retry += 1
                new_task.priority = TaskPriority.HIGH  # Повышаем приоритет для повторной попытки
                new_task.scheduled_at = datetime.now() + timedelta(seconds=2 ** task.current_retry)  # Экспоненциальная задержка

                # Обновляем задачу в бэкенде
                # Для fallback бэкенда обновляем в памяти, для Redis используем redis-операцию
                if hasattr(self.backend, 'redis'):
                    task_key = f"task_queue:task:{task.id}"
                    task_data = json.dumps(new_task.to_dict(), ensure_ascii=False)
                    await self.backend.redis.setex(task_key, 86400, task_data)
                else:
                    # Для fallback бэкенда просто обновляем задачу в словаре
                    await self.backend.enqueue(new_task)

                # Возвращаем задачу в очередь
                await self.backend.enqueue(new_task)

                # Обновляем статус
                await self.backend.update_task_status(task.id, TaskStatus.RETRYING)

                logger.info(f"Task {task.id} rescheduled for retry {new_task.current_retry}/{new_task.max_retries}")
            else:
                # Создаем результат с ошибкой
                error_result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    execution_time=execution_time
                )
                await self.backend.store_result(error_result)
                await self.backend.update_task_status(task.id, TaskStatus.FAILED)

                logger.error(f"Task {task.id} failed after {task.max_retries} retries: {e}")

    async def _process_as_command(self, task: Task) -> TaskResult:
        """Обработка задачи как команды CQRS"""
        if not self.command_bus:
            raise ValueError("CommandBus is required for processing commands")

        # Проверяем, является ли payload командой
        if 'command_type' in task.payload and 'command_data' in task.payload:
            # Создаем команду из данных задачи
            command_type_name = task.payload['command_type']
            command_data = task.payload['command_data']

            # Находим класс команды (упрощенная реализация)
            # В реальном приложении нужно использовать рефлексию или реестр команд
            command_class = self._get_command_class(command_type_name)
            if not command_class:
                raise ValueError(f"Unknown command type: {command_type_name}")

            # Создаем экземпляр команды
            command = command_class(**command_data)

            # Выполняем команду через шину
            result = await self.command_bus.execute(command)

            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                result=result.result if result.success else None,
                error=result.error_message if not result.success else None
            )
        else:
            # Если это не команда, просто возвращаем payload как результат
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=task.payload
            )

    def _get_command_class(self, command_type_name: str) -> Optional[type]:
        """Получение класса команды по имени (упрощенная реализация)"""
        # В реальном приложении здесь должна быть реализация
        # динамического поиска класса команды
        return None

    async def enqueue_task(self, task: Task) -> bool:
        """Добавление задачи в очередь"""
        return await self.backend.enqueue(task)

    def register_task_processor(self, task_name: str, processor: Callable[[Task], Awaitable[TaskResult]]):
        """Регистрация обработчика для специфичных типов задач"""
        self.task_processors[task_name] = processor

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Получение статуса задачи"""
        return await self.backend.get_task_status(task_id)

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Получение результата задачи"""
        return await self.backend.get_result(task_id)

    async def get_queue_size(self) -> int:
        """Получение размера очереди"""
        return await self.backend.get_queue_size()


# Глобальная очередь задач
async def create_distributed_task_queue() -> DistributedTaskQueue:
    """Создание экземпляра распределенной очереди задач"""
    try:
        # Создаем Redis клиент
        redis_client = redis.from_url(config.redis_url)
        
        # Создаем бэкенд
        backend = RedisTaskQueueBackend(redis_client)
        
        logger.info("Redis-based task queue backend initialized")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis, using fallback mechanism: {e}")
        # В случае ошибки подключения к Redis можно использовать альтернативный бэкенд
        # или настроить систему так, чтобы она могла работать без очереди
        from .distributed_task_queue_fallback import FallbackTaskQueueBackend
        backend = FallbackTaskQueueBackend()
        logger.info("Fallback task queue backend initialized")
    
    # Создаем очередь задач
    task_queue = DistributedTaskQueue(backend)
    
    return task_queue


# Примеры задач для AI агента
@dataclass
class ProcessUserQueryTask(Task):
    """Задача обработки пользовательского запроса"""
    user_id: str = ""
    query: str = ""
    context: Dict[str, Any] = None

    def __post_init__(self):
        super().__post_init__()
        if self.context is None:
            self.context = {}


@dataclass
class AnalyzeDataTask(Task):
    """Задача анализа данных"""
    data_source: str = ""
    analysis_type: str = ""
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        super().__post_init__()
        if self.parameters is None:
            self.parameters = {}


@dataclass
class GenerateReportTask(Task):
    """Задача генерации отчета"""
    report_type: str = ""
    data: Dict[str, Any] = None
    format: str = "json"

    def __post_init__(self):
        super().__post_init__()
        if self.data is None:
            self.data = {}


# Пример использования
async def example_usage():
    """Пример использования распределенной очереди задач"""
    # Создаем очередь задач
    task_queue = await create_distributed_task_queue()
    
    # Запускаем очередь
    await task_queue.start()
    
    # Создаем пример задачи
    task = Task(
        id=str(uuid.uuid4()),
        name="example_task",
        payload={"message": "Hello, distributed world!"},
        priority=TaskPriority.NORMAL
    )
    
    # Добавляем задачу в очередь
    success = await task_queue.enqueue_task(task)
    print(f"Task enqueued: {success}")
    
    # Получаем статус задачи
    status = await task_queue.get_task_status(task.id)
    print(f"Task status: {status}")
    
    # Ждем выполнения задачи (в реальном приложении это асинхронно)
    await asyncio.sleep(2)
    
    # Получаем результат
    result = await task_queue.get_task_result(task.id)
    print(f"Task result: {result}")
    
    # Останавливаем очередь
    await task_queue.stop()


if __name__ == "__main__":
    # Запуск примера
    asyncio.run(example_usage())

"""
Запасная реализация распределенной очереди задач для случаев, когда Redis недоступен
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Awaitable
import threading

from config import config

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


class FallbackTaskQueueBackend(TaskQueueBackend):
    """Резервная реализация бэкенда очереди задач в памяти"""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._queue: List[str] = []  # IDs задач в порядке приоритета
        self._statuses: Dict[str, TaskStatus] = {}
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()  # Защита доступа к общим данным
        logger.info("Fallback task queue backend initialized (in-memory)")

    async def enqueue(self, task: Task) -> bool:
        """Добавление задачи в очередь"""
        with self._lock:
            try:
                self._tasks[task.id] = task
                self._statuses[task.id] = TaskStatus.PENDING

                # Добавляем задачу в очередь с учетом приоритета
                # Простая реализация: вставляем в начало для высокого приоритета
                if task.priority == TaskPriority.CRITICAL or task.priority == TaskPriority.HIGH:
                    self._queue.insert(0, task.id)
                else:
                    self._queue.append(task.id)

                logger.info(f"Task {task.id} enqueued with priority {task.priority.name} (fallback)")
                return True
            except Exception as e:
                logger.error(f"Failed to enqueue task {task.id} in fallback: {e}")
                return False

    async def dequeue(self, queue_name: str = "default") -> Optional[Task]:
        """Извлечение задачи из очереди"""
        with self._lock:
            try:
                if not self._queue:
                    return None

                # Получаем задачу с наивысшим приоритетом
                task_id = self._queue.pop(0)
                if task_id in self._tasks:
                    task = self._tasks[task_id]

                    # Проверяем, не истекло ли время жизни задачи
                    if task.is_expired():
                        del self._tasks[task_id]
                        del self._statuses[task_id]
                        return None

                    # Проверяем, запланирована ли задача к выполнению
                    if not task.is_scheduled():
                        # Если задача еще не запланирована, возвращаем в очередь
                        self._queue.append(task_id)
                        return None

                    # Устанавливаем статус PROCESSING
                    self._statuses[task_id] = TaskStatus.PROCESSING

                    logger.info(f"Task {task.id} dequeued for processing (fallback)")
                    return task
                else:
                    # Задача была удалена, но осталась в очереди
                    return None
            except Exception as e:
                logger.error(f"Failed to dequeue task in fallback: {e}")
                return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Получение задачи по ID"""
        with self._lock:
            return self._tasks.get(task_id)

    async def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Обновление статуса задачи"""
        with self._lock:
            try:
                self._statuses[task_id] = status
                return True
            except Exception as e:
                logger.error(f"Failed to update status for task {task_id} in fallback: {e}")
                return False

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Получение статуса задачи"""
        with self._lock:
            return self._statuses.get(task_id)

    async def delete_task(self, task_id: str) -> bool:
        """Удаление задачи из очереди"""
        with self._lock:
            try:
                # Удаляем задачу из всех хранилищ
                self._tasks.pop(task_id, None)
                self._statuses.pop(task_id, None)
                self._results.pop(task_id, None)

                # Удаляем задачу из очереди
                if task_id in self._queue:
                    self._queue.remove(task_id)

                return True
            except Exception as e:
                logger.error(f"Failed to delete task {task_id} in fallback: {e}")
                return False

    async def get_queue_size(self, queue_name: str = "default") -> int:
        """Получение размера очереди"""
        with self._lock:
            return len(self._queue)

    async def store_result(self, result: TaskResult) -> bool:
        """Сохранение результата задачи"""
        with self._lock:
            try:
                self._results[result.task_id] = result
                return True
            except Exception as e:
                logger.error(f"Failed to store result for task {result.task_id} in fallback: {e}")
                return False

    async def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Получение результата задачи"""
        with self._lock:
            return self._results.get(task_id)


class DistributedTaskQueue:
    """
    Распределенная очередь задач с поддержкой обработки команд CQRS
    Использует fallback-реализацию при недоступности Redis
    """

    def __init__(
        self,
        backend: TaskQueueBackend,
        command_bus=None,
        event_publisher=None
    ):
        self.backend = backend
        self.command_bus = command_bus
        self.event_publisher = event_publisher
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.worker_count = min(2, config.api_workers)  # Уменьшаем количество воркеров для fallback
        self.task_processors: Dict[str, Callable[[Task], Awaitable[TaskResult]]] = {}

    async def start(self):
        """Запуск обработчиков очереди"""
        if self.is_running:
            return

        self.is_running = True
        logger.info(f"Starting {self.worker_count} task queue workers (fallback mode)")

        # Запускаем воркеры
        for i in range(self.worker_count):
            worker_task = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker_task)

        logger.info("Task queue workers started (fallback mode)")

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
        logger.info(f"Worker {worker_id} started (fallback mode)")
        consecutive_errors = 0
        max_errors_before_slowdown = 5

        while self.is_running:
            try:
                # Получаем задачу из очереди
                task = await self.backend.dequeue()
                if task:
                    # Сброс счетчика ошибок при успешной операции
                    consecutive_errors = 0
                    # Обрабатываем задачу
                    await self._process_task(task, worker_id)
                else:
                    # Если задач нет, ждем перед следующей проверкой
                    if consecutive_errors >= max_errors_before_slowdown:
                        await asyncio.sleep(2.0)  # Дольше спим при ошибках
                    else:
                        await asyncio.sleep(0.1)  # Короткая пауза в нормальном режиме
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                # Для других ошибок увеличиваем счетчик и делаем паузу
                consecutive_errors += 1
                logger.error(f"Worker {worker_id} error (attempt {consecutive_errors}): {e}")
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
            # Если нет command_bus, просто возвращаем payload как результат
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=task.payload
            )

        # Проверяем, является ли payload командой
        if 'command_type' in task.payload and 'command_data' in task.payload:
            # В fallback режиме просто возвращаем payload как результат
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=task.payload
            )
        else:
            # Если это не команда, просто возвращаем payload как результат
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=task.payload
            )

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
    """Создание экземпляра распределенной очереди задач с fallback"""
    try:
        # Пробуем создать Redis клиент
        import redis.asyncio as redis
        redis_client = redis.from_url(config.redis_url)

        # Если подключение успешно, создаем основной бэкенд
        from distributed_task_queue import RedisTaskQueueBackend
        backend = RedisTaskQueueBackend(redis_client)
        logger.info("Redis-based task queue backend initialized")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis, using fallback mechanism: {e}")
        # Используем fallback-бэкенд
        backend = FallbackTaskQueueBackend()
        logger.info("Fallback task queue backend initialized")

    # Создаем очередь задач
    task_queue = DistributedTaskQueue(backend)

    return task_queue

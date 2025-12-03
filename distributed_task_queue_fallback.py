"""
Резервная реализация очереди задач для работы без Redis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass

from distributed_task_queue import Task, TaskResult, TaskStatus, TaskPriority, TaskQueueBackend

logger = logging.getLogger(__name__)


class FallbackTaskQueueBackend(TaskQueueBackend):
    """
    Резервная реализация бэкенда очереди задач для работы без Redis
    Использует встроенные структуры данных Python для хранения задач
    """
    
    def __init__(self):
        # Хранение задач в памяти
        self._tasks: Dict[str, Task] = {}
        self._statuses: Dict[str, TaskStatus] = {}
        self._results: Dict[str, TaskResult] = {}
        self._queues: Dict[str, List[str]] = {"default": []}  # очередь задач по идентификаторам
        self._lock = asyncio.Lock()  # для потокобезопасности
        logger.info("Fallback task queue backend initialized (in-memory)")

    async def enqueue(self, task: Task) -> bool:
        """Добавление задачи в очередь"""
        async with self._lock:
            try:
                # Сохраняем задачу
                self._tasks[task.id] = task
                self._statuses[task.id] = TaskStatus.PENDING
                
                # Добавляем в очередь
                queue_name = "default" # в фоллбэк-реализации используем только основную очередь
                self._queues[queue_name].append(task.id)
                
                # Сортируем очередь по приоритету (сначала задачи с высоким приоритетом)
                self._queues[queue_name].sort(key=lambda task_id: self._tasks[task_id].priority.value, reverse=True)
                
                logger.info(f"Task {task.id} enqueued with priority {task.priority.name} (fallback)")
                return True
            except Exception as e:
                logger.error(f"Failed to enqueue task in fallback backend: {e}")
                return False

    async def dequeue(self, queue_name: str = "default") -> Optional[Task]:
        """Извлечение задачи из очереди"""
        async with self._lock:
            try:
                queue = self._queues.get(queue_name, [])
                if not queue:
                    return None
                
                # Получаем задачу с наивысшим приоритетом
                task_id = queue[0]
                task = self._tasks.get(task_id)
                
                if not task:
                    # Если задача не найдена, удаляем из очереди
                    queue.pop(0)
                    return None
                
                # Проверяем, не истекло ли время жизни задачи
                if task.is_expired():
                    await self.delete_task(task.id)
                    return None
                
                # Проверяем, запланирована ли задача к выполнению
                if not task.is_scheduled():
                    return None
                
                # Удаляем задачу из очереди
                queue.pop(0)
                
                # Устанавливаем статус PROCESSING
                await self.update_task_status(task.id, TaskStatus.PROCESSING)
                
                logger.info(f"Task {task.id} dequeued for processing (fallback)")
                return task
            except Exception as e:
                logger.error(f"Failed to dequeue task in fallback backend: {e}")
                return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Получение задачи по ID"""
        async with self._lock:
            try:
                return self._tasks.get(task_id)
            except Exception as e:
                logger.error(f"Failed to get task {task_id} in fallback backend: {e}")
                return None

    async def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Обновление статуса задачи"""
        async with self._lock:
            try:
                self._statuses[task_id] = status
                return True
            except Exception as e:
                logger.error(f"Failed to update status for task {task_id} in fallback backend: {e}")
                return False

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Получение статуса задачи"""
        async with self._lock:
            try:
                return self._statuses.get(task_id)
            except Exception as e:
                logger.error(f"Failed to get status for task {task_id} in fallback backend: {e}")
                return None

    async def delete_task(self, task_id: str) -> bool:
        """Удаление задачи из очереди"""
        async with self._lock:
            try:
                # Удаляем задачу из всех хранилищ
                self._tasks.pop(task_id, None)
                self._statuses.pop(task_id, None)
                self._results.pop(task_id, None)
                
                # Удаляем из всех очередей
                for queue in self._queues.values():
                    if task_id in queue:
                        queue.remove(task_id)
                
                return True
            except Exception as e:
                logger.error(f"Failed to delete task {task_id} in fallback backend: {e}")
                return False

    async def get_queue_size(self, queue_name: str = "default") -> int:
        """Получение размера очереди"""
        async with self._lock:
            try:
                queue = self._queues.get(queue_name, [])
                return len(queue)
            except Exception as e:
                logger.error(f"Failed to get queue size in fallback backend: {e}")
                return 0

    async def store_result(self, result: TaskResult) -> bool:
        """Сохранение результата задачи"""
        async with self._lock:
            try:
                self._results[result.task_id] = result
                return True
            except Exception as e:
                logger.error(f"Failed to store result for task {result.task_id} in fallback backend: {e}")
                return False

    async def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Получение результата задачи"""
        async with self._lock:
            try:
                return self._results.get(task_id)
            except Exception as e:
                logger.error(f"Failed to get result for task {task_id} in fallback backend: {e}")
                return None
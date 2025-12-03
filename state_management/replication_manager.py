"""
Менеджер репликации для AI-агента
Обеспечивает синхронизацию и репликацию данных между узлами системы
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import uuid
from enum import Enum
import hashlib
import aiohttp
from dataclasses import dataclass

class ReplicationMode(Enum):
    """Режимы репликации"""
    SYNC = "sync"          # Синхронная репликация
    ASYNC = "async"        # Асинхронная репликация
    SEMI_SYNC = "semi_sync" # Полусинхронная репликация

class ReplicationStatus(Enum):
    """Статусы репликации"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ReplicationNode:
    """Информация о репликационном узле"""
    node_id: str
    url: str
    priority: int
    last_sync: Optional[datetime] = None
    status: str = "active"
    replication_lag: int = 0

@dataclass
class ReplicationTask:
    """Задача репликации"""
    task_id: str
    source_node: str
    target_nodes: List[str]
    data_key: str
    data: Any
    mode: ReplicationMode
    priority: int = 1
    timeout: int = 30
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ReplicationStatus = ReplicationStatus.PENDING
    error_message: Optional[str] = None

class ReplicationManager:
    """Менеджер репликации"""
    
    def __init__(self, 
                 node_id: str,
                 storage_backend: Optional[Any] = None,
                 cluster_nodes: Optional[List[ReplicationNode]] = None):
        self.node_id = node_id
        self.storage = storage_backend
        self.cluster_nodes = cluster_nodes or []
        self.replication_tasks: Dict[str, ReplicationTask] = {}
        self.active_replications: Dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        self.event_handlers: Dict[str, List[Callable]] = {
            "replication_started": [],
            "replication_completed": [],
            "replication_failed": [],
            "node_status_changed": []
        }
        
    def add_node(self, node: ReplicationNode):
        """Добавляет узел в кластер"""
        self.cluster_nodes.append(node)
        self._trigger_event("node_status_changed", {"node": node, "action": "added"})
    
    def remove_node(self, node_id: str):
        """Удаляет узел из кластера"""
        self.cluster_nodes = [node for node in self.cluster_nodes if node.node_id != node_id]
        self._trigger_event("node_status_changed", {"node_id": node_id, "action": "removed"})
    
    def update_node_status(self, node_id: str, status: str):
        """Обновляет статус узла"""
        for node in self.cluster_nodes:
            if node.node_id == node_id:
                node.status = status
                node.last_sync = datetime.now()
                self._trigger_event("node_status_changed", {"node": node, "action": "updated"})
                break
    
    async def replicate_data(self, 
                           data_key: str, 
                           data: Any, 
                           mode: ReplicationMode = ReplicationMode.ASYNC,
                           target_nodes: Optional[List[str]] = None,
                           priority: int = 1) -> str:
        """Инициирует репликацию данных"""
        async with self.lock:
            task_id = str(uuid.uuid4())
            
            # Если не указаны целевые узлы, используем все активные узлы
            if target_nodes is None:
                target_nodes = [node.node_id for node in self.cluster_nodes if node.status == "active" and node.node_id != self.node_id]
            
            task = ReplicationTask(
                task_id=task_id,
                source_node=self.node_id,
                target_nodes=target_nodes,
                data_key=data_key,
                data=data,
                mode=mode,
                priority=priority
            )
            
            self.replication_tasks[task_id] = task
            
            # Запускаем репликацию в зависимости от режима
            if mode == ReplicationMode.SYNC:
                await self._execute_replication_sync(task)
            elif mode == ReplicationMode.ASYNC:
                # Запускаем асинхронную репликацию в фоне
                asyncio_task = asyncio.create_task(self._execute_replication_async(task))
                self.active_replications[task_id] = asyncio_task
            elif mode == ReplicationMode.SEMI_SYNC:
                await self._execute_replication_semi_sync(task)
            
            return task_id
    
    async def _execute_replication_sync(self, task: ReplicationTask):
        """Выполняет синхронную репликацию"""
        task.status = ReplicationStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            success_count = 0
            total_targets = len(task.target_nodes)
            
            for target_node_id in task.target_nodes:
                if await self._replicate_to_node(task, target_node_id):
                    success_count += 1
                    self.update_node_status(target_node_id, "active")
                else:
                    self.update_node_status(target_node_id, "failed")
            
            # Для синхронной репликации требуем успеха на всех узлах
            if success_count == total_targets:
                task.status = ReplicationStatus.COMPLETED
                task.completed_at = datetime.now()
                self._trigger_event("replication_completed", {"task": task})
            else:
                task.status = ReplicationStatus.FAILED
                task.error_message = f"Failed to replicate to {total_targets - success_count} of {total_targets} nodes"
                self._trigger_event("replication_failed", {"task": task})
                
        except Exception as e:
            task.status = ReplicationStatus.FAILED
            task.error_message = str(e)
            self._trigger_event("replication_failed", {"task": task})
    
    async def _execute_replication_async(self, task: ReplicationTask):
        """Выполняет асинхронную репликацию"""
        task.status = ReplicationStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Запускаем репликацию на всех узлах параллельно
            replication_coroutines = [
                self._replicate_to_node(task, target_node_id) 
                for target_node_id in task.target_nodes
            ]
            
            results = await asyncio.gather(*replication_coroutines, return_exceptions=True)
            
            success_count = sum(1 for result in results if result is True)
            total_targets = len(task.target_nodes)
            
            if success_count == total_targets:
                task.status = ReplicationStatus.COMPLETED
            else:
                task.status = ReplicationStatus.FAILED
                task.error_message = f"Failed to replicate to {total_targets - success_count} of {total_targets} nodes"
            
            task.completed_at = datetime.now()
            
            if task.status == ReplicationStatus.COMPLETED:
                self._trigger_event("replication_completed", {"task": task})
            else:
                self._trigger_event("replication_failed", {"task": task})
                
        except Exception as e:
            task.status = ReplicationStatus.FAILED
            task.error_message = str(e)
            self._trigger_event("replication_failed", {"task": task})
    
    async def _execute_replication_semi_sync(self, task: ReplicationTask):
        """Выполняет полусинхронную репликацию"""
        task.status = ReplicationStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Для полусинхронной репликации требуем успеха хотя бы на одном узле
            success_count = 0
            total_targets = len(task.target_nodes)
            
            for target_node_id in task.target_nodes:
                if await self._replicate_to_node(task, target_node_id):
                    success_count += 1
                    self.update_node_status(target_node_id, "active")
                    
                    # Если достигли минимального количества успешных репликаций, можем завершить
                    if success_count >= max(1, total_targets // 2 + 1):
                        break
                else:
                    self.update_node_status(target_node_id, "failed")
            
            if success_count >= max(1, total_targets // 2 + 1):
                task.status = ReplicationStatus.COMPLETED
                task.completed_at = datetime.now()
                self._trigger_event("replication_completed", {"task": task})
            else:
                task.status = ReplicationStatus.FAILED
                task.error_message = f"Failed to replicate to minimum required nodes: {success_count}/{max(1, total_targets // 2 + 1)}"
                self._trigger_event("replication_failed", {"task": task})
                
        except Exception as e:
            task.status = ReplicationStatus.FAILED
            task.error_message = str(e)
            self._trigger_event("replication_failed", {"task": task})
    
    async def _replicate_to_node(self, task: ReplicationTask, target_node_id: str) -> bool:
        """Выполняет репликацию на конкретный узел"""
        try:
            target_node = next((node for node in self.cluster_nodes if node.node_id == target_node_id), None)
            if not target_node:
                self.logger.error(f"Target node {target_node_id} not found")
                return False
            
            # Подготовка данных для отправки
            payload = {
                "data_key": task.data_key,
                "data": task.data,
                "source_node": task.source_node,
                "replication_task_id": task.task_id
            }
            
            # Отправка данных на удаленный узел
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{target_node.url}/replicate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=task.timeout)
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Successfully replicated {task.data_key} to {target_node_id}")
                        return True
                    else:
                        self.logger.error(f"Failed to replicate to {target_node_id}: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Error replicating to {target_node_id}: {e}")
            return False
    
    async def get_replication_status(self, task_id: str) -> Optional[ReplicationTask]:
        """Получает статус задачи репликации"""
        return self.replication_tasks.get(task_id)
    
    async def get_node_replication_status(self, node_id: str) -> Dict[str, Any]:
        """Получает статус репликации для узла"""
        node = next((node for node in self.cluster_nodes if node.node_id == node_id), None)
        if not node:
            return {}
        
        # Подсчет статистики репликации для узла
        completed_tasks = 0
        failed_tasks = 0
        pending_tasks = 0
        
        for task in self.replication_tasks.values():
            if node_id in task.target_nodes:
                if task.status == ReplicationStatus.COMPLETED:
                    completed_tasks += 1
                elif task.status == ReplicationStatus.FAILED:
                    failed_tasks += 1
                else:
                    pending_tasks += 1
        
        return {
            "node_id": node_id,
            "status": node.status,
            "last_sync": node.last_sync,
            "replication_lag": node.replication_lag,
            "statistics": {
                "completed": completed_tasks,
                "failed": failed_tasks,
                "pending": pending_tasks
            }
        }
    
    async def cancel_replication(self, task_id: str) -> bool:
        """Отменяет задачу репликации"""
        if task_id in self.active_replications:
            task = self.active_replications[task_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.active_replications[task_id]
        
        if task_id in self.replication_tasks:
            self.replication_tasks[task_id].status = ReplicationStatus.CANCELLED
            return True
        return False
    
    async def get_active_replications(self) -> List[ReplicationTask]:
        """Получает список активных задач репликации"""
        return [
            task for task in self.replication_tasks.values() 
            if task.status in [ReplicationStatus.PENDING, ReplicationStatus.IN_PROGRESS]
        ]
    
    async def cleanup_completed_tasks(self, retention_hours: int = 24) -> int:
        """Очищает завершенные задачи репликации старше указанного времени"""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        completed_tasks = [
            task_id for task_id, task in self.replication_tasks.items()
            if task.status == ReplicationStatus.COMPLETED and task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in completed_tasks:
            del self.replication_tasks[task_id]
        
        return len(completed_tasks)
    
    async def get_replication_history(self, 
                                    limit: int = 100,
                                    status: Optional[ReplicationStatus] = None,
                                    node_id: Optional[str] = None) -> List[ReplicationTask]:
        """Получает историю репликаций"""
        history = []
        
        for task in self.replication_tasks.values():
            if status and task.status != status:
                continue
            if node_id and node_id not in task.target_nodes:
                continue
            history.append(task)
        
        # Сортируем по времени создания (новые первыми)
        history.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        return history[:limit]
    
    async def calculate_consistency_score(self) -> float:
        """Рассчитывает уровень консистентности данных в кластере"""
        if not self.cluster_nodes:
            return 1.0 # Если нет узлов кроме текущего, то консистентность 100%
        
        # Простой расчет: отношение узлов с актуальными данными к общему количеству
        active_nodes = [node for node in self.cluster_nodes if node.status == "active"]
        if not active_nodes:
            return 0.0
        
        # В реальной системе здесь будет более сложная логика проверки консистентности данных
        # на основе контрольных сумм, версий данных и т.д.
        return len(active_nodes) / len(self.cluster_nodes)
    
    def subscribe_to_events(self, event_type: str, handler: Callable):
        """Подписывается на события репликации"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def unsubscribe_from_events(self, event_type: str, handler: Callable):
        """Отписывается от событий репликации"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].remove(handler)
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Триггерит событие репликации"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def sync_node_data(self, node_id: str, data_key: str) -> bool:
        """Синхронизирует данные с конкретным узлом"""
        node = next((node for node in self.cluster_nodes if node.node_id == node_id), None)
        if not node:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{node.url}/data/{data_key}",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Обновляем локальное состояние
                        if self.storage:
                            await self._save_to_local_storage(data_key, data)
                        
                        # Обновляем статус узла
                        self.update_node_status(node_id, "active")
                        return True
                    else:
                        self.logger.error(f"Failed to sync data from {node_id}: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Error syncing data from {node_id}: {e}")
            self.update_node_status(node_id, "failed")
            return False
    
    async def _save_to_local_storage(self, data_key: str, data: Any):
        """Сохраняет данные в локальное хранилище"""
        # Это абстрактный метод - реализация зависит от конкретного бэкенда
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверяет здоровье системы репликации"""
        active_tasks = await self.get_active_replications()
        consistency_score = await self.calculate_consistency_score()
        
        return {
            "node_id": self.node_id,
            "cluster_size": len(self.cluster_nodes),
            "active_replications": len(active_tasks),
            "consistency_score": consistency_score,
            "status": "healthy" if consistency_score > 0.5 else "degraded"
        }
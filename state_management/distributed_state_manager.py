"""
Распределенный менеджер состояний для AI-агента
Обеспечивает синхронизацию состояний между различными компонентами системы
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from enum import Enum

class StateOperation(Enum):
    """Типы операций с состоянием"""
    SET = "set"
    GET = "get"
    DELETE = "delete"
    UPDATE = "update"
    MERGE = "merge"

class StateVersion:
    """Класс для управления версиями состояния"""
    def __init__(self, version_id: str, timestamp: datetime, data: Dict[str, Any]):
        self.version_id = version_id
        self.timestamp = timestamp
        self.data = data
        self.checksum = self._calculate_checksum(data)

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Рассчитывает контрольную сумму данных"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует версию в словарь"""
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "checksum": self.checksum
        }

class DistributedStateManager:
    """Основной класс распределенного менеджера состояний"""
    
    def __init__(self, node_id: str, storage_backend: Optional[Any] = None):
        self.node_id = node_id
        self.storage = storage_backend or {}
        self.state_history: Dict[str, List[StateVersion]] = {}
        self.current_state: Dict[str, Any] = {}
        self.subscribers: Dict[str, List[callable]] = {}
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def set_state(self, key: str, value: Any, propagate: bool = True) -> str:
        """Устанавливает значение состояния с версионированием"""
        async with self.lock:
            version_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Создаем новую версию
            new_version = StateVersion(version_id, timestamp, {key: value})
            
            # Сохраняем историю версий
            if key not in self.state_history:
                self.state_history[key] = []
            self.state_history[key].append(new_version)
            
            # Обновляем текущее состояние
            self.current_state[key] = value
            
            # Сохраняем в бэкенд
            if self.storage:
                await self._save_to_storage(key, value, new_version)
            
            # Уведомляем подписчиков
            if propagate:
                await self._notify_subscribers(key, value, StateOperation.SET)
            
            self.logger.info(f"State set: {key}={value}, version: {version_id}")
            return version_id
    
    async def get_state(self, key: str, version: Optional[str] = None) -> Optional[Any]:
        """Получает значение состояния по ключу и версии"""
        if version:
            # Возвращаем конкретную версию
            if key in self.state_history:
                for state_version in self.state_history[key]:
                    if state_version.version_id == version:
                        return state_version.data.get(key)
            return None
        
        # Возвращаем текущее состояние
        return self.current_state.get(key)
    
    async def update_state(self, key: str, updates: Dict[str, Any], propagate: bool = True) -> str:
        """Обновляет часть состояния"""
        async with self.lock:
            current_value = await self.get_state(key)
            if current_value is None:
                current_value = {}
            
            # Объединяем текущее значение с обновлениями
            if isinstance(current_value, dict) and isinstance(updates, dict):
                updated_value = {**current_value, **updates}
            else:
                updated_value = updates
            
            return await self.set_state(key, updated_value, propagate)
    
    async def merge_state(self, key: str, data: Dict[str, Any], propagate: bool = True) -> str:
        """Объединяет данные состояния с глубоким слиянием"""
        async with self.lock:
            current_value = await self.get_state(key)
            if current_value is None:
                current_value = {}
            
            merged_value = self._deep_merge(current_value, data)
            return await self.set_state(key, merged_value, propagate)
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Глубокое слияние двух словарей"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def delete_state(self, key: str, propagate: bool = True) -> bool:
        """Удаляет состояние по ключу"""
        async with self.lock:
            if key in self.current_state:
                del self.current_state[key]
                
                # Удаляем из бэкенда
                if self.storage:
                    await self._delete_from_storage(key)
                
                # Уведомляем подписчиков
                if propagate:
                    await self._notify_subscribers(key, None, StateOperation.DELETE)
                
                self.logger.info(f"State deleted: {key}")
                return True
            return False
    
    async def subscribe(self, key: str, callback: callable):
        """Подписывается на изменения состояния"""
        if key not in self.subscribers:
            self.subscribers[key] = []
        self.subscribers[key].append(callback)
    
    async def unsubscribe(self, key: str, callback: callable):
        """Отписывается от изменений состояния"""
        if key in self.subscribers:
            self.subscribers[key].remove(callback)
    
    async def _notify_subscribers(self, key: str, value: Any, operation: StateOperation):
        """Уведомляет подписчиков об изменении состояния"""
        if key in self.subscribers:
            for callback in self.subscribers[key]:
                try:
                    await callback(key, value, operation)
                except Exception as e:
                    self.logger.error(f"Error in subscriber callback: {e}")
    
    async def _save_to_storage(self, key: str, value: Any, version: StateVersion):
        """Сохраняет состояние в бэкенд-хранилище"""
        # Это абстрактный метод - реализация зависит от конкретного бэкенда
        pass
    
    async def _delete_from_storage(self, key: str):
        """Удаляет состояние из бэкенд-хранилища"""
        # Это абстрактный метод - реализация зависит от конкретного бэкенда
        pass
    
    async def get_state_history(self, key: str) -> List[Dict[str, Any]]:
        """Возвращает историю изменений состояния"""
        if key in self.state_history:
            return [version.to_dict() for version in self.state_history[key]]
        return []
    
    async def rollback_state(self, key: str, version_id: str) -> bool:
        """Откатывает состояние к предыдущей версии"""
        async with self.lock:
            if key in self.state_history:
                for state_version in self.state_history[key]:
                    if state_version.version_id == version_id:
                        # Восстанавливаем состояние
                        self.current_state[key] = state_version.data.get(key)
                        
                        # Уведомляем подписчиков
                        await self._notify_subscribers(key, self.current_state[key], StateOperation.UPDATE)
                        
                        self.logger.info(f"State rolled back: {key} to version {version_id}")
                        return True
            return False
    
    async def get_all_keys(self) -> List[str]:
        """Возвращает все ключи состояний"""
        return list(self.current_state.keys())
    
    async def get_state_with_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Возвращает состояние с метаданными"""
        value = await self.get_state(key)
        if value is not None:
            history = await self.get_state_history(key)
            return {
                "key": key,
                "value": value,
                "version_count": len(history),
                "latest_version": history[-1] if history else None,
                "current": True
            }
        return None
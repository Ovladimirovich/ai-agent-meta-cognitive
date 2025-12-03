"""
Менеджер версионирования для AI-агента
Обеспечивает управление версиями состояний, конфигураций и данных
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import uuid
from enum import Enum
import hashlib

class VersionType(Enum):
    """Типы версионируемых объектов"""
    STATE = "state"
    CONFIG = "config"
    DATA = "data"
    MODEL = "model"
    SCHEMA = "schema"

class VersionMetadata:
    """Метаданные версии"""
    def __init__(self, 
                 version_id: str,
                 timestamp: datetime,
                 author: str,
                 change_type: str,
                 description: str,
                 parent_version: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        self.version_id = version_id
        self.timestamp = timestamp
        self.author = author
        self.change_type = change_type
        self.description = description
        self.parent_version = parent_version
        self.tags = tags or []
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Рассчитывает контрольную сумму метаданных"""
        data_str = json.dumps({
            "version_id": self.version_id,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "change_type": self.change_type,
            "description": self.description,
            "parent_version": self.parent_version,
            "tags": sorted(self.tags)
        }, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метаданные в словарь"""
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "change_type": self.change_type,
            "description": self.description,
            "parent_version": self.parent_version,
            "tags": self.tags,
            "checksum": self.checksum
        }

class Version:
    """Класс версии объекта"""
    def __init__(self, 
                 obj_id: str, 
                 version_number: int, 
                 obj_type: VersionType,
                 data: Any,
                 metadata: VersionMetadata):
        self.obj_id = obj_id
        self.version_number = version_number
        self.obj_type = obj_type
        self.data = data
        self.metadata = metadata
        self.dependencies: List[Tuple[str, str]] = []  # (obj_id, version_id)
        self.references: List[Tuple[str, str]] = []    # (obj_id, version_id)

    def add_dependency(self, obj_id: str, version_id: str):
        """Добавляет зависимость"""
        self.dependencies.append((obj_id, version_id))

    def add_reference(self, obj_id: str, version_id: str):
        """Добавляет ссылку"""
        self.references.append((obj_id, version_id))

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует версию в словарь"""
        return {
            "obj_id": self.obj_id,
            "version_number": self.version_number,
            "obj_type": self.obj_type.value,
            "data": self.data,
            "metadata": self.metadata.to_dict(),
            "dependencies": self.dependencies,
            "references": self.references
        }

class VersionManager:
    """Менеджер версионирования"""
    
    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self.versions: Dict[str, List[Version]] = {}  # obj_id -> [versions]
        self.version_index: Dict[str, Version] = {} # version_id -> version
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def create_version(self, 
                           obj_id: str, 
                           obj_type: VersionType, 
                           data: Any,
                           author: str,
                           change_type: str,
                           description: str,
                           parent_version_id: Optional[str] = None,
                           tags: Optional[List[str]] = None) -> str:
        """Создает новую версию объекта"""
        async with self.lock:
            # Получаем последнюю версию для определения номера
            latest_version = await self.get_latest_version(obj_id)
            version_number = 1 if latest_version is None else latest_version.version_number + 1
            
            # Создаем метаданные
            metadata = VersionMetadata(
                version_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                author=author,
                change_type=change_type,
                description=description,
                parent_version=parent_version_id,
                tags=tags
            )
            
            # Создаем версию
            version = Version(obj_id, version_number, obj_type, data, metadata)
            
            # Сохраняем в индекс
            self.version_index[metadata.version_id] = version
            
            # Добавляем в список версий объекта
            if obj_id not in self.versions:
                self.versions[obj_id] = []
            self.versions[obj_id].append(version)
            
            # Сохраняем в бэкенд
            if self.storage:
                await self._save_version_to_storage(version)
            
            self.logger.info(f"Created version {metadata.version_id} for {obj_id} ({obj_type.value})")
            return metadata.version_id
    
    async def get_version(self, version_id: str) -> Optional[Version]:
        """Получает версию по ID"""
        return self.version_index.get(version_id)
    
    async def get_latest_version(self, obj_id: str) -> Optional[Version]:
        """Получает последнюю версию объекта"""
        if obj_id in self.versions and self.versions[obj_id]:
            return max(self.versions[obj_id], key=lambda v: v.version_number)
        return None
    
    async def get_version_by_number(self, obj_id: str, version_number: int) -> Optional[Version]:
        """Получает версию по номеру"""
        if obj_id in self.versions:
            for version in self.versions[obj_id]:
                if version.version_number == version_number:
                    return version
        return None
    
    async def get_all_versions(self, obj_id: str) -> List[Version]:
        """Получает все версии объекта"""
        return self.versions.get(obj_id, [])
    
    async def get_versions_by_type(self, obj_type: VersionType) -> List[Version]:
        """Получает все версии определенного типа"""
        result = []
        for versions_list in self.versions.values():
            for version in versions_list:
                if version.obj_type == obj_type:
                    result.append(version)
        return result
    
    async def get_versions_by_author(self, author: str) -> List[Version]:
        """Получает все версии, созданные автором"""
        result = []
        for version in self.version_index.values():
            if version.metadata.author == author:
                result.append(version)
        return result
    
    async def get_versions_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Version]:
        """Получает версии в диапазоне времени"""
        result = []
        for version in self.version_index.values():
            if start_time <= version.metadata.timestamp <= end_time:
                result.append(version)
        return result
    
    async def get_versions_with_tag(self, tag: str) -> List[Version]:
        """Получает версии с определенным тегом"""
        result = []
        for version in self.version_index.values():
            if tag in version.metadata.tags:
                result.append(version)
        return result
    
    async def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Сравнивает две версии"""
        version1 = await self.get_version(version_id1)
        version2 = await self.get_version(version_id2)
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        # Простое сравнение данных (для сложного сравнения можно использовать сторонние библиотеки)
        import deepdiff
        diff = deepdiff.DeepDiff(version1.data, version2.data, ignore_order=True)
        
        return {
            "version1_id": version_id1,
            "version2_id": version_id2,
            "differences": diff.to_dict() if diff else {},
            "version1_metadata": version1.metadata.to_dict(),
            "version2_metadata": version2.metadata.to_dict()
        }
    
    async def rollback_to_version(self, obj_id: str, version_id: str) -> bool:
        """Откатывает объект к определенной версии"""
        version = await self.get_version(version_id)
        if not version or version.obj_id != obj_id:
            return False
        
        # Создаем новую версию с данными из целевой версии
        latest_version = await self.get_latest_version(obj_id)
        new_version_id = await self.create_version(
            obj_id=obj_id,
            obj_type=version.obj_type,
            data=version.data,
            author="system",
            change_type="rollback",
            description=f"Rollback to version {version_id}",
            parent_version_id=latest_version.metadata.version_id if latest_version else None
        )
        
        return True
    
    async def add_dependency(self, version_id: str, dependency_obj_id: str, dependency_version_id: str) -> bool:
        """Добавляет зависимость между версиями"""
        version = await self.get_version(version_id)
        dependency_version = await self.get_version(dependency_version_id)
        
        if not version or not dependency_version:
            return False
        
        if dependency_version.obj_id != dependency_obj_id:
            return False
        
        version.add_dependency(dependency_obj_id, dependency_version_id)
        
        # Сохраняем изменения в бэкенд
        if self.storage:
            await self._save_version_to_storage(version)
        
        return True
    
    async def add_reference(self, version_id: str, referenced_obj_id: str, referenced_version_id: str) -> bool:
        """Добавляет ссылку между версиями"""
        version = await self.get_version(version_id)
        referenced_version = await self.get_version(referenced_version_id)
        
        if not version or not referenced_version:
            return False
        
        if referenced_version.obj_id != referenced_obj_id:
            return False
        
        version.add_reference(referenced_obj_id, referenced_version_id)
        
        # Сохраняем изменения в бэкенд
        if self.storage:
            await self._save_version_to_storage(version)
        
        return True
    
    async def get_dependencies(self, version_id: str) -> List[Tuple[str, str]]:
        """Получает зависимости версии"""
        version = await self.get_version(version_id)
        return version.dependencies if version else []
    
    async def get_references(self, version_id: str) -> List[Tuple[str, str]]:
        """Получает ссылки на версию"""
        version = await self.get_version(version_id)
        return version.references if version else []
    
    async def search_versions(self, 
                            query: str, 
                            obj_type: Optional[VersionType] = None,
                            author: Optional[str] = None,
                            tags: Optional[List[str]] = None) -> List[Version]:
        """Поиск версий по различным критериям"""
        results = []
        
        for version in self.version_index.values():
            # Проверяем тип
            if obj_type and version.obj_type != obj_type:
                continue
            
            # Проверяем автора
            if author and version.metadata.author != author:
                continue
            
            # Проверяем теги
            if tags:
                if not all(tag in version.metadata.tags for tag in tags):
                    continue
            
            # Проверяем совпадение в данных или описании
            data_str = json.dumps(version.data, default=str)
            if query.lower() in data_str.lower() or query.lower() in version.metadata.description.lower():
                results.append(version)
        
        return results
    
    async def get_version_lineage(self, version_id: str) -> List[Version]:
        """Получает полную линию наследования версии"""
        version = await self.get_version(version_id)
        if not version:
            return []
        
        lineage = []
        current_version = version
        
        # Идем вверх по дереву родительских версий
        while current_version:
            lineage.insert(0, current_version)  # Добавляем в начало списка
            if current_version.metadata.parent_version:
                current_version = await self.get_version(current_version.metadata.parent_version)
            else:
                current_version = None
        
        return lineage
    
    async def _save_version_to_storage(self, version: Version):
        """Сохраняет версию в бэкенд-хранилище"""
        # Это абстрактный метод - реализация зависит от конкретного бэкенда
        pass
    
    async def validate_version_consistency(self, version_id: str) -> Dict[str, Any]:
        """Проверяет целостность и консистентность версии"""
        version = await self.get_version(version_id)
        if not version:
            return {"valid": False, "errors": ["Version not found"]}
        
        errors = []
        
        # Проверяем контрольную сумму метаданных
        expected_checksum = version.metadata._calculate_checksum()
        if version.metadata.checksum != expected_checksum:
            errors.append("Metadata checksum mismatch")
        
        # Проверяем зависимости
        for dep_obj_id, dep_version_id in version.dependencies:
            dep_version = await self.get_version(dep_version_id)
            if not dep_version or dep_version.obj_id != dep_obj_id:
                errors.append(f"Dependency not found: {dep_obj_id}@{dep_version_id}")
        
        # Проверяем ссылки
        for ref_obj_id, ref_version_id in version.references:
            ref_version = await self.get_version(ref_version_id)
            if not ref_version or ref_version.obj_id != ref_obj_id:
                errors.append(f"Reference not found: {ref_obj_id}@{ref_version_id}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": []
        }
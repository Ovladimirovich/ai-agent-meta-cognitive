"""
Фабрика менеджеров состояний
Обеспечивает создание и конфигурацию компонентов управления состоянием
"""
import asyncio
import logging
from typing import Optional, Dict, Any
from .distributed_state_manager import DistributedStateManager
from .version_manager import VersionManager
from .replication_manager import ReplicationManager, ReplicationNode

class StateManagerFactory:
    """Фабрика для создания менеджеров состояний"""
    
    @staticmethod
    def create_distributed_state_manager(
        node_id: str,
        storage_backend: Optional[Any] = None
    ) -> DistributedStateManager:
        """Создает распределенный менеджер состояний"""
        return DistributedStateManager(
            node_id=node_id,
            storage_backend=storage_backend
        )
    
    @staticmethod
    def create_version_manager(
        storage_backend: Optional[Any] = None
    ) -> VersionManager:
        """Создает менеджер версионирования"""
        return VersionManager(
            storage_backend=storage_backend
        )
    
    @staticmethod
    def create_replication_manager(
        node_id: str,
        storage_backend: Optional[Any] = None,
        cluster_nodes: Optional[list] = None
    ) -> ReplicationManager:
        """Создает менеджер репликации"""
        return ReplicationManager(
            node_id=node_id,
            storage_backend=storage_backend,
            cluster_nodes=cluster_nodes or []
        )
    
    @classmethod
    def create_complete_state_system(
        cls,
        node_id: str,
        storage_backend: Optional[Any] = None,
        cluster_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Создает полную систему управления состоянием"""
        # Создаем менеджер состояний
        state_manager = cls.create_distributed_state_manager(
            node_id=node_id,
            storage_backend=storage_backend
        )
        
        # Создаем менеджер версионирования
        version_manager = cls.create_version_manager(
            storage_backend=storage_backend
        )
        
        # Создаем конфигурацию кластера для менеджера репликации
        cluster_nodes = []
        if cluster_config:
            for node_config in cluster_config.get("nodes", []):
                node = ReplicationNode(
                    node_id=node_config["node_id"],
                    url=node_config["url"],
                    priority=node_config.get("priority", 1)
                )
                cluster_nodes.append(node)
        
        # Создаем менеджер репликации
        replication_manager = cls.create_replication_manager(
            node_id=node_id,
            storage_backend=storage_backend,
            cluster_nodes=cluster_nodes
        )
        
        # Интегрируем компоненты между собой
        cls._integrate_components(
            state_manager=state_manager,
            version_manager=version_manager,
            replication_manager=replication_manager
        )
        
        return {
            "state_manager": state_manager,
            "version_manager": version_manager,
            "replication_manager": replication_manager
        }
    
    @classmethod
    def _integrate_components(
        cls,
        state_manager: DistributedStateManager,
        version_manager: VersionManager,
        replication_manager: ReplicationManager
    ):
        """Интегрирует компоненты системы управления состоянием"""
        
        # Подписываем менеджер версионирования на изменения состояния
        async def on_state_change(key: str, value: Any, operation: Any):
            if operation.name in ["SET", "UPDATE", "MERGE"]:
                # Создаем новую версию при изменении состояния
                await version_manager.create_version(
                    obj_id=key,
                    obj_type="state",
                    data=value,
                    author="system",
                    change_type=operation.name.lower(),
                    description=f"State change for {key}"
                )
                
                # Реплицируем изменение состояния
                await replication_manager.replicate_data(
                    data_key=key,
                    data=value,
                    mode="async"
                )
        
        # Подписываемся на изменения состояния
        for key in ["*"]:  # Подписываемся на все изменения
            asyncio.create_task(state_manager.subscribe(key, on_state_change))
        
        # Настройка обработчиков событий репликации
        def on_replication_completed(data: Dict[str, Any]):
            logging.info(f"Replication completed: {data}")
        
        def on_replication_failed(data: Dict[str, Any]):
            logging.error(f"Replication failed: {data}")
        
        replication_manager.subscribe_to_events("replication_completed", on_replication_completed)
        replication_manager.subscribe_to_events("replication_failed", on_replication_failed)

# Пример использования
async def example_usage():
    """Пример использования фабрики"""
    # Конфигурация кластера
    cluster_config = {
        "nodes": [
            {
                "node_id": "node-1",
                "url": "http://node1:8000",
                "priority": 1
            },
            {
                "node_id": "node-2",
                "url": "http://node2:8000",
                "priority": 1
            }
        ]
    }
    
    # Создание полной системы управления состоянием
    state_system = StateManagerFactory.create_complete_state_system(
        node_id="main-node",
        cluster_config=cluster_config
    )
    
    state_manager = state_system["state_manager"]
    version_manager = state_system["version_manager"]
    replication_manager = state_system["replication_manager"]
    
    # Пример использования
    await state_manager.set_state("user_preferences", {"theme": "dark", "language": "ru"})
    
    # Получение истории версий
    history = await version_manager.get_all_versions("user_preferences")
    print(f"Versions count: {len(history)}")
    
    # Получение статуса репликации
    replication_status = await replication_manager.health_check()
    print(f"Replication status: {replication_status}")

if __name__ == "__main__":
    asyncio.run(example_usage())
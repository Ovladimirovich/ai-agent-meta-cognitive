import time
import hashlib
import json
import logging
from typing import Any, Dict
from .base_tool import BaseTool, Task, ToolResult
from ..integrations.circuit_breaker import circuit_breaker_decorator, CircuitBreakerConfig

logger = logging.getLogger(__name__)

class CacheTool(BaseTool):
    name = "cache"
    description = "Кэширование результатов для оптимизации производительности"
    version = "1.0.0"

    def __init__(self):
        self.cache_system = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Инициализация системы кэширования"""
        if self._initialized:
            return True

        try:
            from cache.cache_system_enhanced import EnhancedCache
            self.cache_system = EnhancedCache()
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize CacheTool: {e}")
            return False

    def can_handle(self, task: Task) -> bool:
        """Кэш можно использовать для любой задачи"""
        return True

    @circuit_breaker_decorator("cache_tool", CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=30.0,
        timeout=10.0,
        name="cache_tool"
    ))
    async def execute(self, task: Task) -> ToolResult:
        """Проверка и управление кэшем"""
        start_time = time.time()

        try:
            if not await self.initialize():
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    execution_time=time.time() - start_time,
                    error_message="Cache system not available"
                )

            # Генерация ключа кэша
            cache_key = self._generate_cache_key(task)

            # Проверка наличия в кэше
            cached_result = await self.cache_system.get(cache_key)

            execution_time = time.time() - start_time

            if cached_result:
                return ToolResult(
                    success=True,
                    data=cached_result,
                    metadata={
                        'tool': self.name,
                        'cache_hit': True,
                        'cache_key': cache_key,
                        'is_final': False
                    },
                    execution_time=execution_time
                )
            else:
                return ToolResult(
                    success=True,
                    data=None,
                    metadata={
                        'tool': self.name,
                        'cache_hit': False,
                        'cache_key': cache_key,
                        'is_final': False
                    },
                    execution_time=execution_time
                )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                data=None,
                metadata={'tool': self.name},
                execution_time=execution_time,
                error_message=str(e)
            )

    def _generate_cache_key(self, task: Task) -> str:
        """Генерация ключа кэша для задачи"""
        # Создание deterministic представления задачи
        cache_data = {
            'query': task.query,
            'context_keys': sorted(task.context.keys()) if task.context else [],
            'metadata_keys': sorted(task.metadata.keys()) if task.metadata else []
        }

        # Хэширование для создания ключа
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    async def store_result(self, cache_key: str, result: Any, ttl: int = 300):
        """Метод для сохранения результата в кэш"""
        if self.cache_system and result:
            try:
                await self.cache_system.set(cache_key, result, ttl=ttl)
                logger.debug(f"Cached result for key: {cache_key}")
            except Exception as e:
                logger.error(f"Failed to cache result: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса инструмента"""
        return {
            'name': self.name,
            'initialized': self._initialized,
            'version': self.version,
            'capabilities': ['caching', 'performance_optimization'],
            'cache_type': 'enhanced' if self._initialized else None
        }

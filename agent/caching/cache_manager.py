"""
Multi-level caching system for AI Agent performance optimization.
Implements L1 (in-memory), L2 (Redis), L3 (disk) caching strategy.
"""

import asyncio
import json
import hashlib
import time
import os
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import redis
import psutil
from prometheus_client import Counter, Histogram, Gauge


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    ttl: int
    created_at: float
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0


class CacheLayer(ABC):
    """Abstract base class for cache layers"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class L1Cache(CacheLayer):
    """L1 In-memory cache using LRU eviction"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self.access_order = []  # Для отслеживания порядка доступа к элементам

        # Metrics
        self.hits = Counter('l1_cache_hits_total', 'L1 cache hits')
        self.misses = Counter('l1_cache_misses_total', 'L1 cache misses')
        self.evictions = Counter('l1_cache_evictions_total', 'L1 cache evictions')
        self.memory_usage = Gauge('l1_cache_memory_bytes', 'L1 cache memory usage')
        
        # Добавляем счетчик для отслеживания количества элементов в кэше
        self.entry_count_gauge = Gauge('l1_cache_entries_count', 'L1 cache entries count')

    async def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache"""
        if key in self.cache:
            entry = self.cache[key]

            # Check TTL
            if time.time() - entry.created_at > entry.ttl:
                await self.delete(key)
                self.misses.inc()
                return None

            # Update access metadata
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Обновляем порядок доступа к элементам
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            self.hits.inc()
            return entry.value

        self.misses.inc()
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in L1 cache"""
        # Calculate entry size
        entry_size = self._calculate_size(key, value)

        # Check if we need to evict
        while self.current_memory_bytes + entry_size > self.max_memory_bytes:
            if not await self._evict_lru():
                return False

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            created_at=time.time(),
            size_bytes=entry_size
        )

        # Update memory usage
        if key in self.cache:
            self.current_memory_bytes -= self.cache[key].size_bytes
        self.current_memory_bytes += entry_size

        self.cache[key] = entry
        # Обновляем порядок доступа
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        self.memory_usage.set(self.current_memory_bytes)
        # Обновляем счетчик элементов
        self.entry_count_gauge.set(len(self.cache))
        return True

    async def delete(self, key: str) -> bool:
        """Delete value from L1 cache"""
        if key in self.cache:
            self.current_memory_bytes -= self.cache[key].size_bytes
            del self.cache[key]
            # Удаляем ключ также из списка порядка доступа
            if key in self.access_order:
                self.access_order.remove(key)
            self.memory_usage.set(self.current_memory_bytes)
            # Обновляем счетчик элементов
            self.entry_count_gauge.set(len(self.cache))
            return True
        return False

    async def clear(self) -> bool:
        """Clear all L1 cache entries"""
        self.cache.clear()
        self.current_memory_bytes = 0
        self.access_order.clear()  # Также очищаем порядок доступа
        self.memory_usage.set(0)
        # Обновляем счетчик элементов
        self.entry_count_gauge.set(0)
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get L1 cache statistics"""
        return {
            'entries': len(self.cache),
            'memory_usage_bytes': self.current_memory_bytes,
            'memory_usage_mb': self.current_memory_bytes / (1024 * 1024),
            'hit_rate': self.hits._value / max(self.hits._value + self.misses._value, 1)
        }

    async def _evict_lru(self) -> bool:
        """Evict least recently used entry"""
        if not self.cache:
            return False

        # Find LRU entry using access_order list for efficiency
        if self.access_order:
            lru_key = self.access_order[0] # The least recently accessed key
        else:
            # Fallback to original method if access_order is somehow empty
            lru_key = min(self.cache.keys(),
                         key=lambda k: self.cache[k].last_accessed)

        self.evictions.inc()
        await self.delete(lru_key)
        return True

    def _calculate_size(self, key: str, value: Any) -> int:
        """Calculate approximate size of cache entry"""
        key_size = len(key.encode('utf-8'))
        try:
            value_size = len(json.dumps(value).encode('utf-8'))
        except:
            value_size = len(str(value).encode('utf-8'))

        return key_size + value_size + 128  # Overhead


class L2Cache(CacheLayer):
    """L2 Redis cache"""

    def __init__(self, redis_client: redis.Redis, prefix: str = "ai_agent:"):
        self.redis = redis_client
        self.prefix = prefix

        # Metrics
        self.hits = Counter('l2_cache_hits_total', 'L2 cache hits')
        self.misses = Counter('l2_cache_misses_total', 'L2 cache misses')
        self.errors = Counter('l2_cache_errors_total', 'L2 cache errors')

    async def get(self, key: str) -> Optional[Any]:
        """Get value from L2 cache"""
        try:
            full_key = f"{self.prefix}{key}"
            value = self.redis.get(full_key)

            if value:
                self.hits.inc()
                return json.loads(value.decode('utf-8'))

            self.misses.inc()
            return None

        except Exception as e:
            self.errors.inc()
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in L2 cache"""
        try:
            full_key = f"{self.prefix}{key}"
            json_value = json.dumps(value)
            return bool(self.redis.setex(full_key, ttl, json_value))
        except Exception as e:
            self.errors.inc()
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from L2 cache"""
        try:
            full_key = f"{self.prefix}{key}"
            return bool(self.redis.delete(full_key))
        except Exception as e:
            self.errors.inc()
            return False

    async def clear(self) -> bool:
        """Clear all L2 cache entries"""
        try:
            keys = self.redis.keys(f"{self.prefix}*")
            if keys:
                return bool(self.redis.delete(*keys))
            return True
        except Exception as e:
            self.errors.inc()
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get L2 cache statistics"""
        try:
            info = self.redis.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'total_keys': len(self.redis.keys(f"{self.prefix}*")),
                'hit_rate': self.hits._value / max(self.hits._value + self.misses._value, 1)
            }
        except:
            return {'error': 'Unable to get Redis stats'}


class L3Cache(CacheLayer):
    """L3 Disk-based cache"""

    def __init__(self, cache_dir: str = "/tmp/ai_agent_cache", max_size_gb: float = 10.0):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Metrics
        self.hits = Counter('l3_cache_hits_total', 'L3 cache hits')
        self.misses = Counter('l3_cache_misses_total', 'L3 cache misses')
        self.errors = Counter('l3_cache_errors_total', 'L3 cache errors')

    async def get(self, key: str) -> Optional[Any]:
        """Get value from L3 cache"""
        try:
            file_path = self._get_file_path(key)

            if os.path.exists(file_path):
                # Check if file is too old (TTL)
                if time.time() - os.path.getmtime(file_path) > 86400:  # 24 hours
                    os.remove(file_path)
                    self.misses.inc()
                    return None

                with open(file_path, 'r') as f:
                    data = json.load(f)

                self.hits.inc()
                return data['value']

            self.misses.inc()
            return None

        except Exception as e:
            self.errors.inc()
            return None

    async def set(self, key: str, value: Any, ttl: int = 86400) -> bool:
        """Set value in L3 cache"""
        try:
            # Check if we need to clean up
            await self._cleanup_if_needed()

            file_path = self._get_file_path(key)
            data = {
                'value': value,
                'created_at': time.time(),
                'ttl': ttl
            }

            with open(file_path, 'w') as f:
                json.dump(data, f)

            return True

        except Exception as e:
            self.errors.inc()
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from L3 cache"""
        try:
            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            self.errors.inc()
            return False

    async def clear(self) -> bool:
        """Clear all L3 cache entries"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            return True
        except Exception as e:
            self.errors.inc()
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get L3 cache statistics"""
        try:
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files)

            return {
                'files_count': len(files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'hit_rate': self.hits._value / max(self.hits._value + self.misses._value, 1)
            }
        except:
            return {'error': 'Unable to get disk cache stats'}

    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")

    async def _cleanup_if_needed(self):
        """Clean up old files if cache is too large"""
        try:
            files = [(f, os.path.getmtime(os.path.join(self.cache_dir, f)))
                    for f in os.listdir(self.cache_dir) if f.endswith('.json')]

            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[1])

            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f[0])) for f in files)

            # Remove oldest files until we're under the limit
            while total_size > self.max_size_bytes * 0.8 and files:
                oldest_file = files.pop(0)
                file_path = os.path.join(self.cache_dir, oldest_file[0])
                total_size -= os.path.getsize(file_path)
                os.remove(file_path)

        except Exception:
            pass  # Ignore cleanup errors


class MultiLevelCache:
    """
    Multi-level caching system with automatic promotion/demotion
    L1 -> L2 -> L3 hierarchy
    """

    def __init__(self, redis_client: redis.Redis):
        self.l1_cache = L1Cache(max_size=10000, max_memory_mb=1024)
        self.l2_cache = L2Cache(redis_client)
        self.l3_cache = L3Cache(max_size_gb=50.0)

        # Metrics
        self.total_requests = Counter('cache_total_requests', 'Total cache requests')
        self.cache_hits = Counter('cache_hits_total', 'Cache hits by level', ['level'])
        self.cache_misses = Counter('cache_misses_total', 'Cache misses')
        # Инициализируем задачу для периодической очистки устаревших записей в L1 кэше
        self._cleanup_task = None

    async def __aenter__(self):
        """Асинхронный контекстный менеджер для запуска задачи очистки"""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер для остановки задачи очистки"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass  # Игнорируем исключение отмены задачи

    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        self.total_requests.inc()

        # Try L1 first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.cache_hits.labels(level='l1').inc()
            return value

        # Try L2
        value = await self.l2_cache.get(key)
        if value is not None:
            self.cache_hits.labels(level='l2').inc()
            # Promote to L1
            await self.l1_cache.set(key, value, ttl=1800)  # 30 minutes in L1
            return value

        # Try L3
        value = await self.l3_cache.get(key)
        if value is not None:
            self.cache_hits.labels(level='l3').inc()
            # Promote to L1 and L2
            await self.l1_cache.set(key, value, ttl=1800)
            await self.l2_cache.set(key, value, ttl=3600)  # 1 hour in L2
            return value

        self.cache_misses.inc()
        return None

    async def _cleanup_expired_entries(self):
        """Фоновая задача для очистки устаревших записей в L1 кэше"""
        while True:
            try:
                # Ждем 5 минут перед следующей проверкой
                await asyncio.sleep(300)
                
                # Получаем текущее время
                current_time = time.time()
                expired_keys = []
                
                # Проверяем каждый элемент в кэше на просрочку
                for key, entry in self.l1_cache.cache.items():
                    if current_time - entry.created_at > entry.ttl:
                        expired_keys.append(key)
                
                # Удаляем просроченные элементы
                for key in expired_keys:
                    await self.l1_cache.delete(key)
                    
                # Также вызываем сборщик мусора для предотвращения утечек памяти
                import gc
                gc.collect()
                    
            except asyncio.CancelledError:
                # Завершаем задачу при отмене
                break
            except Exception:
                # Игнорируем ошибки очистки, чтобы не нарушать работу кэша
                pass

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in all cache levels"""
        success = True

        # Set in all levels
        l1_success = await self.l1_cache.set(key, value, ttl=min(ttl, 1800))  # Max 30 min in L1
        l2_success = await self.l2_cache.set(key, value, ttl=ttl)
        l3_success = await self.l3_cache.set(key, value, ttl=max(ttl, 86400))  # Min 24 hours in L3

        return l1_success and l2_success and l3_success

    async def start_cleanup_task(self):
        """Запуск фоновой задачи очистки устаревших записей"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())

    async def stop_cleanup_task(self):
        """Остановка фоновой задачи очистки устаревших записей"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass  # Игнорируем исключение отмены задачи
            self._cleanup_task = None

    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels"""
        l1_deleted = await self.l1_cache.delete(key)
        l2_deleted = await self.l2_cache.delete(key)
        l3_deleted = await self.l3_cache.delete(key)

        return l1_deleted or l2_deleted or l3_deleted

    async def clear(self) -> bool:
        """Clear all cache levels"""
        await self.l1_cache.clear()
        await self.l2_cache.clear()
        await self.l3_cache.clear()
        # Отменяем задачу очистки при очистке кэша
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'l1': await self.l1_cache.get_stats(),
            'l2': await self.l2_cache.get_stats(),
            'l3': await self.l3_cache.get_stats(),
            'overall_hit_rate': sum([
                self.cache_hits.labels(level='l1')._value,
                self.cache_hits.labels(level='l2')._value,
                self.cache_hits.labels(level='l3')._value
            ]) / max(self.total_requests._value, 1)
        }

    async def warmup(self, keys: List[str]):
        """Warm up cache with frequently accessed keys"""
        for key in keys:
            # Try to get from L3 and promote up
            value = await self.l3_cache.get(key)
            if value:
                await self.set(key, value)



# Global cache instance
cache = None

def get_cache_instance() -> MultiLevelCache:
    """Get global cache instance"""
    global cache
    if cache is None:
        # Initialize Redis client
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis-service'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=False
        )
        cache = MultiLevelCache(redis_client)
    return cache


# Utility functions for cache keys
def make_cache_key(*args, **kwargs) -> str:
    """Generate consistent cache key"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


# Cache decorators
def cached(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for automatic caching"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache_instance()

            # Generate cache key
            cache_key = f"{key_prefix}:{make_cache_key(func.__name__, *args, **kwargs)}"

            # Try cache first
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


def invalidate_cache(*keys):
    """Decorator to invalidate cache keys after function execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Invalidate specified keys
            cache = get_cache_instance()
            for key in keys:
                await cache.delete(key)

            return result

        return wrapper
    return decorator

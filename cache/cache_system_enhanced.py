import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Union, List
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import OrderedDict

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available, using basic metrics")

@dataclass
class CacheStats:
    """Статистика кэширования"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    compression_savings: int = 0

@dataclass
class CacheEntry:
    """Запись кэша с метаданными"""
    value: Any
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    size_bytes: int = 0

class LRUCache:
    """
    LRU (Least Recently Used) кэш для эффективного управления памятью.

    Особенности:
    - Автоматическое удаление наименее недавно использованных элементов
    - Ограничение по размеру и памяти
    - TTL (Time To Live) для записей
    - Статистика использования
    """

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 256, name: str = "lru_cache"):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self.name = name

        # Используем OrderedDict для поддержания порядка доступа
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Статистика
        self.stats = CacheStats()

        # Метрики производительности
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        else:
            self.metrics = None

    def _setup_prometheus_metrics(self):
        """Настройка Prometheus метрик для LRU кэша"""
        try:
            self.metrics = {
                'cache_hits': Counter(
                    f'{self.name}_hits_total',
                    f'Number of cache hits for {self.name}',
                    ['cache_name']
                ),
                'cache_misses': Counter(
                    f'{self.name}_misses_total',
                    f'Number of cache misses for {self.name}',
                    ['cache_name']
                ),
                'cache_sets': Counter(
                    f'{self.name}_sets_total',
                    f'Number of cache sets for {self.name}',
                    ['cache_name']
                ),
                'cache_deletes': Counter(
                    f'{self.name}_deletes_total',
                    f'Number of cache deletes for {self.name}',
                    ['cache_name']
                ),
                'cache_evictions': Counter(
                    f'{self.name}_evictions_total',
                    f'Number of cache evictions for {self.name}',
                    ['cache_name']
                ),
                'cache_size': Gauge(
                    f'{self.name}_size',
                    f'Current cache size for {self.name}',
                    ['cache_name']
                ),
                'cache_memory_usage': Gauge(
                    f'{self.name}_memory_bytes',
                    f'Current memory usage for {self.name}',
                    ['cache_name']
                ),
                'cache_hit_rate': Gauge(
                    f'{self.name}_hit_rate',
                    f'Cache hit rate for {self.name}',
                    ['cache_name']
                ),
                'cache_operation_duration': Histogram(
                    f'{self.name}_operation_duration_seconds',
                    f'Cache operation duration for {self.name}',
                    ['cache_name', 'operation']
                )
            }
        except Exception as e:
            logger.warning(f"Failed to setup Prometheus metrics for {self.name}: {e}")
            self.metrics = None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        Установить значение в кэш

        Args:
            key: Ключ
            value: Значение
            ttl: Время жизни в секундах (опционально)

        Returns:
            bool: Успешно ли установлено
        """
        try:
            start_time = time.time()

            # Вычисляем размер значения
            size_bytes = len(json.dumps(value).encode('utf-8')) if value is not None else 0

            # Проверяем лимит памяти
            if self.current_memory_bytes + size_bytes > self.max_memory_bytes:
                self._evict_to_fit(size_bytes)

            # Создаем запись
            expires_at = time.time() + ttl if ttl else float('inf')
            entry = CacheEntry(
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes
            )

            # Если ключ уже существует, обновляем
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_bytes -= old_entry.size_bytes
                del self.cache[key]
            else:
                # Проверяем лимит размера
                if len(self.cache) >= self.max_size:
                    self._evict_lru()

            # Добавляем новую запись
            self.cache[key] = entry
            self.cache.move_to_end(key)  # Перемещаем в конец (most recently used)
            self.current_memory_bytes += size_bytes

            self.stats.sets += 1

            # Обновляем метрики
            if self.metrics:
                self.metrics['cache_sets'].labels(cache_name=self.name).inc()
                self.metrics['cache_size'].labels(cache_name=self.name).set(len(self.cache))
                self.metrics['cache_memory_usage'].labels(cache_name=self.name).set(self.current_memory_bytes)

            duration = time.time() - start_time
            if self.metrics:
                self.metrics['cache_operation_duration'].labels(cache_name=self.name, operation='set').observe(duration)

            return True

        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")
            self.stats.errors += 1
            return False

    def get(self, key: str) -> Any:
        """
        Получить значение из кэша

        Args:
            key: Ключ

        Returns:
            Значение или None если не найдено
        """
        try:
            start_time = time.time()

            if key not in self.cache:
                self.stats.misses += 1
                if self.metrics:
                    self.metrics['cache_misses'].labels(cache_name=self.name).inc()
                duration = time.time() - start_time
                if self.metrics:
                    self.metrics['cache_operation_duration'].labels(cache_name=self.name, operation='get').observe(duration)
                return None

            entry = self.cache[key]

            # Проверяем срок действия
            if time.time() > entry.expires_at:
                self.delete(key)
                self.stats.misses += 1
                if self.metrics:
                    self.metrics['cache_misses'].labels(cache_name=self.name).inc()
                return None

            # Обновляем статистику доступа
            entry.access_count += 1
            entry.last_accessed = time.time()

            # Перемещаем в конец (most recently used)
            self.cache.move_to_end(key)

            self.stats.hits += 1
            if self.metrics:
                self.metrics['cache_hits'].labels(cache_name=self.name).inc()

            duration = time.time() - start_time
            if self.metrics:
                self.metrics['cache_operation_duration'].labels(cache_name=self.name, operation='get').observe(duration)

            return entry.value

        except Exception as e:
            logger.error(f"Failed to get cache entry {key}: {e}")
            self.stats.errors += 1
            return None

    def delete(self, key: str) -> bool:
        """
        Удалить значение из кэша

        Args:
            key: Ключ

        Returns:
            bool: Успешно ли удалено
        """
        try:
            if key in self.cache:
                entry = self.cache[key]
                self.current_memory_bytes -= entry.size_bytes
                del self.cache[key]
                self.stats.deletes += 1

                if self.metrics:
                    self.metrics['cache_deletes'].labels(cache_name=self.name).inc()
                    self.metrics['cache_size'].labels(cache_name=self.name).set(len(self.cache))
                    self.metrics['cache_memory_usage'].labels(cache_name=self.name).set(self.current_memory_bytes)

                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete cache entry {key}: {e}")
            self.stats.errors += 1
            return False

    def clear(self) -> bool:
        """
        Очистить весь кэш

        Returns:
            bool: Успешно ли очищен
        """
        try:
            self.cache.clear()
            self.current_memory_bytes = 0
            self.stats = CacheStats()

            if self.metrics:
                self.metrics['cache_size'].labels(cache_name=self.name).set(0)
                self.metrics['cache_memory_usage'].labels(cache_name=self.name).set(0)

            return True

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику кэша

        Returns:
            Dict[str, Any]: Статистика
        """
        total_requests = self.stats.hits + self.stats.misses
        hit_rate = self.stats.hits / total_requests if total_requests > 0 else 0.0

        return {
            'size': len(self.cache),
            'memory_usage_mb': self.current_memory_bytes / (1024 * 1024),
            'max_size': self.max_size,
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate': hit_rate,
            'sets': self.stats.sets,
            'deletes': self.stats.deletes,
            'errors': self.stats.errors
        }

    def _evict_lru(self):
        """Удалить наименее недавно использованный элемент"""
        if not self.cache:
            return

        # Находим самый старый элемент (первый в OrderedDict)
        key, entry = next(iter(self.cache.items()))
        self.current_memory_bytes -= entry.size_bytes
        del self.cache[key]

        if self.metrics:
            self.metrics['cache_evictions'].labels(cache_name=self.name).inc()

    def _evict_to_fit(self, required_bytes: int):
        """Удалить элементы пока не освободится достаточно места"""
        while self.current_memory_bytes + required_bytes > self.max_memory_bytes and self.cache:
            self._evict_lru()

    def cleanup_expired(self):
        """Очистить истекшие записи"""
        expired_keys = []
        current_time = time.time()

        for key, entry in self.cache.items():
            if current_time > entry.expires_at:
                expired_keys.append(key)

        for key in expired_keys:
            self.delete(key)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Инвалидировать записи по шаблону

        Args:
            pattern: Шаблон для поиска ключей (поддерживает * как wildcard)

        Returns:
            int: Количество удаленных записей
        """
        import fnmatch
        deleted_count = 0

        keys_to_delete = []
        for key in self.cache.keys():
            if fnmatch.fnmatch(key, pattern):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            if self.delete(key):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Invalidated {deleted_count} cache entries matching pattern: {pattern}")

        return deleted_count

    def invalidate_namespace(self, namespace: str) -> int:
        """
        Инвалидировать все записи в пространстве имен

        Args:
            namespace: Префикс пространства имен

        Returns:
            int: Количество удаленных записей
        """
        pattern = f"{namespace}:*"
        return self.invalidate_pattern(pattern)

    def invalidate_by_age(self, max_age_seconds: float) -> int:
        """
        Инвалидировать записи старше указанного возраста

        Args:
            max_age_seconds: Максимальный возраст в секундах

        Returns:
            int: Количество удаленных записей
        """
        current_time = time.time()
        deleted_count = 0

        keys_to_delete = []
        for key, entry in self.cache.items():
            age = current_time - entry.last_accessed
            if age > max_age_seconds:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            if self.delete(key):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Invalidated {deleted_count} cache entries older than {max_age_seconds} seconds")

        return deleted_count

    def invalidate_by_access_count(self, min_access_count: int) -> int:
        """
        Инвалидировать записи с количеством доступов меньше указанного

        Args:
            min_access_count: Минимальное количество доступов

        Returns:
            int: Количество удаленных записей
        """
        deleted_count = 0

        keys_to_delete = []
        for key, entry in self.cache.items():
            if entry.access_count < min_access_count:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            if self.delete(key):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Invalidated {deleted_count} cache entries with access count < {min_access_count}")

        return deleted_count

    def get_cache_health(self) -> Dict[str, Any]:
        """
        Получить показатели здоровья кэша

        Returns:
            Dict[str, Any]: Показатели здоровья
        """
        stats = self.get_stats()
        current_time = time.time()

        # Анализ возраста записей
        ages = []
        access_counts = []
        expired_count = 0

        for entry in self.cache.values():
            age = current_time - entry.last_accessed
            ages.append(age)
            access_counts.append(entry.access_count)

            if current_time > entry.expires_at:
                expired_count += 1

        avg_age = sum(ages) / len(ages) if ages else 0
        avg_access_count = sum(access_counts) / len(access_counts) if access_counts else 0
        max_age = max(ages) if ages else 0
        min_age = min(ages) if ages else 0

        # Расчет эффективности использования памяти
        memory_efficiency = (stats['memory_usage_mb'] / stats['max_memory_mb']) * 100 if stats['max_memory_mb'] > 0 else 0

        # Расчет фрагментации (отношение размера к использованию памяти)
        size_efficiency = (len(self.cache) / stats['max_size']) * 100 if stats['max_size'] > 0 else 0

        return {
            'hit_rate': stats['hit_rate'],
            'memory_efficiency_percent': memory_efficiency,
            'size_efficiency_percent': size_efficiency,
            'expired_entries_count': expired_count,
            'avg_entry_age_seconds': avg_age,
            'max_entry_age_seconds': max_age,
            'min_entry_age_seconds': min_age,
            'avg_access_count': avg_access_count,
            'error_rate': stats['errors'] / (stats['hits'] + stats['misses'] + stats['sets'] + stats['deletes']) if (stats['hits'] + stats['misses'] + stats['sets'] + stats['deletes']) > 0 else 0,
            'health_score': self._calculate_health_score(stats, expired_count, memory_efficiency, size_efficiency)
        }

    def _calculate_health_score(self, stats: Dict, expired_count: int, memory_efficiency: float, size_efficiency: float) -> float:
        """
        Рассчитать оценку здоровья кэша (0-100)

        Args:
            stats: Статистика кэша
            expired_count: Количество истекших записей
            memory_efficiency: Эффективность использования памяти (%)
            size_efficiency: Эффективность использования размера (%)

        Returns:
            float: Оценка здоровья (0-100)
        """
        score = 100.0

        # Штраф за низкий hit rate
        if stats['hit_rate'] < 0.5:
            score -= (0.5 - stats['hit_rate']) * 100
        elif stats['hit_rate'] > 0.8:
            score += (stats['hit_rate'] - 0.8) * 50  # Бонус за высокий hit rate

        # Штраф за истекшие записи
        if expired_count > 0:
            score -= min(expired_count * 5, 20)

        # Штраф за низкую эффективность использования памяти
        if memory_efficiency < 50:
            score -= (50 - memory_efficiency) * 0.5

        # Штраф за низкую эффективность использования размера
        if size_efficiency < 50:
            score -= (50 - size_efficiency) * 0.5

        # Штраф за ошибки
        if stats['errors'] > 0:
            score -= min(stats['errors'] * 2, 15)

        return max(0.0, min(100.0, score))


@dataclass
class CacheThreshold:
    """Пороги для мониторинга кэша"""
    min_hit_rate: float = 0.5
    max_memory_usage_percent: float = 80.0
    max_size_usage_percent: float = 90.0
    max_error_rate: float = 0.05
    cleanup_interval_seconds: int = 300


class EnhancedCache:
    """
    Улучшенная версия кэша с дополнительными функциями
    """

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 256, name: str = "enhanced_cache"):
        self.lru_cache = LRUCache(max_size=max_size, max_memory_mb=max_memory_mb, name=name)
        self.thresholds = CacheThreshold()
        self.name = name

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Установить значение в кэш"""
        return self.lru_cache.set(key, value, ttl)

    def get(self, key: str) -> Any:
        """Получить значение из кэша"""
        return self.lru_cache.get(key)

    def delete(self, key: str) -> bool:
        """Удалить значение из кэша"""
        return self.lru_cache.delete(key)

    def clear(self) -> bool:
        """Очистить кэш"""
        return self.lru_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша"""
        return self.lru_cache.get_stats()

    def get_health(self) -> Dict[str, Any]:
        """Получить показатели здоровья кэша"""
        return self.lru_cache.get_cache_health()

    def cleanup_expired(self):
        """Очистить истекшие записи"""
        self.lru_cache.cleanup_expired()

    def invalidate_pattern(self, pattern: str) -> int:
        """Инвалидировать записи по шаблону"""
        return self.lru_cache.invalidate_pattern(pattern)

    def check_thresholds(self) -> Dict[str, bool]:
        """Проверить превышение порогов"""
        stats = self.get_stats()
        health = self.get_health()

        return {
            'hit_rate_ok': stats.get('hit_rate', 0) >= self.thresholds.min_hit_rate,
            'memory_ok': stats.get('memory_usage_mb', 0) / self.lru_cache.max_memory_bytes * 100 <= self.thresholds.max_memory_usage_percent,
            'size_ok': len(self.lru_cache.cache) / self.lru_cache.max_size * 100 <= self.thresholds.max_size_usage_percent,
            'errors_ok': health.get('error_rate', 0) <= self.thresholds.max_error_rate
        }


class EnhancedCacheSystem:
    """
    Система управления множественными кэшами
    """

    def __init__(self):
        self.caches: Dict[str, EnhancedCache] = {}
        self.default_cache = EnhancedCache(name="default")

    def create_cache(self, name: str, max_size: int = 1000, max_memory_mb: int = 256) -> EnhancedCache:
        """Создать новый кэш"""
        cache = EnhancedCache(max_size=max_size, max_memory_mb=max_memory_mb, name=name)
        self.caches[name] = cache
        return cache

    def get_cache(self, name: str) -> EnhancedCache:
        """Получить кэш по имени"""
        return self.caches.get(name, self.default_cache)

    def set(self, key: str, value: Any, cache_name: str = "default", ttl: Optional[float] = None) -> bool:
        """Установить значение в указанный кэш"""
        cache = self.get_cache(cache_name)
        return cache.set(key, value, ttl)

    def get(self, key: str, cache_name: str = "default") -> Any:
        """Получить значение из указанного кэша"""
        cache = self.get_cache(cache_name)
        return cache.get(key)

    def delete(self, key: str, cache_name: str = "default") -> bool:
        """Удалить значение из указанного кэша"""
        cache = self.get_cache(cache_name)
        return cache.delete(key)

    def clear(self, cache_name: str = "default") -> bool:
        """Очистить указанный кэш"""
        cache = self.get_cache(cache_name)
        return cache.clear()

    def get_stats(self, cache_name: str = "default") -> Dict[str, Any]:
        """Получить статистику указанного кэша"""
        cache = self.get_cache(cache_name)
        return cache.get_stats()

    def get_system_stats(self) -> Dict[str, Any]:
        """Получить статистику всей системы кэширования"""
        system_stats = {
            'total_caches': len(self.caches) + 1,  # +1 для default
            'cache_stats': {}
        }

        # Статистика всех кэшей
        for name, cache in self.caches.items():
            system_stats['cache_stats'][name] = cache.get_stats()

        # Статистика default кэша
        system_stats['cache_stats']['default'] = self.default_cache.get_stats()

        return system_stats

    def cleanup_all(self):
        """Очистить истекшие записи во всех кэшах"""
        for cache in self.caches.values():
            cache.cleanup_expired()
        self.default_cache.cleanup_expired()

    def check_all_thresholds(self) -> Dict[str, Dict[str, bool]]:
        """Проверить пороги для всех кэшей"""
        results = {}
        for name, cache in self.caches.items():
            results[name] = cache.check_thresholds()
        results['default'] = self.default_cache.check_thresholds()
        return results

"""
Асинхронный менеджер памяти с оптимизацией для работы с большими объемами данных
"""

import logging
import asyncio
import gc
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple, AsyncIterator
from datetime import datetime, timedelta
from collections import deque, OrderedDict
import threading
import time
from weakref import WeakValueDictionary
import pickle
import zlib
from concurrent.futures import ThreadPoolExecutor

from ..core.models import MemoryEntry

logger = logging.getLogger("AsyncMemoryManager")

class AsyncMemoryEntry:
    """Асинхронная запись памяти с поддержкой сериализации и сжатия"""
    
    def __init__(self, entry_id: str, data: Any, importance: float = 0.5, tags: Optional[List[str]] = None):
        self.entry_id = entry_id
        self.data = data
        self.importance = importance
        self.tags = tags or []
        self.created_at = datetime.now()
        self.access_count = 0
        self.last_access = datetime.now()
        self.size_bytes = self._estimate_size(data)
        self.is_compressed = False

    def _estimate_size(self, data: Any) -> int:
        """Оценка размера данных в байтах"""
        try:
            # Попробуем сериализовать данные для получения точного размера
            serialized = pickle.dumps(data)
            return len(serialized)
        except:
            # Резервная оценка
            if isinstance(data, (str, bytes)):
                return len(data.encode('utf-8') if isinstance(data, str) else data)
            elif isinstance(data, dict):
                return sum(self._estimate_size(v) for v in data.values())
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_size(item) for item in data)
            else:
                return 1024  # 1KB по умолчанию

    def compress(self) -> bool:
        """Сжатие данных для экономии памяти"""
        if self.is_compressed:
            return True
            
        try:
            serialized_data = pickle.dumps(self.data)
            compressed_data = zlib.compress(serialized_data)
            # Только сжимаем, если получаем экономию > 20%
            if len(compressed_data) < len(serialized_data) * 0.8:
                self.data = compressed_data
                self.is_compressed = True
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to compress entry {self.entry_id}: {e}")
            return False

    def decompress(self) -> bool:
        """Распаковка данных"""
        if not self.is_compressed:
            return True
            
        try:
            decompressed_data = zlib.decompress(self.data)
            self.data = pickle.loads(decompressed_data)
            self.is_compressed = False
            return True
        except Exception as e:
            logger.error(f"Failed to decompress entry {self.entry_id}: {e}")
            return False

    def update_access(self):
        """Обновление статистики доступа"""
        self.access_count += 1
        self.last_access = datetime.now()

    def get_age_days(self) -> float:
        """Получение возраста записи в днях"""
        return (datetime.now() - self.created_at).total_seconds() / (24 * 3600)

    def get_inactivity_days(self) -> float:
        """Получение дней без доступа"""
        return (datetime.now() - self.last_access).total_seconds() / (24 * 3600)

    def calculate_relevance_score(self) -> float:
        """Расчет релевантности записи"""
        # Формула: importance * (1 / (1 + age)) * (1 / (1 + inactivity)) * access_bonus
        age_penalty = 1 / (1 + self.get_age_days())
        inactivity_penalty = 1 / (1 + self.get_inactivity_days())
        access_bonus = min(2.0, 1 + (self.access_count * 0.1))

        return self.importance * age_penalty * inactivity_penalty * access_bonus

class AsyncMemoryPool:
    """Асинхронный пул памяти с поддержкой сжатия и потокобезопасности"""
    
    def __init__(self, max_size_mb: float = 100.0, cleanup_threshold: float = 0.8, compression_threshold: float = 0.7):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cleanup_threshold = cleanup_threshold  # Процент заполнения для запуска очистки
        self.compression_threshold = compression_threshold  # Порог для сжатия
        self.current_size_bytes = 0
        self.entries: OrderedDict[str, AsyncMemoryEntry] = OrderedDict()
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.compression_enabled = True

    async def add_entry(self, entry_id: str, data: Any, importance: float = 0.5, tags: Optional[List[str]] = None) -> bool:
        """
        Асинхронное добавление записи в пул

        Args:
            entry_id: ID записи
            data: Данные записи
            importance: Важность (0.0-1.0)
            tags: Теги для классификации

        Returns:
            True если добавлено, False если нет места
        """
        async with self.lock:
            # Создание асинхронной записи
            entry = AsyncMemoryEntry(entry_id, data, importance, tags)

            # Проверка доступного места
            if self.current_size_bytes + entry.size_bytes > self.max_size_bytes:
                # Попытка очистки
                if not await self._cleanup_space(entry.size_bytes):
                    return False

            # Добавление записи
            self.entries[entry_id] = entry
            self.current_size_bytes += entry.size_bytes

            # Асинхронное сжатие при необходимости
            if self.compression_enabled and self._should_compress():
                asyncio.create_task(self._compress_entries())

            return True

    async def get_entry(self, entry_id: str) -> Optional[Any]:
        """Асинхронное получение записи"""
        async with self.lock:
            if entry_id in self.entries:
                entry = self.entries[entry_id]
                entry.update_access()
                
                # Распаковка при необходимости
                if entry.is_compressed:
                    entry.decompress()
                
                return entry.data
            return None

    async def remove_entry(self, entry_id: str) -> bool:
        """Асинхронное удаление записи"""
        async with self.lock:
            if entry_id in self.entries:
                entry = self.entries[entry_id]
                size_bytes = entry.size_bytes
                del self.entries[entry_id]
                self.current_size_bytes -= size_bytes
                return True
            return False

    async def get_entries_by_tag(self, tag: str) -> List[Tuple[str, Any]]:
        """Получение записей по тегу"""
        async with self.lock:
            results = []
            for entry_id, entry in self.entries.items():
                if tag in entry.tags:
                    # Распаковка при необходимости
                    if entry.is_compressed:
                        entry.decompress()
                    results.append((entry_id, entry.data))
            return results

    async def get_stats(self) -> Dict[str, Any]:
        """Асинхронное получение статистики пула"""
        async with self.lock:
            compressed_count = sum(1 for entry in self.entries.values() if entry.is_compressed)
            return {
                "current_size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization_percent": (self.current_size_bytes / self.max_size_bytes) * 100,
                "entry_count": len(self.entries),
                "compressed_entries": compressed_count,
                "cleanup_threshold": self.cleanup_threshold * 100,
                "compression_enabled": self.compression_enabled
            }

    async def _cleanup_space(self, required_bytes: int) -> bool:
        """
        Асинхронная очистка пространства для новых данных

        Returns:
            True если удалось освободить достаточно места
        """
        async with self.lock:
            # Сортировка записей по релевантности (от менее релевантных к более)
            entries_by_relevance = sorted(
                self.entries.items(),
                key=lambda x: x[1].calculate_relevance_score()
            )

            freed_bytes = 0
            removed_count = 0

            for entry_id, entry in entries_by_relevance:
                if freed_bytes >= required_bytes:
                    break

                # Не удаляем очень важные записи
                if entry.importance >= 0.8:
                    continue

                # Удаление записи
                del self.entries[entry_id]
                freed_bytes += entry.size_bytes
                removed_count += 1

            logger.info(f"Memory pool cleanup: freed {freed_bytes} bytes, removed {removed_count} entries")

            return freed_bytes >= required_bytes

    def _should_compress(self) -> bool:
        """Проверка необходимости сжатия"""
        utilization = self.current_size_bytes / self.max_size_bytes
        return utilization > self.compression_threshold

    async def _compress_entries(self):
        """Асинхронное сжатие записей"""
        async with self.lock:
            # Сортировка по размеру и давности использования
            entries_to_compress = sorted(
                self.entries.items(),
                key=lambda x: (x[1].size_bytes, x[1].get_inactivity_days()),
                reverse=True
            )[:10]  # Сжимаем только 10 самых больших/старых записей
            
            for entry_id, entry in entries_to_compress:
                if not entry.is_compressed and entry.size_bytes > 1024:  # > 1KB
                    entry.compress()
                    # Обновляем размер после сжатия
                    if entry.is_compressed:
                        self.current_size_bytes -= (entry.size_bytes - len(entry.data))

    async def cleanup_expired_entries(self, max_age_days: float = 30.0):
        """Асинхронная очистка устаревших записей"""
        async with self.lock:
            expired_ids = []
            for entry_id, entry in self.entries.items():
                if entry.get_age_days() > max_age_days and entry.importance < 0.7:
                    expired_ids.append(entry_id)

            for entry_id in expired_ids:
                await self.remove_entry(entry_id)

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired memory entries")

    async def clear(self):
        """Очистка всех записей"""
        async with self.lock:
            self.entries.clear()
            self.current_size_bytes = 0

class AsyncMemoryManager:
    """
    Асинхронный менеджер памяти с оптимизацией для больших объемов данных.
    
    Ключевые улучшения:
    - Полностью асинхронные операции
    - Поддержка сжатия данных
    - Потокобезопасность
    - Оптимизация для больших объемов данных
    - Стриминг данных
    """

    def __init__(self,
                 max_episodic_entries: int = 1000,
                 max_working_memory_mb: float = 50.0,
                 max_semantic_memory_mb: float = 100.0,
                 cleanup_interval_seconds: int = 300):
        """
        Инициализация асинхронного менеджера памяти

        Args:
            max_episodic_entries: Максимальное количество эпизодических записей
            max_working_memory_mb: Максимальный размер рабочей памяти (MB)
            max_semantic_memory_mb: Максимальный размер семантической памяти (MB)
            cleanup_interval_seconds: Интервал автоматической очистки (секунды)
        """
        # Асинхронные пулы памяти с ограничениями
        self.episodic_pool = AsyncMemoryPool(max_size_mb=max_semantic_memory_mb)
        self.working_pool = AsyncMemoryPool(max_size_mb=max_working_memory_mb)
        self.semantic_pool = AsyncMemoryPool(max_size_mb=max_semantic_memory_mb)

        # Ограничения для обратной совместимости
        self.max_episodic_entries = max_episodic_entries

        # Статистика и мониторинг
        self.memory_stats = {
            "total_operations": 0,
            "cleanup_operations": 0,
            "memory_warnings": 0,
            "last_cleanup": datetime.now(),
            "start_time": datetime.now()
        }

        # Автоматическая очистка
        self.cleanup_interval = cleanup_interval_seconds
        self._cleanup_task = None

        # Weak references для предотвращения циклических ссылок
        self._weak_references = WeakValueDictionary()

        logger.info("🚀 AsyncMemoryManager initialized with async memory pools")

    async def start_cleanup_task(self):
        """Запуск асинхронной задачи очистки"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())
            logger.info("Started async memory cleanup task")

    async def stop_cleanup_task(self):
        """Остановка асинхронной задачи очистки"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped async memory cleanup task")

    # ===== АСИНХРОННАЯ ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ =====

    async def store_episodic_memory(self, memory_data: Dict[str, Any], importance: float = 0.5, tags: Optional[List[str]] = None):
        """
        Асинхронное сохранение эпизодической памяти с оптимизацией

        Args:
            memory_data: Данные для сохранения
            importance: Важность записи (0.0-1.0)
            tags: Теги для классификации
        """
        try:
            self.memory_stats["total_operations"] += 1

            # Создание ID для записи
            entry_id = f"episodic_{int(time.time() * 1000000)}_{hash(str(memory_data)) % 10000}"

            # Попытка добавления в пул
            if await self.episodic_pool.add_entry(entry_id, memory_data, importance, tags):
                logger.debug(f"Stored episodic memory: {entry_id}")
            else:
                logger.warning(f"Failed to store episodic memory: insufficient space")

        except Exception as e:
            logger.error(f"❌ Failed to store episodic memory: {e}")

    async def retrieve_episodic_memory(self, limit: int = 10, min_importance: float = 0.0) -> List[Any]:
        """
        Асинхронное получение эпизодической памяти с фильтрацией

        Args:
            limit: Максимальное количество записей
            min_importance: Минимальная важность

        Returns:
            Список записей памяти
        """
        # Получение записей, отсортированных по релевантности
        relevant_entries = []

        # Получаем все записи из пула и сортируем по релевантности
        for entry_id in list(self.episodic_pool.entries.keys()):
            data = await self.episodic_pool.get_entry(entry_id)
            if data:
                entry = self.episodic_pool.entries[entry_id]
                if entry.importance >= min_importance:
                    relevant_entries.append((data, entry.calculate_relevance_score()))

        # Сортировка по релевантности и возврат топ-N
        relevant_entries.sort(key=lambda x: x[1], reverse=True)
        return [data for data, score in relevant_entries[:limit]]

    async def retrieve_episodic_memory_by_tag(self, tag: str, limit: int = 10) -> List[Any]:
        """Получение эпизодической памяти по тегу"""
        entries = await self.episodic_pool.get_entries_by_tag(tag)
        # Сортируем по релевантности
        entries_with_relevance = []
        for entry_id, data in entries:
            entry = self.episodic_pool.entries[entry_id]
            entries_with_relevance.append((data, entry.calculate_relevance_score()))
        
        entries_with_relevance.sort(key=lambda x: x[1], reverse=True)
        return [data for data, score in entries_with_relevance[:limit]]

    # ===== АСИНХРОННАЯ РАБОЧАЯ ПАМЯТЬ =====

    async def store_working_memory(self, key: str, value: Any, ttl_seconds: Optional[int] = None, importance: float = 0.7, tags: Optional[List[str]] = None):
        """
        Асинхронное сохранение в рабочей памяти с TTL

        Args:
            key: Ключ
            value: Значение
            ttl_seconds: Время жизни (None - бессрочно)
            importance: Важность
            tags: Теги для классификации
        """
        try:
            self.memory_stats["total_operations"] += 1

            # Добавление TTL в данные
            enriched_value = {
                "data": value,
                "expires_at": (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds else None,
                "stored_at": datetime.now().isoformat()
            }

            if await self.working_pool.add_entry(key, enriched_value, importance, tags):
                logger.debug(f"Stored working memory: {key}")
            else:
                logger.warning(f"Failed to store working memory: {key} - insufficient space")

        except Exception as e:
            logger.error(f"❌ Failed to store working memory: {e}")

    async def retrieve_working_memory(self, key: str) -> Optional[Any]:
        """
        Асинхронное получение из рабочей памяти с проверкой TTL

        Args:
            key: Ключ

        Returns:
            Значение или None
        """
        try:
            enriched_value = await self.working_pool.get_entry(key)
            if not enriched_value:
                return None

            # Проверка TTL
            if enriched_value.get("expires_at"):
                expires_at = datetime.fromisoformat(enriched_value["expires_at"])
                if datetime.now() > expires_at:
                    # Удаление истекшей записи
                    await self.working_pool.remove_entry(key)
                    logger.debug(f"Working memory entry expired: {key}")
                    return None

            return enriched_value["data"]

        except Exception as e:
            logger.error(f"❌ Failed to retrieve working memory: {e}")
            return None

    # ===== АСИНХРОННАЯ СЕМАНТИЧЕСКАЯ ПАМЯТЬ =====

    async def store_semantic_memory(self, key: str, value: Any, importance: float = 0.8, tags: Optional[List[str]] = None):
        """
        Асинхронное сохранение семантической памяти

        Args:
            key: Ключ
            value: Значение
            importance: Важность (высокая по умолчанию для семантической памяти)
            tags: Теги для классификации
        """
        try:
            self.memory_stats["total_operations"] += 1

            if await self.semantic_pool.add_entry(key, value, importance, tags):
                logger.debug(f"Stored semantic memory: {key}")
            else:
                logger.warning(f"Failed to store semantic memory: {key} - insufficient space")

        except Exception as e:
            logger.error(f"❌ Failed to store semantic memory: {e}")

    async def retrieve_semantic_memory(self, key: str) -> Optional[Any]:
        """Асинхронное получение семантической памяти"""
        return await self.semantic_pool.get_entry(key)

    async def retrieve_semantic_memory_by_tag(self, tag: str, limit: int = 10) -> List[Any]:
        """Получение семантической памяти по тегу"""
        entries = await self.semantic_pool.get_entries_by_tag(tag)
        # Сортируем по релевантности
        entries_with_relevance = []
        for entry_id, data in entries:
            entry = self.semantic_pool.entries[entry_id]
            entries_with_relevance.append((data, entry.calculate_relevance_score()))
        
        entries_with_relevance.sort(key=lambda x: x[1], reverse=True)
        return [data for data, score in entries_with_relevance[:limit]]

    # ===== АСИНХРОННЫЙ МОНИТОРИНГ И ОПТИМИЗАЦИЯ =====

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Асинхронное получение подробной статистики использования памяти

        Returns:
            Детальная статистика
        """
        # Сбор статистики от всех пулов
        episodic_stats = await self.episodic_pool.get_stats()
        working_stats = await self.working_pool.get_stats()
        semantic_stats = await self.semantic_pool.get_stats()

        # Системная статистика
        process = psutil.Process(os.getpid())
        system_memory = process.memory_info()

        # Расчет общей эффективности
        total_utilization = (
            episodic_stats["utilization_percent"] +
            working_stats["utilization_percent"] +
            semantic_stats["utilization_percent"]
        ) / 3

        return {
            "pools": {
                "episodic": episodic_stats,
                "working": working_stats,
                "semantic": semantic_stats
            },
            "system": {
                "rss_mb": system_memory.rss / (1024 * 1024),
                "vms_mb": system_memory.vms / (1024 * 1024),
                "cpu_percent": process.cpu_percent(interval=0.1)
            },
            "operations": {
                "total": self.memory_stats["total_operations"],
                "cleanup_count": self.memory_stats["cleanup_operations"],
                "warnings": self.memory_stats["memory_warnings"]
            },
            "performance": {
                "average_utilization_percent": total_utilization,
                "uptime_hours": (datetime.now() - self.memory_stats["start_time"]).total_seconds() / 3600,
                "last_cleanup": self.memory_stats["last_cleanup"].isoformat()
            }
        }

    async def optimize_memory(self) -> Dict[str, Any]:
        """
        Асинхронная оптимизация использования памяти

        Returns:
            Результаты оптимизации
        """
        logger.info("🔄 Starting async memory optimization")

        results = {
            "freed_bytes": 0,
            "removed_entries": 0,
            "pools_optimized": 0,
            "gc_collections": 0
        }

        # Очистка устаревших записей
        await self.episodic_pool.cleanup_expired_entries(max_age_days=7.0)  # Неделя
        await self.working_pool.cleanup_expired_entries(max_age_days=1.0)   # День
        await self.semantic_pool.cleanup_expired_entries(max_age_days=30.0) # Месяц

        # Принудительный сбор мусора в отдельном потоке
        def run_gc():
            return gc.collect()
        collected = await asyncio.get_event_loop().run_in_executor(None, run_gc)
        results["gc_collections"] = collected

        # Обновляем статистику
        self.memory_stats["cleanup_operations"] += 1
        self.memory_stats["last_cleanup"] = datetime.now()

        logger.info(f"✅ Async memory optimization completed: {results}")
        return results

    async def check_memory_health(self) -> Dict[str, Any]:
        """
        Асинхронная проверка здоровья памяти

        Returns:
            Статус здоровья
        """
        stats = await self.get_memory_stats()

        health_status = "healthy"
        issues = []

        # Проверка переполнения пулов
        for pool_name, pool_stats in stats["pools"].items():
            if pool_stats["utilization_percent"] > 90:
                health_status = "critical"
                issues.append(f"{pool_name} pool over 90% full")
            elif pool_stats["utilization_percent"] > 75:
                if health_status == "healthy":
                    health_status = "warning"
                issues.append(f"{pool_name} pool over 75% full")

        # Проверка системной памяти
        if stats["system"]["rss_mb"] > 500:  # 500MB
            health_status = "critical"
            issues.append("High system memory usage")

        return {
            "status": health_status,
            "issues": issues,
            "recommendations": await self._get_memory_recommendations(health_status, issues)
        }

    async def _get_memory_recommendations(self, status: str, issues: List[str]) -> List[str]:
        """Получение рекомендаций по оптимизации памяти"""
        recommendations = []

        if status == "critical":
            recommendations.extend([
                "Immediate memory cleanup required",
                "Consider increasing memory limits",
                "Review memory-intensive operations"
            ])

        if status == "warning":
            recommendations.extend([
                "Schedule memory optimization",
                "Monitor memory usage trends",
                "Consider reducing cache sizes"
            ])

        for issue in issues:
            if "episodic" in issue:
                recommendations.append("Clean up old episodic memory entries")
            elif "working" in issue:
                recommendations.append("Clear expired working memory entries")
            elif "semantic" in issue:
                recommendations.append("Archive unused semantic memory")

        return recommendations

    # ===== АСИНХРОННЫЙ РАБОЧИЙ ЦИКЛ ОЧИСТКИ =====

    async def _cleanup_worker(self):
        """Асинхронный рабочий цикл автоматической очистки"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                # Проверка здоровья и оптимизация если нужно
                health = await self.check_memory_health()
                if health["status"] in ["warning", "critical"]:
                    logger.info("Running scheduled async memory optimization")
                    await self.optimize_memory()
                    self.memory_stats["cleanup_operations"] += 1

            except asyncio.CancelledError:
                logger.info("Async memory cleanup worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in async memory cleanup worker: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке

    # ===== СОВМЕСТИМОСТЬ И УТИЛИТЫ =====

    async def clear_memory(self, memory_type: str = "all"):
        """Асинхронная очистка памяти (для совместимости)"""
        if memory_type in ["episodic", "all"]:
            await self.episodic_pool.clear()

        if memory_type in ["working", "all"]:
            await self.working_pool.clear()

        if memory_type in ["semantic", "all"]:
            await self.semantic_pool.clear()

        logger.info(f"Memory cleared: {memory_type}")

    async def stream_memory_by_importance(self, min_importance: float = 0.5) -> AsyncIterator[Tuple[str, Any]]:
        """Асинхронный стриминг памяти по важности"""
        # Стримим эпизодическую память
        for entry_id in list(self.episodic_pool.entries.keys()):
            entry = self.episodic_pool.entries[entry_id]
            if entry.importance >= min_importance:
                data = await self.episodic_pool.get_entry(entry_id)
                if data:
                    yield ("episodic", entry_id, data)
        
        # Стримим рабочую память
        for entry_id in list(self.working_pool.entries.keys()):
            entry = self.working_pool.entries[entry_id]
            if entry.importance >= min_importance:
                data = await self.working_pool.get_entry(entry_id)
                if data:
                    yield ("working", entry_id, data)
                    
        # Стримим семантическую память
        for entry_id in list(self.semantic_pool.entries.keys()):
            entry = self.semantic_pool.entries[entry_id]
            if entry.importance >= min_importance:
                data = await self.semantic_pool.get_entry(entry_id)
                if data:
                    yield ("semantic", entry_id, data)

    def __str__(self) -> str:
        """Строковое представление"""
        # Для асинхронного метода создаем синхронную обертку
        import asyncio
        try:
            stats = asyncio.run(self.get_memory_stats())
            return (f"AsyncMemoryManager("
                    f"episodic: {stats['pools']['episodic']['utilization_percent']:.1f}%, "
                    f"working: {stats['pools']['working']['utilization_percent']:.1f}%, "
                    f"semantic: {stats['pools']['semantic']['utilization_percent']:.1f}%, "
                    f"system: {stats['system']['rss_mb']:.1f}MB)")
        except:
            return "AsyncMemoryManager(not initialized)"

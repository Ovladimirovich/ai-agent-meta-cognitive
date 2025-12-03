"""
Оптимизированный менеджер памяти с предотвращением утечек
Реализует интеллектуальное управление памятью для AI агента
"""

import logging
import gc
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, OrderedDict
import threading
import time
from weakref import WeakValueDictionary

from ..core.models import MemoryEntry

logger = logging.getLogger("MemoryManagerOptimized")


class MemoryEntryMetadata:
    """Метаданные для записи памяти"""

    def __init__(self, entry_id: str, size_bytes: int, importance: float = 0.5,
                 access_count: int = 0, last_access: datetime = None):
        self.entry_id = entry_id
        self.size_bytes = size_bytes
        self.importance = importance
        self.access_count = access_count
        self.last_access = last_access or datetime.now()
        self.created_at = datetime.now()

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


class MemoryPool:
    """Пул памяти с ограничением размера"""

    def __init__(self, max_size_mb: float = 100.0, cleanup_threshold: float = 0.8):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cleanup_threshold = cleanup_threshold  # Процент заполнения для запуска очистки
        self.current_size_bytes = 0
        self.entries: OrderedDict[str, Any] = OrderedDict()
        self.metadata: Dict[str, MemoryEntryMetadata] = {}
        self.lock = threading.RLock()

    def add_entry(self, entry_id: str, data: Any, importance: float = 0.5) -> bool:
        """
        Добавление записи в пул

        Args:
            entry_id: ID записи
            data: Данные записи
            importance: Важность (0.0-1.0)

        Returns:
            True если добавлено, False если нет места
        """
        with self.lock:
            # Оценка размера данных
            size_bytes = self._estimate_size(data)

            # Проверка доступного места
            if self.current_size_bytes + size_bytes > self.max_size_bytes:
                # Попытка очистки
                if not self._cleanup_space(size_bytes):
                    return False

            # Добавление записи
            self.entries[entry_id] = data
            self.metadata[entry_id] = MemoryEntryMetadata(
                entry_id=entry_id,
                size_bytes=size_bytes,
                importance=importance
            )
            self.current_size_bytes += size_bytes

            return True

    def get_entry(self, entry_id: str) -> Optional[Any]:
        """Получение записи"""
        with self.lock:
            if entry_id in self.entries:
                # Обновление статистики доступа
                if entry_id in self.metadata:
                    self.metadata[entry_id].update_access()
                return self.entries[entry_id]
            return None

    def remove_entry(self, entry_id: str) -> bool:
        """Удаление записи"""
        with self.lock:
            if entry_id in self.entries:
                size_bytes = self.metadata[entry_id].size_bytes
                del self.entries[entry_id]
                del self.metadata[entry_id]
                self.current_size_bytes -= size_bytes
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики пула"""
        with self.lock:
            return {
                "current_size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization_percent": (self.current_size_bytes / self.max_size_bytes) * 100,
                "entry_count": len(self.entries),
                "cleanup_threshold": self.cleanup_threshold * 100
            }

    def _estimate_size(self, data: Any) -> int:
        """Оценка размера данных в байтах"""
        try:
            # Простая оценка - можно улучшить для более точных расчетов
            if isinstance(data, (str, bytes)):
                return len(data.encode('utf-8') if isinstance(data, str) else data)
            elif isinstance(data, dict):
                return sum(self._estimate_size(v) for v in data.values())
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_size(item) for item in data)
            else:
                # Для объектов используем приблизительную оценку
                return 1024  # 1KB по умолчанию
        except:
            return 1024  # Fallback

    def _cleanup_space(self, required_bytes: int) -> bool:
        """
        Очистка пространства для новых данных

        Returns:
            True если удалось освободить достаточно места
        """
        with self.lock:
            # Сортировка записей по релевантности (от менее релевантных к более)
            entries_by_relevance = sorted(
                self.metadata.items(),
                key=lambda x: x[1].calculate_relevance_score()
            )

            freed_bytes = 0
            removed_count = 0

            for entry_id, metadata in entries_by_relevance:
                if freed_bytes >= required_bytes:
                    break

                # Не удаляем очень важные записи
                if metadata.importance >= 0.8:
                    continue

                # Удаление записи
                self.remove_entry(entry_id)
                freed_bytes += metadata.size_bytes
                removed_count += 1

            logger.info(f"Memory pool cleanup: freed {freed_bytes} bytes, removed {removed_count} entries")

            return freed_bytes >= required_bytes

    def cleanup_expired_entries(self, max_age_days: float = 30.0):
        """Очистка устаревших записей"""
        with self.lock:
            expired_ids = []
            for entry_id, metadata in self.metadata.items():
                if metadata.get_age_days() > max_age_days and metadata.importance < 0.7:
                    expired_ids.append(entry_id)

            for entry_id in expired_ids:
                self.remove_entry(entry_id)

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired memory entries")


class OptimizedMemoryManager:
    """
    Оптимизированный менеджер памяти с предотвращением утечек.

    Ключевые улучшения:
    - Интеллектуальное управление размером памяти
    - Автоматическая очистка устаревших данных
    - Мониторинг использования памяти
    - Защита от утечек памяти
    """

    def __init__(self,
                 max_episodic_entries: int = 1000,
                 max_working_memory_mb: float = 50.0,
                 max_semantic_memory_mb: float = 100.0,
                 cleanup_interval_seconds: int = 300):
        """
        Инициализация оптимизированного менеджера памяти

        Args:
            max_episodic_entries: Максимальное количество эпизодических записей
            max_working_memory_mb: Максимальный размер рабочей памяти (MB)
            max_semantic_memory_mb: Максимальный размер семантической памяти (MB)
            cleanup_interval_seconds: Интервал автоматической очистки (секунды)
        """
        # Пулы памяти с ограничениями
        self.episodic_pool = MemoryPool(max_size_mb=max_semantic_memory_mb)
        self.working_pool = MemoryPool(max_size_mb=max_working_memory_mb)
        self.semantic_pool = MemoryPool(max_size_mb=max_semantic_memory_mb)

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
        self._start_cleanup_thread()

        # Weak references для предотвращения циклических ссылок
        self._weak_references = WeakValueDictionary()

        logger.info("🚀 OptimizedMemoryManager initialized with memory limits")

    def __del__(self):
        """Очистка при уничтожении объекта"""
        try:
            self._stop_cleanup_thread()
        except:
            pass

    # ===== ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ =====

    async def store_episodic_memory(self, memory_data: Dict[str, Any], importance: float = 0.5):
        """
        Сохранение эпизодической памяти с оптимизацией

        Args:
            memory_data: Данные для сохранения
            importance: Важность записи (0.0-1.0)
        """
        try:
            self.memory_stats["total_operations"] += 1

            # Создание ID для записи
            entry_id = f"episodic_{int(time.time() * 1000000)}_{hash(str(memory_data)) % 10000}"

            # Попытка добавления в пул
            if self.episodic_pool.add_entry(entry_id, memory_data, importance):
                logger.debug(f"Stored episodic memory: {entry_id}")
            else:
                logger.warning(f"Failed to store episodic memory: insufficient space")

        except Exception as e:
            logger.error(f"❌ Failed to store episodic memory: {e}")

    def retrieve_episodic_memory(self, limit: int = 10, min_importance: float = 0.0) -> List[Any]:
        """
        Получение эпизодической памяти с фильтрацией

        Args:
            limit: Максимальное количество записей
            min_importance: Минимальная важность

        Returns:
            Список записей памяти
        """
        # Получение записей, отсортированных по релевантности
        relevant_entries = []

        for entry_id, metadata in self.episodic_pool.metadata.items():
            if metadata.importance >= min_importance:
                data = self.episodic_pool.get_entry(entry_id)
                if data:
                    relevant_entries.append((data, metadata.calculate_relevance_score()))

        # Сортировка по релевантности и возврат топ-N
        relevant_entries.sort(key=lambda x: x[1], reverse=True)
        return [data for data, score in relevant_entries[:limit]]

    # ===== РАБОЧАЯ ПАМЯТЬ =====

    def store_working_memory(self, key: str, value: Any, ttl_seconds: Optional[int] = None, importance: float = 0.7):
        """
        Сохранение в рабочей памяти с TTL

        Args:
            key: Ключ
            value: Значение
            ttl_seconds: Время жизни (None - бессрочно)
            importance: Важность
        """
        try:
            self.memory_stats["total_operations"] += 1

            # Добавление TTL в данные
            enriched_value = {
                "data": value,
                "expires_at": (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds else None,
                "stored_at": datetime.now().isoformat()
            }

            if self.working_pool.add_entry(key, enriched_value, importance):
                logger.debug(f"Stored working memory: {key}")
            else:
                logger.warning(f"Failed to store working memory: {key} - insufficient space")

        except Exception as e:
            logger.error(f"❌ Failed to store working memory: {e}")

    def retrieve_working_memory(self, key: str) -> Optional[Any]:
        """
        Получение из рабочей памяти с проверкой TTL

        Args:
            key: Ключ

        Returns:
            Значение или None
        """
        try:
            enriched_value = self.working_pool.get_entry(key)
            if not enriched_value:
                return None

            # Проверка TTL
            if enriched_value.get("expires_at"):
                expires_at = datetime.fromisoformat(enriched_value["expires_at"])
                if datetime.now() > expires_at:
                    # Удаление истекшей записи
                    self.working_pool.remove_entry(key)
                    logger.debug(f"Working memory entry expired: {key}")
                    return None

            return enriched_value["data"]

        except Exception as e:
            logger.error(f"❌ Failed to retrieve working memory: {e}")
            return None

    # ===== СЕМАНТИЧЕСКАЯ ПАМЯТЬ =====

    def store_semantic_memory(self, key: str, value: Any, importance: float = 0.8):
        """
        Сохранение семантической памяти

        Args:
            key: Ключ
            value: Значение
            importance: Важность (высокая по умолчанию для семантической памяти)
        """
        try:
            self.memory_stats["total_operations"] += 1

            if self.semantic_pool.add_entry(key, value, importance):
                logger.debug(f"Stored semantic memory: {key}")
            else:
                logger.warning(f"Failed to store semantic memory: {key} - insufficient space")

        except Exception as e:
            logger.error(f"❌ Failed to store semantic memory: {e}")

    def retrieve_semantic_memory(self, key: str) -> Optional[Any]:
        """Получение семантической памяти"""
        return self.semantic_pool.get_entry(key)

    # ===== МОНИТОРИНГ И ОПТИМИЗАЦИЯ =====

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Получение подробной статистики использования памяти

        Returns:
            Детальная статистика
        """
        # Сбор статистики от всех пулов
        episodic_stats = self.episodic_pool.get_stats()
        working_stats = self.working_pool.get_stats()
        semantic_stats = self.semantic_pool.get_stats()

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

    def optimize_memory(self) -> Dict[str, Any]:
        """
        Оптимизация использования памяти

        Returns:
            Результаты оптимизации
        """
        logger.info("🔄 Starting memory optimization")

        results = {
            "freed_bytes": 0,
            "removed_entries": 0,
            "pools_optimized": 0,
            "gc_collections": 0
        }

        # Очистка устаревших записей
        self.episodic_pool.cleanup_expired_entries(max_age_days=7.0)  # Неделя
        self.working_pool.cleanup_expired_entries(max_age_days=1.0)   # День
        self.semantic_pool.cleanup_expired_entries(max_age_days=30.0) # Месяц

        # Принудительный сбор мусора
        collected = gc.collect()
        results["gc_collections"] = collected

        # Расчет освобожденного места (приблизительно)
        # В реальности нужно отслеживать до/после оптимизации

        self.memory_stats["cleanup_operations"] += 1
        self.memory_stats["last_cleanup"] = datetime.now()

        logger.info(f"✅ Memory optimization completed: {results}")
        return results

    def check_memory_health(self) -> Dict[str, Any]:
        """
        Проверка здоровья памяти

        Returns:
            Статус здоровья
        """
        stats = self.get_memory_stats()

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
            "recommendations": self._get_memory_recommendations(health_status, issues)
        }

    def _get_memory_recommendations(self, status: str, issues: List[str]) -> List[str]:
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

    # ===== АВТОМАТИЧЕСКАЯ ОЧИСТКА =====

    def _start_cleanup_thread(self):
        """Запуск потока автоматической очистки"""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="MemoryCleanup"
        )
        self._cleanup_thread.start()

    def _stop_cleanup_thread(self):
        """Остановка потока очистки"""
        if hasattr(self, '_cleanup_thread'):
            self._cleanup_thread = None

    def _cleanup_worker(self):
        """Рабочий поток автоматической очистки"""
        while True:
            try:
                time.sleep(self.cleanup_interval)

                # Проверка здоровья и оптимизация если нужно
                health = self.check_memory_health()
                if health["status"] in ["warning", "critical"]:
                    logger.info("Running scheduled memory optimization")
                    self.optimize_memory()
                    self.memory_stats["cleanup_operations"] += 1

            except Exception as e:
                logger.error(f"Error in memory cleanup worker: {e}")
                time.sleep(60)  # Пауза при ошибке

    # ===== СОВМЕСТИМОСТЬ =====

    def clear_memory(self, memory_type: str = "all"):
        """Очистка памяти (для совместимости)"""
        if memory_type in ["episodic", "all"]:
            # Очистка episodic пула
            entry_ids = list(self.episodic_pool.entries.keys())
            for entry_id in entry_ids:
                self.episodic_pool.remove_entry(entry_id)

        if memory_type in ["working", "all"]:
            # Очистка working пула
            entry_ids = list(self.working_pool.entries.keys())
            for entry_id in entry_ids:
                self.working_pool.remove_entry(entry_id)

        if memory_type in ["semantic", "all"]:
            # Очистка semantic пула
            entry_ids = list(self.semantic_pool.entries.keys())
            for entry_id in entry_ids:
                self.semantic_pool.remove_entry(entry_id)

        logger.info(f"Memory cleared: {memory_type}")

    # ===== УТИЛИТЫ =====

    def __str__(self) -> str:
        """Строковое представление"""
        stats = self.get_memory_stats()
        return (f"OptimizedMemoryManager("
                f"episodic: {stats['pools']['episodic']['utilization_percent']:.1f}%, "
                f"working: {stats['pools']['working']['utilization_percent']:.1f}%, "
                f"semantic: {stats['pools']['semantic']['utilization_percent']:.1f}%, "
                f"system: {stats['system']['rss_mb']:.1f}MB)")

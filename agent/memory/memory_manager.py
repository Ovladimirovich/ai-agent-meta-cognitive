import logging
import gc
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import threading
import time
from weakref import WeakValueDictionary

from ..core.models import MemoryEntry

logger = logging.getLogger("MemoryManager")


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
        return (datetime.now() - self.last_access).total_seconds() / (24 * 360)

    def calculate_relevance_score(self) -> float:
        """Расчет релевантности записи"""
        # Формула: importance * (1 / (1 + age)) * (1 / (1 + inactivity)) * access_bonus
        age_penalty = 1 / (1 + self.get_age_days())
        inactivity_penalty = 1 / (1 + self.get_inactivity_days())
        access_bonus = min(2.0, 1 + (self.access_count * 0.1))

        return self.importance * age_penalty * inactivity_penalty * access_bonus


class MemoryPool:
    """Пул памяти с ограничением размера и асинхронной обработкой"""

    def __init__(self, max_size_mb: float = 100.0, cleanup_threshold: float = 0.8):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cleanup_threshold = cleanup_threshold  # Процент заполнения для запуска очистки
        self.current_size_bytes = 0
        self.entries: OrderedDict[str, Any] = OrderedDict()
        self.metadata: Dict[str, MemoryEntryMetadata] = {}
        self.lock = threading.RLock()
        # Добавляем множество для отслеживания устаревших записей
        self.expired_entries: Set[str] = set()
        # Добавляем время последней очистки
        self.last_cleanup_time = datetime.now()

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
                    # Убираем из устаревших, если был там
                    self.expired_entries.discard(entry_id)
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
                # Убираем из устаревших, если был там
                self.expired_entries.discard(entry_id)
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
                "cleanup_threshold": self.cleanup_threshold * 10,
                "expired_entries_count": len(self.expired_entries)
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
                return 1024 # 1KB по умолчанию
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

    def cleanup_expired_entries(self, max_age_days: float = 30.0, force_cleanup: bool = False):
        """Очистка устаревших записей с оптимизацией для предотвращения утечек памяти"""
        with self.lock:
            current_time = datetime.now()
            # Проверяем, не прошло ли достаточно времени с последней очистки
            if not force_cleanup and (current_time - self.last_cleanup_time).total_seconds() < 300:  # 5 минут
                return

            expired_ids = []
            for entry_id, metadata in self.metadata.items():
                if metadata.get_age_days() > max_age_days and metadata.importance < 0.7:
                    expired_ids.append(entry_id)
                    # Добавляем в множество устаревших
                    self.expired_entries.add(entry_id)

            # Удаляем устаревшие записи
            for entry_id in expired_ids:
                self.remove_entry(entry_id)

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired memory entries")
                self.last_cleanup_time = current_time

    def cleanup_inactive_entries(self, max_inactivity_days: float = 7.0, min_importance: float = 0.3):
        """Очистка неактивных записей для предотвращения утечек памяти"""
        with self.lock:
            inactive_ids = []
            for entry_id, metadata in self.metadata.items():
                if metadata.get_inactivity_days() > max_inactivity_days and metadata.importance < min_importance:
                    inactive_ids.append(entry_id)

            for entry_id in inactive_ids:
                self.remove_entry(entry_id)

            if inactive_ids:
                logger.info(f"Cleaned up {len(inactive_ids)} inactive memory entries")

    def cleanup_memory_pressure(self):
        """Очистка при высоком давлении на память"""
        with self.lock:
            utilization = self.current_size_bytes / self.max_size_bytes
            if utilization > 0.9:  # Если память заполнена более чем на 90%
                # Сортируем по наименьшей важности
                entries_by_importance = sorted(
                    self.metadata.items(),
                    key=lambda x: x[1].importance
                )

                freed_bytes = 0
                removed_count = 0

                for entry_id, metadata in entries_by_importance:
                    if self.current_size_bytes / self.max_size_bytes <= 0.7:  # Цель - снизить до 70%
                        break

                    # Удаляем только записи с низкой важностью
                    if metadata.importance < 0.5:
                        size_to_free = metadata.size_bytes
                        self.remove_entry(entry_id)
                        freed_bytes += size_to_free
                        removed_count += 1

                if removed_count > 0:
                    logger.info(f"Memory pressure cleanup: freed {freed_bytes} bytes, removed {removed_count} entries")


class MemoryManager:
    """
    Оптимизированный менеджер памяти с предотвращением утечек и асинхронной обработкой.

    Ключевые улучшения:
    - Интеллектуальное управление размером памяти
    - Автоматическая очистка устаревших данных
    - Мониторинг использования памяти
    - Защита от утечек памяти
    - Асинхронная обработка для улучшения производительности
    """

    def __init__(self, max_entries: int = 1000, max_working_memory_mb: float = 50.0, max_semantic_memory_mb: float = 100.0):
        # Пулы памяти с ограничениями
        self.episodic_pool = MemoryPool(max_size_mb=max_semantic_memory_mb)
        self.working_pool = MemoryPool(max_size_mb=max_working_memory_mb)
        self.semantic_pool = MemoryPool(max_size_mb=max_semantic_memory_mb)

        # Ограничения для обратной совместимости
        self.max_entries = max_entries

        # Статистика и мониторинг
        self.memory_stats = {
            "total_operations": 0,
            "cleanup_operations": 0,
            "memory_warnings": 0,
            "last_cleanup": datetime.now(),
            "start_time": datetime.now()
        }

        # Weak references для предотвращения циклических ссылок
        self._weak_references = WeakValueDictionary()

        # Добавляем таймер для периодической очистки
        self._cleanup_timer = None

        logger.info(f"🚀 MemoryManager initialized with memory limits: episodic={max_semantic_memory_mb}MB, working={max_working_memory_mb}MB, semantic={max_semantic_memory_mb}MB")

    async def store_episodic_memory(self, memory_data: Dict[str, Any]):
        """
        Сохранение эпизодической памяти с оптимизацией

        Args:
            memory_data: Данные для сохранения
        """
        try:
            self.memory_stats["total_operations"] += 1

            # Создание ID для записи
            entry_id = f"episodic_{int(time.time() * 1000000)}_{hash(str(memory_data)) % 10000}"

            # Определение важности на основе данных
            confidence = memory_data.get("confidence", 0.0)
            importance = min(1.0, max(0.1, confidence))  # Важность от 0.1 до 1.0 в зависимости от уверенности

            # Попытка добавления в пул
            if self.episodic_pool.add_entry(entry_id, memory_data, importance):
                logger.debug(f"Stored episodic memory: {entry_id}, importance={importance:.2f}")
            else:
                logger.warning(f"Failed to store episodic memory: insufficient space")

        except Exception as e:
            logger.error(f"❌ Failed to store episodic memory: {e}")

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

    def retrieve_episodic_memory(self, limit: int = 10, min_importance: float = 0.0) -> List[MemoryEntry]:
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

    def find_similar_episodes(self, current_query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        Поиск похожих эпизодов в памяти с оптимизацией.

        Args:
            current_query: Текущий запрос для сравнения
            limit: Максимальное количество похожих эпизодов

        Returns:
            Список похожих эпизодов
        """
        if not self.episodic_pool.entries:
            return []

        similar_episodes = []
        current_words = set(current_query.lower().split())

        for entry_id, metadata in self.episodic_pool.metadata.items():
            entry = self.episodic_pool.get_entry(entry_id)
            if not entry or not entry.get("request") or not entry["request"].get("query"):
                continue

            # Простое сравнение по пересечению слов
            entry_words = set(entry["request"]["query"].lower().split())
            intersection = current_words.intersection(entry_words)

            if intersection:
                similarity_score = len(intersection) / len(current_words.union(entry_words))

                if similarity_score > 0.1:  # Минимальный порог схожести
                    similar_episodes.append((entry, similarity_score, metadata.calculate_relevance_score()))

        # Сортировка по релевантности и схожести и возврат топ-N
        similar_episodes.sort(key=lambda x: x[2], reverse=True)  # Сортировка по релевантности
        return [entry for entry, similarity, relevance in similar_episodes[:limit]]

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

    def _cleanup_expired_working_memory(self):
        """Очистка истекших записей рабочей памяти"""
        # Очистка устаревших записей в пуле рабочей памяти
        self.working_pool.cleanup_expired_entries(max_age_days=1.0)  # 1 день

    def _cleanup_old_entries(self):
        """Очистка старых записей эпизодической памяти"""
        # Очистка устаревших записей в пуле эпизодической памяти
        self.episodic_pool.cleanup_expired_entries(max_age_days=7.0)  # 7 дней

    def _cleanup_inactive_entries(self):
        """Очистка неактивных записей для предотвращения утечек памяти"""
        # Очистка неактивных записей во всех пулах
        self.episodic_pool.cleanup_inactive_entries(max_inactivity_days=14.0, min_importance=0.2)
        self.working_pool.cleanup_inactive_entries(max_inactivity_days=1.0, min_importance=0.3)
        self.semantic_pool.cleanup_inactive_entries(max_inactivity_days=30.0, min_importance=0.4)

    def _cleanup_memory_pressure(self):
        """Очистка при высоком давлении на память"""
        # Проверка давления на память во всех пулах
        self.episodic_pool.cleanup_memory_pressure()
        self.working_pool.cleanup_memory_pressure()
        self.semantic_pool.cleanup_memory_pressure()

    def perform_memory_cleanup(self, force_cleanup: bool = False):
        """Выполнение комплексной очистки памяти для предотвращения утечек"""
        try:
            # Очистка устаревших записей
            self.episodic_pool.cleanup_expired_entries(max_age_days=7.0, force_cleanup=force_cleanup)
            self.working_pool.cleanup_expired_entries(max_age_days=1.0, force_cleanup=force_cleanup)
            self.semantic_pool.cleanup_expired_entries(max_age_days=30.0, force_cleanup=force_cleanup)

            # Очистка неактивных записей
            self._cleanup_inactive_entries()

            # Очистка при высоком давлении на память
            self._cleanup_memory_pressure()

            # Вызов сборщика мусора
            gc.collect()

            # Обновление статистики
            self.memory_stats["cleanup_operations"] += 1
            self.memory_stats["last_cleanup"] = datetime.now()

            logger.info("Memory cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")

    def clear_memory(self, memory_type: str = "all"):
        """
        Очистка памяти (для совместимости)

        Args:
            memory_type: Тип памяти для очистки (working, episodic, semantic, all)
        """
        if memory_type in ["working", "all"]:
            # Очистка working пула
            entry_ids = list(self.working_pool.entries.keys())
            for entry_id in entry_ids:
                self.working_pool.remove_entry(entry_id)
            logger.info("Working memory cleared")

        if memory_type in ["episodic", "all"]:
            # Очистка episodic пула
            entry_ids = list(self.episodic_pool.entries.keys())
            for entry_id in entry_ids:
                self.episodic_pool.remove_entry(entry_id)
            logger.info("Episodic memory cleared")

        if memory_type in ["semantic", "all"]:
            # Очистка semantic пула
            entry_ids = list(self.semantic_pool.entries.keys())
            for entry_id in entry_ids:
                self.semantic_pool.remove_entry(entry_id)
            logger.info("Semantic memory cleared")

    def export_memory(self, format: str = "dict") -> Any:
        """
        Экспорт памяти в различных форматах.

        Args:
            format: Формат экспорта (dict, json)

        Returns:
            Данные памяти в указанном формате
        """
        memory_data = {
            "episodic_memory": [self.episodic_pool.get_entry(entry_id) for entry_id in self.episodic_pool.entries.keys()],
            "working_memory": {key: self.working_pool.get_entry(key) for key in self.working_pool.entries.keys()},
            "semantic_memory": {key: self.semantic_pool.get_entry(key) for key in self.semantic_pool.entries.keys()},
            "stats": self.get_memory_stats(),
            "export_timestamp": datetime.now().isoformat()
        }

        if format == "json":
            # В будущем можно добавить JSON сериализацию
            return memory_data
        else:
            return memory_data

    def import_memory(self, memory_data: Dict[str, Any]):
        """
        Импорт памяти из данных (асинхронная версия).

        Args:
            memory_data: Данные для импорта
        """
        try:
            # Импорт эпизодической памяти
            if "episodic_memory" in memory_data:
                for i, entry_data in enumerate(memory_data["episodic_memory"]):
                    entry_id = f"episodic_imported_{i}"
                    importance = entry_data.get("confidence", 0.5)
                    self.episodic_pool.add_entry(entry_id, entry_data, importance)

            # Импорт рабочей памяти
            if "working_memory" in memory_data:
                for key, value in memory_data["working_memory"].items():
                    importance = 0.7  # По умолчанию
                    if isinstance(value, dict) and "data" in value:
                        importance = value.get("importance", 0.7)
                    self.working_pool.add_entry(key, value, importance)

            # Импорт семантической памяти
            if "semantic_memory" in memory_data:
                for key, value in memory_data["semantic_memory"].items():
                    importance = 0.8  # По умолчанию для семантической памяти
                    self.semantic_pool.add_entry(key, value, importance)

            logger.info("Memory imported successfully")

        except Exception as e:
            logger.error(f"❌ Failed to import memory: {e}")

    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Получение инсайтов для обучения на основе памяти с оптимизацией.

        Returns:
            Инсайты для улучшения агента
        """
        insights = {
            "total_experiences": len(self.episodic_pool.entries),
            "average_confidence": 0.0,
            "common_intents": {},
            "strategy_effectiveness": {},
            "performance_trends": []
        }

        if not self.episodic_pool.entries:
            return insights

        # Расчет средней уверенности и анализ данных
        total_confidence = 0.0
        confidence_count = 0

        intent_counts = {}
        strategy_stats = {}

        for entry_id, metadata in self.episodic_pool.metadata.items():
            entry = self.episodic_pool.get_entry(entry_id)
            if not entry:
                continue

            # Обработка уверенности
            confidence = entry.get("confidence", 0.0)
            if confidence > 0:
                total_confidence += confidence
                confidence_count += 1

            # Анализ намерений
            if entry.get("analysis") and entry["analysis"].get("intent"):
                intent = entry["analysis"]["intent"]
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

            # Анализ стратегий
            strategy = entry.get("strategy", "unknown")
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"count": 0, "total_confidence": 0.0}

            strategy_stats[strategy]["count"] += 1
            strategy_stats[strategy]["total_confidence"] += confidence

        # Вычисление средней уверенности
        if confidence_count > 0:
            insights["average_confidence"] = total_confidence / confidence_count

        insights["common_intents"] = intent_counts

        # Расчет средней уверенности по стратегиям
        for strategy, stats in strategy_stats.items():
            stats["average_confidence"] = stats["total_confidence"] / stats["count"]

        insights["strategy_effectiveness"] = strategy_stats

        return insights

    def retrieve_semantic_memory(self, query: str, limit: int = 5) -> List[Any]:
        """
        Получение семантической памяти по запросу
        
        Args:
            query: Запрос для поиска семантической памяти
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных записей семантической памяти
        """
        try:
            # Поиск записей в семантическом пуле по запросу
            results = []
            query_lower = query.lower()
            
            for entry_id, metadata in self.semantic_pool.metadata.items():
                entry_data = self.semantic_pool.get_entry(entry_id)
                if entry_data:
                    # Простой поиск по содержимому записи
                    entry_str = str(entry_data).lower()
                    if query_lower in entry_str or query_lower.replace(" ", "") in entry_str.replace(" ", ""):
                        results.append((entry_data, metadata.calculate_relevance_score()))
            
            # Сортировка по релевантности и возврат ограниченного количества результатов
            results.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in results[:limit]]
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve semantic memory: {e}")
            return []

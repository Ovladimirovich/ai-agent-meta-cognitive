#!/usr/bin/env python3
"""
Система мониторинга производительности кэша
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from .cache_system_enhanced import LRUCache

logger = logging.getLogger(__name__)


@dataclass
class CacheAlert:
    """Алерт кэша"""
    alert_type: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    cache_name: str
    timestamp: datetime
    metrics: Dict[str, Any]


@dataclass
class CacheThreshold:
    """Порог для алертов"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str  # 'gt', 'lt', 'eq', 'ne'


class CacheMonitor:
    """
    Система мониторинга производительности кэша

    Предоставляет:
    - Реал-тайм мониторинг метрик
    - Автоматические алерты
    - Историю производительности
    - Отчеты о здоровье
    """

    def __init__(self, check_interval: float = 60.0):
        self.caches: Dict[str, LRUCache] = {}
        self.alerts: List[CacheAlert] = []
        self.thresholds: Dict[str, List[CacheThreshold]] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.check_interval = check_interval
        self.monitoring_active = False
        self.alert_callbacks: List[Callable[[CacheAlert], None]] = []

        # Настройки по умолчанию
        self._setup_default_thresholds()

    def register_cache(self, cache: LRUCache) -> None:
        """
        Регистрация кэша для мониторинга

        Args:
            cache: Экземпляр LRUCache для мониторинга
        """
        self.caches[cache.name] = cache
        self.performance_history[cache.name] = []
        self.thresholds[cache.name] = self._get_default_thresholds()

        logger.info(f"Registered cache '{cache.name}' for monitoring")

    def unregister_cache(self, cache_name: str) -> None:
        """
        Отмена регистрации кэша

        Args:
            cache_name: Имя кэша
        """
        if cache_name in self.caches:
            del self.caches[cache_name]
            del self.performance_history[cache_name]
            del self.thresholds[cache_name]
            logger.info(f"Unregistered cache '{cache_name}' from monitoring")

    def set_threshold(self, cache_name: str, threshold: CacheThreshold) -> None:
        """
        Установка порога для алертов

        Args:
            cache_name: Имя кэша
            threshold: Порог алерта
        """
        if cache_name not in self.thresholds:
            self.thresholds[cache_name] = []

        # Удаляем существующий порог для этой метрики
        self.thresholds[cache_name] = [
            t for t in self.thresholds[cache_name]
            if t.metric_name != threshold.metric_name
        ]

        self.thresholds[cache_name].append(threshold)
        logger.info(f"Set threshold for {cache_name}.{threshold.metric_name}")

    def add_alert_callback(self, callback: Callable[[CacheAlert], None]) -> None:
        """
        Добавление callback для алертов

        Args:
            callback: Функция обратного вызова
        """
        self.alert_callbacks.append(callback)

    async def start_monitoring(self) -> None:
        """Запуск мониторинга"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        logger.info("Started cache monitoring")

        try:
            while self.monitoring_active:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            self.monitoring_active = False

    def stop_monitoring(self) -> None:
        """Остановка мониторинга"""
        self.monitoring_active = False
        logger.info("Stopped cache monitoring")

    async def _perform_health_check(self) -> None:
        """Выполнение проверки здоровья"""
        for cache_name, cache in self.caches.items():
            try:
                # Получаем метрики здоровья
                health = cache.get_cache_health()
                stats = cache.get_stats()

                # Сохраняем в историю
                self._record_performance(cache_name, health, stats)

                # Проверяем пороги
                self._check_thresholds(cache_name, health, stats)

                # Выполняем обслуживание
                await self._perform_maintenance(cache_name, cache, health)

            except Exception as e:
                logger.error(f"Health check failed for cache '{cache_name}': {e}")
                self._create_alert(
                    'health_check_failed',
                    'error',
                    f"Health check failed: {e}",
                    cache_name
                )

    def _record_performance(self, cache_name: str, health: Dict[str, Any], stats: Dict[str, Any]) -> None:
        """Запись производительности в историю"""
        record = {
            'timestamp': datetime.now(),
            'health': health,
            'stats': stats
        }

        history = self.performance_history[cache_name]
        history.append(record)

        # Ограничиваем историю последними 1000 записями
        if len(history) > 1000:
            history.pop(0)

    def _check_thresholds(self, cache_name: str, health: Dict[str, Any], stats: Dict[str, Any]) -> None:
        """Проверка порогов для алертов"""
        thresholds = self.thresholds.get(cache_name, [])

        for threshold in thresholds:
            value = self._get_metric_value(threshold.metric_name, health, stats)

            if value is None:
                continue

            alert_triggered = False
            severity = None

            if threshold.comparison == 'gt':
                if value > threshold.critical_threshold:
                    alert_triggered = True
                    severity = 'critical'
                elif value > threshold.warning_threshold:
                    alert_triggered = True
                    severity = 'warning'
            elif threshold.comparison == 'lt':
                if value < threshold.critical_threshold:
                    alert_triggered = True
                    severity = 'critical'
                elif value < threshold.warning_threshold:
                    alert_triggered = True
                    severity = 'warning'

            if alert_triggered:
                self._create_alert(
                    f'threshold_{threshold.metric_name}',
                    severity,
                    f"{threshold.metric_name} {threshold.comparison} threshold: {value:.2f}",
                    cache_name,
                    {'metric': threshold.metric_name, 'value': value, 'threshold': threshold}
                )

    def _get_metric_value(self, metric_name: str, health: Dict[str, Any], stats: Dict[str, Any]) -> Optional[float]:
        """Получение значения метрики"""
        # Сначала проверяем в health
        if metric_name in health:
            return float(health[metric_name])

        # Затем в stats
        if metric_name in stats:
            return float(stats[metric_name])

        return None

    async def _perform_maintenance(self, cache_name: str, cache: LRUCache, health: Dict[str, Any]) -> None:
        """Выполнение обслуживания кэша"""
        try:
            # Очистка истекших записей
            cache.cleanup_expired()

            # Автоматическая оптимизация при низком здоровье
            if health['health_score'] < 60:
                logger.info(f"Performing automatic optimization for cache '{cache_name}'")

                # Удаляем старые записи
                cache.invalidate_by_age(3600)  # Старше 1 часа

                # Удаляем редко используемые
                cache.invalidate_by_access_count(1)

                self._create_alert(
                    'auto_optimization',
                    'info',
                    f"Performed automatic optimization (health: {health['health_score']:.1f})",
                    cache_name
                )

        except Exception as e:
            logger.error(f"Maintenance failed for cache '{cache_name}': {e}")

    def _create_alert(self, alert_type: str, severity: str, message: str,
                     cache_name: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Создание алерта"""
        alert = CacheAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            cache_name=cache_name,
            timestamp=datetime.now(),
            metrics=metrics or {}
        )

        self.alerts.append(alert)

        # Ограничиваем историю алертов
        if len(self.alerts) > 1000:
            self.alerts.pop(0)

        # Вызываем callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.log(
            getattr(logging, severity.upper()) if hasattr(logging, severity.upper()) else logging.INFO,
            f"Cache alert [{severity}]: {cache_name} - {message}"
        )

    def _setup_default_thresholds(self) -> None:
        """Настройка порогов по умолчанию"""
        self.default_thresholds = [
            CacheThreshold('health_score', 70.0, 50.0, 'lt'),
            CacheThreshold('hit_rate', 0.5, 0.3, 'lt'),
            CacheThreshold('memory_efficiency_percent', 80.0, 95.0, 'lt'),
            CacheThreshold('error_rate', 0.01, 0.05, 'gt'),
            CacheThreshold('expired_entries_count', 10, 50, 'gt'),
        ]

    def _get_default_thresholds(self) -> List[CacheThreshold]:
        """Получение порогов по умолчанию"""
        return self.default_thresholds.copy()

    def get_cache_report(self, cache_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Получение отчета о кэше

        Args:
            cache_name: Имя кэша
            hours: Период в часах

        Returns:
            Dict[str, Any]: Отчет
        """
        if cache_name not in self.performance_history:
            return {}

        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = [
            record for record in self.performance_history[cache_name]
            if record['timestamp'] > cutoff_time
        ]

        if not history:
            return {}

        # Анализ трендов
        health_scores = [r['health']['health_score'] for r in history]
        hit_rates = [r['health']['hit_rate'] for r in history]
        memory_usage = [r['stats']['memory_usage_mb'] for r in history]

        return {
            'cache_name': cache_name,
            'period_hours': hours,
            'total_records': len(history),
            'current_health': history[-1]['health'] if history else {},
            'health_trend': {
                'min': min(health_scores) if health_scores else 0,
                'max': max(health_scores) if health_scores else 0,
                'avg': sum(health_scores) / len(health_scores) if health_scores else 0,
                'latest': health_scores[-1] if health_scores else 0,
            },
            'hit_rate_trend': {
                'min': min(hit_rates) if hit_rates else 0,
                'max': max(hit_rates) if hit_rates else 0,
                'avg': sum(hit_rates) / len(hit_rates) if hit_rates else 0,
                'latest': hit_rates[-1] if hit_rates else 0,
            },
            'memory_trend': {
                'min': min(memory_usage) if memory_usage else 0,
                'max': max(memory_usage) if memory_usage else 0,
                'avg': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'latest': memory_usage[-1] if memory_usage else 0,
            },
            'alerts_count': len([
                alert for alert in self.alerts
                if alert.cache_name == cache_name and alert.timestamp > cutoff_time
            ])
        }

    def get_recent_alerts(self, cache_name: Optional[str] = None, hours: int = 24) -> List[CacheAlert]:
        """
        Получение недавних алертов

        Args:
            cache_name: Имя кэша (опционально)
            hours: Период в часах

        Returns:
            List[CacheAlert]: Список алертов
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]

        if cache_name:
            alerts = [alert for alert in alerts if alert.cache_name == cache_name]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def export_metrics(self, cache_name: str, format: str = 'json') -> str:
        """
        Экспорт метрик кэша

        Args:
            cache_name: Имя кэша
            format: Формат экспорта ('json', 'prometheus')

        Returns:
            str: Экспортированные метрики
        """
        if cache_name not in self.caches:
            return ""

        cache = self.caches[cache_name]
        stats = cache.get_stats()
        health = cache.get_cache_health()

        if format == 'json':
            return json.dumps({
                'cache_name': cache_name,
                'timestamp': datetime.now().isoformat(),
                'stats': stats,
                'health': health
            }, indent=2, default=str)

        elif format == 'prometheus':
            lines = [
                f'# Cache {cache_name} metrics',
                f'cache_size{{cache="{cache_name}"}} {stats["size"]}',
                f'cache_memory_usage_mb{{cache="{cache_name}"}} {stats["memory_usage_mb"]:.2f}',
                f'cache_hit_rate{{cache="{cache_name}"}} {stats["hit_rate"]:.4f}',
                f'cache_health_score{{cache="{cache_name}"}} {health["health_score"]:.2f}',
            ]
            return '\n'.join(lines)

        return ""


# Глобальный монитор
_cache_monitor = None

def get_cache_monitor() -> CacheMonitor:
    """Получение глобального монитора кэша"""
    global _cache_monitor
    if _cache_monitor is None:
        _cache_monitor = CacheMonitor()
    return _cache_monitor

def start_cache_monitoring(check_interval: float = 60.0) -> None:
    """
    Запуск глобального мониторинга кэша

    Args:
        check_interval: Интервал проверки в секундах
    """
    monitor = get_cache_monitor()

    # Настройка логирования алертов
    def log_alert_handler(alert: CacheAlert):
        level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.INFO)

        logger.log(level, f"Cache Alert [{alert.severity.upper()}]: {alert.cache_name} - {alert.message}")

    monitor.add_alert_callback(log_alert_handler)

    # Запуск мониторинга в фоне
    asyncio.create_task(monitor.start_monitoring())

def register_cache_for_monitoring(cache: LRUCache) -> None:
    """
    Регистрация кэша для глобального мониторинга

    Args:
        cache: Кэш для регистрации
    """
    monitor = get_cache_monitor()
    monitor.register_cache(cache)

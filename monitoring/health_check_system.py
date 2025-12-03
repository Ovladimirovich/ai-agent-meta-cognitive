"""
Система health checks для AI Агента
Расширенная система проверки здоровья с интеграцией метрик и алертинга
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import os
from .metrics_collector import metrics_collector


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Статусы health check"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Результат health check"""
    name: str
    status: HealthStatus
    response_time: float
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    critical: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "response_time": self.response_time,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "critical": self.critical
        }


class HealthChecker:
    """
    Базовый класс для health checks
    """

    def __init__(self, name: str, timeout: float = 5.0, critical: bool = True):
        self.name = name
        self.timeout = timeout
        self.critical = critical

    async def check(self) -> HealthCheckResult:
        """Выполнить health check"""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self._check_impl(),
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            # Обновляем время ответа в результате
            result.response_time = response_time
            return result
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                message=f"Health check timed out after {self.timeout}s",
                critical=self.critical
            )
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                message=f"Health check failed: {str(e)}",
                critical=self.critical
            )

    async def _check_impl(self) -> HealthCheckResult:
        """Реализация проверки (должна быть переопределена)"""
        raise NotImplementedError("Subclasses must implement _check_impl")


class SystemHealthChecker(HealthChecker):
    """Проверка системных ресурсов"""

    def __init__(self, cpu_threshold=None, memory_threshold=None, disk_threshold=None, network_threshold=None, **kwargs):
        # Извлекаем специфичные параметры для этого класса
        self.cpu_threshold = cpu_threshold or kwargs.pop('cpu_threshold', 90.0)
        self.memory_threshold = memory_threshold or kwargs.pop('memory_threshold', 90.0)
        self.disk_threshold = disk_threshold or kwargs.pop('disk_threshold', 90.0)
        self.network_threshold = network_threshold or kwargs.pop('network_threshold', 100)  # connections
        
        # Передаем оставшиеся параметры родительскому классу
        super().__init__("system", **kwargs)

    async def _check_impl(self) -> HealthCheckResult:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # Network connections
        net_connections = len(psutil.net_connections())

        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_percent": disk_percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "network_connections": net_connections,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }

        # Обновляем метрики системы
        if metrics_collector.enabled:
            metrics_collector.set_system_metrics(cpu_percent, memory_percent, disk_percent)

        # Определяем статус
        issues = []
        if cpu_percent > self.cpu_threshold:
            issues.append(f"CPU usage too high: {cpu_percent}%")
        if memory_percent > self.memory_threshold:
            issues.append(f"Memory usage too high: {memory_percent}%")
        if disk_percent > self.disk_threshold:
            issues.append(f"Disk usage too high: {disk_percent}%")
        if net_connections > self.network_threshold:
            issues.append(f"Too many network connections: {net_connections}")

        if issues:
            if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"System resources critically high: {', '.join(issues)}"
            else:
                status = HealthStatus.DEGRADED
                message = f"System resources high: {', '.join(issues)}"
        else:
            status = HealthStatus.HEALTHY
            message = "System resources are normal"

        result = HealthCheckResult(
            name=self.name,
            status=status,
            response_time=0.0,
            message=message,
            details=details,
            critical=True
        )

        # Обновляем метрику здоровья
        if metrics_collector.enabled:
            health_value = 1.0 if status == HealthStatus.HEALTHY else (0.5 if status == HealthStatus.DEGRADED else 0.0)
            metrics_collector.set_health_status(self.name, health_value)

        return result


class DatabaseHealthChecker(HealthChecker):
    """Проверка здоровья базы данных"""

    def __init__(self, db_manager, **kwargs):
        super().__init__("database", **kwargs)
        self.db_manager = db_manager

    async def _check_impl(self) -> HealthCheckResult:
        try:
            # Проверяем подключение
            connection_ok = await self.db_manager.health_check()

            # Получаем статистику
            stats = await self.db_manager.get_stats()

            if connection_ok:
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    message="Database connection is healthy",
                    details=stats
                )
            else:
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    message="Database connection failed",
                    details=stats
                )

            # Обновляем метрику здоровья
            if metrics_collector.enabled:
                health_value = 1.0 if result.status == HealthStatus.HEALTHY else (0.5 if result.status == HealthStatus.DEGRADED else 0.0)
                metrics_collector.set_health_status(self.name, health_value)

            return result

        except Exception as e:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=0.0,
                message=f"Database health check failed: {str(e)}",
                details={"error": str(e)}
            )
            
            # Обновляем метрику здоровья
            if metrics_collector.enabled:
                metrics_collector.set_health_status(self.name, 0.0)
                
            return result


class CacheHealthChecker(HealthChecker):
    """Проверка здоровья кэша"""

    def __init__(self, cache_manager, **kwargs):
        super().__init__("cache", **kwargs)
        self.cache_manager = cache_manager

    async def _check_impl(self) -> HealthCheckResult:
        try:
            # Проверяем подключение к кэшу
            cache_ok = await self.cache_manager.health_check()

            # Получаем статистику
            stats = await self.cache_manager.get_stats()

            if cache_ok:
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    message="Cache is healthy",
                    details=stats
                )
            else:
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    message="Cache connection failed",
                    details=stats
                )

            # Обновляем метрику здоровья
            if metrics_collector.enabled:
                health_value = 1.0 if result.status == HealthStatus.HEALTHY else (0.5 if result.status == HealthStatus.DEGRADED else 0.0)
                metrics_collector.set_health_status(self.name, health_value)

            return result

        except Exception as e:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=0.0,
                message=f"Cache health check failed: {str(e)}",
                details={"error": str(e)}
            )
            
            # Обновляем метрику здоровья
            if metrics_collector.enabled:
                metrics_collector.set_health_status(self.name, 0.0)
                
            return result


class AgentHealthChecker(HealthChecker):
    """Проверка здоровья AI агента"""

    def __init__(self, agent_core, **kwargs):
        super().__init__("agent", **kwargs)
        self.agent_core = agent_core

    async def _check_impl(self) -> HealthCheckResult:
        try:
            # Проверяем состояние агента
            agent_status = await self.agent_core.get_status()

            # Проверяем компоненты
            components_ok = await self.agent_core.check_components()

            details = {
                "agent_status": agent_status,
                "components_status": components_ok,
                "active_tasks": len(self.agent_core.active_tasks) if hasattr(self.agent_core, 'active_tasks') else 0,
                "memory_usage": self.agent_core.get_memory_usage() if hasattr(self.agent_core, 'get_memory_usage') else None
            }

            if agent_status == "healthy" and components_ok:
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    message="AI Agent is healthy",
                    details=details
                )
            elif agent_status == "degraded":
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    response_time=0.0,
                    message="AI Agent is degraded",
                    details=details
                )
            else:
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    message="AI Agent is unhealthy",
                    details=details
                )

            # Обновляем метрику здоровья
            if metrics_collector.enabled:
                health_value = 1.0 if result.status == HealthStatus.HEALTHY else (0.5 if result.status == HealthStatus.DEGRADED else 0.0)
                metrics_collector.set_health_status(self.name, health_value)

            return result

        except Exception as e:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=0.0,
                message=f"Agent health check failed: {str(e)}",
                details={"error": str(e)}
            )
            
            # Обновляем метрику здоровья
            if metrics_collector.enabled:
                metrics_collector.set_health_status(self.name, 0.0)
                
            return result


class ExternalServiceHealthChecker(HealthChecker):
    """Проверка здоровья внешних сервисов"""

    def __init__(self, service_name: str, check_func: Callable[[], Awaitable[bool]], **kwargs):
        super().__init__(f"external_{service_name}", **kwargs)
        self.service_name = service_name
        self.check_func = check_func

    async def _check_impl(self) -> HealthCheckResult:
        try:
            service_ok = await self.check_func()

            if service_ok:
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    message=f"External service {self.service_name} is healthy"
                )
            else:
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    message=f"External service {self.service_name} is unhealthy"
                )

            # Обновляем метрику здоровья
            if metrics_collector.enabled:
                health_value = 1.0 if result.status == HealthStatus.HEALTHY else (0.5 if result.status == HealthStatus.DEGRADED else 0.0)
                metrics_collector.set_health_status(self.name, health_value)

            return result

        except Exception as e:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=0.0,
                message=f"External service {self.service_name} check failed: {str(e)}",
                details={"error": str(e)}
            )
            
            # Обновляем метрику здоровья
            if metrics_collector.enabled:
                metrics_collector.set_health_status(self.name, 0.0)
                
            return result


class HealthCheckRegistry:
    """
    Реестр health checks с группировкой и агрегацией результатов
    """

    def __init__(self):
        self.checkers: Dict[str, HealthChecker] = {}
        self.groups: Dict[str, List[str]] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.history: List[Dict[str, Any]] = []
        self.max_history_size = 10
        self.alert_thresholds = {
            'unhealthy_count': 1, # Количество unhealthy чеков для алерта
            'degraded_count': 2,   # Количество degraded чеков для алерта
        }

    def register_checker(self, checker: HealthChecker, group: str = "default"):
        """Регистрация health checker"""
        self.checkers[checker.name] = checker

        if group not in self.groups:
            self.groups[group] = []
        self.groups[group].append(checker.name)

        logger.info(f"Registered health checker: {checker.name} in group: {group}")

    def unregister_checker(self, name: str):
        """Удаление health checker"""
        if name in self.checkers:
            del self.checkers[name]

            # Удаляем из групп
            for group, checkers in self.groups.items():
                if name in checkers:
                    checkers.remove(name)

            # Удаляем из результатов
            if name in self.last_results:
                del self.last_results[name]

            logger.info(f"Unregistered health checker: {name}")

    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Запуск конкретного health check"""
        if name not in self.checkers:
            return None

        result = await self.checkers[name].check()
        self.last_results[name] = result

        return result

    async def run_group(self, group: str) -> Dict[str, HealthCheckResult]:
        """Запуск всех проверок в группе"""
        if group not in self.groups:
            return {}

        results = {}
        tasks = []

        for name in self.groups[group]:
            if name in self.checkers:
                task = asyncio.create_task(self.run_check(name))
                tasks.append((name, task))

        for name, task in tasks:
            try:
                result = await task
                if result:
                    results[name] = result
            except Exception as e:
                logger.error(f"Failed to run health check {name}: {e}")

        return results

    async def run_all(self) -> Dict[str, HealthCheckResult]:
        """Запуск всех health checks"""
        results = {}
        tasks = []

        for name, checker in self.checkers.items():
            task = asyncio.create_task(self.run_check(name))
            tasks.append((name, task))

        for name, task in tasks:
            try:
                result = await task
                if result:
                    results[name] = result
            except Exception as e:
                logger.error(f"Failed to run health check {name}: {e}")

        # Сохраняем в историю
        self._add_to_history(results)

        # Проверяем пороги для алертинга
        await self._check_alerts(results)

        return results

    def get_overall_status(self, results: Optional[Dict[str, HealthCheckResult]] = None) -> HealthStatus:
        """Получение общего статуса системы"""
        if results is None:
            results = self.last_results

        if not results:
            return HealthStatus.UNHEALTHY

        unhealthy_count = 0
        degraded_count = 0
        total_count = len(results)

        for result in results.values():
            if result.status == HealthStatus.UNHEALTHY:
                unhealthy_count += 1
            elif result.status == HealthStatus.DEGRADED:
                degraded_count += 1

        # Критические проверки (если есть unhealthy критические проверки)
        critical_unhealthy = sum(
            1 for result in results.values()
            if result.status == HealthStatus.UNHEALTHY and result.critical
        )

        if critical_unhealthy > 0:
            return HealthStatus.UNHEALTHY
        elif unhealthy_count > 0 or degraded_count > total_count * 0.5:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_summary(self, results: Optional[Dict[str, HealthCheckResult]] = None) -> Dict[str, Any]:
        """Получение сводки по health checks"""
        if results is None:
            results = self.last_results

        summary = {
            "overall_status": self.get_overall_status(results).value,
            "total_checks": len(results),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "groups": {},
            "timestamp": datetime.now().isoformat()
        }

        for result in results.values():
            if result.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif result.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
            elif result.status == HealthStatus.UNHEALTHY:
                summary["unhealthy"] += 1

        # Группировка по группам
        for group, checkers in self.groups.items():
            group_results = {name: results.get(name) for name in checkers if name in results}
            if group_results:
                summary["groups"][group] = {
                    "status": self.get_overall_status(group_results).value,
                    "checks": len(group_results),
                    "results": {name: result.to_dict() for name, result in group_results.items()}
                }

        return summary

    def _add_to_history(self, results: Dict[str, HealthCheckResult]):
        """Добавление результатов в историю"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.get_overall_status(results).value,
            "results": {name: result.to_dict() for name, result in results.items()}
        }

        self.history.append(history_entry)

        # Ограничиваем размер истории
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение истории health checks"""
        return self.history[-limit:] if limit > 0 else self.history

    async def _check_alerts(self, results: Dict[str, HealthCheckResult]):
        """Проверка условий для алертинга"""
        unhealthy_count = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)

        # Проверяем пороги для алертинга
        if unhealthy_count >= self.alert_thresholds['unhealthy_count']:
            logger.critical(f"ALERT: {unhealthy_count} health checks are unhealthy")
            # Здесь можно интегрировать с системой алертинга
            await self._send_alert("health_unhealthy", f"{unhealthy_count} health checks are unhealthy")
        
        elif degraded_count >= self.alert_thresholds['degraded_count']:
            logger.warning(f"WARNING: {degraded_count} health checks are degraded")
            # Здесь можно интегрировать с системой алертинга
            await self._send_alert("health_degraded", f"{degraded_count} health checks are degraded")

    async def _send_alert(self, alert_type: str, message: str):
        """Отправка алерта"""
        # В реальной системе здесь будет интеграция с Alertmanager или другой системой алертинга
        logger.error(f"ALERT [{alert_type}]: {message}")

    async def start_periodic_health_checks(self, interval: int = 30):
        """Запуск периодических проверок здоровья"""
        logger.info(f"Starting periodic health checks every {interval} seconds")
        
        while True:
            try:
                await self.run_all()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in periodic health checks: {e}")
                await asyncio.sleep(interval)


# Глобальный реестр health checks
health_registry = HealthCheckRegistry()


# Вспомогательные функции для создания типичных health checks
def create_system_health_checker(**kwargs) -> SystemHealthChecker:
    """Создание проверки системных ресурсов"""
    return SystemHealthChecker(**kwargs)


def create_database_health_checker(db_manager, **kwargs) -> DatabaseHealthChecker:
    """Создание проверки базы данных"""
    return DatabaseHealthChecker(db_manager, **kwargs)


def create_cache_health_checker(cache_manager, **kwargs) -> CacheHealthChecker:
    """Создание проверки кэша"""
    return CacheHealthChecker(cache_manager, **kwargs)


def create_agent_health_checker(agent_core, **kwargs) -> AgentHealthChecker:
    """Создание проверки AI агента"""
    return AgentHealthChecker(agent_core, **kwargs)


def create_external_service_health_checker(service_name: str, check_func: Callable[[], Awaitable[bool]], **kwargs) -> ExternalServiceHealthChecker:
    """Создание проверки внешнего сервиса"""
    return ExternalServiceHealthChecker(service_name, check_func, **kwargs)


if __name__ == "__main__":
    # Пример использования
    import asyncio
    
    async def main():
        # Создаем реестр и добавляем базовые проверки
        registry = HealthCheckRegistry()
        
        # Пример добавления системной проверки
        system_checker = SystemHealthChecker(cpu_threshold=80, memory_threshold=80)
        registry.register_checker(system_checker, "infrastructure")
        
        # Запускаем проверки
        results = await registry.run_all()
        
        # Выводим результаты
        summary = registry.get_summary(results)
        print("Health Check Summary:")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Healthy: {summary['healthy']}, Degraded: {summary['degraded']}, Unhealthy: {summary['unhealthy']}")
        
        # Показываем результаты по группам
        for group_name, group_info in summary['groups'].items():
            print(f"\nGroup: {group_name}")
            print(f"Status: {group_info['status']}")
            for name, result in group_info['results'].items():
                print(f"  - {name}: {result['status']} ({result['message']})")
    
    # asyncio.run(main())
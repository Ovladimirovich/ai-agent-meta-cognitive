"""
Интеграция всех компонентов мониторинга AI Агента
Объединяет логирование, метрики, health checks и алертинг
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import psutil
import os

from monitoring.metrics_collector import metrics_collector, track_agent_cognitive_load, track_agent_memory_usage, track_tool_usage, track_learning_experience, set_learning_success_rate
from monitoring.centralized_logging import get_app_logger
from monitoring.health_check_system import health_registry
from monitoring.alerting_system import alert_manager, setup_default_alert_rules
from monitoring.metrics_endpoint import register_metrics_endpoint
from api.health_endpoints import register_health_endpoints, initialize_health_checks, initialize_health_checks as init_health_endpoints
from api.monitoring_middleware import setup_monitoring_middleware, update_system_health_metrics


logger = logging.getLogger(__name__)
app_logger = get_app_logger()


class MonitoringIntegration:
    """
    Класс интеграции всех компонентов мониторинга
    """
    
    def __init__(self):
        self.is_initialized = False
        self.agent_core = None
        self.health_check_task = None
        self.alert_evaluation_task = None
        self.system_metrics_task = None
        
    def initialize(
        self, 
        app=None, 
        agent_core=None, 
        enable_health_checks: bool = True,
        enable_alerts: bool = True,
        enable_metrics: bool = True,
        enable_logging: bool = True
    ):
        """
        Инициализация интеграции мониторинга
        
        Args:
            app: FastAPI приложение
            agent_core: Ядро AI агента
            enable_health_checks: Включить health checks
            enable_alerts: Включить алерты
            enable_metrics: Включить метрики
            enable_logging: Включить логирование
        """
        logger.info("Initializing monitoring integration...")
        
        self.agent_core = agent_core
        
        # Настройка логирования
        if enable_logging:
            logger.info("Setting up centralized logging...")
            app_logger.info("Centralized logging initialized")
        
        # Настройка метрик
        if enable_metrics:
            logger.info("Setting up metrics collection...")
            if app:
                register_metrics_endpoint(app)
        
        # Настройка health checks
        if enable_health_checks:
            logger.info("Setting up health checks...")
            initialize_health_checks(agent_core)
            if app:
                register_health_endpoints(app)
                init_health_endpoints(agent_core)
        
        # Настройка алертов
        if enable_alerts:
            logger.info("Setting up alerting system...")
            setup_default_alert_rules()
        
        # Настройка мидлвара мониторинга (только если приложение еще не запущено)
        # Middleware должны добавляться до запуска приложения, а не в рамках lifespan
        # if app and enable_metrics:
        #     setup_monitoring_middleware(app)
        logger.info("Monitoring middleware setup deferred to application startup")
        
        # Запуск фоновых задач
        self.start_background_tasks()
        
        self.is_initialized = True
        logger.info("Monitoring integration initialized successfully")
    
    def start_background_tasks(self):
        """Запуск фоновых задач мониторинга"""
        logger.info("Starting background monitoring tasks...")
        
        # Запуск периодических health checks
        self.health_check_task = asyncio.create_task(
            health_registry.start_periodic_health_checks(interval=30)
        )
        
        # Запуск оценки алертов
        self.alert_evaluation_task = asyncio.create_task(
            alert_manager.start_alert_evaluation()
        )
        
        # Запуск обновления системных метрик
        self.system_metrics_task = asyncio.create_task(
            self._update_system_metrics_loop()
        )
        
        logger.info("Background monitoring tasks started")
    
    async def _update_system_metrics_loop(self):
        """Цикл обновления системных метрик"""
        while True:
            try:
                update_system_health_metrics()
                await asyncio.sleep(10)  # Обновляем каждые 10 секунд
            except Exception as e:
                logger.error(f"Error in system metrics loop: {e}")
                await asyncio.sleep(10)
    
    def track_agent_request(self, request_type: str, duration: float, confidence: float = None):
        """Отслеживание запроса агента"""
        if metrics_collector.enabled:
            metrics_collector.record_agent_request(request_type, duration, confidence)
    
    def track_cognitive_load(self, component: str, load: float):
        """Отслеживание когнитивной нагрузки"""
        track_agent_cognitive_load(component, load)
    
    def track_memory_usage(self, memory_type: str, count: int):
        """Отслеживание использования памяти"""
        track_agent_memory_usage(memory_type, count)
    
    def track_tool_execution(self, tool_name: str, duration: float, success: bool = True):
        """Отслеживание выполнения инструмента"""
        track_tool_usage(tool_name, duration, success)
    
    def track_learning_event(self, success: bool = True):
        """Отслеживание события обучения"""
        track_learning_experience(success)
    
    def set_learning_rate(self, rate: float):
        """Установка процента успешного обучения"""
        set_learning_success_rate(rate)
    
    def log_agent_event(self, level: str, message: str, **context):
        """Логирование события агента"""
        getattr(app_logger, level.lower())(message, extra=context)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Получение статуса мониторинга"""
        return {
            "is_initialized": self.is_initialized,
            "metrics_enabled": metrics_collector.enabled,
            "health_checks_registered": len(health_registry.checkers) > 0,
            "active_alerts_count": len(alert_manager.get_active_alerts()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Завершение работы мониторинга"""
        logger.info("Shutting down monitoring integration...")
        
        # Отменяем фоновые задачи
        if self.health_check_task:
            self.health_check_task.cancel()
        
        if self.alert_evaluation_task:
            self.alert_evaluation_task.cancel()
        
        if self.system_metrics_task:
            self.system_metrics_task.cancel()
        
        logger.info("Monitoring integration shutdown completed")


# Глобальный экземпляр интеграции мониторинга
monitoring_integration = MonitoringIntegration()


def setup_monitoring(
    app=None, 
    agent_core=None, 
    enable_health_checks: bool = True,
    enable_alerts: bool = True,
    enable_metrics: bool = True,
    enable_logging: bool = True
):
    """
    Настройка полной системы мониторинга
    
    Args:
        app: FastAPI приложение
        agent_core: Ядро AI агента
        enable_health_checks: Включить health checks
        enable_alerts: Включить алерты
        enable_metrics: Включить метрики
        enable_logging: Включить логирование
    """
    monitoring_integration.initialize(
        app=app,
        agent_core=agent_core,
        enable_health_checks=enable_health_checks,
        enable_alerts=enable_alerts,
        enable_metrics=enable_metrics,
        enable_logging=enable_logging
    )
    
    logger.info("Complete monitoring system setup completed")


def get_monitoring_status() -> Dict[str, Any]:
    """Получение статуса мониторинга"""
    return monitoring_integration.get_monitoring_status()


# Функции для использования в различных частях приложения
def log_with_monitoring(level: str, message: str, **context):
    """
    Логирование с интеграцией мониторинга
    
    Args:
        level: Уровень логирования
        message: Сообщение
        **context: Контекстные данные
    """
    monitoring_integration.log_agent_event(level, message, **context)


def track_agent_performance(request_type: str, duration: float, confidence: float = None):
    """
    Отслеживание производительности агента
    
    Args:
        request_type: Тип запроса
        duration: Время выполнения
        confidence: Уверенность в ответе
    """
    monitoring_integration.track_agent_request(request_type, duration, confidence)


def update_cognitive_load(component: str, load: float):
    """
    Обновление уровня когнитивной нагрузки
    
    Args:
        component: Компонент агента
        load: Уровень нагрузки (0.0-1.0)
    """
    monitoring_integration.track_cognitive_load(component, load)


def update_memory_stats(memory_type: str, count: int):
    """
    Обновление статистики памяти
    
    Args:
        memory_type: Тип памяти
        count: Количество записей
    """
    monitoring_integration.track_memory_usage(memory_type, count)


def record_tool_execution(tool_name: str, duration: float, success: bool = True):
    """
    Запись выполнения инструмента
    
    Args:
        tool_name: Название инструмента
        duration: Время выполнения
        success: Успешность выполнения
    """
    monitoring_integration.track_tool_execution(tool_name, duration, success)


def record_learning_outcome(success: bool = True):
    """
    Запись результата обучения
    
    Args:
        success: Успешность обучения
    """
    monitoring_integration.track_learning_event(success)


def update_learning_success_rate(rate: float):
    """
    Обновление процента успешного обучения
    
    Args:
        rate: Процент успешного обучения (0.0-1.0)
    """
    monitoring_integration.set_learning_rate(rate)


# Функция для периодического обновления метрик системы
async def update_system_metrics():
    """
    Обновление системных метрик
    """
    if metrics_collector.enabled:
        # Получаем системные метрики
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # Обновляем метрики в collector'е
        metrics_collector.set_system_metrics(cpu_percent, memory_percent, disk_percent)
        
        # Обновляем метрику здоровья системы
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
            health_status = 0.0  # unhealthy
        elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 90:
            health_status = 0.5  # degraded
        else:
            health_status = 1.0  # healthy
        
        metrics_collector.set_health_status("system", health_status)


if __name__ == "__main__":
    # Пример использования
    import asyncio
    
    async def example_usage():
        print("Monitoring integration example...")
        
        # Имитация работы системы мониторинга
        print(f"Metrics collector enabled: {metrics_collector.enabled}")
        print(f"App logger available: {app_logger is not None}")
        
        # Пример логирования
        log_with_monitoring("info", "System started", component="main", version="1.0.0")
        
        # Пример отслеживания производительности
        track_agent_performance("basic", 0.5, 0.85)
        
        # Пример обновления когнитивной нагрузки
        update_cognitive_load("reasoning_engine", 0.6)
        
        # Пример обновления статистики памяти
        update_memory_stats("short_term", 42)
        
        # Пример выполнения инструмента
        record_tool_execution("web_search", 1.2, True)
        
        # Пример результата обучения
        record_learning_outcome(True)
        
        # Пример обновления процента успешного обучения
        update_learning_success_rate(0.87)
        
        print("Example completed successfully")
    
    # asyncio.run(example_usage())
"""
Модуль для сбора и экспорта метрик приложения AI Агента
Интеграция с Prometheus для мониторинга производительности
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from enum import Enum

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, 
        generate_latest, CONTENT_TYPE_LATEST,
        REGISTRY, CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Заглушки для случаев, когда prometheus недоступен
    class Counter:
        def __init__(self, name, documentation, labelnames=None):
            self.name = name
        def inc(self, amount=1, labels=None):
            pass
        def labels(self, **labels):
            return self

    class Histogram:
        def __init__(self, name, documentation, labelnames=None):
            self.name = name
        def observe(self, value):
            pass
        def labels(self, **labels):
            return self
        @contextmanager
        def time(self):
            start = time.time()
            try:
                yield
            finally:
                self.observe(time.time() - start)

    class Gauge:
        def __init__(self, name, documentation, labelnames=None):
            self.name = name
        def set(self, value):
            pass
        def inc(self, amount=1):
            pass
        def dec(self, amount=1):
            pass
        def labels(self, **labels):
            return self

    class Summary:
        def __init__(self, name, documentation, labelnames=None):
            self.name = name
        def observe(self, value):
            pass
        def labels(self, **labels):
            return self

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Сборщик метрик для AI Агента
    """
    
    def __init__(self):
        self.enabled = PROMETHEUS_AVAILABLE
        if not self.enabled:
            logger.warning("Prometheus not available, metrics collection disabled")
            return

        # Метрики HTTP запросов
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.http_active_requests = Gauge(
            'http_active_requests',
            'Number of active HTTP requests',
            ['endpoint']
        )
        
        # Метрики агента
        self.agent_requests_total = Counter(
            'agent_requests_total',
            'Total agent requests processed',
            ['type']  # 'basic', 'meta_cognitive'
        )
        
        self.agent_request_duration_seconds = Histogram(
            'agent_request_duration_seconds',
            'Agent request processing duration in seconds',
            ['type']
        )
        
        self.agent_confidence_score = Summary(
            'agent_confidence_score',
            'Agent confidence score',
            ['type']
        )
        
        # Метрики когнитивной нагрузки
        self.cognitive_load_current = Gauge(
            'cognitive_load_current',
            'Current cognitive load',
            ['component']
        )
        
        self.cognitive_load_threshold = Gauge(
            'cognitive_load_threshold',
            'Cognitive load threshold',
            ['component', 'threshold_type']  # 'warning', 'critical'
        )
        
        # Метрики памяти
        self.memory_entries_total = Gauge(
            'memory_entries_total',
            'Total memory entries',
            ['memory_type']  # 'short_term', 'long_term'
        )
        
        self.memory_access_total = Counter(
            'memory_access_total',
            'Total memory accesses',
            ['operation', 'success']  # 'read', 'write' / 'true', 'false'
        )
        
        # Метрики обучения
        self.learning_experiences_total = Counter(
            'learning_experiences_total',
            'Total learning experiences processed'
        )
        
        self.learning_success_rate = Gauge(
            'learning_success_rate',
            'Learning success rate'
        )
        
        # Метрики инструментов
        self.tool_calls_total = Counter(
            'tool_calls_total',
            'Total tool calls',
            ['tool_name', 'status']  # 'success', 'error'
        )
        
        self.tool_duration_seconds = Histogram(
            'tool_duration_seconds',
            'Tool call duration in seconds',
            ['tool_name']
        )
        
        # Метрики системы
        self.system_cpu_percent = Gauge(
            'system_cpu_percent',
            'CPU usage percentage'
        )
        
        self.system_memory_percent = Gauge(
            'system_memory_percent',
            'Memory usage percentage'
        )
        
        self.system_disk_percent = Gauge(
            'system_disk_percent',
            'Disk usage percentage'
        )
        
        # Метрики состояния
        self.health_status = Gauge(
            'health_status',
            'Health status (0=unhealthy, 0.5=degraded, 1=healthy)',
            ['check']
        )
        
        logger.info("Metrics collector initialized")
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Запись HTTP запроса"""
        if not self.enabled:
            return
            
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_agent_request(self, request_type: str, duration: float, confidence: float = None):
        """Запись запроса агента"""
        if not self.enabled:
            return
            
        self.agent_requests_total.labels(type=request_type).inc()
        
        self.agent_request_duration_seconds.labels(type=request_type).observe(duration)
        
        if confidence is not None:
            self.agent_confidence_score.labels(type=request_type).observe(confidence)
    
    def set_cognitive_load(self, component: str, load: float):
        """Установка уровня когнитивной нагрузки"""
        if not self.enabled:
            return
            
        self.cognitive_load_current.labels(component=component).set(load)
    
    def set_memory_entries(self, memory_type: str, count: int):
        """Установка количества записей в памяти"""
        if not self.enabled:
            return
            
        self.memory_entries_total.labels(memory_type=memory_type).set(count)
    
    def record_memory_access(self, operation: str, success: bool):
        """Запись обращения к памяти"""
        if not self.enabled:
            return
            
        self.memory_access_total.labels(
            operation=operation,
            success=str(success).lower()
        ).inc()
    
    def record_learning_experience(self, success: bool = True):
        """Запись опыта обучения"""
        if not self.enabled:
            return
            
        self.learning_experiences_total.inc()
    
    def set_learning_success_rate(self, rate: float):
        """Установка процента успешного обучения"""
        if not self.enabled:
            return
            
        self.learning_success_rate.set(rate)
    
    def record_tool_call(self, tool_name: str, duration: float, success: bool = True):
        """Запись вызова инструмента"""
        if not self.enabled:
            return
            
        status = 'success' if success else 'error'
        self.tool_calls_total.labels(
            tool_name=tool_name,
            status=status
        ).inc()
        
        self.tool_duration_seconds.labels(tool_name=tool_name).observe(duration)
    
    def set_system_metrics(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Установка системных метрик"""
        if not self.enabled:
            return
            
        self.system_cpu_percent.set(cpu_percent)
        self.system_memory_percent.set(memory_percent)
        self.system_disk_percent.set(disk_percent)
    
    def set_health_status(self, check: str, status: float):
        """Установка статуса здоровья"""
        if not self.enabled:
            return
            
        self.health_status.labels(check=check).set(status)
    
    @contextmanager
    def track_http_request(self, method: str, endpoint: str):
        """Контекстный менеджер для отслеживания HTTP запросов"""
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        self.http_active_requests.labels(endpoint=endpoint).inc()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.http_active_requests.labels(endpoint=endpoint).dec()
            # Статус будет установлен позже в record_http_request
    
    @contextmanager
    def track_agent_request(self, request_type: str):
        """Контекстный менеджер для отслеживания запросов агента"""
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_agent_request(request_type, duration)
    
    @contextmanager
    def track_tool_call(self, tool_name: str):
        """Контекстный менеджер для отслеживания вызовов инструментов"""
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            self.record_tool_call(tool_name, duration, success)
    
    def get_metrics(self) -> str:
        """Получение метрик в формате Prometheus"""
        if not self.enabled:
            return "# Metrics collection disabled\n"
            
        return generate_latest().decode('utf-8')


# Глобальный экземпляр сборщика метрик
metrics_collector = MetricsCollector()


# Декораторы для удобства
def track_agent_request(request_type: str = "basic"):
    """Декоратор для отслеживания запросов агента"""
    def decorator(func):
        if not metrics_collector.enabled:
            return func
            
        def wrapper(*args, **kwargs):
            with metrics_collector.track_agent_request(request_type):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_tool_call(tool_name: str):
    """Декоратор для отслеживания вызовов инструментов"""
    def decorator(func):
        if not metrics_collector.enabled:
            return func
            
        def wrapper(*args, **kwargs):
            with metrics_collector.track_tool_call(tool_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_duration_histogram(histogram_metric):
    """Декоратор для измерения времени выполнения функции"""
    def decorator(func):
        if not metrics_collector.enabled:
            return func
            
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                histogram_metric.observe(duration)
        return wrapper
    return decorator


# Функции-обертки для совместимости с импортами в integration.py
def track_agent_cognitive_load(component: str, load: float):
    """Обертка для установки уровня когнитивной нагрузки"""
    metrics_collector.set_cognitive_load(component, load)


def track_agent_memory_usage(memory_type: str, count: int):
    """Обертка для установки количества записей в памяти"""
    metrics_collector.set_memory_entries(memory_type, count)


def track_tool_usage(tool_name: str, duration: float, success: bool = True):
    """Обертка для записи вызова инструмента"""
    metrics_collector.record_tool_call(tool_name, duration, success)


def track_learning_experience(success: bool = True):
    """Обертка для записи опыта обучения"""
    metrics_collector.record_learning_experience(success)


def set_learning_success_rate(rate: float):
    """Обертка для установки процента успешного обучения"""
    metrics_collector.set_learning_success_rate(rate)


if __name__ == "__main__":
    # Пример использования
    print("Testing metrics collector...")
    
    # Запись примера метрик
    metrics_collector.record_http_request("GET", "/health", 200, 0.05)
    metrics_collector.record_agent_request("basic", 0.2, 0.85)
    metrics_collector.set_cognitive_load("main_loop", 0.6)
    metrics_collector.set_memory_entries("short_term", 42)
    
    # Вывод метрик
    print("\nCurrent metrics:")
    print(metrics_collector.get_metrics())
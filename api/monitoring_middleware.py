"""
Мидлвар для мониторинга API запросов AI Агента
Интеграция с системой метрик и логирования
"""

import time
import logging
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from monitoring.metrics_collector import metrics_collector
from monitoring.centralized_logging import get_app_logger


logger = logging.getLogger(__name__)
app_logger = get_app_logger()


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Мидлвар для мониторинга HTTP запросов
    Собирает метрики производительности и логирует запросы
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Получаем request_id из заголовков или генерируем новый
        request_id = request.headers.get('X-Request-ID', f"req_{int(start_time)}_{hash(str(request.url)) % 1000}")
        
        # Логируем начало запроса
        app_logger.set_context(request_id=request_id)
        app_logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "user_agent": request.headers.get('User-Agent', 'Unknown'),
                "client_host": request.client.host if request.client else None
            }
        )
        
        try:
            # Выполняем запрос
            response = await call_next(request)
            
            # Вычисляем время обработки
            process_time = time.time() - start_time
            
            # Записываем метрики
            if metrics_collector.enabled:
                metrics_collector.record_http_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code,
                    duration=process_time
                )
            
            # Логируем завершение запроса
            app_logger.info(
                f"Request completed: {response.status_code} in {process_time:.3f}s",
                extra={
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "content_length": response.headers.get('content-length', 0)
                }
            )
            
            # Добавляем request_id в заголовки ответа
            response.headers['X-Request-ID'] = request_id
            
            return response
        
        except Exception as e:
            # Вычисляем время до ошибки
            process_time = time.time() - start_time
            
            # Записываем метрики ошибки
            if metrics_collector.enabled:
                metrics_collector.record_http_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=500,
                    duration=process_time
                )
            
            # Логируем ошибку
            app_logger.error(
                f"Request failed with exception: {str(e)}",
                extra={
                    "status_code": 500,
                    "process_time": process_time,
                    "exception_type": type(e).__name__,
                    "method": request.method,
                    "path": request.url.path
                }
            )
            
            # Добавляем request_id в заголовки ответа (если возможно)
            if hasattr(response, 'headers'):
                response.headers['X-Request-ID'] = request_id
            else:
                response = Response(
                    content="Internal Server Error",
                    status_code=500,
                    headers={"X-Request-ID": request_id}
                )
            
            raise


class AgentPerformanceMiddleware:
    """
    Мидлвар для мониторинга производительности AI агента
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        response_status = None
        response_size = 0
        
        async def send_wrapper(message):
            nonlocal response_status, response_size
            
            if message["type"] == "http.response.start":
                response_status = message["status"]
            
            elif message["type"] == "http.response.body":
                response_size += len(message.get("body", b""))
            
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start_time
            
            # Специфичная логика для эндпоинтов агента
            path = scope.get("path", "")
            if "/agent/" in path:
                # Записываем метрики производительности агента
                if metrics_collector.enabled:
                    request_type = "meta_cognitive" if "meta" in path else "basic"
                    metrics_collector.record_agent_request(
                        request_type=request_type,
                        duration=duration
                    )
                
                app_logger.info(
                    f"Agent request completed",
                    extra={
                        "path": path,
                        "duration": duration,
                        "status": response_status,
                        "request_type": request_type
                    }
                )


def setup_monitoring_middleware(app):
    """
    Настройка мидлваров мониторинга для приложения
    
    Args:
        app: FastAPI приложение
    """
    # Добавляем мидлвар мониторинга
    app.add_middleware(MonitoringMiddleware)
    
    # Добавляем мидлвар производительности агента
    app.add_middleware(AgentPerformanceMiddleware)
    
    logger.info("Monitoring middleware configured")


# Метрики для мониторинга состояния системы
def update_system_health_metrics():
    """
    Обновление метрик состояния системы
    """
    if not metrics_collector.enabled:
        return
    
    import psutil
    
    # Обновляем системные метрики
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    
    metrics_collector.set_system_metrics(cpu_percent, memory_percent, disk_percent)
    
    # Обновляем метрику здоровья системы
    if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
        health_status = 0.0  # unhealthy
    elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 90:
        health_status = 0.5  # degraded
    else:
        health_status = 1.0  # healthy
    
    metrics_collector.set_health_status("system", health_status)


def track_agent_cognitive_load(component: str, load: float):
    """
    Отслеживание уровня когнитивной нагрузки
    
    Args:
        component: Компонент агента
        load: Уровень нагрузки (0.0-1.0)
    """
    if metrics_collector.enabled:
        metrics_collector.set_cognitive_load(component, load)
        
        # Обновляем метрику здоровья для когнитивной нагрузки
        if load > 0.9:
            health_status = 0.0  # unhealthy
        elif load > 0.7:
            health_status = 0.5  # degraded
        else:
            health_status = 1.0  # healthy
        
        metrics_collector.set_health_status(f"cognitive_load_{component}", health_status)


def track_agent_memory_usage(memory_type: str, count: int):
    """
    Отслеживание использования памяти агентом
    
    Args:
        memory_type: Тип памяти ('short_term', 'long_term')
        count: Количество записей
    """
    if metrics_collector.enabled:
        metrics_collector.set_memory_entries(memory_type, count)


def track_tool_usage(tool_name: str, duration: float, success: bool = True):
    """
    Отслеживание использования инструментов
    
    Args:
        tool_name: Название инструмента
        duration: Время выполнения
        success: Успешность выполнения
    """
    if metrics_collector.enabled:
        metrics_collector.record_tool_call(tool_name, duration, success)


def track_learning_experience(success: bool = True):
    """
    Отслеживание опыта обучения
    
    Args:
        success: Успешность обучения
    """
    if metrics_collector.enabled:
        metrics_collector.record_learning_experience(success)


def set_learning_success_rate(rate: float):
    """
    Установка процента успешного обучения
    
    Args:
        rate: Процент успешного обучения (0.0-1.0)
    """
    if metrics_collector.enabled:
        metrics_collector.set_learning_success_rate(rate)


if __name__ == "__main__":
    # Пример использования
    print("Monitoring middleware module loaded")
    print(f"Metrics collector enabled: {metrics_collector.enabled}")
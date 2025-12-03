"""
OpenTelemetry интеграция для распределенной трассировки
Инструментирование AI агента для мониторинга производительности
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable, TypeVar, Union
from contextlib import asynccontextmanager
from functools import wraps
import os

logger = logging.getLogger(__name__)

# Проверяем доступность OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextPropagator
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry not available, tracing disabled")
    OPENTELEMETRY_AVAILABLE = False

    # Stub классы для случаев когда OpenTelemetry недоступен
    class TracerProvider:
        pass

    class BatchSpanProcessor:
        def __init__(self, *args, **kwargs):
            pass

    class JaegerExporter:
        def __init__(self, *args, **kwargs):
            pass

    class OTLPSpanExporter:
        def __init__(self, *args, **kwargs):
            pass

    class Resource:
        def __init__(self, *args, **kwargs):
            pass

    class TraceContextPropagator:
        pass

    class W3CBaggagePropagator:
        pass

    class Status:
        OK = "OK"
        ERROR = "ERROR"

    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"

    # Создаем mock tracer
    class MockTracer:
        def start_as_current_span(self, name, **kwargs):
            return MockSpan()

        def start_span(self, name, **kwargs):
            return MockSpan()

    class MockSpan:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_attribute(self, key, value):
            pass

        def set_status(self, status, description=None):
            pass

        def record_exception(self, exception):
            pass

        def add_event(self, name, attributes=None):
            pass

        def end(self):
            pass

    trace = type('MockTrace', (), {'get_tracer': lambda *args: MockTracer()})()

F = TypeVar('F', bound=Callable[..., Any])

class TracingService:
    """
    Сервис для управления трассировкой
    """

    def __init__(self):
        self.enabled = OPENTELEMETRY_AVAILABLE and os.getenv('ENABLE_TRACING', 'false').lower() == 'true'
        self.service_name = os.getenv('SERVICE_NAME', 'ai-agent')
        self.service_version = os.getenv('SERVICE_VERSION', '1.0.0')

        if self.enabled:
            self._setup_tracing()
        else:
            logger.info("Tracing is disabled")

    def _setup_tracing(self):
        """Настройка OpenTelemetry tracing"""
        # Настройка ресурса
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: self.service_version,
        })

        # Настройка tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))

        # Настройка экспортеров
        exporters = []

        # Jaeger экспортер
        jaeger_endpoint = os.getenv('JAEGER_ENDPOINT')
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint.split(':')[0],
                agent_port=int(jaeger_endpoint.split(':')[1]) if ':' in jaeger_endpoint else 6831,
            )
            exporters.append(jaeger_exporter)

        # OTLP экспортер
        otlp_endpoint = os.getenv('OTLP_ENDPOINT')
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True,
            )
            exporters.append(otlp_exporter)

        # Добавляем процессоры для каждого экспортера
        for exporter in exporters:
            span_processor = BatchSpanProcessor(exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)

        # Настройка propagators для распределенной трассировки
        from opentelemetry import propagate
        propagate.set_global_textmap(TraceContextPropagator())
        propagate.set_global_baggage(W3CBaggagePropagator())

        logger.info(f"Tracing enabled with {len(exporters)} exporters")

    def get_tracer(self, name: str = "ai-agent"):
        """Получение tracer'а"""
        return trace.get_tracer(name)

    def is_enabled(self) -> bool:
        """Проверка включена ли трассировка"""
        return self.enabled

# Глобальный экземпляр tracing service
tracing_service = TracingService()

class TracingMiddleware:
    """
    Middleware для автоматической трассировки HTTP запросов
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not tracing_service.is_enabled():
            await self.app(scope, receive, send)
            return

        tracer = tracing_service.get_tracer("http")

        # Извлекаем информацию о запросе
        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        query_string = scope.get("query_string", b"").decode()

        span_name = f"{method} {path}"
        if query_string:
            span_name += f"?{query_string}"

        # Создаем span для HTTP запроса
        with tracer.start_as_current_span(
            span_name,
            kind=trace.SpanKind.SERVER
        ) as span:
            # Добавляем атрибуты
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", f"{path}?{query_string}" if query_string else path)
            span.set_attribute("http.scheme", scope.get("scheme", "http"))

            # Извлекаем заголовки
            headers = dict(scope.get("headers", []))
            headers = {k.decode(): v.decode() for k, v in headers.items()}

            user_agent = headers.get("User-Agent", "Unknown")
            span.set_attribute("http.user_agent", user_agent)

            request_id = headers.get("X-Request-ID", "unknown")
            span.set_attribute("request.id", request_id)

            # Время начала
            start_time = time.time()

            # Обертываем send для трассировки ответа
            original_send = send

            async def traced_send(message):
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    span.set_attribute("http.status_code", status_code)

                    # Определяем статус span'а
                    if status_code >= 400:
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {status_code}"))
                    else:
                        span.set_status(Status(StatusCode.OK))

                elif message["type"] == "http.response.body":
                    # Вычисляем время ответа
                    duration = time.time() - start_time
                    span.set_attribute("http.response_time", duration)

                    # Добавляем событие завершения
                    span.add_event("response_sent", {
                        "duration_ms": duration * 1000
                    })

                await original_send(message)

            try:
                await self.app(scope, receive, traced_send)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    Декоратор для трассировки функций

    Args:
        name: Имя span'а (по умолчанию имя функции)
        attributes: Дополнительные атрибуты
    """
    def decorator(func: F) -> F:
        if not tracing_service.is_enabled():
            return func

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = tracing_service.get_tracer("function")
            span_name = name or f"{func.__module__}.{func.__qualname__}"

            with tracer.start_as_current_span(span_name) as span:
                # Добавляем атрибуты
                span.set_attribute("function.name", func.__qualname__)
                span.set_attribute("function.module", func.__module__)

                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Добавляем аргументы функции (только простые типы)
                if len(args) > 0 and hasattr(args[0], '__class__'):
                    span.set_attribute("function.self", args[0].__class__.__name__)

                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time

                    span.set_attribute("function.duration", duration)
                    span.set_status(Status(StatusCode.OK))

                    # Добавляем событие успешного выполнения
                    span.add_event("function_completed", {
                        "duration_ms": duration * 1000
                    })

                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = tracing_service.get_tracer("function")
            span_name = name or f"{func.__module__}.{func.__qualname__}"

            with tracer.start_as_current_span(span_name) as span:
                # Добавляем атрибуты
                span.set_attribute("function.name", func.__qualname__)
                span.set_attribute("function.module", func.__module__)

                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time

                    span.set_attribute("function.duration", duration)
                    span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

@asynccontextmanager
async def trace_context(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Контекстный менеджер для трассировки блоков кода

    Args:
        name: Имя span'а
        attributes: Атрибуты span'а
    """
    if not tracing_service.is_enabled():
        yield
        return

    tracer = tracing_service.get_tracer("context")

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            start_time = time.time()
            yield span
            duration = time.time() - start_time

            span.set_attribute("context.duration", duration)
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

class TracingMetricsCollector:
    """
    Коллектор метрик трассировки для интеграции с системами мониторинга
    """

    def __init__(self):
        self.metrics = {
            "spans_created": 0,
            "spans_completed": 0,
            "spans_errored": 0,
            "trace_duration_sum": 0.0,
            "trace_duration_count": 0
        }

    def record_span_created(self):
        """Запись создания span'а"""
        self.metrics["spans_created"] += 1

    def record_span_completed(self, duration: float):
        """Запись завершения span'а"""
        self.metrics["spans_completed"] += 1
        self.metrics["trace_duration_sum"] += duration
        self.metrics["trace_duration_count"] += 1

    def record_span_error(self):
        """Запись ошибки span'а"""
        self.metrics["spans_errored"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик"""
        metrics = self.metrics.copy()

        # Вычисляем среднюю продолжительность
        if metrics["trace_duration_count"] > 0:
            metrics["trace_duration_avg"] = metrics["trace_duration_sum"] / metrics["trace_duration_count"]
        else:
            metrics["trace_duration_avg"] = 0.0

        # Вычисляем процент ошибок
        total_spans = metrics["spans_completed"] + metrics["spans_errored"]
        if total_spans > 0:
            metrics["error_rate"] = metrics["spans_errored"] / total_spans
        else:
            metrics["error_rate"] = 0.0

        return metrics

# Глобальный коллектор метрик
tracing_metrics = TracingMetricsCollector()

# Специализированные декораторы для AI агента
def trace_agent_operation(operation_type: str):
    """Декоратор для трассировки операций агента"""
    def decorator(func):
        return trace_function(
            name=f"agent.{operation_type}",
            attributes={
                "agent.operation": operation_type,
                "agent.function": func.__qualname__
            }
        )(func)
    return decorator

def trace_model_inference(model_name: str):
    """Декоратор для трассировки инференса модели"""
    def decorator(func):
        return trace_function(
            name=f"model.{model_name}.inference",
            attributes={
                "model.name": model_name,
                "model.operation": "inference"
            }
        )(func)
    return decorator

def trace_database_operation(operation: str, table: Optional[str] = None):
    """Декоратор для трассировки операций базы данных"""
    def decorator(func):
        attributes = {
            "db.operation": operation,
            "db.system": "postgresql"  # или другой
        }
        if table:
            attributes["db.table"] = table

        return trace_function(
            name=f"db.{operation}",
            attributes=attributes
        )(func)
    return decorator

def trace_cache_operation(operation: str, key_pattern: Optional[str] = None):
    """Декоратор для трассировки операций кэша"""
    def decorator(func):
        attributes = {
            "cache.operation": operation,
            "cache.system": "redis"  # или другой
        }
        if key_pattern:
            attributes["cache.key_pattern"] = key_pattern

        return trace_function(
            name=f"cache.{operation}",
            attributes=attributes
        )(func)
    return decorator

def trace_external_api_call(service_name: str, endpoint: str):
    """Декоратор для трассировки внешних API вызовов"""
    def decorator(func):
        return trace_function(
            name=f"external.{service_name}.{endpoint}",
            attributes={
                "external.service": service_name,
                "external.endpoint": endpoint,
                "span.kind": "client"
            }
        )(func)
    return decorator

# Конфигурация для различных сред
def configure_tracing_for_development():
    """Конфигурация трассировки для разработки"""
    os.environ.setdefault('ENABLE_TRACING', 'true')
    os.environ.setdefault('SERVICE_NAME', 'ai-agent-dev')
    os.environ.setdefault('JAEGER_ENDPOINT', 'localhost:6831')

def configure_tracing_for_production():
    """Конфигурация трассировки для продакшена"""
    os.environ.setdefault('ENABLE_TRACING', 'true')
    os.environ.setdefault('SERVICE_NAME', 'ai-agent-prod')
    os.environ.setdefault('OTLP_ENDPOINT', 'otel-collector:4317')

# Примеры использования
if __name__ == "__main__":
    # Пример конфигурации
    configure_tracing_for_development()

    # Пересоздаем сервис с новыми настройками
    import monitoring.opentelemetry_tracing as tracing_module
    tracing_module.tracing_service = TracingService()

    print(f"Tracing enabled: {tracing_module.tracing_service.is_enabled()}")
    print(f"Service name: {tracing_module.tracing_service.service_name}")

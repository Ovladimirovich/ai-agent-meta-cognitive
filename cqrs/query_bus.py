"""
Query Bus для CQRS паттерна
Реализация шины запросов для операций чтения
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Type, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class Query:
    """Базовый класс запроса"""
    query_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class QueryResult(Generic[T]):
    """Результат выполнения запроса"""
    success: bool
    query_id: str
    data: Optional[T] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class QueryHandlerNotFoundError(Exception):
    """Обработчик запроса не найден"""
    pass

class QueryHandler(Generic[T]):
    """Базовый класс обработчика запросов"""

    async def handle(self, query: Query) -> QueryResult[T]:
        """Обработка запроса"""
        raise NotImplementedError("Subclasses must implement handle method")

class QueryMiddleware:
    """Middleware для обработки запросов"""

    async def __call__(self, query: Query, next_handler: Callable[[Query], Awaitable[QueryResult]]) -> QueryResult:
        """Обработка запроса в middleware"""
        return await next_handler(query)

class CachingMiddleware(QueryMiddleware):
    """Middleware кэширования запросов"""

    def __init__(self, cache_manager, ttl_seconds: int = 300):
        self.cache_manager = cache_manager
        self.ttl_seconds = ttl_seconds

    async def __call__(self, query: Query, next_handler: Callable[[Query], Awaitable[QueryResult]]) -> QueryResult:
        # Создаем ключ кэша на основе типа запроса и его данных
        cache_key = f"query:{type(query).__name__}:{hash(str(query.__dict__))}"

        # Проверяем кэш
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Query cache hit for {type(query).__name__}")
            return cached_result

        # Выполняем запрос
        result = await next_handler(query)

        # Кэшируем успешный результат
        if result.success and result.data is not None:
            await self.cache_manager.set(cache_key, result, ttl=self.ttl_seconds)
            logger.debug(f"Query cached for {type(query).__name__}")

        return result

class LoggingMiddleware(QueryMiddleware):
    """Middleware логирования запросов"""

    async def __call__(self, query: Query, next_handler: Callable[[Query], Awaitable[QueryResult]]) -> QueryResult:
        query_type = type(query).__name__
        logger.info(f"Executing query: {query_type} (ID: {query.query_id})")

        start_time = asyncio.get_event_loop().time()
        try:
            result = await next_handler(query)
            duration = asyncio.get_event_loop().time() - start_time

            if result.success:
                logger.info(f"Query {query_type} completed successfully in {duration:.3f}s")
            else:
                logger.warning(f"Query {query_type} failed: {result.error_message}")

            return result
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(f"Query {query_type} threw exception after {duration:.3f}s: {e}")
            raise

class QueryBus:
    """
    Шина запросов для CQRS

    Управляет регистрацией обработчиков и выполнением запросов через middleware
    """

    def __init__(self):
        self.handlers: Dict[Type[Query], QueryHandler] = {}
        self.middleware: List[QueryMiddleware] = []

    def register_handler(self, query_type: Type[Query], handler: QueryHandler):
        """Регистрация обработчика для типа запроса"""
        self.handlers[query_type] = handler
        logger.info(f"Registered handler for query: {query_type.__name__}")

    def add_middleware(self, middleware: QueryMiddleware):
        """Добавление middleware"""
        self.middleware.append(middleware)
        logger.info(f"Added middleware: {type(middleware).__name__}")

    async def execute(self, query: Query) -> QueryResult:
        """
        Выполнение запроса

        Args:
            query: Запрос для выполнения

        Returns:
            QueryResult: Результат выполнения

        Raises:
            QueryHandlerNotFoundError: Если обработчик не найден
        """
        # Находим обработчик
        handler = self.handlers.get(type(query))
        if not handler:
            raise QueryHandlerNotFoundError(f"No handler found for query: {type(query).__name__}")

        # Строим цепочку middleware
        async def execute_with_handler(q):
            return await handler.handle(q)

        current_handler = execute_with_handler

        # Применяем middleware в обратном порядке
        for middleware in reversed(self.middleware):
            async def create_wrapper(q, mw=middleware, next_handler=current_handler):
                return await mw(q, next_handler)
            current_handler = create_wrapper

        return await current_handler(query)

    def get_registered_queries(self) -> List[str]:
        """Получение списка зарегистрированных запросов"""
        return [query.__name__ for query in self.handlers.keys()]

# Глобальная шина запросов
query_bus = QueryBus()

# Декораторы для удобства
def query_handler(query_type: Type[Query]):
    """Декоратор для регистрации обработчика запросов"""
    def decorator(cls):
        handler = cls()
        query_bus.register_handler(query_type, handler)
        return cls
    return decorator

# Примеры запросов для AI агента (команды определены в command_bus.py)
@dataclass
class GetAgentStatusQuery(Query):
    """Запрос статуса агента"""
    include_details: bool = False

@dataclass
class GetTaskHistoryQuery(Query):
    """Запрос истории задач"""
    agent_id: Optional[str] = None
    limit: int = 10
    offset: int = 0

@dataclass
class GetPerformanceMetricsQuery(Query):
    """Запрос метрик производительности"""
    time_range_hours: int = 24
    metric_types: List[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.metric_types is None:
            self.metric_types = ["cpu", "memory", "response_time"]

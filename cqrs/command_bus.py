"""
Command Bus для CQRS паттерна
Реализация шины команд с middleware и обработчиками
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Type, TypeVar
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class Command:
    """Базовый класс команды"""
    command_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CommandResult:
    """Результат выполнения команды"""
    success: bool
    command_id: str
    result: Any = None
    error_message: Optional[str] = None
    events: List['DomainEvent'] = None

    def __post_init__(self):
        if self.events is None:
            self.events = []

class CommandHandlerNotFoundError(Exception):
    """Обработчик команды не найден"""
    pass

class CommandValidationError(Exception):
    """Ошибка валидации команды"""
    pass

class CommandHandler:
    """Базовый класс обработчика команд"""

    async def handle(self, command: Command) -> CommandResult:
        """Обработка команды"""
        raise NotImplementedError("Subclasses must implement handle method")

class CommandMiddleware:
    """Middleware для обработки команд"""

    async def __call__(self, command: Command, next_handler: Callable[[Command], Awaitable[CommandResult]]) -> CommandResult:
        """Обработка команды в middleware"""
        return await next_handler(command)

class ValidationMiddleware(CommandMiddleware):
    """Middleware валидации команд"""

    def __init__(self, validator: Callable[[Command], Awaitable[None]]):
        self.validator = validator

    async def __call__(self, command: Command, next_handler: Callable[[Command], Awaitable[CommandResult]]) -> CommandResult:
        try:
            await self.validator(command)
            return await next_handler(command)
        except Exception as e:
            logger.error(f"Command validation failed for {type(command).__name__}: {e}")
            raise CommandValidationError(f"Validation failed: {e}")

class LoggingMiddleware(CommandMiddleware):
    """Middleware логирования команд"""

    async def __call__(self, command: Command, next_handler: Callable[[Command], Awaitable[CommandResult]]) -> CommandResult:
        command_type = type(command).__name__
        logger.info(f"Executing command: {command_type} (ID: {command.command_id})")

        start_time = asyncio.get_event_loop().time()
        try:
            result = await next_handler(command)
            duration = asyncio.get_event_loop().time() - start_time

            if result.success:
                logger.info(f"Command {command_type} completed successfully in {duration:.3f}s")
            else:
                logger.error(f"Command {command_type} failed: {result.error_message}")

            return result
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(f"Command {command_type} threw exception after {duration:.3f}s: {e}")
            raise

class CommandBus:
    """
    Шина команд для CQRS

    Управляет регистрацией обработчиков и выполнением команд через middleware
    """

    def __init__(self):
        self.handlers: Dict[Type[Command], CommandHandler] = {}
        self.middleware: List[CommandMiddleware] = []

    def register_handler(self, command_type: Type[Command], handler: CommandHandler):
        """Регистрация обработчика для типа команды"""
        self.handlers[command_type] = handler
        logger.info(f"Registered handler for command: {command_type.__name__}")

    def add_middleware(self, middleware: CommandMiddleware):
        """Добавление middleware"""
        self.middleware.append(middleware)
        logger.info(f"Added middleware: {type(middleware).__name__}")

    async def execute(self, command: Command) -> CommandResult:
        """
        Выполнение команды

        Args:
            command: Команда для выполнения

        Returns:
            CommandResult: Результат выполнения

        Raises:
            CommandHandlerNotFoundError: Если обработчик не найден
        """
        # Находим обработчик
        handler = self.handlers.get(type(command))
        if not handler:
            raise CommandHandlerNotFoundError(f"No handler found for command: {type(command).__name__}")

        # Строим цепочку middleware
        async def execute_with_handler(cmd):
            return await handler.handle(cmd)

        current_handler = execute_with_handler

        # Применяем middleware в обратном порядке
        for middleware in reversed(self.middleware):
            async def create_wrapper(cmd, mw=middleware, next_handler=current_handler):
                return await mw(cmd, next_handler)
            current_handler = create_wrapper

        return await current_handler(command)

    def get_registered_commands(self) -> List[str]:
        """Получение списка зарегистрированных команд"""
        return [cmd.__name__ for cmd in self.handlers.keys()]

# Глобальная шина команд
command_bus = CommandBus()

# Декораторы для удобства
def command_handler(command_type: Type[Command]):
    """Декоратор для регистрации обработчика команд"""
    def decorator(cls):
        handler = cls()
        command_bus.register_handler(command_type, handler)
        return cls
    return decorator

def command_validator(command_type: Type[Command]):
    """Декоратор для регистрации валидатора команд"""
    def decorator(func: Callable[[Command], Awaitable[None]]):
        middleware = ValidationMiddleware(func)
        command_bus.add_middleware(middleware)
        return func
    return decorator

"""
Модуль централизованного логирования для AI Агента
Обеспечивает структурированное логирование с фильтрацией чувствительных данных
и интеграцией системами агрегации логов
"""

import logging
import logging.config
import json
import re
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import structlog
from enum import Enum


class LogLevel(Enum):
    """Уровни логирования"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Форматы логирования"""
    JSON = "json"
    STRUCTURED = "structured"
    CONSOLE = "console"


# Паттерны для фильтрации чувствительных данных
SENSITIVE_PATTERNS = [
    r'password[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]',  # password="value"
    r'password[\'"]?\s*[:=]\s*\w+',  # password=value
    r'token[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]',  # token="value"
    r'api[_-]?key[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]', # api_key="value"
    r'secret[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]',  # secret="value"
    r'Bearer\s+[A-Za-z0-9\-_\.]+',  # Bearer tokens
    r'Authorization:\s*[\w\s]+[^\s]*',  # Authorization headers
    r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',  # Email addresses (basic)
    r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',  # Credit card numbers
]


class SensitiveDataFilter(logging.Filter):
    """Фильтр для удаления чувствительных данных из логов"""
    
    def __init__(self):
        super().__init__()
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SENSITIVE_PATTERNS]
    
    def filter(self, record):
        """Фильтрует чувствительные данные из записи лога"""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._filter_sensitive_data(record.msg)
        
        if hasattr(record, 'args'):
            record.args = tuple(
                self._filter_sensitive_data(str(arg)) if isinstance(arg, (str, dict)) else arg
                for arg in record.args
            )
        
        return True
    
    def _filter_sensitive_data(self, text: str) -> str:
        """Удаляет чувствительные данные из текста"""
        if not isinstance(text, str):
            return str(text)
        
        filtered_text = text
        for pattern in self.compiled_patterns:
            filtered_text = pattern.sub('[FILTERED]', filtered_text)
        
        return filtered_text


class JSONFormatter(logging.Formatter):
    """JSON форматтер для структурированного логирования"""
    
    def format(self, record):
        """Форматирует запись в JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': self._filter_sensitive_data(record.getMessage()),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Добавляем дополнительные поля если есть
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        
        if hasattr(record, 'trace_id'):
            log_entry['trace_id'] = record.trace_id
        
        if hasattr(record, 'span_id'):
            log_entry['span_id'] = record.span_id
        
        # Добавляем exception info если есть
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Добавляем стек вызовов если есть
        if record.stack_info:
            log_entry['stack_info'] = self.formatStack(record.stack_info)
        
        return json.dumps(log_entry, ensure_ascii=False)
    
    def _filter_sensitive_data(self, text: str) -> str:
        """Фильтрует чувствительные данные"""
        filter_obj = SensitiveDataFilter()
        return filter_obj._filter_sensitive_data(text)


class ContextualLogger:
    """Класс для логирования с контекстом"""
    
    def __init__(self, name: str, logger: logging.Logger = None):
        self.name = name
        self.logger = logger or logging.getLogger(name)
        self.context = {}
    
    def set_context(self, **kwargs):
        """Устанавливает контекст для логирования"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Очищает контекст"""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """Логирует сообщение с контекстом"""
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        
        self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)


class CentralizedLoggingConfig:
    """Конфигурация централизованного логирования"""
    
    def __init__(
        self,
        service_name: str = "ai-agent",
        log_level: str = "INFO",
        log_format: LogFormat = LogFormat.JSON,
        enable_console: bool = True,
        enable_file: bool = True,
        log_file_path: Optional[str] = None,
        enable_remote: bool = True,
        remote_endpoint: Optional[str] = None,
        sensitive_data_filtering: bool = True
    ):
        self.service_name = service_name
        self.log_level = log_level.upper()
        self.log_format = log_format
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.log_file_path = log_file_path or f"logs/{service_name}.log"
        self.enable_remote = enable_remote
        self.remote_endpoint = remote_endpoint
        self.sensitive_data_filtering = sensitive_data_filtering
        
        # Создаем директорию для логов
        log_dir = Path(self.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Получение конфигурации логирования"""
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                },
                'json': {
                    '()': JSONFormatter,
                },
            },
            'filters': {},
            'handlers': {},
            'loggers': {}
        }
        
        # Добавляем фильтр чувствительных данных если включено
        if self.sensitive_data_filtering:
            config['filters']['sensitive_filter'] = {
                '()': SensitiveDataFilter,
            }
        
        # Конфигурация консольного хендлера
        if self.enable_console:
            console_handler_config = {
                'class': 'logging.StreamHandler',
                'level': self.log_level,
                'formatter': 'json' if self.log_format == LogFormat.JSON else 'detailed',
                'stream': sys.stdout,
            }
            
            if self.sensitive_data_filtering:
                console_handler_config['filters'] = ['sensitive_filter']
            
            config['handlers']['console'] = console_handler_config
        
        # Конфигурация файлового хендлера
        if self.enable_file:
            file_handler_config = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': self.log_level,
                'formatter': 'json' if self.log_format == LogFormat.JSON else 'detailed',
                'filename': self.log_file_path,
                'maxBytes': 50 * 1024 * 1024,  # 50MB
                'backupCount': 10,
            }
            
            if self.sensitive_data_filtering:
                file_handler_config['filters'] = ['sensitive_filter']
            
            config['handlers']['file'] = file_handler_config
        
        # Конфигурация удаленного хендлера (для отправки в систему агрегации)
        if self.enable_remote and self.remote_endpoint:
            # Используем HTTPHandler для отправки логов в удаленную систему
            remote_handler_config = {
                'class': 'logging.handlers.HTTPHandler',
                'level': self.log_level,
                'host': self.remote_endpoint,
                'url': '/logs',
                'method': 'POST',
                'formatter': 'json' if self.log_format == LogFormat.JSON else 'detailed',
            }
            
            if self.sensitive_data_filtering:
                remote_handler_config['filters'] = ['sensitive_filter']
            
            config['handlers']['remote'] = remote_handler_config
        
        # Конфигурация логгеров
        handlers = []
        if self.enable_console:
            handlers.append('console')
        if self.enable_file:
            handlers.append('file')
        if self.enable_remote and self.remote_endpoint:
            handlers.append('remote')
        
        config['loggers'][''] = {  # Root logger
            'handlers': handlers,
            'level': self.log_level,
            'propagate': True,
        }
        
        # Конфигурация для специфичных логгеров
        config['loggers'].update({
            'uvicorn': {
                'handlers': handlers,
                'level': 'WARNING',
                'propagate': False,
            },
            'uvicorn.access': {
                'handlers': handlers,
                'level': 'WARNING',
                'propagate': False,
            },
            'fastapi': {
                'handlers': handlers,
                'level': 'WARNING',
                'propagate': False,
            },
            'sqlalchemy': {
                'handlers': handlers,
                'level': 'WARNING',
                'propagate': False,
            },
            'ai_agent': {
                'handlers': handlers,
                'level': self.log_level,
                'propagate': False,
            }
        })
        
        return config
    
    def setup_logging(self):
        """Настройка логирования по конфигурации"""
        config = self.get_logging_config()
        logging.config.dictConfig(config)
        
        # Настраиваем structlog если доступен
        try:
            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.stdlib.filter_by_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
        except ImportError:
            pass  # structlog не установлен, продолжаем без него


def setup_centralized_logging(
    service_name: str = "ai-agent",
    log_level: str = "INFO",
    log_format: LogFormat = LogFormat.JSON,
    enable_console: bool = True,
    enable_file: bool = True,
    log_file_path: Optional[str] = None,
    enable_remote: bool = True,
    remote_endpoint: Optional[str] = None,
    sensitive_data_filtering: bool = True
) -> ContextualLogger:
    """
    Настройка централизованного логирования
    
    Args:
        service_name: Название сервиса
        log_level: Уровень логирования
        log_format: Формат логирования
        enable_console: Включить логирование в консоль
        enable_file: Включить логирование в файл
        log_file_path: Путь к файлу логов
        enable_remote: Включить удаленную отправку логов
        remote_endpoint: Адрес удаленной системы логирования
        sensitive_data_filtering: Включить фильтрацию чувствительных данных
    
    Returns:
        ContextualLogger: Логгер с контекстом
    """
    config = CentralizedLoggingConfig(
        service_name=service_name,
        log_level=log_level,
        log_format=log_format,
        enable_console=enable_console,
        enable_file=enable_file,
        log_file_path=log_file_path,
        enable_remote=enable_remote,
        remote_endpoint=remote_endpoint,
        sensitive_data_filtering=sensitive_data_filtering
    )
    
    config.setup_logging()
    
    # Возвращаем контекстный логгер для основного приложения
    return ContextualLogger(service_name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    **extra_context
):
    """
    Логирование с контекстом
    
    Args:
        logger: Логгер
        level: Уровень логирования
        message: Сообщение
        request_id: ID запроса
        user_id: ID пользователя
        session_id: ID сессии
        trace_id: ID трассировки
        **extra_context: Дополнительный контекст
    """
    extra = {}
    if request_id:
        extra['request_id'] = request_id
    if user_id:
        extra['user_id'] = user_id
    if session_id:
        extra['session_id'] = session_id
    if trace_id:
        extra['trace_id'] = trace_id
    
    extra.update(extra_context)
    
    logger.log(level, message, extra=extra)


def create_safe_error_message(error: Exception, include_traceback: bool = False) -> str:
    """
    Создает безопасное сообщение об ошибке без чувствительных данных
    
    Args:
        error: Исключение
        include_traceback: Включать ли traceback
    
    Returns:
        Безопасное сообщение об ошибке
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    # Фильтруем чувствительные данные
    filter_obj = SensitiveDataFilter()
    safe_message = filter_obj._filter_sensitive_data(error_message)
    
    if include_traceback and hasattr(error, '__traceback__'):
        import traceback
        tb_str = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        safe_message += f"\n\nTraceback:\n{filter_obj._filter_sensitive_data(tb_str)}"
    
    return f"{error_type}: {safe_message}"


# Глобальный логгер для приложения
app_logger = None


def get_app_logger() -> ContextualLogger:
    """Получение глобального логгера приложения"""
    global app_logger
    if app_logger is None:
        app_logger = setup_centralized_logging(
            service_name="ai-agent",
            log_level="INFO",
            log_format=LogFormat.JSON,
            enable_console=True,
            enable_file=True,
            log_file_path="logs/ai-agent.log",
            enable_remote=True,
            remote_endpoint="loki:3100"  # Для интеграции с Loki
        )
    return app_logger


if __name__ == "__main__":
    # Пример использования
    logger = setup_centralized_logging(
        service_name="ai-agent-test",
        log_level="DEBUG",
        log_format=LogFormat.JSON,
        enable_console=True,
        enable_file=True,
        log_file_path="logs/test-ai-agent.log"
    )
    
    # Логирование с контекстом
    logger.set_context(user_id="user123", session_id="session456")
    logger.info("Application started", extra={"version": "1.0"})
    logger.error("Test error occurred", extra={"error_code": "E001"})
    
    # Логирование чувствительных данных (они будут отфильтрованы)
    logger.warning("Request with sensitive data", extra={
        "password": "secret123",
        "token": "abc123xyz",
        "api_key": "key123456"
    })
    
    print("Centralized logging setup completed!")
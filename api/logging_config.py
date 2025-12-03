"""
Продакшен конфигурация логирования для AI Агента
Обеспечивает безопасное логирование без утечек чувствительных данных
"""

import logging
import logging.config
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import re
from pathlib import Path

# Список чувствительных паттернов для фильтрации
SENSITIVE_PATTERNS = [
    r'password[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]', # password="value"
    r'password[\'"]?\s*[:=]\s*\w+',  # password=value
    r'token[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]', # token="value"
    r'token[\'"]?\s*[:=]\s*\w+',  # token=value
    r'api[_-]?key[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]', # api_key="value"
    r'api[_-]?key[\'"]?\s*[:=]\s*\w+',  # api_key=value
    r'secret[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]',  # secret="value"
    r'secret[\'"]?\s*[:=]\s*\w+',  # secret=value
    r'Bearer\s+[A-Za-z0-9\-_\.]+', # Bearer tokens
    r'Authorization:\s*[\w\s]+[^\s]*',  # Authorization headers
    r'[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*[_-]?key[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]', # *_key="value"
    r'[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*[_-]?key[\'"]?\s*[:=]\s*\w+',  # *_key=value
    r'client[_-]?id[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]',  # client_id="value"
    r'client[_-]?id[\'"]?\s*[:=]\s*\w+',  # client_id=value
    r'client[_-]?secret[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]', # client_secret="value"
    r'client[_-]?secret[\'"]?\s*[:=]\s*\w+',  # client_secret=value
    r'access[_-]?token[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]',  # access_token="value"
    r'access[_-]?token[\'"]?\s*[:=]\s*\w+',  # access_token=value
    r'refresh[_-]?token[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]', # refresh_token="value"
    r'refresh[_-]?token[\'"]?\s*[:=]\s*\w+', # refresh_token=value
    r'[A-Za-z0-9]{20,}',  # Длинные строки, возможно токены
]

class SensitiveDataFilter(logging.Filter):
    """Фильтр для удаления чувствительных данных из логов"""

    def __init__(self):
        super().__init__()
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SENSITIVE_PATTERNS]
        
        # Создаем специальные паттерны для замены только значений
        self.value_patterns = [
            # Паттерны, которые захватывают только значения в кавычках
            (re.compile(r'(password[\'"]?\s*[:=]\s*[\'"])' + r'([^\'"]+)' + r'([\'"])', re.IGNORECASE), r'\g<1>[FILTERED]\g<3>'),
            (re.compile(r'(token[\'"]?\s*[:=]\s*[\'"])' + r'([^\'"]+)' + r'([\'"])', re.IGNORECASE), r'\g<1>[FILTERED]\g<3>'),
            (re.compile(r'(api[_-]?key[\'"]?\s*[:=]\s*[\'"])' + r'([^\'"]+)' + r'([\'"])', re.IGNORECASE), r'\g<1>[FILTERED]\g<3>'),
            (re.compile(r'(secret[\'"]?\s*[:=]\s*[\'"])' + r'([^\'"]+)' + r'([\'"])', re.IGNORECASE), r'\g<1>[FILTERED]\g<3>'),
            (re.compile(r'(client[_-]?id[\'"]?\s*[:=]\s*[\'"])' + r'([^\'"]+)' + r'([\'"])', re.IGNORECASE), r'\g<1>[FILTERED]\g<3>'),
            (re.compile(r'(client[_-]?secret[\'"]?\s*[:=]\s*[\'"])' + r'([^\'"]+)' + r'([\'"])', re.IGNORECASE), r'\g<1>[FILTERED]\g<3>'),
            (re.compile(r'(access[_-]?token[\'"]?\s*[:=]\s*[\'"])' + r'([^\'"]+)' + r'([\'"])', re.IGNORECASE), r'\g<1>[FILTERED]\g<3>'),
            (re.compile(r'(refresh[_-]?token[\'"]?\s*[:=]\s*[\'"])' + r'([^\'"]+)' + r'([\'"])', re.IGNORECASE), r'\g<1>[FILTERED]\g<3>'),
            (re.compile(r'(([A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*[_-]?key[\'"]?\s*[:=]\s*[\'"])' + r'([^\'"]+)' + r'([\'"]))', re.IGNORECASE), r'\g<2>[FILTERED]\g<4>'),
            
            # Паттерны, которые захватывают значения без кавычек
            (re.compile(r'(password[\'"]?\s*[:=]\s*)' + r'(\w+)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'(token[\'"]?\s*[:=]\s*)' + r'(\w+)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'(api[_-]?key[\'"]?\s*[:=]\s*)' + r'(\w+)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'(secret[\'"]?\s*[:=]\s*)' + r'(\w+)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'(client[_-]?id[\'"]?\s*[:=]\s*)' + r'(\w+)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'(client[_-]?secret[\'"]?\s*[:=]\s*)' + r'(\w+)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'(access[_-]?token[\'"]?\s*[:=]\s*)' + r'(\w+)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'(refresh[_-]?token[\'"]?\s*[:=]\s*)' + r'(\w+)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'(([A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*[_-]?key[\'"]?\s*[:=]\s*)' + r'(\w+))', re.IGNORECASE), r'\g<2>[FILTERED]'),
            
            # Другие специфические паттерны
            (re.compile(r'(Bearer\s+)' + r'([A-Za-z0-9\-_\.]+)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'(Authorization:\s*[\w\s]+)' + r'([^\s]*)', re.IGNORECASE), r'\g<1>[FILTERED]'),
            (re.compile(r'([A-Za-z0-9]{20,})', re.IGNORECASE), r'[FILTERED]'),
        ]

    def filter(self, record):
        """Фильтрует чувствительные данные из записи лога"""
        if hasattr(record, 'msg'):
            if isinstance(record.msg, str):
                record.msg = self._filter_sensitive_data(record.msg)
            elif isinstance(record.msg, dict):
                record.msg = self._filter_sensitive_data_in_dict(record.msg)
            elif isinstance(record.msg, (list, tuple)):
                record.msg = self._filter_sensitive_data_in_list(record.msg)

        if hasattr(record, 'args'):
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    filtered_args.append(self._filter_sensitive_data(arg))
                elif isinstance(arg, dict):
                    filtered_args.append(self._filter_sensitive_data_in_dict(arg))
                elif isinstance(arg, (list, tuple)):
                    filtered_args.append(self._filter_sensitive_data_in_list(arg))
                else:
                    filtered_args.append(arg)
            record.args = tuple(filtered_args)

        return True

    def _filter_sensitive_data_in_dict(self, data: dict) -> dict:
        """Фильтрует чувствительные данные в словаре"""
        if not isinstance(data, dict):
            return data
            
        filtered_dict = {}
        for key, value in data.items():
            # Проверяем ключ на чувствительность
            filtered_key = key
            if isinstance(key, str):
                for pattern in self.compiled_patterns:
                    if pattern.search(key):
                        filtered_key = '[FILTERED_KEY]'
                        break
            
            # Фильтруем значение
            if isinstance(value, str):
                # Для строк в словарях применяем особую логику фильтрации
                # Проверяем, является ли значение потенциально чувствительным
                filtered_value = self._filter_sensitive_value_in_context(key, value)
                filtered_dict[filtered_key] = filtered_value
            elif isinstance(value, dict):
                filtered_dict[filtered_key] = self._filter_sensitive_data_in_dict(value)
            elif isinstance(value, (list, tuple)):
                filtered_dict[filtered_key] = self._filter_sensitive_data_in_list(value)
            else:
                filtered_dict[filtered_key] = value
        
        return filtered_dict
    
    def _filter_sensitive_value_in_context(self, key: str, value: str) -> str:
        """Фильтрует значение в контексте ключа"""
        # Определяем чувствительные ключи
        sensitive_keys = [
            'password', 'token', 'api_key', 'secret',
            'client_secret', 'client_id', 'access_token',
            'refresh_token', 'key', 'auth', 'authorization'
        ]
        
        # Если ключ является чувствительным, фильтруем значение
        if any(sensitive_key.lower() in key.lower() for sensitive_key in sensitive_keys):
            # Просто фильтруем значение как потенциально чувствительное
            # Проверим, похоже ли значение на токен/пароль
            if self._is_value_likely_sensitive(value):
                return '[FILTERED]'
            else:
                return value
        else:
            # Для нечувствительных ключей возвращаем значение как есть
            return value
    
    def _is_value_likely_sensitive(self, value: str) -> bool:
        """Проверяет, является ли значение потенциально чувствительным"""
        # Проверим, похоже ли значение на токен/пароль по его характеристикам
        if not isinstance(value, str):
            return False
            
        # Длинные строки, содержащие цифры и буквы, могут быть токенами
        if len(value) >= 8 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value):
            return True
            
        # Слова, содержащие "secret", "key", "token", "password" (без учета регистра) в значении
        lower_value = value.lower()
        sensitive_indicators = ['secret', 'key', 'token', 'password', 'auth']
        if any(indicator in lower_value for indicator in sensitive_indicators):
            return True
            
        # Значения, похожие на пароли (смешанные символы)
        if len(value) >= 6 and any(c in value for c in ['!', '@', '#', '$', '%', '&', '*']):
            return True
            
        return False
    
    def _is_likely_sensitive_value_simple(self, value: str) -> bool:
        """Простая проверка, является ли значение потенциально чувствительным"""
        # Проверяем, соответствует ли значение паттернам для чувствительных данных
        # без создания сложных комбинаций
        for pattern, _ in self.value_patterns:
            # Проверяем, совпадает ли само значение с каким-либо паттерном
            if '(\w+)' in pattern.pattern or '([^\'"]+)' in pattern.pattern:
                # Для паттернов, которые ищут значения после ключей,
                # создаем упрощенный паттерн, который проверяет только значение
                simple_value_pattern = re.sub(r'.*[=:].*?(\(.*\)).*$', r'^\1$', pattern.pattern)
                if simple_value_pattern != pattern.pattern:  # Если паттерн был изменен
                    # Просто проверим, похоже ли значение на токен/пароль
                    if len(value) > 10 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value):
                        # Если значение выглядит как токен (буквы и цифры, длина > 10)
                        return True
                    elif len(value) > 20:
                        # Длинные строки также могут быть токенами
                        return True
        
        # Также проверим, похоже ли значение на JWT токен (содержит точки)
        if '.' in value and len(value) > 20:
            parts = value.split('.')
            if len(parts) >= 2 and all(len(part) > 5 for part in parts):
                return True
        
        return False
    

    def _filter_sensitive_data_in_list(self, data: list) -> list:
        """Фильтрует чувствительные данные в списке"""
        if not isinstance(data, (list, tuple)):
            return data
            
        filtered_list = []
        for item in data:
            if isinstance(item, str):
                # Для строк в списке проверим, является ли строка сама по себе чувствительной
                # Проверим, похожа ли строка на чувствительные данные
                if self._is_value_likely_sensitive(item):
                    filtered_list.append('[FILTERED]')
                else:
                    # Применим обычную фильтрацию
                    filtered_list.append(self._filter_sensitive_data(item))
            elif isinstance(item, dict):
                filtered_list.append(self._filter_sensitive_data_in_dict(item))
            elif isinstance(item, (list, tuple)):
                filtered_list.append(self._filter_sensitive_data_in_list(item))
            else:
                filtered_list.append(item)
        
        return filtered_list

    def _filter_sensitive_data(self, text: str) -> str:
        """Удаляет чувствительные данные из текста"""
        if not isinstance(text, str):
            return str(text)

        filtered_text = text
        # Сначала применяем специальные паттерны, которые заменяют только значения
        for pattern, replacement in self.value_patterns:
            filtered_text = pattern.sub(replacement, filtered_text)

        return filtered_text


class JSONFormatter(logging.Formatter):
    """JSON форматтер для структурированного логирования"""

    def format(self, record):
        """Форматирует запись в JSON"""
        # Для корректной обработки оригинального сообщения используем record.msg напрямую
        original_msg = record.msg
        if isinstance(original_msg, str):
            formatted_message = self._filter_sensitive_data(original_msg)
        elif isinstance(original_msg, dict):
            formatted_message = self._filter_sensitive_data_in_dict(original_msg)
        elif isinstance(original_msg, (list, tuple)):
            formatted_message = self._filter_sensitive_data_in_list(original_msg)
        else:
            formatted_message = self._filter_sensitive_data(str(original_msg))

        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': formatted_message,
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

        # Добавляем exception info если есть
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)

    def _filter_sensitive_data(self, text: str) -> str:
        """Фильтрует чувствительные данные"""
        filter_obj = SensitiveDataFilter()
        return filter_obj._filter_sensitive_data(text)
    
    def _filter_sensitive_data_in_dict(self, data: dict) -> dict:
        """Фильтрует чувствительные данные в словаре"""
        filter_obj = SensitiveDataFilter()
        return filter_obj._filter_sensitive_data_in_dict(data)
    
    def _filter_sensitive_data_in_list(self, data: list) -> list:
        """Фильтрует чувствительные данные в списке"""
        filter_obj = SensitiveDataFilter()
        return filter_obj._filter_sensitive_data_in_list(data)


class RequestContextFilter(logging.Filter):
    """Фильтр для добавления контекста запроса"""

    def __init__(self):
        super().__init__()
        self._context = {}

    def set_context(self, **kwargs):
        """Устанавливает контекст для логирования"""
        self._context.update(kwargs)

    def clear_context(self):
        """Очищает контекст"""
        self._context.clear()

    def filter(self, record):
        """Добавляет контекст к записи лога"""
        for key, value in self._context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


# Глобальный фильтр контекста запроса
request_context_filter = RequestContextFilter()


def get_production_logging_config(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = True
) -> Dict[str, Any]:
    """
    Получить конфигурацию логирования для продакшена

    Args:
        log_level: Уровень логирования
        log_file: Путь к файлу логов (опционально)
        enable_json: Использовать JSON форматтер

    Returns:
        Конфигурация логирования
    """

    # Базовая конфигурация
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'filters': {
            'sensitive_filter': {
                '()': SensitiveDataFilter,
            },
            'request_context': {
                '()': lambda: request_context_filter,
            },
        },
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'json': {
                '()': JSONFormatter,
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'json' if enable_json else 'detailed',
                'filters': ['sensitive_filter', 'request_context'],
                'stream': sys.stdout,
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console'],
                'level': log_level,
                'propagate': True,
            },
            'uvicorn': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False,
            },
            'uvicorn.access': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False,
            },
            'fastapi': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False,
            },
            'sqlalchemy': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False,
            },
        }
    }

    # Добавляем файловый handler если указан путь
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'json' if enable_json else 'detailed',
            'filters': ['sensitive_filter', 'request_context'],
            'filename': log_file,
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
        }

        # Добавляем file handler к root logger
        config['loggers']['']['handlers'].append('file')

    return config


def setup_production_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = True
):
    """
    Настраивает продакшен логирование

    Args:
        log_level: Уровень логирования
        log_file: Путь к файлу логов
        enable_json: Использовать JSON форматтер
    """
    config = get_production_logging_config(log_level, log_file, enable_json)
    logging.config.dictConfig(config)

    # Настраиваем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Добавляем фильтры
    for handler in logger.handlers:
        handler.addFilter(SensitiveDataFilter())
        handler.addFilter(request_context_filter)


def get_request_logger(request_id: Optional[str] = None) -> logging.LoggerAdapter:
    """
    Получить логгер с контекстом запроса

    Args:
        request_id: ID запроса

    Returns:
        LoggerAdapter с контекстом
    """
    logger = logging.getLogger('request')

    if request_id:
        request_context_filter.set_context(request_id=request_id)

    return logging.LoggerAdapter(logger, {'request_id': request_id})


def log_error_with_context(
    error: Exception,
    message: str = "An error occurred",
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **extra_context
):
    """
    Логирует ошибку с полным контекстом

    Args:
        error: Исключение
        message: Сообщение об ошибке
        request_id: ID запроса
        user_id: ID пользователя
        **extra_context: Дополнительный контекст
    """
    logger = logging.getLogger('error')

    # Устанавливаем контекст
    context = {'error_type': type(error).__name__}
    if request_id:
        context['request_id'] = request_id
    if user_id:
        context['user_id'] = user_id
    context.update(extra_context)

    request_context_filter.set_context(**context)

    # Логируем ошибку
    logger.error(f"{message}: {str(error)}", exc_info=True)

    # Очищаем контекст
    request_context_filter.clear_context()


# Middleware для логирования запросов
async def log_requests(request, call_next):
    """Middleware для логирования HTTP запросов"""
    import time
    from fastapi import Request, Response

    start_time = time.time()

    # Получаем request_id из заголовков или генерируем новый
    request_id = getattr(request, 'headers', {}).get('X-Request-ID', f"req_{int(time.time())}_{hash(str(request.url)) % 1000}")

    # Устанавливаем контекст
    request_context_filter.set_context(
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get('User-Agent', 'Unknown')
    )

    logger = logging.getLogger('http')

    try:
        # Логируем начало запроса
        logger.info(f"Request started: {request.method} {request.url}")

        # Выполняем запрос
        response = await call_next(request)

        # Вычисляем время обработки
        process_time = time.time() - start_time

        # Логируем завершение
        logger.info(
            f"Request completed: {response.status_code} in {process_time:.3f}s",
            extra={
                'status_code': response.status_code,
                'process_time': process_time,
                'response_size': getattr(response, 'content_length', 0)
            }
        )

        # Добавляем request_id в заголовки ответа
        if hasattr(response, 'headers'):
            response.headers['X-Request-ID'] = request_id

        return response

    except Exception as e:
        # Логируем ошибку
        process_time = time.time() - start_time
        log_error_with_context(
            e,
            f"Request failed: {request.method} {request.url}",
            request_id=request_id,
            process_time=process_time
        )

        # Очищаем контекст
        request_context_filter.clear_context()

        # Передаем исключение дальше
        raise

    finally:
        # Очищаем контекст в любом случае
        request_context_filter.clear_context()


# Функция для создания безопасного сообщения об ошибке
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


def setup_environment_logging(environment: str = "production"):
    """
    Настройка логирования в зависимости от окружения

    Args:
        environment: Окружение ("development", "staging", "production", "test")
    """
    if environment == "development":
        setup_development_logging()
    elif environment == "staging":
        setup_staging_logging()
    elif environment == "production":
        setup_production_logging()
    elif environment == "test":
        setup_test_logging()
    else:
        setup_production_logging()


def setup_development_logging():
    """Настройка логирования для разработки - подробные логи с цветами"""
    import sys

    # Создание логгера
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Очистка существующих обработчиков
    logger.handlers.clear()

    # Цветной форматтер для консоли
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[36m',  # Cyan
            'INFO': '\033[32m',   # Green
            'WARNING': '\033[33m', # Yellow
            'ERROR': '\033[31m',   # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'     # Reset
        }

        def format(self, record):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
            return super().format(record)

    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # Настройка специфических логгеров для разработки
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.INFO)

    logger.info("Development logging configured with colors and debug level")


def setup_staging_logging():
    """Настройка логирования для staging - умеренная детализация"""
    setup_production_logging(
        log_level="INFO",
        enable_json=True,
        log_file="logs/staging.log"
    )


def setup_test_logging():
    """Настройка логирования для тестов - минимальные логи"""
    # Создание логгера
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # Очистка существующих обработчиков
    logger.handlers.clear()

    # Простой форматтер для тестов
    formatter = logging.Formatter(
        "%(levelname)s - %(name)s - %(message)s"
    )

    # Консольный обработчик только для ошибок
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    # Отключаем все специфические логгеры
    logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
    logging.getLogger("fastapi").setLevel(logging.CRITICAL)
    logging.getLogger("sqlalchemy").setLevel(logging.CRITICAL)

    logger.info("Test logging configured - minimal output")

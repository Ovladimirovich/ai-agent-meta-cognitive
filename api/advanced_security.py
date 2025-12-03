"""
Расширенная система безопасности для AI Агента
Включает продвинутый rate limiting с обнаружением аномалий и архитектурные слои безопасности
"""

import time
import logging
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import secrets
import json
from functools import wraps

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Типы событий безопасности"""
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ANOMALY_DETECTED = "anomaly_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    PROMPT_INJECTION_ATTEMPT = "prompt_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"


@dataclass
class SecurityEvent:
    """Событие безопасности"""
    event_type: SecurityEventType
    severity: str  # low, medium, high, critical
    timestamp: datetime
    client_id: str
    endpoint: str
    details: Dict[str, Any]
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class AdvancedRateLimitRule:
    """Расширенное правило rate limiting"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = 10
    anomaly_threshold: float = 2.0  # Порог аномального поведения (стандартных отклонений)
    max_concurrent_requests: int = 5  # Максимальное количество одновременных запросов
    block_duration: int = 300  # Время блокировки при обнаружении аномалии (секунды)
    sliding_window_size: int = 60  # Размер скользящего окна в секундах


@dataclass
class RateLimitInfo:
    """Информация о rate limit для пользователя/IP"""
    minute_count: int = 0
    hour_count: int = 0
    day_count: int = 0
    burst_count: int = 0
    concurrent_requests: int = 0
    minute_window: float = 0.0
    hour_window: float = 0.0
    day_window: float = 0.0
    burst_window: float = 0.0
    request_times: deque = None  # Скользящее окно для обнаружения аномалий
    blocked_until: float = 0.0  # Время до которого заблокирован
    last_request: float = 0.0
    suspicious_patterns: int = 0  # Количество подозрительных паттернов

    def __post_init__(self):
        if self.request_times is None:
            self.request_times = deque(maxlen=100)  # Храним последние 100 временных меток


class AdvancedRateLimiter:
    """
    Расширенный rate limiter с обнаружением аномалий
    """

    def __init__(self, use_redis: bool = False):
        self._limits: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        self._rules: Dict[str, AdvancedRateLimitRule] = {}
        self._anomaly_cache: Dict[str, List[float]] = defaultdict(list)
        self._security_events: List[SecurityEvent] = []
        self.use_redis = use_redis
        self.redis_client = None

        if use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host='localhost',  # Конфигурируется через env
                    port=6379,
                    decode_responses=True
                )
                logger.info("✅ Redis connection established for rate limiting")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Redis: {e}, falling back to in-memory")
                self.use_redis = False

    def set_rule(self, endpoint: str, rule: AdvancedRateLimitRule):
        """Установить правило для эндпоинта"""
        self._rules[endpoint] = rule
        logger.info(f"Set advanced rate limit rule for {endpoint}: "
                   f"{rule.requests_per_minute}/min, {rule.requests_per_hour}/hour, "
                   f"{rule.requests_per_day}/day, burst: {rule.burst_limit}")

    def is_allowed(self, key: str, endpoint: str, request_ip: str = None) -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        Проверить, разрешен ли запрос с обнаружением аномалий

        Args:
            key: Ключ (IP или user_id)
            endpoint: Эндпоинт
            request_ip: IP адрес запроса

        Returns:
            Tuple[bool, Optional[Dict]]: (разрешен, headers с лимитами)
        """
        rule = self._rules.get(endpoint)
        if not rule:
            return True, None  # Нет ограничений

        now = time.time()
        info = self._limits[key]

        # Проверка времени блокировки
        if now < info.blocked_until:
            self._log_security_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                "high",
                key,
                endpoint,
                {"reason": "still_blocked", "remaining_time": info.blocked_until - now},
                ip_address=request_ip
            )
            return False, self._get_headers(info, rule, "blocked")

        # Сброс счетчиков по окнам
        if now - info.minute_window >= 60:
            info.minute_count = 0
            info.minute_window = now

        if now - info.hour_window >= 3600:
            info.hour_count = 0
            info.hour_window = now

        if now - info.day_window >= 86400:  # 24 часа
            info.day_count = 0
            info.day_window = now

        if now - info.burst_window >= 1:  # 1 секунда для burst
            info.burst_count = 0
            info.burst_window = now

        # Проверка лимитов
        if info.minute_count >= rule.requests_per_minute:
            self._check_anomaly_and_block(key, endpoint, rule, info, request_ip)
            return False, self._get_headers(info, rule, "minute")

        if info.hour_count >= rule.requests_per_hour:
            self._check_anomaly_and_block(key, endpoint, rule, info, request_ip)
            return False, self._get_headers(info, rule, "hour")

        if info.day_count >= rule.requests_per_day:
            self._check_anomaly_and_block(key, endpoint, rule, info, request_ip)
            return False, self._get_headers(info, rule, "day")

        if info.burst_count >= rule.burst_limit:
            self._check_anomaly_and_block(key, endpoint, rule, info, request_ip)
            return False, self._get_headers(info, rule, "burst")

        # Проверка аномального поведения
        if self._detect_anomaly(key, endpoint, now, rule, info, request_ip):
            self._check_anomaly_and_block(key, endpoint, rule, info, request_ip)
            return False, self._get_headers(info, rule, "anomaly")

        # Увеличение счетчиков
        info.minute_count += 1
        info.hour_count += 1
        info.day_count += 1
        info.burst_count += 1
        info.concurrent_requests += 1
        info.last_request = now

        # Добавляем временные метки для анализа аномалий
        info.request_times.append(now)

        return True, self._get_headers(info, rule, "ok")

    def _detect_anomaly(self, key: str, endpoint: str, now: float, rule: AdvancedRateLimitRule, info: RateLimitInfo, request_ip: str = None) -> bool:
        """Обнаружение аномальных паттернов запросов"""
        if len(info.request_times) < 5:  # Нужно минимум 5 запросов для анализа
            return False

        # Вычисление интервалов между запросами
        intervals = []
        times_list = list(info.request_times)
        for i in range(1, len(times_list)):
            intervals.append(times_list[i] - times_list[i-1])

        if not intervals:
            return False

        # Статистический анализ
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5

        # Если последний интервал значительно меньше среднего (аномальное поведение)
        if intervals and intervals[-1] < avg_interval - (rule.anomaly_threshold * std_dev) and avg_interval > 0.1:
            logger.warning(f"Anomaly detected for {key} on {endpoint}: rapid request pattern (avg_interval: {avg_interval:.3f}s, std_dev: {std_dev:.3f}s)")
            self._log_security_event(
                SecurityEventType.ANOMALY_DETECTED,
                "high",
                key,
                endpoint,
                {
                    "anomaly_type": "rapid_requests",
                    "avg_interval": avg_interval,
                    "std_dev": std_dev,
                    "last_interval": intervals[-1],
                    "total_requests": len(info.request_times)
                },
                ip_address=request_ip
            )
            return True

        # Проверка на аномально высокую частоту за короткий промежуток
        recent_window = 10  # последние 10 секунд
        recent_requests = [t for t in info.request_times if now - t <= recent_window]
        if len(recent_requests) > rule.requests_per_minute * recent_window / 60 * 1.5:  # 1.5x превышение ожидаемого
            logger.warning(f"Anomaly detected for {key} on {endpoint}: high frequency in {recent_window}s ({len(recent_requests)} requests)")
            self._log_security_event(
                SecurityEventType.ANOMALY_DETECTED,
                "high",
                key,
                endpoint,
                {
                    "anomaly_type": "high_frequency",
                    "time_window": recent_window,
                    "requests_count": len(recent_requests),
                    "expected_rate": rule.requests_per_minute * recent_window / 60
                },
                ip_address=request_ip
            )
            return True

        return False

    def _check_anomaly_and_block(self, key: str, endpoint: str, rule: AdvancedRateLimitRule, info: RateLimitInfo, request_ip: str = None):
        """Проверить аномалии и при необходимости заблокировать пользователя"""
        info.suspicious_patterns += 1

        # Если слишком много подозрительных паттернов, блокируем на время
        if info.suspicious_patterns >= 3:
            info.blocked_until = time.time() + rule.block_duration
            logger.warning(f"Blocking {key} for {rule.block_duration}s due to multiple anomalies")
            self._log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                "critical",
                key,
                endpoint,
                {
                    "reason": "multiple_anomalies",
                    "suspicious_patterns_count": info.suspicious_patterns,
                    "block_duration": rule.block_duration
                },
                ip_address=request_ip
            )

    def _get_headers(self, info: RateLimitInfo, rule: AdvancedRateLimitRule, status: str) -> Dict[str, int]:
        """Получить headers для ответа"""
        remaining_minute = max(0, rule.requests_per_minute - info.minute_count)
        remaining_hour = max(0, rule.requests_per_hour - info.hour_count)
        remaining_day = max(0, rule.requests_per_day - info.day_count)
        remaining_burst = max(0, rule.burst_limit - info.burst_count)

        return {
            "X-RateLimit-Limit-Minute": rule.requests_per_minute,
            "X-RateLimit-Remaining-Minute": remaining_minute,
            "X-RateLimit-Limit-Hour": rule.requests_per_hour,
            "X-RateLimit-Remaining-Hour": remaining_hour,
            "X-RateLimit-Limit-Day": rule.requests_per_day,
            "X-RateLimit-Remaining-Day": remaining_day,
            "X-RateLimit-Limit-Burst": rule.burst_limit,
            "X-RateLimit-Remaining-Burst": remaining_burst,
        }

    def _log_security_event(self, event_type: SecurityEventType, severity: str, client_id: str, endpoint: str,
                           details: Dict[str, Any], user_agent: str = None, ip_address: str = None, session_id: str = None):
        """Логирование события безопасности"""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            client_id=client_id,
            endpoint=endpoint,
            details=details,
            user_agent=user_agent,
            ip_address=ip_address,
            session_id=session_id
        )
        self._security_events.append(event)

        # Логируем в основной логгер
        logger.warning(f"SECURITY EVENT: {event_type.value} - {severity} - {client_id} - {endpoint} - {details}")

        # Здесь можно добавить отправку в систему мониторинга безопасности


class InputValidator:
    """Система валидации и санитизации входных данных"""

    def __init__(self):
        # Паттерны для обнаружения prompt injection
        self.injection_patterns = [
            r'(?i)\b(ignore|disregard|forget|override|bypass)\b.*\b(previous|above|following|instructions|rules)\b',
            r'(?i)\b(system|role|function)\b.*[:=].*\b(assistant|user|system)\b',
            r'(?i)\[system\]|\[user\]|\[assistant\]',
            r'(?i)###\s*system\s*:',  # Markdown стили для системных сообщений
            r'(?i)```\s*(javascript|python|bash|sql)',  # Попытки выполнить код
            r'(?i)<script[^>]*>.*?</script>',  # XSS попытки
            r'(?i)javascript:',
            r'(?i)on\w+\s*=',
        ]

        # Разрешенные теги и атрибуты для безопасного HTML
        self.allowed_html_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
        ]
        self.allowed_html_attrs = {
            '*': ['class'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title']
        }

        self.max_query_length = 10000  # Максимальная длина запроса
        self.max_context_depth = 10    # Максимальная глубина контекста
        self.max_nested_objects = 5    # Максимальное количество вложенных объектов

    async def validate_and_sanitize(self, query: str, context: Optional[Dict] = None) -> Tuple[str, List[str]]:
        """
        Валидация и санитизация входного запроса

        Args:
            query: Входной запрос
            context: Контекст запроса

        Returns:
            Tuple[valid_query, warnings]
        """
        warnings = []

        # Проверка длины запроса
        if len(query) > self.max_query_length:
            warnings.append(f"Query too long (max {self.max_query_length} characters)")
            query = query[:self.max_query_length]

        # Проверка на prompt injection
        injection_detected, patterns = self._detect_prompt_injection(query)
        if injection_detected:
            warnings.append(f"Prompt injection detected: {patterns}")
            query = self._sanitize_prompt(query)

        # Санитизация HTML если присутствует
        query = self._sanitize_html(query)

        # Валидация контекста
        if context:
            context_warnings = self._validate_context(context)
            warnings.extend(context_warnings)

        return query, warnings

    def _detect_prompt_injection(self, prompt: str) -> Tuple[bool, List[str]]:
        """Обнаружение потенциальных prompt injection атак"""
        detected_patterns = []

        for pattern in self.injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE | re.DOTALL):
                detected_patterns.append(pattern)

        return len(detected_patterns) > 0, detected_patterns

    def _sanitize_prompt(self, prompt: str) -> str:
        """Санитизация prompt для предотвращения инъекций"""
        sanitized = prompt

        # Удаление потенциальных команд системы
        sanitized = re.sub(r'(?i)\[system\].*?\[/system\]', '[REDACTED_SYSTEM]', sanitized, flags=re.DOTALL)
        sanitized = re.sub(r'(?i)\[user\].*?\[/user\]', '[REDACTED_USER]', sanitized, flags=re.DOTALL)
        sanitized = re.sub(r'(?i)\[assistant\].*?\[/assistant\]', '[REDACTED_ASSISTANT]', sanitized, flags=re.DOTALL)

        # Удаление потенциальных команд
        sanitized = re.sub(r'(?i)(ignore|disregard|forget|override|bypass)\s+(previous|above|following|instructions|rules)',
                          '[COMMAND_REMOVED]', sanitized)

        # Ограничение длины потенциальных кодовых блоков
        def replace_code_blocks(match):
            code_content = match.group(0)
            if len(code_content) > 1000:  # Ограничение длины кода
                return f"[CODE_BLOCK_TOO_LONG: {len(code_content)} chars removed]"
            return code_content

        # Обработка потенциальных кодовых блоков
        sanitized = re.sub(r'```[\s\S]*?```', replace_code_blocks, sanitized)
        sanitized = re.sub(r'`[^`]{50,}`', '[LONG_CODE_REMOVED]', sanitized)

        return sanitized

    def _sanitize_html(self, text: str) -> str:
        """Базовая санитизация HTML"""
        # Проверка на XSS паттерны
        is_xss_detected, xss_patterns = self._detect_xss(text)
        if is_xss_detected:
            logger.warning(f"XSS attempt detected: {xss_patterns}")
            # Удаление подозрительных паттернов
            for pattern in xss_patterns:
                text = re.sub(pattern, '[REMOVED_XSS_CONTENT]', text, flags=re.IGNORECASE | re.DOTALL)

        # Простая очистка HTML
        return self._simple_html_sanitize(text)

    def _detect_xss(self, text: str) -> Tuple[bool, List[str]]:
        """Обнаружение потенциальных XSS атак"""
        xss_patterns = [
            r'(?i)<script[^>]*>.*?</script>',
            r'(?i)javascript:',
            r'(?i)on\w+\s*=',
            r'(?i)<iframe[^>]*>',
            r'(?i)<object[^>]*>',
            r'(?i)<embed[^>]*>',
            r'(?i)<form[^>]*>',
            r'(?i)vbscript:',
            r'(?i)data:',
        ]
        detected_patterns = []

        for pattern in xss_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                detected_patterns.append(pattern)

        return len(detected_patterns) > 0, detected_patterns

    def _simple_html_sanitize(self, text: str) -> str:
        """Простая санитизация HTML"""
        # Удаляем потенциально опасные теги
        dangerous_tags = ['script', 'iframe', 'object', 'embed', 'form']
        for tag in dangerous_tags:
            text = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(rf'<{tag}[^>]*/>', '', text, flags=re.IGNORECASE)  # Self-closing tags

        # Удаляем потенциально опасные атрибуты
        dangerous_attrs = ['on\w+', 'href', 'src']  # Пример, более детализированная очистка
        for attr in dangerous_attrs:
            text = re.sub(rf'\b{attr}\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)

        return text

    def _validate_context(self, context: Dict) -> List[str]:
        """Валидация контекста запроса"""
        warnings = []
        context_str = json.dumps(context, default=str)

        # Проверка на prompt injection в контексте
        injection_detected, patterns = self._detect_prompt_injection(context_str)
        if injection_detected:
            warnings.append(f"Prompt injection detected in context: {patterns}")

        # Проверка на XSS в контексте
        xss_detected, patterns = self._detect_xss(context_str)
        if xss_detected:
            warnings.append(f"XSS attempt detected in context: {patterns}")

        return warnings


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware для расширенной безопасности"""

    def __init__(self, app, rate_limiter: AdvancedRateLimiter, validator: InputValidator):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.validator = validator

    async def dispatch(self, request: Request, call_next):
        # Получение ключа для rate limiting (IP или user_id)
        client_ip = self._get_client_ip(request)
        user_id = getattr(request.state, 'user_id', None) if hasattr(request, 'state') else None

        # Используем user_id если доступен, иначе IP
        key = user_id or client_ip
        endpoint = request.url.path

        # Проверка rate limit
        allowed, headers = self.rate_limiter.is_allowed(key, endpoint, request_ip=client_ip)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {key} on {endpoint}")
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": 60 # секунд
                }
            )
            if headers:
                for header, value in headers.items():
                    response.headers[header] = str(value)
            return response

        # Валидация и санитизация тела запроса если это POST/PUT запрос
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
                if "query" in body:
                    sanitized_query, warnings = await self.validator.validate_and_sanitize(body["query"])
                    if warnings:
                        logger.warning(f"Security warnings for {key}: {warnings}")

                    # Заменяем оригинальный запрос на санитизированный
                    request._body = json.dumps({**body, "query": sanitized_query}).encode()
            except Exception as e:
                logger.error(f"Error validating request body: {e}")

        # Продолжение обработки запроса
        response = await call_next(request)

        # Добавление headers с информацией о rate limit
        if headers:
            for header, value in headers.items():
                response.headers[header] = str(value)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Получение IP адреса клиента"""
        # Проверка X-Forwarded-For для прокси
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Проверка X-Real-IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Стандартный client IP
        return request.client.host if request.client else "unknown"


# Глобальный экземпляр расширенного rate limiter
advanced_rate_limiter = AdvancedRateLimiter()

# Настройка стандартных правил
def setup_advanced_rate_limits():
    """Настройка стандартных правил расширенного rate limiting"""

    # Агент API - более строгие лимиты
    agent_rule = AdvancedRateLimitRule(
        requests_per_minute=30,   # 30 запросов в минуту
        requests_per_hour=500,    # 500 в час
        requests_per_day=2000,    # 2000 в день
        burst_limit=5,            # 5 в секунду
        anomaly_threshold=1.5,    # Порог аномального поведения
        max_concurrent_requests=3, # Максимум 3 одновременных запроса
        block_duration=600        # Блокировка на 10 минут при аномалии
    )
    advanced_rate_limiter.set_rule("/agent/process", agent_rule)
    advanced_rate_limiter.set_rule("/agent/process-meta", agent_rule)

    # Мета API - средние лимиты
    meta_rule = AdvancedRateLimitRule(
        requests_per_minute=20,
        requests_per_hour=300,
        requests_per_day=1000,
        burst_limit=3,
        anomaly_threshold=2.0,
        max_concurrent_requests=2,
        block_duration=300
    )
    advanced_rate_limiter.set_rule("/learning/metrics", meta_rule)
    advanced_rate_limiter.set_rule("/system/status", meta_rule)
    advanced_rate_limiter.set_rule("/system/info", meta_rule)

    # Health checks - более мягкие лимиты
    health_rule = AdvancedRateLimitRule(
        requests_per_minute=120,
        requests_per_hour=2000,
        requests_per_day=10000,
        burst_limit=20,
        anomaly_threshold=3.0,
        max_concurrent_requests=5,
        block_duration=60
    )
    advanced_rate_limiter.set_rule("/health", health_rule)

    # Остальные эндпоинты - базовые лимиты
    default_rule = AdvancedRateLimitRule(
        requests_per_minute=100,
        requests_per_hour=2000,
        requests_per_day=10000,
        burst_limit=15,
        anomaly_threshold=2.5,
        max_concurrent_requests=5,
        block_duration=120
    )
    advanced_rate_limiter.set_rule("/", default_rule)
    advanced_rate_limiter.set_rule("/system/optimize", default_rule)
    advanced_rate_limiter.set_rule("/debug/logs", default_rule)

    logger.info("✅ Advanced rate limiting rules configured")


# Создание валидатора
input_validator = InputValidator()


def create_security_middleware():
    """Создание middleware для безопасности"""
    return SecurityMiddleware(None, advanced_rate_limiter, input_validator)

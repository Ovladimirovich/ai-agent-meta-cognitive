"""
Rate limiting middleware для API
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

@dataclass
class RateLimitRule:
    """Правило rate limiting"""
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int = 10

@dataclass
class RateLimitInfo:
    """Информация о rate limit для пользователя/IP"""
    minute_count: int = 0
    hour_count: int = 0
    burst_count: int = 0
    minute_window: float = 0.0
    hour_window: float = 0.0
    burst_window: float = 0.0
    last_request: float = 0.0

class InMemoryRateLimiter:
    """
    In-memory rate limiter для простых случаев.
    Для продакшена рекомендуется Redis.
    """

    def __init__(self):
        self._limits: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        self._rules: Dict[str, RateLimitRule] = {}

    def set_rule(self, endpoint: str, rule: RateLimitRule):
        """Установить правило для эндпоинта"""
        self._rules[endpoint] = rule
        logger.info(f"Set rate limit rule for {endpoint}: {rule.requests_per_minute}/min, {rule.requests_per_hour}/hour")

    def is_allowed(self, key: str, endpoint: str) -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        Проверить, разрешен ли запрос

        Args:
            key: Ключ (IP или user_id)
            endpoint: Эндпоинт

        Returns:
            Tuple[bool, Optional[Dict]]: (разрешен, headers с лимитами)
        """
        rule = self._rules.get(endpoint)
        if not rule:
            return True, None  # Нет ограничений

        now = time.time()
        info = self._limits[key]

        # Сброс счетчиков по окнам
        if now - info.minute_window >= 60:
            info.minute_count = 0
            info.minute_window = now

        if now - info.hour_window >= 3600:
            info.hour_count = 0
            info.hour_window = now

        if now - info.burst_window >= 1:  # 1 секунда для burst
            info.burst_count = 0
            info.burst_window = now

        # Проверка лимитов
        if info.minute_count >= rule.requests_per_minute:
            return False, self._get_headers(info, rule, "minute")

        if info.hour_count >= rule.requests_per_hour:
            return False, self._get_headers(info, rule, "hour")

        if info.burst_count >= rule.burst_limit:
            return False, self._get_headers(info, rule, "burst")

        # Увеличение счетчиков
        info.minute_count += 1
        info.hour_count += 1
        info.burst_count += 1
        info.last_request = now

        return True, self._get_headers(info, rule, "ok")

    def _get_headers(self, info: RateLimitInfo, rule: RateLimitRule, status: str) -> Dict[str, int]:
        """Получить headers для ответа"""
        remaining_minute = max(0, rule.requests_per_minute - info.minute_count)
        remaining_hour = max(0, rule.requests_per_hour - info.hour_count)
        remaining_burst = max(0, rule.burst_limit - info.burst_count)

        return {
            "X-RateLimit-Limit-Minute": rule.requests_per_minute,
            "X-RateLimit-Remaining-Minute": remaining_minute,
            "X-RateLimit-Limit-Hour": rule.requests_per_hour,
            "X-RateLimit-Remaining-Hour": remaining_hour,
            "X-RateLimit-Limit-Burst": rule.burst_limit,
            "X-RateLimit-Remaining-Burst": remaining_burst,
        }

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware для rate limiting
    """

    def __init__(self, app, limiter: InMemoryRateLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        # Получение ключа для rate limiting (IP или user_id)
        client_ip = self._get_client_ip(request)
        user_id = getattr(request.state, 'user_id', None) if hasattr(request, 'state') else None

        # Используем user_id если доступен, иначе IP
        key = user_id or client_ip
        endpoint = request.url.path

        # Проверка rate limit
        allowed, headers = self.limiter.is_allowed(key, endpoint)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {key} on {endpoint}")
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": 60  # секунд
                }
            )
            if headers:
                for header, value in headers.items():
                    response.headers[header] = str(value)
            return response

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

# Глобальный экземпляр rate limiter
rate_limiter = InMemoryRateLimiter()

# Настройка стандартных правил
def setup_default_rate_limits():
    """Настройка стандартных правил rate limiting"""

    # Агент API - более строгие лимиты
    agent_rule = RateLimitRule(
        requests_per_minute=60,  # 60 запросов в минуту
        requests_per_hour=1000,  # 1000 в час
        burst_limit=10  # 10 в секунду
    )
    rate_limiter.set_rule("/agent/process", agent_rule)
    rate_limiter.set_rule("/agent/process-meta", agent_rule)

    # Мета API - средние лимиты
    meta_rule = RateLimitRule(
        requests_per_minute=30,
        requests_per_hour=500,
        burst_limit=5
    )
    rate_limiter.set_rule("/learning/metrics", meta_rule)
    rate_limiter.set_rule("/system/status", meta_rule)
    rate_limiter.set_rule("/system/info", meta_rule)

    # Health checks - более мягкие лимиты
    health_rule = RateLimitRule(
        requests_per_minute=120,
        requests_per_hour=2000,
        burst_limit=20
    )
    rate_limiter.set_rule("/health", health_rule)

    # Остальные эндпоинты - базовые лимиты
    default_rule = RateLimitRule(
        requests_per_minute=100,
        requests_per_hour=2000,
        burst_limit=15
    )
    rate_limiter.set_rule("/", default_rule)
    rate_limiter.set_rule("/system/optimize", default_rule)
    rate_limiter.set_rule("/debug/logs", default_rule)

    logger.info("✅ Default rate limiting rules configured")

# Функция для создания middleware
def create_rate_limit_middleware():
    """Создание middleware для rate limiting"""
    return RateLimitMiddleware(None, rate_limiter)

# Функция для настройки расширенных правил rate limiting
def setup_advanced_rate_limits():
    """Настройка расширенных правил rate limiting для продакшена"""
    # В текущей реализации используем стандартные правила
    # В будущем можно добавить более сложную логику, например:
    # - интеграцию с Redis для распределенного rate limiting
    # - динамическую настройку лимитов через конфигурацию
    # - более сложные правила на основе пользовательских ролей
    setup_default_rate_limits()
    logger.info("✅ Advanced rate limiting rules configured")

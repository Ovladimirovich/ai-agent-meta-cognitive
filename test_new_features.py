#!/usr/bin/env python3
"""
Тестирование новых функций улучшений
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.self_awareness.state_manager import StateManager, StateTransitionResult
from agent.core.models import AgentState
from api.rate_limiter import InMemoryRateLimiter, RateLimitRule
from integrations.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
from api.audit_logger import AuditLogger, AuditEventType, AuditEventSeverity

class TestStateManager:
    """Тесты для улучшенного StateManager"""

    def test_transition_to_safe_success(self):
        """Тест успешного безопасного перехода"""
        manager = StateManager()

        result = manager.transition_to_safe(AgentState.ANALYZING, "test transition")

        assert result.success == True
        assert result.from_state == AgentState.IDLE
        assert result.to_state == AgentState.ANALYZING
        assert result.reason == "test transition"
        assert result.error_message is None

    def test_transition_to_safe_failure(self):
        """Тест неудачного безопасного перехода"""
        manager = StateManager()

        # Попытка недопустимого перехода
        result = manager.transition_to_safe(AgentState.COMPLETED, "invalid transition")

        assert result.success == False
        assert result.from_state == AgentState.IDLE
        assert result.to_state == AgentState.COMPLETED
        assert result.error_message is not None
        assert "Cannot transition" in result.error_message

    def test_backward_compatibility(self):
        """Тест обратной совместимости с transition_to"""
        manager = StateManager()

        # Успешный переход
        assert manager.transition_to(AgentState.ANALYZING) == True

        # Неудачный переход должен бросить исключение
        with pytest.raises(Exception):
            manager.transition_to(AgentState.IDLE)  # Недопустимый переход из ANALYZING

class TestRateLimiter:
    """Тесты для rate limiter"""

    def test_rate_limit_creation(self):
        """Тест создания rate limiter"""
        limiter = InMemoryRateLimiter()

        rule = RateLimitRule(
            requests_per_minute=10,
            requests_per_hour=100,
            burst_limit=5
        )

        limiter.set_rule("/test", rule)

        # Первый запрос должен пройти
        allowed, headers = limiter.is_allowed("user1", "/test")
        assert allowed == True
        assert headers is not None
        assert headers["X-RateLimit-Limit-Minute"] == 10

    def test_rate_limit_exceeded(self):
        """Тест превышения лимита"""
        limiter = InMemoryRateLimiter()

        rule = RateLimitRule(requests_per_minute=2, requests_per_hour=10, burst_limit=1)
        limiter.set_rule("/test", rule)

        # Два запроса должны пройти
        assert limiter.is_allowed("user1", "/test")[0] == True
        assert limiter.is_allowed("user1", "/test")[0] == True

        # Третий должен быть заблокирован
        allowed, headers = limiter.is_allowed("user1", "/test")
        assert allowed == False
        assert headers is not None

class TestCircuitBreaker:
    """Тесты для circuit breaker"""

    def test_circuit_breaker_creation(self):
        """Тест создания circuit breaker"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            timeout=5.0
        )
        cb = CircuitBreaker(config)

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.config.failure_threshold == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Тест успешной работы circuit breaker"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)
        cb = CircuitBreaker(config)

        async def success_func():
            return "success"

        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self):
        """Тест открытия circuit breaker при ошибках"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)
        cb = CircuitBreaker(config)

        async def failure_func():
            raise Exception("Test error")

        # Первая ошибка
        with pytest.raises(Exception):
            await cb.call(failure_func)

        # Вторая ошибка - circuit breaker должен открыться
        with pytest.raises(Exception):
            await cb.call(failure_func)

        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Тест восстановления circuit breaker"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,  # Быстрое восстановление для теста
            success_threshold=1
        )
        cb = CircuitBreaker(config)

        async def failure_func():
            raise Exception("Test error")

        async def success_func():
            return "success"

        # Вызываем ошибки для открытия
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call(failure_func)

        assert cb.state == CircuitBreakerState.OPEN

        # Ждем восстановления
        await asyncio.sleep(0.2)

        # Следующий вызов должен перейти в HALF_OPEN и затем в CLOSED при успехе
        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED

class TestAuditLogger:
    """Тесты для audit logger"""

    @pytest.mark.asyncio
    async def test_audit_logger_creation(self):
        """Тест создания audit logger"""
        logger = AuditLogger(log_file="test_audit.log", buffer_size=10)

        await logger.start()

        # Проверяем статистику
        stats = logger.get_stats()
        assert stats['buffer_size'] == 0
        assert 'events_logged' in stats

        await logger.stop()

        # Очищаем тестовый файл
        if os.path.exists("test_audit.log"):
            os.remove("test_audit.log")

    @pytest.mark.asyncio
    async def test_audit_event_logging(self):
        """Тест логирования audit событий"""
        logger = AuditLogger(log_file="test_audit.log", buffer_size=1)  # Маленький буфер для быстрого сброса

        await logger.start()

        # Логируем событие
        await logger.log(
            event_type=AuditEventType.API_ACCESS,
            severity=AuditEventSeverity.LOW,
            resource="/test",
            action="GET",
            status="success",
            user_id="test_user",
            request_id="test_req"
        )

        # Ждем сброса буфера
        await asyncio.sleep(0.1)

        # Проверяем статистику
        stats = logger.get_stats()
        assert stats['events_logged'] >= 1

        await logger.stop()

        # Очищаем тестовый файл
        if os.path.exists("test_audit.log"):
            os.remove("test_audit.log")

def run_tests():
    """Запуск всех тестов"""
    print("🧪 Запуск тестов новых функций...")

    # Тесты StateManager
    print("\n📋 Тестирование StateManager...")
    state_tests = TestStateManager()
    try:
        state_tests.test_transition_to_safe_success()
        print("✅ transition_to_safe_success - OK")

        state_tests.test_transition_to_safe_failure()
        print("✅ transition_to_safe_failure - OK")

        state_tests.test_backward_compatibility()
        print("✅ backward_compatibility - OK")

    except Exception as e:
        print(f"❌ StateManager tests failed: {e}")

    # Тесты RateLimiter
    print("\n📊 Тестирование RateLimiter...")
    rate_tests = TestRateLimiter()
    try:
        rate_tests.test_rate_limit_creation()
        print("✅ rate_limit_creation - OK")

        rate_tests.test_rate_limit_exceeded()
        print("✅ rate_limit_exceeded - OK")

    except Exception as e:
        print(f"❌ RateLimiter tests failed: {e}")

    # Тесты CircuitBreaker
    print("\n🔄 Тестирование CircuitBreaker...")
    cb_tests = TestCircuitBreaker()
    try:
        cb_tests.test_circuit_breaker_creation()
        print("✅ circuit_breaker_creation - OK")

        asyncio.run(cb_tests.test_circuit_breaker_success())
        print("✅ circuit_breaker_success - OK")

        asyncio.run(cb_tests.test_circuit_breaker_failure())
        print("✅ circuit_breaker_failure - OK")

        asyncio.run(cb_tests.test_circuit_breaker_recovery())
        print("✅ circuit_breaker_recovery - OK")

    except Exception as e:
        print(f"❌ CircuitBreaker tests failed: {e}")

    # Тесты AuditLogger
    print("\n📝 Тестирование AuditLogger...")
    audit_tests = TestAuditLogger()
    try:
        asyncio.run(audit_tests.test_audit_logger_creation())
        print("✅ audit_logger_creation - OK")

        asyncio.run(audit_tests.test_audit_event_logging())
        print("✅ audit_event_logging - OK")

    except Exception as e:
        print(f"❌ AuditLogger tests failed: {e}")

    print("\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    run_tests()

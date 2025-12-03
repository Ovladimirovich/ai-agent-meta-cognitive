"""
Circuit Breaker паттерн для защиты от каскадных сбоев
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Состояния circuit breaker"""
    CLOSED = "closed"      # Нормальная работа
    OPEN = "open"          # Разомкнуто - блокирует запросы
    HALF_OPEN = "half_open"  # Полуразомкнуто - тестирует восстановление

@dataclass
class CircuitBreakerConfig:
    """Конфигурация circuit breaker"""
    failure_threshold: int = 5  # Количество неудач для открытия
    recovery_timeout: float = 60.0  # Время ожидания перед тестированием (сек)
    expected_exception: tuple = (Exception,)  # Ожидаемые исключения
    success_threshold: int = 3  # Количество успехов для закрытия
    timeout: float = 30.0  # Таймаут для запросов (сек)
    name: str = "default"

@dataclass
class CircuitBreakerMetrics:
    """Метрики circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: List[Dict[str, Any]] = field(default_factory=list)

class CircuitBreakerOpenException(Exception):
    """Исключение при разомкнутом circuit breaker"""
    pass

class CircuitBreaker:
    """
    Реализация Circuit Breaker паттерна

    Предотвращает каскадные сбои, блокируя запросы к нестабильному сервису
    и позволяя ему восстановиться.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()

        logger.info(f"🚀 Circuit breaker '{config.name}' initialized in {self.state.value} state")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Выполнить функцию через circuit breaker

        Args:
            func: Функция для выполнения
            *args: Позиционные аргументы
            **kwargs: Именованные аргументы

        Returns:
            Результат выполнения функции

        Raises:
            CircuitBreakerOpenException: Если circuit breaker разомкнут
        """
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if not self._should_attempt_reset():
                    self.metrics.total_requests += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.config.name}' is OPEN"
                    )
                else:
                    self._transition_to(CircuitBreakerState.HALF_OPEN)
                    logger.info(f"🔄 Circuit breaker '{self.config.name}' testing recovery")

            self.metrics.total_requests += 1

        try:
            # Выполняем запрос с таймаутом
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                # Для синхронных функций
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: func(*args, **kwargs)
                )

            await self._on_success()
            return result

        except self.config.expected_exception as e:
            await self._on_failure()
            raise e
        except asyncio.TimeoutError as e:
            await self._on_failure()
            raise e

    def _on_success_sync(self):
        """Обработка успешного запроса (синхронная версия)"""
        self.metrics.successful_requests += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.last_success_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.metrics.consecutive_successes >= self.config.success_threshold:
                self._transition_to(CircuitBreakerState.CLOSED)
                logger.info(f"✅ Circuit breaker '{self.config.name}' recovered and CLOSED")

    def _on_failure_sync(self):
        """Обработка неудачного запроса (синхронная версия)"""
        self.metrics.failed_requests += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            if self.metrics.consecutive_failures >= self.config.failure_threshold:
                self._transition_to(CircuitBreakerState.OPEN)
                logger.warning(f"❌ Circuit breaker '{self.config.name}' OPENED due to {self.metrics.consecutive_failures} consecutive failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self._transition_to(CircuitBreakerState.OPEN)
            logger.warning(f"❌ Circuit breaker '{self.config.name}' failed recovery test, back to OPEN")

    async def _on_success(self):
        """Обработка успешного запроса"""
        async with self._lock:
            self._on_success_sync()

    async def _on_failure(self):
        """Обработка неудачного запроса"""
        async with self._lock:
            self._on_failure_sync()

    def _should_attempt_reset(self) -> bool:
        """Проверить, пора ли пытаться восстановиться"""
        if self.metrics.last_failure_time is None:
            return True

        elapsed = time.time() - self.metrics.last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _transition_to(self, new_state: CircuitBreakerState):
        """Переход в новое состояние"""
        old_state = self.state
        self.state = new_state

        state_change = {
            "timestamp": datetime.now().isoformat(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes
        }
        self.metrics.state_changes.append(state_change)

        logger.info(f"🔄 Circuit breaker '{self.config.name}' state: {old_state.value} → {new_state.value}")

    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики circuit breaker"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "success_rate": (
                    self.metrics.successful_requests / self.metrics.total_requests
                    if self.metrics.total_requests > 0 else 0
                ),
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
                "state_changes_count": len(self.metrics.state_changes)
            },
            "recent_state_changes": self.metrics.state_changes[-5:]  # Последние 5 изменений
        }

    def reset(self):
        """Сброс circuit breaker в начальное состояние"""
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        logger.info(f"🔄 Circuit breaker '{self.config.name}' reset to CLOSED state")

class CircuitBreakerRegistry:
    """
    Реестр circuit breakers для управления множественными экземплярами
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Получить или создать circuit breaker"""
        if name not in self._breakers:
            config.name = name
            self._breakers[name] = CircuitBreaker(config)
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Получить circuit breaker по имени"""
        return self._breakers.get(name)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Получить метрики всех circuit breakers"""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self):
        """Сбросить все circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("🔄 All circuit breakers reset")

# Глобальный реестр
circuit_breaker_registry = CircuitBreakerRegistry()

# Декоратор для применения circuit breaker к функциям
def circuit_breaker_decorator(name: str, config: CircuitBreakerConfig):
    """
    Декоратор для применения circuit breaker к функциям

    Args:
        name: Имя circuit breaker
        config: Конфигурация
    """
    def decorator(func: Callable):
        breaker = circuit_breaker_registry.get_or_create(name, config)

        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            # Для синхронных функций просто вызываем напрямую
            # Circuit breaker для синхронных функций требует дополнительной настройки
            logger.warning(f"Circuit breaker '{name}' not supported for sync functions, calling directly")
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def create_database_circuit_breaker(name: str) -> CircuitBreaker:
    """Создать circuit breaker для базы данных"""
    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        success_threshold=3,
        timeout=5.0,
        name=name
    )
    return circuit_breaker_registry.get_or_create(name, config)

def create_external_service_circuit_breaker(name: str) -> CircuitBreaker:
    """Создать circuit breaker для внешних сервисов"""
    config = CircuitBreakerConfig(
        failure_threshold=2,  # Быстрее открывать для внешних сервисов
        recovery_timeout=120.0,  # Дольше ждать восстановления
        success_threshold=3,
        timeout=15.0,
        name=name
    )
    return circuit_breaker_registry.get_or_create(name, config)

class FallbackStrategy:
    """
    Стратегия плавного ухудшения функциональности
    """
    
    def __init__(self, primary_func: Callable, fallback_func: Callable, 
                 degradation_threshold: float = 0.5):
        self.primary_func = primary_func
        self.fallback_func = fallback_func
        self.degradation_threshold = degradation_threshold
        self.success_rate = 1.0  # Начальный успех
        self.request_count = 0
        self.failure_count = 0
        self.last_check = time.time()
        
    async def call(self, *args, **kwargs) -> Any:
        """
        Вызов с возможностью перехода к fallback стратегии
        """
        # Обновляем статистику каждые 10 запросов или раз в 5 минут
        if (self.request_count % 10 == 0 or 
            time.time() - self.last_check > 300):
            self._update_success_rate()
            self.last_check = time.time()
        
        # Если успех ниже порога, используем fallback
        if self.success_rate < self.degradation_threshold:
            logger.warning(f"Falling back to degraded mode for {self.primary_func.__name__}")
            try:
                return await self._call_with_timeout(self.fallback_func, *args, **kwargs)
            except Exception as e:
                logger.error(f"Fallback function failed: {e}")
                raise
        
        # Используем основную функцию
        try:
            result = await self._call_with_timeout(self.primary_func, *args, **kwargs)
            self.request_count += 1
            return result
        except Exception as e:
            self.request_count += 1
            self.failure_count += 1
            logger.warning(f"Primary function failed: {e}, trying fallback")
            
            # Попробуем fallback
            try:
                return await self._call_with_timeout(self.fallback_func, *args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Both primary and fallback functions failed: {e}, {fallback_error}")
                raise
    
    async def _call_with_timeout(self, func: Callable, *args, timeout: float = 10.0, **kwargs):
        """Вызов функции с таймаутом"""
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: func(*args, **kwargs)
            )
    
    def _update_success_rate(self):
        """Обновление коэффициента успеха"""
        if self.request_count > 0:
            self.success_rate = (self.request_count - self.failure_count) / self.request_count
        else:
            self.success_rate = 1.0

class DegradableService:
    """
    Сервис с поддержкой плавного ухудшения функциональности
    """
    
    def __init__(self, name: str):
        self.name = name
        self.fallback_strategies: Dict[str, FallbackStrategy] = {}
        self.health_score = 1.0
        self.last_degradation = time.time()
        
    def register_fallback(self, operation: str, primary_func: Callable, 
                         fallback_func: Callable, degradation_threshold: float = 0.5):
        """Регистрация операции с возможностью fallback"""
        self.fallback_strategies[operation] = FallbackStrategy(
            primary_func, fallback_func, degradation_threshold
        )
        
    async def execute(self, operation: str, *args, **kwargs):
        """Выполнение операции с возможностью ухудшения"""
        if operation not in self.fallback_strategies:
            raise ValueError(f"Operation {operation} not registered")
            
        try:
            result = await self.fallback_strategies[operation].call(*args, **kwargs)
            self._update_health_score(True)
            return result
        except Exception as e:
            self._update_health_score(False)
            raise e
    
    def _update_health_score(self, success: bool):
        """Обновление оценки здоровья сервиса"""
        if success:
            # Увеличиваем здоровье, но не больше 1.0
            self.health_score = min(1.0, self.health_score + 0.01)
        else:
            # Уменьшаем здоровье, но не меньше 0.0
            self.health_score = max(0.0, self.health_score - 0.05)
    
    def get_health_score(self) -> float:
        """Получить оценку здоровья сервиса"""
        return self.health_score
    
    def is_degraded(self) -> bool:
        """Проверить, находится ли сервис в ухудшенном состоянии"""
        return self.health_score < 0.7

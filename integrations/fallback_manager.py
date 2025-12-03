"""
Менеджер плавного ухудшения функциональности (Graceful Degradation)
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .circuit_breaker import DegradableService

logger = logging.getLogger(__name__)

class DegradationLevel(Enum):
    """Уровни ухудшения функциональности"""
    FULL_FUNCTIONAL = "full_functional"      # Полная функциональность
    PARTIAL_FUNCTIONAL = "partial_functional"  # Частичная функциональность
    MINIMAL_FUNCTIONAL = "minimal_functional"  # Минимальная функциональность
    DEGRADED = "degraded"                    # Ухудшенное состояние
    UNAVAILABLE = "unavailable"              # Недоступно

@dataclass
class DegradationConfig:
    """Конфигурация плавного ухудшения"""
    degradation_threshold: float = 0.5  # Порог ухудшения (0.0-1.0)
    minimal_threshold: float = 0.2      # Порог минимальной функциональности
    recovery_threshold: float = 0.8     # Порог восстановления
    check_interval: float = 30.0        # Интервал проверки состояния (сек)
    max_degradation_steps: int = 3      # Максимальное количество шагов ухудшения
    name: str = "default"

@dataclass
class DegradationMetrics:
    """Метрики ухудшения функциональности"""
    health_score: float = 1.0
    degradation_level: DegradationLevel = DegradationLevel.FULL_FUNCTIONAL
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    degraded_requests: int = 0
    last_degradation_time: Optional[float] = None
    last_recovery_time: Optional[float] = None
    degradation_history: List[Dict[str, Any]] = field(default_factory=list)

class GracefulDegradationManager:
    """
    Менеджер плавного ухудшения функциональности
    
    Обеспечивает постепенное снижение функциональности при ухудшении состояния системы
    и восстановление функциональности при улучшении состояния.
    """
    
    def __init__(self, config: DegradationConfig):
        self.config = config
        self.metrics = DegradationMetrics()
        self.degradable_services: Dict[str, DegradableService] = {}
        self.degradation_strategies: Dict[str, List[Callable]] = {}
        self._last_check_time = time.time()
        self._lock = asyncio.Lock()
        
        logger.info(f"🚀 Graceful degradation manager '{config.name}' initialized")
    
    def register_service(self, service_name: str, degradable_service: DegradableService):
        """Регистрация сервиса с поддержкой ухудшения"""
        self.degradable_services[service_name] = degradable_service
        logger.info(f"Registered degradable service: {service_name}")
    
    def register_degradation_strategy(self, service_name: str, strategies: List[Callable]):
        """Регистрация стратегий ухудшения для сервиса"""
        self.degradation_strategies[service_name] = strategies
        logger.info(f"Registered {len(strategies)} degradation strategies for {service_name}")
    
    async def execute_with_fallback(self, service_name: str, operation: str, *args, **kwargs):
        """
        Выполнение операции с возможностью плавного ухудшения
        
        Args:
            service_name: Имя сервиса
            operation: Имя операции
            *args: Аргументы операции
            **kwargs: Именованные аргументы операции
            
        Returns:
            Результат выполнения операции
        """
        if service_name not in self.degradable_services:
            raise ValueError(f"Service {service_name} not registered as degradable")
        
        async with self._lock:
            # Обновляем метрики
            self.metrics.total_requests += 1
            
            try:
                # Выполняем операцию
                result = await self.degradable_services[service_name].execute(operation, *args, **kwargs)
                
                # Обновляем метрики успеха
                self.metrics.successful_requests += 1
                
                # Проверяем необходимость восстановления
                await self._check_recovery()
                
                return result
            except Exception as e:
                # Обновляем метрики неудачи
                self.metrics.failed_requests += 1
                
                # Пробуем стратегии ухудшения
                fallback_result = await self._try_degradation_strategies(service_name, operation, *args, **kwargs)
                
                if fallback_result is not None:
                    self.metrics.degraded_requests += 1
                    return fallback_result
                else:
                    raise e
    
    async def _try_degradation_strategies(self, service_name: str, operation: str, *args, **kwargs):
        """Попытка выполнения с использованием стратегий ухудшения"""
        if service_name in self.degradation_strategies:
            strategies = self.degradation_strategies[service_name]
            
            for i, strategy in enumerate(strategies):
                try:
                    logger.info(f"Trying degradation strategy {i+1}/{len(strategies)} for {service_name}")
                    result = await strategy(*args, **kwargs)
                    return result
                except Exception as e:
                    logger.warning(f"Degradation strategy {i+1} failed: {e}")
                    continue
        
        return None
    
    async def _check_degradation(self):
        """Проверка необходимости ухудшения функциональности"""
        current_time = time.time()
        
        if current_time - self._last_check_time < self.config.check_interval:
            return
        
        # Рассчитываем текущий уровень здоровья
        health_score = self._calculate_health_score()
        
        # Обновляем уровень ухудшения
        old_level = self.metrics.degradation_level
        new_level = self._determine_degradation_level(health_score)
        
        if new_level != old_level:
            self._transition_degradation_level(old_level, new_level, health_score)
        
        self._last_check_time = current_time
    
    async def _check_recovery(self):
        """Проверка возможности восстановления функциональности"""
        current_time = time.time()
        
        if current_time - self._last_check_time < self.config.check_interval:
            return
        
        # Рассчитываем текущий уровень здоровья
        health_score = self._calculate_health_score()
        
        # Проверяем возможность восстановления
        if (health_score >= self.config.recovery_threshold and 
            self.metrics.degradation_level != DegradationLevel.FULL_FUNCTIONAL):
            
            old_level = self.metrics.degradation_level
            self.metrics.degradation_level = DegradationLevel.FULL_FUNCTIONAL
            self.metrics.health_score = health_score
            self.metrics.last_recovery_time = current_time
            
            logger.info(f"✅ Recovery to full functionality: {old_level.value} → {self.metrics.degradation_level.value}")
            
            # Записываем в историю
            self._add_to_history({
                "timestamp": datetime.now().isoformat(),
                "from_level": old_level.value,
                "to_level": self.metrics.degradation_level.value,
                "health_score": health_score,
                "reason": "recovery"
            })
        
        self._last_check_time = current_time
    
    def _calculate_health_score(self) -> float:
        """Расчет оценки здоровья системы"""
        if self.metrics.total_requests == 0:
            return 1.0
        
        # Базовая оценка на основе соотношения успехов к неудачам
        success_rate = self.metrics.successful_requests / self.metrics.total_requests
        
        # Учитываем среднюю оценку здоровья зарегистрированных сервисов
        if self.degradable_services:
            service_health_scores = [
                service.get_health_score() 
                for service in self.degradable_services.values()
            ]
            avg_service_health = sum(service_health_scores) / len(service_health_scores)
            
            # Комбинируем оценки (70% от успешности запросов, 30% от здоровья сервисов)
            combined_score = (success_rate * 0.7) + (avg_service_health * 0.3)
        else:
            combined_score = success_rate
        
        # Ограничиваем значение в диапазоне [0, 1]
        self.metrics.health_score = max(0.0, min(1.0, combined_score))
        return self.metrics.health_score
    
    def _determine_degradation_level(self, health_score: float) -> DegradationLevel:
        """Определение уровня ухудшения на основе оценки здоровья"""
        if health_score >= 0.9:
            return DegradationLevel.FULL_FUNCTIONAL
        elif health_score >= 0.7:
            return DegradationLevel.PARTIAL_FUNCTIONAL
        elif health_score >= 0.5:
            return DegradationLevel.MINIMAL_FUNCTIONAL
        elif health_score >= 0.3:
            return DegradationLevel.DEGRADED
        else:
            return DegradationLevel.UNAVAILABLE
    
    def _transition_degradation_level(self, old_level: DegradationLevel, 
                                    new_level: DegradationLevel, health_score: float):
        """Переход между уровнями ухудшения"""
        self.metrics.degradation_level = new_level
        self.metrics.health_score = health_score
        self.metrics.last_degradation_time = time.time()
        
        logger.warning(f"⚠️ Degradation level changed: {old_level.value} → {new_level.value} "
                      f"(health: {health_score:.2f})")
        
        # Записываем в историю
        self._add_to_history({
            "timestamp": datetime.now().isoformat(),
            "from_level": old_level.value,
            "to_level": new_level.value,
            "health_score": health_score,
            "reason": "degradation"
        })
    
    def _add_to_history(self, entry: Dict[str, Any]):
        """Добавление записи в историю ухудшения"""
        self.metrics.degradation_history.append(entry)
        
        # Ограничиваем историю последними 50 записями
        if len(self.metrics.degradation_history) > 50:
            self.metrics.degradation_history = self.metrics.degradation_history[-50:]
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Получение статуса ухудшения функциональности"""
        return {
            "service": self.config.name,
            "degradation_level": self.metrics.degradation_level.value,
            "health_score": self.metrics.health_score,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "degraded_requests": self.metrics.degraded_requests,
                "success_rate": (
                    self.metrics.successful_requests / self.metrics.total_requests
                    if self.metrics.total_requests > 0 else 0
                ),
                "last_degradation_time": self.metrics.last_degradation_time,
                "last_recovery_time": self.metrics.last_recovery_time
            },
            "degradation_history": self.metrics.degradation_history[-10:]  # Последние 10 изменений
        }
    
    def force_degradation_level(self, level: DegradationLevel):
        """Принудительная установка уровня ухудшения (для тестирования)"""
        old_level = self.metrics.degradation_level
        if level != old_level:
            self.metrics.degradation_level = level
            self.metrics.last_degradation_time = time.time()
            
            logger.info(f"🔧 Forced degradation level: {old_level.value} → {level.value}")
            
            self._add_to_history({
                "timestamp": datetime.now().isoformat(),
                "from_level": old_level.value,
                "to_level": level.value,
                "health_score": self.metrics.health_score,
                "reason": "forced"
            })

# Глобальный менеджер плавного ухудшения
graceful_degradation_manager = GracefulDegradationManager(
    DegradationConfig(name="global_degradation_manager")
)

def create_degradation_manager(name: str, config: Optional[DegradationConfig] = None) -> GracefulDegradationManager:
    """Создание менеджера плавного ухудшения"""
    if config is None:
        config = DegradationConfig(name=name)
    else:
        config.name = name
    
    return GracefulDegradationManager(config)

# Примеры стратегий ухудшения
async def cache_fallback_strategy(*args, **kwargs):
    """Стратегия: возврат данных из кэша"""
    logger.info("Using cache fallback strategy")
    # Здесь будет реализация получения данных из кэша
    return {"result": "fallback_from_cache", "source": "cache", "degraded": True}

async def simplified_response_strategy(*args, **kwargs):
    """Стратегия: упрощенный ответ"""
    logger.info("Using simplified response strategy")
    # Здесь будет реализация упрощенного ответа
    return {"result": "simplified_response", "source": "simplified", "degraded": True}

async def cached_summary_strategy(*args, **kwargs):
    """Стратегия: возврат кэшированного суммарного ответа"""
    logger.info("Using cached summary strategy")
    # Здесь будет реализация получения кэшированного суммарного ответа
    return {"result": "cached_summary", "source": "summary_cache", "degraded": True}

# Стандартные стратегии ухудшения
STANDARD_DEGRADATION_STRATEGIES = [
    cache_fallback_strategy,
    simplified_response_strategy,
    cached_summary_strategy
]
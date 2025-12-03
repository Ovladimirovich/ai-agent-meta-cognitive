from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from dataclasses import dataclass


class ModelProvider(Enum):
    """Провайдеры AI моделей"""
    QWEN = "qwen"
    GEMINI = "gemini"
    OPENAI = "openai"
    LOCAL = "local"


class ModelCapabilities(Enum):
    """Возможности моделей"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    CREATIVE_WRITING = "creative_writing"
    MATHEMATICS = "mathematics"
    TRANSLATION = "translation"


class ModelConfig(BaseModel):
    """Конфигурация модели"""
    provider: ModelProvider
    model_name: str
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    cost_per_token: float = 0.0
    capabilities: list[ModelCapabilities] = []
    priority: int = 1  # Выше число = выше приоритет
    max_requests_per_minute: int = 60
    context_window: int = 4096


class ModelMetrics(BaseModel):
    """Метрики использования модели"""
    requests_count: int = 0
    tokens_used: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    error_count: int = 0


class ModelHealth(BaseModel):
    """Здоровье модели"""
    is_available: bool = True
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    average_latency: float = 0.0
    quota_remaining: Optional[int] = None


class RoutingDecision(BaseModel):
    """Решение о маршрутизации"""
    selected_provider: ModelProvider
    selected_model: str
    confidence: float
    reasoning: str
    alternatives: list[Dict[str, Any]] = []
    estimated_cost: float = 0.0
    estimated_time: float = 0.0


class FallbackStrategy(Enum):
    """Стратегии fallback"""
    NEXT_PRIORITY = "next_priority"
    SAME_PROVIDER = "same_provider"
    CHEAPEST_AVAILABLE = "cheapest_available"
    FASTEST_AVAILABLE = "fastest_available"


class ModelConstraints(BaseModel):
    """Ограничения модели"""
    max_input_length: int = 4096
    max_output_length: int = 4096
    supported_languages: list[str] = ["en", "ru"]
    content_policies: list[str] = []
    rate_limits: Dict[str, Any] = {}


class ModelInstance(BaseModel):
    """Экземпляр модели с состоянием"""
    config: ModelConfig
    metrics: ModelMetrics = ModelMetrics()
    health: ModelHealth = ModelHealth()
    constraints: ModelConstraints = ModelConstraints()
    is_active: bool = True

    def update_metrics(self, response_time: float, tokens: int, cost: float, success: bool):
        """Обновление метрик после запроса"""
        self.metrics.requests_count += 1
        self.metrics.tokens_used += tokens
        self.metrics.total_cost += cost
        self.metrics.last_used = datetime.now()

        if success:
            # Обновление средней задержки
            if self.metrics.requests_count == 1:
                self.metrics.average_response_time = response_time
            else:
                self.metrics.average_response_time = (
                    (self.metrics.average_response_time * (self.metrics.requests_count - 1)) +
                    response_time
                ) / self.metrics.requests_count

            # Сброс счетчика ошибок при успехе
            self.health.consecutive_failures = 0
            self.metrics.success_rate = (
                (self.metrics.success_rate * (self.metrics.requests_count - 1)) + 1
            ) / self.metrics.requests_count
        else:
            self.metrics.error_count += 1
            self.health.consecutive_failures += 1
            self.metrics.success_rate = (
                (self.metrics.success_rate * (self.metrics.requests_count - 1))
            ) / self.metrics.requests_count

    def is_healthy(self) -> bool:
        """Проверка здоровья модели"""
        if not self.is_active or not self.config.enabled:
            return False

        # Проверка количества последовательных ошибок
        if self.health.consecutive_failures >= 3:
            return False

        # Проверка времени последнего использования (не старше 24 часов)
        if self.metrics.last_used:
            hours_since_last_use = (datetime.now() - self.metrics.last_used).total_seconds() / 3600
            if hours_since_last_use > 24:
                return False

        return self.health.is_available

    def get_effective_priority(self) -> float:
        """Получение эффективного приоритета с учетом здоровья"""
        base_priority = self.config.priority

        # Штраф за ошибки
        error_penalty = self.health.consecutive_failures * 0.1

        # Бонус за недавнее успешное использование
        recency_bonus = 0.0
        if self.metrics.last_used:
            hours_since_last_use = (datetime.now() - self.metrics.last_used).total_seconds() / 3600
            if hours_since_last_use < 1:  # Последний час
                recency_bonus = 0.2

        return max(0.1, base_priority - error_penalty + recency_bonus)


@dataclass
class ModelResponse:
    """Ответ от модели"""
    answer: str
    model_name: str
    provider_used: 'ModelProvider'
    confidence: float
    tokens_used: int
    cost: float
    processing_time: float
    used_fallback: bool = False
    error_message: Optional[str] = None

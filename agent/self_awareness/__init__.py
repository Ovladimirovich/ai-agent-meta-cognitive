"""
Самосознание AI Агента (Фаза 2)

Рефлексия, анализ ошибок, само-диагностика, адаптация и генерация insights.
"""

from .reasoning_trace import ReasoningTraceAnalyzer
from .error_classifier import ErrorClassifier
from .performance_analyzer import PerformanceAnalyzer
from .feedback_processor import FeedbackProcessor, SentimentAnalyzer, FeedbackClassifier
from .self_reflection_engine import SelfReflectionEngine
from .adaptation_engine import AdaptationEngine, StrategyAdaptation, ParameterAdaptation
from .reflection_cycle import ReflectionCycle
from .self_monitoring import SelfMonitoringSystem, ComponentHealthChecker, PerformanceMonitor, CognitiveHealthAssessor

__all__ = [
    # Анализаторы
    "ReasoningTraceAnalyzer",
    "ErrorClassifier",
    "PerformanceAnalyzer",

    # Обработка обратной связи
    "FeedbackProcessor",
    "SentimentAnalyzer",
    "FeedbackClassifier",

    # Рефлексия и адаптация
    "SelfReflectionEngine",
    "AdaptationEngine",
    "StrategyAdaptation",
    "ParameterAdaptation",

    # Циклы и мониторинг
    "ReflectionCycle",
    "SelfMonitoringSystem",
    "ComponentHealthChecker",
    "PerformanceMonitor",
    "CognitiveHealthAssessor",
]

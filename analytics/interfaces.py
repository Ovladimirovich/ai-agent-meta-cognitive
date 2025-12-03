"""
Интерфейсы для аналитического движка, реализующие принципы SOLID
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Результат аналитического анализа"""
    analysis_type: str
    insights: List[str]
    metrics: Dict[str, Any]
    recommendations: List[str]
    confidence: float
    data_quality_score: float
    processing_time: float
    timestamp: datetime


class DataQualityAssessor(ABC):
    """Интерфейс для оценки качества данных"""
    
    @abstractmethod
    def assess(self, data: Any) -> float:
        """Оценка качества данных (0.0 - 1.0)"""
        pass


class TimeSeriesProcessor(ABC):
    """Интерфейс для обработки временных рядов"""
    
    @abstractmethod
    def extract(self, data: Any) -> List[tuple[datetime, float]]:
        """Извлечение временных рядов из данных"""
        pass
    
    @abstractmethod
    def calculate_trend(self, time_series: List[tuple[datetime, float]]) -> tuple[str, float]:
        """Расчет тренда временного ряда"""
        pass
    
    @abstractmethod
    def detect_seasonality(self, time_series: List[tuple[datetime, float]]) -> Optional[str]:
        """Обнаружение сезонности"""
        pass
    
    @abstractmethod
    def forecast(self, time_series: List[tuple[datetime, float]], periods: int) -> Optional[List[float]]:
        """Прогноз"""
        pass


class StatisticalAnalyzer(ABC):
    """Интерфейс для статистического анализа"""
    
    @abstractmethod
    def calculate_basic_statistics(self, numeric_data: List[float]) -> Dict[str, float]:
        """Расчет базовой статистики"""
        pass
    
    @abstractmethod
    def analyze_distribution(self, numeric_data: List[float]) -> Dict[str, Any]:
        """Анализ распределения данных"""
        pass
    
    @abstractmethod
    def detect_outliers(self, numeric_data: List[float]) -> Dict[str, Any]:
        """Обнаружение выбросов"""
        pass


class CorrelationAnalyzer(ABC):
    """Интерфейс для корреляционного анализа"""
    
    @abstractmethod
    def extract_correlation_data(self, data: Any) -> Dict[str, List[float]]:
        """Извлечение данных для корреляции"""
        pass
    
    @abstractmethod
    def calculate_correlation_matrix(self, correlation_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Расчет корреляционной матрицы"""
        pass
    
    @abstractmethod
    def find_strong_correlations(self, corr_matrix: Dict[str, Dict[str, float]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Поиск сильных корреляций"""
        pass
    
    @abstractmethod
    def check_multicollinearity(self, corr_matrix: Dict[str, Dict[str, float]], threshold: float = 0.8) -> Optional[List[List[str]]]:
        """Проверка мультиколлинеарности"""
        pass


class ComparativeAnalyzer(ABC):
    """Интерфейс для сравнительного анализа"""
    
    @abstractmethod
    def extract_comparison_data(self, data: Any) -> Dict[str, Dict[str, Any]]:
        """Извлечение данных для сравнения"""
        pass
    
    @abstractmethod
    def calculate_differences(self, comparison_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Расчет различий между объектами"""
        pass
    
    @abstractmethod
    def test_significance(self, comparison_data: Dict[str, Dict[str, Any]]) -> bool:
        """Тест статистической значимости"""
        pass
    
    @abstractmethod
    def calculate_correlations(self, comparison_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Расчет корреляций"""
        pass


class PerformanceAnalyzer(ABC):
    """Интерфейс для анализа производительности"""
    
    @abstractmethod
    def extract_performance_data(self, data: Any) -> Dict[str, Any]:
        """Извлечение данных производительности"""
        pass
    
    @abstractmethod
    def calculate_efficiency_score(self, performance_data: Dict[str, Any]) -> float:
        """Расчет оценки эффективности"""
        pass
    
    @abstractmethod
    def identify_bottlenecks(self, performance_data: Dict[str, Any]) -> List[str]:
        """Идентификация bottleneck'ов"""
        pass
    
    @abstractmethod
    def analyze_performance_trend(self, performance_data: Dict[str, Any]) -> Optional[str]:
        """Анализ тренда производительности"""
        pass


class DataProcessor(ABC):
    """Интерфейс для обработки данных"""
    
    @abstractmethod
    def extract_numeric_data(self, data: Any) -> List[float]:
        """Извлечение числовых данных"""
        pass
    
    @abstractmethod
    def summarize_data(self, data: Any) -> Dict[str, Any]:
        """Общая сводка данных"""
        pass
    
    @abstractmethod
    def identify_data_types(self, data: Any) -> Dict[str, str]:
        """Идентификация типов данных"""
        pass
    
    @abstractmethod
    def check_completeness(self, data: Any) -> float:
        """Проверка полноты данных (0.0 - 1.0)"""
        pass


class AnalysisCache(ABC):
    """Интерфейс для кэширования результатов анализа"""
    
    @abstractmethod
    async def get(self, cache_key: str) -> Optional[AnalysisResult]:
        """Получение результата из кэша"""
        pass
    
    @abstractmethod
    async def set(self, cache_key: str, result: AnalysisResult) -> None:
        """Сохранение результата в кэш"""
        pass
    
    @abstractmethod
    async def clear_expired(self) -> None:
        """Очистка устаревших записей"""
        pass
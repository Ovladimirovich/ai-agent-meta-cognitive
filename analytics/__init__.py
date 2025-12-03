"""
Модуль аналитики с внедрением принципов SOLID
"""

from .interfaces import (
    AnalysisResult, DataQualityAssessor, TimeSeriesProcessor, StatisticalAnalyzer,
    CorrelationAnalyzer, ComparativeAnalyzer, PerformanceAnalyzer, DataProcessor, AnalysisCache
)
from .implementations import (
    BasicDataQualityAssessor, BasicTimeSeriesProcessor, BasicStatisticalAnalyzer,
    BasicCorrelationAnalyzer, BasicComparativeAnalyzer, BasicPerformanceAnalyzer,
    BasicDataProcessor, InMemoryAnalysisCache
)
from .analytics_engine import AnalyticsEngine
from .async_analytics_engine import AsyncAnalyticsEngine

__all__ = [
    # Интерфейсы
    'AnalysisResult', 'DataQualityAssessor', 'TimeSeriesProcessor', 'StatisticalAnalyzer',
    'CorrelationAnalyzer', 'ComparativeAnalyzer', 'PerformanceAnalyzer', 'DataProcessor', 'AnalysisCache',
    
    # Реализации
    'BasicDataQualityAssessor', 'BasicTimeSeriesProcessor', 'BasicStatisticalAnalyzer',
    'BasicCorrelationAnalyzer', 'BasicComparativeAnalyzer', 'BasicPerformanceAnalyzer',
    'BasicDataProcessor', 'InMemoryAnalysisCache',
    
    # Движки
    'AnalyticsEngine', 'AsyncAnalyticsEngine'
]

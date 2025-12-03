"""
Реализации интерфейсов аналитического движка, реализующие принципы SOLID
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .interfaces import (
    AnalysisResult, DataQualityAssessor, TimeSeriesProcessor, StatisticalAnalyzer,
    CorrelationAnalyzer, ComparativeAnalyzer, PerformanceAnalyzer, DataProcessor, AnalysisCache
)

logger = logging.getLogger(__name__)


class BasicDataQualityAssessor(DataQualityAssessor):
    """Базовая реализация оценки качества данных"""
    
    def assess(self, data: Any) -> float:
        """Оценка качества данных (0.0 - 1.0)"""
        try:
            score = 1.0

            # Проверка на None/пустые значения
            if data is None:
                return 0.0

            # Для списков/массивов
            if isinstance(data, (list, tuple)):
                if not data:
                    return 0.0
                none_count = sum(1 for x in data if x is None)
                score -= (none_count / len(data)) * 0.5

            # Для словарей
            elif isinstance(data, dict):
                if not data:
                    return 0.0
                none_values = sum(1 for v in data.values() if v is None)
                score -= (none_values / len(data)) * 0.5

            # Проверка на корректность типов
            # (дополнительные проверки могут быть добавлены)

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.0


class BasicTimeSeriesProcessor(TimeSeriesProcessor):
    """Базовая реализация обработки временных рядов"""
    
    def extract(self, data: Any) -> List[Tuple[datetime, float]]:
        """Извлечение временных рядов из данных"""
        # Заглушка - в реальности нужно парсить различные форматы данных
        return []

    def calculate_trend(self, time_series: List[Tuple[datetime, float]]) -> Tuple[str, float]:
        """Расчет тренда временного ряда"""
        if len(time_series) < 3:
            return 'insufficient_data', 0.0

        # Простой линейный тренд
        values = [v for _, v in time_series]
        n = len(values)
        x = list(range(n))

        # Линейная регрессия
        slope = np.polyfit(x, values, 1)[0]

        # Определение направления
        if slope > 0.01:
            direction = 'upward'
        elif slope < -0.01:
            direction = 'downward'
        else:
            direction = 'stable'

        # Сила тренда (коэффициент детерминации)
        y_pred = [slope * i for i in x]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(values, y_pred))
        ss_tot = sum((y - statistics.mean(values)) ** 2 for y in values)
        strength = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return direction, strength

    def detect_seasonality(self, time_series: List[Tuple[datetime, float]]) -> Optional[str]:
        """Обнаружение сезонности"""
        # Заглушка - в реальности нужен более сложный анализ
        return None

    def forecast(self, time_series: List[Tuple[datetime, float]], periods: int) -> Optional[List[float]]:
        """Простой прогноз"""
        if len(time_series) < 3:
            return None

        values = [v for _, v in time_series]
        slope = np.polyfit(range(len(values)), values, 1)[0]

        last_value = values[-1]
        forecast = []
        for i in range(1, periods + 1):
            forecast.append(last_value + slope * i)

        return forecast


class BasicStatisticalAnalyzer(StatisticalAnalyzer):
    """Базовая реализация статистического анализа"""
    
    def calculate_basic_statistics(self, numeric_data: List[float]) -> Dict[str, float]:
        """Расчет базовой статистики"""
        if not numeric_data:
            return {}

        return {
            'mean': statistics.mean(numeric_data),
            'median': statistics.median(numeric_data),
            'std_dev': statistics.stdev(numeric_data) if len(numeric_data) > 1 else 0,
            'min': min(numeric_data),
            'max': max(numeric_data),
            'range': max(numeric_data) - min(numeric_data)
        }

    def analyze_distribution(self, numeric_data: List[float]) -> Dict[str, Any]:
        """Анализ распределения данных"""
        # Заглушка
        return {}

    def detect_outliers(self, numeric_data: List[float]) -> Dict[str, Any]:
        """Обнаружение выбросов"""
        if len(numeric_data) < 4:
            return {'count': 0, 'outliers': []}

        # Простой метод IQR
        sorted_data = sorted(numeric_data)
        q1 = np.percentile(sorted_data, 25)
        q3 = np.percentile(sorted_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [x for x in numeric_data if x < lower_bound or x > upper_bound]

        return {
            'count': len(outliers),
            'outliers': outliers,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }


class BasicCorrelationAnalyzer(CorrelationAnalyzer):
    """Базовая реализация корреляционного анализа"""
    
    def extract_correlation_data(self, data: Any) -> Dict[str, List[float]]:
        """Извлечение данных для корреляции"""
        # Заглушка
        return {}

    def calculate_correlation_matrix(self, correlation_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Расчет корреляционной матрицы"""
        variables = list(correlation_data.keys())
        matrix = {}

        for var1 in variables:
            matrix[var1] = {}
            for var2 in variables:
                if var1 == var2:
                    matrix[var1][var2] = 1.0
                else:
                    try:
                        corr = np.corrcoef(correlation_data[var1], correlation_data[var2])[0, 1]
                        matrix[var1][var2] = float(corr)
                    except:
                        matrix[var1][var2] = 0.0

        return matrix

    def find_strong_correlations(self, corr_matrix: Dict[str, Dict[str, float]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Поиск сильных корреляций"""
        strong_corr = []

        variables = list(corr_matrix.keys())
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Избегать дублирования
                    corr = abs(corr_matrix[var1][var2])
                    if corr >= threshold:
                        strong_corr.append({
                            'variables': [var1, var2],
                            'coefficient': corr_matrix[var1][var2],
                            'strength': 'strong' if corr > 0.7 else 'moderate'
                        })

        # Сортировка по силе корреляции
        strong_corr.sort(key=lambda x: abs(x['coefficient']), reverse=True)
        return strong_corr

    def check_multicollinearity(self, corr_matrix: Dict[str, Dict[str, float]], threshold: float = 0.8) -> Optional[List[List[str]]]:
        """Проверка мультиколлинеарности"""
        multicollinear_groups = []
        variables = list(corr_matrix.keys())

        # Поиск групп сильно коррелированных переменных
        for var1 in variables:
            group = [var1]
            for var2 in variables:
                if var1 != var2 and abs(corr_matrix[var1][var2]) > threshold:
                    group.append(var2)

            if len(group) > 2 and group not in multicollinear_groups:
                multicollinear_groups.append(group)

        return multicollinear_groups if multicollinear_groups else None


class BasicComparativeAnalyzer(ComparativeAnalyzer):
    """Базовая реализация сравнительного анализа"""
    
    def extract_comparison_data(self, data: Any) -> Dict[str, Dict[str, Any]]:
        """Извлечение данных для сравнения"""
        # Заглушка
        return {}

    def calculate_differences(self, comparison_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Расчет различий между объектами"""
        # Заглушка
        return {}

    def test_significance(self, comparison_data: Dict[str, Dict[str, Any]]) -> bool:
        """Тест статистической значимости"""
        # Заглушка
        return False

    def calculate_correlations(self, comparison_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Расчет корреляций"""
        # Заглушка
        return {}


class BasicPerformanceAnalyzer(PerformanceAnalyzer):
    """Базовая реализация анализа производительности"""
    
    def extract_performance_data(self, data: Any) -> Dict[str, Any]:
        """Извлечение данных производительности"""
        # Заглушка
        return {}

    def calculate_efficiency_score(self, performance_data: Dict[str, Any]) -> float:
        """Расчет оценки эффективности"""
        # Заглушка
        return 0.5

    def identify_bottlenecks(self, performance_data: Dict[str, Any]) -> List[str]:
        """Идентификация bottleneck'ов"""
        # Заглушка
        return []

    def analyze_performance_trend(self, performance_data: Dict[str, Any]) -> Optional[str]:
        """Анализ тренда производительности"""
        # Заглушка
        return None


class BasicDataProcessor(DataProcessor):
    """Базовая реализация обработки данных"""
    
    def extract_numeric_data(self, data: Any) -> List[float]:
        """Извлечение числовых данных"""
        # Заглушка
        return []

    def summarize_data(self, data: Any) -> Dict[str, Any]:
        """Общая сводка данных"""
        summary = {'total_elements': 0}

        try:
            if isinstance(data, (list, tuple)):
                summary['total_elements'] = len(data)
                summary['type'] = 'array'
            elif isinstance(data, dict):
                summary['total_elements'] = len(data)
                summary['type'] = 'object'
                summary['keys'] = list(data.keys())
            elif isinstance(data, str):
                summary['total_elements'] = len(data)
                summary['type'] = 'string'
            else:
                summary['type'] = str(type(data).__name__)
        except:
            summary['type'] = 'unknown'

        return summary

    def identify_data_types(self, data: Any) -> Dict[str, str]:
        """Идентификация типов данных"""
        types = {}

        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    types[key] = type(value).__name__
            elif isinstance(data, (list, tuple)):
                for i, item in enumerate(data):
                    types[f'item_{i}'] = type(item).__name__
        except:
            pass

        return types

    def check_completeness(self, data: Any) -> float:
        """Проверка полноты данных (0.0 - 1.0)"""
        try:
            if isinstance(data, dict):
                total = len(data)
                complete = sum(1 for v in data.values() if v is not None)
                return complete / total if total > 0 else 0.0
            elif isinstance(data, (list, tuple)):
                total = len(data)
                complete = sum(1 for item in data if item is not None)
                return complete / total if total > 0 else 0.0
            else:
                return 1.0 if data is not None else 0.0
        except:
            return 0.0


class InMemoryAnalysisCache(AnalysisCache):
    """Реализация кэша анализа в памяти"""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, AnalysisResult] = {}
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, cache_key: str) -> Optional[AnalysisResult]:
        """Получение результата из кэша"""
        async with self._lock:
            if cache_key in self._cache:
                # Проверяем, не устарел ли кэш (например, по времени)
                timestamp = self._timestamps.get(cache_key, 0)
                if time.time() - timestamp < 3600:  # 1 час
                    return self._cache[cache_key]
                else:
                    # Удаляем устаревший кэш
                    del self._cache[cache_key]
                    del self._timestamps[cache_key]
            return None

    async def set(self, cache_key: str, result: AnalysisResult) -> None:
        """Сохранение результата в кэш"""
        async with self._lock:
            if len(self._cache) >= self._max_size:
                # Удаляем старые записи
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[cache_key] = result
            self._timestamps[cache_key] = time.time()

    async def clear_expired(self) -> None:
        """Очистка устаревших записей"""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self._timestamps.items()
                if current_time - timestamp >= 3600  # 1 час
            ]
            
            for key in expired_keys:
                del self._cache[key]
                del self._timestamps[key]
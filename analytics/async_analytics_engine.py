"""
Асинхронный движок аналитики с поддержкой стриминга и улучшенной производительности
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
from functools import lru_cache
from collections import defaultdict

from .interfaces import (
    AnalysisResult, DataQualityAssessor, TimeSeriesProcessor, StatisticalAnalyzer,
    CorrelationAnalyzer, ComparativeAnalyzer, PerformanceAnalyzer, DataProcessor, AnalysisCache
)
from .implementations import (
    BasicDataQualityAssessor, BasicTimeSeriesProcessor, BasicStatisticalAnalyzer,
    BasicCorrelationAnalyzer, BasicComparativeAnalyzer, BasicPerformanceAnalyzer,
    BasicDataProcessor, InMemoryAnalysisCache
)

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Событие для стриминга аналитики"""
    event_type: str # 'chunk', 'progress', 'complete', 'error'
    data: Any
    progress: Optional[float] = None
    timestamp: Optional[datetime] = None


class AsyncAnalyticsEngine:
    """
    Асинхронный движок аналитики для AI агента с поддержкой стриминга, реализующий принципы SOLID.
    
    Основные улучшения:
    - Полностью асинхронная обработка
    - Поддержка стриминга результатов
    - Параллельная обработка данных
    - Кэширование результатов
    - Улучшенные алгоритмы анализа
    """

    def __init__(
        self,
        data_quality_assessor: Optional[DataQualityAssessor] = None,
        time_series_processor: Optional[TimeSeriesProcessor] = None,
        statistical_analyzer: Optional[StatisticalAnalyzer] = None,
        correlation_analyzer: Optional[CorrelationAnalyzer] = None,
        comparative_analyzer: Optional[ComparativeAnalyzer] = None,
        performance_analyzer: Optional[PerformanceAnalyzer] = None,
        data_processor: Optional[DataProcessor] = None,
        cache: Optional[AnalysisCache] = None
    ):
        self._initialized = False
        self.analysis_history: List[AnalysisResult] = []
        
        # Используем зависимости или создаем стандартные реализации
        self.data_quality_assessor = data_quality_assessor or BasicDataQualityAssessor()
        self.time_series_processor = time_series_processor or BasicTimeSeriesProcessor()
        self.statistical_analyzer = statistical_analyzer or BasicStatisticalAnalyzer()
        self.correlation_analyzer = correlation_analyzer or BasicCorrelationAnalyzer()
        self.comparative_analyzer = comparative_analyzer or BasicComparativeAnalyzer()
        self.performance_analyzer = performance_analyzer or BasicPerformanceAnalyzer()
        self.data_processor = data_processor or BasicDataProcessor()
        self.cache = cache or InMemoryAnalysisCache()
        
        self._executor = ThreadPoolExecutor(max_workers=8)  # Пул для CPU-интенсивных задач
        self._cache_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Инициализация аналитического движка"""
        if self._initialized:
            return True

        try:
            self._initialized = True
            logger.info("AsyncAnalyticsEngine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize analytics engine: {e}")
            return False

    async def analyze(
        self,
        data: Any,
        analysis_type: str,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> AnalysisResult:
        """
        Асинхронное выполнение анализа данных с кэшированием
        """
        if not await self.initialize():
            return AnalysisResult(
                analysis_type=analysis_type,
                insights=[],
                metrics={},
                recommendations=[],
                confidence=0.0,
                data_quality_score=0.0,
                processing_time=0.0,
                timestamp=datetime.now()
            )

        start_time = time.time()
        
        # Генерация ключа для кэша
        cache_key = f"{analysis_type}_{hash(str(data))}_{hash(str(context))}"
        
        # Проверка кэша
        if use_cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for analysis: {analysis_type}")
                return cached_result

        try:
            # Оценка качества данных
            data_quality = await self._assess_data_quality_async(data)

            # Выполнение соответствующего типа анализа
            if analysis_type == 'trend_analysis':
                result = await self._trend_analysis_async(data, context)
            elif analysis_type == 'comparative_analysis':
                result = await self._comparative_analysis_async(data, context)
            elif analysis_type == 'performance_analysis':
                result = await self._performance_analysis_async(data, context)
            elif analysis_type == 'statistical_analysis':
                result = await self._statistical_analysis_async(data, context)
            elif analysis_type == 'correlation_analysis':
                result = await self._correlation_analysis_async(data, context)
            else:
                result = await self._general_analysis_async(data, context)

            # Добавление метаданных
            result.data_quality_score = data_quality
            result.processing_time = time.time() - start_time
            result.timestamp = datetime.now()

            # Сохранение в историю
            self.analysis_history.append(result)
            
            # Кэширование результата
            await self.cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                analysis_type=analysis_type,
                insights=[f"Ошибка анализа: {str(e)}"],
                metrics={'error': str(e)},
                recommendations=["Проверьте корректность входных данных"],
                confidence=0.0,
                data_quality_score=0.0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )

    async def analyze_stream(
        self,
        data: Any,
        analysis_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Асинхронный стриминг анализа данных
        """
        start_time = time.time()
        
        try:
            # Отправляем событие начала
            yield StreamEvent(
                event_type='progress',
                data={'status': 'started', 'analysis_type': analysis_type},
                progress=0.0,
                timestamp=datetime.now()
            )
            
            # Оценка качества данных
            data_quality = await self._assess_data_quality_async(data)
            
            yield StreamEvent(
                event_type='progress',
                data={'status': 'data_quality_assessed', 'quality_score': data_quality},
                progress=0.1,
                timestamp=datetime.now()
            )

            # Выполнение соответствующего типа анализа с прогрессом
            if analysis_type == 'trend_analysis':
                async for event in self._trend_analysis_stream_async(data, context):
                    yield event
            elif analysis_type == 'comparative_analysis':
                async for event in self._comparative_analysis_stream_async(data, context):
                    yield event
            elif analysis_type == 'performance_analysis':
                async for event in self._performance_analysis_stream_async(data, context):
                    yield event
            elif analysis_type == 'statistical_analysis':
                async for event in self._statistical_analysis_stream_async(data, context):
                    yield event
            elif analysis_type == 'correlation_analysis':
                async for event in self._correlation_analysis_stream_async(data, context):
                    yield event
            else:
                async for event in self._general_analysis_stream_async(data, context):
                    yield event

            # Отправляем событие завершения
            yield StreamEvent(
                event_type='complete',
                data={'status': 'completed', 'processing_time': time.time() - start_time},
                progress=1.0,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Streaming analysis failed: {e}")
            yield StreamEvent(
                event_type='error',
                data={'error': str(e)},
                progress=0.0,
                timestamp=datetime.now()
            )

    async def _assess_data_quality_async(self, data: Any) -> float:
        """Асинхронная оценка качества данных (0.0 - 1.0)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.data_quality_assessor.assess, data)

    # Асинхронные методы анализа с поддержкой стриминга

    async def _trend_analysis_async(self, data: Any, context: Dict[str, Any]) -> AnalysisResult:
        """Асинхронный трендовый анализ данных"""
        insights = []
        metrics = {}
        recommendations = []

        try:
            # Извлечение временных рядов
            time_series = await self._extract_time_series_async(data)

            if not time_series:
                return AnalysisResult(
                    analysis_type='trend_analysis',
                    insights=["Недостаточно данных для трендового анализа"],
                    metrics={},
                    recommendations=["Соберите больше исторических данных"],
                    confidence=0.3,
                    data_quality_score=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now()
                )

            # Расчет тренда
            trend_direction, trend_strength = await self._calculate_trend_async(time_series)

            metrics.update({
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'data_points': len(time_series),
                'time_range': await self._calculate_time_range_async(time_series)
            })

            # Формирование инсайтов
            if trend_direction == 'upward':
                insights.append(f"Обнаружен восходящий тренд (сила: {trend_strength:.2f})")
                if trend_strength > 0.7:
                    insights.append("Тренд очень сильный - рекомендуется дальнейший рост")
            elif trend_direction == 'downward':
                insights.append(f"Обнаружен нисходящий тренд (сила: {trend_strength:.2f})")
                if trend_strength > 0.7:
                    insights.append("Тренд очень сильный - требуется внимание")
            else:
                insights.append("Тренд отсутствует или очень слабый")

            # Расчет сезонности
            seasonality = await self._detect_seasonality_async(time_series)
            if seasonality:
                insights.append(f"Обнаружена сезонность: {seasonality}")
                metrics['seasonality'] = seasonality

            # Прогноз
            forecast = await self._simple_forecast_async(time_series, periods=5)
            if forecast:
                metrics['forecast'] = forecast
                insights.append(f"Прогноз на 5 периодов: {forecast}")

            # Рекомендации
            if trend_direction == 'downward' and trend_strength > 0.5:
                recommendations.append("Рассмотрите меры по улучшению показателей")
            elif trend_direction == 'upward' and trend_strength > 0.8:
                recommendations.append("Поддерживайте текущую положительную динамику")

            confidence = min(0.9, trend_strength + 0.3)

        except Exception as e:
            insights.append(f"Ошибка трендового анализа: {str(e)}")
            confidence = 0.1

        return AnalysisResult(
            analysis_type='trend_analysis',
            insights=insights,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence,
            data_quality_score=0.0,
            processing_time=0.0,
            timestamp=datetime.now()
        )

    async def _trend_analysis_stream_async(self, data: Any, context: Dict[str, Any]) -> AsyncIterator[StreamEvent]:
        """Асинхронный стриминг трендового анализа"""
        yield StreamEvent(
            event_type='progress',
            data={'step': 'extracting_time_series'},
            progress=0.2,
            timestamp=datetime.now()
        )
        
        time_series = await self._extract_time_series_async(data)
        
        yield StreamEvent(
            event_type='progress',
            data={'step': 'calculating_trend', 'data_points': len(time_series) if time_series else 0},
            progress=0.4,
            timestamp=datetime.now()
        )
        
        if time_series:
            trend_direction, trend_strength = await self._calculate_trend_async(time_series)
            
            yield StreamEvent(
                event_type='progress',
                data={'step': 'detecting_seasonality'},
                progress=0.6,
                timestamp=datetime.now()
            )
            
            seasonality = await self._detect_seasonality_async(time_series)
            
            yield StreamEvent(
                event_type='progress',
                data={'step': 'generating_forecast'},
                progress=0.8,
                timestamp=datetime.now()
            )
            
            forecast = await self._simple_forecast_async(time_series, periods=5)
            
            result = await self._trend_analysis_async(data, context)
            yield StreamEvent(
                event_type='chunk',
                data=result,
                progress=1.0,
                timestamp=datetime.now()
            )

    async def _comparative_analysis_async(self, data: Any, context: Dict[str, Any]) -> AnalysisResult:
        """Асинхронный сравнительный анализ данных"""
        insights = []
        metrics = {}
        recommendations = []

        try:
            # Извлечение данных для сравнения
            comparison_data = await self._extract_comparison_data_async(data)

            if len(comparison_data) < 2:
                return AnalysisResult(
                    analysis_type='comparative_analysis',
                    insights=["Недостаточно данных для сравнения"],
                    metrics={},
                    recommendations=["Предоставьте данные минимум для двух объектов сравнения"],
                    confidence=0.2,
                    data_quality_score=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now()
                )

            # Расчет статистических различий
            differences = await self._calculate_differences_async(comparison_data)

            metrics.update({
                'num_subjects': len(comparison_data),
                'differences': differences,
                'statistical_significance': await self._test_significance_async(comparison_data)
            })

            # Формирование инсайтов
            if comparison_data:
                best_key = max(comparison_data.keys(), key=lambda x: comparison_data[x].get('mean', 0))
                worst_key = min(comparison_data.keys(), key=lambda x: comparison_data[x].get('mean', 0))

                insights.append(f"Лучший результат: {best_key}")
                insights.append(f"Худший результат: {worst_key}")

            # Анализ различий
            if differences.get('max_diff', 0) > 0.5:
                insights.append("Значительные различия между сравниваемыми объектами")
            elif differences.get('max_diff', 0) < 0.1:
                insights.append("Различия между объектами минимальны")

            # Корреляционный анализ
            correlations = await self._calculate_correlations_async(comparison_data)
            if correlations:
                metrics['correlations'] = correlations
                strong_corr = [(k, v) for k, v in correlations.items() if abs(v) > 0.7]
                if strong_corr:
                    insights.append(f"Сильные корреляции: {strong_corr}")

            # Рекомендации
            if best_key:
                recommendations.append(f"Изучите практики {best_key} для применения")
            if differences.get('statistical_significance', False):
                recommendations.append("Различия статистически значимы")

            confidence = 0.8 if differences.get('statistical_significance', False) else 0.6

        except Exception as e:
            insights.append(f"Ошибка сравнительного анализа: {str(e)}")
            confidence = 0.1

        return AnalysisResult(
            analysis_type='comparative_analysis',
            insights=insights,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence,
            data_quality_score=0.0,
            processing_time=0.0,
            timestamp=datetime.now()
        )

    async def _performance_analysis_async(self, data: Any, context: Dict[str, Any]) -> AnalysisResult:
        """Асинхронный анализ производительности"""
        insights = []
        metrics = {}
        recommendations = []

        try:
            # Извлечение метрик производительности
            performance_data = await self._extract_performance_data_async(data)

            # Расчет ключевых метрик
            avg_response_time = statistics.mean(performance_data.get('response_times', [0]))
            success_rate = performance_data.get('success_count', 0) / max(1, performance_data.get('total_count', 1))
            throughput = performance_data.get('throughput', 0)

            metrics.update({
                'avg_response_time': avg_response_time,
                'success_rate': success_rate,
                'throughput': throughput,
                'error_rate': 1 - success_rate,
                'efficiency_score': await self._calculate_efficiency_score_async(performance_data)
            })

            # Анализ bottleneck'ов
            bottlenecks = await self._identify_bottlenecks_async(performance_data)
            if bottlenecks:
                insights.extend([f"Bottleneck: {b}" for b in bottlenecks])
                metrics['bottlenecks'] = bottlenecks

            # Оценка производительности
            if success_rate > 0.95:
                insights.append("Отличная производительность системы")
            elif success_rate > 0.8:
                insights.append("Хорошая производительность с возможностями улучшения")
            else:
                insights.append("Производительность требует улучшения")

            if avg_response_time < 1.0:
                insights.append("Быстрое время отклика")
            elif avg_response_time > 5.0:
                insights.append("Медленное время отклика - требуется оптимизация")

            # Тренды производительности
            if len(performance_data.get('response_times', [])) > 10:
                trend = await self._analyze_performance_trend_async(performance_data)
                if trend:
                    insights.append(f"Тренд производительности: {trend}")
                    metrics['performance_trend'] = trend

            # Рекомендации
            if success_rate < 0.9:
                recommendations.append("Улучшите обработку ошибок")
            if avg_response_time > 3.0:
                recommendations.append("Оптимизируйте время отклика")
            if bottlenecks:
                recommendations.append("Устраните выявленные bottleneck'ы")

            confidence = min(0.9, success_rate + 0.4)

        except Exception as e:
            insights.append(f"Ошибка анализа производительности: {str(e)}")
            confidence = 0.1

        return AnalysisResult(
            analysis_type='performance_analysis',
            insights=insights,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence,
            data_quality_score=0.0,
            processing_time=0.0,
            timestamp=datetime.now()
        )

    async def _statistical_analysis_async(self, data: Any, context: Dict[str, Any]) -> AnalysisResult:
        """Асинхронный статистический анализ данных"""
        insights = []
        metrics = {}
        recommendations = []

        try:
            # Извлечение числовых данных
            numeric_data = await self._extract_numeric_data_async(data)

            if not numeric_data:
                return AnalysisResult(
                    analysis_type='statistical_analysis',
                    insights=["Недостаточно числовых данных для статистического анализа"],
                    metrics={},
                    recommendations=["Предоставьте числовые данные"],
                    confidence=0.2,
                    data_quality_score=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now()
                )

            # Базовая статистика
            basic_stats = await self._calculate_basic_statistics_async(numeric_data)

            # Распределение
            distribution_info = await self._analyze_distribution_async(numeric_data)

            # Выбросы
            outliers = await self._detect_outliers_async(numeric_data)

            metrics.update({
                'basic_stats': basic_stats,
                'distribution': distribution_info,
                'outliers': outliers,
                'data_points': len(numeric_data)
            })

            # Формирование инсайтов
            if basic_stats.get('std_dev', 0) < basic_stats.get('mean', 0) * 0.1:
                insights.append("Данные имеют низкую вариативность")
            elif basic_stats.get('std_dev', 0) > basic_stats.get('mean', 0) * 0.5:
                insights.append("Данные имеют высокую вариативность")

            if distribution_info.get('normality_test', {}).get('is_normal', False):
                insights.append("Данные соответствуют нормальному распределению")
            else:
                insights.append("Данные не соответствуют нормальному распределению")

            if outliers.get('count', 0) > 0:
                insights.append(f"Обнаружено {outliers['count']} выбросов")

            # Рекомендации
            if outliers.get('count', 0) > len(numeric_data) * 0.1:
                recommendations.append("Рассмотрите удаление или обработку выбросов")
            if not distribution_info.get('normality_test', {}).get('is_normal', True):
                recommendations.append("Для ненормальных данных используйте непараметрические методы")

            confidence = 0.85

        except Exception as e:
            insights.append(f"Ошибка статистического анализа: {str(e)}")
            confidence = 0.1

        return AnalysisResult(
            analysis_type='statistical_analysis',
            insights=insights,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence,
            data_quality_score=0.0,
            processing_time=0.0,
            timestamp=datetime.now()
        )

    async def _correlation_analysis_async(self, data: Any, context: Dict[str, Any]) -> AnalysisResult:
        """Асинхронный корреляционный анализ"""
        insights = []
        metrics = {}
        recommendations = []

        try:
            # Извлечение данных для корреляции
            correlation_data = await self._extract_correlation_data_async(data)

            if len(correlation_data) < 2:
                return AnalysisResult(
                    analysis_type='correlation_analysis',
                    insights=["Недостаточно переменных для корреляционного анализа"],
                    metrics={},
                    recommendations=["Предоставьте данные минимум для двух переменных"],
                    confidence=0.2,
                    data_quality_score=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now()
                )

            # Расчет корреляционной матрицы
            corr_matrix = await self._calculate_correlation_matrix_async(correlation_data)

            # Поиск сильных корреляций
            strong_correlations = await self._find_strong_correlations_async(corr_matrix)

            metrics.update({
                'correlation_matrix': corr_matrix,
                'strong_correlations': strong_correlations,
                'variables': list(correlation_data.keys())
            })

            # Формирование инсайтов
            if strong_correlations:
                insights.append(f"Найдено {len(strong_correlations)} сильных корреляций")
                for corr in strong_correlations[:5]:  # Показать топ-5
                    var1, var2, coeff = corr['variables'][0], corr['variables'][1], corr['coefficient']
                    direction = "положительная" if coeff > 0 else "отрицательная"
                    strength = "сильная" if abs(coeff) > 0.7 else "умеренная"
                    insights.append(f"{strength} {direction} корреляция между {var1} и {var2} (r={coeff:.2f})")
            else:
                insights.append("Сильные корреляции не обнаружены")

            # Анализ мультиколлинеарности
            multicollinearity = await self._check_multicollinearity_async(corr_matrix)
            if multicollinearity:
                insights.append("Обнаружена мультиколлинеарность между переменными")
                metrics['multicollinearity'] = multicollinearity

            # Рекомендации
            if strong_correlations:
                recommendations.append("Изучите причинно-следственные связи для сильных корреляций")
            if multicollinearity:
                recommendations.append("Рассмотрите удаление коррелированных переменных для моделирования")

            confidence = 0.8

        except Exception as e:
            insights.append(f"Ошибка корреляционного анализа: {str(e)}")
            confidence = 0.1

        return AnalysisResult(
            analysis_type='correlation_analysis',
            insights=insights,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence,
            data_quality_score=0.0,
            processing_time=0.0,
            timestamp=datetime.now()
        )

    async def _general_analysis_async(self, data: Any, context: Dict[str, Any]) -> AnalysisResult:
        """Асинхронный общий анализ данных"""
        insights = []
        metrics = {}
        recommendations = []

        try:
            # Общая характеристика данных
            data_summary = await self._summarize_data_async(data)

            metrics.update({
                'data_summary': data_summary,
                'data_types': await self._identify_data_types_async(data),
                'completeness': await self._check_completeness_async(data)
            })

            # Формирование инсайтов
            insights.append(f"Проанализировано {data_summary.get('total_elements', 0)} элементов данных")

            data_types = metrics['data_types']
            if data_types:
                type_counts = {}
                for dt in data_types.values():
                    type_counts[dt] = type_counts.get(dt, 0) + 1
                insights.append(f"Типы данных: {type_counts}")

            completeness = metrics['completeness']
            if completeness < 0.8:
                insights.append(f"Низкая полнота данных: {completeness:.1%}")
                recommendations.append("Улучшите сбор данных для повышения полноты")

            # Общие рекомендации
            recommendations.append("Рассмотрите более специфичный тип анализа для глубоких инсайтов")

            confidence = 0.6

        except Exception as e:
            insights.append(f"Ошибка общего анализа: {str(e)}")
            confidence = 0.1

        return AnalysisResult(
            analysis_type='general_analysis',
            insights=insights,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence,
            data_quality_score=0.0,
            processing_time=0.0,
            timestamp=datetime.now()
        )

    # Асинхронные вспомогательные методы

    async def _extract_time_series_async(self, data: Any) -> List[Tuple[datetime, float]]:
        """Асинхронное извлечение временных рядов из данных"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.time_series_processor.extract, data)

    async def _calculate_trend_async(self, time_series: List[Tuple[datetime, float]]) -> Tuple[str, float]:
        """Асинхронный расчет тренда временного ряда"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.time_series_processor.calculate_trend, time_series)

    async def _detect_seasonality_async(self, time_series: List[Tuple[datetime, float]]) -> Optional[str]:
        """Асинхронное обнаружение сезонности"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.time_series_processor.detect_seasonality, time_series)

    async def _simple_forecast_async(self, time_series: List[Tuple[datetime, float]], periods: int) -> Optional[List[float]]:
        """Асинхронный простой прогноз"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.time_series_processor.forecast, time_series, periods)

    async def _extract_comparison_data_async(self, data: Any) -> Dict[str, Dict[str, Any]]:
        """Асинхронное извлечение данных для сравнения"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.comparative_analyzer.extract_comparison_data, data)

    async def _calculate_differences_async(self, comparison_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Асинхронный расчет различий между объектами"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.comparative_analyzer.calculate_differences, comparison_data)

    async def _test_significance_async(self, comparison_data: Dict[str, Dict[str, Any]]) -> bool:
        """Асинхронный тест статистической значимости"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.comparative_analyzer.test_significance, comparison_data)

    async def _calculate_correlations_async(self, comparison_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Асинхронный расчет корреляций"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.comparative_analyzer.calculate_correlations, comparison_data)

    async def _extract_performance_data_async(self, data: Any) -> Dict[str, Any]:
        """Асинхронное извлечение данных производительности"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.performance_analyzer.extract_performance_data, data)

    async def _calculate_efficiency_score_async(self, performance_data: Dict[str, Any]) -> float:
        """Асинхронный расчет оценки эффективности"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.performance_analyzer.calculate_efficiency_score, performance_data)

    async def _identify_bottlenecks_async(self, performance_data: Dict[str, Any]) -> List[str]:
        """Асинхронная идентификация bottleneck'ов"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.performance_analyzer.identify_bottlenecks, performance_data)

    async def _analyze_performance_trend_async(self, performance_data: Dict[str, Any]) -> Optional[str]:
        """Асинхронный анализ тренда производительности"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.performance_analyzer.analyze_performance_trend, performance_data)

    async def _extract_numeric_data_async(self, data: Any) -> List[float]:
        """Асинхронное извлечение числовых данных"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.data_processor.extract_numeric_data, data)

    async def _calculate_basic_statistics_async(self, numeric_data: List[float]) -> Dict[str, float]:
        """Асинхронный расчет базовой статистики"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.statistical_analyzer.calculate_basic_statistics, numeric_data)

    async def _analyze_distribution_async(self, numeric_data: List[float]) -> Dict[str, Any]:
        """Асинхронный анализ распределения данных"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.statistical_analyzer.analyze_distribution, numeric_data)

    async def _detect_outliers_async(self, numeric_data: List[float]) -> Dict[str, Any]:
        """Асинхронное обнаружение выбросов"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.statistical_analyzer.detect_outliers, numeric_data)

    async def _extract_correlation_data_async(self, data: Any) -> Dict[str, List[float]]:
        """Асинхронное извлечение данных для корреляции"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.correlation_analyzer.extract_correlation_data, data)

    async def _calculate_correlation_matrix_async(self, correlation_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Асинхронный расчет корреляционной матрицы"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.correlation_analyzer.calculate_correlation_matrix, correlation_data)

    async def _find_strong_correlations_async(self, corr_matrix: Dict[str, Dict[str, float]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Асинхронный поиск сильных корреляций"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.correlation_analyzer.find_strong_correlations, corr_matrix, threshold)

    async def _check_multicollinearity_async(self, corr_matrix: Dict[str, Dict[str, float]], threshold: float = 0.8) -> Optional[List[List[str]]]:
        """Асинхронная проверка мультиколлинеарности"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.correlation_analyzer.check_multicollinearity, corr_matrix, threshold)

    async def _summarize_data_async(self, data: Any) -> Dict[str, Any]:
        """Асинхронная общая сводка данных"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.data_processor.summarize_data, data)

    async def _identify_data_types_async(self, data: Any) -> Dict[str, str]:
        """Асинхронная идентификация типов данных"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.data_processor.identify_data_types, data)

    async def _check_completeness_async(self, data: Any) -> float:
        """Асинхронная проверка полноты данных (0.0 - 1.0)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.data_processor.check_completeness, data)

    async def _calculate_time_range_async(self, time_series: List[Tuple[datetime, float]]) -> Optional[Dict[str, Any]]:
        """Асинхронный расчет временного диапазона"""
        if not time_series:
            return None

        timestamps = [ts for ts, _ in time_series]
        start = min(timestamps)
        end = max(timestamps)
        duration = end - start

        return {
            'start': start.isoformat(),
            'end': end.isoformat(),
            'duration_days': duration.days,
            'duration_hours': duration.total_seconds() / 3600
        }

    async def get_analysis_history(self, limit: int = 10) -> List[AnalysisResult]:
        """Получение истории анализов"""
        return self.analysis_history[-limit:] if self.analysis_history else []

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса аналитического движка"""
        return {
            'initialized': self._initialized,
            'supported_analysis_types': [
                'trend_analysis', 'comparative_analysis', 'performance_analysis',
                'statistical_analysis', 'correlation_analysis', 'general_analysis'
            ],
            'analysis_history_count': len(self.analysis_history),
            'cache_size': len(self.analysis_history),  # В реальной реализации нужно отдельно отслеживать размер кэша
            'max_cache_size': 1000,  # В реальной реализации нужно отслеживать это в кэше
            'last_analysis': self.analysis_history[-1].timestamp.isoformat() if self.analysis_history else None
        }

    async def clear_cache(self):
        """Очистка кэша"""
        await self.cache.clear_expired()
        logger.info("Analytics cache cleared")
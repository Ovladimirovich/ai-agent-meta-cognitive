"""
Анализатор когнитивной нагрузки агента
Фаза 3: Расширенные мета-когнитивные функции
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LoadLevel(Enum):
    """Уровни когнитивной нагрузки"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CognitiveLoadMetrics:
    """Метрики когнитивной нагрузки"""
    response_time: float  # Время ответа
    processing_time: float # Время обработки
    memory_usage: float  # Использование памяти
    cpu_usage: float # Использование CPU
    active_tasks: int  # Активные задачи
    confidence_level: float  # Уровень уверенности
    error_rate: float  # Уровень ошибок
    complexity_score: float  # Оценка сложности задачи
    resource_pressure: float  # Давление на ресурсы
    timestamp: datetime  # Время измерения


@dataclass
class CognitiveLoadAnalysis:
    """Результат анализа когнитивной нагрузки"""
    load_level: LoadLevel  # Уровень нагрузки
    load_score: float  # Оценка нагрузки (0.0-1.0)
    metrics: CognitiveLoadMetrics  # Метрики
    contributing_factors: List[str]  # Факторы нагрузки
    recommendations: List[str] # Рекомендации
    trends: Dict[str, Any]  # Тренды нагрузки


class CognitiveLoadAnalyzer:
    """
    Анализатор когнитивной нагрузки агента
    
    Выполняет анализ текущей когнитивной нагрузки на основе различных метрик:
    - Время отклика
    - Использование ресурсов
    - Сложность задач
    - История ошибок
    - Активные процессы
    """

    def __init__(self):
        self.load_history: List[CognitiveLoadMetrics] = []
        self.max_history_size = 1000
        self.thresholds = {
            'low_to_medium': 0.3,
            'medium_to_high': 0.6,
            'high_to_critical': 0.8
        }
        
        # Веса для различных факторов нагрузки
        self.factor_weights = {
            'response_time': 0.2,
            'memory_usage': 0.2,
            'cpu_usage': 0.2,
            'active_tasks': 0.15,
            'error_rate': 0.15,
            'complexity_score': 0.1
        }

    def add_metrics(self, metrics: CognitiveLoadMetrics):
        """Добавление метрик в историю"""
        self.load_history.append(metrics)
        if len(self.load_history) > self.max_history_size:
            self.load_history = self.load_history[-self.max_history_size:]

    def calculate_load_score(self, metrics: CognitiveLoadMetrics) -> float:
        """
        Расчет общей оценки когнитивной нагрузки
        
        Args:
            metrics: Метрики для анализа
            
        Returns:
            float: Оценка нагрузки (0.0-1.0)
        """
        # Нормализация метрик к диапазону 0-1
        normalized_response_time = min(metrics.response_time / 5.0, 1.0)  # 5 секунд как максимум
        normalized_memory = metrics.memory_usage
        normalized_cpu = metrics.cpu_usage
        normalized_tasks = min(metrics.active_tasks / 50.0, 1.0)  # 50 задач как максимум
        normalized_error_rate = metrics.error_rate
        normalized_complexity = metrics.complexity_score
        
        # Взвешенное суммирование
        load_score = (
            normalized_response_time * self.factor_weights['response_time'] +
            normalized_memory * self.factor_weights['memory_usage'] +
            normalized_cpu * self.factor_weights['cpu_usage'] +
            normalized_tasks * self.factor_weights['active_tasks'] +
            normalized_error_rate * self.factor_weights['error_rate'] +
            normalized_complexity * self.factor_weights['complexity_score']
        )
        
        # Учет уровня уверенности - низкая уверенность увеличивает нагрузку
        confidence_factor = max(0, (1.0 - metrics.confidence_level) * 0.1)
        load_score = min(load_score + confidence_factor, 1.0)
        
        return load_score

    def determine_load_level(self, load_score: float) -> LoadLevel:
        """Определение уровня нагрузки по оценке"""
        if load_score >= self.thresholds['high_to_critical']:
            return LoadLevel.CRITICAL
        elif load_score >= self.thresholds['medium_to_high']:
            return LoadLevel.HIGH
        elif load_score >= self.thresholds['low_to_medium']:
            return LoadLevel.MEDIUM
        else:
            return LoadLevel.LOW

    def identify_contributing_factors(self, metrics: CognitiveLoadMetrics) -> List[str]:
        """Определение факторов, способствующих нагрузке"""
        factors = []
        
        if metrics.response_time > 3.0:
            factors.append("long_response_time")
        if metrics.memory_usage > 0.8:
            factors.append("high_memory_usage")
        if metrics.cpu_usage > 0.85:
            factors.append("high_cpu_usage")
        if metrics.active_tasks > 30:
            factors.append("many_active_tasks")
        if metrics.error_rate > 0.2:
            factors.append("high_error_rate")
        if metrics.complexity_score > 0.7:
            factors.append("high_complexity")
        if metrics.confidence_level < 0.3:
            factors.append("low_confidence")
        if metrics.resource_pressure > 0.7:
            factors.append("high_resource_pressure")
            
        return factors

    def generate_recommendations(self, load_level: LoadLevel, factors: List[str]) -> List[str]:
        """Генерация рекомендаций на основе уровня нагрузки и факторов"""
        recommendations = []
        
        if load_level == LoadLevel.CRITICAL:
            recommendations.append("Срочно снизить нагрузку - возможен сбой системы")
            recommendations.append("Приостановить выполнение новых задач")
            recommendations.append("Выполнить оптимизацию ресурсов")
        elif load_level == LoadLevel.HIGH:
            recommendations.append("Рассмотреть возможность распределения нагрузки")
            recommendations.append("Оптимизировать текущие процессы")
            recommendations.append("Проверить доступные ресурсы")
        elif load_level == LoadLevel.MEDIUM:
            recommendations.append("Мониторить уровень нагрузки")
            recommendations.append("Подготовиться к увеличению нагрузки")
        else:
            recommendations.append("Нагрузка в пределах нормы")
            recommendations.append("Можно принимать дополнительные задачи")
        
        # Добавление специфических рекомендаций по факторам
        for factor in factors:
            if factor == "long_response_time":
                recommendations.append("Оптимизировать время отклика")
            elif factor == "high_memory_usage":
                recommendations.append("Очистить память или увеличить лимит")
            elif factor == "high_cpu_usage":
                recommendations.append("Оптимизировать вычисления")
            elif factor == "many_active_tasks":
                recommendations.append("Уменьшить количество параллельных задач")
            elif factor == "high_error_rate":
                recommendations.append("Проверить стабильность системы")
            elif factor == "high_complexity":
                recommendations.append("Разбить сложные задачи на подзадачи")
            elif factor == "low_confidence":
                recommendations.append("Улучшить качество входных данных")
            elif factor == "high_resource_pressure":
                recommendations.append("Выделить дополнительные ресурсы")
        
        return recommendations

    def analyze_trends(self) -> Dict[str, Any]:
        """Анализ трендов когнитивной нагрузки"""
        if len(self.load_history) < 5:
            return {
                'trend_direction': 'insufficient_data',
                'load_trend': [],
                'peak_times': [],
                'average_load': 0.0
            }
        
        recent_metrics = self.load_history[-20:]  # Последние 20 измерений
        load_scores = [self.calculate_load_score(m) for m in recent_metrics]
        
        # Определение тренда
        if len(load_scores) >= 2:
            recent_avg = sum(load_scores[-5:]) / len(load_scores[-5:])
            older_avg = sum(load_scores[:-5]) / max(1, len(load_scores) - 5)
            
            if recent_avg > older_avg * 1.1:
                trend_direction = 'increasing'
            elif recent_avg < older_avg * 0.9:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'
        
        # Найти пики нагрузки
        peak_threshold = sum(load_scores) / len(load_scores) * 1.5
        peak_times = [
            metrics.timestamp for metrics in recent_metrics 
            if self.calculate_load_score(metrics) > peak_threshold
        ]
        
        return {
            'trend_direction': trend_direction,
            'load_trend': load_scores,
            'peak_times': peak_times,
            'average_load': sum(load_scores) / len(load_scores)
        }

    async def analyze_cognitive_load(self, metrics: CognitiveLoadMetrics) -> CognitiveLoadAnalysis:
        """
        Полный анализ когнитивной нагрузки
        
        Args:
            metrics: Метрики для анализа
            
        Returns:
            CognitiveLoadAnalysis: Результат анализа
        """
        try:
            # Добавление метрик в историю
            self.add_metrics(metrics)
            
            # Расчет оценки нагрузки
            load_score = self.calculate_load_score(metrics)
            
            # Определение уровня нагрузки
            load_level = self.determine_load_level(load_score)
            
            # Определение факторов нагрузки
            contributing_factors = self.identify_contributing_factors(metrics)
            
            # Генерация рекомендаций
            recommendations = self.generate_recommendations(load_level, contributing_factors)
            
            # Анализ трендов
            trends = self.analyze_trends()
            
            analysis = CognitiveLoadAnalysis(
                load_level=load_level,
                load_score=load_score,
                metrics=metrics,
                contributing_factors=contributing_factors,
                recommendations=recommendations,
                trends=trends
            )
            
            logger.info(f"Completed cognitive load analysis: {load_level.value} ({load_score:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in cognitive load analysis: {e}")
            # Возвращаем анализ с критическим уровнем при ошибке
            return CognitiveLoadAnalysis(
                load_level=LoadLevel.CRITICAL,
                load_score=1.0,
                metrics=metrics,
                contributing_factors=["analysis_error"],
                recommendations=["Система не может выполнить анализ нагрузки"],
                trends={}
            )

    def get_load_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Получение сводки по нагрузке за указанный период"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        relevant_metrics = [m for m in self.load_history if m.timestamp >= cutoff_time]
        
        if not relevant_metrics:
            return {
                'period': f"last_{hours}_hours",
                'total_measurements': 0,
                'average_load': 0.0,
                'peak_load': 0.0,
                'load_distribution': {
                    'low': 0,
                    'medium': 0,
                    'high': 0,
                    'critical': 0
                },
                'trend_direction': 'no_data'
            }
        
        load_scores = [self.calculate_load_score(m) for m in relevant_metrics]
        load_levels = [self.determine_load_level(score) for score in load_scores]
        
        # Подсчет распределения уровней
        level_counts = {
            LoadLevel.LOW: 0,
            LoadLevel.MEDIUM: 0,
            LoadLevel.HIGH: 0,
            LoadLevel.CRITICAL: 0
        }
        for level in load_levels:
            level_counts[level] += 1
        
        # Определение тренда
        if len(load_scores) >= 10:
            recent_avg = sum(load_scores[-5:]) / 5
            older_avg = sum(load_scores[:-5]) / max(1, len(load_scores) - 5)
            
            if recent_avg > older_avg * 1.1:
                trend_direction = 'increasing'
            elif recent_avg < older_avg * 0.9:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'insufficient_data'
        
        return {
            'period': f"last_{hours}_hours",
            'total_measurements': len(relevant_metrics),
            'average_load': sum(load_scores) / len(load_scores),
            'peak_load': max(load_scores),
            'load_distribution': {
                'low': level_counts[LoadLevel.LOW],
                'medium': level_counts[LoadLevel.MEDIUM],
                'high': level_counts[LoadLevel.HIGH],
                'critical': level_counts[LoadLevel.CRITICAL]
            },
            'trend_direction': trend_direction
        }


class CognitiveLoadVisualizer:
    """
    Визуализатор когнитивной нагрузки
    
    Генерирует данные для визуализации нагрузки в различных форматах
    """
    
    def __init__(self, analyzer: CognitiveLoadAnalyzer):
        self.analyzer = analyzer

    def generate_time_series_data(self, hours: int = 24) -> Dict[str, Any]:
        """Генерация данных для временного ряда нагрузки"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        relevant_metrics = [m for m in self.analyzer.load_history if m.timestamp >= cutoff_time]
        
        if not relevant_metrics:
            return {
                'timestamps': [],
                'load_scores': [],
                'load_levels': [],
                'response_times': [],
                'memory_usage': [],
                'cpu_usage': []
            }
        
        # Сортировка по времени
        sorted_metrics = sorted(relevant_metrics, key=lambda m: m.timestamp)
        
        return {
            'timestamps': [m.timestamp.isoformat() for m in sorted_metrics],
            'load_scores': [self.analyzer.calculate_load_score(m) for m in sorted_metrics],
            'load_levels': [self.analyzer.determine_load_level(
                self.analyzer.calculate_load_score(m)
            ).value for m in sorted_metrics],
            'response_times': [m.response_time for m in sorted_metrics],
            'memory_usage': [m.memory_usage for m in sorted_metrics],
            'cpu_usage': [m.cpu_usage for m in sorted_metrics]
        }

    def generate_distribution_data(self) -> Dict[str, Any]:
        """Генерация данных для распределения нагрузки"""
        if not self.analyzer.load_history:
            return {
                'load_levels': [],
                'counts': [],
                'percentages': []
            }
        
        load_scores = [self.analyzer.calculate_load_score(m) for m in self.analyzer.load_history]
        load_levels = [self.analyzer.determine_load_level(score) for score in load_scores]
        
        # Подсчет распределения
        level_counts = {
            LoadLevel.LOW: 0,
            LoadLevel.MEDIUM: 0,
            LoadLevel.HIGH: 0,
            LoadLevel.CRITICAL: 0
        }
        for level in load_levels:
            level_counts[level] += 1
        
        total = len(load_levels)
        return {
            'load_levels': [level.value for level in LoadLevel],
            'counts': [level_counts[LoadLevel.LOW], level_counts[LoadLevel.MEDIUM], 
                      level_counts[LoadLevel.HIGH], level_counts[LoadLevel.CRITICAL]],
            'percentages': [level_counts[LoadLevel.LOW]/total*100, 
                           level_counts[LoadLevel.MEDIUM]/total*100,
                           level_counts[LoadLevel.HIGH]/total*100,
                           level_counts[LoadLevel.CRITICAL]/total*100]
        }

    def generate_real_time_data(self) -> Dict[str, Any]:
        """Генерация данных для реального времени"""
        if not self.analyzer.load_history:
            return {
                'current_load_score': 0.0,
                'current_load_level': LoadLevel.LOW.value,
                'current_metrics': None,
                'trend': 'stable',
                'recommendations': []
            }
        
        latest_metrics = self.analyzer.load_history[-1]
        current_score = self.analyzer.calculate_load_score(latest_metrics)
        current_level = self.analyzer.determine_load_level(current_score)
        
        # Определение тренда за последние 10 измерений
        recent_metrics = self.analyzer.load_history[-10:]
        if len(recent_metrics) >= 2:
            recent_scores = [self.analyzer.calculate_load_score(m) for m in recent_metrics]
            avg_recent = sum(recent_scores[-3:]) / len(recent_scores[-3:])
            avg_older = sum(recent_scores[:-3]) / max(1, len(recent_scores) - 3)
            
            if avg_recent > avg_older * 1.1:
                trend = 'increasing'
            elif avg_recent < avg_older * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'current_load_score': current_score,
            'current_load_level': current_level.value,
            'current_metrics': {
                'response_time': latest_metrics.response_time,
                'memory_usage': latest_metrics.memory_usage,
                'cpu_usage': latest_metrics.cpu_usage,
                'active_tasks': latest_metrics.active_tasks,
                'confidence_level': latest_metrics.confidence_level
            },
            'trend': trend,
            'recommendations': self.analyzer.generate_recommendations(current_level, [])
        }

    def generate_advanced_visualization_data(self, hours: int = 24) -> Dict[str, Any]:
        """
        Генерация расширенных данных для визуализации когнитивной нагрузки
        
        Args:
            hours: Количество часов для анализа
            
        Returns:
            Dict: Расширенные данные для визуализации
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        relevant_metrics = [m for m in self.analyzer.load_history if m.timestamp >= cutoff_time]
        
        if not relevant_metrics:
            return {
                'time_series': [],
                'load_distribution': {},
                'peak_analysis': [],
                'trend_analysis': {},
                'correlation_analysis': {},
                'prediction_data': []
            }
        
        # Сортировка по времени
        sorted_metrics = sorted(relevant_metrics, key=lambda m: m.timestamp)
        
        # Временной ряд
        time_series = []
        for metrics in sorted_metrics:
            time_series.append({
                'timestamp': metrics.timestamp.isoformat(),
                'load_score': self.analyzer.calculate_load_score(metrics),
                'load_level': self.analyzer.determine_load_level(
                    self.analyzer.calculate_load_score(metrics)
                ).value,
                'response_time': metrics.response_time,
                'memory_usage': metrics.memory_usage,
                'cpu_usage': metrics.cpu_usage,
                'active_tasks': metrics.active_tasks,
                'confidence_level': metrics.confidence_level,
                'error_rate': metrics.error_rate,
                'complexity_score': metrics.complexity_score
            })
        
        # Распределение нагрузки
        load_scores = [self.analyzer.calculate_load_score(m) for m in sorted_metrics]
        load_levels = [self.analyzer.determine_load_level(score) for score in load_scores]
        
        load_distribution = {
            LoadLevel.LOW.value: load_levels.count(LoadLevel.LOW),
            LoadLevel.MEDIUM.value: load_levels.count(LoadLevel.MEDIUM),
            LoadLevel.HIGH.value: load_levels.count(LoadLevel.HIGH),
            LoadLevel.CRITICAL.value: load_levels.count(LoadLevel.CRITICAL)
        }
        
        # Анализ пиков
        avg_load = sum(load_scores) / len(load_scores)
        peak_threshold = avg_load * 1.5
        peak_analysis = [
            {
                'timestamp': m.timestamp.isoformat(),
                'load_score': self.analyzer.calculate_load_score(m),
                'factors': self.analyzer.identify_contributing_factors(m)
            }
            for m in sorted_metrics
            if self.analyzer.calculate_load_score(m) > peak_threshold
        ]
        
        # Анализ трендов
        if len(load_scores) >= 10:
            recent_avg = sum(load_scores[-5:]) / 5
            older_avg = sum(load_scores[:-5]) / max(1, len(load_scores) - 5)
            trend_direction = 'increasing' if recent_avg > older_avg * 1.1 else 'decreasing' if recent_avg < older_avg * 0.9 else 'stable'
        else:
            trend_direction = 'insufficient_data'
        
        trend_analysis = {
            'direction': trend_direction,
            'current_avg': sum(load_scores[-5:]) / min(5, len(load_scores)) if load_scores else 0,
            'historical_avg': avg_load,
            'change_rate': ((recent_avg - older_avg) / older_avg) if older_avg > 0 else 0 if len(load_scores) >= 10 else 0
        }
        
        # Анализ корреляций между метриками
        if len(sorted_metrics) > 2:
            response_times = [m.response_time for m in sorted_metrics]
            memory_usage = [m.memory_usage for m in sorted_metrics]
            cpu_usage = [m.cpu_usage for m in sorted_metrics]
            active_tasks = [m.active_tasks for m in sorted_metrics]
            load_scores = [self.analyzer.calculate_load_score(m) for m in sorted_metrics]
            
            # Простой анализ корреляций (коэффициент Пирсона)
            def calculate_correlation(x, y):
                n = len(x)
                if n <= 1:
                    return 0
                sum_x = sum(x)
                sum_y = sum(y)
                sum_x_sq = sum(i**2 for i in x)
                sum_y_sq = sum(i**2 for i in y)
                sum_xy = sum(x[i]*y[i] for i in range(n))
                
                numerator = n * sum_xy - sum_x * sum_y
                denominator = ((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))**0.5
                return numerator / denominator if denominator != 0 else 0
            
            correlation_analysis = {
                'load_vs_response_time': calculate_correlation(load_scores, response_times),
                'load_vs_memory': calculate_correlation(load_scores, memory_usage),
                'load_vs_cpu': calculate_correlation(load_scores, cpu_usage),
                'load_vs_active_tasks': calculate_correlation(load_scores, active_tasks),
                'response_vs_memory': calculate_correlation(response_times, memory_usage),
                'response_vs_cpu': calculate_correlation(response_times, cpu_usage)
            }
        else:
            correlation_analysis = {}
        
        # Прогнозные данные (простая экстраполяция)
        prediction_data = []
        if len(load_scores) >= 5:
            # Простая линейная экстраполяция на основе последних значений
            recent_trend = load_scores[-5:]
            avg_change = 0
            for i in range(1, len(recent_trend)):
                avg_change += (recent_trend[i] - recent_trend[i-1])
            avg_change = avg_change / (len(recent_trend) - 1) if len(recent_trend) > 1 else 0
            
            last_score = load_scores[-1] if load_scores else 0
            for i in range(1, 6):  # Прогноз на 5 временных точек
                predicted_score = last_score + (avg_change * i)
                predicted_score = max(0, min(1, predicted_score))  # Ограничение в диапазоне [0,1]
                prediction_data.append({
                    'time_offset_minutes': i * 10,  # Условно каждые 10 минут
                    'predicted_load_score': predicted_score,
                    'predicted_load_level': self.analyzer.determine_load_level(predicted_score).value
                })
        
        return {
            'time_series': time_series,
            'load_distribution': load_distribution,
            'peak_analysis': peak_analysis,
            'trend_analysis': trend_analysis,
            'correlation_analysis': correlation_analysis,
            'prediction_data': prediction_data
        }
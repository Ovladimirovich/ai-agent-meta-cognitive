import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..core.models import (
    PerformanceAnalysis, PerformanceMetrics, PerformanceComparison,
    InefficientStrategy, AgentInteraction
)

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Анализатор производительности агента"""

    def __init__(self):
        self.baseline_metrics = {
            'avg_execution_time': 5.0,  # секунды
            'avg_confidence': 0.8,
            'avg_tools_used': 2.0,
            'avg_memory_usage': 50.0,  # MB
            'avg_api_calls': 1.0,
            'max_error_rate': 0.1
        }

        self.performance_history = []
        self.strategy_patterns = {}

    async def analyze_performance(self, interaction: AgentInteraction) -> PerformanceAnalysis:
        """Анализ производительности взаимодействия"""
        # Расчет метрик текущего взаимодействия
        metrics = self._calculate_metrics(interaction)

        # Сравнение с базовыми показателями
        comparison = self._compare_with_baseline(metrics)

        # Выявление неэффективных стратегий
        inefficient_strategies = self._identify_inefficient_strategies(
            interaction, metrics, comparison
        )

        # Анализ использования ресурсов
        resource_usage = self._analyze_resource_usage(interaction)

        # Прогноз будущей производительности
        forecast = self._forecast_performance(metrics, self.performance_history)

        analysis = PerformanceAnalysis(
            metrics=metrics,
            comparison=comparison,
            inefficient_strategies=inefficient_strategies,
            resource_usage=resource_usage,
            forecast=forecast
        )

        self.performance_history.append(analysis)
        return analysis

    def _calculate_metrics(self, interaction: AgentInteraction) -> PerformanceMetrics:
        """Расчет метрик производительности"""
        # Качество ответа (простая оценка на основе confidence и длины)
        quality_score = self._calculate_quality_score(interaction)

        return PerformanceMetrics(
            execution_time=interaction.execution_time,
            confidence_score=interaction.response.confidence if interaction.response else 0.0,
            tool_usage_count=len(interaction.tools_used) if interaction.tools_used else 0,
            memory_usage=interaction.memory_usage or 0.0,
            api_calls_count=interaction.api_calls_count or 0,
            error_count=len(interaction.errors) if interaction.errors else 0,
            quality_score=quality_score
        )

    def _calculate_quality_score(self, interaction: AgentInteraction) -> float:
        """Расчет оценки качества ответа"""
        if not interaction.response:
            return 0.0

        base_score = interaction.response.confidence

        # Факторы качества
        factors = {
            'response_length': self._assess_response_length(interaction.response),
            'reasoning_depth': self._assess_reasoning_depth(interaction.reasoning_trace),
            'error_penalty': -0.2 if interaction.errors else 0.0,
            'tool_effectiveness': self._assess_tool_effectiveness(interaction)
        }

        # Взвешенная сумма факторов
        quality = base_score
        for factor, weight in factors.items():
            quality += weight * 0.1  # Небольшое влияние каждого фактора

        return max(0.0, min(1.0, quality))

    def _assess_response_length(self, response) -> float:
        """Оценка длины ответа (0.5 - оптимальная длина)"""
        response_text = str(response.result) if hasattr(response, 'result') else ""
        length = len(response_text)

        if 100 <= length <= 2000:  # Оптимальная длина
            return 0.5
        elif length < 100:
            return 0.2  # Слишком короткий
        else:
            return 0.3  # Слишком длинный

    def _assess_reasoning_depth(self, trace: List[Dict]) -> float:
        """Оценка глубины рассуждений"""
        if not trace:
            return 0.0

        depth_indicators = ['analyze', 'evaluate', 'reason', 'compare', 'conclude']
        depth_steps = 0

        for step in trace:
            step_type = step.get('type', '').lower()
            if any(indicator in step_type for indicator in depth_indicators):
                depth_steps += 1

        return min(depth_steps / 5.0, 1.0)  # Нормализация к 5 шагам

    def _assess_tool_effectiveness(self, interaction: AgentInteraction) -> float:
        """Оценка эффективности использования инструментов"""
        if not interaction.tools_used:
            return 0.5  # Нейтрально, если инструменты не использовались

        tools_count = len(interaction.tools_used)
        execution_time = interaction.execution_time

        # Эффективность = результат / ресурсы
        effectiveness = interaction.response.confidence / (tools_count * execution_time + 1)

        return min(effectiveness * 10, 1.0)  # Нормализация

    def _compare_with_baseline(self, metrics: PerformanceMetrics) -> PerformanceComparison:
        """Сравнение с базовыми показателями"""
        return PerformanceComparison(
            expected_time=self.baseline_metrics['avg_execution_time'],
            expected_tools=int(self.baseline_metrics['avg_tools_used']),
            expected_confidence=self.baseline_metrics['avg_confidence'],
            deviation_time=metrics.execution_time - self.baseline_metrics['avg_execution_time'],
            deviation_tools=metrics.tool_usage_count - self.baseline_metrics['avg_tools_used'],
            deviation_confidence=metrics.confidence_score - self.baseline_metrics['avg_confidence']
        )

    def _identify_inefficient_strategies(self, interaction: AgentInteraction,
                                       metrics: PerformanceMetrics,
                                       comparison: PerformanceComparison) -> List[InefficientStrategy]:
        """Выявление неэффективных стратегий"""
        strategies = []

        # Анализ времени выполнения
        if metrics.execution_time > comparison.expected_time * 1.5:
            strategies.append(InefficientStrategy(
                name="Долгое время выполнения",
                reason=".2f",
                alternative="Оптимизировать выбор инструментов или кэширование результатов",
                confidence=self._calculate_strategy_confidence(metrics, 'time'),
                metrics={
                    'actual_time': metrics.execution_time,
                    'expected_time': comparison.expected_time,
                    'ratio': metrics.execution_time / comparison.expected_time
                }
            ))

        # Анализ использования инструментов
        if metrics.tool_usage_count > comparison.expected_tools * 2:
            strategies.append(InefficientStrategy(
                name="Избыточное использование инструментов",
                reason=f"Использовано {metrics.tool_usage_count} инструментов вместо ожидаемых {comparison.expected_tools}",
                alternative="Упростить цепочку инструментов или использовать более эффективные инструменты",
                confidence=self._calculate_strategy_confidence(metrics, 'tools'),
                metrics={
                    'actual_tools': metrics.tool_usage_count,
                    'expected_tools': comparison.expected_tools,
                    'ratio': metrics.tool_usage_count / max(comparison.expected_tools, 1)
                }
            ))

        # Анализ низкой уверенности
        if metrics.confidence_score < comparison.expected_confidence * 0.7:
            strategies.append(InefficientStrategy(
                name="Низкая уверенность в ответе",
                reason=".2f",
                alternative="Улучшить анализ запроса или использовать дополнительные проверки",
                confidence=self._calculate_strategy_confidence(metrics, 'confidence'),
                metrics={
                    'actual_confidence': metrics.confidence_score,
                    'expected_confidence': comparison.expected_confidence,
                    'ratio': metrics.confidence_score / comparison.expected_confidence
                }
            ))

        # Анализ высокой ошибки
        error_rate = metrics.error_count / max(metrics.tool_usage_count + 1, 1)
        if error_rate > self.baseline_metrics['max_error_rate']:
            strategies.append(InefficientStrategy(
                name="Высокий уровень ошибок",
                reason=f"Уровень ошибок {error_rate:.2f} превышает допустимый {self.baseline_metrics['max_error_rate']}",
                alternative="Добавить проверки ошибок и механизмы восстановления",
                confidence=self._calculate_strategy_confidence(metrics, 'errors'),
                metrics={
                    'error_rate': error_rate,
                    'max_allowed': self.baseline_metrics['max_error_rate'],
                    'error_count': metrics.error_count
                }
            ))

        # Анализ использования памяти
        if metrics.memory_usage > self.baseline_metrics['avg_memory_usage'] * 2:
            strategies.append(InefficientStrategy(
                name="Высокое потребление памяти",
                reason=".1f",
                alternative="Оптимизировать хранение данных или использовать streaming",
                confidence=self._calculate_strategy_confidence(metrics, 'memory'),
                metrics={
                    'actual_memory': metrics.memory_usage,
                    'expected_memory': self.baseline_metrics['avg_memory_usage'],
                    'ratio': metrics.memory_usage / self.baseline_metrics['avg_memory_usage']
                }
            ))

        return strategies

    def _calculate_strategy_confidence(self, metrics: PerformanceMetrics, strategy_type: str) -> float:
        """Расчет уверенности в выявленной неэффективной стратегии"""
        base_confidence = 0.7

        # Корректировка на основе общего качества
        quality_factor = metrics.quality_score

        # Специфические факторы для разных типов стратегий
        if strategy_type == 'time':
            factor = min(metrics.execution_time / 10.0, 2.0)  # Нормализация
        elif strategy_type == 'tools':
            factor = min(metrics.tool_usage_count / 5.0, 2.0)
        elif strategy_type == 'confidence':
            factor = (1.0 - metrics.confidence_score) * 2  # Инвертированная уверенность
        elif strategy_type == 'errors':
            factor = metrics.error_count / 3.0
        elif strategy_type == 'memory':
            factor = metrics.memory_usage / 100.0
        else:
            factor = 1.0

        confidence = base_confidence * quality_factor / factor
        return max(0.1, min(1.0, confidence))

    def _analyze_resource_usage(self, interaction: AgentInteraction) -> Dict[str, Any]:
        """Анализ использования ресурсов"""
        return {
            'memory_peak': interaction.memory_usage or 0.0,
            'cpu_time': interaction.execution_time,
            'api_calls': interaction.api_calls_count or 0,
            'tools_used': len(interaction.tools_used) if interaction.tools_used else 0,
            'efficiency_ratio': self._calculate_efficiency_ratio(interaction),
            'resource_score': self._calculate_resource_score(interaction)
        }

    def _calculate_efficiency_ratio(self, interaction: AgentInteraction) -> float:
        """Расчет коэффициента эффективности использования ресурсов"""
        if not interaction.response:
            return 0.0

        # Эффективность = качество / ресурсы
        quality = interaction.response.confidence
        resources = (interaction.execution_time +
                    (interaction.memory_usage or 0) / 10 +
                    (interaction.api_calls_count or 0) * 0.5)

        return quality / max(resources, 0.1)

    def _calculate_resource_score(self, interaction: AgentInteraction) -> float:
        """Расчет оценки использования ресурсов (0-1, где 1 - оптимально)"""
        memory_score = 1.0 - min((interaction.memory_usage or 0) / 200.0, 1.0)
        time_score = 1.0 - min(interaction.execution_time / 15.0, 1.0)
        api_score = 1.0 - min((interaction.api_calls_count or 0) / 5.0, 1.0)

        return (memory_score + time_score + api_score) / 3.0

    def _forecast_performance(self, current_metrics: PerformanceMetrics,
                            history: List[PerformanceAnalysis]) -> Dict[str, Any]:
        """Прогноз будущей производительности"""
        if len(history) < 3:
            return {
                'trend': 'insufficient_data',
                'predicted_improvement': 0.0,
                'confidence': 0.5
            }

        # Простой трендовый анализ
        recent_analyses = history[-5:]  # Последние 5 анализов

        # Тренд времени выполнения
        time_trend = self._calculate_trend([a.metrics.execution_time for a in recent_analyses])
        confidence_trend = self._calculate_trend([a.metrics.confidence_score for a in recent_analyses])

        # Прогноз улучшения
        predicted_improvement = (time_trend + confidence_trend) / 2.0

        trend_direction = 'improving' if predicted_improvement > 0.1 else 'stable' if predicted_improvement > -0.1 else 'declining'

        return {
            'trend': trend_direction,
            'predicted_improvement': predicted_improvement,
            'confidence': 0.8 if len(history) > 10 else 0.6,
            'time_trend': time_trend,
            'confidence_trend': confidence_trend
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Расчет тренда по последовательности значений"""
        if len(values) < 2:
            return 0.0

        # Линейная регрессия (упрощенная)
        n = len(values)
        x = list(range(n))
        y = values

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / max(n * sum_xx - sum_x * sum_x, 1)

        # Нормализация тренда
        return slope / max(abs(max(values) - min(values)), 1)

    async def analyze_performance_trends(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Анализ трендов производительности по множеству взаимодействий"""
        if not interactions:
            return {'trend': 'no_data', 'description': 'Нет данных для анализа'}

        analyses = []
        for interaction in interactions:
            analysis = await self.analyze_performance(interaction)
            analyses.append(analysis)

        # Агрегация метрик
        avg_execution_time = sum(a.metrics.execution_time for a in analyses) / len(analyses)
        avg_confidence = sum(a.metrics.confidence_score for a in analyses) / len(analyses)
        total_inefficient_strategies = sum(len(a.inefficient_strategies) for a in analyses)

        # Определение общего тренда
        trend = self._forecast_performance(analyses[-1].metrics, analyses[:-1])

        return {
            'trend': trend['trend'],
            'avg_execution_time': avg_execution_time,
            'avg_confidence': avg_confidence,
            'total_inefficient_strategies': total_inefficient_strategies,
            'description': f"Общий тренд производительности: {trend['trend']}",
            'recommendations': self._generate_performance_recommendations(analyses)
        }

    def _generate_performance_recommendations(self, analyses: List[PerformanceAnalysis]) -> List[str]:
        """Генерация рекомендаций по улучшению производительности"""
        recommendations = []

        # Анализ наиболее частых неэффективных стратегий
        all_strategies = []
        for analysis in analyses:
            all_strategies.extend([s.name for s in analysis.inefficient_strategies])

        if all_strategies:
            from collections import Counter
            common_strategies = Counter(all_strategies).most_common(3)

            for strategy, count in common_strategies:
                if strategy == "Долгое время выполнения":
                    recommendations.append("Оптимизировать алгоритмы и кэширование для сокращения времени выполнения")
                elif strategy == "Избыточное использование инструментов":
                    recommendations.append("Упростить цепочки инструментов и использовать более эффективные комбинации")
                elif strategy == "Низкая уверенность в ответе":
                    recommendations.append("Улучшить механизмы анализа и валидации для повышения уверенности")
                elif strategy == "Высокий уровень ошибок":
                    recommendations.append("Добавить дополнительные проверки и обработку ошибок")
                elif strategy == "Высокое потребление памяти":
                    recommendations.append("Оптимизировать использование памяти и добавить очистку кэша")

        return recommendations

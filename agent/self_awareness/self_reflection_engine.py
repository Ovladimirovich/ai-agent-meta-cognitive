import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.models import (
    ReflectionResult, Insight, InsightType, AgentInteraction
)
from .reasoning_trace import ReasoningTraceAnalyzer
from .error_classifier import ErrorClassifier
from .performance_analyzer import PerformanceAnalyzer
from .feedback_processor import FeedbackProcessor

logger = logging.getLogger(__name__)


class SelfReflectionEngine:
    """Движок само-рефлексии агента"""

    def __init__(self, agent_core=None, hybrid_manager=None):
        self.agent_core = agent_core
        self.hybrid_manager = hybrid_manager

        # Инициализация компонентов анализа
        self.reasoning_analyzer = ReasoningTraceAnalyzer(hybrid_manager)
        self.error_classifier = ErrorClassifier()
        self.performance_analyzer = PerformanceAnalyzer()
        self.feedback_processor = FeedbackProcessor(hybrid_manager)

        # История рефлексии
        self.reflection_history: List[ReflectionResult] = []
        self.insights: List[Insight] = []  # Полученные insights

        logger.info("SelfReflectionEngine initialized")

    async def reflect_on_interaction(self, interaction: AgentInteraction) -> ReflectionResult:
        """Рефлексия над завершенным взаимодействием"""
        reflection_start = time.time()

        logger.info(f"Starting reflection on interaction {interaction.id}")

        try:
            # Анализ трассировки рассуждений
            reasoning_analysis = await self.reasoning_analyzer.analyze_trace(
                interaction.reasoning_trace
            )

            # Классификация ошибок и проблем
            error_analysis = await self.error_classifier.classify_errors(interaction)

            # Анализ производительности
            performance_analysis = await self.performance_analyzer.analyze_performance(
                interaction
            )

            # Обработка обратной связи (если есть)
            feedback_analysis = None
            if interaction.feedback:
                feedback_analysis = await self.feedback_processor.process_feedback(
                    interaction.feedback
                )

            # Генерация insights и рекомендаций
            insights = await self._generate_insights(
                reasoning_analysis, error_analysis, performance_analysis, feedback_analysis
            )

            # Сохранение рефлексии
            reflection = ReflectionResult(
                interaction_id=interaction.id,
                reasoning_analysis=reasoning_analysis,
                error_analysis=error_analysis,
                performance_analysis=performance_analysis,
                feedback_analysis=feedback_analysis,
                insights=insights,
                reflection_time=time.time() - reflection_start,
                timestamp=datetime.now()
            )

            self.reflection_history.append(reflection)
            self.insights.extend(insights)

            logger.info(f"Reflection completed for interaction {interaction.id}, "
                       f"generated {len(insights)} insights")

            return reflection

        except Exception as e:
            logger.error(f"Error during reflection on interaction {interaction.id}: {e}")
            # Возвращаем базовый результат рефлексии в случае ошибки
            return ReflectionResult(
                interaction_id=interaction.id,
                reasoning_analysis=None,
                error_analysis=None,
                performance_analysis=None,
                feedback_analysis=None,
                insights=[],
                reflection_time=time.time() - reflection_start,
                timestamp=datetime.now()
            )

    async def _generate_insights(self, reasoning_analysis, error_analysis,
                               performance_analysis, feedback_analysis) -> List[Insight]:
        """Генерация insights на основе всех анализов"""
        insights = []

        # Insights из анализа ошибок
        if error_analysis and error_analysis.patterns:
            error_insights = await self._generate_error_insights(error_analysis)
            insights.extend(error_insights)

        # Insights из анализа производительности
        if performance_analysis and performance_analysis.inefficient_strategies:
            performance_insights = await self._generate_performance_insights(performance_analysis)
            insights.extend(performance_insights)

        # Insights из анализа рассуждений
        if reasoning_analysis and reasoning_analysis.issues:
            reasoning_insights = await self._generate_reasoning_insights(reasoning_analysis)
            insights.extend(reasoning_insights)

        # Insights из обратной связи
        if feedback_analysis and feedback_analysis.insights:
            insights.extend(feedback_analysis.insights)

        # Генерация мета-insights (insights об patterns)
        if len(insights) > 2:
            meta_insights = await self._generate_meta_insights(insights)
            insights.extend(meta_insights)

        return insights

    async def _generate_error_insights(self, error_analysis) -> List[Insight]:
        """Генерация insights из анализа ошибок"""
        insights = []

        for pattern in error_analysis.patterns:
            insight = Insight(
                id=f"error_pattern_{pattern.pattern_id}_{int(time.time())}",
                type=InsightType.ERROR_PATTERN,
                description=f"Обнаружен паттерн ошибок: {pattern.description}",
                confidence=pattern.confidence,
                recommendation=pattern.recommendation,
                data={
                    'pattern_id': pattern.pattern_id,
                    'examples_count': len(pattern.examples),
                    'error_category': pattern.pattern_id.split('_')[0]
                },
                timestamp=datetime.now()
            )
            insights.append(insight)

        # Insight о высокой частоте ошибок
        if error_analysis.metrics.get('total_errors', 0) > 3:
            insight = Insight(
                id=f"high_error_rate_{int(time.time())}",
                type=InsightType.ERROR_PATTERN,
                description="Высокий уровень ошибок во взаимодействии",
                confidence=0.8,
                recommendation="Провести дополнительный анализ причин ошибок и улучшить механизмы валидации",
                data={
                    'total_errors': error_analysis.metrics.get('total_errors', 0),
                    'most_common_category': error_analysis.metrics.get('most_common_category')
                },
                timestamp=datetime.now()
            )
            insights.append(insight)

        return insights

    async def _generate_performance_insights(self, performance_analysis) -> List[Insight]:
        """Генерация insights из анализа производительности"""
        insights = []

        for strategy in performance_analysis.inefficient_strategies:
            insight = Insight(
                id=f"performance_{strategy.name.lower().replace(' ', '_')}_{int(time.time())}",
                type=InsightType.PERFORMANCE_ISSUE,
                description=strategy.reason,
                confidence=strategy.confidence,
                recommendation=strategy.alternative,
                data=strategy.metrics,
                timestamp=datetime.now()
            )
            insights.append(insight)

        # Insight о тренде производительности
        if performance_analysis.forecast.get('trend') == 'declining':
            insight = Insight(
                id=f"performance_trend_declining_{int(time.time())}",
                type=InsightType.PERFORMANCE_ISSUE,
                description="Производительность агента имеет тенденцию к снижению",
                confidence=performance_analysis.forecast.get('confidence', 0.5),
                recommendation="Провести оптимизацию и мониторинг ключевых метрик производительности",
                data={
                    'trend': performance_analysis.forecast.get('trend'),
                    'predicted_improvement': performance_analysis.forecast.get('predicted_improvement')
                },
                timestamp=datetime.now()
            )
            insights.append(insight)

        return insights

    async def _generate_reasoning_insights(self, reasoning_analysis) -> List[Insight]:
        """Генерация insights из анализа рассуждений"""
        insights = []

        # Insights о проблемах в рассуждениях
        for issue in reasoning_analysis.issues:
            insight = Insight(
                id=f"reasoning_issue_{int(time.time())}_{len(insights)}",
                type=InsightType.STRATEGY_IMPROVEMENT,
                description=f"Проблема в рассуждениях: {issue}",
                confidence=0.7,
                recommendation=self._map_reasoning_issue_to_recommendation(issue),
                data={
                    'quality_score': reasoning_analysis.quality_score,
                    'efficiency_score': reasoning_analysis.efficiency.efficiency_score
                },
                timestamp=datetime.now()
            )
            insights.append(insight)

        # Insights о паттернах рассуждений
        effective_patterns = [p for p in reasoning_analysis.patterns if p.effectiveness > 0.7]
        if effective_patterns:
            insight = Insight(
                id=f"effective_reasoning_patterns_{int(time.time())}",
                type=InsightType.STRATEGY_IMPROVEMENT,
                description=f"Обнаружены эффективные паттерны рассуждений: {', '.join(p.pattern_type for p in effective_patterns)}",
                confidence=0.8,
                recommendation="Увеличивать использование эффективных паттернов рассуждений",
                data={
                    'effective_patterns': [p.pattern_type for p in effective_patterns],
                    'avg_effectiveness': sum(p.effectiveness for p in effective_patterns) / len(effective_patterns)
                },
                timestamp=datetime.now()
            )
            insights.append(insight)

        return insights

    def _map_reasoning_issue_to_recommendation(self, issue: str) -> str:
        """Преобразование проблемы рассуждений в рекомендацию"""
        issue_lower = issue.lower()

        if "эффективность" in issue_lower:
            return "Оптимизировать процесс рассуждений для повышения эффективности"
        elif "разнообразие" in issue_lower:
            return "Внедрить больше различных стратегий и паттернов рассуждений"
        elif "глубина" in issue_lower:
            return "Увеличить глубину анализа и детализации рассуждений"
        elif "ветвление" in issue_lower:
            return "Сбалансировать ветвление рассуждений для оптимальной эффективности"
        else:
            return "Проанализировать и улучшить процесс рассуждений"

    async def _generate_meta_insights(self, insights: List[Insight]) -> List[Insight]:
        """Генерация мета-insights (insights об insights)"""
        meta_insights = []

        if len(insights) < 3:
            return meta_insights

        # Анализ распределения типов insights
        insight_types = {}
        for insight in insights:
            insight_types[insight.type.value] = insight_types.get(insight.type.value, 0) + 1

        # Мета-insight о преобладающем типе проблем
        most_common_type = max(insight_types.items(), key=lambda x: x[1])

        if most_common_type[0] == InsightType.ERROR_PATTERN.value and most_common_type[1] > 2:
            meta_insight = Insight(
                id=f"meta_error_focus_{int(time.time())}",
                type=InsightType.STRATEGY_IMPROVEMENT,
                description="Частые ошибки указывают на системные проблемы в работе агента",
                confidence=0.9,
                recommendation="Провести комплексный анализ и рефакторинг ключевых компонентов",
                data={
                    'dominant_type': most_common_type[0],
                    'count': most_common_type[1],
                    'total_insights': len(insights)
                },
                timestamp=datetime.now()
            )
            meta_insights.append(meta_insight)

        elif most_common_type[0] == InsightType.PERFORMANCE_ISSUE.value and most_common_type[1] > 2:
            meta_insight = Insight(
                id=f"meta_performance_focus_{int(time.time())}",
                type=InsightType.PERFORMANCE_ISSUE,
                description="Множественные проблемы производительности требуют оптимизации",
                confidence=0.8,
                recommendation="Приоритизировать оптимизацию производительности и ресурсов",
                data={
                    'dominant_type': most_common_type[0],
                    'count': most_common_type[1],
                    'total_insights': len(insights)
                },
                timestamp=datetime.now()
            )
            meta_insights.append(meta_insight)

        # Мета-insight о высокой уверенности insights
        high_confidence_insights = [i for i in insights if i.confidence > 0.8]
        if len(high_confidence_insights) > len(insights) * 0.6:  # >60% высокая уверенность
            meta_insight = Insight(
                id=f"meta_high_confidence_{int(time.time())}",
                type=InsightType.STRATEGY_IMPROVEMENT,
                description="Большинство выявленных проблем имеют высокую степень уверенности",
                confidence=0.85,
                recommendation="Приоритизировать реализацию рекомендаций с высокой уверенностью",
                data={
                    'high_confidence_count': len(high_confidence_insights),
                    'total_insights': len(insights),
                    'avg_confidence': sum(i.confidence for i in insights) / len(insights)
                },
                timestamp=datetime.now()
            )
            meta_insights.append(meta_insights)

        return meta_insights

    async def get_recent_insights(self, limit: int = 50, insight_type: Optional[InsightType] = None) -> List[Insight]:
        """Получение недавних insights"""
        insights = self.insights[-limit:] if limit > 0 else self.insights

        if insight_type:
            insights = [i for i in insights if i.type == insight_type]

        return insights

    async def get_insights_summary(self) -> Dict[str, Any]:
        """Получение сводки по insights"""
        if not self.insights:
            return {'total_insights': 0, 'summary': 'Нет insights для анализа'}

        # Статистика по типам
        type_stats = {}
        for insight in self.insights:
            type_stats[insight.type.value] = type_stats.get(insight.type.value, 0) + 1

        # Статистика по уверенности
        confidence_stats = {
            'high': len([i for i in self.insights if i.confidence > 0.8]),
            'medium': len([i for i in self.insights if 0.6 <= i.confidence <= 0.8]),
            'low': len([i for i in self.insights if i.confidence < 0.6])
        }

        # Последние insights
        recent_insights = self.insights[-10:]

        return {
            'total_insights': len(self.insights),
            'type_distribution': type_stats,
            'confidence_distribution': confidence_stats,
            'recent_insights': [i.description for i in recent_insights],
            'most_common_type': max(type_stats.items(), key=lambda x: x[1])[0] if type_stats else None
        }

    async def analyze_reflection_effectiveness(self) -> Dict[str, Any]:
        """Анализ эффективности процесса рефлексии"""
        if not self.reflection_history:
            return {'effectiveness': 0, 'summary': 'Нет данных о рефлексии'}

        total_reflections = len(self.reflection_history)
        total_insights = sum(len(r.insights) for r in self.reflection_history)
        avg_reflection_time = sum(r.reflection_time for r in self.reflection_history) / total_reflections

        # Эффективность = insights / время
        effectiveness = total_insights / max(avg_reflection_time, 0.1)

        # Анализ качества insights (на основе уверенности)
        all_insights = []
        for reflection in self.reflection_history:
            all_insights.extend(reflection.insights)

        if all_insights:
            avg_confidence = sum(i.confidence for i in all_insights) / len(all_insights)
            high_quality_insights = len([i for i in all_insights if i.confidence > 0.8])
        else:
            avg_confidence = 0
            high_quality_insights = 0

        return {
            'total_reflections': total_reflections,
            'total_insights': total_insights,
            'avg_reflection_time': avg_reflection_time,
            'effectiveness': effectiveness,
            'avg_insight_confidence': avg_confidence,
            'high_quality_insights': high_quality_insights,
            'insights_per_minute': total_insights / max(avg_reflection_time * total_reflections / 60, 0.1)
        }

    def clear_history(self):
        """Очистка истории рефлексии"""
        self.reflection_history.clear()
        self.insights.clear()
        logger.info("Reflection history cleared")

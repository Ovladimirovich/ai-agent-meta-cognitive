import asyncio
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta

from ..core.models import (
    ReflectionCycleResult, AgentInteraction
)

if TYPE_CHECKING:
    from ..core.agent_core import AgentCore

from .self_reflection_engine import SelfReflectionEngine
from .adaptation_engine import AdaptationEngine

logger = logging.getLogger(__name__)


class ReflectionCycle:
    """Цикл рефлексии для регулярного самоанализа агента"""

    def __init__(self, agent: 'AgentCore', reflection_engine: SelfReflectionEngine,
                 adaptation_engine: AdaptationEngine):
        self.agent = agent
        self.reflection_engine = reflection_engine
        self.adaptation_engine = adaptation_engine

        self.cycle_count = 0
        self.cycle_history: List[ReflectionCycleResult] = []
        self.last_cycle_time: Optional[datetime] = None

        # Настройки цикла
        self.cycle_interval_hours = 24  # Раз в сутки
        self.interaction_limit = 10  # Максимум взаимодействий для анализа
        self.min_interactions_for_cycle = 3  # Минимум взаимодействий для запуска цикла

        # Автоматический цикл
        self.auto_cycle_enabled = True
        self.auto_cycle_task: Optional[asyncio.Task] = None

        logger.info("ReflectionCycle initialized")

    async def run_reflection_cycle(self, force: bool = False) -> ReflectionCycleResult:
        """Выполнение полного цикла рефлексии"""
        self.cycle_count += 1
        cycle_start = datetime.now()

        logger.info(f"Starting reflection cycle #{self.cycle_count}")

        try:
            # Сбор недавних взаимодействий
            recent_interactions = await self._collect_recent_interactions()

            if not force and len(recent_interactions) < self.min_interactions_for_cycle:
                logger.info(f"Not enough interactions for cycle: {len(recent_interactions)} < {self.min_interactions_for_cycle}")
                return ReflectionCycleResult(
                    cycle_number=self.cycle_count,
                    interactions_processed=0,
                    reflections_generated=0,
                    insights_discovered=0,
                    meta_insights_generated=0,
                    adaptations_applied=0
                )

            # Рефлексия над каждым взаимодействием
            reflections = []
            for interaction in recent_interactions:
                try:
                    reflection = await self.reflection_engine.reflect_on_interaction(interaction)
                    reflections.append(reflection)
                except Exception as e:
                    logger.error(f"Error reflecting on interaction {interaction.id}: {e}")
                    continue

            # Агрегация insights из всех рефлексий
            all_insights = []
            for reflection in reflections:
                all_insights.extend(reflection.insights)

            # Генерация мета-insights
            meta_insights = await self._generate_meta_insights(all_insights, reflections)

            # Адаптация на основе всех insights
            all_insights_for_adaptation = all_insights + meta_insights
            adaptation_result = await self.adaptation_engine.adapt_based_on_insights(
                all_insights_for_adaptation
            )

            # Сохранение результатов цикла
            cycle_result = ReflectionCycleResult(
                cycle_number=self.cycle_count,
                interactions_processed=len(recent_interactions),
                reflections_generated=len(reflections),
                insights_discovered=len(all_insights),
                meta_insights_generated=len(meta_insights),
                adaptations_applied=adaptation_result.adaptations_applied
            )

            self.cycle_history.append(cycle_result)
            self.last_cycle_time = datetime.now()

            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Reflection cycle #{self.cycle_count} completed in {cycle_duration:.2f}s: "
                       f"{len(reflections)} reflections, {len(all_insights)} insights, "
                       f"{adaptation_result.adaptations_applied} adaptations")

            return cycle_result

        except Exception as e:
            logger.error(f"Error in reflection cycle #{self.cycle_count}: {e}")
            return ReflectionCycleResult(
                cycle_number=self.cycle_count,
                interactions_processed=0,
                reflections_generated=0,
                insights_discovered=0,
                meta_insights_generated=0,
                adaptations_applied=0
            )

    async def _collect_recent_interactions(self, hours: int = 24) -> List[AgentInteraction]:
        """Сбор недавних взаимодействий для анализа"""
        try:
            if hasattr(self.agent, 'memory_manager') and self.agent.memory_manager:
                # Получение взаимодействий из менеджера памяти
                interactions = await self.agent.memory_manager.get_recent_interactions(
                    limit=self.interaction_limit,
                    hours=hours
                )
                return interactions
            else:
                logger.warning("No memory manager available for collecting interactions")
                return []
        except Exception as e:
            logger.error(f"Error collecting recent interactions: {e}")
            return []

    async def _generate_meta_insights(self, insights: List[Any], reflections: List[Any]) -> List[Any]:
        """Генерация мета-insights на основе паттернов в рефлексиях"""
        meta_insights = []

        if len(insights) < 5:
            return meta_insights

        try:
            # Анализ трендов в ошибках
            error_trend = await self._analyze_error_trends(reflections)
            if error_trend:
                meta_insights.append(error_trend)

            # Анализ трендов производительности
            performance_trend = await self._analyze_performance_trends(reflections)
            if performance_trend:
                meta_insights.append(performance_trend)

            # Анализ эффективности рефлексии
            reflection_effectiveness = await self._analyze_reflection_effectiveness()
            if reflection_effectiveness:
                meta_insights.append(reflection_effectiveness)

            # Генерация стратегических рекомендаций
            strategic_recommendations = await self._generate_strategic_recommendations(insights)
            meta_insights.extend(strategic_recommendations)

        except Exception as e:
            logger.error(f"Error generating meta-insights: {e}")

        return meta_insights

    async def _analyze_error_trends(self, reflections: List[Any]) -> Optional[Any]:
        """Анализ трендов ошибок"""
        try:
            # Сбор всех ошибок из рефлексий
            all_errors = []
            for reflection in reflections:
                if reflection.error_analysis and reflection.error_analysis.errors:
                    all_errors.extend(reflection.error_analysis.errors)

            if len(all_errors) < 3:
                return None

            # Анализ частоты ошибок
            error_categories = {}
            for error in all_errors:
                category = error.category
                error_categories[category] = error_categories.get(category, 0) + 1

            # Определение наиболее частых ошибок
            most_common_error = max(error_categories.items(), key=lambda x: x[1])

            if most_common_error[1] >= 3:  # Если ошибка повторяется 3+ раза
                from ..core.models import Insight, InsightType

                return Insight(
                    id=f"meta_error_trend_{int(datetime.now().timestamp())}",
                    type=InsightType.ERROR_PATTERN,
                    description=f"Частые ошибки типа '{most_common_error[0]}' в нескольких взаимодействиях",
                    confidence=0.8,
                    recommendation="Провести глубокий анализ причин повторяющихся ошибок и внедрить системные улучшения",
                    data={
                        'error_category': most_common_error[0],
                        'frequency': most_common_error[1],
                        'total_interactions': len(reflections)
                    },
                    timestamp=datetime.now()
                )

        except Exception as e:
            logger.error(f"Error analyzing error trends: {e}")

        return None

    async def _analyze_performance_trends(self, reflections: List[Any]) -> Optional[Any]:
        """Анализ трендов производительности"""
        try:
            # Сбор метрик производительности
            performance_scores = []
            for reflection in reflections:
                if reflection.performance_analysis and reflection.performance_analysis.metrics:
                    quality = reflection.performance_analysis.metrics.quality_score
                    if quality is not None:
                        performance_scores.append(quality)

            if len(performance_scores) < 3:
                return None

            # Анализ тренда
            avg_performance = sum(performance_scores) / len(performance_scores)
            recent_performance = sum(performance_scores[-3:]) / 3  # Последние 3

            if recent_performance < avg_performance * 0.8:  # Снижение на 20%
                from ..core.models import Insight, InsightType

                return Insight(
                    id=f"meta_performance_decline_{int(datetime.now().timestamp())}",
                    type=InsightType.PERFORMANCE_ISSUE,
                    description="Производительность агента имеет тенденцию к снижению",
                    confidence=0.75,
                    recommendation="Провести анализ причин снижения производительности и оптимизацию",
                    data={
                        'avg_performance': avg_performance,
                        'recent_performance': recent_performance,
                        'decline_ratio': recent_performance / avg_performance,
                        'measurements': len(performance_scores)
                    },
                    timestamp=datetime.now()
                )

        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")

        return None

    async def _analyze_reflection_effectiveness(self) -> Optional[Any]:
        """Анализ эффективности самого процесса рефлексии"""
        try:
            if len(self.cycle_history) < 2:
                return None

            recent_cycles = self.cycle_history[-3:]  # Последние 3 цикла

            # Анализ роста insights
            insights_trend = [cycle.insights_discovered for cycle in recent_cycles]
            if len(insights_trend) >= 2:
                growth_rate = (insights_trend[-1] - insights_trend[0]) / max(insights_trend[0], 1)

                if growth_rate < -0.5:  # Снижение на 50%
                    from ..core.models import Insight, InsightType

                    return Insight(
                        id=f"meta_reflection_ineffective_{int(datetime.now().timestamp())}",
                        type=InsightType.STRATEGY_IMPROVEMENT,
                        description="Эффективность рефлексии снижается - меньше insights генерируется",
                        confidence=0.7,
                        recommendation="Пересмотреть подход к рефлексии или увеличить частоту циклов",
                        data={
                            'insights_trend': insights_trend,
                            'growth_rate': growth_rate,
                            'cycles_analyzed': len(recent_cycles)
                        },
                        timestamp=datetime.now()
                    )

        except Exception as e:
            logger.error(f"Error analyzing reflection effectiveness: {e}")

        return None

    async def _generate_strategic_recommendations(self, insights: List[Any]) -> List[Any]:
        """Генерация стратегических рекомендаций на основе паттернов"""
        strategic_insights = []

        try:
            # Анализ разнообразия insights
            insight_types = {}
            for insight in insights:
                insight_types[insight.type.value] = insight_types.get(insight.type.value, 0) + 1

            # Если преобладают ошибки - стратегическая рекомендация
            if insight_types.get('error_pattern', 0) > len(insights) * 0.6:
                from ..core.models import Insight, InsightType

                strategic_insights.append(Insight(
                    id=f"strategic_error_focus_{int(datetime.now().timestamp())}",
                    type=InsightType.STRATEGY_IMPROVEMENT,
                    description="Стратегический фокус: систематическое решение проблем с ошибками",
                    confidence=0.9,
                    recommendation="Приоритизировать разработку более надежных компонентов и механизмов обработки ошибок",
                    data={
                        'error_insights_ratio': insight_types.get('error_pattern', 0) / len(insights),
                        'total_insights': len(insights)
                    },
                    timestamp=datetime.now()
                ))

            # Если много проблем производительности
            elif insight_types.get('performance_issue', 0) > len(insights) * 0.4:
                strategic_insights.append(Insight(
                    id=f"strategic_performance_focus_{int(datetime.now().timestamp())}",
                    type=InsightType.PERFORMANCE_ISSUE,
                    description="Стратегический фокус: оптимизация производительности",
                    confidence=0.85,
                    recommendation="Сосредоточиться на оптимизации алгоритмов, кэшировании и распределении ресурсов",
                    data={
                        'performance_insights_ratio': insight_types.get('performance_issue', 0) / len(insights),
                        'total_insights': len(insights)
                    },
                    timestamp=datetime.now()
                ))

        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {e}")

        return strategic_insights

    async def start_auto_cycle(self):
        """Запуск автоматических циклов рефлексии"""
        if self.auto_cycle_enabled and not self.auto_cycle_task:
            self.auto_cycle_task = asyncio.create_task(self._auto_cycle_loop())
            logger.info("Auto reflection cycle started")

    async def stop_auto_cycle(self):
        """Остановка автоматических циклов рефлексии"""
        if self.auto_cycle_task:
            self.auto_cycle_task.cancel()
            try:
                await self.auto_cycle_task
            except asyncio.CancelledError:
                pass
            self.auto_cycle_task = None
            logger.info("Auto reflection cycle stopped")

    async def _auto_cycle_loop(self):
        """Основной цикл автоматической рефлексии"""
        while self.auto_cycle_enabled:
            try:
                # Проверка необходимости запуска цикла
                if self._should_run_cycle():
                    await self.run_reflection_cycle()

                # Ожидание до следующего цикла
                await asyncio.sleep(self.cycle_interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto reflection cycle: {e}")
                await asyncio.sleep(300)  # Ожидание 5 минут перед повтором

    def _should_run_cycle(self) -> bool:
        """Проверка необходимости запуска цикла"""
        if not self.last_cycle_time:
            return True

        # Проверка интервала
        time_since_last_cycle = datetime.now() - self.last_cycle_time
        if time_since_last_cycle.total_seconds() < self.cycle_interval_hours * 3600:
            return False

        return True

    def get_cycle_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение истории циклов рефлексии"""
        recent_cycles = self.cycle_history[-limit:] if limit > 0 else self.cycle_history

        return [
            {
                'cycle_number': cycle.cycle_number,
                'interactions_processed': cycle.interactions_processed,
                'reflections_generated': cycle.reflections_generated,
                'insights_discovered': cycle.insights_discovered,
                'meta_insights_generated': cycle.meta_insights_generated,
                'adaptations_applied': cycle.adaptations_applied,
                'timestamp': self.last_cycle_time if cycle.cycle_number == self.cycle_count else None
            }
            for cycle in recent_cycles
        ]

    async def get_cycle_effectiveness(self) -> Dict[str, Any]:
        """Оценка эффективности циклов рефлексии"""
        if not self.cycle_history:
            return {'effectiveness': 0, 'summary': 'Нет данных о циклах рефлексии'}

        total_cycles = len(self.cycle_history)
        total_insights = sum(cycle.insights_discovered for cycle in self.cycle_history)
        total_adaptations = sum(cycle.adaptations_applied for cycle in self.cycle_history)

        # Эффективность = (insights + adaptations) / cycles
        effectiveness = (total_insights + total_adaptations * 2) / max(total_cycles, 1)  # Адаптации весят больше

        # Анализ тренда
        recent_cycles = self.cycle_history[-5:] if len(self.cycle_history) > 5 else self.cycle_history
        recent_effectiveness = sum((c.insights_discovered + c.adaptations_applied * 2) for c in recent_cycles) / len(recent_cycles)

        trend = 'stable'
        if recent_effectiveness > effectiveness * 1.2:
            trend = 'improving'
        elif recent_effectiveness < effectiveness * 0.8:
            trend = 'declining'

        return {
            'total_cycles': total_cycles,
            'total_insights': total_insights,
            'total_adaptations': total_adaptations,
            'average_effectiveness': effectiveness,
            'recent_effectiveness': recent_effectiveness,
            'trend': trend,
            'recommendations': self._generate_cycle_recommendations(effectiveness, trend)
        }

    def _generate_cycle_recommendations(self, effectiveness: float, trend: str) -> List[str]:
        """Генерация рекомендаций по циклам рефлексии"""
        recommendations = []

        if effectiveness < 2.0:
            recommendations.append("Эффективность циклов рефлексии низкая - рассмотреть увеличение частоты или глубины анализа")

        if trend == 'declining':
            recommendations.append("Эффективность рефлексии снижается - пересмотреть стратегию анализа")

        if trend == 'improving':
            recommendations.append("Эффективность рефлексии растет - продолжить текущий подход")

        return recommendations if recommendations else ["Циклы рефлексии работают оптимально"]

    def configure_cycle(self, interval_hours: int = None, interaction_limit: int = None,
                       min_interactions: int = None, auto_enabled: bool = None):
        """Настройка параметров цикла рефлексии"""
        if interval_hours is not None:
            self.cycle_interval_hours = interval_hours
        if interaction_limit is not None:
            self.interaction_limit = interaction_limit
        if min_interactions is not None:
            self.min_interactions_for_cycle = min_interactions
        if auto_enabled is not None:
            self.auto_cycle_enabled = auto_enabled

        logger.info(f"Reflection cycle configured: interval={self.cycle_interval_hours}h, "
                   f"limit={self.interaction_limit}, min={self.min_interactions_for_cycle}, "
                   f"auto={self.auto_cycle_enabled}")

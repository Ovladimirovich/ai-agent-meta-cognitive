"""
Система предиктивных адаптаций агента
Фаза 3: Обучение и Адаптация
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
import statistics
from collections import defaultdict, deque

from .models import (
    PredictiveInsight,
    AdaptationTrigger,
    ProactiveAdaptation,
    PerformancePrediction,
    RiskAssessment,
    Pattern,
    ProcessedExperience,
    SkillInfo
)

logger = logging.getLogger(__name__)


class PredictiveAdaptation:
    """
    Система предиктивных адаптаций для proactive улучшения работы агента
    """

    def __init__(self, learning_engine=None, skill_development=None, cognitive_maps=None):
        """
        Инициализация системы предиктивных адаптаций

        Args:
            learning_engine: Движок обучения для доступа к данным
            skill_development: Система развития навыков
            cognitive_maps: Когнитивные карты
        """
        self.learning_engine = learning_engine
        self.skill_development = skill_development
        self.cognitive_maps = cognitive_maps

        # История предиктивных инсайтов
        self.insights_history: List[PredictiveInsight] = []

        # Активные адаптационные триггеры
        self.active_triggers: Dict[str, AdaptationTrigger] = {}

        # История примененных адаптаций
        self.adaptation_history: List[ProactiveAdaptation] = []

        # Метрики предиктивной эффективности
        self.predictive_metrics = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'prevented_issues': 0,
            'false_positives': 0,
            'prediction_accuracy': 0.0,
            'adaptation_success_rate': 0.0
        }

        # Кэш предсказаний для оптимизации
        self.prediction_cache: Dict[str, PerformancePrediction] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # Пороги для автоматических адаптаций
        self.adaptation_thresholds = {
            'performance_drop_threshold': 0.7,  # Порог падения производительности
            'error_rate_threshold': 0.15,       # Порог частоты ошибок
            'skill_gap_threshold': 0.3,         # Порог разрыва в навыках
            'resource_pressure_threshold': 0.8  # Порог давления на ресурсы
        }

        # Окно анализа для предсказаний (последние N событий)
        self.analysis_window_size = 50

        logger.info("PredictiveAdaptation system initialized")

    async def analyze_and_predict(self, current_context: Dict[str, Any]) -> List[PredictiveInsight]:
        """
        Анализ текущего состояния и генерация предиктивных инсайтов

        Args:
            current_context: Текущий контекст работы агента

        Returns:
            List[PredictiveInsight]: Список предиктивных инсайтов
        """
        insights = []

        try:
            logger.info("Starting predictive analysis")

            # Анализ трендов производительности
            performance_insights = await self._analyze_performance_trends(current_context)
            insights.extend(performance_insights)

            # Предсказание потенциальных проблем
            risk_insights = await self._predict_potential_issues(current_context)
            insights.extend(risk_insights)

            # Анализ навыков и выявление пробелов
            skill_insights = await self._analyze_skill_gaps(current_context)
            insights.extend(skill_insights)

            # Предсказание нагрузки на ресурсы
            resource_insights = await self._predict_resource_pressure(current_context)
            insights.extend(resource_insights)

            # Анализ паттернов поведения пользователя
            user_behavior_insights = await self._analyze_user_behavior_patterns(current_context)
            insights.extend(user_behavior_insights)

            # Фильтрация и приоритизация инсайтов
            prioritized_insights = await self._prioritize_insights(insights, current_context)

            # Сохранение инсайтов в историю
            for insight in prioritized_insights:
                self.insights_history.append(insight)

            # Ограничение размера истории
            if len(self.insights_history) > 1000:
                self.insights_history = self.insights_history[-1000:]

            logger.info(f"Generated {len(prioritized_insights)} predictive insights")

            return prioritized_insights

        except Exception as e:
            logger.error(f"Predictive analysis failed: {e}")
            return []

    async def _analyze_performance_trends(self, context: Dict[str, Any]) -> List[PredictiveInsight]:
        """
        Анализ трендов производительности

        Args:
            context: Текущий контекст

        Returns:
            List[PredictiveInsight]: Инсайты по трендам производительности
        """
        insights = []

        if not self.learning_engine:
            return insights

        # Получение метрик обучения
        learning_metrics = await self.learning_engine.get_learning_metrics()

        # Анализ эффективности обучения
        current_effectiveness = learning_metrics.average_learning_effectiveness
        recent_trend = self._calculate_trend('learning_effectiveness', current_effectiveness)

        if recent_trend < -0.1:  # Падение эффективности
            insight = PredictiveInsight(
                insight_id=f"perf_trend_{int(datetime.now().timestamp())}",
                type="performance_degradation",
                description="Обнаружено снижение эффективности обучения",
                confidence=abs(recent_trend),
                predicted_impact={
                    'performance_drop': abs(recent_trend) * 0.3,
                    'time_increase': abs(recent_trend) * 0.2
                },
                recommended_actions=[
                    "Оптимизировать алгоритмы извлечения паттернов",
                    "Увеличить частоту обновления когнитивных карт",
                    "Провести анализ качества входных данных"
                ],
                time_horizon=timedelta(hours=24),
                trigger_conditions={'performance_trend': 'decreasing'},
                generated_at=datetime.now()
            )
            insights.append(insight)

        elif recent_trend > 0.1:  # Рост эффективности
            insight = PredictiveInsight(
                insight_id=f"perf_improvement_{int(datetime.now().timestamp())}",
                type="performance_improvement",
                description="Обнаружен рост эффективности обучения",
                confidence=recent_trend,
                predicted_impact={
                    'performance_boost': recent_trend * 0.4,
                    'time_reduction': recent_trend * 0.25
                },
                recommended_actions=[
                    "Зафиксировать текущие настройки как оптимальные",
                    "Расширить применение успешных паттернов",
                    "Увеличить частоту обучения на похожих задачах"
                ],
                time_horizon=timedelta(hours=48),
                trigger_conditions={'performance_trend': 'increasing'},
                generated_at=datetime.now()
            )
            insights.append(insight)

        return insights

    async def _predict_potential_issues(self, context: Dict[str, Any]) -> List[PredictiveInsight]:
        """
        Предсказание потенциальных проблем

        Args:
            context: Текущий контекст

        Returns:
            List[PredictiveInsight]: Инсайты по потенциальным проблемам
        """
        insights = []

        # Анализ частоты ошибок
        error_rate = context.get('recent_error_rate', 0.0)
        if error_rate > self.adaptation_thresholds['error_rate_threshold']:
            insight = PredictiveInsight(
                insight_id=f"error_spike_{int(datetime.now().timestamp())}",
                type="error_rate_increase",
                description=f"Прогнозируется рост частоты ошибок ({error_rate:.2%})",
                confidence=min(error_rate * 2, 1.0),
                predicted_impact={
                    'error_increase': error_rate * 0.5,
                    'user_satisfaction_drop': error_rate * 0.3
                },
                recommended_actions=[
                    "Увеличить валидацию входных данных",
                    "Добавить дополнительные проверки ошибок",
                    "Внедрить резервные стратегии обработки"
                ],
                time_horizon=timedelta(hours=12),
                trigger_conditions={'error_rate': f'>{self.adaptation_thresholds["error_rate_threshold"]}'},
                generated_at=datetime.now()
            )
            insights.append(insight)

        # Предсказание перегрузки на основе паттернов
        if self.learning_engine and hasattr(self.learning_engine, 'pattern_memory'):
            recent_patterns = await self.learning_engine.pattern_memory.get_recent_patterns(limit=20)
            overload_patterns = [p for p in recent_patterns if 'overload' in str(p.trigger_conditions).lower()]

            if len(overload_patterns) > 3:
                insight = PredictiveInsight(
                    insight_id=f"overload_risk_{int(datetime.now().timestamp())}",
                    type="system_overload",
                    description="Обнаружены паттерны перегрузки системы",
                    confidence=len(overload_patterns) / 10.0,
                    predicted_impact={
                        'response_time_increase': 0.4,
                        'error_rate_increase': 0.25
                    },
                    recommended_actions=[
                        "Внедрить ограничение нагрузки",
                        "Оптимизировать использование ресурсов",
                        "Добавить очередь запросов"
                    ],
                    time_horizon=timedelta(hours=6),
                    trigger_conditions={'overload_patterns': f'>{len(overload_patterns)}'},
                    generated_at=datetime.now()
                )
                insights.append(insight)

        return insights

    async def _analyze_skill_gaps(self, context: Dict[str, Any]) -> List[PredictiveInsight]:
        """
        Анализ пробелов в навыках

        Args:
            context: Текущий контекст

        Returns:
            List[PredictiveInsight]: Инсайты по пробелам в навыках
        """
        insights = []

        if not self.skill_development:
            return insights

        # Получение статистики навыков
        skill_stats = self.skill_development.get_skill_stats()

        # Анализ среднего уровня навыков
        avg_skill_level = skill_stats['development']['average_skill_level']

        if avg_skill_level < self.adaptation_thresholds['skill_gap_threshold']:
            insight = PredictiveInsight(
                insight_id=f"skill_gap_{int(datetime.now().timestamp())}",
                type="skill_deficiency",
                description=f"Средний уровень навыков ниже порога ({avg_skill_level:.2f})",
                confidence=1 - avg_skill_level,
                predicted_impact={
                    'task_success_decrease': (1 - avg_skill_level) * 0.3,
                    'adaptation_time_increase': (1 - avg_skill_level) * 0.4
                },
                recommended_actions=[
                    "Интенсифицировать развитие ключевых навыков",
                    "Добавить практику в слабых областях",
                    "Внедрить целенаправленное обучение"
                ],
                time_horizon=timedelta(hours=72),
                trigger_conditions={'avg_skill_level': f'<{self.adaptation_thresholds["skill_gap_threshold"]}'},
                generated_at=datetime.now()
            )
            insights.append(insight)

        # Анализ конкретных слабых навыков
        skill_tree = skill_stats['skill_tree']
        for skill_name, skill_info in self.skill_development.skill_tree.skills.items():
            if skill_info.level < 0.4:  # Критически низкий уровень
                insight = PredictiveInsight(
                    insight_id=f"weak_skill_{skill_name}_{int(datetime.now().timestamp())}",
                    type="specific_skill_gap",
                    description=f"Критически низкий уровень навыка: {skill_name} ({skill_info.level:.2f})",
                    confidence=1 - skill_info.level,
                    predicted_impact={
                        'task_failure_risk': (1 - skill_info.level) * 0.5,
                        'compensation_time': (1 - skill_info.level) * 10
                    },
                    recommended_actions=[
                        f"Приоритетное развитие навыка {skill_name}",
                        f"Дополнительная практика в {skill_info.category.value}",
                        "Временное использование альтернативных подходов"
                    ],
                    time_horizon=timedelta(hours=48),
                    trigger_conditions={'skill_level': f'{skill_name}<0.4'},
                    generated_at=datetime.now()
                )
                insights.append(insight)

        return insights

    async def _predict_resource_pressure(self, context: Dict[str, Any]) -> List[PredictiveInsight]:
        """
        Предсказание давления на ресурсы

        Args:
            context: Текущий контекст

        Returns:
            List[PredictiveInsight]: Инсайты по давлению на ресурсы
        """
        insights = []

        # Анализ использования ресурсов
        memory_usage = context.get('memory_usage', 0.0)
        cpu_usage = context.get('cpu_usage', 0.0)
        active_connections = context.get('active_connections', 0)

        resource_pressure = (memory_usage + cpu_usage) / 2.0

        if resource_pressure > self.adaptation_thresholds['resource_pressure_threshold']:
            insight = PredictiveInsight(
                insight_id=f"resource_pressure_{int(datetime.now().timestamp())}",
                type="resource_overload",
                description=f"Прогнозируется перегрузка ресурсов ({resource_pressure:.1%})",
                confidence=resource_pressure,
                predicted_impact={
                    'performance_degradation': resource_pressure * 0.3,
                    'response_time_increase': resource_pressure * 0.4,
                    'stability_risk': resource_pressure * 0.2
                },
                recommended_actions=[
                    "Оптимизировать использование памяти",
                    "Внедрить кэширование результатов",
                    "Добавить ограничение одновременных задач",
                    "Масштабировать ресурсы при необходимости"
                ],
                time_horizon=timedelta(hours=4),
                trigger_conditions={'resource_pressure': f'>{self.adaptation_thresholds["resource_pressure_threshold"]}'},
                generated_at=datetime.now()
            )
            insights.append(insight)

        # Предсказание на основе трендов
        if len(self.insights_history) > 10:
            recent_resource_insights = [
                i for i in self.insights_history[-20:]
                if i.type == 'resource_overload'
            ]

            if len(recent_resource_insights) >= 3:
                insight = PredictiveInsight(
                    insight_id=f"chronic_resource_issue_{int(datetime.now().timestamp())}",
                    type="chronic_resource_problem",
                    description="Обнаружена хроническая проблема с ресурсами",
                    confidence=0.8,
                    predicted_impact={
                        'long_term_performance_drop': 0.25,
                        'scalability_limitations': 0.4
                    },
                    recommended_actions=[
                        "Провести аудит использования ресурсов",
                        "Реорганизовать архитектуру для лучшей эффективности",
                        "Внедрить автоматическое масштабирование",
                        "Оптимизировать алгоритмы обработки"
                    ],
                    time_horizon=timedelta(days=7),
                    trigger_conditions={'chronic_resource_issues': 'true'},
                    generated_at=datetime.now()
                )
                insights.append(insight)

        return insights

    async def _analyze_user_behavior_patterns(self, context: Dict[str, Any]) -> List[PredictiveInsight]:
        """
        Анализ паттернов поведения пользователя

        Args:
            context: Текущий контекст

        Returns:
            List[PredictiveInsight]: Инсайты по поведению пользователя
        """
        insights = []

        if not self.cognitive_maps:
            return insights

        # Анализ предпочтений пользователя
        user_prefs = await self.cognitive_maps.query_maps({
            'query_type': 'user_preferences',
            'parameters': {}
        })

        if user_prefs.results:
            # Анализ изменения предпочтений
            preference_stability = user_prefs.confidence

            if preference_stability < 0.6:  # Нестабильные предпочтения
                insight = PredictiveInsight(
                    insight_id=f"preference_shift_{int(datetime.now().timestamp())}",
                    type="user_preference_change",
                    description="Обнаружено изменение предпочтений пользователя",
                    confidence=1 - preference_stability,
                    predicted_impact={
                        'response_relevance_decrease': (1 - preference_stability) * 0.3,
                        'user_satisfaction_drop': (1 - preference_stability) * 0.2
                    },
                    recommended_actions=[
                        "Обновить модель предпочтений пользователя",
                        "Адаптировать стиль ответов",
                        "Увеличить сбор обратной связи"
                    ],
                    time_horizon=timedelta(hours=24),
                    trigger_conditions={'preference_stability': f'<0.6'},
                    generated_at=datetime.now()
                )
                insights.append(insight)

        # Анализ паттернов запросов
        query_patterns = await self.cognitive_maps.query_maps({
            'query_type': 'query_patterns',
            'parameters': {}
        })

        if query_patterns.results:
            # Предсказание будущих потребностей
            complexity_trend = query_patterns.results.get('complexity_trend', 0)

            if complexity_trend > 0.2:  # Рост сложности запросов
                insight = PredictiveInsight(
                    insight_id=f"complexity_increase_{int(datetime.now().timestamp())}",
                    type="increasing_complexity",
                    description="Прогнозируется рост сложности запросов пользователя",
                    confidence=complexity_trend,
                    predicted_impact={
                        'processing_time_increase': complexity_trend * 0.4,
                        'skill_requirements_increase': complexity_trend * 0.3
                    },
                    recommended_actions=[
                        "Подготовиться к более сложным задачам",
                        "Развить навыки обработки комплексных запросов",
                        "Оптимизировать алгоритмы анализа"
                    ],
                    time_horizon=timedelta(hours=48),
                    trigger_conditions={'complexity_trend': f'>{complexity_trend}'},
                    generated_at=datetime.now()
                )
                insights.append(insight)

        return insights

    async def _prioritize_insights(self, insights: List[PredictiveInsight], context: Dict[str, Any]) -> List[PredictiveInsight]:
        """
        Приоритизация инсайтов по важности и срочности

        Args:
            insights: Список инсайтов
            context: Контекст для приоритизации

        Returns:
            List[PredictiveInsight]: Приоритизированные инсайты
        """
        if not insights:
            return insights

        # Расчет приоритета для каждого инсайта
        for insight in insights:
            priority_score = self._calculate_insight_priority(insight, context)
            insight.priority_score = priority_score

        # Сортировка по приоритету (высший приоритет первый)
        prioritized = sorted(insights, key=lambda x: x.priority_score, reverse=True)

        # Ограничение количества инсайтов (топ-10)
        return prioritized[:10]

    def _calculate_insight_priority(self, insight: PredictiveInsight, context: Dict[str, Any]) -> float:
        """
        Расчет приоритета инсайта

        Args:
            insight: Предиктивный инсайт
            context: Контекст

        Returns:
            float: Счет приоритета
        """
        priority = 0.0

        # Базовый приоритет по типу инсайта
        type_priorities = {
            'performance_degradation': 0.9,
            'error_rate_increase': 0.85,
            'system_overload': 0.8,
            'resource_overload': 0.75,
            'skill_deficiency': 0.7,
            'specific_skill_gap': 0.65,
            'user_preference_change': 0.6,
            'increasing_complexity': 0.55,
            'performance_improvement': 0.4,
            'chronic_resource_problem': 0.35
        }

        priority += type_priorities.get(insight.type, 0.5)

        # Модификатор по уверенности
        priority *= insight.confidence

        # Модификатор по временному горизонту (срочные выше)
        if insight.time_horizon < timedelta(hours=12):
            priority *= 1.3
        elif insight.time_horizon < timedelta(hours=24):
            priority *= 1.2
        elif insight.time_horizon > timedelta(days=3):
            priority *= 0.8

        # Модификатор по потенциальному воздействию
        max_impact = max(insight.predicted_impact.values()) if insight.predicted_impact else 0
        priority *= (1 + max_impact * 0.5)

        return min(priority, 1.0)

    def _calculate_trend(self, metric_name: str, current_value: float) -> float:
        """
        Расчет тренда метрики на основе истории

        Args:
            metric_name: Название метрики
            current_value: Текущее значение

        Returns:
            float: Тренд (положительный - рост, отрицательный - падение)
        """
        # Получение значений из истории инсайтов
        historical_values = []

        for insight in self.insights_history[-20:]:  # Последние 20 инсайтов
            if hasattr(insight, 'metrics') and metric_name in insight.metrics:
                historical_values.append(insight.metrics[metric_name])

        if len(historical_values) < 3:
            return 0.0

        # Расчет линейного тренда
        try:
            # Простой расчет тренда как разница между средними
            mid_point = len(historical_values) // 2
            first_half = historical_values[:mid_point]
            second_half = historical_values[mid_point:]

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            if first_avg == 0:
                return 0.0

            trend = (second_avg - first_avg) / first_avg

            # Нормализация тренда
            return max(min(trend, 1.0), -1.0)

        except Exception:
            return 0.0

    async def generate_adaptation_plan(self, insights: List[PredictiveInsight]) -> List[ProactiveAdaptation]:
        """
        Генерация плана адаптаций на основе инсайтов

        Args:
            insights: Список предиктивных инсайтов

        Returns:
            List[ProactiveAdaptation]: План адаптаций
        """
        adaptations = []

        for insight in insights:
            if insight.priority_score > 0.6:  # Только высокоприоритетные инсайты
                adaptation = ProactiveAdaptation(
                    adaptation_id=f"adaptation_{insight.insight_id}",
                    trigger_insight=insight,
                    description=f"Адаптация для решения: {insight.description}",
                    changes=self._generate_adaptation_changes(insight),
                    expected_benefits=insight.predicted_impact,
                    risk_assessment=self._assess_adaptation_risk(insight),
                    implementation_priority=insight.priority_score,
                    estimated_effort=self._estimate_adaptation_effort(insight),
                    rollback_plan=self._generate_rollback_plan(insight),
                    created_at=datetime.now()
                )
                adaptations.append(adaptation)

        # Сортировка по приоритету
        adaptations.sort(key=lambda x: x.implementation_priority, reverse=True)

        return adaptations

    def _generate_adaptation_changes(self, insight: PredictiveInsight) -> Dict[str, Any]:
        """
        Генерация изменений для адаптации

        Args:
            insight: Предиктивный инсайт

        Returns:
            Dict[str, Any]: Изменения для адаптации
        """
        changes = {}

        if insight.type == 'performance_degradation':
            changes.update({
                'learning_algorithm': 'optimize_pattern_extraction',
                'cognitive_maps': 'increase_update_frequency',
                'data_quality': 'add_validation_checks'
            })

        elif insight.type == 'error_rate_increase':
            changes.update({
                'error_handling': 'add_fallback_strategies',
                'input_validation': 'strengthen_checks',
                'monitoring': 'increase_error_tracking'
            })

        elif insight.type == 'skill_deficiency':
            changes.update({
                'skill_development': 'intensify_training',
                'task_assignment': 'match_skill_levels',
                'learning_focus': 'target_weak_areas'
            })

        elif insight.type == 'resource_overload':
            changes.update({
                'resource_management': 'implement_limits',
                'caching': 'add_result_caching',
                'load_balancing': 'distribute_workload'
            })

        return changes

    def _assess_adaptation_risk(self, insight: PredictiveInsight) -> RiskAssessment:
        """
        Оценка рисков адаптации

        Args:
            insight: Предиктивный инсайт

        Returns:
            RiskAssessment: Оценка рисков
        """
        # Базовая оценка рисков
        risk_level = "medium"
        risk_factors = []

        # Анализ на основе типа инсайта
        if insight.type in ['performance_degradation', 'system_overload']:
            risk_level = "high"
            risk_factors.extend([
                "Возможное временное снижение производительности",
                "Риск конфликта с существующими оптимизациями"
            ])

        elif insight.type in ['skill_deficiency', 'resource_overload']:
            risk_level = "medium"
            risk_factors.extend([
                "Возможные изменения в поведении агента",
                "Необходимость дополнительного обучения"
            ])

        else:
            risk_level = "low"
            risk_factors.append("Минимальный риск побочных эффектов")

        # Расчет вероятности успеха
        success_probability = insight.confidence * 0.8  # С учетом рисков

        return RiskAssessment(
            risk_level=risk_level,
            risk_factors=risk_factors,
            mitigation_strategies=self._generate_risk_mitigation(insight),
            success_probability=success_probability,
            impact_assessment={
                'worst_case': f"Временное ухудшение {insight.type.replace('_', ' ')}",
                'best_case': f"Улучшение {insight.type.replace('_', ' ')} на {insight.confidence:.1%}"
            }
        )

    def _generate_risk_mitigation(self, insight: PredictiveInsight) -> List[str]:
        """
        Генерация стратегий снижения рисков

        Args:
            insight: Предиктивный инсайт

        Returns:
            List[str]: Стратегии снижения рисков
        """
        mitigation = [
            "Провести тестирование в изолированной среде",
            "Подготовить план отката изменений",
            "Мониторить метрики после внедрения"
        ]

        # Специфические стратегии по типу
        if insight.type == 'performance_degradation':
            mitigation.extend([
                "Начать с минимальных изменений",
                "Иметь возможность быстрого отката"
            ])

        elif insight.type == 'resource_overload':
            mitigation.extend([
                "Мониторить использование ресурсов",
                "Подготовить план масштабирования"
            ])

        return mitigation

    def _estimate_adaptation_effort(self, insight: PredictiveInsight) -> timedelta:
        """
        Оценка трудозатрат на адаптацию

        Args:
            insight: Предиктивный инсайт

        Returns:
            timedelta: Оценка времени
        """
        base_effort_hours = 4  # Базовое время

        # Модификаторы по типу инсайта
        effort_multipliers = {
            'performance_degradation': 2.0,
            'error_rate_increase': 1.5,
            'skill_deficiency': 3.0,
            'resource_overload': 2.5,
            'system_overload': 2.0,
            'user_preference_change': 1.0,
            'increasing_complexity': 1.5
        }

        multiplier = effort_multipliers.get(insight.type, 1.0)
        effort_hours = base_effort_hours * multiplier

        # Корректировка по уверенности (менее уверенные требуют больше времени)
        effort_hours *= (2 - insight.confidence)

        return timedelta(hours=int(effort_hours))

    def _generate_rollback_plan(self, insight: PredictiveInsight) -> Dict[str, Any]:
        """
        Генерация плана отката изменений

        Args:
            insight: Предиктивный инсайт

        Returns:
            Dict[str, Any]: План отката
        """
        return {
            'rollback_steps': [
                "Отменить последние изменения",
                "Восстановить предыдущую конфигурацию",
                "Перезапустить систему",
                "Проверить восстановление функциональности"
            ],
            'rollback_time_estimate': timedelta(minutes=30),
            'data_backup_required': True,
            'system_restart_required': insight.type in ['system_overload', 'resource_overload'],
            'monitoring_after_rollback': True
        }

    async def apply_adaptation(self, adaptation: ProactiveAdaptation) -> Dict[str, Any]:
        """
        Применение адаптации

        Args:
            adaptation: Адаптация для применения

        Returns:
            Dict[str, Any]: Результат применения
        """
        try:
            logger.info(f"Applying adaptation: {adaptation.adaptation_id}")

            # Имитация применения адаптации
            # В реальной реализации здесь был бы код для фактического применения изменений

            success = True
            applied_changes = []
            performance_impact = {}

            # Сохранение в историю
            self.adaptation_history.append(adaptation)

            # Обновление метрик
            self.predictive_metrics['total_predictions'] += 1
            if success:
                self.predictive_metrics['accurate_predictions'] += 1

            self.predictive_metrics['prediction_accuracy'] = (
                self.predictive_metrics['accurate_predictions'] /
                self.predictive_metrics['total_predictions']
            )

            result = {
                'success': success,
                'adaptation_id': adaptation.adaptation_id,
                'applied_changes': applied_changes,
                'performance_impact': performance_impact,
                'timestamp': datetime.now()
            }

            logger.info(f"Adaptation applied successfully: {adaptation.adaptation_id}")
            return result

        except Exception as e:
            logger.error(f"Adaptation application failed: {e}")
            return {
                'success': False,
                'adaptation_id': adaptation.adaptation_id,
                'error': str(e),
                'timestamp': datetime.now()
            }

    def get_predictive_stats(self) -> Dict[str, Any]:
        """
        Получение статистики предиктивной системы

        Returns:
            Dict[str, Any]: Статистика
        """
        total_insights = len(self.insights_history)
        recent_insights = len([i for i in self.insights_history
                              if datetime.now() - i.generated_at < timedelta(hours=24)])

        insight_types = {}
        for insight in self.insights_history[-100:]:  # Последние 100
            insight_types[insight.type] = insight_types.get(insight.type, 0) + 1

        return {
            'predictive_metrics': self.predictive_metrics.copy(),
            'insights_stats': {
                'total_insights': total_insights,
                'recent_insights_24h': recent_insights,
                'insight_types_distribution': insight_types
            },
            'adaptation_stats': {
                'total_adaptations': len(self.adaptation_history),
                'active_triggers': len(self.active_triggers),
                'recent_adaptations': len([a for a in self.adaptation_history
                                          if datetime.now() - a.created_at < timedelta(hours=24)])
            },
            'system_health': {
                'prediction_cache_size': len(self.prediction_cache),
                'analysis_window_size': self.analysis_window_size,
                'adaptation_thresholds': self.adaptation_thresholds.copy()
            }
        }

    async def optimize_predictive_model(self) -> Dict[str, Any]:
        """
        Оптимизация предиктивной модели на основе обратной связи

        Returns:
            Dict[str, Any]: Результаты оптимизации
        """
        recommendations = []

        stats = self.get_predictive_stats()

        # Анализ точности предсказаний
        accuracy = stats['predictive_metrics']['prediction_accuracy']
        if accuracy < 0.7:
            recommendations.append({
                'type': 'accuracy_improvement',
                'description': f'Точность предсказаний низкая ({accuracy:.1%})',
                'suggestion': 'Улучшить алгоритмы анализа трендов и корреляций'
            })

        # Анализ частоты инсайтов
        insight_rate = stats['insights_stats']['recent_insights_24h']
        if insight_rate > 20:  # Слишком много инсайтов
            recommendations.append({
                'type': 'insight_filtering',
                'description': f'Слишком много инсайтов ({insight_rate} за 24ч)',
                'suggestion': 'Увеличить пороги срабатывания и улучшить фильтрацию'
            })
        elif insight_rate < 5:  # Слишком мало инсайтов
            recommendations.append({
                'type': 'sensitivity_increase',
                'description': f'Слишком мало инсайтов ({insight_rate} за 24ч)',
                'suggestion': 'Уменьшить пороги обнаружения проблем'
            })

        # Оптимизация порогов
        if recommendations:
            # Автоматическая корректировка порогов
            if 'accuracy_improvement' in [r['type'] for r in recommendations]:
                # Увеличить пороги для снижения ложных срабатываний
                for threshold_name in self.adaptation_thresholds:
                    self.adaptation_thresholds[threshold_name] *= 1.1

            if 'sensitivity_increase' in [r['type'] for r in recommendations]:
                # Уменьшить пороги для увеличения чувствительности
                for threshold_name in self.adaptation_thresholds:
                    self.adaptation_thresholds[threshold_name] *= 0.9

        return {
            'recommendations': recommendations,
            'thresholds_adjusted': bool(recommendations),
            'new_thresholds': self.adaptation_thresholds.copy(),
            'optimization_potential': len(recommendations) * 0.2
        }

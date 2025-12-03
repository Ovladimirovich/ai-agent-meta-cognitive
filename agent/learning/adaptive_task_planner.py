"""
Адаптивный планировщик задач агента
Фаза 3: Обучение и Адаптация
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import logging
import heapq

from .models import (
    TaskPlan,
    TaskStep,
    TaskContext,
    TaskPriority,
    TaskComplexity,
    TaskExecutionResult,
    AdaptationStrategy,
    Pattern,
    ProcessedExperience
)

logger = logging.getLogger(__name__)


class AdaptiveTaskPlanner:
    """
    Адаптивный планировщик задач, который учится на опыте и оптимизирует планирование
    """

    def __init__(self):
        """
        Инициализация адаптивного планировщика задач
        """
        self.task_history: List[TaskExecutionResult] = []
        self.adaptation_strategies: Dict[str, AdaptationStrategy] = {}
        self.task_patterns: Dict[str, List[Pattern]] = {}

        # Статистика планирования
        self.stats = {
            'total_tasks_planned': 0,
            'successful_executions': 0,
            'average_planning_time': 0,
            'adaptation_success_rate': 0,
            'task_complexity_distribution': {},
            'priority_distribution': {}
        }

        # Кэш планов для повторяющихся задач
        self.plan_cache: Dict[str, TaskPlan] = {}

        # Метрики эффективности
        self.performance_metrics = {
            'planning_accuracy': 0.0,
            'execution_efficiency': 0.0,
            'adaptation_effectiveness': 0.0
        }

        logger.info("AdaptiveTaskPlanner initialized")

    async def plan_task(self, task_description: str, context: TaskContext) -> TaskPlan:
        """
        Планирование задачи с учетом контекста и предыдущего опыта

        Args:
            task_description: Описание задачи
            context: Контекст выполнения задачи

        Returns:
            TaskPlan: План выполнения задачи
        """
        planning_start = datetime.now()

        try:
            logger.info(f"Planning task: {task_description}")

            # Проверка кэша планов
            cached_plan = await self._check_plan_cache(task_description, context)
            if cached_plan:
                logger.info("Using cached plan")
                return cached_plan

            # Анализ сложности задачи
            complexity = await self._analyze_task_complexity(task_description, context)

            # Определение приоритета
            priority = await self._determine_task_priority(task_description, context, complexity)

            # Генерация шагов плана
            steps = await self._generate_task_steps(task_description, context, complexity)

            # Оптимизация плана на основе опыта
            optimized_steps = await self._optimize_plan_with_experience(steps, context)

            # Расчет временных оценок
            time_estimates = await self._estimate_execution_times(optimized_steps, context)

            # Создание плана
            plan = TaskPlan(
                task_id=f"task_{int(datetime.now().timestamp())}_{hash(task_description) % 1000}",
                description=task_description,
                steps=optimized_steps,
                estimated_duration=sum(time_estimates.values()),
                priority=priority,
                complexity=complexity,
                context=context,
                created_at=datetime.now(),
                confidence_score=await self._calculate_plan_confidence(optimized_steps, context)
            )

            # Кэширование плана
            await self._cache_plan(plan)

            # Обновление статистики
            self._update_planning_stats(plan, datetime.now() - planning_start)

            logger.info(f"Task planning completed. Steps: {len(plan.steps)}, "
                       f"Duration: {plan.estimated_duration:.1f}s")

            return plan

        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            # Возврат базового плана в случае ошибки
            return TaskPlan(
                task_id=f"fallback_{int(datetime.now().timestamp())}",
                description=task_description,
                steps=[
                    TaskStep(
                        step_id="fallback_step",
                        description="Execute task with basic approach",
                        dependencies=[],
                        estimated_time=30.0,
                        required_skills=["basic_execution"],
                        success_criteria=["Task completed"]
                    )
                ],
                estimated_duration=30.0,
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.MEDIUM,
                context=context,
                created_at=datetime.now(),
                confidence_score=0.3
            )

    async def _check_plan_cache(self, task_description: str, context: TaskContext) -> Optional[TaskPlan]:
        """
        Проверка кэша планов для повторяющихся задач

        Args:
            task_description: Описание задачи
            context: Контекст задачи

        Returns:
            Optional[TaskPlan]: Кэшированный план или None
        """
        cache_key = f"{hash(task_description)}_{context.user_id}_{context.session_id}"

        if cache_key in self.plan_cache:
            cached_plan = self.plan_cache[cache_key]

            # Проверка актуальности кэша (не старше 24 часов)
            if datetime.now() - cached_plan.created_at < timedelta(hours=24):
                # Адаптация кэшированного плана под текущий контекст
                adapted_plan = await self._adapt_cached_plan(cached_plan, context)
                return adapted_plan

            # Удаление устаревшего кэша
            del self.plan_cache[cache_key]

        return None

    async def _adapt_cached_plan(self, cached_plan: TaskPlan, new_context: TaskContext) -> TaskPlan:
        """
        Адаптация кэшированного плана под новый контекст

        Args:
            cached_plan: Кэшированный план
            new_context: Новый контекст

        Returns:
            TaskPlan: Адаптированный план
        """
        adapted_plan = cached_plan.copy()

        # Обновление контекста
        adapted_plan.context = new_context
        adapted_plan.task_id = f"adapted_{cached_plan.task_id}_{int(datetime.now().timestamp())}"

        # Корректировка приоритета на основе контекста
        if new_context.time_pressure:
            adapted_plan.priority = TaskPriority.HIGH
        elif new_context.complexity == "low":
            adapted_plan.priority = TaskPriority.LOW

        # Корректировка временных оценок
        time_multiplier = 1.0
        if new_context.time_pressure:
            time_multiplier = 0.8  # Ускорение под давлением времени
        elif new_context.user_experience_level == "beginner":
            time_multiplier = 1.3  # Замедление для новичков

        adapted_plan.estimated_duration *= time_multiplier
        for step in adapted_plan.steps:
            step.estimated_time *= time_multiplier

        return adapted_plan

    async def _analyze_task_complexity(self, task_description: str, context: TaskContext) -> TaskComplexity:
        """
        Анализ сложности задачи

        Args:
            task_description: Описание задачи
            context: Контекст задачи

        Returns:
            TaskComplexity: Уровень сложности
        """
        complexity_score = 0.5  # Базовая сложность

        # Анализ по ключевым словам
        complex_keywords = ['optimize', 'integrate', 'complex', 'advanced', 'multiple', 'system']
        simple_keywords = ['basic', 'simple', 'single', 'straightforward']

        description_lower = task_description.lower()

        for keyword in complex_keywords:
            if keyword in description_lower:
                complexity_score += 0.2

        for keyword in simple_keywords:
            if keyword in description_lower:
                complexity_score -= 0.1

        # Корректировка по контексту
        if context.time_pressure:
            complexity_score += 0.1  # Давление времени увеличивает сложность

        if context.user_experience_level == "expert":
            complexity_score -= 0.1  # Опыт уменьшает сложность

        if len(context.available_resources) < 3:
            complexity_score += 0.1  # Ограниченные ресурсы увеличивают сложность

        # Определение уровня сложности
        if complexity_score < 0.3:
            return TaskComplexity.LOW
        elif complexity_score < 0.7:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.HIGH

    async def _determine_task_priority(self, task_description: str, context: TaskContext, complexity: TaskComplexity) -> TaskPriority:
        """
        Определение приоритета задачи

        Args:
            task_description: Описание задачи
            context: Контекст задачи
            complexity: Сложность задачи

        Returns:
            TaskPriority: Приоритет задачи
        """
        priority_score = 0.5  # Базовый приоритет

        # Анализ по ключевым словам
        high_priority_keywords = ['urgent', 'critical', 'emergency', 'immediate', 'deadline']
        low_priority_keywords = ['optional', 'nice-to-have', 'future', 'eventually']

        description_lower = task_description.lower()

        for keyword in high_priority_keywords:
            if keyword in description_lower:
                priority_score += 0.3

        for keyword in low_priority_keywords:
            if keyword in description_lower:
                priority_score -= 0.2

        # Корректировка по контексту
        if context.time_pressure:
            priority_score += 0.4

        if context.user_importance == "high":
            priority_score += 0.2

        if complexity == TaskComplexity.HIGH:
            priority_score += 0.1

        # Определение уровня приоритета
        if priority_score < 0.3:
            return TaskPriority.LOW
        elif priority_score < 0.7:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.HIGH

    async def _generate_task_steps(self, task_description: str, context: TaskContext, complexity: TaskComplexity) -> List[TaskStep]:
        """
        Генерация шагов выполнения задачи

        Args:
            task_description: Описание задачи
            context: Контекст задачи
            complexity: Сложность задачи

        Returns:
            List[TaskStep]: Список шагов
        """
        steps = []

        # Базовая декомпозиция задачи
        if complexity == TaskComplexity.LOW:
            # Простые задачи - 2-3 шага
            steps = [
                TaskStep(
                    step_id="analyze_requirements",
                    description="Анализ требований и подготовка",
                    dependencies=[],
                    estimated_time=5.0,
                    required_skills=["analysis"],
                    success_criteria=["Требования поняты"]
                ),
                TaskStep(
                    step_id="execute_task",
                    description=f"Выполнение: {task_description}",
                    dependencies=["analyze_requirements"],
                    estimated_time=15.0,
                    required_skills=["execution"],
                    success_criteria=["Задача выполнена"]
                ),
                TaskStep(
                    step_id="verify_result",
                    description="Проверка результата",
                    dependencies=["execute_task"],
                    estimated_time=3.0,
                    required_skills=["verification"],
                    success_criteria=["Результат проверен"]
                )
            ]
        elif complexity == TaskComplexity.MEDIUM:
            # Средние задачи - 4-5 шагов
            steps = [
                TaskStep(
                    step_id="analyze_requirements",
                    description="Подробный анализ требований",
                    dependencies=[],
                    estimated_time=8.0,
                    required_skills=["analysis", "planning"],
                    success_criteria=["Требования проанализированы"]
                ),
                TaskStep(
                    step_id="gather_resources",
                    description="Сбор необходимых ресурсов",
                    dependencies=["analyze_requirements"],
                    estimated_time=5.0,
                    required_skills=["resource_management"],
                    success_criteria=["Ресурсы собраны"]
                ),
                TaskStep(
                    step_id="execute_main_task",
                    description=f"Основное выполнение: {task_description}",
                    dependencies=["gather_resources"],
                    estimated_time=20.0,
                    required_skills=["execution", "problem_solving"],
                    success_criteria=["Основная задача выполнена"]
                ),
                TaskStep(
                    step_id="handle_edge_cases",
                    description="Обработка особых случаев",
                    dependencies=["execute_main_task"],
                    estimated_time=7.0,
                    required_skills=["error_handling"],
                    success_criteria=["Особые случаи обработаны"]
                ),
                TaskStep(
                    step_id="final_verification",
                    description="Финальная проверка и оптимизация",
                    dependencies=["handle_edge_cases"],
                    estimated_time=5.0,
                    required_skills=["verification", "optimization"],
                    success_criteria=["Результат оптимизирован"]
                )
            ]
        else:
            # Сложные задачи - 6+ шагов
            steps = [
                TaskStep(
                    step_id="comprehensive_analysis",
                    description="Комплексный анализ требований и рисков",
                    dependencies=[],
                    estimated_time=12.0,
                    required_skills=["analysis", "risk_assessment"],
                    success_criteria=["Анализ завершен"]
                ),
                TaskStep(
                    step_id="design_solution",
                    description="Проектирование решения",
                    dependencies=["comprehensive_analysis"],
                    estimated_time=15.0,
                    required_skills=["design", "architecture"],
                    success_criteria=["Решение спроектировано"]
                ),
                TaskStep(
                    step_id="setup_environment",
                    description="Подготовка окружения",
                    dependencies=["design_solution"],
                    estimated_time=8.0,
                    required_skills=["setup", "configuration"],
                    success_criteria=["Окружение готово"]
                ),
                TaskStep(
                    step_id="implement_solution",
                    description="Реализация решения",
                    dependencies=["setup_environment"],
                    estimated_time=30.0,
                    required_skills=["implementation", "coding"],
                    success_criteria=["Решение реализовано"]
                ),
                TaskStep(
                    step_id="test_solution",
                    description="Тестирование решения",
                    dependencies=["implement_solution"],
                    estimated_time=10.0,
                    required_skills=["testing", "validation"],
                    success_criteria=["Тесты пройдены"]
                ),
                TaskStep(
                    step_id="deploy_and_monitor",
                    description="Развертывание и мониторинг",
                    dependencies=["test_solution"],
                    estimated_time=8.0,
                    required_skills=["deployment", "monitoring"],
                    success_criteria=["Решение развернуто"]
                ),
                TaskStep(
                    step_id="documentation_and_cleanup",
                    description="Документирование и очистка",
                    dependencies=["deploy_and_monitor"],
                    estimated_time=5.0,
                    required_skills=["documentation"],
                    success_criteria=["Документация готова"]
                )
            ]

        # Нумерация шагов
        for i, step in enumerate(steps):
            step.step_id = f"step_{i+1}_{step.step_id}"

        return steps

    async def _optimize_plan_with_experience(self, steps: List[TaskStep], context: TaskContext) -> List[TaskStep]:
        """
        Оптимизация плана на основе предыдущего опыта

        Args:
            steps: Исходные шаги плана
            context: Контекст задачи

        Returns:
            List[TaskStep]: Оптимизированные шаги
        """
        optimized_steps = steps.copy()

        # Поиск похожих задач в истории
        similar_tasks = await self._find_similar_tasks(context)

        if similar_tasks:
            # Извлечение успешных паттернов
            success_patterns = await self._extract_success_patterns(similar_tasks)

            # Применение оптимизаций
            optimized_steps = await self._apply_pattern_optimizations(optimized_steps, success_patterns, context)

        # Оптимизация зависимостей
        optimized_steps = await self._optimize_step_dependencies(optimized_steps)

        # Параллелизация где возможно
        optimized_steps = await self._parallelize_steps(optimized_steps, context)

        return optimized_steps

    async def _find_similar_tasks(self, context: TaskContext) -> List[TaskExecutionResult]:
        """
        Поиск похожих задач в истории выполнения

        Args:
            context: Контекст текущей задачи

        Returns:
            List[TaskExecutionResult]: Список похожих задач
        """
        similar_tasks = []

        for task_result in self.task_history[-50:]:  # Последние 50 задач
            similarity_score = await self._calculate_task_similarity(task_result, context)

            if similarity_score > 0.6:  # Порог схожести
                similar_tasks.append(task_result)

        return similar_tasks[:5]  # Ограничение до 5 самых похожих

    async def _calculate_task_similarity(self, task_result: TaskExecutionResult, context: TaskContext) -> float:
        """
        Расчет схожести задач

        Args:
            task_result: Результат выполнения задачи
            context: Контекст текущей задачи

        Returns:
            float: Степень схожести (0.0 to 1.0)
        """
        similarity = 0.0

        # Сравнение по сложности
        if task_result.plan.complexity == context.complexity:
            similarity += 0.3

        # Сравнение по приоритету
        if task_result.plan.priority == context.priority:
            similarity += 0.2

        # Сравнение по доступным ресурсам
        common_resources = set(task_result.plan.context.available_resources) & set(context.available_resources)
        resource_similarity = len(common_resources) / max(len(context.available_resources), 1)
        similarity += resource_similarity * 0.3

        # Сравнение по опыту пользователя
        if task_result.plan.context.user_experience_level == context.user_experience_level:
            similarity += 0.2

        return min(similarity, 1.0)

    async def _extract_success_patterns(self, similar_tasks: List[TaskExecutionResult]) -> List[Dict[str, Any]]:
        """
        Извлечение паттернов успешного выполнения

        Args:
            similar_tasks: Список похожих задач

        Returns:
            List[Dict[str, Any]]: Паттерны успеха
        """
        patterns = []

        successful_tasks = [t for t in similar_tasks if t.success]

        for task in successful_tasks:
            pattern = {
                'duration': task.actual_duration,
                'steps_count': len(task.plan.steps),
                'efficiency': task.efficiency_score,
                'bottlenecks': task.bottlenecks_identified,
                'optimizations': task.applied_optimizations
            }
            patterns.append(pattern)

        return patterns

    async def _apply_pattern_optimizations(self, steps: List[TaskStep], patterns: List[Dict[str, Any]], context: TaskContext) -> List[TaskStep]:
        """
        Применение оптимизаций на основе паттернов

        Args:
            steps: Исходные шаги
            patterns: Паттерны успеха
            context: Контекст задачи

        Returns:
            List[TaskStep]: Оптимизированные шаги
        """
        optimized_steps = steps.copy()

        if not patterns:
            return optimized_steps

        # Анализ паттернов для оптимизаций
        avg_efficiency = sum(p['efficiency'] for p in patterns) / len(patterns)
        avg_duration = sum(p['duration'] for p in patterns) / len(patterns)

        # Оптимизация времени выполнения
        if avg_efficiency > 0.8:
            # Высокая эффективность - можно ускорить некоторые шаги
            for step in optimized_steps:
                if step.estimated_time > 5.0:  # Только значимые шаги
                    step.estimated_time *= 0.9  # Ускорение на 10%

        # Добавление шагов проверки качества если нужно
        if context.quality_requirements == "high":
            quality_step = TaskStep(
                step_id="quality_assurance",
                description="Проверка качества выполнения",
                dependencies=[optimized_steps[-1].step_id],
                estimated_time=5.0,
                required_skills=["quality_assurance"],
                success_criteria=["Качество подтверждено"]
            )
            optimized_steps.append(quality_step)

        return optimized_steps

    async def _optimize_step_dependencies(self, steps: List[TaskStep]) -> List[TaskStep]:
        """
        Оптимизация зависимостей между шагами

        Args:
            steps: Шаги плана

        Returns:
            List[TaskStep]: Шаги с оптимизированными зависимостями
        """
        # Простая топологическая сортировка для выявления возможности параллелизации
        # В данной реализации оставляем как есть, но можно расширить

        return steps

    async def _parallelize_steps(self, steps: List[TaskStep], context: TaskContext) -> List[TaskStep]:
        """
        Параллелизация независимых шагов

        Args:
            steps: Шаги плана
            context: Контекст задачи

        Returns:
            List[TaskStep]: Шаги с возможной параллелизацией
        """
        # Простая логика: шаги без зависимостей могут выполняться параллельно
        parallel_candidates = []

        for step in steps:
            if not step.dependencies:  # Независимые шаги
                parallel_candidates.append(step)

        # Если есть несколько независимых шагов и достаточно ресурсов
        if len(parallel_candidates) > 1 and len(context.available_resources) > 2:
            # Отмечаем возможность параллельного выполнения
            for step in parallel_candidates:
                step.description += " (может выполняться параллельно)"

        return steps

    async def _estimate_execution_times(self, steps: List[TaskStep], context: TaskContext) -> Dict[str, float]:
        """
        Оценка времени выполнения шагов

        Args:
            steps: Шаги плана
            context: Контекст задачи

        Returns:
            Dict[str, float]: Оценки времени по шагам
        """
        time_estimates = {}

        for step in steps:
            base_time = step.estimated_time

            # Корректировка по уровню опыта пользователя
            if context.user_experience_level == "beginner":
                base_time *= 1.5
            elif context.user_experience_level == "expert":
                base_time *= 0.8

            # Корректировка по доступным ресурсам
            resource_multiplier = 1.0
            if len(context.available_resources) < 2:
                resource_multiplier = 1.3  # Недостаток ресурсов замедляет
            elif len(context.available_resources) > 4:
                resource_multiplier = 0.9  # Достаток ресурсов ускоряет

            base_time *= resource_multiplier

            # Корректировка по давлению времени
            if context.time_pressure:
                base_time *= 0.85  # Ускорение под давлением

            time_estimates[step.step_id] = base_time
            step.estimated_time = base_time

        return time_estimates

    async def _calculate_plan_confidence(self, steps: List[TaskStep], context: TaskContext) -> float:
        """
        Расчет уверенности в плане

        Args:
            steps: Шаги плана
            context: Контекст задачи

        Returns:
            float: Уровень уверенности (0.0 to 1.0)
        """
        confidence = 0.5  # Базовая уверенность

        # Фактор количества похожих задач в истории
        similar_tasks_count = len(await self._find_similar_tasks(context))
        confidence += min(similar_tasks_count * 0.1, 0.3)

        # Фактор детализации плана
        if len(steps) > 3:
            confidence += 0.1

        # Фактор доступности ресурсов
        if len(context.available_resources) >= len(steps) * 0.5:
            confidence += 0.1

        return min(confidence, 1.0)

    async def _cache_plan(self, plan: TaskPlan):
        """
        Кэширование плана для повторного использования

        Args:
            plan: План для кэширования
        """
        cache_key = f"{hash(plan.description)}_{plan.context.user_id}_{plan.context.session_id}"
        self.plan_cache[cache_key] = plan

        # Ограничение размера кэша
        if len(self.plan_cache) > 100:
            # Удаление самого старого плана
            oldest_key = min(self.plan_cache.keys(),
                           key=lambda k: self.plan_cache[k].created_at)
            del self.plan_cache[oldest_key]

    def _update_planning_stats(self, plan: TaskPlan, planning_time: timedelta):
        """
        Обновление статистики планирования

        Args:
            plan: Созданный план
            planning_time: Время планирования
        """
        self.stats['total_tasks_planned'] += 1

        # Обновление среднего времени планирования
        current_avg = self.stats['average_planning_time']
        self.stats['average_planning_time'] = (
            current_avg + planning_time.total_seconds()
        ) / 2

        # Обновление распределения сложности
        complexity = plan.complexity.value
        if complexity not in self.stats['task_complexity_distribution']:
            self.stats['task_complexity_distribution'][complexity] = 0
        self.stats['task_complexity_distribution'][complexity] += 1

        # Обновление распределения приоритетов
        priority = plan.priority.value
        if priority not in self.stats['priority_distribution']:
            self.stats['priority_distribution'][priority] = 0
        self.stats['priority_distribution'][priority] += 1

    async def record_task_execution(self, plan: TaskPlan, actual_duration: float, success: bool,
                                  bottlenecks: List[str] = None, optimizations: List[str] = None) -> TaskExecutionResult:
        """
        Запись результатов выполнения задачи

        Args:
            plan: План задачи
            actual_duration: Фактическое время выполнения
            success: Успешность выполнения
            bottlenecks: Выявленные узкие места
            optimizations: Примененные оптимизации

        Returns:
            TaskExecutionResult: Результат выполнения
        """
        efficiency_score = plan.estimated_duration / actual_duration if actual_duration > 0 else 0
        if efficiency_score > 1:
            efficiency_score = 1 / efficiency_score  # Нормализация

        result = TaskExecutionResult(
            task_id=plan.task_id,
            plan=plan,
            actual_duration=actual_duration,
            success=success,
            efficiency_score=efficiency_score,
            bottlenecks_identified=bottlenecks or [],
            applied_optimizations=optimizations or [],
            executed_at=datetime.now()
        )

        self.task_history.append(result)

        # Обновление статистики
        if success:
            self.stats['successful_executions'] += 1

        # Обновление метрик эффективности
        self._update_performance_metrics(result)

        # Извлечение уроков для будущих планов
        await self._learn_from_execution(result)

        logger.info(f"Task execution recorded: {plan.task_id}, success: {success}, "
                   f"efficiency: {efficiency_score:.2f}")

        return result

    def _update_performance_metrics(self, result: TaskExecutionResult):
        """
        Обновление метрик эффективности

        Args:
            result: Результат выполнения задачи
        """
        # Обновление точности планирования
        duration_accuracy = 1 - abs(result.plan.estimated_duration - result.actual_duration) / max(result.plan.estimated_duration, result.actual_duration)
        self.performance_metrics['planning_accuracy'] = (
            self.performance_metrics['planning_accuracy'] + duration_accuracy
        ) / 2

        # Обновление эффективности выполнения
        self.performance_metrics['execution_efficiency'] = (
            self.performance_metrics['execution_efficiency'] + result.efficiency_score
        ) / 2

        # Обновление эффективности адаптаций
        adaptation_effectiveness = 1.0 if result.success else 0.0
        if result.applied_optimizations:
            adaptation_effectiveness *= 1.2  # Бонус за применение оптимизаций

        self.performance_metrics['adaptation_effectiveness'] = (
            self.performance_metrics['adaptation_effectiveness'] + adaptation_effectiveness
        ) / 2

    async def _learn_from_execution(self, result: TaskExecutionResult):
        """
        Извлечение уроков из выполнения задачи

        Args:
            result: Результат выполнения
        """
        # Создание паттернов на основе результатов
        if result.success and result.efficiency_score > 0.8:
            # Успешный эффективный паттерн
            pattern = Pattern(
                id=f"success_pattern_{result.task_id}",
                type="success",
                trigger_conditions={
                    'complexity': result.plan.complexity.value,
                    'priority': result.plan.priority.value,
                    'context': str(result.plan.context.__dict__)
                },
                description=f"Successful execution pattern for {result.plan.description}",
                confidence=result.efficiency_score,
                frequency=1.0,
                last_updated=datetime.now()
            )

            task_key = f"{result.plan.complexity.value}_{result.plan.priority.value}"
            if task_key not in self.task_patterns:
                self.task_patterns[task_key] = []
            self.task_patterns[task_key].append(pattern)

        # Обновление стратегий адаптации
        if result.bottlenecks_identified:
            await self._update_adaptation_strategies(result)

    async def _update_adaptation_strategies(self, result: TaskExecutionResult):
        """
        Обновление стратегий адаптации на основе результатов

        Args:
            result: Результат выполнения задачи
        """
        for bottleneck in result.bottlenecks_identified:
            strategy_key = f"bottleneck_{bottleneck.replace(' ', '_')}"

            if strategy_key not in self.adaptation_strategies:
                self.adaptation_strategies[strategy_key] = AdaptationStrategy(
                    strategy_id=strategy_key,
                    description=f"Strategy to handle {bottleneck}",
                    trigger_conditions={'bottleneck': bottleneck},
                    adaptation_actions=[f"Optimize {bottleneck}"],
                    expected_improvement=0.1,
                    success_rate=0.0,
                    usage_count=0
                )

            strategy = self.adaptation_strategies[strategy_key]
            strategy.usage_count += 1

            if result.success:
                strategy.success_rate = (strategy.success_rate + 1) / 2
            else:
                strategy.success_rate = strategy.success_rate * 0.9  # Плавное снижение

    def get_planning_stats(self) -> Dict[str, Any]:
        """
        Получение статистики планирования

        Returns:
            Dict[str, Any]: Статистика планировщика
        """
        success_rate = (
            self.stats['successful_executions'] / max(self.stats['total_tasks_planned'], 1)
        )

        return {
            'planning_stats': self.stats.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'success_rate': success_rate,
            'cache_size': len(self.plan_cache),
            'patterns_count': sum(len(patterns) for patterns in self.task_patterns.values()),
            'adaptation_strategies_count': len(self.adaptation_strategies),
            'recent_tasks': len(self.task_history[-10:])
        }

    async def optimize_planning_strategy(self) -> Dict[str, Any]:
        """
        Оптимизация стратегии планирования на основе анализа

        Returns:
            Dict[str, Any]: Рекомендации по оптимизации
        """
        recommendations = []

        stats = self.get_planning_stats()

        # Анализ точности планирования
        if stats['performance_metrics']['planning_accuracy'] < 0.7:
            recommendations.append({
                'type': 'planning_accuracy',
                'description': 'Низкая точность оценки времени выполнения',
                'suggestion': 'Улучшить алгоритмы оценки времени на основе исторических данных'
            })

        # Анализ эффективности выполнения
        if stats['performance_metrics']['execution_efficiency'] < 0.6:
            recommendations.append({
                'type': 'execution_efficiency',
                'description': 'Низкая эффективность выполнения планов',
                'suggestion': 'Оптимизировать последовательность шагов и распределение ресурсов'
            })

        # Анализ использования кэша
        cache_hit_rate = len([p for p in self.plan_cache.values()
                             if datetime.now() - p.created_at < timedelta(hours=1)]) / max(len(self.plan_cache), 1)
        if cache_hit_rate < 0.3:
            recommendations.append({
                'type': 'cache_utilization',
                'description': 'Низкая эффективность использования кэша планов',
                'suggestion': 'Улучшить логику кэширования и адаптации планов'
            })

        return {
            'recommendations': recommendations,
            'current_metrics': stats,
            'optimization_potential': len(recommendations) * 0.15
        }

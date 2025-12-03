"""
Мета-система обучения для оптимизации самого процесса обучения
Фаза 3: Обучение и Адаптация
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
import statistics
import json
from collections import defaultdict, deque

from .models import (
    MetaLearningResult,
    LearningResult,
    LearningMetrics,
    Pattern,
    SkillUpdate,
    AdaptationResult,
    AgentExperience,
    ProcessedExperience
)

logger = logging.getLogger(__name__)


class MetaLearningSystem:
    """
    Мета-система обучения, которая анализирует и оптимизирует сам процесс обучения агента
    """

    def __init__(self, learning_engine=None):
        """
        Инициализация мета-системы обучения

        Args:
            learning_engine: Движок обучения для анализа и оптимизации
        """
        self.learning_engine = learning_engine

        # История мета-анализов
        self.meta_analysis_history: List[MetaLearningResult] = []

        # Стратегии обучения и их эффективность
        self.learning_strategies: Dict[str, Dict[str, Any]] = {}

        # Метрики эффективности стратегий
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)

        # Оптимальные условия для различных типов задач
        self.optimal_conditions: Dict[str, Dict[str, Any]] = {}

        # История оптимизаций
        self.optimization_history: List[Dict[str, Any]] = []

        # Параметры мета-обучения
        self.meta_parameters = {
            'analysis_window_days': 7,
            'min_samples_for_analysis': 10,
            'confidence_threshold': 0.7,
            'optimization_frequency_hours': 24,
            'strategy_evaluation_period_days': 3
        }

        # Инициализация базовых стратегий обучения
        self._initialize_learning_strategies()

        logger.info("MetaLearningSystem initialized")

    def _initialize_learning_strategies(self):
        """
        Инициализация базовых стратегий обучения
        """
        self.learning_strategies = {
            'conservative': {
                'description': 'Консервативная стратегия с медленным обучением',
                'parameters': {
                    'pattern_extraction_threshold': 0.8,
                    'skill_update_frequency': 'low',
                    'cognitive_map_updates': 'selective',
                    'adaptation_aggressiveness': 0.3
                },
                'target_scenarios': ['stable_environment', 'high_reliability_required'],
                'expected_effectiveness': 0.7
            },
            'aggressive': {
                'description': 'Агрессивная стратегия с быстрым обучением',
                'parameters': {
                    'pattern_extraction_threshold': 0.6,
                    'skill_update_frequency': 'high',
                    'cognitive_map_updates': 'frequent',
                    'adaptation_aggressiveness': 0.8
                },
                'target_scenarios': ['dynamic_environment', 'rapid_adaptation_needed'],
                'expected_effectiveness': 0.6
            },
            'balanced': {
                'description': 'Сбалансированная стратегия обучения',
                'parameters': {
                    'pattern_extraction_threshold': 0.7,
                    'skill_update_frequency': 'medium',
                    'cognitive_map_updates': 'moderate',
                    'adaptation_aggressiveness': 0.5
                },
                'target_scenarios': ['mixed_environment', 'general_purpose'],
                'expected_effectiveness': 0.8
            },
            'adaptive': {
                'description': 'Адаптивная стратегия, подстраивающаяся под контекст',
                'parameters': {
                    'pattern_extraction_threshold': 'dynamic',
                    'skill_update_frequency': 'context_dependent',
                    'cognitive_map_updates': 'intelligent',
                    'adaptation_aggressiveness': 'context_aware'
                },
                'target_scenarios': ['complex_environment', 'intelligent_adaptation'],
                'expected_effectiveness': 0.9
            }
        }

        logger.info(f"Initialized {len(self.learning_strategies)} learning strategies")

    async def analyze_learning_effectiveness(self, time_window_days: int = 7) -> MetaLearningResult:
        """
        Анализ эффективности обучения за указанный период

        Args:
            time_window_days: Период анализа в днях

        Returns:
            MetaLearningResult: Результаты мета-анализа
        """
        if not self.learning_engine:
            return MetaLearningResult(
                strategy_effectiveness={},
                optimal_conditions={},
                improved_strategies={},
                application_result={}
            )

        try:
            logger.info(f"Starting meta-analysis for {time_window_days} days")

            # Получение данных обучения за период
            learning_data = await self._gather_learning_data(time_window_days)

            if not learning_data['results']:
                logger.warning("No learning data available for analysis")
                return MetaLearningResult(
                    strategy_effectiveness={},
                    optimal_conditions={},
                    improved_strategies={},
                    application_result={'status': 'insufficient_data'}
                )

            # Анализ эффективности стратегий
            strategy_effectiveness = await self._analyze_strategy_effectiveness(learning_data)

            # Определение оптимальных условий
            optimal_conditions = await self._identify_optimal_conditions(learning_data)

            # Генерация улучшенных стратегий
            improved_strategies = await self._generate_improved_strategies(strategy_effectiveness, optimal_conditions)

            # Применение оптимизаций
            application_result = await self._apply_meta_optimizations(improved_strategies)

            result = MetaLearningResult(
                strategy_effectiveness=strategy_effectiveness,
                optimal_conditions=optimal_conditions,
                improved_strategies=improved_strategies,
                application_result=application_result
            )

            # Сохранение результатов
            self.meta_analysis_history.append(result)
            self._update_strategy_performance(strategy_effectiveness)

            logger.info("Meta-analysis completed successfully")
            return result

        except Exception as e:
            logger.error(f"Meta-analysis failed: {e}")
            return MetaLearningResult(
                strategy_effectiveness={},
                optimal_conditions={},
                improved_strategies={},
                application_result={'error': str(e)}
            )

    async def _gather_learning_data(self, days: int) -> Dict[str, Any]:
        """
        Сбор данных обучения за указанный период

        Args:
            days: Количество дней для анализа

        Returns:
            Dict[str, Any]: Собранные данные обучения
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Получение метрик обучения
        metrics = await self.learning_engine.get_learning_metrics(f"{days}d")

        # Получение истории обучения (если доступна)
        learning_history = getattr(self.learning_engine, 'learning_history', [])

        # Фильтрация по времени
        recent_results = [
            result for result in learning_history
            if result.timestamp > cutoff_date
        ]

        # Группировка данных по типам задач
        task_types = defaultdict(list)
        for result in recent_results:
            task_category = self._categorize_task(result.experience_processed)
            task_types[task_category].append(result)

        return {
            'metrics': metrics,
            'results': recent_results,
            'task_types': dict(task_types),
            'time_period': days
        }

    def _categorize_task(self, experience: ProcessedExperience) -> str:
        """
        Категоризация задачи по опыту

        Args:
            experience: Обработанный опыт

        Returns:
            str: Категория задачи
        """
        categories = experience.categories

        # Определение основной категории
        if 'analysis' in categories:
            return 'analysis'
        elif 'search' in categories:
            return 'search'
        elif 'generation' in categories:
            return 'generation'
        elif 'optimization' in categories:
            return 'optimization'
        elif 'error' in categories:
            return 'error_handling'
        else:
            return 'general'

    async def _analyze_strategy_effectiveness(self, learning_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Анализ эффективности различных стратегий обучения

        Args:
            learning_data: Данные обучения

        Returns:
            Dict[str, float]: Эффективность стратегий
        """
        strategy_effectiveness = {}

        results = learning_data['results']
        if not results:
            return strategy_effectiveness

        # Группировка результатов по стратегиям (имитация)
        # В реальной системе стратегии могут быть помечены в результатах
        for strategy_name in self.learning_strategies.keys():
            strategy_results = [
                r for r in results
                if self._classify_result_strategy(r) == strategy_name
            ]

            if strategy_results:
                # Расчет средней эффективности для стратегии
                effectiveness_scores = [r.learning_effectiveness for r in strategy_results]
                avg_effectiveness = statistics.mean(effectiveness_scores)
                strategy_effectiveness[strategy_name] = avg_effectiveness
            else:
                # Использование ожидаемой эффективности для стратегий без данных
                strategy_effectiveness[strategy_name] = self.learning_strategies[strategy_name]['expected_effectiveness']

        return strategy_effectiveness

    def _classify_result_strategy(self, result: LearningResult) -> str:
        """
        Классификация стратегии обучения для результата

        Args:
            result: Результат обучения

        Returns:
            str: Название стратегии
        """
        # Простая логика классификации на основе характеристик результата
        effectiveness = result.learning_effectiveness
        patterns_count = result.patterns_extracted
        skills_count = result.skills_developed

        if effectiveness > 0.8 and patterns_count > 2:
            return 'aggressive'
        elif effectiveness > 0.7 and skills_count > 1:
            return 'balanced'
        elif effectiveness < 0.6:
            return 'conservative'
        else:
            return 'adaptive'

    async def _identify_optimal_conditions(self, learning_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Определение оптимальных условий для различных типов задач

        Args:
            learning_data: Данные обучения

        Returns:
            Dict[str, Dict[str, Any]]: Оптимальные условия
        """
        optimal_conditions = {}

        task_types = learning_data['task_types']

        for task_type, results in task_types.items():
            if len(results) < self.meta_parameters['min_samples_for_analysis']:
                continue

            # Анализ успешных результатов
            successful_results = [r for r in results if r.learning_effectiveness > 0.7]

            if successful_results:
                # Определение общих характеристик успешных результатов
                avg_effectiveness = statistics.mean([r.learning_effectiveness for r in successful_results])
                avg_patterns = statistics.mean([r.patterns_extracted for r in successful_results])
                avg_skills = statistics.mean([r.skills_developed for r in successful_results])

                optimal_conditions[task_type] = {
                    'avg_effectiveness': avg_effectiveness,
                    'optimal_patterns_count': avg_patterns,
                    'optimal_skills_improved': avg_skills,
                    'sample_size': len(successful_results),
                    'confidence': min(len(successful_results) / 10, 1.0)
                }

        return optimal_conditions

    async def _generate_improved_strategies(self, strategy_effectiveness: Dict[str, float],
                                          optimal_conditions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Генерация улучшенных стратегий обучения

        Args:
            strategy_effectiveness: Эффективность текущих стратегий
            optimal_conditions: Оптимальные условия

        Returns:
            Dict[str, Dict[str, Any]]: Улучшенные стратегии
        """
        improved_strategies = {}

        for strategy_name, effectiveness in strategy_effectiveness.items():
            if effectiveness < 0.7:  # Стратегия нуждается в улучшении
                improvements = await self._generate_strategy_improvements(strategy_name, effectiveness, optimal_conditions)
                if improvements:
                    improved_strategies[strategy_name] = {
                        'original_effectiveness': effectiveness,
                        'improvements': improvements,
                        'expected_gain': sum(imp.get('expected_impact', 0) for imp in improvements)
                    }

        return improved_strategies

    async def _generate_strategy_improvements(self, strategy_name: str, current_effectiveness: float,
                                            optimal_conditions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Генерация улучшений для конкретной стратегии

        Args:
            strategy_name: Название стратегии
            current_effectiveness: Текущая эффективность
            optimal_conditions: Оптимальные условия

        Returns:
            List[Dict[str, Any]]: Список улучшений
        """
        improvements = []

        strategy_config = self.learning_strategies[strategy_name]

        # Анализ параметров стратегии
        if strategy_name == 'conservative' and current_effectiveness < 0.7:
            improvements.extend([
                {
                    'parameter': 'pattern_extraction_threshold',
                    'change': 'decrease_by_0.1',
                    'reason': 'Слишком строгий порог извлечения паттернов',
                    'expected_impact': 0.1
                },
                {
                    'parameter': 'skill_update_frequency',
                    'change': 'increase_to_medium',
                    'reason': 'Слишком редкое обновление навыков',
                    'expected_impact': 0.15
                }
            ])

        elif strategy_name == 'aggressive' and current_effectiveness < 0.7:
            improvements.extend([
                {
                    'parameter': 'pattern_extraction_threshold',
                    'change': 'increase_by_0.1',
                    'reason': 'Слишком низкий порог приводит к шуму',
                    'expected_impact': 0.1
                },
                {
                    'parameter': 'adaptation_aggressiveness',
                    'change': 'decrease_by_0.2',
                    'reason': 'Слишком агрессивные адаптации вызывают нестабильность',
                    'expected_impact': 0.2
                }
            ])

        elif strategy_name == 'balanced':
            # Для сбалансированной стратегии улучшения на основе оптимальных условий
            if optimal_conditions:
                avg_optimal_effectiveness = statistics.mean([
                    cond['avg_effectiveness'] for cond in optimal_conditions.values()
                ])

                if avg_optimal_effectiveness > current_effectiveness:
                    improvements.append({
                        'parameter': 'cognitive_map_updates',
                        'change': 'optimize_based_on_optimal_conditions',
                        'reason': 'Оптимизация обновлений карт на основе успешных паттернов',
                        'expected_impact': avg_optimal_effectiveness - current_effectiveness
                    })

        return improvements

    async def _apply_meta_optimizations(self, improved_strategies: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Применение мета-оптимизаций

        Args:
            improved_strategies: Улучшенные стратегии

        Returns:
            Dict[str, Any]: Результаты применения
        """
        application_results = {
            'applied_optimizations': 0,
            'failed_optimizations': 0,
            'performance_impact': {},
            'status': 'completed'
        }

        try:
            for strategy_name, improvements in improved_strategies.items():
                for improvement in improvements['improvements']:
                    success = await self._apply_single_optimization(strategy_name, improvement)
                    if success:
                        application_results['applied_optimizations'] += 1
                        application_results['performance_impact'][f"{strategy_name}_{improvement['parameter']}"] = \
                            improvement.get('expected_impact', 0)
                    else:
                        application_results['failed_optimizations'] += 1

            # Сохранение в историю оптимизаций
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'improved_strategies': improved_strategies,
                'application_results': application_results
            })

        except Exception as e:
            application_results['status'] = 'error'
            application_results['error'] = str(e)
            logger.error(f"Failed to apply meta-optimizations: {e}")

        return application_results

    async def _apply_single_optimization(self, strategy_name: str, improvement: Dict[str, Any]) -> bool:
        """
        Применение одиночной оптимизации

        Args:
            strategy_name: Название стратегии
            improvement: Улучшение для применения

        Returns:
            bool: Успешность применения
        """
        try:
            parameter = improvement['parameter']
            change = improvement['change']

            # Обновление параметров стратегии
            if strategy_name in self.learning_strategies:
                strategy = self.learning_strategies[strategy_name]

                if parameter in strategy['parameters']:
                    current_value = strategy['parameters'][parameter]

                    # Применение изменения
                    if change == 'decrease_by_0.1' and isinstance(current_value, (int, float)):
                        strategy['parameters'][parameter] = current_value - 0.1
                    elif change == 'increase_by_0.1' and isinstance(current_value, (int, float)):
                        strategy['parameters'][parameter] = current_value + 0.1
                    elif change == 'decrease_by_0.2' and isinstance(current_value, (int, float)):
                        strategy['parameters'][parameter] = current_value - 0.2
                    elif change == 'increase_to_medium' and current_value == 'low':
                        strategy['parameters'][parameter] = 'medium'
                    elif change == 'optimize_based_on_optimal_conditions':
                        # Специальная логика для оптимизации
                        strategy['parameters'][parameter] = 'optimized'

                    logger.info(f"Applied optimization to {strategy_name}.{parameter}: {change}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to apply optimization {improvement}: {e}")
            return False

    def _update_strategy_performance(self, strategy_effectiveness: Dict[str, float]):
        """
        Обновление истории эффективности стратегий

        Args:
            strategy_effectiveness: Эффективность стратегий
        """
        for strategy_name, effectiveness in strategy_effectiveness.items():
            self.strategy_performance[strategy_name].append(effectiveness)

            # Ограничение истории (последние 50 измерений)
            if len(self.strategy_performance[strategy_name]) > 50:
                self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-50:]

    async def recommend_learning_strategy(self, context: Dict[str, Any]) -> str:
        """
        Рекомендация оптимальной стратегии обучения для текущего контекста

        Args:
            context: Контекст для рекомендации

        Returns:
            str: Рекомендуемая стратегия
        """
        # Анализ контекста
        context_factors = self._analyze_context_factors(context)

        # Оценка пригодности каждой стратегии
        strategy_scores = {}
        for strategy_name, strategy_config in self.learning_strategies.items():
            score = self._calculate_strategy_fit(strategy_name, context_factors)
            strategy_scores[strategy_name] = score

        # Выбор лучшей стратегии
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])

        logger.info(f"Recommended strategy '{best_strategy[0]}' with score {best_strategy[1]:.2f}")
        return best_strategy[0]

    def _analyze_context_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализ факторов контекста

        Args:
            context: Контекст

        Returns:
            Dict[str, Any]: Факторы контекста
        """
        factors = {
            'environment_stability': context.get('environment_stability', 'medium'),
            'task_complexity': context.get('task_complexity', 'medium'),
            'time_pressure': context.get('time_pressure', False),
            'reliability_requirement': context.get('reliability_requirement', 'medium'),
            'adaptation_needed': context.get('adaptation_needed', 'medium')
        }

        return factors

    def _calculate_strategy_fit(self, strategy_name: str, context_factors: Dict[str, Any]) -> float:
        """
        Расчет пригодности стратегии для контекста

        Args:
            strategy_name: Название стратегии
            context_factors: Факторы контекста

        Returns:
            float: Счет пригодности
        """
        strategy = self.learning_strategies[strategy_name]
        score = 0.0

        # Базовый счет от целевых сценариев
        target_scenarios = strategy['target_scenarios']

        if context_factors['environment_stability'] == 'stable' and 'stable_environment' in target_scenarios:
            score += 0.3
        elif context_factors['environment_stability'] == 'dynamic' and 'dynamic_environment' in target_scenarios:
            score += 0.3

        if context_factors['reliability_requirement'] == 'high' and 'high_reliability_required' in target_scenarios:
            score += 0.3

        if context_factors['adaptation_needed'] == 'high' and 'rapid_adaptation_needed' in target_scenarios:
            score += 0.2

        # Корректировка по эффективности стратегии
        recent_performance = self.strategy_performance[strategy_name]
        if recent_performance:
            avg_performance = statistics.mean(recent_performance[-5:])  # Последние 5 измерений
            score += avg_performance * 0.4

        return min(score, 1.0)

    async def optimize_learning_parameters(self) -> Dict[str, Any]:
        """
        Оптимизация параметров обучения на основе мета-анализа

        Returns:
            Dict[str, Any]: Результаты оптимизации
        """
        optimizations = {
            'parameter_adjustments': {},
            'strategy_recommendations': {},
            'expected_improvements': {},
            'confidence_levels': {}
        }

        try:
            # Анализ последних мета-результатов
            recent_analyses = self.meta_analysis_history[-5:] if self.meta_analysis_history else []

            if recent_analyses:
                # Агрегация рекомендаций
                all_improvements = []
                for analysis in recent_analyses:
                    all_improvements.extend(analysis.improved_strategies.items())

                # Группировка улучшений по параметрам
                parameter_improvements = defaultdict(list)
                for strategy_name, improvements in all_improvements:
                    for improvement in improvements.get('improvements', []):
                        param = improvement['parameter']
                        parameter_improvements[param].append(improvement)

                # Определение наиболее рекомендуемых изменений
                for param, improvements in parameter_improvements.items():
                    if len(improvements) >= 3:  # Консенсус из нескольких анализов
                        avg_impact = statistics.mean([imp.get('expected_impact', 0) for imp in improvements])
                        if avg_impact > 0.05:  # Значимое улучшение
                            optimizations['parameter_adjustments'][param] = {
                                'recommended_change': improvements[0]['change'],
                                'expected_impact': avg_impact,
                                'confidence': min(len(improvements) / 5, 1.0)
                            }

            # Рекомендации по стратегиям
            strategy_scores = {}
            for strategy_name in self.learning_strategies.keys():
                recent_perf = self.strategy_performance[strategy_name][-10:] if self.strategy_performance[strategy_name] else []
                if recent_perf:
                    avg_score = statistics.mean(recent_perf)
                    strategy_scores[strategy_name] = avg_score

            if strategy_scores:
                best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
                optimizations['strategy_recommendations']['primary'] = best_strategy[0]
                optimizations['strategy_recommendations']['confidence'] = best_strategy[1]

        except Exception as e:
            logger.error(f"Failed to optimize learning parameters: {e}")
            optimizations['error'] = str(e)

        return optimizations

    def get_meta_learning_stats(self) -> Dict[str, Any]:
        """
        Получение статистики мета-обучения

        Returns:
            Dict[str, Any]: Статистика мета-системы
        """
        total_analyses = len(self.meta_analysis_history)
        recent_analyses = len([a for a in self.meta_analysis_history
                              if datetime.now() - a.application_result.get('timestamp', datetime.min) < timedelta(days=7)])

        # Статистика стратегий
        strategy_stats = {}
        for strategy_name, performances in self.strategy_performance.items():
            if performances:
                strategy_stats[strategy_name] = {
                    'avg_effectiveness': statistics.mean(performances),
                    'trend': 'improving' if len(performances) > 1 and performances[-1] > performances[0] else 'stable',
                    'measurements': len(performances)
                }

        # Статистика оптимизаций
        total_optimizations = len(self.optimization_history)
        successful_optimizations = sum(1 for opt in self.optimization_history
                                     if opt['application_results'].get('applied_optimizations', 0) > 0)

        return {
            'meta_analysis': {
                'total_analyses': total_analyses,
                'recent_analyses_7d': recent_analyses,
                'avg_analysis_frequency': total_analyses / max((datetime.now() - datetime.min).days, 1)
            },
            'strategy_performance': strategy_stats,
            'optimizations': {
                'total_optimizations': total_optimizations,
                'successful_optimizations': successful_optimizations,
                'success_rate': successful_optimizations / max(total_optimizations, 1)
            },
            'learning_strategies': {
                'total_strategies': len(self.learning_strategies),
                'active_strategies': len([s for s in self.learning_strategies.keys() if self.strategy_performance[s]])
            },
            'system_health': {
                'analysis_window_days': self.meta_parameters['analysis_window_days'],
                'optimization_frequency_hours': self.meta_parameters['optimization_frequency_hours']
            }
        }

    async def run_automated_optimization(self) -> Dict[str, Any]:
        """
        Запуск автоматизированной оптимизации обучения

        Returns:
            Dict[str, Any]: Результаты автоматизированной оптимизации
        """
        logger.info("Starting automated learning optimization")

        results = {
            'meta_analysis_completed': False,
            'optimizations_applied': False,
            'performance_improved': False,
            'details': {}
        }

        try:
            # Проведение мета-анализа
            meta_result = await self.analyze_learning_effectiveness()
            results['meta_analysis_completed'] = True
            results['details']['meta_analysis'] = {
                'strategies_analyzed': len(meta_result.strategy_effectiveness),
                'improvements_found': len(meta_result.improved_strategies)
            }

            # Применение оптимизаций
            if meta_result.improved_strategies:
                optimization_result = await self.optimize_learning_parameters()
                results['optimizations_applied'] = len(optimization_result.get('parameter_adjustments', {})) > 0
                results['details']['optimizations'] = optimization_result

                # Оценка улучшения производительности
                if optimization_result.get('expected_improvements'):
                    total_improvement = sum(optimization_result['expected_improvements'].values())
                    results['performance_improved'] = total_improvement > 0.1

            logger.info("Automated optimization completed successfully")

        except Exception as e:
            logger.error(f"Automated optimization failed: {e}")
            results['error'] = str(e)

        return results

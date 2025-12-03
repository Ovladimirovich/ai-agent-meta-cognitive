"""
Оптимизатор производительности мета-познания
Фаза 4: Продвинутые функции
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Метрики производительности
    """

    def __init__(self):
        self.response_times: List[float] = []
        self.throughput_values: List[float] = []
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        self.error_rates: List[float] = []
        self.cache_hit_rates: List[float] = []
        self.optimization_history: List[Dict[str, Any]] = []

    def add_response_time(self, time: float):
        """Добавление времени ответа"""
        self.response_times.append(time)
        self._limit_history(self.response_times, 1000)

    def add_throughput(self, throughput: float):
        """Добавление пропускной способности"""
        self.throughput_values.append(throughput)
        self._limit_history(self.throughput_values, 1000)

    def add_memory_usage(self, usage: float):
        """Добавление использования памяти"""
        self.memory_usage.append(usage)
        self._limit_history(self.memory_usage, 1000)

    def add_cpu_usage(self, usage: float):
        """Добавление использования CPU"""
        self.cpu_usage.append(usage)
        self._limit_history(self.cpu_usage, 1000)

    def add_error_rate(self, rate: float):
        """Добавление уровня ошибок"""
        self.error_rates.append(rate)
        self._limit_history(self.error_rates, 1000)

    def add_cache_hit_rate(self, rate: float):
        """Добавление коэффициента попадания в кэш"""
        self.cache_hit_rates.append(rate)
        self._limit_history(self.cache_hit_rates, 1000)

    def record_optimization(self, optimization: Dict[str, Any]):
        """Запись оптимизации"""
        self.optimization_history.append({
            'timestamp': datetime.now(),
            **optimization
        })
        self._limit_history(self.optimization_history, 500)

    def _limit_history(self, history_list: list, max_size: int):
        """Ограничение размера истории"""
        if len(history_list) > max_size:
            history_list[:] = history_list[-max_size:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Получение сводки производительности"""
        return {
            'average_response_time': self._calculate_average(self.response_times),
            'average_throughput': self._calculate_average(self.throughput_values),
            'average_memory_usage': self._calculate_average(self.memory_usage),
            'average_cpu_usage': self._calculate_average(self.cpu_usage),
            'average_error_rate': self._calculate_average(self.error_rates),
            'average_cache_hit_rate': self._calculate_average(self.cache_hit_rates),
            'performance_trends': self._analyze_performance_trends(),
            'bottlenecks': self._identify_bottlenecks(),
            'optimization_effectiveness': self._calculate_optimization_effectiveness()
        }

    def _calculate_average(self, values: List[float]) -> float:
        """Расчет среднего значения"""
        return sum(values) / len(values) if values else 0.0

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Анализ трендов производительности"""
        trends = {}

        # Анализ тренда времени ответа
        if len(self.response_times) >= 10:
            recent_avg = sum(self.response_times[-10:]) / 10
            older_avg = sum(self.response_times[:-10]) / max(1, len(self.response_times) - 10)
            if older_avg > 0:
                response_trend = (recent_avg - older_avg) / older_avg
                trends['response_time'] = {
                    'change_percent': response_trend * 100,
                    'direction': 'improving' if response_trend < 0 else 'degrading'
                }

        # Анализ тренда пропускной способности
        if len(self.throughput_values) >= 10:
            recent_avg = sum(self.throughput_values[-10:]) / 10
            older_avg = sum(self.throughput_values[:-10]) / max(1, len(self.throughput_values) - 10)
            if older_avg > 0:
                throughput_trend = (recent_avg - older_avg) / older_avg
                trends['throughput'] = {
                    'change_percent': throughput_trend * 100,
                    'direction': 'improving' if throughput_trend > 0 else 'degrading'
                }

        return trends

    def _identify_bottlenecks(self) -> List[str]:
        """Идентификация узких мест"""
        bottlenecks = []

        # Проверка на высокое использование памяти
        if self.memory_usage and self._calculate_average(self.memory_usage) > 0.8:
            bottlenecks.append("High memory usage detected")

        # Проверка на высокое использование CPU
        if self.cpu_usage and self._calculate_average(self.cpu_usage) > 0.9:
            bottlenecks.append("High CPU usage detected")

        # Проверка на низкую пропускную способность
        if self.throughput_values and self._calculate_average(self.throughput_values) < 5:
            bottlenecks.append("Low throughput detected")

        # Проверка на высокий уровень ошибок
        if self.error_rates and self._calculate_average(self.error_rates) > 0.1:
            bottlenecks.append("High error rate detected")

        return bottlenecks

    def _calculate_optimization_effectiveness(self) -> Dict[str, Any]:
        """Расчет эффективности оптимизаций"""
        if not self.optimization_history:
            return {'effectiveness_score': 0.0, 'optimizations_count': 0}

        total_improvement = sum(opt.get('improvement', 0) for opt in self.optimization_history)
        avg_improvement = total_improvement / len(self.optimization_history)

        return {
            'effectiveness_score': avg_improvement,
            'optimizations_count': len(self.optimization_history),
            'total_improvement': total_improvement
        }


class OptimizationStrategy:
    """
    Стратегия оптимизации
    """

    def __init__(self, name: str, conditions: Dict[str, Any], actions: List[Dict[str, Any]], priority: int = 1):
        self.name = name
        self.conditions = conditions
        self.actions = actions
        self.priority = priority

    async def evaluate(self, metrics: PerformanceMetrics) -> Tuple[bool, float]:
        """
        Оценка стратегии

        Args:
            metrics: Метрики производительности

        Returns:
            Tuple[bool, float]: (применима, уверенность)
        """
        confidence = 1.0

        for condition_key, condition_value in self.conditions.items():
            if not await self._check_condition(condition_key, condition_value, metrics):
                return False, 0.0
            else:
                confidence *= await self._get_condition_confidence(condition_key, condition_value, metrics)

        return True, confidence

    async def _check_condition(self, key: str, value: Any, metrics: PerformanceMetrics) -> bool:
        """Проверка условия"""
        summary = metrics.get_performance_summary()

        if key == 'response_time_threshold':
            return summary['average_response_time'] > value
        elif key == 'memory_usage_threshold':
            return summary['average_memory_usage'] > value
        elif key == 'cpu_usage_threshold':
            return summary['average_cpu_usage'] > value
        elif key == 'error_rate_threshold':
            return summary['average_error_rate'] > value
        elif key == 'throughput_threshold':
            return summary['average_throughput'] < value
        elif key == 'bottleneck_present':
            return value in summary['bottlenecks']

        return False

    async def _get_condition_confidence(self, key: str, value: Any, metrics: PerformanceMetrics) -> float:
        """Получение уверенности условия"""
        return 1.0  # Для простоты


class PerformanceOptimizer:
    """
    Оптимизатор производительности
    Автоматически оптимизирует производительность системы мета-познания
    """

    def __init__(self):
        """
        Инициализация оптимизатора
        """
        self.metrics = PerformanceMetrics()
        self.optimization_strategies: List[OptimizationStrategy] = []
        self._initialize_optimization_strategies()

        # История оптимизаций
        self.optimization_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # Метрики оптимизатора
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.average_optimization_time = 0.0

        logger.info("PerformanceOptimizer initialized")

    def _initialize_optimization_strategies(self):
        """Инициализация стратегий оптимизации"""

        # Стратегия 1: Оптимизация кэширования при низкой пропускной способности
        caching_strategy = OptimizationStrategy(
            name="caching_optimization",
            conditions={
                'throughput_threshold': 10,
                'response_time_threshold': 1.0
            },
            actions=[
                {'type': 'cache', 'action': 'increase_cache_size', 'parameter': 'size_mb', 'value': 100},
                {'type': 'cache', 'action': 'enable_prefetching', 'parameter': 'enabled', 'value': True},
                {'type': 'cache', 'action': 'optimize_cache_policy', 'parameter': 'policy', 'value': 'lru'}
            ],
            priority=3
        )
        self.optimization_strategies.append(caching_strategy)

        # Стратегия 2: Оптимизация памяти при высоком использовании
        memory_strategy = OptimizationStrategy(
            name="memory_optimization",
            conditions={
                'memory_usage_threshold': 0.8
            },
            actions=[
                {'type': 'memory', 'action': 'enable_garbage_collection', 'parameter': 'aggressive', 'value': True},
                {'type': 'memory', 'action': 'reduce_cache_size', 'parameter': 'size_mb', 'value': 50},
                {'type': 'memory', 'action': 'enable_memory_pooling', 'parameter': 'enabled', 'value': True}
            ],
            priority=4
        )
        self.optimization_strategies.append(memory_strategy)

        # Стратегия 3: Оптимизация CPU при высоком использовании
        cpu_strategy = OptimizationStrategy(
            name="cpu_optimization",
            conditions={
                'cpu_usage_threshold': 0.9
            },
            actions=[
                {'type': 'cpu', 'action': 'enable_parallel_processing', 'parameter': 'max_workers', 'value': 4},
                {'type': 'cpu', 'action': 'optimize_algorithms', 'parameter': 'algorithm', 'value': 'async'},
                {'type': 'cpu', 'action': 'reduce_computation_complexity', 'parameter': 'level', 'value': 'medium'}
            ],
            priority=4
        )
        self.optimization_strategies.append(cpu_strategy)

        # Стратегия 4: Оптимизация при высоком уровне ошибок
        error_strategy = OptimizationStrategy(
            name="error_rate_optimization",
            conditions={
                'error_rate_threshold': 0.05
            },
            actions=[
                {'type': 'error_handling', 'action': 'enable_circuit_breaker', 'parameter': 'threshold', 'value': 0.1},
                {'type': 'error_handling', 'action': 'add_retry_logic', 'parameter': 'max_retries', 'value': 3},
                {'type': 'error_handling', 'action': 'improve_error_recovery', 'parameter': 'recovery_time', 'value': 30}
            ],
            priority=5
        )
        self.optimization_strategies.append(error_strategy)

        # Стратегия 5: Комплексная оптимизация при наличии узких мест
        bottleneck_strategy = OptimizationStrategy(
            name="bottleneck_optimization",
            conditions={
                'bottleneck_present': 'High memory usage detected'
            },
            actions=[
                {'type': 'system', 'action': 'enable_resource_monitoring', 'parameter': 'interval', 'value': 10},
                {'type': 'system', 'action': 'optimize_resource_allocation', 'parameter': 'strategy', 'value': 'dynamic'},
                {'type': 'system', 'action': 'enable_auto_scaling', 'parameter': 'enabled', 'value': True}
            ],
            priority=5
        )
        self.optimization_strategies.append(bottleneck_strategy)

    async def optimize_based_on_feedback(self, agent_response: Any, reflection_result: Dict[str, Any],
                                       learning_result: Any) -> Dict[str, Any]:
        """
        Оптимизация на основе обратной связи

        Args:
            agent_response: Ответ агента
            reflection_result: Результат рефлексии
            learning_result: Результат обучения

        Returns:
            Dict[str, Any]: Результат оптимизации
        """
        try:
            # Сбор метрик из обратной связи
            response_time = getattr(agent_response, 'processing_time', 0.5) if agent_response else 0.5
            confidence = getattr(agent_response, 'confidence', 0.5) if agent_response else 0.5

            self.metrics.add_response_time(response_time)

            # Анализ обратной связи
            feedback_analysis = self._analyze_feedback(reflection_result, learning_result)

            # Выбор стратегии оптимизации
            applicable_strategies = []
            for strategy in self.optimization_strategies:
                applicable, confidence_score = await strategy.evaluate(self.metrics)
                if applicable:
                    applicable_strategies.append((strategy, confidence_score))

            if not applicable_strategies:
                return {'optimizations_applied': [], 'improvement': 0.0}

            # Выбор лучшей стратегии
            best_strategy, _ = max(applicable_strategies, key=lambda x: x[0].priority)

            # Применение оптимизации
            optimization_result = await self._apply_optimization_strategy(best_strategy)

            # Оценка улучшения
            improvement = self._estimate_improvement(best_strategy, feedback_analysis)

            result = {
                'strategy_used': best_strategy.name,
                'optimizations_applied': optimization_result,
                'improvement': improvement,
                'feedback_analysis': feedback_analysis
            }

            # Сохранение в историю
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'result': result
            })

            if len(self.optimization_history) > self.max_history_size:
                self.optimization_history = self.optimization_history[-self.max_history_size:]

            return result

        except Exception as e:
            logger.error(f"Feedback-based optimization failed: {e}")
            return {'error': str(e), 'optimizations_applied': [], 'improvement': 0.0}

    async def perform_full_optimization(self) -> Dict[str, Any]:
        """
        Выполнение полной оптимизации

        Returns:
            Dict[str, Any]: Результат оптимизации
        """
        import time
        opt_start = time.time()

        try:
            logger.info("Starting full performance optimization")

            # Анализ текущей производительности
            performance_summary = self.metrics.get_performance_summary()

            # Идентификация проблем
            issues = self._identify_performance_issues(performance_summary)

            # Выбор стратегий оптимизации
            selected_strategies = await self._select_optimization_strategies(issues)

            # Применение оптимизаций
            applied_optimizations = []
            total_improvement = 0.0

            for strategy in selected_strategies:
                result = await self._apply_optimization_strategy(strategy)
                improvement = self._estimate_strategy_improvement(strategy)
                applied_optimizations.append({
                    'strategy': strategy.name,
                    'result': result,
                    'estimated_improvement': improvement
                })
                total_improvement += improvement

            optimization_time = time.time() - opt_start

            # Обновление метрик
            self.total_optimizations += 1
            self.successful_optimizations += 1 if total_improvement > 0 else 0
            self.average_optimization_time = (
                self.average_optimization_time * 0.9 + optimization_time * 0.1
            )

            result = {
                'applied_optimizations': applied_optimizations,
                'total_improvement': total_improvement,
                'optimization_time': optimization_time,
                'performance_before': performance_summary,
                'issues_addressed': len(issues)
            }

            logger.info(f"Full optimization completed with {total_improvement:.2f} improvement")
            return result

        except Exception as e:
            logger.error(f"Full optimization failed: {e}")
            optimization_time = time.time() - opt_start

            return {
                'error': str(e),
                'applied_optimizations': [],
                'total_improvement': 0.0,
                'optimization_time': optimization_time
            }

    def _analyze_feedback(self, reflection_result: Dict[str, Any], learning_result: Any) -> Dict[str, Any]:
        """Анализ обратной связи"""
        analysis = {
            'reflection_quality': 0.0,
            'learning_effectiveness': 0.0,
            'performance_indicators': {}
        }

        # Анализ рефлексии
        if reflection_result and 'insights' in reflection_result:
            analysis['reflection_quality'] = min(len(reflection_result['insights']) / 10, 1.0)

        # Анализ обучения
        if learning_result and hasattr(learning_result, 'learning_effectiveness'):
            analysis['learning_effectiveness'] = learning_result.learning_effectiveness

        # Индикаторы производительности
        if hasattr(learning_result, 'learning_time'):
            analysis['performance_indicators']['learning_time'] = learning_result.learning_time

        return analysis

    def _identify_performance_issues(self, performance_summary: Dict[str, Any]) -> List[str]:
        """Идентификация проблем производительности"""
        issues = []

        # Проверка времени ответа
        if performance_summary['average_response_time'] > 2.0:
            issues.append('high_response_time')

        # Проверка использования памяти
        if performance_summary['average_memory_usage'] > 0.8:
            issues.append('high_memory_usage')

        # Проверка использования CPU
        if performance_summary['average_cpu_usage'] > 0.9:
            issues.append('high_cpu_usage')

        # Проверка уровня ошибок
        if performance_summary['average_error_rate'] > 0.05:
            issues.append('high_error_rate')

        # Проверка пропускной способности
        if performance_summary['average_throughput'] < 10:
            issues.append('low_throughput')

        return issues

    async def _select_optimization_strategies(self, issues: List[str]) -> List[OptimizationStrategy]:
        """Выбор стратегий оптимизации"""
        selected_strategies = []

        for issue in issues:
            # Поиск подходящих стратегий для каждой проблемы
            for strategy in self.optimization_strategies:
                if self._strategy_addresses_issue(strategy, issue):
                    if strategy not in selected_strategies:
                        selected_strategies.append(strategy)

        # Сортировка по приоритету
        selected_strategies.sort(key=lambda s: s.priority, reverse=True)

        return selected_strategies[:3]  # Ограничение до 3 стратегий

    def _strategy_addresses_issue(self, strategy: OptimizationStrategy, issue: str) -> bool:
        """Проверка, решает ли стратегия проблему"""
        issue_mappings = {
            'high_response_time': ['caching_optimization', 'cpu_optimization'],
            'high_memory_usage': ['memory_optimization', 'bottleneck_optimization'],
            'high_cpu_usage': ['cpu_optimization', 'bottleneck_optimization'],
            'high_error_rate': ['error_rate_optimization'],
            'low_throughput': ['caching_optimization', 'cpu_optimization']
        }

        return strategy.name in issue_mappings.get(issue, [])

    async def _apply_optimization_strategy(self, strategy: OptimizationStrategy) -> List[Dict[str, Any]]:
        """Применение стратегии оптимизации"""
        applied_actions = []

        for action in strategy.actions:
            try:
                # Имитация применения действия
                await asyncio.sleep(0.01)  # Имитация времени применения

                applied_actions.append({
                    'action': action['action'],
                    'type': action['type'],
                    'parameter': action['parameter'],
                    'value': action['value'],
                    'status': 'applied'
                })

                # Запись в метрики
                self.metrics.record_optimization({
                    'strategy': strategy.name,
                    'action': action['action'],
                    'improvement': self._estimate_action_improvement(action)
                })

            except Exception as e:
                applied_actions.append({
                    'action': action['action'],
                    'type': action['type'],
                    'error': str(e),
                    'status': 'failed'
                })

        return applied_actions

    def _estimate_improvement(self, strategy: OptimizationStrategy, feedback_analysis: Dict[str, Any]) -> float:
        """Оценка улучшения"""
        base_improvement = 0.0

        # Оценка на основе типа стратегии
        if 'caching' in strategy.name:
            base_improvement = 0.15  # 15% улучшение
        elif 'memory' in strategy.name:
            base_improvement = 0.10  # 10% улучшение
        elif 'cpu' in strategy.name:
            base_improvement = 0.12  # 12% улучшение
        elif 'error' in strategy.name:
            base_improvement = 0.08  # 8% улучшение
        elif 'bottleneck' in strategy.name:
            base_improvement = 0.20  # 20% улучшение

        # Корректировка на основе обратной связи
        feedback_multiplier = 1.0
        if feedback_analysis.get('reflection_quality', 0) > 0.7:
            feedback_multiplier += 0.1
        if feedback_analysis.get('learning_effectiveness', 0) > 0.8:
            feedback_multiplier += 0.1

        return base_improvement * feedback_multiplier

    def _estimate_strategy_improvement(self, strategy: OptimizationStrategy) -> float:
        """Оценка улучшения стратегии"""
        if 'caching' in strategy.name:
            return 0.15
        elif 'memory' in strategy.name:
            return 0.10
        elif 'cpu' in strategy.name:
            return 0.12
        elif 'error' in strategy.name:
            return 0.08
        elif 'bottleneck' in strategy.name:
            return 0.20
        else:
            return 0.05

    def _estimate_action_improvement(self, action: Dict[str, Any]) -> float:
        """Оценка улучшения действия"""
        action_improvements = {
            'increase_cache_size': 0.10,
            'enable_prefetching': 0.05,
            'optimize_cache_policy': 0.08,
            'enable_garbage_collection': 0.07,
            'reduce_cache_size': 0.03,
            'enable_memory_pooling': 0.06,
            'enable_parallel_processing': 0.12,
            'optimize_algorithms': 0.09,
            'reduce_computation_complexity': 0.11,
            'enable_circuit_breaker': 0.04,
            'add_retry_logic': 0.06,
            'improve_error_recovery': 0.05,
            'enable_resource_monitoring': 0.02,
            'optimize_resource_allocation': 0.08,
            'enable_auto_scaling': 0.15
        }

        return action_improvements.get(action['action'], 0.01)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик оптимизатора

        Returns:
            Dict[str, Any]: Метрики
        """
        return {
            'total_optimizations': self.total_optimizations,
            'successful_optimizations': self.successful_optimizations,
            'success_rate': self.successful_optimizations / max(self.total_optimizations, 1),
            'average_optimization_time': self.average_optimization_time,
            'performance_summary': self.metrics.get_performance_summary(),
            'optimization_history_size': len(self.optimization_history)
        }

    def get_optimization_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Получение истории оптимизаций

        Args:
            limit: Максимальное количество записей

        Returns:
            List[Dict[str, Any]]: История оптимизаций
        """
        return self.optimization_history[-limit:] if limit > 0 else self.optimization_history

    async def reset_optimizations(self) -> Dict[str, Any]:
        """
        Сброс оптимизаций

        Returns:
            Dict[str, Any]: Результат сброса
        """
        try:
            # Имитация сброса оптимизаций
            await asyncio.sleep(0.1)

            # Очистка истории
            self.optimization_history.clear()
            self.total_optimizations = 0
            self.successful_optimizations = 0
            self.average_optimization_time = 0.0

            return {
                'status': 'reset',
                'optimizations_cleared': True,
                'metrics_reset': True
            }

        except Exception as e:
            return {'status': 'error', 'error': str(e)}

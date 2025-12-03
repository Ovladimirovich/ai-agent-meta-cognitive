"""
Strategy Selector - Выбор оптимальной стратегии выполнения задач
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import time

logger = logging.getLogger(__name__)


class Strategy(BaseModel):
    """Структура стратегии выполнения"""
    name: str
    tool_chain: List[str]
    priority: int
    estimated_time: float
    resource_requirements: Dict[str, Any]
    success_probability: float
    adaptation_rules: Dict[str, Any] = {}
    constraints: List[str] = []


class StrategyEvaluation(BaseModel):
    """Результат оценки стратегии"""
    strategy: Strategy
    score: float
    reasons: List[str]
    risks: List[str]
    benefits: List[str]


class StrategySelector:
    """Выбор оптимальной стратегии для выполнения задач"""

    def __init__(self, decision_engine=None, performance_monitor=None):
        self.decision_engine = decision_engine
        self.performance_monitor = performance_monitor

        # Предопределенные стратегии
        self.strategies = self._initialize_strategies()

        # История выбора стратегий
        self.selection_history = []

        # Статистика производительности стратегий
        self.strategy_stats = {}

    def _initialize_strategies(self) -> Dict[str, Strategy]:
        """Инициализация базовых стратегий"""
        return {
            'simple_chat': Strategy(
                name='simple_chat',
                tool_chain=['hybrid_models'],
                priority=1,
                estimated_time=1.0,
                resource_requirements={'cpu': 0.2, 'memory': 0.1},
                success_probability=0.8,
                adaptation_rules={'max_complexity': 'low'},
                constraints=['basic_tools_only']
            ),
            'rag_enhanced': Strategy(
                name='rag_enhanced',
                tool_chain=['rag_tool', 'hybrid_models'],
                priority=2,
                estimated_time=2.0,
                resource_requirements={'cpu': 0.4, 'memory': 0.3},
                success_probability=0.9,
                adaptation_rules={'min_complexity': 'low', 'requires_search': True},
                constraints=['rag_available']
            ),
            'multi_tool': Strategy(
                name='multi_tool',
                tool_chain=['rag_tool', 'web_research', 'hybrid_models'],
                priority=3,
                estimated_time=3.0,
                resource_requirements={'cpu': 0.6, 'memory': 0.5},
                success_probability=0.85,
                adaptation_rules={'min_complexity': 'medium', 'requires_research': True},
                constraints=['web_research_available', 'rag_available']
            ),
            'research_mode': Strategy(
                name='research_mode',
                tool_chain=['web_research', 'rag_tool', 'hybrid_models'],
                priority=4,
                estimated_time=4.0,
                resource_requirements={'cpu': 0.8, 'memory': 0.7},
                success_probability=0.75,
                adaptation_rules={'complexity': 'high', 'research_focused': True},
                constraints=['web_research_available', 'rag_available', 'time_tolerant']
            ),
            'learning_mode': Strategy(
                name='learning_mode',
                tool_chain=['rag_tool', 'learning_engine', 'hybrid_models'],
                priority=5,
                estimated_time=2.5,
                resource_requirements={'cpu': 0.5, 'memory': 0.4},
                success_probability=0.7,
                adaptation_rules={'learning_opportunity': True},
                constraints=['learning_engine_available']
            ),
            'fast_response': Strategy(
                name='fast_response',
                tool_chain=['cache_tool', 'hybrid_models'],
                priority=0,
                estimated_time=0.5,
                resource_requirements={'cpu': 0.1, 'memory': 0.05},
                success_probability=0.6,
                adaptation_rules={'time_critical': True, 'cached_available': True},
                constraints=['cache_available']
            ),
            'expert_mode': Strategy(
                name='expert_mode',
                tool_chain=['rag_tool', 'web_research', 'learning_engine', 'hybrid_models'],
                priority=6,
                estimated_time=5.0,
                resource_requirements={'cpu': 0.9, 'memory': 0.8},
                success_probability=0.95,
                adaptation_rules={'complexity': 'expert', 'high_accuracy_required': True},
                constraints=['all_tools_available', 'resource_rich']
            )
        }

    async def select_optimal_strategy(self, task_analysis: Dict[str, Any],
                                    context: Dict[str, Any]) -> Strategy:
        """Выбор оптимальной стратегии для задачи"""
        try:
            start_time = time.time()

            # Оценка всех доступных стратегий
            evaluations = []
            for strategy in self.strategies.values():
                evaluation = await self._evaluate_strategy(strategy, task_analysis, context)
                evaluations.append(evaluation)

            # Выбор лучшей стратегии
            best_evaluation = max(evaluations, key=lambda x: x.score)

            # Сохранение выбора в истории
            self.selection_history.append({
                'task_analysis': task_analysis,
                'context': context,
                'selected_strategy': best_evaluation.strategy.name,
                'score': best_evaluation.score,
                'alternatives': [e.strategy.name for e in evaluations if e != best_evaluation],
                'timestamp': time.time(),
                'selection_time': time.time() - start_time
            })

            # Ограничение истории
            if len(self.selection_history) > 1000:
                self.selection_history = self.selection_history[-500:]

            logger.info(f"Selected strategy: {best_evaluation.strategy.name} "
                       f"with score {best_evaluation.score:.2f}")

            return best_evaluation.strategy

        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            # Возврат fallback стратегии
            return self.strategies['simple_chat']

    async def _evaluate_strategy(self, strategy: Strategy, task_analysis: Dict[str, Any],
                               context: Dict[str, Any]) -> StrategyEvaluation:
        """Оценка стратегии для конкретной задачи"""
        score = 0.0
        reasons = []
        risks = []
        benefits = []

        # Проверка ограничений
        constraints_met, constraint_reasons = self._check_constraints(strategy, context)
        if not constraints_met:
            return StrategyEvaluation(
                strategy=strategy,
                score=0.0,
                reasons=constraint_reasons,
                risks=['Constraints not met'],
                benefits=[]
            )

        # Оценка соответствия задаче
        task_fit_score = self._evaluate_task_fit(strategy, task_analysis)
        score += task_fit_score * 0.4
        reasons.append(f"Task fit: {task_fit_score:.2f}")

        # Оценка производительности
        performance_score = await self._evaluate_performance(strategy)
        score += performance_score * 0.3
        reasons.append(f"Performance: {performance_score:.2f}")

        # Оценка ресурсов
        resource_score = self._evaluate_resources(strategy, context)
        score += resource_score * 0.2
        reasons.append(f"Resources: {resource_score:.2f}")

        # Оценка рисков
        risk_score = self._evaluate_risks(strategy, task_analysis, context)
        score += risk_score * 0.1
        if risk_score < 0.5:
            risks.append("High risk strategy")

        # Определение преимуществ
        if strategy.success_probability > 0.9:
            benefits.append("High success probability")
        if strategy.estimated_time < 2.0:
            benefits.append("Fast execution")
        if len(strategy.tool_chain) > 2:
            benefits.append("Comprehensive approach")

        return StrategyEvaluation(
            strategy=strategy,
            score=min(1.0, score),
            reasons=reasons,
            risks=risks,
            benefits=benefits
        )

    def _check_constraints(self, strategy: Strategy, context: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Проверка ограничений стратегии"""
        available_tools = set(context.get('available_tools', []))
        system_resources = context.get('system_resources', {})

        reasons = []

        for constraint in strategy.constraints:
            if constraint == 'basic_tools_only':
                if not available_tools.intersection({'hybrid_models'}):
                    reasons.append("Basic tools not available")
                    return False, reasons

            elif constraint == 'rag_available':
                if 'rag_tool' not in available_tools:
                    reasons.append("RAG tool not available")
                    return False, reasons

            elif constraint == 'web_research_available':
                if 'web_research' not in available_tools:
                    reasons.append("Web research not available")
                    return False, reasons

            elif constraint == 'learning_engine_available':
                if 'learning_engine' not in available_tools:
                    reasons.append("Learning engine not available")
                    return False, reasons

            elif constraint == 'cache_available':
                if 'cache_tool' not in available_tools:
                    reasons.append("Cache tool not available")
                    return False, reasons

            elif constraint == 'all_tools_available':
                required_tools = {'rag_tool', 'web_research', 'learning_engine', 'hybrid_models'}
                if not required_tools.issubset(available_tools):
                    missing = required_tools - available_tools
                    reasons.append(f"Missing tools: {missing}")
                    return False, reasons

            elif constraint == 'time_tolerant':
                if context.get('time_sensitivity') == 'high':
                    reasons.append("Time critical task")
                    return False, reasons

            elif constraint == 'resource_rich':
                cpu_available = system_resources.get('cpu', 1.0)
                memory_available = system_resources.get('memory', 1.0)
                if cpu_available < 0.8 or memory_available < 0.8:
                    reasons.append("Insufficient resources")
                    return False, reasons

        return True, []

    def _evaluate_task_fit(self, strategy: Strategy, task_analysis: Dict[str, Any]) -> float:
        """Оценка соответствия стратегии задаче"""
        score = 0.0

        # Сложность задачи
        task_complexity = task_analysis.get('complexity', 'medium')
        strategy_complexity = self._get_strategy_complexity(strategy)

        if task_complexity == strategy_complexity:
            score += 0.4
        elif abs(self._complexity_level(task_complexity) - self._complexity_level(strategy_complexity)) == 1:
            score += 0.2

        # Требуемые инструменты
        required_tools = set(task_analysis.get('required_tools', []))
        strategy_tools = set(strategy.tool_chain)

        if required_tools.issubset(strategy_tools):
            score += 0.3
        else:
            overlap = len(required_tools.intersection(strategy_tools))
            if overlap > 0:
                score += 0.3 * (overlap / len(required_tools))

        # Адаптационные правила
        adaptation_score = self._evaluate_adaptation_rules(strategy, task_analysis)
        score += adaptation_score * 0.3

        return min(1.0, score)

    async def _evaluate_performance(self, strategy: Strategy) -> float:
        """Оценка производительности стратегии"""
        # Историческая производительность
        historical_perf = self.strategy_stats.get(strategy.name, {}).get('avg_success_rate', 0.5)

        # Базовая вероятность успеха
        base_perf = strategy.success_probability

        # Комбинированная оценка
        return (historical_perf * 0.7) + (base_perf * 0.3)

    def _evaluate_resources(self, strategy: Strategy, context: Dict[str, Any]) -> float:
        """Оценка доступности ресурсов"""
        system_resources = context.get('system_resources', {'cpu': 1.0, 'memory': 1.0})

        required_cpu = strategy.resource_requirements.get('cpu', 0)
        required_memory = strategy.resource_requirements.get('memory', 0)

        available_cpu = system_resources.get('cpu', 1.0)
        available_memory = system_resources.get('memory', 1.0)

        cpu_score = min(1.0, available_cpu / max(0.1, required_cpu)) if required_cpu > 0 else 1.0
        memory_score = min(1.0, available_memory / max(0.1, required_memory)) if required_memory > 0 else 1.0

        return (cpu_score + memory_score) / 2

    def _evaluate_risks(self, strategy: Strategy, task_analysis: Dict[str, Any],
                       context: Dict[str, Any]) -> float:
        """Оценка рисков стратегии"""
        risk_score = 1.0  # Начинаем с низкого риска

        # Риск недоступности инструментов
        tool_risk = len(strategy.tool_chain) / 10  # Чем больше инструментов, тем выше риск
        risk_score -= tool_risk

        # Риск по времени
        time_sensitivity = task_analysis.get('time_sensitivity', 'normal')
        if time_sensitivity == 'high' and strategy.estimated_time > 2.0:
            risk_score -= 0.3

        # Риск по ресурсам
        resource_usage = sum(strategy.resource_requirements.values())
        if resource_usage > 1.0:
            risk_score -= 0.2

        return max(0.0, risk_score)

    def _evaluate_adaptation_rules(self, strategy: Strategy, task_analysis: Dict[str, Any]) -> float:
        """Оценка соответствия правилам адаптации"""
        score = 0.0
        rules = strategy.adaptation_rules

        # Правило сложности
        if 'min_complexity' in rules:
            task_complexity_level = self._complexity_level(task_analysis.get('complexity', 'medium'))
            min_level = self._complexity_level(rules['min_complexity'])
            if task_complexity_level >= min_level:
                score += 0.3

        if 'max_complexity' in rules:
            task_complexity_level = self._complexity_level(task_analysis.get('complexity', 'medium'))
            max_level = self._complexity_level(rules['max_complexity'])
            if task_complexity_level <= max_level:
                score += 0.3

        # Специфические правила
        if rules.get('requires_search') and task_analysis.get('needs_search'):
            score += 0.2

        if rules.get('requires_research') and task_analysis.get('needs_research'):
            score += 0.2

        if rules.get('time_critical') and task_analysis.get('time_sensitivity') == 'high':
            score += 0.2

        return min(1.0, score)

    def _get_strategy_complexity(self, strategy: Strategy) -> str:
        """Определение сложности стратегии"""
        if strategy.priority <= 1:
            return 'low'
        elif strategy.priority <= 3:
            return 'medium'
        else:
            return 'high'

    def _complexity_level(self, complexity: str) -> int:
        """Преобразование сложности в числовой уровень"""
        levels = {'low': 1, 'medium': 2, 'high': 3, 'expert': 4}
        return levels.get(complexity, 2)

    def update_strategy_stats(self, strategy_name: str, success: bool, execution_time: float):
        """Обновление статистики стратегии"""
        if strategy_name not in self.strategy_stats:
            self.strategy_stats[strategy_name] = {
                'total_uses': 0,
                'successes': 0,
                'total_time': 0.0,
                'avg_success_rate': 0.0,
                'avg_execution_time': 0.0
            }

        stats = self.strategy_stats[strategy_name]
        stats['total_uses'] += 1
        stats['total_time'] += execution_time

        if success:
            stats['successes'] += 1

        # Обновление средних значений
        stats['avg_success_rate'] = stats['successes'] / stats['total_uses']
        stats['avg_execution_time'] = stats['total_time'] / stats['total_uses']

    async def get_strategy_recommendations(self, task_analysis: Dict[str, Any],
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Получение рекомендаций по выбору стратегии"""
        recommendations = []

        for strategy in self.strategies.values():
            evaluation = await self._evaluate_strategy(strategy, task_analysis, context)
            if evaluation.score > 0.5:  # Только хорошие варианты
                recommendations.append({
                    'strategy': strategy.name,
                    'score': evaluation.score,
                    'reasons': evaluation.reasons,
                    'benefits': evaluation.benefits,
                    'risks': evaluation.risks
                })

        # Сортировка по оценке
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations[:5]  # Топ 5 рекомендаций

    def get_selection_stats(self) -> Dict[str, Any]:
        """Получение статистики выбора стратегий"""
        total_selections = len(self.selection_history)

        if total_selections == 0:
            return {'total_selections': 0}

        strategy_counts = {}
        avg_scores = {}

        for selection in self.selection_history:
            strategy = selection['selected_strategy']
            score = selection['score']

            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            if strategy not in avg_scores:
                avg_scores[strategy] = []
            avg_scores[strategy].append(score)

        # Расчет средних оценок
        for strategy in avg_scores:
            avg_scores[strategy] = sum(avg_scores[strategy]) / len(avg_scores[strategy])

        return {
            'total_selections': total_selections,
            'strategy_counts': strategy_counts,
            'avg_scores': avg_scores,
            'most_used_strategy': max(strategy_counts, key=strategy_counts.get) if strategy_counts else None
        }

    def add_custom_strategy(self, strategy: Strategy):
        """Добавление пользовательской стратегии"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added custom strategy: {strategy.name}")

    def remove_strategy(self, strategy_name: str):
        """Удаление стратегии"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logger.info(f"Removed strategy: {strategy_name}")
        else:
            logger.warning(f"Strategy not found: {strategy_name}")

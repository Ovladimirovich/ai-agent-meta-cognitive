"""
Advanced Decision Engine - Расширенный движок принятия решений
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class Decision(BaseModel):
    """Структура решения агента"""
    strategy: Dict[str, Any]
    confidence: float
    reasoning: str
    alternatives: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class Strategy(BaseModel):
    """Структура стратегии"""
    name: str
    tool_chain: List[str]
    priority: int
    estimated_time: float
    resource_requirements: Dict[str, Any]
    success_probability: float


class MetaDecisionEngine:
    """Мета-движок принятия решений для мета-познания"""

    def __init__(self):
        self.decision_history = []
        self.strategy_performance = {}

    async def make_meta_decision(self, request, meta_state, cognitive_load):
        """Принятие мета-решения"""
        return {
            'strategy': 'standard_processing',
            'use_multimodal': False,
            'use_distributed': False,
            'confidence_threshold': 0.7,
            'processing_priority': 'normal'
        }


class DecisionEngine:
    """
    Основной движок принятия решений для агента
    """

    def __init__(self, learning_engine=None, performance_monitor=None, memory_manager=None):
        self.learning_engine = learning_engine
        self.performance_monitor = performance_monitor
        self.memory_manager = memory_manager

        self.decision_history = []
        self.strategy_performance = {}

    async def make_decision(self, task: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        """Принятие решения на основе анализа задачи и контекста"""
        try:
            # Простая логика для базового движка
            strategy = Strategy(
                name='basic_processing',
                tool_chain=['hybrid_models'],
                priority=1,
                estimated_time=1.0,
                resource_requirements={'cpu': 0.2, 'memory': 0.1},
                success_probability=0.8
            )

            decision = Decision(
                strategy=strategy.dict(),
                confidence=0.8,
                reasoning="Basic decision for standard processing",
                alternatives=[],
                metadata={'decision_type': 'basic'}
            )

            return decision

        except Exception as e:
            logger.error(f"Ошибка принятия решения: {e}")
            # Возврат fallback решения
            return Decision(
                strategy={'name': 'fallback', 'tool_chain': ['hybrid_models']},
                confidence=0.5,
                reasoning=f"Fallback decision due to error: {str(e)}",
                alternatives=[],
                metadata={'error': str(e)}
            )

    def get_decision_stats(self) -> Dict[str, Any]:
        """Получение статистики принятия решений"""
        return {
            'total_decisions': len(self.decision_history),
            'strategy_performance': self.strategy_performance.copy()
        }


class AdvancedDecisionEngine:
    """Расширенный движок принятия решений"""

    def __init__(self, learning_engine=None, performance_monitor=None, memory_manager=None):
        self.learning_engine = learning_engine
        self.performance_monitor = performance_monitor
        self.memory_manager = memory_manager

        self.decision_history = []
        self.strategy_performance = {}

        # Базовые стратегии
        self.base_strategies = {
            'simple_chat': Strategy(
                name='simple_chat',
                tool_chain=['hybrid_models'],
                priority=1,
                estimated_time=1.0,
                resource_requirements={'cpu': 0.2, 'memory': 0.1},
                success_probability=0.8
            ),
            'rag_enhanced': Strategy(
                name='rag_enhanced',
                tool_chain=['rag_tool', 'hybrid_models'],
                priority=2,
                estimated_time=2.0,
                resource_requirements={'cpu': 0.4, 'memory': 0.3},
                success_probability=0.9
            ),
            'multi_tool': Strategy(
                name='multi_tool',
                tool_chain=['rag_tool', 'web_research', 'hybrid_models'],
                priority=3,
                estimated_time=3.0,
                resource_requirements={'cpu': 0.6, 'memory': 0.5},
                success_probability=0.85
            ),
            'research_mode': Strategy(
                name='research_mode',
                tool_chain=['web_research', 'rag_tool', 'hybrid_models'],
                priority=4,
                estimated_time=4.0,
                resource_requirements={'cpu': 0.8, 'memory': 0.7},
                success_probability=0.75
            ),
            'learning_mode': Strategy(
                name='learning_mode',
                tool_chain=['rag_tool', 'learning_engine', 'hybrid_models'],
                priority=5,
                estimated_time=2.5,
                resource_requirements={'cpu': 0.5, 'memory': 0.4},
                success_probability=0.7
            )
        }

    async def make_decision(self, task: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        """Принятие решения на основе анализа задачи и контекста"""
        try:
            start_time = time.time()

            # Анализ задачи
            task_analysis = await self._analyze_task(task, context)

            # Оценка доступных инструментов
            tool_evaluation = await self._evaluate_tools(task, context)

            # Выбор стратегии
            strategy = await self._select_strategy(task_analysis, tool_evaluation)

            # Оптимизация на основе предыдущего опыта
            optimized_strategy = await self._optimize_strategy(strategy, context)

            # Расчет уверенности
            confidence = self._calculate_confidence(task_analysis, tool_evaluation, optimized_strategy)

            # Генерация объяснения
            reasoning = self._generate_reasoning(task_analysis, tool_evaluation, optimized_strategy)

            # Альтернативные стратегии
            alternatives = self._generate_alternatives(optimized_strategy, task_analysis)

            decision = Decision(
                strategy=optimized_strategy.dict(),
                confidence=confidence,
                reasoning=reasoning,
                alternatives=[alt.dict() for alt in alternatives],
                metadata={
                    'task_analysis': task_analysis,
                    'tool_evaluation': tool_evaluation,
                    'decision_time': time.time() - start_time,
                    'context_used': list(context.keys()) if context else []
                }
            )

            # Сохранение решения для обучения
            self.decision_history.append({
                'task': task,
                'decision': decision,
                'timestamp': time.time(),
                'context': context
            })

            # Ограничение истории
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]

            return decision

        except Exception as e:
            logger.error(f"Ошибка принятия решения: {e}")
            # Возврат fallback решения
            return Decision(
                strategy=self.base_strategies['simple_chat'].dict(),
                confidence=0.5,
                reasoning=f"Fallback decision due to error: {str(e)}",
                alternatives=[],
                metadata={'error': str(e)}
            )

    async def _analyze_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ задачи"""
        query = task.get('query', '')
        task_type = task.get('type', 'general')

        analysis = {
            'complexity': self._assess_complexity(query),
            'domain': self._identify_domain(query),
            'required_tools': self._identify_required_tools(query),
            'time_sensitivity': self._assess_time_sensitivity(query),
            'creativity_required': self._assess_creativity(query),
            'knowledge_depth': self._assess_knowledge_depth(query),
            'user_expertise': context.get('user_expertise', 'unknown'),
            'context_relevance': self._assess_context_relevance(context),
            'historical_performance': await self._get_historical_performance(query, context)
        }

        return analysis

    async def _evaluate_tools(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Оценка доступных инструментов"""
        available_tools = context.get('available_tools', ['hybrid_models', 'rag_tool'])

        tool_evaluation = {}

        for tool_name in available_tools:
            evaluation = {
                'available': True,
                'performance_score': await self._get_tool_performance(tool_name),
                'resource_usage': self._get_tool_resource_usage(tool_name),
                'compatibility_score': self._assess_tool_compatibility(tool_name, task),
                'last_used': await self._get_tool_last_used(tool_name),
                'success_rate': await self._get_tool_success_rate(tool_name)
            }

            # Общий скор инструмента
            evaluation['overall_score'] = (
                evaluation['performance_score'] * 0.4 +
                evaluation['compatibility_score'] * 0.3 +
                evaluation['success_rate'] * 0.3
            )

            tool_evaluation[tool_name] = evaluation

        return tool_evaluation

    async def _select_strategy(self, task_analysis: Dict[str, Any],
                              tool_evaluation: Dict[str, Any]) -> Strategy:
        """Выбор оптимальной стратегии"""
        scores = {}

        for strategy_name, strategy in self.base_strategies.items():
            score = await self._evaluate_strategy_fit(strategy, task_analysis, tool_evaluation)
            scores[strategy_name] = score

        # Выбор стратегии с максимальным скором
        best_strategy_name = max(scores, key=scores.get)

        # Проверка доступности инструментов стратегии
        best_strategy = self.base_strategies[best_strategy_name]
        available_tools = set(tool_evaluation.keys())

        if not all(tool in available_tools for tool in best_strategy.tool_chain):
            # Fallback на простую стратегию
            logger.warning(f"Strategy {best_strategy_name} requires unavailable tools, using fallback")
            best_strategy = self.base_strategies['simple_chat']

        return best_strategy

    async def _evaluate_strategy_fit(self, strategy: Strategy, task_analysis: Dict[str, Any],
                                    tool_evaluation: Dict[str, Any]) -> float:
        """Оценка соответствия стратегии задаче"""
        score = 0.0

        # Оценка на основе сложности задачи
        complexity_multiplier = self._get_complexity_multiplier(task_analysis['complexity'])
        score += strategy.priority * complexity_multiplier

        # Оценка доступности инструментов
        tool_availability = sum(
            tool_evaluation.get(tool, {}).get('overall_score', 0)
            for tool in strategy.tool_chain
        ) / len(strategy.tool_chain) if strategy.tool_chain else 0
        score += tool_availability * 0.3

        # Оценка исторической производительности
        historical_score = self.strategy_performance.get(strategy.name, 0.5)
        score += historical_score * 0.2

        # Штраф за время выполнения
        if task_analysis.get('time_sensitivity', 'normal') == 'high':
            time_penalty = max(0, strategy.estimated_time - 2.0) * 0.1
            score -= time_penalty

        return max(0.0, score)

    async def _optimize_strategy(self, strategy: Strategy, context: Dict[str, Any]) -> Strategy:
        """Оптимизация стратегии на основе опыта"""
        optimized = strategy.copy()

        # Оптимизация цепочки инструментов
        if self.learning_engine:
            try:
                optimization_suggestions = await self.learning_engine.get_optimization_suggestions(
                    strategy.dict(), context
                )

                if optimization_suggestions:
                    # Применение оптимизаций
                    optimized.tool_chain = optimization_suggestions.get(
                        'optimized_chain', strategy.tool_chain
                    )
                    optimized.estimated_time = optimization_suggestions.get(
                        'estimated_time', strategy.estimated_time
                    )

            except Exception as e:
                logger.warning(f"Ошибка оптимизации стратегии: {e}")

        return optimized

    def _calculate_confidence(self, task_analysis: Dict[str, Any],
                            tool_evaluation: Dict[str, Any],
                            strategy: Strategy) -> float:
        """Расчет уверенности в решении"""
        confidence = 0.5  # Базовая уверенность

        # Уверенность на основе анализа задачи
        task_confidence = task_analysis.get('historical_performance', 0.5)
        confidence += task_confidence * 0.3

        # Уверенность на основе инструментов
        tool_scores = [eval.get('overall_score', 0.5) for eval in tool_evaluation.values()]
        avg_tool_score = sum(tool_scores) / len(tool_scores) if tool_scores else 0.5
        confidence += avg_tool_score * 0.4

        # Уверенность на основе стратегии
        strategy_confidence = strategy.success_probability
        confidence += strategy_confidence * 0.3

        return min(1.0, max(0.0, confidence))

    def _generate_reasoning(self, task_analysis: Dict[str, Any],
                           tool_evaluation: Dict[str, Any],
                           strategy: Strategy) -> str:
        """Генерация объяснения решения"""
        reasoning_parts = []

        # Объяснение выбора стратегии
        reasoning_parts.append(f"Выбрана стратегия '{strategy.name}' на основе:")

        # Анализ сложности
        complexity = task_analysis.get('complexity', 'medium')
        reasoning_parts.append(f"- Сложность задачи: {complexity}")

        # Доступные инструменты
        available_tools = list(tool_evaluation.keys())
        reasoning_parts.append(f"- Доступные инструменты: {', '.join(available_tools)}")

        # Оценка уверенности
        confidence_level = "высокая" if strategy.success_probability > 0.8 else "средняя" if strategy.success_probability > 0.6 else "низкая"
        reasoning_parts.append(f"- Ожидаемая эффективность: {confidence_level} ({strategy.success_probability:.2f})")

        return " ".join(reasoning_parts)

    def _generate_alternatives(self, selected_strategy: Strategy,
                              task_analysis: Dict[str, Any]) -> List[Strategy]:
        """Генерация альтернативных стратегий"""
        alternatives = []

        for strategy_name, strategy in self.base_strategies.items():
            if strategy.name != selected_strategy.name:
                # Проверка, может ли стратегия быть альтернативой
                if self._is_viable_alternative(strategy, task_analysis):
                    alternatives.append(strategy)

        # Ограничение до 3 альтернатив
        return alternatives[:3]

    def _is_viable_alternative(self, strategy: Strategy, task_analysis: Dict[str, Any]) -> bool:
        """Проверка, может ли стратегия быть жизнеспособной альтернативой"""
        complexity = task_analysis.get('complexity', 'medium')

        # Простые стратегии подходят для простых задач
        if complexity == 'low' and strategy.priority <= 2:
            return True

        # Сложные стратегии для сложных задач
        if complexity == 'high' and strategy.priority >= 3:
            return True

        # Средние стратегии для средних задач
        if complexity == 'medium' and 2 <= strategy.priority <= 4:
            return True

        return False

    def _assess_complexity(self, query: str) -> str:
        """Оценка сложности запроса"""
        query_length = len(query.split())

        if query_length < 5:
            return 'low'
        elif query_length < 15:
            return 'medium'
        else:
            return 'high'

    def _identify_domain(self, query: str) -> str:
        """Идентификация домена запроса"""
        # Простая логика определения домена
        query_lower = query.lower()

        if any(word in query_lower for word in ['программирование', 'python', 'javascript', 'code']):
            return 'programming'
        elif any(word in query_lower for word in ['математика', 'физика', 'наука']):
            return 'science'
        elif any(word in query_lower for word in ['бизнес', 'финансы', 'экономика']):
            return 'business'
        else:
            return 'general'

    def _identify_required_tools(self, query: str) -> List[str]:
        """Идентификация требуемых инструментов"""
        tools = []

        query_lower = query.lower()

        if any(word in query_lower for word in ['поиск', 'найти', 'искать']):
            tools.append('rag_tool')

        if any(word in query_lower for word in ['сгенерировать', 'написать', 'создать']):
            tools.append('hybrid_models')

        if any(word in query_lower for word in ['исследовать', 'анализировать']):
            tools.append('web_research')

        return tools or ['hybrid_models']

    def _assess_time_sensitivity(self, query: str) -> str:
        """Оценка чувствительности ко времени"""
        urgent_words = ['срочно', 'быстро', 'немедленно', 'urgent', 'asap']
        query_lower = query.lower()

        if any(word in query_lower for word in urgent_words):
            return 'high'
        else:
            return 'normal'

    def _assess_creativity(self, query: str) -> float:
        """Оценка требуемой креативности"""
        creative_words = ['креативно', 'оригинально', 'творчески', 'creative', 'innovative']
        query_lower = query.lower()

        creative_score = sum(1 for word in creative_words if word in query_lower)
        return min(1.0, creative_score / 3)

    def _assess_knowledge_depth(self, query: str) -> str:
        """Оценка требуемой глубины знаний"""
        deep_words = ['подробно', 'детально', 'глубоко', 'explain', 'detailed']
        query_lower = query.lower()

        if any(word in query_lower for word in deep_words):
            return 'deep'
        else:
            return 'standard'

    def _assess_context_relevance(self, context: Dict[str, Any]) -> float:
        """Оценка релевантности контекста"""
        if not context:
            return 0.0

        relevance_score = 0.0

        if context.get('conversation_history'):
            relevance_score += 0.3

        if context.get('user_preferences'):
            relevance_score += 0.2

        if context.get('domain_knowledge'):
            relevance_score += 0.3

        if context.get('temporal_context'):
            relevance_score += 0.2

        return relevance_score

    async def _get_historical_performance(self, query: str, context: Dict[str, Any]) -> float:
        """Получение исторической производительности для похожих запросов"""
        # Заглушка - в реальности поиск в истории
        return 0.7

    async def _get_tool_performance(self, tool_name: str) -> float:
        """Получение оценки производительности инструмента"""
        # Заглушка
        tool_scores = {
            'rag_tool': 0.85,
            'hybrid_models': 0.9,
            'web_research': 0.75,
            'cache_tool': 0.95
        }
        return tool_scores.get(tool_name, 0.7)

    def _get_tool_resource_usage(self, tool_name: str) -> Dict[str, float]:
        """Получение использования ресурсов инструментом"""
        # Заглушка
        resource_usage = {
            'rag_tool': {'cpu': 0.3, 'memory': 0.4},
            'hybrid_models': {'cpu': 0.5, 'memory': 0.3},
            'web_research': {'cpu': 0.4, 'memory': 0.6},
            'cache_tool': {'cpu': 0.1, 'memory': 0.2}
        }
        return resource_usage.get(tool_name, {'cpu': 0.2, 'memory': 0.2})

    def _assess_tool_compatibility(self, tool_name: str, task: Dict[str, Any]) -> float:
        """Оценка совместимости инструмента с задачей"""
        # Заглушка - простая логика
        return 0.8

    async def _get_tool_last_used(self, tool_name: str) -> Optional[float]:
        """Получение времени последнего использования инструмента"""
        # Заглушка
        return time.time() - 3600  # 1 час назад

    async def _get_tool_success_rate(self, tool_name: str) -> float:
        """Получение рейтинга успешности инструмента"""
        # Заглушка
        success_rates = {
            'rag_tool': 0.88,
            'hybrid_models': 0.92,
            'web_research': 0.78,
            'cache_tool': 0.96
        }
        return success_rates.get(tool_name, 0.8)

    def _get_complexity_multiplier(self, complexity: str) -> float:
        """Получение множителя сложности"""
        multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        }
        return multipliers.get(complexity, 1.0)

    def update_strategy_performance(self, strategy_name: str, success: bool):
        """Обновление статистики производительности стратегии"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = 0.5

        # Простое экспоненциальное сглаживание
        current = self.strategy_performance[strategy_name]
        alpha = 0.1  # Скорость обучения

        if success:
            self.strategy_performance[strategy_name] = current + alpha * (1.0 - current)
        else:
            self.strategy_performance[strategy_name] = current + alpha * (0.0 - current)

    async def select_tools(self, task_analysis: Dict[str, Any]) -> List[str]:
        """Выбор цепочки инструментов на основе анализа задачи"""
        try:
            # Простая логика выбора инструментов
            complexity = task_analysis.get('complexity', 'medium')
            needs_search = task_analysis.get('needs_search', False)
            needs_generation = task_analysis.get('needs_generation', True)

            tool_chain = []

            # Всегда добавляем аутентификацию
            tool_chain.append('auth')

            # Добавляем кэш для оптимизации
            tool_chain.append('cache')

            # Добавляем RAG если нужен поиск
            if needs_search or complexity in ['high', 'medium']:
                tool_chain.append('rag')

            # Добавляем гибридные модели для генерации
            if needs_generation:
                tool_chain.append('hybrid_models')

            return tool_chain

        except Exception as e:
            logger.error(f"Ошибка выбора инструментов: {e}")
            return ['auth', 'hybrid_models']  # Fallback

    def get_decision_stats(self) -> Dict[str, Any]:
        """Получение статистики принятия решений"""
        return {
            'total_decisions': len(self.decision_history),
            'avg_confidence': sum(d['decision'].confidence for d in self.decision_history) / len(self.decision_history) if self.decision_history else 0,
            'strategy_performance': self.strategy_performance.copy()
        }

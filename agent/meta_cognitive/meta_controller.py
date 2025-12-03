"""
Главный контроллер мета-познания
Фаза 4: Продвинутые функции
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..core.models import AgentRequest, AgentResponse
from ..learning.models import AgentExperience, LearningResult
from .system_coordinator import SystemCoordinator
from .cognitive_health_monitor import CognitiveHealthMonitor
from .decision_engine import MetaDecisionEngine
from .performance_optimizer import PerformanceOptimizer
from .distributed_learning import DistributedLearningManager
from .multimodality_learning import MultimodalLearningEngine
from .ethical_learning import EthicalLearningEngine

logger = logging.getLogger(__name__)


class MetaCognitiveState:
    """
    Состояние мета-познания агента
    """

    def __init__(self):
        self.cognitive_load = 0.0
        self.confidence_level = 0.5
        self.learning_sessions_count = 0
        self.adaptations_applied = 0
        self.insights_generated = 0
        self.subsystem_states: Dict[str, Dict[str, Any]] = {}
        self.active_optimizations: List[Dict[str, Any]] = []
        self.current_reasoning_trace = None
        self.average_meta_processing_time = 0.0
        self.last_health_assessment = None

    def snapshot(self) -> Dict[str, Any]:
        """Создание снимка состояния"""
        return {
            'cognitive_load': self.cognitive_load,
            'confidence_level': self.confidence_level,
            'learning_sessions_count': self.learning_sessions_count,
            'adaptations_applied': self.adaptations_applied,
            'insights_generated': self.insights_generated,
            'subsystem_states': self.subsystem_states.copy(),
            'active_optimizations': self.active_optimizations.copy(),
            'average_meta_processing_time': self.average_meta_processing_time,
            'timestamp': datetime.now()
        }


class MetaCognitiveResponse:
    """
    Ответ с мета-познанием
    """

    def __init__(self, **kwargs):
        self.agent_response = kwargs.get('agent_response')
        self.meta_decision = kwargs.get('meta_decision')
        self.coordination_result = kwargs.get('coordination_result')
        self.reflection_result = kwargs.get('reflection_result')
        self.learning_result = kwargs.get('learning_result')
        self.optimization_result = kwargs.get('optimization_result')
        self.cognitive_load = kwargs.get('cognitive_load', 0.0)
        self.processing_time = kwargs.get('processing_time', 0.0)
        self.meta_state_snapshot = kwargs.get('meta_state_snapshot')


class CognitiveLoadMonitor:
    """
    Монитор когнитивной нагрузки
    """

    def __init__(self):
        self.load_history: List[float] = []
        self.max_history_size = 100

    async def assess_load(self) -> float:
        """
        Оценка текущей когнитивной нагрузки

        Returns:
            float: Уровень нагрузки (0.0-1.0)
        """
        # Факторы нагрузки:
        # - Количество активных задач
        # - Сложность текущих операций
        # - Доступные ресурсы
        # - История нагрузки

        base_load = 0.3  # Базовая нагрузка

        # Фактор количества активных задач (заглушка)
        active_tasks_factor = min(len(asyncio.all_tasks()) / 50, 0.3)

        # Фактор сложности (заглушка)
        complexity_factor = 0.2

        # Фактор ресурсов (заглушка)
        resource_factor = 0.1

        # Фактор истории
        history_factor = sum(self.load_history[-10:]) / max(len(self.load_history[-10:]), 1) * 0.1

        load = base_load + active_tasks_factor + complexity_factor + resource_factor + history_factor
        load = min(max(load, 0.0), 1.0)

        self.load_history.append(load)
        if len(self.load_history) > self.max_history_size:
            self.load_history = self.load_history[-self.max_history_size:]

        return load


class MetaCognitiveController:
    """
    Главный контроллер мета-познания
    Координирует все мета-познавательные процессы
    """

    def __init__(self, agent_core):
        """
        Инициализация контроллера мета-познания

        Args:
            agent_core: Ядро агента
        """
        self.agent_core = agent_core

        # Базовые компоненты мета-познания
        self.self_reflection = agent_core.self_reflection_engine if hasattr(agent_core, 'self_reflection_engine') else None
        self.learning_engine = agent_core.learning_engine if hasattr(agent_core, 'learning_engine') else None
        self.system_coordinator = SystemCoordinator(agent_core)
        self.cognitive_load = CognitiveLoadMonitor()
        self.decision_engine = MetaDecisionEngine()
        self.performance_optimizer = PerformanceOptimizer()
        self.health_monitor = CognitiveHealthMonitor(self)

        # Продвинутые функции Фазы 4
        self.distributed_learning = DistributedLearningManager(
            agent_id="meta_controller_agent",
            learning_engine=self.learning_engine
        ) if self.learning_engine else None

        self.multimodal_learning = MultimodalLearningEngine(
            self.learning_engine
        ) if self.learning_engine else None

        self.ethical_learning = EthicalLearningEngine(
            self.learning_engine
        ) if self.learning_engine else None

        # Состояние мета-познания
        self.meta_state = MetaCognitiveState()

        # История обработки
        self.processing_history: List[MetaCognitiveResponse] = []
        self.max_history_size = 1000

        logger.info("MetaCognitiveController initialized")

    async def process_with_meta_cognition(self, request: AgentRequest) -> MetaCognitiveResponse:
        """
        Обработка запроса с полным мета-познанием

        Args:
            request: Запрос агента

        Returns:
            MetaCognitiveResponse: Ответ с мета-познанием
        """
        meta_start = time.time()

        try:
            logger.info(f"Starting meta-cognitive processing for request: {request.id}")

            # Шаг 1: Оценка когнитивной нагрузки
            cognitive_load = await self.cognitive_load.assess_load()

            # Шаг 2: Принятие мета-решения о стратегии обработки
            meta_decision = await self.decision_engine.make_meta_decision(
                request, self.meta_state, cognitive_load
            )

            # Шаг 3: Координация подсистем
            coordination_result = await self.system_coordinator.coordinate_processing(
                request, meta_decision
            )

            # Шаг 4: Основная обработка агентом
            agent_response = await self._process_agent_request(request, meta_decision)

            # Шаг 5: Пост-обработка с мета-познанием
            post_processing = await self._perform_post_processing(
                request, agent_response, coordination_result
            )

            # Шаг 6: Само-рефлексия
            reflection_result = await self._perform_self_reflection(
                request, agent_response, post_processing
            )

            # Шаг 7: Обучение на взаимодействии
            learning_result = await self._perform_learning(
                request, agent_response, reflection_result
            )

            # Шаг 8: Оптимизация производительности
            optimization_result = await self.performance_optimizer.optimize_based_on_feedback(
                agent_response, reflection_result, learning_result
            )

            # Шаг 9: Обновление мета-состояния
            await self._update_meta_state(
                reflection_result, learning_result, optimization_result
            )

            # Шаг 10: Продвинутые функции (Фаза 4)
            await self._apply_advanced_features(request, agent_response)

            processing_time = time.time() - meta_start

            response = MetaCognitiveResponse(
                agent_response=agent_response,
                meta_decision=meta_decision,
                coordination_result=coordination_result,
                reflection_result=reflection_result,
                learning_result=learning_result,
                optimization_result=optimization_result,
                cognitive_load=cognitive_load,
                processing_time=processing_time,
                meta_state_snapshot=self.meta_state.snapshot()
            )

            # Сохранение в историю
            self.processing_history.append(response)
            if len(self.processing_history) > self.max_history_size:
                self.processing_history = self.processing_history[-self.max_history_size:]

            logger.info(f"Meta-cognitive processing completed in {processing_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Meta-cognitive processing failed: {e}")
            # Возврат минимального ответа в случае ошибки
            return MetaCognitiveResponse(
                agent_response=AgentResponse(
                    id=f"error_{request.id}",
                    result="Извините, произошла ошибка в мета-познавательной обработке.",
                    confidence=0.0,
                    reasoning_trace=[],
                    execution_time=time.time() - meta_start,
                    timestamp=datetime.now(),
                    metadata={}
                ),
                cognitive_load=1.0,
                processing_time=time.time() - meta_start,
                meta_state_snapshot=self.meta_state.snapshot()
            )

    async def _process_agent_request(self, request: AgentRequest, meta_decision: Dict[str, Any]) -> AgentResponse:
        """
        Обработка запроса агентом с учетом мета-решения

        Args:
            request: Запрос
            meta_decision: Мета-решение

        Returns:
            AgentResponse: Ответ агента
        """
        try:
            # Применение стратегии из мета-решения
            if meta_decision.get('use_multimodal', False) and self.multimodal_learning:
                # Мульти-модальная обработка
                return await self._process_multimodal_request(request)
            elif meta_decision.get('use_distributed', False) and self.distributed_learning:
                # Обработка с учетом распределенного обучения
                return await self._process_distributed_request(request)
            else:
                # Стандартная обработка
                return await self.agent_core.process_request(request)

        except Exception as e:
            logger.warning(f"Agent request processing failed, using fallback: {e}")
            return await self.agent_core.process_request(request)

    async def _process_multimodal_request(self, request: AgentRequest) -> AgentResponse:
        """
        Мульти-модальная обработка запроса

        Args:
            request: Запрос

        Returns:
            AgentResponse: Ответ
        """
        if not self.multimodal_learning:
            return await self.agent_core.process_request(request)

        # Создание мульти-модального опыта
        experience = AgentExperience(
            id=f"multimodal_{request.id}",
            query=getattr(request, 'query', ''),
            response="",  # Будет заполнен после обработки
            timestamp=datetime.now(),
            significance_score=0.8,
            metadata=getattr(request, 'metadata', {})
        )

        # Мульти-модальное обучение
        learning_result = await self.multimodal_learning.learn_from_multimodal_experience(experience)

        # Получение ответа от агента
        agent_response = await self.agent_core.process_request(request)

        # Обогащение ответа мульти-модальными insights
        enhanced_content = self._enhance_response_with_multimodal(
            agent_response.result, learning_result
        )

        return AgentResponse(
            id=agent_response.id,
            result=enhanced_content,
            confidence=max(agent_response.confidence, 0.7),  # Повышение уверенности
            reasoning_trace=agent_response.reasoning_trace if hasattr(agent_response, 'reasoning_trace') else [],
            execution_time=agent_response.execution_time if hasattr(agent_response, 'execution_time') else 0.0,
            timestamp=agent_response.timestamp,
            metadata={
                **getattr(agent_response, 'metadata', {}),
                'multimodal_enhanced': True,
                'modalities_used': learning_result.modalities_used if hasattr(learning_result, 'modalities_used') else []
            }
        )

    async def _process_distributed_request(self, request: AgentRequest) -> AgentResponse:
        """
        Обработка запроса с учетом распределенного обучения

        Args:
            request: Запрос

        Returns:
            AgentResponse: Ответ
        """
        if not self.distributed_learning:
            return await self.agent_core.process_request(request)

        # Запрос опыта от сети
        query = {'topic': getattr(request, 'query', '')[:50], 'domain': 'general'}
        network_experiences = await self.distributed_learning.request_experience(query)

        # Основная обработка
        agent_response = await self.agent_core.process_request(request)
        # Возвращаем обогащенный ответ
        if network_experiences:
            enhanced_content = self._enhance_response_with_network_knowledge(
                agent_response.result, network_experiences
            )
            return AgentResponse(
                id=agent_response.id,
                result=enhanced_content,
                confidence=agent_response.confidence,
                reasoning_trace=agent_response.reasoning_trace if hasattr(agent_response, 'reasoning_trace') else [],
                execution_time=agent_response.execution_time if hasattr(agent_response, 'execution_time') else 0.0,
                timestamp=agent_response.timestamp,
                metadata=agent_response.metadata if hasattr(agent_response, 'metadata') else {}
            )
        # Возвращаем обогащенный ответ
        return agent_response

    async def _perform_post_processing(self, request: AgentRequest, response: AgentResponse,
                                     coordination: Dict[str, Any]) -> Dict[str, Any]:
        """
        Пост-обработка с использованием мета-познания

        Args:
            request: Запрос
            response: Ответ
            coordination: Результат координации

        Returns:
            Dict[str, Any]: Результат пост-обработки
        """
        # Анализ качества ответа
        quality_analysis = await self._analyze_response_quality(response)

        # Генерация объяснений
        explanations = await self._generate_explanations(response, coordination)

        # Оценка уверенности с учетом мета-факторов
        enhanced_confidence = await self._enhance_confidence_assessment(
            response.confidence, quality_analysis, coordination
        )

        # Рекомендации по улучшению
        improvement_suggestions = await self._generate_improvement_suggestions(
            request, response, quality_analysis
        )

        return {
            'quality_analysis': quality_analysis,
            'explanations': explanations,
            'enhanced_confidence': enhanced_confidence,
            'improvement_suggestions': improvement_suggestions
        }

    async def _perform_self_reflection(self, request: AgentRequest, response: AgentResponse,
                                     post_processing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение само-рефлексии

        Args:
            request: Запрос
            response: Ответ
            post_processing: Пост-обработка

        Returns:
            Dict[str, Any]: Результат рефлексии
        """
        if not self.self_reflection:
            return {'reflection': 'Self-reflection not available', 'insights': []}

        try:
            # Создание записи взаимодействия
            interaction_record = {
                'request': request,
                'response': response,
                'post_processing': post_processing,
                'timestamp': datetime.now()
            }

            # Выполнение рефлексии
            reflection_result = await self.self_reflection.reflect_on_interaction(interaction_record)

            return reflection_result

        except Exception as e:
            logger.warning(f"Self-reflection failed: {e}")
            return {'reflection': f'Error: {e}', 'insights': []}

    async def _perform_learning(self, request: AgentRequest, response: AgentResponse,
                              reflection: Dict[str, Any]) -> LearningResult:
        """
        Выполнение обучения на взаимодействии

        Args:
            request: Запрос
            response: Ответ
            reflection: Результат рефлексии

        Returns:
            LearningResult: Результат обучения
        """
        if not self.learning_engine:
            return LearningResult(
                experience_processed=None,
                patterns_extracted=0,
                cognitive_updates=0,
                skills_developed=0,
                adaptation_applied=None,
                learning_effectiveness=0.0,
                learning_time=0.0,
                timestamp=datetime.now()
            )

        try:
            # Создание опыта для обучения
            experience = AgentExperience(
                id=f"learning_{request.id}",
                query=getattr(request, 'query', ''),
                response=response.result,
                timestamp=datetime.now(),
                significance_score=reflection.get('enhanced_confidence', response.confidence),
                metadata={
                    'reflection_insights': reflection.get('insights', []),
                    'quality_score': 0.5  # Используем значение по умолчанию, т.к. post_processing не передается в эту функцию
                }
            )

            # Этическая проверка перед обучением
            if self.ethical_learning:
                safety_check = await self.ethical_learning.ethical_checker.check_learning_safety(experience)
                if not safety_check['safe_to_learn']:
                    logger.warning(f"Learning blocked for safety: {safety_check['risk_level']}")
                    return LearningResult(
                        experience_processed=experience,
                        patterns_extracted=0,
                        cognitive_updates=0,
                        skills_developed=0,
                        adaptation_applied=None,
                        learning_effectiveness=0.0,
                        learning_time=0.0,
                        timestamp=datetime.now()
                    )

            # Обучение
            learning_result = await self.learning_engine.learn_from_experience(experience)

            # Обновление счетчика сессий обучения
            self.meta_state.learning_sessions_count += 1

            return learning_result

        except Exception as e:
            logger.error(f"Learning failed: {e}")
            return LearningResult(
                experience_processed=None,
                patterns_extracted=0,
                cognitive_updates=0,
                skills_developed=0,
                adaptation_applied=None,
                learning_effectiveness=0.0,
                learning_time=0.0,
                timestamp=datetime.now()
            )

    async def _apply_advanced_features(self, request: AgentRequest, response: AgentResponse):
        """
        Применение продвинутых функций Фазы 4

        Args:
            request: Запрос
            response: Ответ
        """
        try:
            # Создание опыта для продвинутых функций
            experience = AgentExperience(
                id=f"advanced_{request.id}",
                query=getattr(request, 'query', ''),
                response=response.result,
                timestamp=datetime.now(),
                significance_score=response.confidence,
                metadata=getattr(request, 'metadata', {})
            )

            # Распределенное обучение
            if self.distributed_learning:
                await self.distributed_learning.share_experience(experience)

            # Этическое обучение
            if self.ethical_learning:
                await self.ethical_learning.learn_ethically(experience)

        except Exception as e:
            logger.warning(f"Advanced features application failed: {e}")

    async def _update_meta_state(self, reflection: Dict[str, Any], learning: LearningResult,
                               optimization: Dict[str, Any]):
        """
        Обновление мета-состояния

        Args:
            reflection: Результат рефлексии
            learning: Результат обучения
            optimization: Результат оптимизации
        """
        # Обновление когнитивной нагрузки
        self.meta_state.cognitive_load = await self.cognitive_load.assess_load()

        # Обновление уверенности
        if hasattr(learning, 'learning_effectiveness'):
            self.meta_state.confidence_level = (
                self.meta_state.confidence_level * 0.9 + learning.learning_effectiveness * 0.1
            )

        # Обновление счетчиков
        if hasattr(learning, 'patterns_extracted'):
            self.meta_state.insights_generated += learning.patterns_extracted

        # Обновление активных оптимизаций
        if optimization and 'optimizations' in optimization:
            self.meta_state.active_optimizations = optimization['optimizations']

        # Обновление среднего времени обработки
        if hasattr(self, 'processing_history') and self.processing_history:
            recent_times = [r.processing_time for r in self.processing_history[-10:]]
            self.meta_state.average_meta_processing_time = sum(recent_times) / len(recent_times)

    def _enhance_response_with_multimodal(self, content: str, learning_result: Any) -> str:
        """
        Обогащение ответа мульти-модальными insights

        Args:
            content: Исходный контент
            learning_result: Результат мульти-модального обучения

        Returns:
            str: Обогащенный контент
        """
        if not hasattr(learning_result, 'modalities_used'):
            return content

        modalities = learning_result.modalities_used
        if modalities:
            enhancement = f"\n\n*Обработано модальностей: {', '.join(modalities)}*"
            return content + enhancement

        return content

    def _enhance_response_with_network_knowledge(self, content: str, network_experiences: List[AgentExperience]) -> str:
        """
        Обогащение ответа знаниями из сети

        Args:
            content: Исходный контент
            network_experiences: Опыт из сети

        Returns:
            str: Обогащенный контент
        """
        if network_experiences:
            enhancement = f"\n\n*Использовано знаний из сети агентов: {len(network_experiences)} источников*"
            return content + enhancement

        return content

    async def _analyze_response_quality(self, response: AgentResponse) -> Dict[str, Any]:
        """
        Анализ качества ответа

        Args:
            response: Ответ

        Returns:
            Dict[str, Any]: Анализ качества
        """
        content = response.result
        confidence = response.confidence

        # Простой анализ качества
        quality_score = confidence

        # Факторы качества
        length_appropriate = 50 <= len(content) <= 2000
        has_structure = any(indicator in content for indicator in ['•', '-', '1.', '2.'])
        has_examples = any(word in content.lower() for word in ['например', 'example', 'for instance'])

        quality_score *= (0.7 + 0.1 * length_appropriate + 0.1 * has_structure + 0.1 * has_examples)

        return {
            'score': min(quality_score, 1.0),
            'length_appropriate': length_appropriate,
            'has_structure': has_structure,
            'has_examples': has_examples,
            'confidence': confidence
        }

    async def _generate_explanations(self, response: AgentResponse, coordination: Dict[str, Any]) -> List[str]:
        """
        Генерация объяснений ответа

        Args:
            response: Ответ
            coordination: Результат координации

        Returns:
            List[str]: Объяснения
        """
        explanations = []

        if coordination.get('involved_subsystems'):
            subsystems = coordination['involved_subsystems']
            explanations.append(f"Для обработки запроса были задействованы подсистемы: {', '.join(subsystems)}")

        if response.confidence > 0.8:
            explanations.append("Ответ дан с высокой уверенностью")
        elif response.confidence < 0.5:
            explanations.append("Ответ дан с низкой уверенностью - рекомендуется дополнительная проверка")

        return explanations

    async def _enhance_confidence_assessment(self, base_confidence: float, quality_analysis: Dict[str, Any],
                                          coordination: Dict[str, Any]) -> float:
        """
        Улучшение оценки уверенности

        Args:
            base_confidence: Базовая уверенность
            quality_analysis: Анализ качества
            coordination: Координация

        Returns:
            float: Улучшенная уверенность
        """
        enhanced = base_confidence

        # Бонус за качество
        quality_bonus = quality_analysis.get('score', 0.5) * 0.1
        enhanced += quality_bonus

        # Бонус за координацию
        if coordination.get('success', False):
            enhanced += 0.05

        return min(enhanced, 1.0)

    async def _generate_improvement_suggestions(self, request: AgentRequest, response: AgentResponse,
                                             quality_analysis: Dict[str, Any]) -> List[str]:
        """
        Генерация предложений по улучшению

        Args:
            request: Запрос
            response: Ответ
            quality_analysis: Анализ качества

        Returns:
            List[str]: Предложения
        """
        suggestions = []

        if not quality_analysis.get('length_appropriate', True):
            if len(response.result) < 50:
                suggestions.append("Ответ слишком короткий - добавить больше деталей")
            else:
                suggestions.append("Ответ слишком длинный - сократить до ключевых моментов")

        if not quality_analysis.get('has_structure', False):
            suggestions.append("Добавить структуру в ответ (списки, абзацы)")

        if not quality_analysis.get('has_examples', False):
            suggestions.append("Добавить примеры для лучшего понимания")

        if response.confidence < 0.7:
            suggestions.append("Повысить уверенность в ответах через дополнительное обучение")

        return suggestions

    async def get_meta_cognitive_state(self) -> Dict[str, Any]:
        """
        Получение состояния мета-познания

        Returns:
            Dict[str, Any]: Состояние
        """
        return {
            'meta_state': self.meta_state.snapshot(),
            'health_assessment': await self.health_monitor.assess_cognitive_health(),
            'system_coordination_status': self.system_coordinator.get_status(),
            'performance_metrics': self.performance_optimizer.get_metrics(),
            'advanced_features_status': {
                'distributed_learning': self.distributed_learning.get_distributed_metrics() if self.distributed_learning else None,
                'multimodal_learning': self.multimodal_learning.get_multimodal_metrics() if self.multimodal_learning else None,
                'ethical_learning': self.ethical_learning.get_ethical_learning_metrics() if self.ethical_learning else None
            }
        }

    async def optimize_meta_cognitive_system(self) -> Dict[str, Any]:
        """
        Оптимизация мета-познавательной системы

        Returns:
            Dict[str, Any]: Результат оптимизации
        """
        optimization_results = []

        # Оптимизация производительности
        perf_result = await self.performance_optimizer.perform_full_optimization()
        optimization_results.append({'component': 'performance', 'result': perf_result})

        # Оптимизация здоровья
        health_result = await self.health_monitor.optimize_health()
        optimization_results.append({'component': 'health', 'result': health_result})

        # Оптимизация координации
        coord_result = await self.system_coordinator.optimize_coordination()
        optimization_results.append({'component': 'coordination', 'result': coord_result})

        return {
            'optimizations_applied': optimization_results,
            'overall_improvement': sum(r['result'].get('improvement', 0) for r in optimization_results),
            'timestamp': datetime.now()
        }

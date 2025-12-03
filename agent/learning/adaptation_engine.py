"""
Движок адаптации агента на основе мета-когнитивных данных
Фаза 3: Расширенные мета-когнитивные функции
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import json

from ..core.models import AgentRequest, AgentResponse, QueryAnalysis
from ..self_awareness.confidence_calculator import ConfidenceCalculator
from ..meta_cognitive.cognitive_load_analyzer import CognitiveLoadAnalyzer, CognitiveLoadMetrics

if TYPE_CHECKING:
    from ..core.agent_core import AgentCore

logger = logging.getLogger(__name__)


class AdaptationType(Enum):
    """Типы адаптаций"""
    STRATEGY_CHANGE = "strategy_change"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    TOOL_PREFERENCE = "tool_preference"
    MEMORY_OPTIMIZATION = "memory_optimization"
    RESOURCE_MANAGEMENT = "resource_management"
    LEARNING_RATE_ADJUSTMENT = "learning_rate_adjustment"


class AdaptationStatus(Enum):
    """Статус адаптации"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Adaptation:
    """Описание адаптации"""
    id: str
    type: AdaptationType
    description: str
    trigger_condition: Dict[str, Any]
    action: Dict[str, Any]
    priority: int
    created_at: datetime
    status: AdaptationStatus
    effectiveness_score: Optional[float] = None
    last_applied: Optional[datetime] = None
    rollback_reason: Optional[str] = None


class AdaptationEngine:
    """
    Движок адаптации агента
    
    Основан на мета-когнитивных данных и анализе производительности
    для автоматической адаптации стратегий, параметров и поведения агента
    """

    def __init__(self, agent_core: 'AgentCore'):
        self.agent_core = agent_core
        self.analyzer = CognitiveLoadAnalyzer()
        self.confidence_calculator = ConfidenceCalculator()
        
        # Хранилище адаптаций
        self.adaptations: Dict[str, Adaptation] = {}
        self.active_adaptations: List[str] = []
        self.adaptation_history: List[Adaptation] = []
        
        # Правила адаптации
        self.adaptation_rules = self._initialize_adaptation_rules()
        
        # Пороги для принятия решений
        self.thresholds = {
            'low_confidence': 0.4,
            'high_error_rate': 0.2,
            'high_cognitive_load': 0.7,
            'low_performance': 0.5,
            'resource_pressure': 0.8
        }
        
        logger.info("AdaptationEngine initialized")

    def _initialize_adaptation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Инициализация правил адаптации"""
        return {
            # Правило для снижения когнитивной нагрузки
            'reduce_cognitive_load': {
                'condition': lambda metrics: self._is_high_cognitive_load(metrics),
                'action': self._reduce_cognitive_load_action,
                'priority': 1,
                'description': 'Снижение когнитивной нагрузки'
            },
            # Правило для повышения уверенности
            'increase_confidence': {
                'condition': lambda metrics: self._is_low_confidence(metrics),
                'action': self._increase_confidence_action,
                'priority': 2,
                'description': 'Повышение уровня уверенности'
            },
            # Правило для оптимизации ресурсов
            'optimize_resources': {
                'condition': lambda metrics: self._is_resource_pressure(metrics),
                'action': self._optimize_resources_action,
                'priority': 3,
                'description': 'Оптимизация использования ресурсов'
            },
            # Правило для изменения стратегии
            'change_strategy': {
                'condition': lambda metrics: self._is_performance_degradation(metrics),
                'action': self._change_strategy_action,
                'priority': 4,
                'description': 'Изменение стратегии обработки'
            },
            # Правило для адаптации инструментов
            'adapt_tools': {
                'condition': lambda metrics: self._is_tool_inefficiency(metrics),
                'action': self._adapt_tools_action,
                'priority': 5,
                'description': 'Адаптация используемых инструментов'
            }
        }

    def _is_high_cognitive_load(self, metrics: CognitiveLoadMetrics) -> bool:
        """Проверка высокой когнитивной нагрузки"""
        load_score = self.analyzer.calculate_load_score(metrics)
        return load_score > self.thresholds['high_cognitive_load']

    def _is_low_confidence(self, metrics: CognitiveLoadMetrics) -> bool:
        """Проверка низкой уверенности"""
        return metrics.confidence_level < self.thresholds['low_confidence']

    def _is_resource_pressure(self, metrics: CognitiveLoadMetrics) -> bool:
        """Проверка давления на ресурсы"""
        return (metrics.memory_usage > self.thresholds['resource_pressure'] or
                metrics.cpu_usage > self.thresholds['resource_pressure'] or
                metrics.resource_pressure > self.thresholds['resource_pressure'])

    def _is_performance_degradation(self, metrics: CognitiveLoadMetrics) -> bool:
        """Проверка деградации производительности"""
        # Комплексная проверка на основе нескольких факторов
        error_rate_too_high = metrics.error_rate > self.thresholds['high_error_rate']
        response_time_too_long = metrics.response_time > 3.0  # больше 3 секунд
        confidence_too_low = metrics.confidence_level < self.thresholds['low_confidence']
        
        return error_rate_too_high or response_time_too_long or confidence_too_low

    def _is_tool_inefficiency(self, metrics: CognitiveLoadMetrics) -> bool:
        """Проверка неэффективности инструментов"""
        # Пока используем базовую проверку
        return metrics.error_rate > self.thresholds['high_error_rate']

    def _reduce_cognitive_load_action(self, metrics: CognitiveLoadMetrics) -> Dict[str, Any]:
        """Действие по снижению когнитивной нагрузки"""
        return {
            'type': 'resource_optimization',
            'parameters': {
                'reduce_parallelism': True,
                'simplify_processing': True,
                'increase_timeout': True
            },
            'description': 'Снижение параллелизма и упрощение обработки'
        }

    def _increase_confidence_action(self, metrics: CognitiveLoadMetrics) -> Dict[str, Any]:
        """Действие по повышению уверенности"""
        return {
            'type': 'confidence_enhancement',
            'parameters': {
                'additional_verification': True,
                'source_diversification': True,
                'confidence_threshold_adjustment': 0.1
            },
            'description': 'Дополнительная проверка и диверсификация источников'
        }

    def _optimize_resources_action(self, metrics: CognitiveLoadMetrics) -> Dict[str, Any]:
        """Действие по оптимизации ресурсов"""
        return {
            'type': 'resource_management',
            'parameters': {
                'memory_cleanup': True,
                'cache_optimization': True,
                'resource_allocation': 'increase'
            },
            'description': 'Оптимизация использования ресурсов'
        }

    def _change_strategy_action(self, metrics: CognitiveLoadMetrics) -> Dict[str, Any]:
        """Действие по изменению стратегии"""
        return {
            'type': 'strategy_adaptation',
            'parameters': {
                'strategy_type': 'simplification' if metrics.complexity_score > 0.7 else 'optimization',
                'processing_depth': 'shallow' if metrics.complexity_score > 0.7 else 'normal'
            },
            'description': 'Изменение стратегии обработки запросов'
        }

    def _adapt_tools_action(self, metrics: CognitiveLoadMetrics) -> Dict[str, Any]:
        """Действие по адаптации инструментов"""
        return {
            'type': 'tool_adaptation',
            'parameters': {
                'tool_selection': 'conservative',
                'fallback_tools': True,
                'tool_timeout_increase': True
            },
            'description': 'Адаптация выбора и использования инструментов'
        }

    async def analyze_for_adaptations(self, request: AgentRequest, response: AgentResponse) -> List[Adaptation]:
        """
        Анализ необходимости адаптаций на основе запроса и ответа
        
        Args:
            request: Запрос агента
            response: Ответ агента
            
        Returns:
            List[Adaptation]: Список предложенных адаптаций
        """
        try:
            # Сбор метрик для анализа
            metrics = await self._collect_cognitive_metrics(request, response)
            
            # Определение необходимых адаптаций
            required_adaptations = []
            
            for rule_name, rule in self.adaptation_rules.items():
                if rule['condition'](metrics):
                    adaptation_id = f"adapt_{rule_name}_{int(datetime.now().timestamp())}"
                    
                    adaptation = Adaptation(
                        id=adaptation_id,
                        type=AdaptationType.STRATEGY_CHANGE, # Временно используем общий тип
                        description=rule['description'],
                        trigger_condition={'rule': rule_name},
                        action=rule['action'](metrics),
                        priority=rule['priority'],
                        created_at=datetime.now(),
                        status=AdaptationStatus.PENDING
                    )
                    
                    required_adaptations.append(adaptation)
            
            # Сортировка по приоритету
            required_adaptations.sort(key=lambda x: x.priority)
            
            logger.info(f"Identified {len(required_adaptations)} adaptations needed")
            return required_adaptations
            
        except Exception as e:
            logger.error(f"Error in adaptation analysis: {e}")
            return []

    async def _collect_cognitive_metrics(self, request: AgentRequest, response: AgentResponse) -> CognitiveLoadMetrics:
        """Сбор когнитивных метрик на основе запроса и ответа"""
        # Используем базовые метрики, в реальности это будет более комплексный анализ
        return CognitiveLoadMetrics(
            response_time=response.execution_time if hasattr(response, 'execution_time') else 1.0,
            processing_time=response.execution_time if hasattr(response, 'execution_time') else 1.0,
            memory_usage=0.5,  # Заглушка - в реальности нужно получать из системы
            cpu_usage=0.3,     # Заглушка - в реальности нужно получать из системы
            active_tasks=5,    # Заглушка
            confidence_level=response.confidence if hasattr(response, 'confidence') else 0.7,
            error_rate=0.1 if "ошибка" in str(response.result).lower() else 0.02,
            complexity_score=0.5,  # Заглушка - в реальности анализировать сложность запроса
            resource_pressure=0.4,  # Заглушка
            timestamp=datetime.now()
        )

    async def apply_adaptations(self, adaptations: List[Adaptation]) -> List[Dict[str, Any]]:
        """
        Применение адаптаций
        
        Args:
            adaptations: Список адаптаций для применения
            
        Returns:
            List[Dict]: Результаты применения адаптаций
        """
        results = []
        
        for adaptation in adaptations:
            try:
                result = await self._apply_single_adaptation(adaptation)
                results.append(result)
                
                # Обновление статуса адаптации
                adaptation.status = AdaptationStatus.ACTIVE
                adaptation.last_applied = datetime.now()
                self.active_adaptations.append(adaptation.id)
                self.adaptations[adaptation.id] = adaptation
                
                logger.info(f"Applied adaptation: {adaptation.id} - {adaptation.description}")
                
            except Exception as e:
                logger.error(f"Failed to apply adaptation {adaptation.id}: {e}")
                results.append({
                    'adaptation_id': adaptation.id,
                    'status': 'failed',
                    'error': str(e)
                })
                
                # Обновление статуса адаптации
                adaptation.status = AdaptationStatus.FAILED
                self.adaptations[adaptation.id] = adaptation
        
        return results

    async def _apply_single_adaptation(self, adaptation: Adaptation) -> Dict[str, Any]:
        """Применение одиночной адаптации"""
        action_type = adaptation.action.get('type', '')
        
        if action_type == 'resource_optimization':
            return await self._apply_resource_optimization(adaptation.action['parameters'])
        elif action_type == 'confidence_enhancement':
            return await self._apply_confidence_enhancement(adaptation.action['parameters'])
        elif action_type == 'resource_management':
            return await self._apply_resource_management(adaptation.action['parameters'])
        elif action_type == 'strategy_adaptation':
            return await self._apply_strategy_adaptation(adaptation.action['parameters'])
        elif action_type == 'tool_adaptation':
            return await self._apply_tool_adaptation(adaptation.action['parameters'])
        else:
            return {
                'adaptation_id': adaptation.id,
                'status': 'unknown_action_type',
                'applied': False,
                'details': f"Unknown action type: {action_type}"
            }

    async def _apply_resource_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Применение оптимизации ресурсов"""
        try:
            # Применение параметров оптимизации
            if params.get('reduce_parallelism'):
                # Уменьшение параллелизма
                if hasattr(self.agent_core, 'max_concurrent_tasks'):
                    self.agent_core.max_concurrent_tasks = max(1, self.agent_core.max_concurrent_tasks // 2)
            
            if params.get('simplify_processing'):
                # Упрощение обработки (например, отключение сложных проверок)
                pass  # Реализация зависит от конкретной архитектуры
            
            if params.get('increase_timeout'):
                # Увеличение таймаутов
                if hasattr(self.agent_core, 'tool_timeout'):
                    self.agent_core.tool_timeout *= 1.5
            
            return {
                'adaptation_id': 'resource_optimization',
                'status': 'success',
                'applied': True,
                'details': 'Resource optimization applied',
                'parameters_applied': params
            }
        except Exception as e:
            return {
                'adaptation_id': 'resource_optimization',
                'status': 'failed',
                'applied': False,
                'error': str(e)
            }

    async def _apply_confidence_enhancement(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Применение повышения уверенности"""
        try:
            # Применение параметров повышения уверенности
            if params.get('additional_verification'):
                # Включение дополнительной проверки
                pass  # Реализация зависит от конкретной архитектуры
            
            if params.get('source_diversification'):
                # Диверсификация источников
                pass  # Реализация зависит от конкретной архитектуры
            
            if 'confidence_threshold_adjustment' in params:
                # Корректировка порога уверенности
                if hasattr(self.agent_core, 'config') and hasattr(self.agent_core.config, 'confidence_threshold'):
                    self.agent_core.config.confidence_threshold -= params['confidence_threshold_adjustment']
            
            return {
                'adaptation_id': 'confidence_enhancement',
                'status': 'success',
                'applied': True,
                'details': 'Confidence enhancement applied',
                'parameters_applied': params
            }
        except Exception as e:
            return {
                'adaptation_id': 'confidence_enhancement',
                'status': 'failed',
                'applied': False,
                'error': str(e)
            }

    async def _apply_resource_management(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Применение управления ресурсами"""
        try:
            # Применение параметров управления ресурсами
            if params.get('memory_cleanup'):
                # Очистка памяти
                if hasattr(self.agent_core, 'memory_manager'):
                    self.agent_core.memory_manager.cleanup_old_entries()
            
            if params.get('cache_optimization'):
                # Оптимизация кэша
                pass  # Реализация зависит от конкретной архитектуры
            
            if params.get('resource_allocation') == 'increase':
                # Увеличение выделения ресурсов
                pass  # Реализация зависит от конкретной архитектуры
            
            return {
                'adaptation_id': 'resource_management',
                'status': 'success',
                'applied': True,
                'details': 'Resource management applied',
                'parameters_applied': params
            }
        except Exception as e:
            return {
                'adaptation_id': 'resource_management',
                'status': 'failed',
                'applied': False,
                'error': str(e)
            }

    async def _apply_strategy_adaptation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Применение адаптации стратегии"""
        try:
            # Применение параметров адаптации стратегии
            if 'strategy_type' in params:
                strategy_type = params['strategy_type']
                # Изменение стратегии обработки
                if hasattr(self.agent_core, 'processing_strategy'):
                    self.agent_core.processing_strategy = strategy_type
            
            if 'processing_depth' in params:
                processing_depth = params['processing_depth']
                # Изменение глубины обработки
                if hasattr(self.agent_core, 'processing_depth'):
                    self.agent_core.processing_depth = processing_depth
            
            return {
                'adaptation_id': 'strategy_adaptation',
                'status': 'success',
                'applied': True,
                'details': 'Strategy adaptation applied',
                'parameters_applied': params
            }
        except Exception as e:
            return {
                'adaptation_id': 'strategy_adaptation',
                'status': 'failed',
                'applied': False,
                'error': str(e)
            }

    async def _apply_tool_adaptation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Применение адаптации инструментов"""
        try:
            # Применение параметров адаптации инструментов
            if params.get('tool_selection') == 'conservative':
                # Консервативный выбор инструментов
                if hasattr(self.agent_core, 'tool_orchestrator'):
                    self.agent_core.tool_orchestrator.use_conservative_selection = True
            
            if params.get('fallback_tools'):
                # Включение резервных инструментов
                if hasattr(self.agent_core, 'tool_orchestrator'):
                    self.agent_core.tool_orchestrator.enable_fallback_tools = True
            
            if params.get('tool_timeout_increase'):
                # Увеличение таймаутов инструментов
                if hasattr(self.agent_core, 'tool_timeout'):
                    self.agent_core.tool_timeout *= 1.3
            
            return {
                'adaptation_id': 'tool_adaptation',
                'status': 'success',
                'applied': True,
                'details': 'Tool adaptation applied',
                'parameters_applied': params
            }
        except Exception as e:
            return {
                'adaptation_id': 'tool_adaptation',
                'status': 'failed',
                'applied': False,
                'error': str(e)
            }

    async def evaluate_adaptation_effectiveness(self, adaptations: List[Adaptation], 
                                              before_metrics: CognitiveLoadMetrics,
                                              after_metrics: CognitiveLoadMetrics) -> Dict[str, Any]:
        """
        Оценка эффективности адаптаций
        
        Args:
            adaptations: Примененные адаптации
            before_metrics: Метрики до адаптации
            after_metrics: Метрики после адаптации
            
        Returns:
            Dict: Результаты оценки
        """
        try:
            # Расчет изменений метрик
            load_improvement = before_metrics.response_time - after_metrics.response_time
            confidence_improvement = after_metrics.confidence_level - before_metrics.confidence_level
            error_reduction = before_metrics.error_rate - after_metrics.error_rate
            
            # Общий показатель эффективности
            effectiveness_score = (
                load_improvement * 0.4 + 
                confidence_improvement * 0.3 + 
                error_reduction * 0.3
            )
            
            # Обновление эффективности для каждой адаптации
            for adaptation in adaptations:
                adaptation.effectiveness_score = effectiveness_score
                self.adaptations[adaptation.id] = adaptation
            
            results = {
                'overall_effectiveness': effectiveness_score,
                'load_improvement': load_improvement,
                'confidence_improvement': confidence_improvement,
                'error_reduction': error_reduction,
                'adaptations_evaluated': len(adaptations),
                'improvement_percentage': effectiveness_score * 100
            }
            
            logger.info(f"Adaptation effectiveness: {effectiveness_score:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating adaptation effectiveness: {e}")
            return {
                'error': str(e),
                'overall_effectiveness': 0.0,
                'adaptations_evaluated': 0
            }

    async def rollback_adaptation(self, adaptation_id: str, reason: str = "") -> bool:
        """
        Откат адаптации
        
        Args:
            adaptation_id: ID адаптации для отката
            reason: Причина отката
            
        Returns:
            bool: Успешность отката
        """
        try:
            if adaptation_id not in self.adaptations:
                logger.warning(f"Adaptation {adaptation_id} not found for rollback")
                return False
            
            adaptation = self.adaptations[adaptation_id]
            
            # Выполнение отката в зависимости от типа адаптации
            rollback_result = await self._perform_rollback(adaptation)
            
            if rollback_result:
                adaptation.status = AdaptationStatus.ROLLED_BACK
                adaptation.rollback_reason = reason
                self.active_adaptations.remove(adaptation_id)
                
                logger.info(f"Successfully rolled back adaptation: {adaptation_id}")
                return True
            else:
                logger.warning(f"Failed to rollback adaptation: {adaptation_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error rolling back adaptation {adaptation_id}: {e}")
            return False

    async def _perform_rollback(self, adaptation: Adaptation) -> bool:
        """Выполнение отката адаптации"""
        action_type = adaptation.action.get('type', '')
        
        try:
            if action_type == 'resource_optimization':
                return await self._rollback_resource_optimization(adaptation.action['parameters'])
            elif action_type == 'confidence_enhancement':
                return await self._rollback_confidence_enhancement(adaptation.action['parameters'])
            elif action_type == 'resource_management':
                return await self._rollback_resource_management(adaptation.action['parameters'])
            elif action_type == 'strategy_adaptation':
                return await self._rollback_strategy_adaptation(adaptation.action['parameters'])
            elif action_type == 'tool_adaptation':
                return await self._rollback_tool_adaptation(adaptation.action['parameters'])
            else:
                logger.warning(f"Unknown action type for rollback: {action_type}")
                return False
        except Exception as e:
            logger.error(f"Error in rollback operation: {e}")
            return False

    async def _rollback_resource_optimization(self, params: Dict[str, Any]) -> bool:
        """Откат оптимизации ресурсов"""
        try:
            # Восстановление исходных значений
            if hasattr(self.agent_core, 'max_concurrent_tasks'):
                self.agent_core.max_concurrent_tasks = getattr(self.agent_core, 'default_max_concurrent_tasks', 10)
            
            if hasattr(self.agent_core, 'tool_timeout'):
                self.agent_core.tool_timeout /= 1.5  # Вернуть к исходному значению
            
            return True
        except Exception:
            return False

    async def _rollback_confidence_enhancement(self, params: Dict[str, Any]) -> bool:
        """Откат повышения уверенности"""
        try:
            # Восстановление исходного порога уверенности
            if (hasattr(self.agent_core, 'config') and 
                hasattr(self.agent_core.config, 'confidence_threshold') and
                'confidence_threshold_adjustment' in params):
                self.agent_core.config.confidence_threshold += params['confidence_threshold_adjustment']
            
            return True
        except Exception:
            return False

    async def _rollback_resource_management(self, params: Dict[str, Any]) -> bool:
        """Откат управления ресурсами"""
        try:
            # Восстановление исходного состояния
            if params.get('memory_cleanup') and hasattr(self.agent_core, 'memory_manager'):
                # Восстановление памяти не требуется, так как очистка необратима
                pass
            
            return True
        except Exception:
            return False

    async def _rollback_strategy_adaptation(self, params: Dict[str, Any]) -> bool:
        """Откат адаптации стратегии"""
        try:
            # Восстановление исходной стратегии
            if 'strategy_type' in params and hasattr(self.agent_core, 'default_processing_strategy'):
                self.agent_core.processing_strategy = self.agent_core.default_processing_strategy
            
            if 'processing_depth' in params and hasattr(self.agent_core, 'default_processing_depth'):
                self.agent_core.processing_depth = self.agent_core.default_processing_depth
            
            return True
        except Exception:
            return False

    async def _rollback_tool_adaptation(self, params: Dict[str, Any]) -> bool:
        """Откат адаптации инструментов"""
        try:
            # Восстановление исходных настроек инструментов
            if params.get('tool_selection') == 'conservative' and hasattr(self.agent_core, 'tool_orchestrator'):
                self.agent_core.tool_orchestrator.use_conservative_selection = False
            
            if params.get('fallback_tools') and hasattr(self.agent_core, 'tool_orchestrator'):
                self.agent_core.tool_orchestrator.enable_fallback_tools = False
            
            if params.get('tool_timeout_increase') and hasattr(self.agent_core, 'tool_timeout'):
                self.agent_core.tool_timeout /= 1.3 # Вернуть к исходному значению
            
            return True
        except Exception:
            return False

    def get_active_adaptations(self) -> List[Dict[str, Any]]:
        """Получение списка активных адаптаций"""
        active_list = []
        
        for adaptation_id in self.active_adaptations:
            if adaptation_id in self.adaptations:
                adaptation = self.adaptations[adaptation_id]
                active_list.append({
                    'id': adaptation.id,
                    'type': adaptation.type.value,
                    'description': adaptation.description,
                    'created_at': adaptation.created_at.isoformat(),
                    'last_applied': adaptation.last_applied.isoformat() if adaptation.last_applied else None,
                    'effectiveness_score': adaptation.effectiveness_score,
                    'action': adaptation.action
                })
        
        return active_list

    def get_adaptation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Получение истории адаптаций"""
        history_list = []
        
        # Берем последние N адаптаций из истории
        recent_adaptations = self.adaptation_history[-limit:] if len(self.adaptation_history) > limit else self.adaptation_history
        
        for adaptation in reversed(recent_adaptations):
            history_list.append({
                'id': adaptation.id,
                'type': adaptation.type.value,
                'description': adaptation.description,
                'status': adaptation.status.value,
                'created_at': adaptation.created_at.isoformat(),
                'last_applied': adaptation.last_applied.isoformat() if adaptation.last_applied else None,
                'effectiveness_score': adaptation.effectiveness_score,
                'rollback_reason': adaptation.rollback_reason,
                'action': adaptation.action
            })
        
        return history_list

    async def auto_adapt(self, request: AgentRequest, response: AgentResponse) -> Dict[str, Any]:
        """
        Автоматическая адаптация на основе взаимодействия
        
        Args:
            request: Запрос агента
            response: Ответ агента
            
        Returns:
            Dict: Результат адаптации
        """
        try:
            logger.info("Starting auto adaptation process")
            
            # Анализ необходимости адаптаций
            required_adaptations = await self.analyze_for_adaptations(request, response)
            
            if not required_adaptations:
                logger.info("No adaptations needed")
                return {
                    'adaptations_applied': 0,
                    'adaptations_identified': 0,
                    'message': 'No adaptations needed'
                }
            
            # Применение адаптаций
            application_results = await self.apply_adaptations(required_adaptations)
            
            # Добавление адаптаций в историю
            for adaptation in required_adaptations:
                self.adaptation_history.append(adaptation)
                # Ограничение истории
                if len(self.adaptation_history) > 1000:
                    self.adaptation_history = self.adaptation_history[-500:]
            
            successful_applications = [r for r in application_results if r.get('applied', False)]
            
            result = {
                'adaptations_identified': len(required_adaptations),
                'adaptations_applied': len(successful_applications),
                'application_results': application_results,
                'active_adaptations': len(self.active_adaptations),
                'message': f'Applied {len(successful_applications)} out of {len(required_adaptations)} adaptations'
            }
            
            logger.info(f"Auto adaptation completed: {result['message']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in auto adaptation: {e}")
            return {
                'error': str(e),
                'adaptations_applied': 0,
                'adaptations_identified': 0
            }

    async def get_adaptation_recommendations(self, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Получение рекомендаций по адаптациям
        
        Args:
            context: Контекст для анализа (необязательно)
            
        Returns:
            List[Dict]: Рекомендации по адаптациям
        """
        recommendations = []
        
        # Временно возвращаем общие рекомендации
        # В реальности это будет зависеть от текущего состояния системы
        base_recommendations = [
            {
                'type': 'performance_optimization',
                'priority': 'high',
                'description': 'Оптимизировать производительность при высокой нагрузке',
                'conditions': 'cognitive_load > 0.7'
            },
            {
                'type': 'confidence_enhancement',
                'priority': 'medium',
                'description': 'Повысить уверенность при низких оценках',
                'conditions': 'confidence < 0.5'
            },
            {
                'type': 'resource_management',
                'priority': 'medium',
                'description': 'Управление ресурсами при высоком использовании',
                'conditions': 'memory_usage > 0.8 OR cpu_usage > 0.85'
            }
        ]
        
        recommendations.extend(base_recommendations)
        
        return recommendations

    def enhance_adaptation_algorithms(self):
        """
        Улучшение алгоритмов адаптации с использованием продвинутых методов
        """
        # Реализация улучшенных алгоритмов адаптации
        # 1. Введение машинного обучения для прогнозирования необходимости адаптаций
        # 2. Использование методов усиленного обучения для оптимизации адаптаций
        # 3. Внедрение адаптивных порогов на основе исторических данных
        
        logger.info("Enhanced adaptation algorithms initialized")
        
        # Обновление правил адаптации с использованием улучшенных методов
        self._update_adaptation_rules_with_ml()
        self._implement_predictive_adaptations()
        self._optimize_adaptation_thresholds()

    def _update_adaptation_rules_with_ml(self):
        """Обновление правил адаптации с использованием машинного обучения"""
        # Здесь будет реализация ML-моделей для определения адаптаций
        # Пока что добавим улучшенные правила
        enhanced_rules = {
            # Правило для предсказания перегрузки системы
            'predictive_load_reduction': {
                'condition': lambda metrics: self._is_predictive_load_risk(metrics),
                'action': self._predictive_load_reduction_action,
                'priority': 1,
                'description': 'Предиктивное снижение нагрузки'
            },
            # Правило для динамической настройки скорости обучения
            'dynamic_learning_rate': {
                'condition': lambda metrics: self._is_learning_optimization_needed(metrics),
                'action': self._dynamic_learning_rate_action,
                'priority': 6,
                'description': 'Динамическая настройка скорости обучения'
            },
            # Правило для адаптации стратегии обучения
            'learning_strategy_adaptation': {
                'condition': lambda metrics: self._is_learning_strategy_update_needed(metrics),
                'action': self._learning_strategy_adaptation_action,
                'priority': 7,
                'description': 'Адаптация стратегии обучения'
            }
        }
        self.adaptation_rules.update(enhanced_rules)

    def _is_predictive_load_risk(self, metrics: CognitiveLoadMetrics) -> bool:
        """Проверка риска перегрузки на основе предиктивного анализа"""
        # Пример: если тренд нагрузки увеличивается в течение последних 5 измерений
        if len(self.analyzer.load_history) >= 5:
            recent_metrics = self.analyzer.load_history[-5:]
            recent_scores = [self.analyzer.calculate_load_score(m) for m in recent_metrics]
            # Если средняя нагрузка за последние 3 измерения выше, чем за предыдущие 2
            if len(recent_scores) >= 5:
                recent_avg = sum(recent_scores[-3:]) / 3
                previous_avg = sum(recent_scores[:-3]) / max(1, len(recent_scores) - 3) if len(recent_scores) > 3 else recent_scores[0]
                return recent_avg > previous_avg * 1.1  # Рост более чем на 10%
        return False

    def _is_learning_optimization_needed(self, metrics: CognitiveLoadMetrics) -> bool:
        """Проверка необходимости оптимизации скорости обучения"""
        # Оптимизация нужна, если уверенность снижается или ошибка увеличивается
        return (metrics.confidence_level < 0.5 or metrics.error_rate > 0.2)

    def _is_learning_strategy_update_needed(self, metrics: CognitiveLoadMetrics) -> bool:
        """Проверка необходимости обновления стратегии обучения"""
        # Обновление нужно, если текущая стратегия неэффективна
        return (metrics.error_rate > 0.3 or metrics.complexity_score > 0.8)

    def _predictive_load_reduction_action(self, metrics: CognitiveLoadMetrics) -> Dict[str, Any]:
        """Действие по предиктивному снижению нагрузки"""
        return {
            'type': 'predictive_load_reduction',
            'parameters': {
                'proactive_resource_allocation': True,
                'early_optimization': True,
                'preventive_measures': True
            },
            'description': 'Предиктивное снижение нагрузки'
        }

    def _dynamic_learning_rate_action(self, metrics: CognitiveLoadMetrics) -> Dict[str, Any]:
        """Действие по динамической настройке скорости обучения"""
        # Адаптируем скорость обучения в зависимости от текущей уверенности и ошибки
        base_rate = getattr(self, 'learning_rate', 0.01)
        confidence_factor = metrics.confidence_level
        error_factor = 1 + metrics.error_rate  # Увеличиваем скорость при высокой ошибке
        
        adjusted_rate = base_rate * confidence_factor * error_factor
        # Ограничиваем скорость в пределах разумного диапазона
        adjusted_rate = max(0.001, min(0.1, adjusted_rate))
        
        return {
            'type': 'dynamic_learning_rate_adjustment',
            'parameters': {
                'new_learning_rate': adjusted_rate,
                'confidence_factor': confidence_factor,
                'error_factor': error_factor
            },
            'description': f'Динамическая настройка скорости обучения: {adjusted_rate}'
        }

    def _learning_strategy_adaptation_action(self, metrics: CognitiveLoadMetrics) -> Dict[str, Any]:
        """Действие по адаптации стратегии обучения"""
        # Выбираем стратегию в зависимости от текущих метрик
        if metrics.complexity_score > 0.7:
            strategy = 'deep_learning'
        elif metrics.error_rate > 0.3:
            strategy = 'error_focused_learning'
        else:
            strategy = 'balanced_learning'
            
        return {
            'type': 'learning_strategy_adaptation',
            'parameters': {
                'new_strategy': strategy,
                'complexity_threshold': metrics.complexity_score,
                'error_threshold': metrics.error_rate
            },
            'description': f'Адаптация стратегии обучения: {strategy}'
        }

    def _implement_predictive_adaptations(self):
        """Реализация предиктивных адаптаций"""
        # Здесь будет реализация предиктивных моделей
        # Пока добавим базовую логику
        logger.info("Predictive adaptations framework initialized")

    def _optimize_adaptation_thresholds(self):
        """Оптимизация порогов адаптации на основе исторических данных"""
        # Адаптируем пороги на основе исторических данных
        if self.analyzer.load_history:
            # Рассчитываем статистику по историческим данным
            load_scores = [self.analyzer.calculate_load_score(m) for m in self.analyzer.load_history]
            avg_load = sum(load_scores) / len(load_scores) if load_scores else 0.5
            
            # Адаптируем пороги
            self.thresholds['high_cognitive_load'] = min(0.9, avg_load * 1.3)  # Порог на 30% выше среднего
            self.thresholds['low_confidence'] = max(0.2, avg_load * 0.5)      # Порог на основе средней нагрузки
            self.thresholds['high_error_rate'] = max(0.1, avg_load * 0.3)    # Адаптируем под нагрузку системы
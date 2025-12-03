import time
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime

from ..core.models import (
    Adaptation, AdaptationResult, AdaptationType, Insight,
    AgentConfig
)

if TYPE_CHECKING:
    from ..core.agent_core import AgentCore

logger = logging.getLogger(__name__)


class StrategyAdaptation:
    """Адаптация стратегии"""

    def __init__(self, id: str, insight_id: str, type: AdaptationType,
                 description: str, changes: Dict[str, Any], confidence: float):
        self.id = id
        self.insight_id = insight_id
        self.type = type
        self.description = description
        self.changes = changes
        self.confidence = confidence
        self.applied_at = None
        self.effectiveness = None  # Будет измерено после применения


class ParameterAdaptation:
    """Адаптация параметров"""

    def __init__(self, id: str, insight_id: str, type: AdaptationType,
                 description: str, parameters: Dict[str, Any], confidence: float):
        self.id = id
        self.insight_id = insight_id
        self.type = type
        self.description = description
        self.parameters = parameters
        self.confidence = confidence
        self.applied_at = None
        self.effectiveness = None


class ToolPreferenceAdaptation:
    """Адаптация предпочтений инструментов"""

    def __init__(self, id: str, insight_id: str, type: AdaptationType,
                 description: str, tool_preferences: Dict[str, Any], confidence: float):
        self.id = id
        self.insight_id = insight_id
        self.type = type
        self.description = description
        self.tool_preferences = tool_preferences
        self.confidence = confidence
        self.applied_at = None
        self.effectiveness = None


class MemoryOptimizationAdaptation:
    """Адаптация оптимизации памяти"""

    def __init__(self, id: str, insight_id: str, type: AdaptationType,
                 description: str, memory_settings: Dict[str, Any], confidence: float):
        self.id = id
        self.insight_id = insight_id
        self.type = type
        self.description = description
        self.memory_settings = memory_settings
        self.confidence = confidence
        self.applied_at = None
        self.effectiveness = None


class AdaptationEngine:
    """Двигатель адаптации агента"""

    def __init__(self, agent_core: 'AgentCore', reflection_engine):
        self.agent_core = agent_core
        self.reflection_engine = reflection_engine

        # Активные адаптации
        self.active_adaptations: List[Any] = []
        self.adaptation_history: List[Dict[str, Any]] = []

        # Базовые стратегии адаптации
        self.adaptation_strategies = {
            'error_pattern': self._create_error_adaptation,
            'performance_issue': self._create_performance_adaptation,
            'strategy_improvement': self._create_strategy_adaptation,
            'feedback_insight': self._create_feedback_adaptation
        }

        logger.info("AdaptationEngine initialized")

    async def adapt_based_on_insights(self, insights: List[Insight]) -> AdaptationResult:
        """Адаптация агента на основе полученных insights"""
        adaptations = []

        for insight in insights:
            if insight.confidence > 0.7:  # Только высоконадежные insights
                adaptation = await self._create_adaptation(insight)
                if adaptation:
                    adaptations.append(adaptation)

        # Применение адаптаций
        applied_adaptations = []
        for adaptation in adaptations:
            try:
                success = await self._apply_adaptation(adaptation)
                if success:
                    applied_adaptations.append(adaptation)
                    self.active_adaptations.append(adaptation)
                    logger.info(f"Successfully applied adaptation: {adaptation.id}")
                else:
                    logger.warning(f"Failed to apply adaptation: {adaptation.id}")
            except Exception as e:
                logger.error(f"Error applying adaptation {adaptation.id}: {e}")

        result = AdaptationResult(
            insights_processed=len(insights),
            adaptations_created=len(adaptations),
            adaptations_applied=len(applied_adaptations),
            active_adaptations=len(self.active_adaptations)
        )

        # Сохранение в истории
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'insights_count': len(insights),
            'adaptations_applied': len(applied_adaptations),
            'result': result
        })

        return result

    async def _create_adaptation(self, insight: Insight) -> Optional[Any]:
        """Создание адаптации на основе insight"""
        strategy = self.adaptation_strategies.get(insight.type.value)
        if strategy:
            return await strategy(insight)
        else:
            logger.warning(f"No adaptation strategy for insight type: {insight.type.value}")
            return None

    async def _create_error_adaptation(self, insight: Insight) -> Optional[Any]:
        """Создание адаптации для паттернов ошибок"""
        if 'error' in insight.description.lower():
            # Адаптация параметров валидации
            return ParameterAdaptation(
                id=f"error_param_{insight.id}_{int(time.time())}",
                insight_id=insight.id,
                type=AdaptationType.PARAMETER_TUNING,
                description=f"Настройка параметров для исправления: {insight.description}",
                parameters={
                    'validation_level': 'strict',
                    'error_tolerance': 0.1,
                    'retry_attempts': 3
                },
                confidence=insight.confidence
            )
        return None

    async def _create_performance_adaptation(self, insight: Insight) -> Optional[Any]:
        """Создание адаптации для проблем производительности"""
        description_lower = insight.description.lower()

        if 'время' in description_lower or 'time' in description_lower:
            # Оптимизация времени выполнения
            return StrategyAdaptation(
                id=f"perf_time_{insight.id}_{int(time.time())}",
                insight_id=insight.id,
                type=AdaptationType.STRATEGY_CHANGE,
                description="Оптимизация стратегии для сокращения времени выполнения",
                changes={
                    'tool_selection': 'fastest_available',
                    'parallel_processing': True,
                    'caching_level': 'aggressive'
                },
                confidence=insight.confidence
            )

        elif 'память' in description_lower or 'memory' in description_lower:
            # Оптимизация памяти
            return MemoryOptimizationAdaptation(
                id=f"perf_memory_{insight.id}_{int(time.time())}",
                insight_id=insight.id,
                type=AdaptationType.MEMORY_OPTIMIZATION,
                description="Оптимизация использования памяти",
                memory_settings={
                    'max_cache_size': 100,
                    'cleanup_interval': 300,
                    'compression_enabled': True
                },
                confidence=insight.confidence
            )

        elif 'инструмент' in description_lower or 'tool' in description_lower:
            # Изменение предпочтений инструментов
            return ToolPreferenceAdaptation(
                id=f"perf_tools_{insight.id}_{int(time.time())}",
                insight_id=insight.id,
                type=AdaptationType.TOOL_PREFERENCE,
                description="Изменение предпочтений инструментов для повышения эффективности",
                tool_preferences={
                    'prefer_fast_tools': True,
                    'avoid_slow_tools': True,
                    'tool_timeout': 5.0
                },
                confidence=insight.confidence
            )

        return None

    async def _create_strategy_adaptation(self, insight: Insight) -> Optional[Any]:
        """Создание адаптации для улучшения стратегии"""
        return StrategyAdaptation(
            id=f"strategy_{insight.id}_{int(time.time())}",
            insight_id=insight.id,
            type=AdaptationType.STRATEGY_CHANGE,
            description=f"Изменение стратегии на основе: {insight.description}",
            changes={
                'reasoning_depth': 'adaptive',
                'tool_selection_strategy': 'performance_based',
                'fallback_enabled': True
            },
            confidence=insight.confidence
        )

    async def _create_feedback_adaptation(self, insight: Insight) -> Optional[Any]:
        """Создание адаптации на основе обратной связи"""
        # Адаптация на основе обратной связи обычно менее конкретна
        return ParameterAdaptation(
            id=f"feedback_{insight.id}_{int(time.time())}",
            insight_id=insight.id,
            type=AdaptationType.PARAMETER_TUNING,
            description=f"Настройка параметров на основе обратной связи: {insight.description}",
            parameters={
                'response_detail_level': 'adaptive',
                'user_preference_learning': True,
                'feedback_weight': 0.8
            },
            confidence=insight.confidence * 0.8  # Немного снижаем уверенность
        )

    async def _apply_adaptation(self, adaptation: Any) -> bool:
        """Применение адаптации к агенту"""
        try:
            if isinstance(adaptation, StrategyAdaptation):
                return await self._apply_strategy_change(adaptation)
            elif isinstance(adaptation, ParameterAdaptation):
                return await self._apply_parameter_change(adaptation)
            elif isinstance(adaptation, ToolPreferenceAdaptation):
                return await self._apply_tool_preference_change(adaptation)
            elif isinstance(adaptation, MemoryOptimizationAdaptation):
                return await self._apply_memory_optimization(adaptation)
            else:
                logger.warning(f"Unknown adaptation type: {type(adaptation)}")
                return False
        except Exception as e:
            logger.error(f"Error applying adaptation {adaptation.id}: {e}")
            return False

    async def _apply_strategy_change(self, adaptation: StrategyAdaptation) -> bool:
        """Применение изменения стратегии"""
        try:
            # Изменение конфигурации агента
            if hasattr(self.agent_core, 'config'):
                for key, value in adaptation.changes.items():
                    if hasattr(self.agent_core.config, key):
                        setattr(self.agent_core.config, key, value)
                        logger.info(f"Updated agent config {key} = {value}")

            # Изменение стратегии в tool orchestrator
            if hasattr(self.agent_core, 'tool_orchestrator'):
                for key, value in adaptation.changes.items():
                    if hasattr(self.agent_core.tool_orchestrator, key):
                        setattr(self.agent_core.tool_orchestrator, key, value)
                        logger.info(f"Updated tool orchestrator {key} = {value}")

            adaptation.applied_at = datetime.now()
            return True

        except Exception as e:
            logger.error(f"Failed to apply strategy change: {e}")
            return False

    async def _apply_parameter_change(self, adaptation: ParameterAdaptation) -> bool:
        """Применение изменения параметров"""
        try:
            # Изменение конфигурации агента
            if hasattr(self.agent_core, 'config'):
                for param, value in adaptation.parameters.items():
                    if hasattr(self.agent_core.config, param):
                        setattr(self.agent_core.config, param, value)
                        logger.info(f"Updated agent parameter {param} = {value}")

            # Специфические параметры для компонентов
            if 'validation_level' in adaptation.parameters:
                # Изменение уровня валидации в query analyzer
                if hasattr(self.agent_core, 'query_analyzer'):
                    self.agent_core.query_analyzer.validation_level = adaptation.parameters['validation_level']

            adaptation.applied_at = datetime.now()
            return True

        except Exception as e:
            logger.error(f"Failed to apply parameter change: {e}")
            return False

    async def _apply_tool_preference_change(self, adaptation: ToolPreferenceAdaptation) -> bool:
        """Применение изменения предпочтений инструментов"""
        try:
            if hasattr(self.agent_core, 'tool_orchestrator'):
                orchestrator = self.agent_core.tool_orchestrator

                for pref, value in adaptation.tool_preferences.items():
                    if hasattr(orchestrator, pref):
                        setattr(orchestrator, pref, value)
                        logger.info(f"Updated tool preference {pref} = {value}")

                # Пересчет предпочтений инструментов
                if hasattr(orchestrator, 'recalculate_preferences'):
                    await orchestrator.recalculate_preferences()

            adaptation.applied_at = datetime.now()
            return True

        except Exception as e:
            logger.error(f"Failed to apply tool preference change: {e}")
            return False

    async def _apply_memory_optimization(self, adaptation: MemoryOptimizationAdaptation) -> bool:
        """Применение оптимизации памяти"""
        try:
            if hasattr(self.agent_core, 'memory_manager'):
                memory_mgr = self.agent_core.memory_manager

                for setting, value in adaptation.memory_settings.items():
                    if hasattr(memory_mgr, setting):
                        setattr(memory_mgr, setting, value)
                        logger.info(f"Updated memory setting {setting} = {value}")

                # Запуск оптимизации памяти
                if hasattr(memory_mgr, 'optimize_memory'):
                    await memory_mgr.optimize_memory()

            adaptation.applied_at = datetime.now()
            return True

        except Exception as e:
            logger.error(f"Failed to apply memory optimization: {e}")
            return False

    async def evaluate_adaptation_effectiveness(self) -> Dict[str, Any]:
        """Оценка эффективности примененных адаптаций"""
        if not self.active_adaptations:
            return {'effectiveness': 0, 'summary': 'Нет активных адаптаций'}

        effective_adaptations = 0
        total_adaptations = len(self.active_adaptations)

        for adaptation in self.active_adaptations:
            if adaptation.effectiveness is not None:
                if adaptation.effectiveness > 0.5:  # Считаем эффективными если > 0.5
                    effective_adaptations += 1
            else:
                # Если эффективность не измерена, оцениваем по времени применения
                days_since_applied = (datetime.now() - adaptation.applied_at).days
                if days_since_applied < 7:  # Новые адаптации
                    effective_adaptations += 0.5  # Предполагаем 50% эффективности

        effectiveness_ratio = effective_adaptations / total_adaptations

        # Анализ по типам адаптаций
        type_effectiveness = {}
        for adaptation in self.active_adaptations:
            type_key = adaptation.type.value
            if type_key not in type_effectiveness:
                type_effectiveness[type_key] = {'count': 0, 'effective': 0}

            type_effectiveness[type_key]['count'] += 1
            if adaptation.effectiveness and adaptation.effectiveness > 0.5:
                type_effectiveness[type_key]['effective'] += 1

        return {
            'total_adaptations': total_adaptations,
            'effective_adaptations': effective_adaptations,
            'effectiveness_ratio': effectiveness_ratio,
            'type_effectiveness': type_effectiveness,
            'recommendations': self._generate_adaptation_recommendations(effectiveness_ratio, type_effectiveness)
        }

    def _generate_adaptation_recommendations(self, effectiveness_ratio: float,
                                           type_effectiveness: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по адаптации"""
        recommendations = []

        if effectiveness_ratio < 0.5:
            recommendations.append("Эффективность адаптаций низкая - рассмотреть пересмотр стратегий адаптации")

        # Рекомендации по типам
        for type_name, stats in type_effectiveness.items():
            if stats['count'] > 0:
                type_ratio = stats['effective'] / stats['count']
                if type_ratio < 0.3:
                    recommendations.append(f"Тип адаптации '{type_name}' имеет низкую эффективность - рассмотреть альтернативы")

        if not recommendations:
            recommendations.append("Адаптации работают эффективно - продолжить текущий подход")

        return recommendations

    async def rollback_adaptation(self, adaptation_id: str) -> bool:
        """Откат адаптации"""
        try:
            # Найти адаптацию
            adaptation = None
            for active in self.active_adaptations:
                if active.id == adaptation_id:
                    adaptation = active
                    break

            if not adaptation:
                logger.warning(f"Adaptation {adaptation_id} not found in active adaptations")
                return False

            # Откат изменений (упрощенная версия - в реальности нужна история изменений)
            # Здесь можно реализовать откат конкретных изменений

            # Удалить из активных
            self.active_adaptations.remove(adaptation)

            logger.info(f"Rolled back adaptation: {adaptation_id}")
            return True

        except Exception as e:
            logger.error(f"Error rolling back adaptation {adaptation_id}: {e}")
            return False

    def get_active_adaptations(self) -> List[Dict[str, Any]]:
        """Получение списка активных адаптаций"""
        return [
            {
                'id': adaptation.id,
                'type': adaptation.type.value,
                'description': adaptation.description,
                'confidence': adaptation.confidence,
                'applied_at': adaptation.applied_at,
                'effectiveness': adaptation.effectiveness
            }
            for adaptation in self.active_adaptations
        ]

    def clear_adaptations(self):
        """Очистка всех адаптаций"""
        self.active_adaptations.clear()
        logger.info("All adaptations cleared")

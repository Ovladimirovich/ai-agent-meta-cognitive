"""
Двигатель обучения AI агента
Фаза 3: Обучение и Адаптация
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .models import (
    AgentExperience,
    ProcessedExperience,
    Pattern,
    LearningResult,
    LearningMetrics,
    AdaptationResult,
    SkillUpdate
)
from .experience_processor import ExperienceProcessor
from .pattern_memory import PatternMemory
from .cognitive_maps import CognitiveMaps
from .skill_development import SkillDevelopment
from .predictive_adaptation import PredictiveAdaptation
from .meta_learning_system import MetaLearningSystem

logger = logging.getLogger(__name__)


class LearningEngine:
    """
    Основной двигатель обучения агента, координирующий все компоненты обучения
    """

    def __init__(self, agent_core=None, memory_manager=None, redis_client=None):
        """
        Инициализация двигателя обучения

        Args:
            agent_core: Ядро агента для доступа к адаптациям
            memory_manager: Менеджер памяти агента
            redis_client: Redis клиент для персистентности
        """
        self.agent_core = agent_core
        self.memory_manager = memory_manager

        # Компоненты обучения
        self.pattern_memory = PatternMemory(redis_client)
        self.experience_processor = ExperienceProcessor()
        self.cognitive_maps = CognitiveMaps()
        self.skill_development = SkillDevelopment()
        self.predictive_adaptation = PredictiveAdaptation(
            learning_engine=self,
            skill_development=self.skill_development,
            cognitive_maps=self.cognitive_maps
        )
        self.meta_learning_system = MetaLearningSystem(learning_engine=self)

        # Метрики обучения
        self.learning_metrics = LearningMetrics(
            total_experiences_processed=0,
            average_learning_effectiveness=0,
            patterns_discovered=0,
            skills_improved=0,
            cognitive_maps_updated=0,
            adaptation_success_rate=0,
            time_period="all"
        )

        # История обучения
        self.learning_history: List[LearningResult] = []

        # Статистика производительности
        self.performance_stats = {
            'total_learning_sessions': 0,
            'average_session_time': 0,
            'learning_effectiveness_trend': [],
            'pattern_discovery_rate': 0
        }

        logger.info("LearningEngine initialized")

    async def learn_from_experience(self, experience: AgentExperience) -> LearningResult:
        """
        Обучение на основе опыта агента

        Args:
            experience: Опыт агента для обучения

        Returns:
            LearningResult: Результаты обучения
        """
        learning_start = time.time()

        try:
            logger.info(f"Starting learning from experience {experience.id}")

            # Шаг 1: Обработка опыта
            processed_experience = await self.experience_processor.process(experience)
            # Конвертируем original_experience в dict для совместимости с моделью
            if hasattr(processed_experience, 'original_experience') and hasattr(processed_experience.original_experience, 'model_dump'):
                processed_experience.original_experience = processed_experience.original_experience.model_dump()

            # Шаг 2: Извлечение паттернов
            patterns = await self.pattern_memory.extract_patterns(processed_experience)

            # Шаг 3: Обновление когнитивных карт
            cognitive_updates = await self.cognitive_maps.update_maps(patterns, processed_experience)

            # Шаг 4: Развитие навыков
            skill_updates = await self.skill_development.develop_skills(patterns, cognitive_updates)

            # Шаг 5: Применение обучения через адаптацию
            adaptation_result = await self._apply_learning(skill_updates, cognitive_updates)

            # Шаг 6: Оценка эффективности обучения
            learning_effectiveness = self._evaluate_learning_effectiveness(
                processed_experience, adaptation_result
            )

            # Шаг 7: Создание результата обучения
            result = LearningResult(
                experience_processed=processed_experience,
                patterns_extracted=len(patterns),
                cognitive_updates=len(cognitive_updates),
                skills_developed=len(skill_updates),
                adaptation_applied=adaptation_result,
                learning_effectiveness=learning_effectiveness,
                learning_time=time.time() - learning_start,
                timestamp=datetime.now()
            )

            # Шаг 8: Сохранение результатов
            await self._store_learning_result(result)

            # Шаг 9: Обновление метрик
            self._update_learning_metrics(result)

            logger.info(f"Learning completed for experience {experience.id}. "
                       f"Effectiveness: {learning_effectiveness:.2f}")

            return result

        except Exception as e:
            logger.error(f"Learning failed for experience {experience.id}: {e}")
            # Возврат минимального результата в случае ошибки
            return LearningResult(
                experience_processed=ProcessedExperience(
                    original_experience=experience,
                    key_elements=[],
                    significance_score=0.0,
                    categories=[],
                    lessons=[],
                    processing_timestamp=datetime.now()
                ),
                patterns_extracted=0,
                cognitive_updates=0,
                skills_developed=0,
                adaptation_applied=AdaptationResult(
                    adaptations_created=0,
                    adaptations_applied=0,
                    success_rate=0.0,
                    performance_impact={},
                    applied_at=datetime.now()
                ),
                learning_effectiveness=0.0,
                learning_time=time.time() - learning_start,
                timestamp=datetime.now()
            )

    async def _update_cognitive_maps_placeholder(self, patterns: List[Pattern], experience: ProcessedExperience) -> List[Dict[str, Any]]:
        """
        Заглушка для обновления когнитивных карт
        (Будет заменено на реальную реализацию CognitiveMaps)

        Args:
            patterns: Извлеченные паттерны
            experience: Обработанный опыт

        Returns:
            List[Dict[str, Any]]: Список обновлений карт
        """
        # Временная реализация - просто подсчет обновлений
        updates = []

        if patterns:
            updates.append({
                'map_name': 'domain_knowledge',
                'changes': {'patterns_added': len(patterns)},
                'confidence': 0.7,
                'impact_assessment': {'knowledge_breadth': len(patterns) * 0.1},
                'timestamp': datetime.now()
            })

        return updates

    async def _develop_skills_placeholder(self, patterns: List[Pattern], cognitive_updates: List[Dict[str, Any]]) -> List[SkillUpdate]:
        """
        Заглушка для развития навыков
        (Будет заменено на реальную реализацию SkillDevelopment)

        Args:
            patterns: Извлеченные паттерны
            cognitive_updates: Обновления когнитивных карт

        Returns:
            List[SkillUpdate]: Список обновлений навыков
        """
        # Временная реализация - базовое развитие навыков
        skill_updates = []

        if patterns:
            # Развитие навыка анализа запросов
            query_analysis_skill = SkillUpdate(
                skill_name="query_analysis",
                previous_level=0.5,  # Предполагаемое предыдущее значение
                new_level=min(0.5 + len(patterns) * 0.05, 1.0),
                progress=len(patterns) * 0.05,
                confidence=0.8,
                improvements=["Улучшен анализ паттернов запросов", "Расширено понимание контекста"]
            )
            skill_updates.append(query_analysis_skill)

        return skill_updates

    async def _apply_learning(self, skill_updates: List[SkillUpdate], cognitive_updates: List[Dict[str, Any]]) -> AdaptationResult:
        """
        Применение результатов обучения через систему адаптации

        Args:
            skill_updates: Обновления навыков
            cognitive_updates: Обновления когнитивных карт

        Returns:
            AdaptationResult: Результат применения адаптаций
        """
        adaptations_applied = 0
        performance_impact = {}

        try:
            # Применение обновлений навыков
            for skill_update in skill_updates:
                if self.agent_core and hasattr(self.agent_core, 'adaptation_engine'):
                    # Использование адаптационного движка агента
                    adaptation = {
                        'id': f"skill_adaptation_{skill_update.skill_name}_{int(time.time())}",
                        'type': 'skill_improvement',
                        'description': f"Improvement of {skill_update.skill_name} skill",
                        'changes': {
                            'skill_name': skill_update.skill_name,
                            'level_increase': skill_update.new_level - skill_update.previous_level
                        },
                        'confidence': skill_update.confidence,
                        'estimated_impact': {
                            'performance_boost': skill_update.progress * 0.1,
                            'accuracy_improvement': skill_update.progress * 0.05
                        }
                    }

                    # Применение адаптации
                    result = await self.agent_core.adaptation_engine.apply_adaptation(adaptation)
                    if result.get('success', False):
                        adaptations_applied += 1
                        performance_impact.update(result.get('impact', {}))
                else:
                    # Fallback: просто логирование
                    logger.info(f"Applied skill update: {skill_update.skill_name} -> {skill_update.new_level}")
                    adaptations_applied += 1

            # Применение когнитивных обновлений
            for cognitive_update in cognitive_updates:
                if self.agent_core and hasattr(self.agent_core, 'adaptation_engine'):
                    adaptation = {
                        'id': f"cognitive_adaptation_{cognitive_update['map_name']}_{int(time.time())}",
                        'type': 'cognitive_update',
                        'description': f"Update of {cognitive_update['map_name']} cognitive map",
                        'changes': cognitive_update['changes'],
                        'confidence': cognitive_update['confidence'],
                        'estimated_impact': cognitive_update['impact_assessment']
                    }

                    result = await self.agent_core.adaptation_engine.apply_adaptation(adaptation)
                    if result.get('success', False):
                        adaptations_applied += 1
                        performance_impact.update(result.get('impact', {}))
                else:
                    logger.info(f"Applied cognitive update: {cognitive_update['map_name']}")
                    adaptations_applied += 1

        except Exception as e:
            logger.error(f"Failed to apply learning adaptations: {e}")

        success_rate = adaptations_applied / max(len(skill_updates) + len(cognitive_updates), 1)

        return AdaptationResult(
            adaptations_created=len(skill_updates) + len(cognitive_updates),
            adaptations_applied=adaptations_applied,
            success_rate=success_rate,
            performance_impact=performance_impact,
            applied_at=datetime.now()
        )

    def _evaluate_learning_effectiveness(self, experience: ProcessedExperience, adaptation_result: AdaptationResult) -> float:
        """
        Оценка эффективности обучения

        Args:
            experience: Обработанный опыт
            adaptation_result: Результат применения адаптаций

        Returns:
            float: Оценка эффективности (0.0 to 1.0)
        """
        effectiveness = 0.0

        # Фактор значимости опыта
        effectiveness += experience.significance_score * 0.3

        # Фактор успешности адаптаций
        effectiveness += adaptation_result.success_rate * 0.4

        # Фактор количества примененных адаптаций
        adaptation_ratio = adaptation_result.adaptations_applied / max(adaptation_result.adaptations_created, 1)
        effectiveness += adaptation_ratio * 0.3

        return min(max(effectiveness, 0.0), 1.0)

    async def _store_learning_result(self, result: LearningResult):
        """
        Сохранение результатов обучения в историю

        Args:
            result: Результат обучения
        """
        self.learning_history.append(result)

        # Ограничение размера истории (последние 1000 результатов)
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]

        # Сохранение в память агента, если доступна
        if self.memory_manager:
            try:
                await self.memory_manager.store_learning_result(result)
            except Exception as e:
                logger.warning(f"Failed to store learning result in memory: {e}")

    def _update_learning_metrics(self, result: LearningResult):
        """
        Обновление метрик обучения

        Args:
            result: Результат обучения
        """
        # Обновление основных метрик
        self.learning_metrics.total_experiences_processed += 1
        self.learning_metrics.patterns_discovered += result.patterns_extracted
        self.learning_metrics.cognitive_maps_updated += result.cognitive_updates
        self.learning_metrics.skills_improved += result.skills_developed

        # Обновление средней эффективности (экспоненциальное скользящее среднее)
        alpha = 0.1
        current_avg = self.learning_metrics.average_learning_effectiveness
        self.learning_metrics.average_learning_effectiveness = (
            alpha * result.learning_effectiveness + (1 - alpha) * current_avg
        )

        # Обновление рейтинга успеха адаптаций
        if result.adaptation_applied.adaptations_created > 0:
            adaptation_success = result.adaptation_applied.success_rate
            current_rate = self.learning_metrics.adaptation_success_rate
            self.learning_metrics.adaptation_success_rate = (
                alpha * adaptation_success + (1 - alpha) * current_rate
            )

        # Обновление статистики производительности
        self.performance_stats['total_learning_sessions'] += 1

        # Обновление среднего времени сессии
        session_time = result.learning_time
        current_avg_time = self.performance_stats['average_session_time']
        self.performance_stats['average_session_time'] = (
            alpha * session_time + (1 - alpha) * current_avg_time
        )

        # Добавление в тренд эффективности
        self.performance_stats['learning_effectiveness_trend'].append(result.learning_effectiveness)
        if len(self.performance_stats['learning_effectiveness_trend']) > 100:
            self.performance_stats['learning_effectiveness_trend'] = self.performance_stats['learning_effectiveness_trend'][-100:]

        # Обновление рейтинга обнаружения паттернов
        if result.experience_processed.significance_score > 0:
            pattern_rate = result.patterns_extracted / result.experience_processed.significance_score
            current_rate = self.performance_stats['pattern_discovery_rate']
            self.performance_stats['pattern_discovery_rate'] = (
                alpha * pattern_rate + (1 - alpha) * current_rate
            )

    async def get_learning_metrics(self, timeframe: str = "7d") -> LearningMetrics:
        """
        Получение метрик обучения за указанный период

        Args:
            timeframe: Период времени ("1d", "7d", "30d", "all")

        Returns:
            LearningMetrics: Метрики обучения
        """
        # Для простоты возвращаем текущие метрики
        # В продвинутой версии можно фильтровать по времени
        metrics = self.learning_metrics.copy()
        metrics.time_period = timeframe

        return metrics

    async def get_relevant_patterns(self, context: Dict[str, Any], limit: int = 10) -> List[Pattern]:
        """
        Получение релевантных паттернов для контекста

        Args:
            context: Контекст для поиска паттернов
            limit: Максимальное количество паттернов

        Returns:
            List[Pattern]: Релевантные паттерны
        """
        return await self.pattern_memory.retrieve_relevant_patterns(context, limit)

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Получение общей статистики обучения

        Returns:
            Dict[str, Any]: Статистика обучения
        """
        return {
            'learning_metrics': self.learning_metrics.model_dump(),
            'performance_stats': self.performance_stats.copy(),
            'pattern_memory_stats': self.pattern_memory.get_pattern_stats(),
            'experience_processor_stats': self.experience_processor.get_processing_stats(),
            'learning_history_size': len(self.learning_history)
        }

    async def optimize_learning_process(self) -> Dict[str, Any]:
        """
        Оптимизация процесса обучения на основе анализа истории

        Returns:
            Dict[str, Any]: Рекомендации по оптимизации
        """
        recommendations = []

        # Анализ эффективности обучения
        if self.learning_metrics.average_learning_effectiveness < 0.5:
            recommendations.append({
                'type': 'effectiveness_improvement',
                'description': 'Средняя эффективность обучения ниже порога',
                'suggestion': 'Улучшить алгоритмы извлечения паттернов и оценки значимости'
            })

        # Анализ времени обучения
        if self.performance_stats['average_session_time'] > 5.0:
            recommendations.append({
                'type': 'performance_optimization',
                'description': 'Среднее время обучения превышает норму',
                'suggestion': 'Оптимизировать обработку опыта и извлечение паттернов'
            })

        # Анализ паттернов
        pattern_stats = self.pattern_memory.get_pattern_stats()
        if pattern_stats['extraction_stats']['successful_extractions'] == 0:
            recommendations.append({
                'type': 'pattern_extraction_fix',
                'description': 'Нет успешных извлечений паттернов',
                'suggestion': 'Проверить работу PatternMemory и входные данные'
            })

        return {
            'recommendations': recommendations,
            'current_metrics': self.get_learning_stats(),
            'optimization_potential': len(recommendations) * 0.1  # Предполагаемый потенциал улучшения
        }

    async def reset_learning_state(self):
        """
        Сброс состояния обучения (для тестирования или перезагрузки)
        """
        self.learning_history.clear()
        self.learning_metrics = LearningMetrics(
            total_experiences_processed=0,
            average_learning_effectiveness=0,
            patterns_discovered=0,
            skills_improved=0,
            cognitive_maps_updated=0,
            adaptation_success_rate=0,
            time_period="all"
        )
        self.performance_stats = {
            'total_learning_sessions': 0,
            'average_session_time': 0,
            'learning_effectiveness_trend': [],
            'pattern_discovery_rate': 0
        }

        await self.pattern_memory.cleanup_old_patterns(max_age_days=0)  # Удалить все паттерны
        self.experience_processor.clear_cache()

        logger.info("Learning state reset completed")

"""
Обработчик опыта агента
Фаза 3: Обучение и Адаптация
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .models import (
    AgentExperience,
    ProcessedExperience,
    KeyElements
)

logger = logging.getLogger(__name__)


class ExperienceProcessor:
    """
    Обрабатывает сырой опыт агента для извлечения ключевых элементов,
    анализа значимости и подготовки к обучению
    """

    def __init__(self, hybrid_manager=None):
        """
        Инициализация обработчика опыта

        Args:
            hybrid_manager: Менеджер гибридных моделей для анализа
        """
        self.hybrid_manager = hybrid_manager
        self.experience_cache = {}
        self.processing_stats = {
            'total_processed': 0,
            'average_processing_time': 0,
            'cache_hit_rate': 0
        }

    async def process(self, raw_experience: AgentExperience) -> ProcessedExperience:
        """
        Обработка сырого опыта агента

        Args:
            raw_experience: Сырой опыт агента

        Returns:
            ProcessedExperience: Обработанный опыт с извлеченными элементами
        """
        start_time = time.time()

        # Проверка кэша
        cache_key = f"{raw_experience.id}_{hash(str(raw_experience))}"
        if cache_key in self.experience_cache:
            logger.debug(f"Cache hit for experience {raw_experience.id}")
            cached = self.experience_cache[cache_key]
            cached.processing_timestamp = datetime.now()
            return cached

        logger.info(f"Processing experience {raw_experience.id}")

        # Извлечение ключевых элементов
        key_elements = await self._extract_key_elements(raw_experience)

        # Анализ эмоционального контекста
        emotional_context = await self._analyze_emotional_context(raw_experience)

        # Оценка значимости опыта
        significance_score = await self._assess_significance(raw_experience, key_elements)

        # Категоризация опыта
        categories = await self._categorize_experience(raw_experience, key_elements)

        # Извлечение уроков
        lessons = await self._extract_lessons(raw_experience, emotional_context)

        processed = ProcessedExperience(
            original_experience=raw_experience,
            key_elements=key_elements,
            emotional_context=emotional_context,
            significance_score=significance_score,
            categories=categories,
            lessons=lessons,
            processing_timestamp=datetime.now()
        )

        # Кэширование для будущих ссылок
        self.experience_cache[cache_key] = processed

        # Обновление статистики
        processing_time = time.time() - start_time
        self._update_processing_stats(processing_time)

        logger.info(f"Experience {raw_experience.id} processed in {processing_time:.2f}s")
        return processed

    async def _extract_key_elements(self, experience: AgentExperience) -> KeyElements:
        """
        Извлечение ключевых элементов из опыта

        Args:
            experience: Опыт агента

        Returns:
            KeyElements: Ключевые элементы опыта
        """
        if self.hybrid_manager:
            # Использование AI для анализа опыта
            analysis_prompt = f"""
            Проанализируй этот опыт агента и выдели ключевые элементы:

            Запрос: {experience.query}
            Результат: {experience.result}
            Уверенность: {experience.confidence}
            Время выполнения: {experience.execution_time}

            Выдели:
            1. Ключевые решения и действия
            2. Критические факторы успеха/неудачи
            3. Важные наблюдения
            4. Потенциальные улучшения

            Формат: JSON с полями key_decisions, critical_factors, observations, improvements
            """

            try:
                response = await self.hybrid_manager.process_request({
                    'query': analysis_prompt,
                    'model_preference': 'local',
                    'temperature': 0.2
                })

                # Парсинг ответа AI
                return self._parse_key_elements(response)

            except Exception as e:
                logger.warning(f"AI analysis failed, using fallback: {e}")
                return self._extract_key_elements_fallback(experience)
        else:
            # Fallback метод без AI
            return self._extract_key_elements_fallback(experience)

    def _extract_key_elements_fallback(self, experience: AgentExperience) -> KeyElements:
        """
        Fallback метод извлечения ключевых элементов без AI

        Args:
            experience: Опыт агента

        Returns:
            KeyElements: Ключевые элементы опыта
        """
        key_decisions = []
        critical_factors = []
        observations = []
        improvements = []

        # Анализ на основе эвристик
        if experience.confidence > 0.8:
            key_decisions.append("Высокая уверенность в результате")
            observations.append("Успешное выполнение с высокой уверенностью")
        elif experience.confidence < 0.3:
            critical_factors.append({
                "factor": "низкая уверенность",
                "impact": "негативный",
                "description": "Результат получен с низкой уверенностью"
            })
            improvements.append("Улучшить механизм оценки уверенности")

        if experience.execution_time > 10.0:
            critical_factors.append({
                "factor": "длительное выполнение",
                "impact": "негативный",
                "description": f"Время выполнения {experience.execution_time:.2f}s превышает норму"
            })
            improvements.append("Оптимизировать время выполнения")

        if experience.success_indicators:
            for indicator in experience.success_indicators:
                observations.append(f"Успешный индикатор: {indicator}")

        if experience.error_indicators:
            for indicator in experience.error_indicators:
                critical_factors.append({
                    "factor": "ошибка",
                    "impact": "негативный",
                    "description": f"Обнаружена ошибка: {indicator}"
                })

        return KeyElements(
            key_decisions=key_decisions,
            critical_factors=critical_factors,
            observations=observations,
            improvements=improvements
        )

    def _parse_key_elements(self, ai_response: str) -> KeyElements:
        """
        Парсинг ответа AI в KeyElements

        Args:
            ai_response: Ответ от AI модели

        Returns:
            KeyElements: Разобранные ключевые элементы
        """
        try:
            # Предполагаем, что ответ в формате JSON
            data = json.loads(ai_response)

            return KeyElements(
                key_decisions=data.get('key_decisions', []),
                critical_factors=data.get('critical_factors', []),
                observations=data.get('observations', []),
                improvements=data.get('improvements', [])
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse AI response: {e}")
            # Возврат пустых элементов в случае ошибки парсинга
            return KeyElements(
                key_decisions=[],
                critical_factors=[],
                observations=[],
                improvements=[]
            )

    async def _analyze_emotional_context(self, experience: AgentExperience) -> Optional[Dict[str, Any]]:
        """
        Анализ эмоционального контекста опыта

        Args:
            experience: Опыт агента

        Returns:
            Optional[Dict[str, Any]]: Эмоциональный контекст или None
        """
        # Для базовой реализации возвращаем None
        # В продвинутой версии можно анализировать тональность,
        # эмоциональные маркеры в запросе и ответе
        return None

    async def _assess_significance(self, experience: AgentExperience, key_elements: KeyElements) -> float:
        """
        Оценка значимости опыта для обучения

        Args:
            experience: Опыт агента
            key_elements: Ключевые элементы опыта

        Returns:
            float: Оценка значимости (0.0 to 1.0)
        """
        significance_factors = {
            'uniqueness': self._calculate_uniqueness(experience),
            'impact': self._calculate_impact(experience),
            'learning_potential': self._calculate_learning_potential(key_elements),
            'rarity': self._calculate_rarity(experience)
        }

        # Взвешенная сумма
        weights = {'uniqueness': 0.3, 'impact': 0.3, 'learning_potential': 0.3, 'rarity': 0.1}

        significance = sum(significance_factors[factor] * weights[factor]
                          for factor in significance_factors)

        return min(max(significance, 0.0), 1.0)

    def _calculate_uniqueness(self, experience: AgentExperience) -> float:
        """
        Расчет уникальности опыта

        Args:
            experience: Опыт агента

        Returns:
            float: Оценка уникальности (0.0 to 1.0)
        """
        # Простая эвристика: более низкая уверенность = более уникальный опыт
        uniqueness = 1.0 - experience.confidence

        # Корректировка на основе времени выполнения
        if experience.execution_time > 5.0:
            uniqueness += 0.2
        elif experience.execution_time < 1.0:
            uniqueness -= 0.1

        return min(max(uniqueness, 0.0), 1.0)

    def _calculate_impact(self, experience: AgentExperience) -> float:
        """
        Расчет воздействия опыта

        Args:
            experience: Опыт агента

        Returns:
            float: Оценка воздействия (0.0 to 1.0)
        """
        impact = 0.0

        # Высокая уверенность = высокий impact
        impact += experience.confidence * 0.5

        # Длительное выполнение может указывать на сложность
        if experience.execution_time > 2.0:
            impact += 0.3

        # Наличие индикаторов успеха/ошибок
        if experience.success_indicators:
            impact += len(experience.success_indicators) * 0.1
        if experience.error_indicators:
            impact += len(experience.error_indicators) * 0.1

        return min(impact, 1.0)

    def _calculate_learning_potential(self, key_elements: KeyElements) -> float:
        """
        Расчет потенциала обучения

        Args:
            key_elements: Ключевые элементы опыта

        Returns:
            float: Оценка потенциала обучения (0.0 to 1.0)
        """
        potential = 0.0

        # Наличие улучшений указывает на потенциал обучения
        potential += len(key_elements.improvements) * 0.2

        # Критические факторы тоже важны
        potential += len(key_elements.critical_factors) * 0.15

        # Наблюдения добавляют контекст
        potential += len(key_elements.observations) * 0.1

        return min(potential, 1.0)

    def _calculate_rarity(self, experience: AgentExperience) -> float:
        """
        Расчет редкости опыта

        Args:
            experience: Опыт агента

        Returns:
            float: Оценка редкости (0.0 to 1.0)
        """
        # Для базовой реализации используем простую логику
        rarity = 0.5  # Средняя редкость по умолчанию

        # Очень низкая или очень высокая уверенность = более редкий опыт
        if experience.confidence < 0.2 or experience.confidence > 0.95:
            rarity += 0.3

        # Экстремальное время выполнения
        if experience.execution_time > 15.0 or experience.execution_time < 0.5:
            rarity += 0.2

        return min(rarity, 1.0)

    async def _categorize_experience(self, experience: AgentExperience, key_elements: KeyElements) -> List[str]:
        """
        Категоризация опыта

        Args:
            experience: Опыт агента
            key_elements: Ключевые элементы опыта

        Returns:
            List[str]: Категории опыта
        """
        categories = []

        # Категоризация на основе уверенности
        if experience.confidence > 0.8:
            categories.append("high_confidence")
        elif experience.confidence < 0.4:
            categories.append("low_confidence")

        # Категоризация на основе времени выполнения
        if experience.execution_time > 5.0:
            categories.append("time_consuming")
        elif experience.execution_time < 1.0:
            categories.append("fast_execution")

        # Категоризация на основе ключевых элементов
        if key_elements.improvements:
            categories.append("has_improvements")
        if key_elements.critical_factors:
            categories.append("has_critical_factors")

        # Категоризация на основе индикаторов
        if experience.success_indicators:
            categories.append("successful")
        if experience.error_indicators:
            categories.append("has_errors")

        return categories

    async def _extract_lessons(self, experience: AgentExperience, emotional_context: Optional[Dict[str, Any]]) -> List[str]:
        """
        Извлечение уроков из опыта

        Args:
            experience: Опыт агента
            emotional_context: Эмоциональный контекст

        Returns:
            List[str]: Извлеченные уроки
        """
        lessons = []

        # Уроки на основе уверенности
        if experience.confidence > 0.9:
            lessons.append("Высокая уверенность часто коррелирует с качественными результатами")
        elif experience.confidence < 0.3:
            lessons.append("Низкая уверенность может указывать на необходимость дополнительных проверок")

        # Уроки на основе времени выполнения
        if experience.execution_time > 10.0:
            lessons.append("Длительное выполнение может требовать оптимизации или разбиения на подзадачи")

        # Уроки на основе ошибок
        if experience.error_indicators:
            for error in experience.error_indicators:
                lessons.append(f"Избегать паттерна, приводящего к ошибке: {error}")

        # Уроки на основе успехов
        if experience.success_indicators:
            for success in experience.success_indicators:
                lessons.append(f"Повторять успешный паттерн: {success}")

        return lessons

    def _update_processing_stats(self, processing_time: float):
        """
        Обновление статистики обработки

        Args:
            processing_time: Время обработки опыта
        """
        self.processing_stats['total_processed'] += 1

        # Экспоненциальное скользящее среднее для времени обработки
        alpha = 0.1
        current_avg = self.processing_stats['average_processing_time']
        self.processing_stats['average_processing_time'] = (
            alpha * processing_time + (1 - alpha) * current_avg
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Получение статистики обработки

        Returns:
            Dict[str, Any]: Статистика обработки
        """
        return self.processing_stats.copy()

    def clear_cache(self):
        """Очистка кэша обработанных опытов"""
        self.experience_cache.clear()
        logger.info("Experience processing cache cleared")

"""
Память паттернов агента
Фаза 3: Обучение и Адаптация
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging

from .models import (
    Pattern,
    PatternType,
    SuccessPattern,
    ErrorPattern,
    EfficiencyPattern,
    ProcessedExperience
)

logger = logging.getLogger(__name__)


class PatternMemory:
    """
    Управляет памятью паттернов агента, извлекает паттерны из опыта
    и предоставляет релевантные паттерны для текущего контекста
    """

    def __init__(self, redis_client=None):
        """
        Инициализация памяти паттернов

        Args:
            redis_client: Redis клиент для персистентного хранения
        """
        self.redis = redis_client
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_strength: Dict[str, float] = {}  # Сила/частота паттернов

        # Статистика паттернов
        self.stats = {
            'total_patterns': 0,
            'patterns_by_type': {},
            'extraction_stats': {
                'total_extractions': 0,
                'successful_extractions': 0,
                'average_extraction_time': 0
            }
        }

        # Загрузка существующих паттернов из Redis
        if self.redis:
            self._load_patterns_from_redis()

    async def extract_patterns(self, experience: ProcessedExperience) -> List[Pattern]:
        """
        Извлечение паттернов из обработанного опыта

        Args:
            experience: Обработанный опыт агента

        Returns:
            List[Pattern]: Извлеченные паттерны
        """
        start_time = time.time()

        patterns = []

        try:
            # Паттерны успешных взаимодействий
            success_patterns = await self._extract_success_patterns(experience)
            patterns.extend(success_patterns)

            # Паттерны ошибок
            error_patterns = await self._extract_error_patterns(experience)
            patterns.extend(error_patterns)

            # Паттерны эффективности
            efficiency_patterns = await self._extract_efficiency_patterns(experience)
            patterns.extend(efficiency_patterns)

            # Паттерны пользовательского поведения
            user_patterns = await self._extract_user_patterns(experience)
            patterns.extend(user_patterns)

            # Укрепление существующих паттернов
            await self._reinforce_patterns(patterns)

            # Сохранение новых паттернов
            await self._store_patterns(patterns)

            # Обновление статистики
            extraction_time = time.time() - start_time
            self._update_extraction_stats(len(patterns), extraction_time)

            logger.info(f"Extracted {len(patterns)} patterns from experience {experience.original_experience.id}")

        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            self.stats['extraction_stats']['total_extractions'] += 1

        return patterns

    async def _extract_success_patterns(self, experience: ProcessedExperience) -> List[SuccessPattern]:
        """
        Извлечение паттернов успешных взаимодействий

        Args:
            experience: Обработанный опыт

        Returns:
            List[SuccessPattern]: Паттерны успеха
        """
        patterns = []

        if experience.original_experience.success_indicators:
            for indicator in experience.original_experience.success_indicators:
                # Создание уникального ID паттерна
                pattern_key = self._generate_pattern_key(indicator)
                existing_strength = self.pattern_strength.get(pattern_key, 1.0)

                pattern = SuccessPattern(
                    id=f"success_{hash(pattern_key)}",
                    type=PatternType.SUCCESS,
                    trigger_conditions=indicator.get('conditions', {}),
                    description=f"Success pattern: {indicator.get('description', 'Unknown success')}",
                    confidence=indicator.get('confidence', experience.original_experience.confidence),
                    frequency=existing_strength,
                    successful_actions=indicator.get('actions', []),
                    outcome=indicator.get('outcome', 'Positive result'),
                    examples=[indicator],
                    last_updated=datetime.now()
                )
                patterns.append(pattern)

        # Дополнительные паттерны на основе высокой уверенности
        if experience.original_experience.confidence > 0.8:
            confidence_pattern = SuccessPattern(
                id=f"high_confidence_{experience.original_experience.id}",
                type=PatternType.SUCCESS,
                trigger_conditions={
                    'confidence_threshold': 0.8,
                    'query_type': self._classify_query_type(experience.original_experience.query)
                },
                description="High confidence execution pattern",
                confidence=experience.original_experience.confidence,
                frequency=self.pattern_strength.get('high_confidence', 1.0),
                successful_actions=['execute_with_high_confidence'],
                outcome='Reliable result with high confidence',
                examples=[{
                    'query': experience.original_experience.query,
                    'confidence': experience.original_experience.confidence
                }],
                last_updated=datetime.now()
            )
            patterns.append(confidence_pattern)

        return patterns

    async def _extract_error_patterns(self, experience: ProcessedExperience) -> List[ErrorPattern]:
        """
        Извлечение паттернов ошибок

        Args:
            experience: Обработанный опыт

        Returns:
            List[ErrorPattern]: Паттерны ошибок
        """
        patterns = []

        if experience.original_experience.error_indicators:
            for indicator in experience.original_experience.error_indicators:
                pattern_key = self._generate_pattern_key(indicator)
                existing_strength = self.pattern_strength.get(pattern_key, 1.0)

                pattern = ErrorPattern(
                    id=f"error_{hash(pattern_key)}",
                    type=PatternType.ERROR,
                    trigger_conditions=indicator.get('conditions', {}),
                    description=f"Error pattern: {indicator.get('description', 'Unknown error')}",
                    confidence=1.0 - experience.original_experience.confidence,  # Обратная уверенность
                    frequency=existing_strength,
                    error_type=indicator.get('type', 'unknown_error'),
                    recovery_actions=indicator.get('recovery_actions', []),
                    prevention_measures=indicator.get('prevention', []),
                    examples=[indicator],
                    last_updated=datetime.now()
                )
                patterns.append(pattern)

        # Паттерны низкой уверенности
        if experience.original_experience.confidence < 0.4:
            low_confidence_pattern = ErrorPattern(
                id=f"low_confidence_{experience.original_experience.id}",
                type=PatternType.ERROR,
                trigger_conditions={
                    'confidence_threshold': 0.4,
                    'max_execution_time': experience.original_experience.execution_time
                },
                description="Low confidence execution pattern",
                confidence=1.0 - experience.original_experience.confidence,
                frequency=self.pattern_strength.get('low_confidence', 1.0),
                error_type='low_confidence_error',
                recovery_actions=['request_additional_verification', 'use_fallback_strategy'],
                prevention_measures=['implement_confidence_checks', 'add_validation_steps'],
                examples=[{
                    'query': experience.original_experience.query,
                    'confidence': experience.original_experience.confidence
                }],
                last_updated=datetime.now()
            )
            patterns.append(low_confidence_pattern)

        return patterns

    async def _extract_efficiency_patterns(self, experience: ProcessedExperience) -> List[EfficiencyPattern]:
        """
        Извлечение паттернов эффективности

        Args:
            experience: Обработанный опыт

        Returns:
            List[EfficiencyPattern]: Паттерны эффективности
        """
        patterns = []

        execution_time = experience.original_experience.execution_time

        # Паттерны быстрого выполнения
        if execution_time < 2.0:
            fast_execution_pattern = EfficiencyPattern(
                id=f"fast_execution_{experience.original_experience.id}",
                type=PatternType.EFFICIENCY,
                trigger_conditions={
                    'max_execution_time': 2.0,
                    'query_complexity': self._assess_query_complexity(experience.original_experience.query)
                },
                description="Fast execution pattern for simple queries",
                confidence=experience.original_experience.confidence,
                frequency=self.pattern_strength.get('fast_execution', 1.0),
                optimization_actions=['use_cached_results', 'simplify_processing'],
                performance_gain=1.0 - (execution_time / 5.0),  # Нормализация к 5 секундам
                resource_savings={'time': 5.0 - execution_time},
                examples=[{
                    'query': experience.original_experience.query,
                    'execution_time': execution_time
                }],
                last_updated=datetime.now()
            )
            patterns.append(fast_execution_pattern)

        # Паттерны оптимизации для длительных задач
        elif execution_time > 10.0:
            optimization_pattern = EfficiencyPattern(
                id=f"optimization_needed_{experience.original_experience.id}",
                type=PatternType.EFFICIENCY,
                trigger_conditions={
                    'min_execution_time': 10.0,
                    'query_type': self._classify_query_type(experience.original_experience.query)
                },
                description="Pattern indicating need for optimization",
                confidence=0.7,
                frequency=self.pattern_strength.get('needs_optimization', 1.0),
                optimization_actions=[
                    'implement_caching',
                    'optimize_algorithm',
                    'parallelize_operations',
                    'reduce_external_calls'
                ],
                performance_gain=0.3,  # Предполагаемая оптимизация на 30%
                resource_savings={'time': execution_time * 0.3},
                examples=[{
                    'query': experience.original_experience.query,
                    'execution_time': execution_time
                }],
                last_updated=datetime.now()
            )
            patterns.append(optimization_pattern)

        return patterns

    async def _extract_user_patterns(self, experience: ProcessedExperience) -> List[Pattern]:
        """
        Извлечение паттернов пользовательского поведения

        Args:
            experience: Обработанный опыт

        Returns:
            List[Pattern]: Паттерны пользовательского поведения
        """
        patterns = []

        # Анализ на основе feedback пользователя
        if experience.original_experience.user_feedback:
            feedback = experience.original_experience.user_feedback

            user_satisfaction_pattern = Pattern(
                id=f"user_satisfaction_{experience.original_experience.id}",
                type=PatternType.USER_BEHAVIOR,
                trigger_conditions={
                    'feedback_type': feedback.get('type'),
                    'satisfaction_level': feedback.get('satisfaction', 0.5)
                },
                description=f"User behavior pattern: {feedback.get('description', 'Unknown feedback')}",
                confidence=feedback.get('confidence', 0.5),
                frequency=self.pattern_strength.get('user_feedback', 1.0),
                examples=[feedback],
                last_updated=datetime.now()
            )
            patterns.append(user_satisfaction_pattern)

        return patterns

    def _generate_pattern_key(self, indicator: Dict[str, Any]) -> str:
        """
        Генерация ключа паттерна для идентификации

        Args:
            indicator: Индикатор для генерации ключа

        Returns:
            str: Уникальный ключ паттерна
        """
        # Создание детерминированного ключа на основе содержания
        content = json.dumps(indicator, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _classify_query_type(self, query: str) -> str:
        """
        Классификация типа запроса

        Args:
            query: Текст запроса

        Returns:
            str: Тип запроса
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ['analyze', 'анализ', 'analyze']):
            return 'analysis'
        elif any(word in query_lower for word in ['search', 'поиск', 'find']):
            return 'search'
        elif any(word in query_lower for word in ['create', 'создать', 'generate']):
            return 'creation'
        elif any(word in query_lower for word in ['calculate', 'вычислить', 'compute']):
            return 'calculation'
        else:
            return 'general'

    def _assess_query_complexity(self, query: str) -> str:
        """
        Оценка сложности запроса

        Args:
            query: Текст запроса

        Returns:
            str: Уровень сложности
        """
        word_count = len(query.split())

        if word_count < 5:
            return 'simple'
        elif word_count < 15:
            return 'medium'
        else:
            return 'complex'

    async def retrieve_relevant_patterns(self, context: Dict[str, Any], limit: int = 10) -> List[Pattern]:
        """
        Получение релевантных паттернов для контекста

        Args:
            context: Контекст для поиска паттернов
            limit: Максимальное количество паттернов

        Returns:
            List[Pattern]: Релевантные паттерны
        """
        relevant_patterns = []

        for pattern_id, pattern in self.patterns.items():
            relevance_score = await self._calculate_pattern_relevance(pattern, context)
            if relevance_score > 0.3:  # Минимальный порог релевантности
                pattern.relevance_score = relevance_score
                relevant_patterns.append(pattern)

        # Сортировка по релевантности и силе
        relevant_patterns.sort(
            key=lambda p: (p.relevance_score or 0) * p.frequency,
            reverse=True
        )

        return relevant_patterns[:limit]

    async def _calculate_pattern_relevance(self, pattern: Pattern, context: Dict[str, Any]) -> float:
        """
        Расчет релевантности паттерна для контекста

        Args:
            pattern: Паттерн для оценки
            context: Контекст

        Returns:
            float: Оценка релевантности (0.0 to 1.0)
        """
        relevance = 0.0

        # Проверка совпадения условий паттерна с контекстом
        trigger_conditions = pattern.trigger_conditions

        for key, value in trigger_conditions.items():
            if key in context:
                context_value = context[key]

                # Простое сравнение значений
                if isinstance(value, (int, float)) and isinstance(context_value, (int, float)):
                    # Для числовых значений - проверка диапазона
                    if key.endswith('_threshold') or key.endswith('_time'):
                        # Для порогов - проверка превышения
                        if context_value >= value:
                            relevance += 0.3
                    else:
                        # Для обычных значений - близость
                        diff = abs(value - context_value)
                        similarity = max(0, 1.0 - diff / max(abs(value), abs(context_value), 1))
                        relevance += similarity * 0.2
                elif value == context_value:
                    relevance += 0.4

        # Корректировка на основе типа паттерна
        type_multipliers = {
            PatternType.SUCCESS: 1.0,
            PatternType.ERROR: 0.8,
            PatternType.EFFICIENCY: 1.2,
            PatternType.USER_BEHAVIOR: 1.1,
            PatternType.PERFORMANCE: 1.0
        }

        relevance *= type_multipliers.get(pattern.type, 1.0)

        # Корректировка на основе силы паттерна
        relevance *= min(pattern.frequency / 5.0, 1.0)  # Нормализация к 5 повторениям

        return min(relevance, 1.0)

    async def _reinforce_patterns(self, new_patterns: List[Pattern]):
        """
        Укрепление паттернов на основе повторений

        Args:
            new_patterns: Новые паттерны для укрепления
        """
        for pattern in new_patterns:
            pattern_key = str(pattern.trigger_conditions)

            if pattern_key in self.pattern_strength:
                # Экспоненциальное укрепление
                self.pattern_strength[pattern_key] *= 1.1
            else:
                self.pattern_strength[pattern_key] = 1.0

            # Ограничение максимальной силы
            self.pattern_strength[pattern_key] = min(self.pattern_strength[pattern_key], 10.0)

            # Сохранение в Redis для персистентности
            if self.redis:
                await self.redis.hset("pattern_strength", pattern_key, self.pattern_strength[pattern_key])

    async def _store_patterns(self, patterns: List[Pattern]):
        """
        Сохранение паттернов в памяти и Redis

        Args:
            patterns: Паттерны для сохранения
        """
        for pattern in patterns:
            self.patterns[pattern.id] = pattern
            self.stats['total_patterns'] = len(self.patterns)

            # Обновление статистики по типам
            type_key = pattern.type.value
            if type_key not in self.stats['patterns_by_type']:
                self.stats['patterns_by_type'][type_key] = 0
            self.stats['patterns_by_type'][type_key] += 1

            # Сохранение в Redis
            if self.redis:
                pattern_data = pattern.model_dump()
                await self.redis.hset("patterns", pattern.id, json.dumps(pattern_data))

    async def _load_patterns_from_redis(self):
        """
        Загрузка паттернов из Redis при инициализации
        """
        try:
            if not self.redis:
                return

            # Загрузка паттернов
            patterns_data = await self.redis.hgetall("patterns")
            for pattern_id, pattern_json in patterns_data.items():
                try:
                    pattern_data = json.loads(pattern_json)
                    # Восстановление типа паттерна
                    pattern_type = pattern_data.get('type')
                    if pattern_type == 'success':
                        pattern = SuccessPattern(**pattern_data)
                    elif pattern_type == 'error':
                        pattern = ErrorPattern(**pattern_data)
                    elif pattern_type == 'efficiency':
                        pattern = EfficiencyPattern(**pattern_data)
                    else:
                        pattern = Pattern(**pattern_data)

                    self.patterns[pattern_id] = pattern
                except Exception as e:
                    logger.warning(f"Failed to load pattern {pattern_id}: {e}")

            # Загрузка силы паттернов
            strength_data = await self.redis.hgetall("pattern_strength")
            for key, value in strength_data.items():
                try:
                    self.pattern_strength[key] = float(value)
                except ValueError:
                    self.pattern_strength[key] = 1.0

            logger.info(f"Loaded {len(self.patterns)} patterns from Redis")

        except Exception as e:
            logger.error(f"Failed to load patterns from Redis: {e}")

    def _update_extraction_stats(self, patterns_count: int, extraction_time: float):
        """
        Обновление статистики извлечения паттернов

        Args:
            patterns_count: Количество извлеченных паттернов
            extraction_time: Время извлечения
        """
        stats = self.stats['extraction_stats']
        stats['total_extractions'] += 1

        if patterns_count > 0:
            stats['successful_extractions'] += 1

        # Экспоненциальное скользящее среднее
        alpha = 0.1
        current_avg = stats['average_extraction_time']
        stats['average_extraction_time'] = alpha * extraction_time + (1 - current_avg) * (1 - alpha)

    def get_pattern_stats(self) -> Dict[str, Any]:
        """
        Получение статистики паттернов

        Returns:
            Dict[str, Any]: Статистика паттернов
        """
        return {
            **self.stats,
            'active_patterns': len(self.patterns),
            'strongest_patterns': sorted(
                self.pattern_strength.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    async def cleanup_old_patterns(self, max_age_days: int = 30):
        """
        Очистка устаревших паттернов

        Args:
            max_age_days: Максимальный возраст паттернов в днях
        """
        current_time = datetime.now()
        patterns_to_remove = []

        for pattern_id, pattern in self.patterns.items():
            age_days = (current_time - pattern.last_updated).days
            if age_days > max_age_days and pattern.frequency < 2.0:
                patterns_to_remove.append(pattern_id)

        for pattern_id in patterns_to_remove:
            del self.patterns[pattern_id]
            # Также удалить из Redis
            if self.redis:
                await self.redis.hdel("patterns", pattern_id)

        logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")

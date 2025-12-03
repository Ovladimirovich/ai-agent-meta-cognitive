"""
Система развития навыков агента
Фаза 3: Обучение и Адаптация
"""

import json
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging

from .models import (
    SkillInfo,
    SkillUpdate,
    SkillAssessment,
    SkillCategory,
    Pattern,
    ProcessedExperience
)

logger = logging.getLogger(__name__)


class SkillTree:
    """
    Дерево навыков агента с зависимостями и прогрессией
    """

    def __init__(self):
        """
        Инициализация дерева навыков
        """
        self.skills: Dict[str, SkillInfo] = {}
        self.skill_dependencies: Dict[str, List[str]] = {}
        self.skill_progression: Dict[str, List[str]] = {}  # Следующие уровни навыков

        # Инициализация базовых навыков
        self._initialize_base_skills()

    def _initialize_base_skills(self):
        """
        Инициализация базовых навыков агента
        """
        base_skills = {
            # Навыки анализа запросов
            "query_analysis": {
                "category": SkillCategory.QUERY_ANALYSIS,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": [],
                "description": "Анализ и понимание запросов пользователей"
            },
            "intent_recognition": {
                "category": SkillCategory.QUERY_ANALYSIS,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["query_analysis"],
                "description": "Распознавание намерений в запросах"
            },
            "context_understanding": {
                "category": SkillCategory.QUERY_ANALYSIS,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["intent_recognition"],
                "description": "Понимание контекста запросов"
            },

            # Навыки оркестрации инструментов
            "tool_selection": {
                "category": SkillCategory.TOOL_ORCHESTRATION,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": [],
                "description": "Выбор подходящих инструментов для задач"
            },
            "workflow_optimization": {
                "category": SkillCategory.TOOL_ORCHESTRATION,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["tool_selection"],
                "description": "Оптимизация рабочих процессов"
            },
            "parallel_processing": {
                "category": SkillCategory.TOOL_ORCHESTRATION,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["workflow_optimization"],
                "description": "Параллельная обработка задач"
            },

            # Навыки обработки ошибок
            "error_detection": {
                "category": SkillCategory.ERROR_HANDLING,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": [],
                "description": "Обнаружение ошибок в работе"
            },
            "error_recovery": {
                "category": SkillCategory.ERROR_HANDLING,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["error_detection"],
                "description": "Восстановление после ошибок"
            },
            "robustness": {
                "category": SkillCategory.ERROR_HANDLING,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["error_recovery"],
                "description": "Повышение устойчивости к ошибкам"
            },

            # Навыки оптимизации производительности
            "efficiency": {
                "category": SkillCategory.PERFORMANCE_OPTIMIZATION,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": [],
                "description": "Повышение эффективности работы"
            },
            "resource_management": {
                "category": SkillCategory.PERFORMANCE_OPTIMIZATION,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["efficiency"],
                "description": "Управление ресурсами"
            },
            "adaptive_scaling": {
                "category": SkillCategory.PERFORMANCE_OPTIMIZATION,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["resource_management"],
                "description": "Адаптивное масштабирование"
            },

            # Навыки взаимодействия с пользователями
            "communication": {
                "category": SkillCategory.USER_INTERACTION,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": [],
                "description": "Коммуникация с пользователями"
            },
            "personalization": {
                "category": SkillCategory.USER_INTERACTION,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["communication"],
                "description": "Персонализация взаимодействия"
            },
            "feedback_processing": {
                "category": SkillCategory.USER_INTERACTION,
                "level": 0.1,
                "experience_points": 0,
                "dependencies": ["personalization"],
                "description": "Обработка обратной связи"
            }
        }

        for skill_name, skill_data in base_skills.items():
            skill_info = SkillInfo(
                name=skill_name,
                category=skill_data["category"],
                level=skill_data["level"],
                experience_points=skill_data["experience_points"],
                dependencies=skill_data["dependencies"],
                proficiency_trend=[skill_data["level"]]
            )
            self.skills[skill_name] = skill_info

            # Устанавливаем зависимости
            self.skill_dependencies[skill_name] = skill_data["dependencies"]

    def get_skill(self, skill_name: str) -> Optional[SkillInfo]:
        """
        Получение информации о навыке

        Args:
            skill_name: Название навыка

        Returns:
            Optional[SkillInfo]: Информация о навыке или None
        """
        return self.skills.get(skill_name)

    def get_available_skills(self) -> List[str]:
        """
        Получение списка доступных навыков

        Returns:
            List[str]: Список названий навыков
        """
        return list(self.skills.keys())

    def can_upgrade_skill(self, skill_name: str) -> bool:
        """
        Проверка возможности улучшения навыка

        Args:
            skill_name: Название навыка

        Returns:
            bool: True если навык можно улучшить
        """
        if skill_name not in self.skills:
            return False

        skill = self.skills[skill_name]

        # Проверяем зависимости
        for dependency in skill.dependencies:
            if dependency not in self.skills or self.skills[dependency].level < 0.5:
                return False

        # Проверяем, что навык не на максимальном уровне
        return skill.level < 1.0

    def get_skill_progression_path(self, skill_name: str) -> List[str]:
        """
        Получение пути прогрессии навыка

        Args:
            skill_name: Название навыка

        Returns:
            List[str]: Список навыков в цепочке прогрессии
        """
        if skill_name not in self.skill_dependencies:
            return [skill_name]

        # Рекурсивно собираем цепочку зависимостей
        path = []
        visited = set()

        def collect_dependencies(current_skill: str):
            if current_skill in visited:
                return
            visited.add(current_skill)

            # Добавляем зависимости сначала
            for dep in self.skill_dependencies.get(current_skill, []):
                collect_dependencies(dep)

            path.append(current_skill)

        collect_dependencies(skill_name)
        return path


class SkillDevelopment:
    """
    Система развития навыков агента
    """

    def __init__(self):
        """
        Инициализация системы развития навыков
        """
        self.skill_tree = SkillTree()
        self.skill_assessments: Dict[str, SkillAssessment] = {}
        self.development_history: List[SkillUpdate] = []

        # Статистика развития навыков
        self.stats = {
            'total_skill_updates': 0,
            'skills_improved': 0,
            'average_skill_level': 0.1,
            'skill_categories': {}
        }

        logger.info("SkillDevelopment system initialized")

    async def develop_skills(self, patterns: List[Pattern], cognitive_updates: List[Dict[str, Any]]) -> List[SkillUpdate]:
        """
        Развитие навыков на основе паттернов и когнитивных обновлений

        Args:
            patterns: Извлеченные паттерны
            cognitive_updates: Обновления когнитивных карт

        Returns:
            List[SkillUpdate]: Список обновлений навыков
        """
        skill_updates = []

        # Определяем навыки для развития
        skill_candidates = await self._identify_skill_candidates(patterns)

        for skill_name in skill_candidates:
            if self.skill_tree.can_upgrade_skill(skill_name):
                # Оцениваем прогресс в навыке
                progress = await self._assess_skill_progress(skill_name, patterns, cognitive_updates)

                if progress > 0.05:  # Минимальный порог прогресса
                    current_skill = self.skill_tree.skills[skill_name]
                    previous_level = current_skill.level
                    new_level = min(previous_level + progress, 1.0)

                    # Обновляем навык
                    current_skill.level = new_level
                    current_skill.experience_points += int(progress * 100)
                    current_skill.last_practiced = datetime.now()
                    current_skill.proficiency_trend.append(new_level)

                    # Ограничиваем историю тренда
                    if len(current_skill.proficiency_trend) > 20:
                        current_skill.proficiency_trend = current_skill.proficiency_trend[-20:]

                    # Создаем обновление навыка
                    skill_update = SkillUpdate(
                        skill_name=skill_name,
                        previous_level=previous_level,
                        new_level=new_level,
                        progress=progress,
                        confidence=self._calculate_skill_confidence(skill_name, patterns),
                        improvements=await self._generate_skill_improvements(skill_name, new_level)
                    )

                    skill_updates.append(skill_update)
                    self.development_history.append(skill_update)

                    # Обновляем статистику
                    self._update_skill_stats(skill_update)

                    logger.debug(f"Skill '{skill_name}' improved: {previous_level:.2f} -> {new_level:.2f}")

        return skill_updates

    async def _identify_skill_candidates(self, patterns: List[Pattern]) -> Set[str]:
        """
        Определение навыков, которые можно развить на основе паттернов

        Args:
            patterns: Извлеченные паттерны

        Returns:
            Set[str]: Множество названий навыков-кандидатов
        """
        candidates = set()

        skill_mappings = {
            'query_analysis': ['query_analysis', 'intent_recognition', 'context_understanding'],
            'tool_orchestration': ['tool_selection', 'workflow_optimization', 'parallel_processing'],
            'error_handling': ['error_detection', 'error_recovery', 'robustness'],
            'performance_optimization': ['efficiency', 'resource_management', 'adaptive_scaling'],
            'user_interaction': ['communication', 'personalization', 'feedback_processing']
        }

        for pattern in patterns:
            # Определяем категорию навыка на основе типа паттерна
            pattern_type = pattern.type.value

            if pattern_type in ['success', 'efficiency']:
                # Успешные паттерны улучшают соответствующие навыки
                if 'query' in pattern.trigger_conditions.get('query_type', '').lower():
                    candidates.update(skill_mappings['query_analysis'])
                elif 'tool' in str(pattern.trigger_conditions).lower():
                    candidates.update(skill_mappings['tool_orchestration'])
                elif pattern.trigger_conditions.get('execution_time', 10) < 2.0:
                    candidates.update(skill_mappings['performance_optimization'])

            elif pattern_type == 'error':
                # Паттерны ошибок улучшают навыки обработки ошибок
                candidates.update(skill_mappings['error_handling'])

            elif pattern_type == 'user_behavior':
                # Паттерны поведения пользователя улучшают навыки взаимодействия
                candidates.update(skill_mappings['user_interaction'])

        return candidates

    async def _assess_skill_progress(self, skill_name: str, patterns: List[Pattern], cognitive_updates: List[Dict[str, Any]]) -> float:
        """
        Оценка прогресса в навыке

        Args:
            skill_name: Название навыка
            patterns: Извлеченные паттерны
            cognitive_updates: Обновления когнитивных карт

        Returns:
            float: Оценка прогресса (0.0 to 1.0)
        """
        progress_indicators = []

        # Анализ паттернов
        pattern_progress = await self._calculate_pattern_based_progress(skill_name, patterns)
        progress_indicators.append(pattern_progress)

        # Анализ когнитивных обновлений
        cognitive_progress = await self._calculate_cognitive_based_progress(skill_name, cognitive_updates)
        progress_indicators.append(cognitive_progress)

        # Анализ практического применения
        application_progress = await self._calculate_application_based_progress(skill_name)
        progress_indicators.append(application_progress)

        # Взвешенное среднее
        if progress_indicators:
            return sum(progress_indicators) / len(progress_indicators)
        return 0.0

    async def _calculate_pattern_based_progress(self, skill_name: str, patterns: List[Pattern]) -> float:
        """
        Расчет прогресса на основе паттернов

        Args:
            skill_name: Название навыка
            patterns: Паттерны

        Returns:
            float: Прогресс на основе паттернов
        """
        relevant_patterns = 0
        total_relevance = 0.0

        skill_category_mapping = {
            'query_analysis': ['query_analysis', 'intent_recognition', 'context_understanding'],
            'tool_orchestration': ['tool_selection', 'workflow_optimization', 'parallel_processing'],
            'error_handling': ['error_detection', 'error_recovery', 'robustness'],
            'performance_optimization': ['efficiency', 'resource_management', 'adaptive_scaling'],
            'user_interaction': ['communication', 'personalization', 'feedback_processing']
        }

        # Находим категорию навыка
        skill_category = None
        for category, skills in skill_category_mapping.items():
            if skill_name in skills:
                skill_category = category
                break

        if not skill_category:
            return 0.0

        for pattern in patterns:
            relevance = 0.0

            # Оцениваем релевантность паттерна для категории навыка
            if skill_category == 'query_analysis' and 'query' in str(pattern.trigger_conditions).lower():
                relevance = 0.8
            elif skill_category == 'tool_orchestration' and 'tool' in str(pattern.trigger_conditions).lower():
                relevance = 0.8
            elif skill_category == 'error_handling' and pattern.type.value == 'error':
                relevance = 0.9
            elif skill_category == 'performance_optimization' and pattern.type.value == 'efficiency':
                relevance = 0.8
            elif skill_category == 'user_interaction' and pattern.type.value == 'user_behavior':
                relevance = 0.9

            if relevance > 0:
                relevant_patterns += 1
                total_relevance += relevance * pattern.frequency

        if relevant_patterns > 0:
            return min(total_relevance / relevant_patterns * 0.1, 0.3)  # Максимум 30% прогресса от паттернов
        return 0.0

    async def _calculate_cognitive_based_progress(self, skill_name: str, cognitive_updates: List[Dict[str, Any]]) -> float:
        """
        Расчет прогресса на основе когнитивных обновлений

        Args:
            skill_name: Название навыка
            cognitive_updates: Обновления когнитивных карт

        Returns:
            float: Прогресс на основе когнитивных обновлений
        """
        progress = 0.0

        for update in cognitive_updates:
            map_name = update.get('map_name', '')

            # Разные карты дают разный вклад в развитие навыков
            if map_name == 'domain_knowledge':
                progress += 0.05  # Знания домена улучшают навыки
            elif map_name == 'user_preferences':
                if skill_name in ['communication', 'personalization', 'feedback_processing']:
                    progress += 0.08  # Предпочтения пользователей улучшают навыки взаимодействия
            elif map_name == 'task_strategies':
                if skill_name in ['tool_selection', 'workflow_optimization']:
                    progress += 0.07  # Стратегии задач улучшают навыки оркестрации

        return min(progress, 0.2)  # Максимум 20% прогресса от когнитивных обновлений

    async def _calculate_application_based_progress(self, skill_name: str) -> float:
        """
        Расчет прогресса на основе практического применения

        Args:
            skill_name: Название навыка

        Returns:
            float: Прогресс на основе применения
        """
        skill = self.skill_tree.get_skill(skill_name)
        if not skill:
            return 0.0

        # Простая модель: навыки растут медленнее со временем
        base_progress = 0.02  # Базовый рост

        # Модификаторы на основе текущего уровня
        if skill.level < 0.3:
            base_progress *= 1.5  # Быстрый рост на начальных уровнях
        elif skill.level > 0.8:
            base_progress *= 0.5  # Медленный рост на продвинутых уровнях

        # Модификатор на основе недавней практики
        if skill.last_practiced:
            hours_since_practice = (datetime.now() - skill.last_practiced).total_seconds() / 3600
            if hours_since_practice < 24:  # Практика в последние 24 часа
                base_progress *= 1.2

        return min(base_progress, 0.1)  # Максимум 10% прогресса от применения

    def _calculate_skill_confidence(self, skill_name: str, patterns: List[Pattern]) -> float:
        """
        Расчет уверенности в развитии навыка

        Args:
            skill_name: Название навыка
            patterns: Паттерны

        Returns:
            float: Уверенность в развитии навыка
        """
        skill = self.skill_tree.get_skill(skill_name)
        if not skill:
            return 0.5

        # Уверенность растет с уровнем навыка и количеством паттернов
        confidence = skill.level * 0.7  # 70% от уровня навыка

        # Бонус за релевантные паттерны
        relevant_patterns = len([p for p in patterns if self._is_pattern_relevant_to_skill(p, skill_name)])
        confidence += min(relevant_patterns * 0.05, 0.2)  # До 20% бонуса

        return min(confidence, 1.0)

    def _is_pattern_relevant_to_skill(self, pattern: Pattern, skill_name: str) -> bool:
        """
        Проверка релевантности паттерна для навыка

        Args:
            pattern: Паттерн
            skill_name: Название навыка

        Returns:
            bool: True если паттерн релевантен для навыка
        """
        # Простая логика релевантности
        skill_keywords = {
            'query_analysis': ['query', 'analyze', 'understand'],
            'tool_selection': ['tool', 'select', 'choose'],
            'error_recovery': ['error', 'fail', 'recovery'],
            'efficiency': ['fast', 'efficient', 'optimize'],
            'communication': ['user', 'feedback', 'response']
        }

        keywords = skill_keywords.get(skill_name, [])
        pattern_text = str(pattern.trigger_conditions).lower() + str(pattern.description).lower()

        return any(keyword in pattern_text for keyword in keywords)

    async def _generate_skill_improvements(self, skill_name: str, new_level: float) -> List[str]:
        """
        Генерация описаний улучшений навыка

        Args:
            skill_name: Название навыка
            new_level: Новый уровень навыка

        Returns:
            List[str]: Список улучшений
        """
        improvements = []

        if new_level < 0.3:
            improvements.append(f"Базовое развитие навыка {skill_name}")
        elif new_level < 0.6:
            improvements.append(f"Улучшенное применение навыка {skill_name}")
        elif new_level < 0.9:
            improvements.append(f"Продвинутое мастерство в навыке {skill_name}")
        else:
            improvements.append(f"Экспертный уровень навыка {skill_name}")

        # Добавляем специфические улучшения
        skill_specific_improvements = {
            'query_analysis': ["Лучшее понимание запросов", "Более точный анализ намерений"],
            'tool_selection': ["Оптимальный выбор инструментов", "Эффективная оркестрация"],
            'error_recovery': ["Быстрое обнаружение ошибок", "Надежное восстановление"],
            'efficiency': ["Повышенная производительность", "Оптимизация ресурсов"],
            'communication': ["Лучшее взаимодействие", "Персонализированные ответы"]
        }

        if skill_name in skill_specific_improvements:
            improvements.extend(skill_specific_improvements[skill_name])

        return improvements

    def _update_skill_stats(self, skill_update: SkillUpdate):
        """
        Обновление статистики развития навыков

        Args:
            skill_update: Обновление навыка
        """
        self.stats['total_skill_updates'] += 1
        self.stats['skills_improved'] = len(set([u.skill_name for u in self.development_history]))

        # Пересчитываем средний уровень навыков
        total_level = sum(skill.level for skill in self.skill_tree.skills.values())
        self.stats['average_skill_level'] = total_level / len(self.skill_tree.skills)

        # Обновляем статистику по категориям
        skill = self.skill_tree.get_skill(skill_update.skill_name)
        if skill:
            category = skill.category.value
            if category not in self.stats['skill_categories']:
                self.stats['skill_categories'][category] = {'count': 0, 'avg_level': 0}

            category_stats = self.stats['skill_categories'][category]
            category_stats['count'] += 1
            # Пересчитываем средний уровень категории
            category_skills = [s for s in self.skill_tree.skills.values() if s.category.value == category]
            category_stats['avg_level'] = sum(s.level for s in category_skills) / len(category_skills)

    async def assess_skill(self, skill_name: str, context: Dict[str, Any]) -> SkillAssessment:
        """
        Оценка уровня навыка в конкретном контексте

        Args:
            skill_name: Название навыка
            context: Контекст для оценки

        Returns:
            SkillAssessment: Оценка навыка
        """
        skill = self.skill_tree.get_skill(skill_name)
        if not skill:
            return SkillAssessment(
                skill_name=skill_name,
                current_level=0.0,
                context_performance=0.0,
                recommendations=["Навык не найден"],
                next_practice_suggestion="Изучить основы навыка"
            )

        # Оцениваем производительность в контексте
        context_performance = await self._evaluate_context_performance(skill_name, context)

        # Генерируем рекомендации
        recommendations = await self._generate_skill_recommendations(skill_name, skill.level, context_performance)

        # Предлагаем следующую практику
        next_practice = await self._suggest_next_practice(skill_name, skill.level, context_performance)

        assessment = SkillAssessment(
            skill_name=skill_name,
            current_level=skill.level,
            context_performance=context_performance,
            recommendations=recommendations,
            next_practice_suggestion=next_practice
        )

        self.skill_assessments[skill_name] = assessment
        return assessment

    async def _evaluate_context_performance(self, skill_name: str, context: Dict[str, Any]) -> float:
        """
        Оценка производительности навыка в контексте

        Args:
            skill_name: Название навыка
            context: Контекст

        Returns:
            float: Оценка производительности
        """
        # Простая оценка на основе контекста
        performance = 0.5  # Базовая производительность

        # Модификаторы на основе контекста
        if context.get('complexity') == 'high':
            if skill_name in ['query_analysis', 'context_understanding']:
                performance += 0.2  # Эти навыки важны для сложных задач
        elif context.get('complexity') == 'low':
            performance += 0.1  # Легче на простых задачах

        if context.get('time_pressure', False):
            if skill_name in ['efficiency', 'workflow_optimization']:
                performance += 0.15  # Эти навыки важны под давлением времени

        if context.get('user_feedback'):
            if skill_name in ['communication', 'personalization']:
                performance += 0.1  # Эти навыки важны для пользовательского взаимодействия

        return min(performance, 1.0)

    async def _generate_skill_recommendations(self, skill_name: str, current_level: float, context_performance: float) -> List[str]:
        """
        Генерация рекомендаций по развитию навыка

        Args:
            skill_name: Название навыка
            current_level: Текущий уровень
            context_performance: Производительность в контексте

        Returns:
            List[str]: Рекомендации
        """
        recommendations = []

        if current_level < 0.4:
            recommendations.append("Фокус на базовых практиках для повышения уровня навыка")
        elif current_level < 0.7:
            recommendations.append("Практика в разнообразных сценариях для углубления навыка")
        else:
            recommendations.append("Фокус на совершенствовании и оптимизации применения навыка")

        if context_performance < 0.6:
            recommendations.append("Дополнительная практика в текущем контексте")
        else:
            recommendations.append("Навык хорошо развит для текущего контекста")

        # Специфические рекомендации
        skill_specific_recs = {
            'query_analysis': ["Анализировать разнообразные типы запросов", "Практиковать распознавание намерений"],
            'tool_selection': ["Экспериментировать с разными инструментами", "Изучать оптимальные комбинации"],
            'error_recovery': ["Симулировать сценарии ошибок", "Практиковать стратегии восстановления"],
            'efficiency': ["Оптимизировать время выполнения", "Изучать техники эффективной работы"],
            'communication': ["Практиковать разные стили общения", "Анализировать обратную связь"]
        }

        if skill_name in skill_specific_recs:
            recommendations.extend(skill_specific_recs[skill_name][:2])  # Ограничиваем до 2 специфических рекомендаций

        return recommendations

    async def _suggest_next_practice(self, skill_name: str, current_level: float, context_performance: float) -> str:
        """
        Предложение следующей практики для развития навыка

        Args:
            skill_name: Название навыка
            current_level: Текущий уровень
            context_performance: Производительность в контексте

        Returns:
            str: Предложение следующей практики
        """
        if current_level < 0.3:
            return f"Базовые упражнения по {skill_name}"
        elif current_level < 0.6:
            return f"Промежуточные задачи для развития {skill_name}"
        elif current_level < 0.9:
            return f"Сложные сценарии для совершенствования {skill_name}"
        else:
            return f"Экспертные вызовы для поддержания мастерства в {skill_name}"

    def get_skill_stats(self) -> Dict[str, Any]:
        """
        Получение статистики развития навыков

        Returns:
            Dict[str, Any]: Статистика навыков
        """
        return {
            'skill_tree': {
                'total_skills': len(self.skill_tree.skills),
                'available_skills': self.skill_tree.get_available_skills(),
                'skill_categories': list(set(s.category.value for s in self.skill_tree.skills.values()))
            },
            'development': self.stats.copy(),
            'assessments': {name: assessment.model_dump() for name, assessment in self.skill_assessments.items()},
            'recent_updates': [update.model_dump() for update in self.development_history[-10:]]
        }

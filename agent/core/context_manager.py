"""
Context Manager - Управление контекстом для агента
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ContextData(BaseModel):
    """Структура данных контекста"""
    user_context: Dict[str, Any] = {}
    conversation_history: List[Dict[str, Any]] = []
    domain_context: Dict[str, Any] = {}
    system_context: Dict[str, Any] = {}
    temporal_context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class ContextManager:
    """Менеджер контекста агента"""

    def __init__(self, memory_manager=None, tool_registry=None, auth_manager=None):
        self.memory_manager = memory_manager
        self.tool_registry = tool_registry
        self.auth_manager = auth_manager

        # Кэш контекста для производительности
        self._context_cache = {}
        self._cache_ttl = 300  # 5 минут

    async def gather_context(self, task: Dict[str, Any]) -> ContextData:
        """Сбор полного контекста для задачи"""
        try:
            context = ContextData()

            # Параллельный сбор различных типов контекста
            import asyncio

            tasks = [
                self._get_user_context(task.get('user_id')),
                self._get_conversation_history(task.get('user_id'), task.get('session_id')),
                self._get_domain_context(task.get('query', '')),
                self._get_system_context(),
                self._get_temporal_context()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Обработка результатов
            context.user_context = results[0] if not isinstance(results[0], Exception) else {}
            context.conversation_history = results[1] if not isinstance(results[1], Exception) else []
            context.domain_context = results[2] if not isinstance(results[2], Exception) else {}
            context.system_context = results[3] if not isinstance(results[3], Exception) else {}
            context.temporal_context = results[4] if not isinstance(results[4], Exception) else {}

            # Добавление метаданных
            context.metadata = {
                'gathered_at': time.time(),
                'task_id': task.get('id'),
                'context_sources': self._get_context_sources(context),
                'cache_used': False
            }

            return context

        except Exception as e:
            logger.error(f"Ошибка сбора контекста: {e}")
            return ContextData(metadata={'error': str(e)})

    async def _get_user_context(self, user_id: Optional[str]) -> Dict[str, Any]:
        """Получение контекста пользователя"""
        if not user_id:
            return {}

        try:
            # Попытка получить из кэша
            cache_key = f"user_context_{user_id}"
            if cache_key in self._context_cache:
                cached_data, timestamp = self._context_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return cached_data

            user_context = {}

            # Базовая информация о пользователе
            user_context['user_id'] = user_id
            user_context['preferences'] = await self._get_user_preferences(user_id)
            user_context['history_stats'] = await self._get_user_history_stats(user_id)
            user_context['behavior_patterns'] = await self._get_user_behavior_patterns(user_id)

            # Кэширование
            self._context_cache[cache_key] = (user_context, time.time())

            return user_context

        except Exception as e:
            logger.warning(f"Ошибка получения контекста пользователя {user_id}: {e}")
            return {'user_id': user_id, 'error': str(e)}

    async def _get_conversation_history(self, user_id: Optional[str],
                                      session_id: Optional[str],
                                      limit: int = 10) -> List[Dict[str, Any]]:
        """Получение истории разговора"""
        try:
            if not user_id or not session_id:
                return []

            # Получение из памяти
            if self.memory_manager:
                history = await self.memory_manager.retrieve_conversation_history(
                    user_id, session_id, limit
                )
                return history

            # Fallback - пустая история
            return []

        except Exception as e:
            logger.warning(f"Ошибка получения истории разговора: {e}")
            return []

    async def _get_domain_context(self, query: str) -> Dict[str, Any]:
        """Определение домена и получение соответствующего контекста"""
        try:
            domain_info = self._analyze_query_domain(query)

            domain_context = {
                'primary_domain': domain_info['domain'],
                'confidence': domain_info['confidence'],
                'related_domains': domain_info['related'],
                'domain_knowledge': await self._get_domain_knowledge(domain_info['domain']),
                'specialized_tools': self._get_domain_tools(domain_info['domain'])
            }

            return domain_context

        except Exception as e:
            logger.warning(f"Ошибка анализа домена: {e}")
            return {'domain': 'general', 'confidence': 0.0}

    async def _get_system_context(self) -> Dict[str, Any]:
        """Получение системного контекста"""
        try:
            system_context = {
                'available_tools': self._get_available_tools(),
                'system_load': self._get_system_load(),
                'active_sessions': self._get_active_sessions_count(),
                'memory_usage': self._get_memory_usage(),
                'timestamp': time.time()
            }

            return system_context

        except Exception as e:
            logger.warning(f"Ошибка получения системного контекста: {e}")
            return {'error': str(e)}

    async def _get_temporal_context(self) -> Dict[str, Any]:
        """Получение временного контекста"""
        now = datetime.now()

        temporal_context = {
            'current_time': now.isoformat(),
            'day_of_week': now.weekday(),
            'hour_of_day': now.hour,
            'is_business_hours': self._is_business_hours(now),
            'season': self._get_season(now),
            'time_since_last_interaction': await self._get_time_since_last_interaction()
        }

        return temporal_context

    def _analyze_query_domain(self, query: str) -> Dict[str, Any]:
        """Анализ домена запроса"""
        query_lower = query.lower()

        # Простые правила определения домена
        domain_rules = {
            'programming': ['python', 'javascript', 'java', 'code', 'программирование', 'код'],
            'science': ['физика', 'математика', 'химия', 'биология', 'наука'],
            'business': ['бизнес', 'финансы', 'экономика', 'компания', 'рынок'],
            'health': ['здоровье', 'медицина', 'болезнь', 'лечение'],
            'education': ['образование', 'школа', 'университет', 'учеба'],
            'entertainment': ['фильм', 'музыка', 'игра', 'развлечение', 'movie', 'game']
        }

        domain_scores = {}
        for domain, keywords in domain_rules.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            confidence = min(domain_scores[primary_domain] / 3, 1.0)  # Нормализация
            related_domains = [d for d, s in domain_scores.items() if s > 0 and d != primary_domain]
        else:
            primary_domain = 'general'
            confidence = 0.0
            related_domains = []

        return {
            'domain': primary_domain,
            'confidence': confidence,
            'related': related_domains
        }

    async def _get_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """Получение знаний по домену"""
        # Заглушка - в реальности здесь будет запрос к knowledge base
        domain_knowledge = {
            'programming': {
                'languages': ['Python', 'JavaScript', 'Java'],
                'concepts': ['OOP', 'algorithms', 'data structures']
            },
            'science': {
                'fields': ['physics', 'chemistry', 'biology'],
                'methods': ['experiment', 'observation', 'modeling']
            }
        }

        return domain_knowledge.get(domain, {})

    def _get_domain_tools(self, domain: str) -> List[str]:
        """Получение инструментов для домена"""
        domain_tools = {
            'programming': ['code_executor', 'rag_tool'],
            'science': ['calculator', 'rag_tool'],
            'business': ['rag_tool', 'web_research'],
            'health': ['rag_tool', 'web_research']
        }

        return domain_tools.get(domain, ['rag_tool'])

    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Получение предпочтений пользователя"""
        # Заглушка - в реальности из базы данных
        return {
            'language': 'ru',
            'response_style': 'detailed',
            'preferred_tools': ['rag_tool', 'web_research']
        }

    async def _get_user_history_stats(self, user_id: str) -> Dict[str, Any]:
        """Получение статистики истории пользователя"""
        # Заглушка
        return {
            'total_interactions': 42,
            'avg_response_time': 2.3,
            'preferred_domains': ['programming', 'science']
        }

    async def _get_user_behavior_patterns(self, user_id: str) -> Dict[str, Any]:
        """Получение паттернов поведения пользователя"""
        # Заглушка
        return {
            'time_patterns': ['morning', 'evening'],
            'query_complexity': 'medium',
            'feedback_score': 4.2
        }

    def _get_available_tools(self) -> List[str]:
        """Получение списка доступных инструментов"""
        if self.tool_registry:
            return list(self.tool_registry.get_available_tools())
        return ['rag_tool', 'web_research', 'calculator']

    def _get_system_load(self) -> float:
        """Получение нагрузки системы"""
        # Заглушка - в реальности системные метрики
        return 0.3

    def _get_active_sessions_count(self) -> int:
        """Получение количества активных сессий"""
        # Заглушка
        return 5

    def _get_memory_usage(self) -> Dict[str, Any]:
        """Получение использования памяти"""
        # Заглушка
        return {
            'used_mb': 256,
            'total_mb': 1024,
            'percentage': 25
        }

    def _is_business_hours(self, dt: datetime) -> bool:
        """Проверка, является ли время рабочим"""
        return 9 <= dt.hour <= 18 and dt.weekday() < 5

    def _get_season(self, dt: datetime) -> str:
        """Определение сезона"""
        month = dt.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'

    async def _get_time_since_last_interaction(self) -> Optional[float]:
        """Получение времени с последнего взаимодействия"""
        # Заглушка
        return 3600.0  # 1 час

    def _get_context_sources(self, context: ContextData) -> List[str]:
        """Получение списка источников контекста"""
        sources = []
        if context.user_context:
            sources.append('user')
        if context.conversation_history:
            sources.append('conversation')
        if context.domain_context:
            sources.append('domain')
        if context.system_context:
            sources.append('system')
        if context.temporal_context:
            sources.append('temporal')

        return sources

    def clear_cache(self):
        """Очистка кэша контекста"""
        self._context_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        return {
            'cache_size': len(self._context_cache),
            'cache_ttl': self._cache_ttl
        }

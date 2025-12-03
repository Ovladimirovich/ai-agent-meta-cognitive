"""
Распределенное обучение между агентами
Фаза 4: Продвинутые функции
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import hashlib
import uuid

from ..learning.models import AgentExperience, LearningResult, Pattern
from ..learning.learning_engine import LearningEngine
from ..core.models import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


class DistributedLearningManager:
    """
    Менеджер распределенного обучения между агентами
    Координирует обмен опытом и коллективное обучение
    """

    def __init__(self, agent_id: str, learning_engine: LearningEngine, redis_client=None):
        """
        Инициализация менеджера распределенного обучения

        Args:
            agent_id: Уникальный ID агента
            learning_engine: Двигатель обучения агента
            redis_client: Redis клиент для координации
        """
        self.agent_id = agent_id
        self.learning_engine = learning_engine
        self.redis_client = redis_client

        # Сеть агентов
        self.connected_agents: Set[str] = set()
        self.agent_capabilities: Dict[str, Dict[str, Any]] = {}

        # Обмен опытом
        self.experience_pool: List[AgentExperience] = []
        self.shared_patterns: Dict[str, Pattern] = {}
        self.learning_sessions: Dict[str, Dict[str, Any]] = {}

        # Метрики распределенного обучения
        self.distributed_metrics = {
            'total_experiences_shared': 0,
            'patterns_exchanged': 0,
            'collaborative_sessions': 0,
            'knowledge_transfer_efficiency': 0.0,
            'network_size': 0
        }

        # Настройки
        self.sharing_threshold = 0.7  # Минимальная уверенность для обмена
        self.max_pool_size = 1000
        self.sync_interval = 300  # 5 минут

        logger.info(f"DistributedLearningManager initialized for agent {agent_id}")

    async def connect_to_network(self, network_config: Dict[str, Any]) -> bool:
        """
        Подключение к сети агентов

        Args:
            network_config: Конфигурация сети

        Returns:
            bool: Успешность подключения
        """
        try:
            # Получение списка доступных агентов
            available_agents = await self._discover_agents(network_config)

            # Подключение к агентам
            for agent_info in available_agents:
                if agent_info['id'] != self.agent_id:
                    success = await self._establish_connection(agent_info)
                    if success:
                        self.connected_agents.add(agent_info['id'])
                        self.agent_capabilities[agent_info['id']] = agent_info.get('capabilities', {})

            self.distributed_metrics['network_size'] = len(self.connected_agents)

            # Запуск фоновой синхронизации
            asyncio.create_task(self._background_sync())

            logger.info(f"Connected to {len(self.connected_agents)} agents in network")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to network: {e}")
            return False

    async def share_experience(self, experience: AgentExperience) -> Dict[str, Any]:
        """
        Обмен опытом с другими агентами

        Args:
            experience: Опыт для обмена

        Returns:
            Dict[str, Any]: Результат обмена
        """
        if not self.connected_agents:
            return {'shared': False, 'reason': 'no_connected_agents'}

        # Оценка ценности опыта для обмена
        value_score = await self._assess_experience_value(experience)

        if value_score < self.sharing_threshold:
            return {'shared': False, 'reason': 'low_value', 'score': value_score}

        # Добавление в пул общего опыта
        self.experience_pool.append(experience)
        self._maintain_pool_size()

        # Асинхронный обмен с подключенными агентами
        sharing_results = []
        for agent_id in self.connected_agents:
            try:
                result = await self._share_with_agent(agent_id, experience)
                sharing_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to share with agent {agent_id}: {e}")

        # Агрегация результатов
        successful_shares = sum(1 for r in sharing_results if r.get('success', False))

        result = {
            'shared': successful_shares > 0,
            'total_agents': len(self.connected_agents),
            'successful_shares': successful_shares,
            'value_score': value_score,
            'experience_id': experience.id
        }

        self.distributed_metrics['total_experiences_shared'] += successful_shares

        logger.info(f"Experience {experience.id} shared with {successful_shares}/{len(self.connected_agents)} agents")
        return result

    async def request_experience(self, query: Dict[str, Any]) -> List[AgentExperience]:
        """
        Запрос опыта от других агентов

        Args:
            query: Критерии поиска опыта

        Returns:
            List[AgentExperience]: Полученный опыт
        """
        if not self.connected_agents:
            return []

        requested_experiences = []

        for agent_id in self.connected_agents:
            try:
                experiences = await self._request_from_agent(agent_id, query)
                requested_experiences.extend(experiences)
            except Exception as e:
                logger.warning(f"Failed to request from agent {agent_id}: {e}")

        # Фильтрация и дедупликация
        unique_experiences = self._deduplicate_experiences(requested_experiences)

        logger.info(f"Requested {len(unique_experiences)} experiences from network")
        return unique_experiences

    async def collaborative_learning_session(self, topic: str, participants: List[str] = None) -> Dict[str, Any]:
        """
        Организация совместной сессии обучения

        Args:
            topic: Тема обучения
            participants: Список участников (None = все подключенные)

        Returns:
            Dict[str, Any]: Результат сессии
        """
        session_id = str(uuid.uuid4())
        participants = participants or list(self.connected_agents)

        if not participants:
            return {'success': False, 'reason': 'no_participants'}

        # Создание сессии
        session = {
            'id': session_id,
            'topic': topic,
            'participants': participants + [self.agent_id],
            'start_time': datetime.now(),
            'status': 'active',
            'contributions': {},
            'consensus_patterns': []
        }

        self.learning_sessions[session_id] = session

        try:
            # Координация сессии
            contributions = await self._coordinate_session(session_id, topic, participants)

            # Синтез коллективного знания
            consensus_patterns = await self._synthesize_knowledge(contributions)

            # Распространение результатов
            await self._distribute_session_results(session_id, consensus_patterns)

            session['status'] = 'completed'
            session['end_time'] = datetime.now()
            session['contributions'] = contributions
            session['consensus_patterns'] = consensus_patterns

            self.distributed_metrics['collaborative_sessions'] += 1

            result = {
                'success': True,
                'session_id': session_id,
                'participants': len(participants) + 1,
                'patterns_synthesized': len(consensus_patterns),
                'duration': (session['end_time'] - session['start_time']).total_seconds()
            }

            logger.info(f"Collaborative learning session {session_id} completed: {result}")
            return result

        except Exception as e:
            session['status'] = 'failed'
            session['error'] = str(e)
            logger.error(f"Collaborative session {session_id} failed: {e}")
            return {'success': False, 'reason': str(e)}

    async def _discover_agents(self, network_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Поиск доступных агентов в сети

        Args:
            network_config: Конфигурация сети

        Returns:
            List[Dict[str, Any]]: Список найденных агентов
        """
        # В реальной реализации здесь может быть:
        # - Поиск через сервис discovery (Consul, etcd)
        # - Проверка heartbeat агентов
        # - Чтение из конфигурации

        # Заглушка - возвращаем тестовых агентов
        return [
            {
                'id': 'agent_001',
                'address': 'localhost:8001',
                'capabilities': {'learning': True, 'reasoning': True}
            },
            {
                'id': 'agent_002',
                'address': 'localhost:8002',
                'capabilities': {'learning': True, 'specialized_domain': 'math'}
            }
        ]

    async def _establish_connection(self, agent_info: Dict[str, Any]) -> bool:
        """
        Установление соединения с агентом

        Args:
            agent_info: Информация об агенте

        Returns:
            bool: Успешность подключения
        """
        # В реальной реализации:
        # - Проверка доступности агента
        # - Аутентификация
        # - Установление защищенного соединения

        # Заглушка
        return True

    async def _assess_experience_value(self, experience: AgentExperience) -> float:
        """
        Оценка ценности опыта для обмена

        Args:
            experience: Опыт агента

        Returns:
            float: Оценка ценности (0.0-1.0)
        """
        # Факторы ценности:
        # - Значимость опыта
        # - Уникальность
        # - Потенциал обучения
        # - Актуальность

        value = 0.0

        # Базовая значимость
        if hasattr(experience, 'significance_score'):
            value += experience.significance_score * 0.4

        # Уникальность (проверка на дубликаты в пуле)
        experience_hash = self._hash_experience(experience)
        is_unique = experience_hash not in [self._hash_experience(e) for e in self.experience_pool[-50:]]
        value += (1.0 if is_unique else 0.3) * 0.3

        # Потенциал обучения (наличие уроков)
        if hasattr(experience, 'lessons') and experience.lessons:
            value += min(len(experience.lessons) * 0.1, 0.3)

        # Актуальность (время)
        if hasattr(experience, 'timestamp'):
            age_hours = (datetime.now() - experience.timestamp).total_seconds() / 3600
            recency_factor = max(0, 1.0 - age_hours / 24)  # Убывает за 24 часа
            value += recency_factor * 0.2

        return min(value, 1.0)

    async def _share_with_agent(self, agent_id: str, experience: AgentExperience) -> Dict[str, Any]:
        """
        Обмен опытом с конкретным агентом

        Args:
            agent_id: ID агента
            experience: Опыт для обмена

        Returns:
            Dict[str, Any]: Результат обмена
        """
        # В реальной реализации - отправка по сети
        # Заглушка
        return {
            'success': True,
            'agent_id': agent_id,
            'experience_id': experience.id,
            'acknowledged': True
        }

    async def _request_from_agent(self, agent_id: str, query: Dict[str, Any]) -> List[AgentExperience]:
        """
        Запрос опыта от конкретного агента

        Args:
            agent_id: ID агента
            query: Критерии поиска

        Returns:
            List[AgentExperience]: Полученный опыт
        """
        # В реальной реализации - запрос по сети
        # Заглушка - возвращаем пустой список
        return []

    async def _coordinate_session(self, session_id: str, topic: str, participants: List[str]) -> Dict[str, Any]:
        """
        Координация совместной сессии обучения

        Args:
            session_id: ID сессии
            topic: Тема
            participants: Участники

        Returns:
            Dict[str, Any]: Вклады участников
        """
        contributions = {}

        # Сбор вклада от текущего агента
        my_contributions = await self._gather_agent_contributions(topic)
        contributions[self.agent_id] = my_contributions

        # Сбор вкладов от других участников
        for participant in participants:
            try:
                participant_contributions = await self._request_contributions(participant, session_id, topic)
                contributions[participant] = participant_contributions
            except Exception as e:
                logger.warning(f"Failed to get contributions from {participant}: {e}")
                contributions[participant] = []

        return contributions

    async def _synthesize_knowledge(self, contributions: Dict[str, Any]) -> List[Pattern]:
        """
        Синтез коллективного знания из вкладов

        Args:
            contributions: Вклады участников

        Returns:
            List[Pattern]: Синтезированные паттерны
        """
        all_patterns = []

        # Сбор всех паттернов
        for agent_contributions in contributions.values():
            if isinstance(agent_contributions, list):
                all_patterns.extend(agent_contributions)

        # Группировка похожих паттернов
        pattern_groups = self._group_similar_patterns(all_patterns)

        # Создание консенсусных паттернов
        consensus_patterns = []
        for group in pattern_groups:
            if len(group) >= 2:  # Минимум 2 агента согласны
                consensus = self._create_consensus_pattern(group)
                consensus_patterns.append(consensus)

        return consensus_patterns

    async def _gather_agent_contributions(self, topic: str) -> List[Pattern]:
        """
        Сбор вклада от текущего агента

        Args:
            topic: Тема

        Returns:
            List[Pattern]: Паттерны агента
        """
        # Получение релевантных паттернов из памяти обучения
        context = {'topic': topic, 'domain': 'general'}
        patterns = await self.learning_engine.get_relevant_patterns(context, limit=10)

        return patterns

    async def _request_contributions(self, agent_id: str, session_id: str, topic: str) -> List[Pattern]:
        """
        Запрос вклада от другого агента

        Args:
            agent_id: ID агента
            session_id: ID сессии
            topic: Тема

        Returns:
            List[Pattern]: Вклад агента
        """
        # В реальной реализации - сетевой запрос
        # Заглушка
        return []

    def _group_similar_patterns(self, patterns: List[Pattern]) -> List[List[Pattern]]:
        """
        Группировка похожих паттернов

        Args:
            patterns: Список паттернов

        Returns:
            List[List[Pattern]]: Группы похожих паттернов
        """
        groups = []
        used = set()

        for i, pattern in enumerate(patterns):
            if i in used:
                continue

            group = [pattern]
            used.add(i)

            # Поиск похожих паттернов
            for j, other in enumerate(patterns):
                if j not in used and self._patterns_similar(pattern, other):
                    group.append(other)
                    used.add(j)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _patterns_similar(self, pattern1: Pattern, pattern2: Pattern) -> bool:
        """
        Проверка схожести паттернов

        Args:
            pattern1, pattern2: Паттерны для сравнения

        Returns:
            bool: Схожи ли паттерны
        """
        # Простая проверка по типу и ключевым элементам
        if pattern1.pattern_type != pattern2.pattern_type:
            return False

        # Проверка пересечения элементов
        elements1 = set(pattern1.elements.keys()) if hasattr(pattern1, 'elements') else set()
        elements2 = set(pattern2.elements.keys()) if hasattr(pattern2, 'elements') else set()

        intersection = elements1.intersection(elements2)
        union = elements1.union(elements2)

        if not union:
            return False

        similarity = len(intersection) / len(union)
        return similarity > 0.5

    def _create_consensus_pattern(self, pattern_group: List[Pattern]) -> Pattern:
        """
        Создание консенсусного паттерна из группы

        Args:
            pattern_group: Группа похожих паттернов

        Returns:
            Pattern: Консенсусный паттерн
        """
        # Использование первого паттерна как основу
        base_pattern = pattern_group[0]

        # Увеличение уверенности на основе консенсуса
        consensus_confidence = min(base_pattern.confidence * (1 + len(pattern_group) * 0.1), 1.0)

        # Создание нового паттерна с повышенной уверенностью
        consensus = Pattern(
            id=f"consensus_{base_pattern.id}_{len(pattern_group)}",
            pattern_type=base_pattern.pattern_type,
            elements=base_pattern.elements.copy(),
            confidence=consensus_confidence,
            frequency=base_pattern.frequency * len(pattern_group),
            context=base_pattern.context,
            metadata={
                **base_pattern.metadata,
                'consensus_agents': len(pattern_group),
                'synthesized_at': datetime.now()
            }
        )

        return consensus

    async def _distribute_session_results(self, session_id: str, patterns: List[Pattern]):
        """
        Распространение результатов сессии

        Args:
            session_id: ID сессии
            patterns: Синтезированные паттерны
        """
        # Сохранение паттернов в общую память
        for pattern in patterns:
            self.shared_patterns[pattern.id] = pattern

        # Распространение среди участников
        for agent_id in self.connected_agents:
            try:
                await self._send_session_results(agent_id, session_id, patterns)
            except Exception as e:
                logger.warning(f"Failed to send results to {agent_id}: {e}")

    async def _send_session_results(self, agent_id: str, session_id: str, patterns: List[Pattern]):
        """
        Отправка результатов сессии агенту

        Args:
            agent_id: ID агента
            session_id: ID сессии
            patterns: Паттерны
        """
        # В реальной реализации - отправка по сети
        pass

    async def _background_sync(self):
        """
        Фоновая синхронизация с сетью
        """
        while True:
            try:
                await asyncio.sleep(self.sync_interval)

                # Синхронизация пула опыта
                await self._sync_experience_pool()

                # Обновление метрик
                self._update_network_metrics()

            except Exception as e:
                logger.error(f"Background sync failed: {e}")

    async def _sync_experience_pool(self):
        """
        Синхронизация пула общего опыта
        """
        # Запрос нового опыта от подключенных агентов
        for agent_id in self.connected_agents:
            try:
                new_experiences = await self._request_recent_experiences(agent_id)
                for exp in new_experiences:
                    if exp not in self.experience_pool:
                        self.experience_pool.append(exp)
            except Exception as e:
                logger.warning(f"Failed to sync with {agent_id}: {e}")

        self._maintain_pool_size()

    async def _request_recent_experiences(self, agent_id: str) -> List[AgentExperience]:
        """
        Запрос недавнего опыта от агента

        Args:
            agent_id: ID агента

        Returns:
            List[AgentExperience]: Недавний опыт
        """
        # Заглушка
        return []

    def _maintain_pool_size(self):
        """
        Поддержание размера пула опыта
        """
        if len(self.experience_pool) > self.max_pool_size:
            # Удаление oldest experiences
            self.experience_pool = self.experience_pool[-self.max_pool_size:]

    def _deduplicate_experiences(self, experiences: List[AgentExperience]) -> List[AgentExperience]:
        """
        Удаление дубликатов опыта

        Args:
            experiences: Список опыта

        Returns:
            List[AgentExperience]: Уникальный опыт
        """
        seen_hashes = set()
        unique = []

        for exp in experiences:
            exp_hash = self._hash_experience(exp)
            if exp_hash not in seen_hashes:
                seen_hashes.add(exp_hash)
                unique.append(exp)

        return unique

    def _hash_experience(self, experience: AgentExperience) -> str:
        """
        Создание хэша опыта для дедупликации

        Args:
            experience: Опыт

        Returns:
            str: Хэш
        """
        content = f"{experience.id}{experience.query}{experience.response}"
        return hashlib.md5(content.encode()).hexdigest()

    def _update_network_metrics(self):
        """
        Обновление метрик сети
        """
        self.distributed_metrics['network_size'] = len(self.connected_agents)

        # Расчет эффективности передачи знаний
        if self.distributed_metrics['total_experiences_shared'] > 0:
            efficiency = self.distributed_metrics['patterns_exchanged'] / self.distributed_metrics['total_experiences_shared']
            self.distributed_metrics['knowledge_transfer_efficiency'] = efficiency

    def get_distributed_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик распределенного обучения

        Returns:
            Dict[str, Any]: Метрики
        """
        return {
            **self.distributed_metrics,
            'connected_agents': list(self.connected_agents),
            'experience_pool_size': len(self.experience_pool),
            'shared_patterns_count': len(self.shared_patterns),
            'active_sessions': len([s for s in self.learning_sessions.values() if s['status'] == 'active'])
        }

    async def disconnect_from_network(self):
        """
        Отключение от сети агентов
        """
        # Очистка соединений
        self.connected_agents.clear()
        self.agent_capabilities.clear()

        # Очистка пула (опционально)
        self.experience_pool.clear()
        self.shared_patterns.clear()

        logger.info("Disconnected from distributed learning network")

"""
Когнитивные карты агента
Фаза 3: Обучение и Адаптация
"""

import json
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging

from .models import (
    MapType,
    MapUpdate,
    MapQuery,
    MapQueryResult,
    CognitiveMap,
    Pattern,
    ProcessedExperience
)

logger = logging.getLogger(__name__)


class BaseCognitiveMap:
    """
    Базовый класс для когнитивных карт
    """

    def __init__(self, map_type: MapType, name: str):
        """
        Инициализация базовой когнитивной карты

        Args:
            map_type: Тип карты
            name: Название карты
        """
        self.map_type = map_type
        self.name = name
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, List[str]] = {}
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'node_count': 0,
            'connection_count': 0
        }

    async def update(self, patterns: List[Pattern], experience: ProcessedExperience) -> Optional[MapUpdate]:
        """
        Обновление карты на основе паттернов и опыта

        Args:
            patterns: Паттерны для обновления
            experience: Обработанный опыт

        Returns:
            Optional[MapUpdate]: Результат обновления или None
        """
        # Базовая реализация - должна быть переопределена в подклассах
        return None

    async def query(self, query: MapQuery) -> Optional[MapQueryResult]:
        """
        Запрос к карте

        Args:
            query: Запрос к карте

        Returns:
            Optional[MapQueryResult]: Результат запроса или None
        """
        # Базовая реализация - должна быть переопределена в подклассах
        return None

    def add_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """
        Добавление узла в карту

        Args:
            node_id: ID узла
            node_data: Данные узла
        """
        self.nodes[node_id] = {
            **node_data,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 0
        }
        self.metadata['node_count'] = len(self.nodes)
        self.metadata['last_updated'] = datetime.now()

    def add_connection(self, from_node: str, to_node: str, connection_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Добавление связи между узлами

        Args:
            from_node: ID исходного узла
            to_node: ID целевого узла
            connection_data: Данные связи
        """
        if from_node not in self.connections:
            self.connections[from_node] = []

        connection = {
            'to_node': to_node,
            'created_at': datetime.now(),
            'strength': 1.0,
            **(connection_data or {})
        }

        self.connections[from_node].append(connection)
        self.metadata['connection_count'] = sum(len(conns) for conns in self.connections.values())
        self.metadata['last_updated'] = datetime.now()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение узла по ID

        Args:
            node_id: ID узла

        Returns:
            Optional[Dict[str, Any]]: Данные узла или None
        """
        node = self.nodes.get(node_id)
        if node:
            node['last_accessed'] = datetime.now()
            node['access_count'] += 1
        return node

    def get_connected_nodes(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Получение связанных узлов

        Args:
            node_id: ID узла

        Returns:
            List[Dict[str, Any]]: Список связанных узлов
        """
        connected = []
        if node_id in self.connections:
            for connection in self.connections[node_id]:
                to_node_id = connection['to_node']
                to_node = self.get_node(to_node_id)
                if to_node:
                    connected.append({
                        'node': to_node,
                        'connection': connection
                    })
        return connected

    def get_map_stats(self) -> Dict[str, Any]:
        """
        Получение статистики карты

        Returns:
            Dict[str, Any]: Статистика карты
        """
        return {
            'name': self.name,
            'type': self.map_type.value,
            'nodes': len(self.nodes),
            'connections': sum(len(conns) for conns in self.connections.values()),
            'metadata': self.metadata.copy()
        }


class DomainKnowledgeMap(BaseCognitiveMap):
    """
    Карта доменных знаний - структурирует знания по предметным областям
    """

    def __init__(self):
        super().__init__(MapType.DOMAIN_KNOWLEDGE, "domain_knowledge")
        self.domains: Dict[str, Dict[str, Any]] = {}

    async def update(self, patterns: List[Pattern], experience: ProcessedExperience) -> Optional[MapUpdate]:
        """
        Обновление карты доменных знаний

        Args:
            patterns: Паттерны для обновления
            experience: Обработанный опыт

        Returns:
            Optional[MapUpdate]: Результат обновления
        """
        changes = {}
        new_nodes = 0
        new_connections = 0

        for pattern in patterns:
            # Извлекаем домен из паттерна
            domain = self._extract_domain_from_pattern(pattern, experience)

            if domain:
                # Создаем или обновляем узел домена
                domain_key = f"domain_{domain}"
                if domain_key not in self.nodes:
                    self.add_node(domain_key, {
                        'type': 'domain',
                        'name': domain,
                        'patterns': [],
                        'confidence': 0.5,
                        'last_pattern_added': datetime.now()
                    })
                    new_nodes += 1

                # Добавляем паттерн к домену
                domain_node = self.nodes[domain_key]
                if pattern.id not in domain_node['patterns']:
                    domain_node['patterns'].append(pattern.id)
                    domain_node['last_pattern_added'] = datetime.now()

                    # Обновляем уверенность домена
                    domain_node['confidence'] = min(domain_node['confidence'] + 0.1, 1.0)

                # Создаем связи между доменами на основе опыта
                related_domains = self._find_related_domains(pattern, experience)
                for related_domain in related_domains:
                    related_key = f"domain_{related_domain}"
                    if related_key != domain_key and related_key in self.nodes:
                        connection_data = {
                            'relationship_type': 'related_experience',
                            'pattern_id': pattern.id,
                            'strength': 0.5
                        }
                        self.add_connection(domain_key, related_key, connection_data)
                        new_connections += 1

        if new_nodes > 0 or new_connections > 0:
            changes = {
                'new_domains': new_nodes,
                'new_domain_connections': new_connections,
                'updated_domains': len([p for p in patterns if self._extract_domain_from_pattern(p, experience)])
            }

            return MapUpdate(
                map_name=self.name,
                changes=changes,
                confidence=0.8,
                impact_assessment={
                    'knowledge_breadth_increase': new_nodes * 0.1,
                    'knowledge_connectivity_increase': new_connections * 0.05
                },
                timestamp=datetime.now()
            )

        return None

    async def query(self, query: MapQuery) -> Optional[MapQueryResult]:
        """
        Запрос к карте доменных знаний

        Args:
            query: Запрос

        Returns:
            Optional[MapQueryResult]: Результат запроса
        """
        if query.query_type == 'domain_info':
            domain_name = query.parameters.get('domain')
            if domain_name:
                domain_key = f"domain_{domain_name}"
                domain_node = self.get_node(domain_key)

                if domain_node:
                    connected_domains = self.get_connected_nodes(domain_key)
                    return MapQueryResult(
                        results={
                            'domain': domain_node,
                            'connected_domains': [conn['node'] for conn in connected_domains],
                            'total_patterns': len(domain_node.get('patterns', []))
                        },
                        confidence=domain_node.get('confidence', 0.5),
                        reasoning=f"Found domain '{domain_name}' with {len(domain_node.get('patterns', []))} patterns",
                        related_maps=[]
                    )

        elif query.query_type == 'all_domains':
            domains_info = []
            for node_id, node_data in self.nodes.items():
                if node_data.get('type') == 'domain':
                    domains_info.append({
                        'name': node_data['name'],
                        'patterns_count': len(node_data.get('patterns', [])),
                        'confidence': node_data.get('confidence', 0.5)
                    })

            return MapQueryResult(
                results={'domains': domains_info},
                confidence=0.9,
                reasoning=f"Found {len(domains_info)} domains",
                related_maps=[]
            )

        return None

    def _extract_domain_from_pattern(self, pattern: Pattern, experience: ProcessedExperience) -> Optional[str]:
        """
        Извлечение домена из паттерна

        Args:
            pattern: Паттерн
            experience: Опыт

        Returns:
            Optional[str]: Название домена или None
        """
        # Анализируем запрос для определения домена
        query = experience.original_experience.query.lower()

        # Простые правила определения домена
        if any(word in query for word in ['analyze', 'анализ', 'data', 'данные']):
            return 'data_analysis'
        elif any(word in query for word in ['search', 'поиск', 'find', 'найти']):
            return 'information_retrieval'
        elif any(word in query for word in ['create', 'создать', 'generate', 'сгенерировать']):
            return 'content_generation'
        elif any(word in query for word in ['calculate', 'вычислить', 'compute']):
            return 'computation'
        elif any(word in query for word in ['chat', 'разговор', 'conversation']):
            return 'conversation'
        else:
            return 'general'

    def _find_related_domains(self, pattern: Pattern, experience: ProcessedExperience) -> List[str]:
        """
        Поиск связанных доменов

        Args:
            pattern: Паттерн
            experience: Опыт

        Returns:
            List[str]: Список связанных доменов
        """
        current_domain = self._extract_domain_from_pattern(pattern, experience)
        if not current_domain:
            return []

        # Определяем связи между доменами
        domain_relations = {
            'data_analysis': ['information_retrieval', 'computation'],
            'information_retrieval': ['data_analysis', 'conversation'],
            'content_generation': ['conversation', 'computation'],
            'computation': ['data_analysis', 'content_generation'],
            'conversation': ['information_retrieval', 'content_generation'],
            'general': ['conversation']
        }

        return domain_relations.get(current_domain, [])


class UserPreferencesMap(BaseCognitiveMap):
    """
    Карта предпочтений пользователя - отслеживает предпочтения и паттерны поведения
    """

    def __init__(self):
        super().__init__(MapType.USER_PREFERENCES, "user_preferences")
        self.preferences: Dict[str, Dict[str, Any]] = {}

    async def update(self, patterns: List[Pattern], experience: ProcessedExperience) -> Optional[MapUpdate]:
        """
        Обновление карты предпочтений пользователя

        Args:
            patterns: Паттерны для обновления
            experience: Обработанный опыт

        Returns:
            Optional[MapUpdate]: Результат обновления
        """
        changes = {}
        updated_preferences = 0

        # Анализируем feedback пользователя
        user_feedback = experience.original_experience.user_feedback
        if user_feedback:
            preference_type = user_feedback.get('type', 'general')
            satisfaction = user_feedback.get('satisfaction', 0.5)

            pref_key = f"pref_{preference_type}"
            if pref_key not in self.nodes:
                self.add_node(pref_key, {
                    'type': 'user_preference',
                    'preference_type': preference_type,
                    'satisfaction_history': [],
                    'average_satisfaction': 0.5,
                    'interaction_count': 0
                })

            # Обновляем предпочтение
            pref_node = self.nodes[pref_key]
            pref_node['satisfaction_history'].append({
                'value': satisfaction,
                'timestamp': datetime.now(),
                'experience_id': experience.original_experience.id
            })

            pref_node['interaction_count'] += 1

            # Пересчитываем среднюю удовлетворенность
            history = pref_node['satisfaction_history']
            pref_node['average_satisfaction'] = sum(h['value'] for h in history) / len(history)

            updated_preferences += 1

        # Анализируем паттерны для выявления предпочтений
        for pattern in patterns:
            if hasattr(pattern, 'type') and pattern.type.value == 'user_behavior':
                pref_type = pattern.trigger_conditions.get('feedback_type', 'behavior')
                pref_key = f"pref_{pref_type}"

                if pref_key not in self.nodes:
                    self.add_node(pref_key, {
                        'type': 'behavior_preference',
                        'preference_type': pref_type,
                        'patterns': [],
                        'strength': 0.5
                    })

                pref_node = self.nodes[pref_key]
                if pattern.id not in pref_node['patterns']:
                    pref_node['patterns'].append(pattern.id)
                    pref_node['strength'] = min(pref_node['strength'] + 0.1, 1.0)
                    updated_preferences += 1

        if updated_preferences > 0:
            changes = {
                'updated_preferences': updated_preferences,
                'new_preferences': len([n for n in self.nodes.keys() if n.startswith('pref_') and
                                       self.nodes[n]['interaction_count'] == 1])
            }

            return MapUpdate(
                map_name=self.name,
                changes=changes,
                confidence=0.7,
                impact_assessment={
                    'preference_accuracy_increase': updated_preferences * 0.05
                },
                timestamp=datetime.now()
            )

        return None

    async def query(self, query: MapQuery) -> Optional[MapQueryResult]:
        """
        Запрос к карте предпочтений пользователя

        Args:
            query: Запрос

        Returns:
            Optional[MapQueryResult]: Результат запроса
        """
        if query.query_type == 'user_preferences':
            preferences = []
            for node_id, node_data in self.nodes.items():
                if node_data.get('type') == 'user_preference':
                    preferences.append({
                        'type': node_data['preference_type'],
                        'average_satisfaction': node_data.get('average_satisfaction', 0.5),
                        'interaction_count': node_data.get('interaction_count', 0)
                    })

            return MapQueryResult(
                results={'preferences': preferences},
                confidence=0.8,
                reasoning=f"Found {len(preferences)} user preferences",
                related_maps=[]
            )

        return None


class TaskStrategiesMap(BaseCognitiveMap):
    """
    Карта стратегий задач - хранит эффективные стратегии решения различных типов задач
    """

    def __init__(self):
        super().__init__(MapType.TASK_STRATEGIES, "task_strategies")
        self.strategies: Dict[str, Dict[str, Any]] = {}

    async def update(self, patterns: List[Pattern], experience: ProcessedExperience) -> Optional[MapUpdate]:
        """
        Обновление карты стратегий задач

        Args:
            patterns: Паттерны для обновления
            experience: Обработанный опыт

        Returns:
            Optional[MapUpdate]: Результат обновления
        """
        changes = {}
        new_strategies = 0
        updated_strategies = 0

        # Определяем тип задачи
        task_type = self._classify_task_type(experience)

        for pattern in patterns:
            if hasattr(pattern, 'type') and pattern.type.value in ['success', 'efficiency']:
                strategy_key = f"strategy_{task_type}_{pattern.type.value}"

                if strategy_key not in self.nodes:
                    self.add_node(strategy_key, {
                        'type': 'task_strategy',
                        'task_type': task_type,
                        'pattern_type': pattern.type.value,
                        'patterns': [],
                        'success_rate': 0.5,
                        'usage_count': 0
                    })
                    new_strategies += 1

                # Обновляем стратегию
                strategy_node = self.nodes[strategy_key]
                if pattern.id not in strategy_node['patterns']:
                    strategy_node['patterns'].append(pattern.id)
                    strategy_node['usage_count'] += 1

                    # Обновляем рейтинг успеха
                    if pattern.type.value == 'success':
                        strategy_node['success_rate'] = min(strategy_node['success_rate'] + 0.1, 1.0)
                    elif pattern.type.value == 'efficiency':
                        strategy_node['success_rate'] = min(strategy_node['success_rate'] + 0.05, 1.0)

                    updated_strategies += 1

        if new_strategies > 0 or updated_strategies > 0:
            changes = {
                'new_strategies': new_strategies,
                'updated_strategies': updated_strategies,
                'task_type': task_type
            }

            return MapUpdate(
                map_name=self.name,
                changes=changes,
                confidence=0.75,
                impact_assessment={
                    'strategy_diversity_increase': new_strategies * 0.1,
                    'strategy_effectiveness_increase': updated_strategies * 0.05
                },
                timestamp=datetime.now()
            )

        return None

    async def query(self, query: MapQuery) -> Optional[MapQueryResult]:
        """
        Запрос к карте стратегий задач

        Args:
            query: Запрос

        Returns:
            Optional[MapQueryResult]: Результат запроса
        """
        if query.query_type == 'task_strategies':
            task_type = query.parameters.get('task_type')
            if task_type:
                strategies = []
                for node_id, node_data in self.nodes.items():
                    if (node_data.get('type') == 'task_strategy' and
                        node_data.get('task_type') == task_type):
                        strategies.append({
                            'strategy_id': node_id,
                            'pattern_type': node_data['pattern_type'],
                            'success_rate': node_data.get('success_rate', 0.5),
                            'usage_count': node_data.get('usage_count', 0)
                        })

                if strategies:
                    return MapQueryResult(
                        results={
                            'task_type': task_type,
                            'strategies': strategies,
                            'best_strategy': max(strategies, key=lambda s: s['success_rate'])
                        },
                        confidence=0.8,
                        reasoning=f"Found {len(strategies)} strategies for task type '{task_type}'",
                        related_maps=[]
                    )

        return None

    def _classify_task_type(self, experience: ProcessedExperience) -> str:
        """
        Классификация типа задачи

        Args:
            experience: Обработанный опыт

        Returns:
            str: Тип задачи
        """
        query = experience.original_experience.query.lower()

        if any(word in query for word in ['analyze', 'анализ', 'process', 'обработать']):
            return 'analysis'
        elif any(word in query for word in ['search', 'поиск', 'find', 'найти']):
            return 'search'
        elif any(word in query for word in ['create', 'создать', 'generate', 'сгенерировать']):
            return 'generation'
        elif any(word in query for word in ['calculate', 'вычислить', 'compute']):
            return 'calculation'
        elif any(word in query for word in ['help', 'помочь', 'assist']):
            return 'assistance'
        else:
            return 'general'


class CognitiveMaps:
    """
    Система когнитивных карт агента
    """

    def __init__(self):
        """
        Инициализация системы когнитивных карт
        """
        self.maps = {
            'domain_knowledge': DomainKnowledgeMap(),
            'user_preferences': UserPreferencesMap(),
            'task_strategies': TaskStrategiesMap(),
            'error_recovery': ErrorRecoveryMap(),
            'performance_patterns': PerformancePatternsMap()
        }

        self.map_connections: Dict[str, List[Dict[str, Any]]] = {}
        self.query_history: List[Dict[str, Any]] = []

        logger.info("CognitiveMaps system initialized")

    async def update_maps(self, patterns: List[Pattern], experience: ProcessedExperience) -> Dict[str, MapUpdate]:
        """
        Обновление всех когнитивных карт

        Args:
            patterns: Паттерны для обновления
            experience: Обработанный опыт

        Returns:
            Dict[str, MapUpdate]: Результаты обновления карт
        """
        updates = {}

        for map_name, cognitive_map in self.maps.items():
            try:
                update = await cognitive_map.update(patterns, experience)
                if update:
                    updates[map_name] = update

                    # Обновление связей между картами
                    await self._update_map_connections(map_name, update, patterns)

                    logger.debug(f"Updated cognitive map '{map_name}': {update.changes}")

            except Exception as e:
                logger.error(f"Failed to update cognitive map '{map_name}': {e}")

        return updates

    async def query_maps(self, query: MapQuery) -> MapQueryResult:
        """
        Запрос к когнитивным картам

        Args:
            query: Запрос

        Returns:
            MapQueryResult: Результат запроса
        """
        results = {}
        related_maps = []

        # Опрашиваем все карты
        for map_name, cognitive_map in self.maps.items():
            try:
                map_result = await cognitive_map.query(query)
                if map_result and map_result.confidence > 0.4:
                    results[map_name] = map_result
                    related_maps.extend(map_result.related_maps)
            except Exception as e:
                logger.error(f"Failed to query cognitive map '{map_name}': {e}")

        # Интегрируем результаты из связанных карт
        integrated_result = await self._integrate_map_results(results, query)

        # Сохраняем историю запросов
        self.query_history.append({
            'query': query.model_dump(),
            'results': integrated_result.model_dump() if integrated_result else None,
            'timestamp': datetime.now()
        })

        return integrated_result

    async def _update_map_connections(self, updated_map: str, update: MapUpdate, patterns: List[Pattern]):
        """
        Обновление связей между картами

        Args:
            updated_map: Название обновленной карты
            update: Результат обновления
            patterns: Паттерны, использованные для обновления
        """
        for pattern in patterns:
            # Ищем связи с другими картами на основе паттерна
            related_maps = await self._find_related_maps(pattern, updated_map)

            for related_map in related_maps:
                connection_key = f"{updated_map}:{related_map}"

                if connection_key not in self.map_connections:
                    self.map_connections[connection_key] = []

                # Добавляем связь
                connection = {
                    'pattern_id': pattern.id,
                    'strength': 0.5,
                    'created_at': datetime.now(),
                    'update_type': update.changes
                }

                self.map_connections[connection_key].append(connection)

                # Укрепляем существующие связи
                for existing_conn in self.map_connections[connection_key]:
                    existing_conn['strength'] = min(existing_conn['strength'] * 1.05, 2.0)

    async def _find_related_maps(self, pattern: Pattern, current_map: str) -> List[str]:
        """
        Поиск связанных карт для паттерна

        Args:
            pattern: Паттерн
            current_map: Текущая карта

        Returns:
            List[str]: Список связанных карт
        """
        related_maps = []

        # Определяем связи на основе типа паттерна
        if pattern.type.value == 'success':
            related_maps.extend(['task_strategies', 'performance_patterns'])
        elif pattern.type.value == 'error':
            related_maps.extend(['error_recovery', 'task_strategies'])
        elif pattern.type.value == 'efficiency':
            related_maps.extend(['performance_patterns', 'task_strategies'])
        elif pattern.type.value == 'user_behavior':
            related_maps.extend(['user_preferences', 'task_strategies'])

        # Убираем текущую карту из списка связанных
        related_maps = [m for m in related_maps if m != current_map]

        return related_maps

    async def _integrate_map_results(self, results: Dict[str, MapQueryResult], query: MapQuery) -> MapQueryResult:
        """
        Интеграция результатов из нескольких карт

        Args:
            results: Результаты от разных карт
            query: Исходный запрос

        Returns:
            MapQueryResult: Интегрированный результат
        """
        if not results:
            return MapQueryResult(
                results={},
                confidence=0.0,
                reasoning="No relevant information found in cognitive maps",
                related_maps=[]
            )

        # Собираем все результаты
        integrated_results = {}
        total_confidence = 0
        reasoning_parts = []
        all_related_maps = set()

        for map_name, result in results.items():
            integrated_results[map_name] = result.results
            total_confidence += result.confidence
            reasoning_parts.append(f"{map_name}: {result.reasoning}")
            all_related_maps.update(result.related_maps)

        # Средняя уверенность
        avg_confidence = total_confidence / len(results)

        # Комбинированное объяснение
        combined_reasoning = "; ".join(reasoning_parts)

        return MapQueryResult(
            results=integrated_results,
            confidence=min(avg_confidence, 1.0),
            reasoning=f"Integrated results from {len(results)} maps: {combined_reasoning}",
            related_maps=list(all_related_maps)
        )

    def get_map_stats(self) -> Dict[str, Any]:
        """
        Получение статистики всех карт

        Returns:
            Dict[str, Any]: Статистика карт
        """
        stats = {}
        for map_name, cognitive_map in self.maps.items():
            stats[map_name] = cognitive_map.get_map_stats()

        stats['system'] = {
            'total_maps': len(self.maps),
            'map_connections': len(self.map_connections),
            'query_history_size': len(self.query_history),
            'last_query': self.query_history[-1] if self.query_history else None
        }

        return stats

    def get_map(self, map_name: str) -> Optional[BaseCognitiveMap]:
        """
        Получение конкретной карты по имени

        Args:
            map_name: Название карты

        Returns:
            Optional[BaseCognitiveMap]: Карта или None
        """
        return self.maps.get(map_name)


# Заглушки для остальных карт (будут реализованы позже)
class ErrorRecoveryMap(BaseCognitiveMap):
    def __init__(self):
        super().__init__(MapType.ERROR_RECOVERY, "error_recovery")

class PerformancePatternsMap(BaseCognitiveMap):
    def __init__(self):
        super().__init__(MapType.PERFORMANCE_PATTERNS, "performance_patterns")

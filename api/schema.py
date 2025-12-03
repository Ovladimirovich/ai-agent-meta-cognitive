"""
GraphQL схема для AI Агента с Мета-Познанием
Фаза 5: Инфраструктура и интеграции
"""

import strawberry
from strawberry import type as strawberry_type
from strawberry.fastapi import GraphQLRouter
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from agent.core.agent_core import AgentCore
from agent.meta_cognitive.meta_controller import MetaCognitiveController
from agent.learning.learning_engine import LearningEngine
from agent.self_awareness.self_monitoring import SelfMonitoringSystem


# Strawberry GraphQL типы


@strawberry_type
class AgentRequestType:
    """Тип запроса к агенту"""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[str] = None  # JSON string
    context: Optional[str] = None   # JSON string


@strawberry_type
class AgentResponseType:
    """Тип ответа агента"""
    id: str
    content: str
    confidence: float
    timestamp: datetime
    metadata: Optional[str] = None  # JSON string
    processing_time: Optional[float] = None


@strawberry_type
class MetaCognitiveResponseType:
    """Тип ответа с мета-познанием"""
    agent_response: AgentResponseType
    meta_decision: str  # JSON string
    coordination_result: str  # JSON string
    reflection_result: str  # JSON string
    learning_result: str  # JSON string
    optimization_result: str  # JSON string
    cognitive_load: float
    processing_time: float
    meta_state_snapshot: str  # JSON string


@strawberry_type
class HealthStatusType:
    """Тип статуса здоровья"""
    status: str
    health_score: float
    issues_count: int
    last_check: datetime


@strawberry_type
class LearningMetricsType:
    """Тип метрик обучения"""
    total_experiences_processed: int
    average_learning_effectiveness: float
    patterns_discovered: int
    skills_improved: int
    cognitive_maps_updated: int
    adaptation_success_rate: float
    time_period: str


@strawberry_type
class SystemInfoType:
    """Тип информации о системе"""
    version: str
    uptime: float
    active_connections: int
    total_requests: int
    average_response_time: float


@strawberry_type
class SystemStatusType:
    """Тип статуса системы"""
    system_status: str
    meta_cognitive_state: str  # JSON string
    timestamp: datetime


# Strawberry Query


@strawberry.type
class Query:
    """GraphQL запросы"""

    @strawberry.field(description="Проверка здоровья системы")
    async def health(self) -> HealthStatusType:
        """Разрешение запроса здоровья"""
        try:
            # Получаем экземпляр self_monitoring через dependencies
            from .main import dependencies

            if not dependencies.self_monitoring:
                raise Exception("Self monitoring not available")

            health = await dependencies.self_monitoring.get_agent_health()

            return HealthStatusType(
                status=health.status,
                health_score=health.health_score,
                issues_count=health.issues_count,
                last_check=health.last_diagnosis
            )

        except Exception as e:
            raise Exception(f"Health check failed: {str(e)}")

    @strawberry.field(description="Получение метрик обучения")
    async def learning_metrics(self, timeframe: str = "7d") -> LearningMetricsType:
        """Разрешение запроса метрик обучения"""
        try:
            # Получаем экземпляр learning_engine через dependencies
            from .main import dependencies

            if not dependencies.learning_engine:
                raise Exception("Learning engine not available")

            metrics = await dependencies.learning_engine.get_learning_metrics(timeframe)

            return LearningMetricsType(
                total_experiences_processed=metrics.total_experiences_processed,
                average_learning_effectiveness=metrics.average_learning_effectiveness,
                patterns_discovered=metrics.patterns_discovered,
                skills_improved=metrics.skills_improved,
                cognitive_maps_updated=metrics.cognitive_maps_updated,
                adaptation_success_rate=metrics.adaptation_success_rate,
                time_period=metrics.time_period
            )

        except Exception as e:
            raise Exception(f"Failed to get learning metrics: {str(e)}")

    @strawberry.field(description="Получение информации о системе")
    async def system_info(self) -> SystemInfoType:
        """Разрешение запроса информации о системе"""
        return SystemInfoType(
            version="1.0.0",
            uptime=0.0,  # Заглушка
            active_connections=0,  # Заглушка
            total_requests=0,  # Заглушка
            average_response_time=0.0  # Заглушка
        )

    @strawberry.field(description="Получение статуса системы")
    async def system_status(self) -> SystemStatusType:
        """Разрешение запроса статуса системы"""
        try:
            # Получаем экземпляр meta_controller через dependencies
            from .main import dependencies

            if not dependencies.meta_controller:
                raise Exception("Meta controller not available")

            status = await dependencies.meta_controller.get_meta_cognitive_state()

            return SystemStatusType(
                system_status='operational',
                meta_cognitive_state=json.dumps(status),
                timestamp=datetime.now()
            )

        except Exception as e:
            raise Exception(f"Failed to get system status: {str(e)}")


# Strawberry Mutations


@strawberry.type
class Mutation:
    """GraphQL мутации"""

    @strawberry.mutation(description="Обработка запроса агентом")
    async def process_agent_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[str] = None,
        context: Optional[str] = None
    ) -> AgentResponseType:
        """Выполнение мутации обработки запроса"""
        try:
            # Получаем глобальный экземпляр agent_core
            from .main import agent_core
            import time

            if not agent_core:
                raise Exception("Agent core not available")

            # Парсим JSON строки
            metadata_dict = json.loads(metadata) if metadata else {}
            context_dict = json.loads(context) if context else {}

            # Преобразование в объект AgentRequest агента
            agent_request = type('AgentRequest', (), {
                'id': f"graphql_{int(time.time())}_{hash(query) % 1000}",
                'query': query,
                'user_id': user_id or 'graphql_user',
                'session_id': session_id or f"session_{int(time.time())}",
                'metadata': metadata_dict,
                'context': context_dict,
                'timestamp': datetime.now()
            })()

            # Обработка запроса
            start_time = time.time()
            response = await agent_core.process_request(agent_request)
            processing_time = time.time() - start_time

            return AgentResponseType(
                id=response.id,
                content=response.content,
                confidence=response.confidence,
                timestamp=getattr(response, 'timestamp', datetime.now()),
                metadata=json.dumps(getattr(response, 'metadata', {})),
                processing_time=processing_time
            )

        except Exception as e:
            raise Exception(f"Request processing failed: {str(e)}")

    @strawberry.mutation(description="Обработка запроса с мета-познанием")
    async def process_meta_cognitive_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[str] = None,
        context: Optional[str] = None
    ) -> MetaCognitiveResponseType:
        """Выполнение мутации мета-познавательной обработки"""
        try:
            # Получаем глобальный экземпляр meta_controller
            from .main import meta_controller
            import time

            if not meta_controller:
                raise Exception("Meta controller not available")

            # Парсим JSON строки
            metadata_dict = json.loads(metadata) if metadata else {}
            context_dict = json.loads(context) if context else {}

            # Преобразование в объект AgentRequest агента
            agent_request = type('AgentRequest', (), {
                'id': f"meta_graphql_{int(time.time())}_{hash(query) % 1000}",
                'query': query,
                'user_id': user_id or 'graphql_user',
                'session_id': session_id or f"session_{int(time.time())}",
                'metadata': metadata_dict,
                'context': context_dict,
                'timestamp': datetime.now()
            })()

            # Обработка с мета-познанием
            meta_response = await meta_controller.process_with_meta_cognition(agent_request)

            return MetaCognitiveResponseType(
                agent_response=AgentResponseType(
                    id=meta_response.agent_response.id,
                    content=meta_response.agent_response.content,
                    confidence=meta_response.agent_response.confidence,
                    timestamp=getattr(meta_response.agent_response, 'timestamp', datetime.now()),
                    metadata=json.dumps(getattr(meta_response.agent_response, 'metadata', {})),
                    processing_time=meta_response.processing_time
                ),
                meta_decision=json.dumps(meta_response.meta_decision),
                coordination_result=json.dumps(meta_response.coordination_result),
                reflection_result=json.dumps(meta_response.reflection_result),
                learning_result=json.dumps(meta_response.learning_result),
                optimization_result=json.dumps(meta_response.optimization_result),
                cognitive_load=meta_response.cognitive_load,
                processing_time=meta_response.processing_time,
                meta_state_snapshot=json.dumps(meta_response.meta_state_snapshot)
            )

        except Exception as e:
            raise Exception(f"Meta-cognitive processing failed: {str(e)}")

    @strawberry.mutation(description="Оптимизация системы")
    async def optimize_system(self) -> str:
        """Выполнение оптимизации системы"""
        try:
            # Получаем глобальный экземпляр meta_controller
            from .main import meta_controller

            if not meta_controller:
                raise Exception("Meta controller not available")

            optimization_result = await meta_controller.optimize_meta_cognitive_system()

            return json.dumps({
                'result': optimization_result,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            raise Exception(f"System optimization failed: {str(e)}")


# Создание схемы
schema = strawberry.Schema(query=Query, mutation=Mutation)

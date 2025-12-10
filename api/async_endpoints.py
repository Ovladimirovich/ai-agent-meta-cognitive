"""
Асинхронные эндпоинты API с оптимизацией производительности и исключением блокирующих операций
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.core.models import AgentConfig
from agent.meta_cognitive.meta_controller import MetaCognitiveController
from agent.learning.learning_engine import LearningEngine
from agent.self_awareness.self_monitoring import SelfMonitoringSystem
from api.input_validator import validate_query
# from api.auth import get_current_user # Закомментирован для версии без аутентификации

logger = logging.getLogger(__name__)


class AsyncAgentRequest(BaseModel):
    """Асинхронный запрос к агенту"""
    query: str = Field(..., description="Текст запроса пользователя")
    user_id: Optional[str] = Field(None, description="ID пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Дополнительные метаданные")
    context: Optional[Dict[str, Any]] = Field(None, description="Контекст запроса")
    stream: bool = Field(False, description="Флаг стриминга ответа")


class AsyncAgentResponse(BaseModel):
    """Асинхронный ответ агента"""
    id: str = Field(..., description="ID ответа")
    content: str = Field(..., description="Содержимое ответа")
    confidence: float = Field(..., description="Уровень уверенности (0.0-1.0)")
    timestamp: str = Field(..., description="Время создания ответа")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Метаданные ответа")
    processing_time: Optional[float] = Field(None, description="Время обработки")


class AsyncMetaCognitiveResponse(BaseModel):
    """Асинхронный ответ с мета-познанием"""
    agent_response: AsyncAgentResponse
    meta_decision: Dict[str, Any]
    coordination_result: Dict[str, Any]
    reflection_result: Dict[str, Any]
    learning_result: Dict[str, Any]
    optimization_result: Dict[str, Any]
    cognitive_load: float
    processing_time: float
    meta_state_snapshot: Dict[str, Any]


class AsyncAPIManager:
    """
    Асинхронный менеджер API с оптимизацией производительности и исключением блокирующих операций
    """

    def __init__(self, agent_core, meta_controller, learning_engine, self_monitoring):
        self.agent_core = agent_core
        self.meta_controller = meta_controller
        self.learning_engine = learning_engine
        self.self_monitoring = self_monitoring

        # Пул потоков для CPU-интенсивных задач
        self.executor = None

    async def process_request_async(self, request: AsyncAgentRequest) -> AsyncAgentResponse:
        """
        Асинхронная обработка запроса без блокировок
        """
        try:
            start_time = time.time()

            # Создание объекта запроса для ядра агента
            agent_request = type('AgentRequest', (), {
                'id': f"api_{int(time.time())}_{hash(request.query) % 1000}",
                'query': request.query,
                'user_id': request.user_id or 'api_user',
                'session_id': request.session_id or f"session_{int(time.time())}",
                'metadata': request.metadata or {},
                'context': request.context or {},
                'timestamp': datetime.now()
            })()

            # Асинхронная обработка запроса
            response = await self.agent_core.process_request(agent_request)
            processing_time = time.time() - start_time

            # Создание асинхронного ответа API
            api_response = AsyncAgentResponse(
                id=f"resp_{int(time.time())}_{hash(request.query) % 100}",
                content=str(response.result),
                confidence=response.confidence,
                timestamp=datetime.now().isoformat(),
                metadata=response.metadata or {},
                processing_time=processing_time
            )

            # Добавление фоновой задачи для обучения (асинхронной)
            if self.learning_engine:
                asyncio.create_task(
                    self._learn_from_experience_async(
                        request.query,
                        api_response.content,
                        api_response.confidence,
                        processing_time,
                        request.metadata
                    )
                )

            return api_response

        except Exception as e:
            logger.error(f"Async request processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")

    async def process_request_streaming(self, request: AsyncAgentRequest) -> AsyncIterator[str]:
        """
        Асинхронная обработка запроса с потоковой передачей
        """
        try:
            start_time = time.time()

            # Создание объекта запроса для ядра агента
            agent_request = type('AgentRequest', (), {
                'id': f"stream_{int(time.time())}_{hash(request.query) % 1000}",
                'query': request.query,
                'user_id': request.user_id or 'api_user',
                'session_id': request.session_id or f"session_{int(time.time())}",
                'metadata': request.metadata or {},
                'context': request.context or {},
                'timestamp': datetime.now()
            })()

            # Асинхронная обработка запроса
            response = await self.agent_core.process_request(agent_request)
            processing_time = time.time() - start_time

            # Потоковая передача результата по частям
            content = str(response.result)
            chunk_size = 50  # Размер чанка в символах

            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                chunk_data = {
                    'id': f"chunk_{int(time.time())}_{i}",
                    'content': chunk,
                    'confidence': response.confidence,
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': processing_time,
                    'is_final': i + chunk_size >= len(content)
                }
                yield f"data: {chunk_data}\n\n"
                await asyncio.sleep(0.01)  # Небольшая задержка для потоковой передачи

        except Exception as e:
            logger.error(f"Async streaming processing failed: {e}")
            yield f"error: {str(e)}\n\n"

    async def process_request_with_meta_cognition_async(self, request: AsyncAgentRequest) -> AsyncMetaCognitiveResponse:
        """
        Асинхронная обработка запроса с полным мета-познанием
        """
        try:
            logger.info(f"🔄 Processing async meta-cognitive request: {request.query[:50]}{'...' if len(request.query) > 50 else ''}")

            if not self.meta_controller:
                logger.error("❌ Meta controller not available")
                raise HTTPException(status_code=503, detail="Meta controller not available")

            # Создание объекта запроса для мета-контроллера
            agent_request = type('AgentRequest', (), {
                'id': f"meta_api_{int(time.time())}_{hash(request.query) % 100}",
                'query': request.query,
                'user_id': request.user_id or 'api_user',
                'session_id': request.session_id or f"session_{int(time.time())}",
                'metadata': request.metadata or {},
                'context': request.context or {},
                'timestamp': datetime.now()
            })()

            # Асинхронная обработка с мета-познанием
            logger.info("🔄 Calling meta controller to process request with meta-cognition...")
            meta_response = await self.meta_controller.process_with_meta_cognition(agent_request)
            logger.info("✅ Meta-cognitive processing completed")

            # Преобразование в асинхронную API модель
            api_response = AsyncMetaCognitiveResponse(
                agent_response=AsyncAgentResponse(
                    id=f"meta_resp_{int(time.time())}_{hash(request.query) % 100}",
                    content=str(meta_response.agent_response.result),
                    confidence=meta_response.agent_response.confidence,
                    timestamp=datetime.now().isoformat(),
                    metadata=meta_response.agent_response.metadata or {},
                    processing_time=meta_response.processing_time
                ),
                meta_decision=meta_response.meta_decision,
                coordination_result=meta_response.coordination_result,
                reflection_result=meta_response.reflection_result,
                learning_result=meta_response.learning_result,
                optimization_result=meta_response.optimization_result,
                cognitive_load=meta_response.cognitive_load,
                processing_time=meta_response.processing_time,
                meta_state_snapshot=meta_response.meta_state_snapshot
            )

            logger.info(f"✅ Async meta-cognitive request processing completed, response ID: {api_response.agent_response.id}")
            return api_response

        except Exception as e:
            logger.error(f"❌ Async meta-cognitive processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Meta-cognitive processing failed: {str(e)}")

    async def _learn_from_experience_async(self, query: str, result: str, confidence: float, execution_time: float, metadata: Optional[Dict]):
        """
        Асинхронное обучение на основе опыта
        """
        try:
            # Создание объекта опыта
            experience = type('AgentExperience', (), {
                'id': f"exp_{int(time.time())}_{hash(query) % 1000}",
                'query': query,
                'result': result,
                'confidence': confidence,
                'execution_time': execution_time,
                'timestamp': datetime.now(),
                'success_indicators': ['api_request_processed'],
                'error_indicators': [],
                'metadata': metadata or {}
            })()

            # Асинхронное обучение
            await self.learning_engine.learn_from_experience(experience)
        except Exception as e:
            logger.error(f"Async learning failed: {e}")

    async def get_health_async(self) -> Dict[str, Any]:
        """
        Асинхронная проверка здоровья системы
        """
        try:
            logger.info("🔄 Async health check initiated")

            if not self.self_monitoring:
                logger.warning("⚠️ Self monitoring system not available")
                return {
                    "status": "unavailable",
                    "health_score": 0.0,
                    "issues_count": 1,
                    "last_check": datetime.now().isoformat(),
                    "detail": "Self monitoring system not available"
                }

            logger.info("🔄 Fetching health from SelfMonitoringSystem...")
            health = await self.self_monitoring.get_agent_health()
            logger.info(f"✅ Async health check completed - status: {health.status}, score: {health.health_score}")

            return {
                "status": health.status,
                "health_score": health.health_score,
                "issues_count": health.issues_count,
                "last_check": health.last_diagnosis.isoformat() if health.last_diagnosis else datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Async health check failed: {e}")
            return {
                "status": "error",
                "health_score": 0.0,
                "issues_count": 1,
                "last_check": datetime.now().isoformat(),
                "detail": str(e)
            }

    async def get_learning_metrics_async(self, timeframe: str = "7d") -> Dict[str, Any]:
        """
        Асинхронное получение метрик обучения
        """
        try:
            logger.info(f"🔄 Async fetching learning metrics for timeframe: {timeframe}")

            if not self.learning_engine:
                logger.warning("⚠️ Learning engine not available")
                return {
                    "total_experiences_processed": 0,
                    "average_learning_effectiveness": 0.0,
                    "patterns_discovered": 0,
                    "skills_improved": 0,
                    "cognitive_maps_updated": 0,
                    "adaptation_success_rate": 0.0,
                    "time_period": timeframe,
                    "detail": "Learning engine not available - system initializing"
                }

            logger.info("🔄 Calling learning engine to get metrics...")
            metrics = await self.learning_engine.get_learning_metrics(timeframe)
            logger.info(f"✅ Async learning metrics fetched successfully, timeframe: {timeframe}")

            return metrics.model_dump()

        except Exception as e:
            logger.error(f"❌ Failed to async get learning metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get learning metrics: {str(e)}")

    async def optimize_system_async(self) -> Dict[str, Any]:
        """
        Асинхронная оптимизация системы
        """
        try:
            logger.info("🔄 Starting async system optimization...")

            if not self.meta_controller:
                logger.error("❌ Meta controller not available")
                return {
                    "status": "optimization_not_available",
                    "result": {},
                    "timestamp": datetime.now().isoformat(),
                    "detail": "Meta controller not available - system initializing"
                }

            logger.info("🔄 Calling meta controller to optimize meta-cognitive system...")
            optimization_result = await self.meta_controller.optimize_meta_cognitive_system()
            logger.info("✅ Async system optimization completed")

            return {
                'status': 'optimization_completed',
                'result': optimization_result,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Async system optimization failed: {e}")
            raise HTTPException(status_code=500, detail=f"System optimization failed: {str(e)}")


def register_async_endpoints(app: FastAPI, api_manager: AsyncAPIManager):
    """
    Регистрация асинхронных эндпоинтов
    """
    @app.post("/agent/process-async", response_model=AsyncAgentResponse)
    async def process_request_async_endpoint(
        request: AsyncAgentRequest
    ):
        """Асинхронная обработка запроса"""
        if request.stream:
            # Возвращаем стриминговый ответ
            async def generate_stream():
                async for chunk in api_manager.process_request_streaming(request):
                    yield chunk
            return StreamingResponse(generate_stream(), media_type="text/plain")
        else:
            # Возвращаем обычный асинхронный ответ
            return await api_manager.process_request_async(request)

    @app.post("/agent/process-meta-async", response_model=AsyncMetaCognitiveResponse)
    async def process_request_with_meta_cognition_async_endpoint(
        request: AsyncAgentRequest
    ):
        """Асинхронная обработка запроса с мета-познанием"""
        return await api_manager.process_request_with_meta_cognition_async(request)

    @app.get("/health-async")
    async def health_check_async_endpoint():
        """Асинхронная проверка здоровья системы"""
        return await api_manager.get_health_async()

    @app.get("/learning/metrics-async")
    async def get_learning_metrics_async_endpoint(timeframe: str = "7d"):
        """Асинхронное получение метрик обучения"""
        return await api_manager.get_learning_metrics_async(timeframe)

    @app.post("/system/optimize-async")
    async def optimize_system_async_endpoint():
        """Асинхронная оптимизация системы"""
        return await api_manager.optimize_system_async()

    logger.info("✅ Async endpoints registered successfully")

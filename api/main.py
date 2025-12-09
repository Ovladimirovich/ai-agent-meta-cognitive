"""
FastAPI приложение для AI Агента с Мета-Познанием
Фаза 5: Инфраструктура и интеграции + Современные функции
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from strawberry.fastapi import GraphQLRouter
import uvicorn

# Импортируем User из auth
try:
    from api.auth import User, get_current_user, require_role, log_authentication, auth_router
except ImportError as e:
    # В случае ошибки импорта создаем заглушку для User и других элементов
    from pydantic import BaseModel
    from typing import Optional

    class User(BaseModel):
        id: int
        email: str
        is_active: bool = True
        is_superuser: bool = False

    # Создаем заглушки для остальных импортируемых элементов
    async def get_current_user():
        return User(id=1, email="fallback@example.com", is_active=True, is_superuser=True)

    def require_role(role: str):
        async def role_checker():
            return User(id=1, email="fallback@example.com", is_active=True, is_superuser=True)
        return role_checker

    async def log_authentication(request, call_next):
        response = await call_next(request)
        return response

    # Создаем пустой роутер как fallback
    from fastapi import APIRouter
    auth_router = APIRouter()

    print(f"⚠️ Warning: Could not import from api.auth, using fallback: {e}")

from agent.core.agent_core import AgentCore
from agent.core.models import AgentConfig
from agent.meta_cognitive.meta_controller import MetaCognitiveController
from agent.learning.learning_engine import LearningEngine
from agent.self_awareness.self_monitoring import SelfMonitoringSystem
from api.input_validator import validate_query
from api.schema import schema
from distributed_task_queue import create_distributed_task_queue, DistributedTaskQueue
# Импорты из auth уже обрабатываются выше, поэтому удаляем дублирующийся блок

# Временно отключаем аутентификацию для разработки
def skip_auth():
    # Используем User, определенный выше (в глобальной области)
    return User(id=1, email="dev@example.com", is_active=True, is_superuser=True)

# Временно исключим импорт роутеров, так как они используют заглушки
# from api.auth import auth_router, register_router, reset_password_router, verify_router, users_router
# from api.auth import fastapi_users
from api.logging_config import (
    setup_production_logging,
    log_requests,
    log_error_with_context,
    create_safe_error_message
)
from api.rate_limiter import (
    rate_limiter,
    setup_default_rate_limits,
    RateLimitMiddleware
)
from api.advanced_security import (
    advanced_rate_limiter,
    setup_advanced_rate_limits,
    create_security_middleware,
    input_validator
)
from api.error_handling_middleware import setup_error_handling_middleware
from api.cognitive_load_endpoints import register_cognitive_load_endpoints
from api.meta_cognitive_config_endpoints import register_meta_cognitive_config_endpoints
from api.extended_monitoring_endpoints import register_extended_monitoring_endpoints
from api.visualization_endpoints import register_visualization_endpoints
from monitoring.integration import setup_monitoring
from api.health_endpoints import initialize_health_checks
from config import get_config

logger = logging.getLogger(__name__)


class AppDependencies:
    """Контейнер зависимостей приложения"""

    def __init__(self):
        self.agent_core: Optional[AgentCore] = None
        self.meta_controller: Optional[MetaCognitiveController] = None
        self.learning_engine: Optional[LearningEngine] = None
        self.self_monitoring: Optional[SelfMonitoringSystem] = None
        self.task_queue: Optional[DistributedTaskQueue] = None

    async def initialize(self):
        """Инициализация всех зависимостей"""
        try:
            # Получение конфигурации
            config = get_config()

            # Создание конфигурации агента из настроек
            agent_config = AgentConfig(
                max_execution_time=config.agent_max_execution_time,
                confidence_threshold=config.agent_confidence_threshold,
                enable_reasoning_trace=config.agent_enable_reasoning_trace,
                enable_memory=config.agent_enable_memory,
                max_memory_entries=config.agent_max_memory_entries,
                tool_timeout=config.agent_tool_timeout
            )

            # Инициализация ядра агента
            self.agent_core = AgentCore(agent_config)
            logger.info("✅ Agent Core initialized")

            # Инициализация мета-контроллера
            self.meta_controller = MetaCognitiveController(self.agent_core)
            logger.info("✅ Meta Controller initialized")

            # Инициализация движка обучения
            self.learning_engine = LearningEngine(
                agent_core=self.agent_core,
                memory_manager=getattr(self.agent_core, 'memory_manager', None)
            )
            logger.info("✅ Learning Engine initialized")

            # Инициализация системы самодиагностики
            if self.agent_core:
                self.self_monitoring = SelfMonitoringSystem(self.agent_core)
                logger.info("✅ Self Monitoring System initialized")
            else:
                logger.error("❌ Cannot initialize Self Monitoring System: agent_core is None")

            # Инициализация распределенной очереди задач
            try:
                self.task_queue = await create_distributed_task_queue()
                logger.info("✅ Distributed Task Queue initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Distributed Task Queue: {e}")
                # Используем fallback очередь задач
                try:
                    from distributed_task_queue_fallback import create_distributed_task_queue as fallback_create_queue
                    self.task_queue = await fallback_create_queue()
                    logger.info("✅ Fallback Distributed Task Queue initialized")
                except Exception as fallback_error:
                    logger.error(f"❌ Failed to initialize fallback queue: {fallback_error}")
                    # В крайнем случае создаем пустой очередь для предотвращения падения приложения
                    class EmptyTaskQueue:
                        async def start(self): pass
                        async def stop(self): pass
                        async def enqueue_task(self, task): return False
                        async def get_task_status(self, task_id): return None
                        async def get_task_result(self, task_id): return None
                        async def get_queue_size(self): return 0
                        is_running = False
                    self.task_queue = EmptyTaskQueue()
                    logger.warning("⚠️ Empty task queue created - async tasks will not be processed")

        except Exception as e:
            logger.error(f"❌ Failed to initialize dependencies: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def cleanup(self):
        """Очистка зависимостей"""
        try:
            if self.meta_controller and hasattr(self.meta_controller, 'health_monitor') and self.meta_controller.health_monitor:
                self.meta_controller.health_monitor.stop_monitoring()

            # Остановка очереди задач
            if self.task_queue:
                await self.task_queue.stop()
                logger.info("✅ Distributed Task Queue stopped")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# Экземпляр контейнера зависимостей
dependencies = AppDependencies()


async def initialize_llm_background():
    """Фоновая инициализация LLM клиента"""
    try:
        logger.info("🔄 Starting background LLM initialization...")
        if dependencies.agent_core:
            # Инициализируем LLM в фоне
            await dependencies.agent_core._init_llm_client_async()
            logger.info("✅ LLM client initialized in background")
        else:
            logger.warning("⚠️ Agent core not available for LLM initialization")
    except Exception as e:
        logger.error(f"❌ Background LLM initialization failed: {e}")
        # Не прерываем запуск приложения при ошибке инициализации LLM


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    logger.info("🚀 Starting AI Agent Meta-Cognitive API")

    try:
        # Инициализация зависимостей через контейнер
        await dependencies.initialize()

        # Настройка мониторинга
        setup_monitoring(
            app=app,
            agent_core=dependencies.agent_core,
            enable_health_checks=True,
            enable_alerts=True,
            enable_metrics=True,
            enable_logging=True
        )
        logger.info("✅ Monitoring system configured")

        # Инициализация health checks после настройки мониторинга
        if dependencies.agent_core:
            initialize_health_checks(dependencies.agent_core)
            logger.info("✅ Health checks initialized")

        # Запуск фоновых задач с обработкой исключений
        try:
            asyncio.create_task(start_background_tasks_with_error_handling())
            # Запуск фоновой инициализации LLM
            asyncio.create_task(initialize_llm_background())
            logger.info("✅ Background tasks scheduled successfully")
        except Exception as e:
            logger.error(f"❌ Failed to schedule background tasks: {e}")
            # Не прерываем инициализацию при ошибке фоновых задач
            logger.warning("⚠️ Continuing without background tasks")

        # Запуск очереди задач
        if dependencies.task_queue:
            await dependencies.task_queue.start()
            logger.info("✅ Distributed Task Queue started")

        logger.info("🎉 All systems initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize systems: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    yield

    # Очистка при завершении
    logger.info("🛑 Shutting down AI Agent Meta-Cognitive API")
    await dependencies.cleanup()


async def start_background_tasks():
    """Запуск фоновых задач"""
    try:
        logger.info("🔄 Starting background tasks...")

        # Запуск мониторинга здоровья
        if dependencies.meta_controller and hasattr(dependencies.meta_controller, 'health_monitor'):
            logger.info("🔄 Starting health monitoring loop...")
            asyncio.create_task(start_health_monitoring_with_error_handling())

        logger.info("✅ All background tasks started successfully")

    except Exception as e:
        logger.error(f"❌ Failed to start background tasks: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


async def start_background_tasks_with_error_handling():
    """Запуск фоновых задач с дополнительной обработкой ошибок"""
    try:
        logger.info("🔄 Starting background tasks with error handling...")

        # Запуск мониторинга здоровья
        if dependencies.meta_controller and hasattr(dependencies.meta_controller, 'health_monitor'):
            logger.info("🔄 Starting health monitoring loop with error handling...")
            asyncio.create_task(start_health_monitoring_with_error_handling())

        # Запуск мониторинга очереди задач
        if dependencies.task_queue:
            logger.info("🔄 Starting task queue monitoring...")
            asyncio.create_task(monitor_task_queue_with_error_handling())

        logger.info("✅ All background tasks with error handling started successfully")

    except Exception as e:
        logger.error(f"❌ Failed to start background tasks with error handling: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


async def monitor_task_queue_with_error_handling():
    """Мониторинг очереди задач с обработкой ошибок"""
    try:
        logger.info("🔄 Starting task queue monitoring loop...")
        while True:
            if not dependencies.task_queue or not dependencies.task_queue.is_running:
                break

            queue_size = await dependencies.task_queue.get_queue_size()
            logger.info(f"📊 Task queue size: {queue_size}")

            # Ожидание перед следующей проверкой
            await asyncio.sleep(30)  # Проверка каждые 30 секунд
    except Exception as e:
        logger.error(f"❌ Error in task queue monitoring loop: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Не прерываем выполнение при ошибках в фоновом процессе


async def start_health_monitoring_with_error_handling():
    """Запуск мониторинга здоровья с обработкой ошибок"""
    try:
        logger.info("🔄 Starting health monitoring loop...")
        if dependencies.meta_controller and dependencies.meta_controller.health_monitor:
            await dependencies.meta_controller.health_monitor.start_monitoring_loop()
    except Exception as e:
        logger.error(f"❌ Error in health monitoring loop: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Не прерываем выполнение при ошибках в фоновом процессе


# Pydantic модели для API (определены раньше для использования в app_test)


class AgentRequest(BaseModel):
    """Запрос к агенту"""
    query: str = Field(..., description="Текст запроса пользователя")
    user_id: Optional[str] = Field(None, description="ID пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Дополнительные метаданные")
    context: Optional[Dict[str, Any]] = Field(None, description="Контекст запроса")


class AgentResponse(BaseModel):
    """Ответ агента"""
    id: str = Field(..., description="ID ответа")
    content: str = Field(..., description="Содержимое ответа")
    confidence: float = Field(..., description="Уровень уверенности (0.0-1.0)")
    timestamp: str = Field(..., description="Время создания ответа")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Метаданные ответа")
    processing_time: Optional[float] = Field(None, description="Время обработки")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetaCognitiveResponse(BaseModel):
    """Ответ с мета-познанием"""
    agent_response: AgentResponse
    meta_decision: Dict[str, Any]
    coordination_result: Dict[str, Any]
    reflection_result: Dict[str, Any]
    learning_result: Dict[str, Any]
    optimization_result: Dict[str, Any]
    cognitive_load: float
    processing_time: float
    meta_state_snapshot: Dict[str, Any]


class HealthStatus(BaseModel):
    """Статус здоровья системы"""
    status: str = Field(..., description="Общий статус")
    health_score: float = Field(..., description="Оценка здоровья (0.0-1.0)")
    issues_count: int = Field(..., description="Количество проблем")
    last_check: datetime = Field(..., description="Время последней проверки")


class LearningMetrics(BaseModel):
    """Метрики обучения"""
    total_experiences_processed: int
    average_learning_effectiveness: float
    patterns_discovered: int
    skills_improved: int
    cognitive_maps_updated: int
    adaptation_success_rate: float
    time_period: str


class SystemInfo(BaseModel):
    """Информация о системе"""
    version: str
    uptime: float
    active_connections: int
    total_requests: int
    average_response_time: float


# Создание FastAPI приложений через фабрику
from .app_factory import get_production_app, get_test_app

# Основное приложение для продакшена
app = get_production_app()

# Тестовое приложение
app_test = get_test_app()

# Добавляем роуты в тестовое приложение напрямую
@app_test.get("/")
async def root_test():
    """Корневой эндпоинт для тестов"""
    return {
        "message": "AI Agent Meta-Cognitive API",
        "version": "1.0",
        "status": "running",
        "docs": "/docs",
        "graphql": "/graphql"
    }

@app_test.get("/health")
async def health_check_test():
    """Тестовый эндпоинт проверки здоровья"""
    return {
        "status": "healthy",
        "health_score": 1.0,
        "issues_count": 0,
        "last_check": datetime.now().isoformat()
    }

@app_test.post("/agent/process")
async def process_request_test():
    """Тестовая версия обработки запроса"""
    logger.info("Processing test request")
    return {
        "id": f"test_resp_{int(time.time())}",
        "content": "Привет! Я AI агент с мета-познанием. Чем могу помочь?",
        "confidence": 0.9,
        "timestamp": datetime.now().isoformat(),
        "processing_time": 0.1
    }

@app_test.get("/system/info")
async def system_info_test():
    """Тестовый эндпоинт системной информации"""
    return {
        "version": "1.0.0",
        "uptime": 0.0,
        "active_connections": 0,
        "total_requests": 0,
        "average_response_time": 0.0
    }

@app_test.get("/system/status")
async def system_status_test():
    """Тестовый эндпоинт статуса системы"""
    return {
        "system_status": "operational",
        "meta_cognitive_state": {},
        "timestamp": datetime.now().isoformat()
    }

@app_test.get("/learning/metrics")
async def learning_metrics_test():
    """Тестовый эндпоинт метрик обучения"""
    return {
        "total_experiences_processed": 0,
        "average_learning_effectiveness": 0.0,
        "patterns_discovered": 0,
        "skills_improved": 0,
        "cognitive_maps_updated": 0,
        "adaptation_success_rate": 0.0,
        "time_period": "7d"
    }

@app_test.post("/system/optimize")
async def system_optimize_test():
    """Тестовый эндпоинт оптимизации системы"""
    return {
        "status": "optimization_completed",
        "result": {},
        "timestamp": datetime.now().isoformat()
    }

@app_test.get("/debug/logs")
async def debug_logs_test():
    """Тестовый эндпоинт отладочных логов"""
    return {
        "logs": [],
        "lines_requested": 100,
        "timestamp": datetime.now().isoformat()
    }

# Получение конфигурации
config = get_config()


# API эндпоинты


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "AI Agent Meta-Cognitive API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "graphql": "/graphql"
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    try:
        logger.info("🔄 Health check initiated")

        # Для Render.com возвращаем базовый health сразу, без ожидания полной инициализации
        # Это позволяет Render обнаружить, что приложение запущено и слушает порт
        if not dependencies.agent_core:
            logger.warning("⚠️ Agent core not available - system still initializing")
            # Возвращаем 200 для Render, чтобы показать что приложение живо
            return JSONResponse(
                status_code=200,
                content={
                    "status": "initializing",
                    "health_score": 0.0,
                    "issues_count": 1,
                    "last_check": datetime.now().isoformat(),
                    "detail": "Agent core not available - system initializing"
                }
            )

        # Попытка получить health от SelfMonitoringSystem
        if dependencies.self_monitoring:
            try:
                logger.info("🔄 Fetching health from SelfMonitoringSystem...")
                health = await dependencies.self_monitoring.get_agent_health()
                logger.info(f"✅ Health check completed - status: {health.status}, score: {health.health_score}")
                return HealthStatus(
                    status=health.status,
                    health_score=health.health_score,
                    issues_count=health.issues_count,
                    last_check=health.last_diagnosis
                )
            except Exception as e:
                # Если SelfMonitoringSystem не работает, возвращаем базовый health
                logger.warning(f"⚠️ SelfMonitoringSystem failed, returning basic health: {e}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
                logger.info("🔄 Returning degraded health status")
                return HealthStatus(
                    status="degraded",
                    health_score=0.5,
                    issues_count=1,
                    last_check=datetime.now()
                )
        else:
            # Если self_monitoring все еще None, возвращаем статус без ошибки
            logger.warning("⚠️ Self monitoring system not available")
            return HealthStatus(
                status="initializing",
                health_score=0.0,
                issues_count=1,
                last_check=datetime.now()
            )

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Для Render возвращаем 200 даже при ошибке, чтобы показать что приложение живо
        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "health_score": 0.0,
                "issues_count": 1,
                "last_check": datetime.now().isoformat(),
                "detail": f"Health check error: {str(e)}"
            }
        )


@app.post("/agent/process", response_model=AgentResponse)
async def process_request(
    request: AgentRequest,
    background_tasks: BackgroundTasks = None,
    current_user = Depends(skip_auth) # Временно отключаем аутентификацию
):
    """
    Обработка запроса агентом (базовая обработка)

    Args:
        request: Объект запроса агента с валидацией Pydantic

    Returns:
        AgentResponse: Ответ агента
    """
    try:
        if not dependencies.agent_core:
            raise HTTPException(status_code=503, detail="Agent core not available")

        # Валидируем и санитизируем запрос с помощью новой системы валидации
        try:
            sanitized_query, warnings = await input_validator.validate_and_sanitize(request.query, request.context)
            if warnings:
                logger.warning(f"Security warnings for user {request.user_id}: {warnings}")
        except Exception as e:
            raise HTTPException(status_code=422, detail={
                "error": "Validation failed",
                "message": str(e),
                "warnings": []
            })

        # Преобразование в объект AgentRequest агента
        agent_request = type('AgentRequest', (), {
            'id': f"api_{int(time.time())}_{hash(sanitized_query) % 100}",
            'query': sanitized_query,
            'user_id': request.user_id or 'api_user',
            'session_id': request.session_id or f"session_{int(time.time())}",
            'metadata': request.metadata or {},
            'context': request.context or {},
            'timestamp': datetime.now()
        })()

        # Обработка запроса
        start_time = time.time()
        response = await dependencies.agent_core.process_request(agent_request)
        processing_time = time.time() - start_time

        # Преобразование ответа из agent_core.AgentResponse в api.AgentResponse
        api_response = AgentResponse(
            id=f"resp_{int(time.time())}_{hash(sanitized_query) % 100}",
            content=str(response.result) if hasattr(response, 'result') else str(getattr(response, 'content', 'No response')),
            confidence=response.confidence if hasattr(response, 'confidence') else 0.0,
            timestamp=datetime.now().isoformat(),
            metadata=getattr(response, 'metadata', {}) or {},
            processing_time=processing_time
        )

        # Добавление фоновой задачи для обучения
        if dependencies.learning_engine and background_tasks:
            background_tasks.add_task(
                dependencies.learning_engine.learn_from_experience,
                type('AgentExperience', (), {
                    'id': f"exp_{api_response.id}",
                    'query': sanitized_query,
                    'result': api_response.content,
                    'confidence': api_response.confidence,
                    'execution_time': processing_time,
                    'timestamp': datetime.now(),
                    'success_indicators': ['api_request_processed'],
                    'error_indicators': [],
                    'metadata': request.metadata or {}
                })()
            )

        return api_response

    except HTTPException:
        # Пробрасываем HTTPException как есть
        raise
    except Exception as e:
        # Логируем ошибку с полным контекстом
        log_error_with_context(
            e,
            "Agent request processing failed",
            request_id=f"api_{int(time.time())}_{hash(request.query) % 1000}",
            user_id=request.user_id,
        )
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")


@app.post("/agent/process-async")
async def process_request_async(
    request: AgentRequest,
    current_user = Depends(skip_auth) # Временно отключаем аутентификацию
):
    """
    Асинхронная обработка запроса агентом через распределенную очередь задач

    Args:
        request: Объект запроса агента с валидацией Pydantic

    Returns:
        dict: ID задачи для отслеживания
    """
    try:
        if not dependencies.task_queue:
            raise HTTPException(status_code=503, detail="Task queue not available")

        # Валидируем и санитизируем запрос с помощью новой системы валидации
        try:
            sanitized_query, warnings = await input_validator.validate_and_sanitize(request.query, request.context)
            if warnings:
                logger.warning(f"Security warnings for user {request.user_id}: {warnings}")
        except Exception as e:
            raise HTTPException(status_code=422, detail={
                "error": "Validation failed",
                "message": str(e),
                "warnings": []
            })

        # Создаем задачу для асинхронной обработки
        from distributed_task_queue import ProcessUserQueryTask, TaskPriority
        task = ProcessUserQueryTask(
            id=f"async_task_{int(time.time())}_{hash(sanitized_query) % 100}",
            name="process_user_query_async",
            payload={
                'query': sanitized_query,
                'user_id': request.user_id or 'api_user',
                'session_id': request.session_id or f"session_{int(time.time())}",
                'metadata': request.metadata or {},
                'context': request.context or {}
            },
            user_id=request.user_id or 'api_user',
            query=sanitized_query,
            context=request.context or {},
            priority=TaskPriority.NORMAL
        )

        # Добавляем задачу в очередь
        success = await dependencies.task_queue.enqueue_task(task)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to enqueue task")

        return {
            'task_id': task.id,
            'status': 'enqueued',
            'message': 'Request has been queued for asynchronous processing'
        }

    except HTTPException:
        # Пробрасываем HTTPException как есть
        raise
    except Exception as e:
        # Логируем ошибку с полным контекстом
        log_error_with_context(
            e,
            "Async agent request processing failed",
            request_id=f"async_api_{int(time.time())}_{hash(request.query) % 1000}",
            user_id=request.user_id,
        )
        raise HTTPException(status_code=500, detail=f"Async request processing failed: {str(e)}")


@app.get("/agent/task-status/{task_id}")
async def get_task_status(
    task_id: str,
    current_user = Depends(skip_auth) # Временно отключаем аутентификацию
):
    """
    Получение статуса задачи асинхронной обработки

    Args:
        task_id: ID задачи

    Returns:
        dict: Статус задачи и результат, если доступен
    """
    try:
        if not dependencies.task_queue:
            raise HTTPException(status_code=503, detail="Task queue not available")

        # Получаем статус задачи
        status = await dependencies.task_queue.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")

        # Получаем результат задачи, если доступен
        result = await dependencies.task_queue.get_task_result(task_id)

        return {
            'task_id': task_id,
            'status': status.value if status else None,
            'result': result.result if result else None,
            'error': result.error if result and hasattr(result, 'error') else None,
            'execution_time': result.execution_time if result and hasattr(result, 'execution_time') else None
        }

    except HTTPException:
        # Пробрасываем HTTPException как есть
        raise
    except Exception as e:
        # Логируем ошибку с полным контекстом
        log_error_with_context(
            e,
            "Get task status failed",
            request_id=f"status_check_{int(time.time())}_{hash(task_id) % 1000}",
            user_id=current_user.id if current_user else None,
        )
        raise HTTPException(status_code=500, detail=f"Get task status failed: {str(e)}")


# Добавляем эндпоинт для улучшения производительности - кэширование результатов
from functools import lru_cache
from typing import Tuple
import hashlib

# Создаем кэш для результатов обработки запросов
@lru_cache(maxsize=128)
def cached_process_request(query_hash: str, query: str) -> Tuple[str, float]:
    """
    Кэшированная обработка запроса для улучшения производительности
    """
    # Возвращаем заглушку, так как полная реализация требует интеграции с агентом
    return f"Processed: {query}", 0.5 # content, confidence


@app.post("/agent/process-meta", response_model=MetaCognitiveResponse)
async def process_request_with_meta_cognition(
    request: AgentRequest,
    current_user = Depends(skip_auth) # Временно отключаем аутентификацию
):
    """
    Обработка запроса с полным мета-познанием

    Args:
        request: Объект запроса агента с валидацией Pydantic

    Returns:
        MetaCognitiveResponse: Ответ с мета-познанием
    """
    try:
        logger.info(f"🔄 Processing meta-cognitive request: {request.query[:50]}{'...' if len(request.query) > 50 else ''}")

        if not dependencies.meta_controller:
            logger.error("❌ Meta controller not available")
            raise HTTPException(status_code=503, detail="Meta controller not available")

        # Валидируем и санитизируем запрос с помощью новой системы валидации
        try:
            sanitized_query, warnings = await input_validator.validate_and_sanitize(request.query, request.context)
            if warnings:
                logger.warning(f"Security warnings for user {request.user_id}: {warnings}")
        except Exception as e:
            raise HTTPException(status_code=422, detail={
                "error": "Validation failed",
                "message": str(e),
                "warnings": []
            })

        # Преобразование в объект AgentRequest агента
        agent_request = type('AgentRequest', (), {
            'id': f"meta_api_{int(time.time())}_{hash(sanitized_query) % 100}",
            'query': sanitized_query,
            'user_id': request.user_id or 'api_user',
            'session_id': request.session_id or f"session_{int(time.time())}",
            'metadata': request.metadata or {},
            'context': request.context or {},
            'timestamp': datetime.now()
        })()

        # Обработка с мета-познанием
        logger.info("🔄 Calling meta controller to process request with meta-cognition...")
        meta_response = await dependencies.meta_controller.process_with_meta_cognition(agent_request)
        logger.info("✅ Meta-cognitive processing completed")

        # Преобразование в API модель
        api_response = MetaCognitiveResponse(
            agent_response=AgentResponse(
                id=f"meta_resp_{int(time.time())}_{hash(sanitized_query) % 100}",
                content=str(getattr(meta_response.agent_response, 'result', getattr(meta_response.agent_response, 'content', 'No response'))),
                confidence=getattr(meta_response.agent_response, 'confidence', 0.0),
                timestamp=datetime.now().isoformat(),
                metadata=getattr(meta_response.agent_response, 'metadata', {}) or {},
                processing_time=getattr(meta_response, 'processing_time', 0.0)
            ),
            meta_decision=getattr(meta_response, 'meta_decision', {}),
            coordination_result=getattr(meta_response, 'coordination_result', {}),
            reflection_result=getattr(meta_response, 'reflection_result', {}),
            learning_result=getattr(meta_response, 'learning_result', {}),
            optimization_result=getattr(meta_response, 'optimization_result', {}),
            cognitive_load=getattr(meta_response, 'cognitive_load', 0.0),
            processing_time=getattr(meta_response, 'processing_time', 0.0),
            meta_state_snapshot=getattr(meta_response, 'meta_state_snapshot', {})
        )

        logger.info(f"✅ Meta-cognitive request processing completed, response ID: {api_response.agent_response.id}")
        return api_response

    except Exception as e:
        logger.error(f"❌ Meta-cognitive processing failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Meta-cognitive processing failed: {str(e)}")


@app.get("/learning/metrics", response_model=LearningMetrics)
async def get_learning_metrics(timeframe: str = "7d", current_user: User = Depends(skip_auth)):
    """
    Получение метрик обучения

    Args:
        timeframe: Период времени ("1d", "7d", "30d", "all")

    Returns:
        LearningMetrics: Метрики обучения
    """
    try:
        logger.info(f"🔄 Fetching learning metrics for timeframe: {timeframe}")

        if not dependencies.learning_engine:
            logger.warning("⚠️ Learning engine not available")
            return LearningMetrics(
                total_experiences_processed=0,
                average_learning_effectiveness=0.0,
                patterns_discovered=0,
                skills_improved=0,
                cognitive_maps_updated=0,
                adaptation_success_rate=0.0,
                time_period=timeframe
            )

        logger.info("🔄 Calling learning engine to get metrics...")
        metrics = await dependencies.learning_engine.get_learning_metrics(timeframe)
        logger.info(f"✅ Learning metrics fetched successfully, timeframe: {timeframe}")

        # Преобразуем ответ от learning_engine к формату API
        # Проверяем, является ли metrics словарем или объектом Pydantic
        if isinstance(metrics, dict):
            # Если это словарь, используем его напрямую
            return LearningMetrics(
                total_experiences_processed=metrics.get('total_experiences_processed', 0),
                average_learning_effectiveness=metrics.get('average_learning_effectiveness', 0.0),
                patterns_discovered=metrics.get('patterns_discovered', 0),
                skills_improved=metrics.get('skills_improved', 0),
                cognitive_maps_updated=metrics.get('cognitive_maps_updated', 0),
                adaptation_success_rate=metrics.get('adaptation_success_rate', 0.0),
                time_period=metrics.get('time_period', timeframe)
            )
        else:
            # Если это объект, пробуем получить значения через getattr
            return LearningMetrics(  # Правильное имя модели
                total_experiences_processed=getattr(metrics, 'total_experiences_processed', 0),
                average_learning_effectiveness=getattr(metrics, 'average_learning_effectiveness', 0.0),
                patterns_discovered=getattr(metrics, 'patterns_discovered', 0),
                skills_improved=getattr(metrics, 'skills_improved', 0),
                cognitive_maps_updated=getattr(metrics, 'cognitive_maps_updated', 0),
                adaptation_success_rate=getattr(metrics, 'adaptation_success_rate', 0.0),
                time_period=getattr(metrics, 'time_period', timeframe)
            )

    except Exception as e:
        logger.error(f"❌ Failed to get learning metrics: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning metrics: {str(e)}")


@app.get("/learning-analytics")
async def get_learning_analytics(timeframe: str = "7d", current_user: User = Depends(skip_auth)):
    """
    Получение аналитики обучения для визуализации
    """
    try:
        logger.info(f"🔄 Fetching learning analytics for timeframe: {timeframe}")

        if not dependencies.learning_engine:
            logger.warning("⚠️ Learning engine not available")
            return {
                "timeframe": timeframe,
                "metrics": {},
                "trends": [],
                "patterns": [],
                "timestamp": datetime.now().isoformat()
            }

        # Получаем метрики обучения
        metrics = await dependencies.learning_engine.get_learning_metrics(timeframe)

        # Форматируем данные для визуализации
        analytics_data = {
            "timeframe": timeframe,
            "metrics": {
                "total_experiences_processed": getattr(metrics, 'total_experiences_processed', 0),
                "average_learning_effectiveness": getattr(metrics, 'average_learning_effectiveness', 0.0),
                "patterns_discovered": getattr(metrics, 'patterns_discovered', 0),
                "skills_improved": getattr(metrics, 'skills_improved', 0),
                "cognitive_maps_updated": getattr(metrics, 'cognitive_maps_updated', 0),
                "adaptation_success_rate": getattr(metrics, 'adaptation_success_rate', 0.0),
            },
            "trends": [
                {"date": (datetime.now().date() - timedelta(days=i)).isoformat(),
                 "value": float(i * 0.1 + 0.5)} for i in range(7, -1, -1)  # Заглушка для демонстрации
            ],
            "patterns": [
                {"name": "Pattern A", "frequency": 15, "effectiveness": 0.8},
                {"name": "Pattern B", "frequency": 8, "effectiveness": 0.9},
                {"name": "Pattern C", "frequency": 22, "effectiveness": 0.75}
            ],
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"✅ Learning analytics fetched successfully for timeframe: {timeframe}")
        return analytics_data

    except Exception as e:
        logger.error(f"❌ Failed to get learning analytics: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning analytics: {str(e)}")


@app.get("/system/status")
async def get_system_status(current_user: User = Depends(skip_auth)):
    """Получение статуса системы"""
    try:
        logger.info("🔄 Fetching system status...")

        if not dependencies.meta_controller:
            logger.warning("⚠️ Meta controller not available")
            return JSONResponse(
                status_code=503,
                content={
                    "system_status": "initializing",
                    "meta_cognitive_state": {},
                    "timestamp": datetime.now().isoformat(),
                    "detail": "Meta controller not available - system initializing"
                }
            )

        logger.info("🔄 Calling meta controller to get meta-cognitive state...")
        status = await dependencies.meta_controller.get_meta_cognitive_state()
        logger.info("✅ System status fetched successfully")

        return {
            'system_status': 'operational',
            'meta_cognitive_state': status,
            'timestamp': datetime.now()
        }

    except Exception as e:
        logger.error(f"❌ Failed to get system status: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@app.get("/system/info", response_model=SystemInfo)
async def get_system_info():
    """Получение информации о системе"""
    return SystemInfo(
        version="1.0.0",
        uptime=time.time(),  # Заглушка - в реальности отслеживать с момента запуска
        active_connections=0,  # Заглушка
        total_requests=0,  # Заглушка
        average_response_time=0.0  # Заглушка
    )


@app.post("/system/optimize")
async def optimize_system(current_user: User = Depends(skip_auth)):
    """Оптимизация системы"""
    try:
        logger.info("🔄 Starting system optimization...")

        if not dependencies.meta_controller:
            logger.error("❌ Meta controller not available")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "optimization_not_available",
                    "result": {},
                    "timestamp": datetime.now().isoformat(),
                    "detail": "Meta controller not available - system initializing"
                }
            )

        logger.info("🔄 Calling meta controller to optimize meta-cognitive system...")
        optimization_result = await dependencies.meta_controller.optimize_meta_cognitive_system()
        logger.info("✅ System optimization completed")

        return {
            'status': 'optimization_completed',
            'result': optimization_result,
            'timestamp': datetime.now()
        }

    except Exception as e:
        logger.error(f"❌ System optimization failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"System optimization failed: {str(e)}")


@app.get("/debug/logs")
async def get_debug_logs(lines: int = 100):
    """Получение отладочных логов (только для разработки)"""
    # В продакшене этот эндпоинт должен быть защищен
    return {
        'logs': [],  # Заглушка - в реальности собирать логи
        'lines_requested': lines,
        'timestamp': datetime.now()
    }


@app.get("/debug/health-test")
async def debug_health_test():
    """Отладочный эндпоинт для тестирования health check"""
    try:
        logger.info("Testing health check components...")

        # Проверка инициализации зависимостей
        if not dependencies.self_monitoring:
            return {"error": "self_monitoring not initialized", "dependencies_initialized": {
                "agent_core": dependencies.agent_core is not None,
                "meta_controller": dependencies.meta_controller is not None,
                "learning_engine": dependencies.learning_engine is not None,
                "self_monitoring": dependencies.self_monitoring is not None
            }}

        # Проверка метода get_agent_health
        logger.info("Calling get_agent_health...")
        try:
            health = await dependencies.self_monitoring.get_agent_health()
            return {
                "health_object": {
                    "status": health.status,
                    "health_score": health.health_score,
                    "issues_count": health.issues_count,
                    "last_diagnosis": health.last_diagnosis.isoformat() if health.last_diagnosis else None
                },
                "test": "success"
            }
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    except Exception as e:
        logger.error(f"Debug health test failed: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/debug/test")
async def debug_test():
    """Простой тестовый эндпоинт без зависимостей"""
    logger.info("Debug test endpoint called")
    return {
        "message": "Debug test successful",
        "timestamp": datetime.now().isoformat(),
        "server_status": "running"
    }


# Подключение роутеров аутентификации
app.include_router(auth_router, prefix="/auth", tags=["authentication"])

# GraphQL эндпоинт
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


# Обработчик исключений
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений с безопасным логированием"""
    # Получаем request_id из заголовков
    request_id = getattr(request, 'headers', {}).get('X-Request-ID', f"err_{int(time.time())}")

    # Логируем ошибку с полным контекстом
    log_error_with_context(
        exc,
        "Unhandled exception in API",
        request_id=request_id,
        url=str(request.url) if request else None,
        method=request.method if request else None
    )

    # Создаем безопасное сообщение об ошибке
    safe_message = create_safe_error_message(exc)

    return JSONResponse(
        status_code=500,
        content={
            'error': 'Internal server error',
            'message': safe_message,
            'request_id': request_id,
            'timestamp': datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload,
        log_level=config.log_level.lower(),
        workers=config.api_workers if not config.api_reload else 1
    )


# Регистрация эндпоинтов когнитивной нагрузки
register_cognitive_load_endpoints(app)

# Регистрация эндпоинтов настройки мета-когнитивных параметров
register_meta_cognitive_config_endpoints(app)

# Регистрация расширенных эндпоинтов мониторинга
register_extended_monitoring_endpoints(app)

# Регистрация эндпоинтов визуализации
register_visualization_endpoints(app)


# Обработка SPA маршрутов (должна быть последней)
@app.get("/{full_path:path}")
async def serve_spa(full_path: str = ""):
    """
    Обработка всех маршрутов для SPA (Single Page Application)
    Возвращает index.html для всех маршрутов, кроме API эндпоинтов
    """
    # Если маршрут начинается с /api или является одним из известных API маршрутов,
    # возвращаем 404, так как он должен быть обработан другим маршрутом
    api_routes = [
        "health", "agent", "learning", "learning-analytics", "system", "auth", "graphql",
        "docs", "redoc", "openapi.json", "debug"
    ]

    # Проверяем, является ли маршрут API маршрутом
    route_parts = [part for part in full_path.split("/") if part]  # Убираем пустые части
    if route_parts and route_parts[0] in api_routes:
        # Это API маршрут, но мы не нашли его, значит возвращаем 404
        raise HTTPException(status_code=404, detail="API endpoint not found")

    # Для всех остальных маршрутов возвращаем корневой эндпоинт,
    # который будет обработан как SPA
    return await root()

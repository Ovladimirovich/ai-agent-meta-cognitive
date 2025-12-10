"""
FastAPI приложение для AI Агента с Мета-Познанием
Минимальная версия для Render.com
"""

import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Настройка базового логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(title="AI Agent Meta-Cognitive API", version="1.0.0")

# Добавление CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Простой health endpoint
@app.get("/health", response_model=HealthStatusResponse)
async def health_check():
    """Проверка здоровья системы"""
    try:
        # Запускаем все проверки
        results = await health_registry.run_all()

        # Получаем сводку
        summary = health_registry.get_summary(results)

        # Рассчитываем health score
        total_checks = summary['total_checks']
        if total_checks > 0:
            health_score = (
                (summary['healthy'] * 1.0 + summary['degraded'] * 0.5) / total_checks
            )
        else:
            health_score = 1.0  # Если нет проверок, считаем систему здоровой

        return HealthStatusResponse(
            status=summary['overall_status'],
            health_score=round(health_score, 2),
            issues_count=summary['degraded'] + summary['unhealthy'],
            last_check=summary['timestamp'],
            details=summary
        )
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        return HealthStatusResponse(
            status="unhealthy",
            health_score=0.0,
            issues_count=1,
            last_check=datetime.now().isoformat(),
            details={"error": str(e)}
        )

# Дополнительный health endpoint для API (совместимость с фронтендом)
@app.get("/api/health", response_model=HealthStatusResponse)
async def api_health_check():
    """Проверка здоровья системы (совместимость с API)"""
    try:
        # Запускаем все проверки
        results = await health_registry.run_all()

        # Получаем сводку
        summary = health_registry.get_summary(results)

        # Рассчитываем health score
        total_checks = summary['total_checks']
        if total_checks > 0:
            health_score = (
                (summary['healthy'] * 1.0 + summary['degraded'] * 0.5) / total_checks
            )
        else:
            health_score = 1.0  # Если нет проверок, считаем систему здоровой

        return HealthStatusResponse(
            status=summary['overall_status'],
            health_score=round(health_score, 2),
            issues_count=summary['degraded'] + summary['unhealthy'],
            last_check=summary['timestamp'],
            details=summary
        )
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        return HealthStatusResponse(
            status="unhealthy",
            health_score=0.0,
            issues_count=1,
            last_check=datetime.now().isoformat(),
            details={"error": str(e)}
        )

# Корневой endpoint
@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "AI Agent Meta-Cognitive API",
        "version": "1.0.0",
        "status": "running",
        "health": "/health"
    }

# Импорт базовых зависимостей и компонентов для health check
from config import get_config
from monitoring.health_check_system import health_registry
from api.health_endpoints import HealthStatusResponse, HealthCheckResponse, HealthSummaryResponse

# Попытка импорта дополнительных компонентов
try:
    logger.info("🔄 Attempting to load advanced features...")

    # Настройка rate limiting
    try:
        from api.rate_limiter import setup_default_rate_limits
        setup_default_rate_limits(app)
        logger.info("✅ Rate limiting configured")
    except Exception as e:
        logger.warning(f"⚠️ Failed to setup rate limiting: {e}")

    # Настройка middleware безопасности - отключена для Render.com
    # try:
    #     from api.advanced_security import create_security_middleware
    #     security_middleware = create_security_middleware()
    #     if security_middleware:
    #         app.add_middleware(type(security_middleware), **security_middleware.__dict__ if hasattr(security_middleware, '__dict__') else {})
    #     logger.info("✅ Security middleware configured")
    # except Exception as e:
    #     logger.warning(f"⚠️ Failed to setup security middleware: {e}")

    # Регистрация дополнительных эндпоинтов
    try:
        from api.cognitive_load_endpoints import register_cognitive_load_endpoints
        register_cognitive_load_endpoints(app)
        logger.info("✅ Cognitive load endpoints registered")
    except Exception as e:
        logger.warning(f"⚠️ Failed to register cognitive load endpoints: {e}")

    try:
        from api.meta_cognitive_config_endpoints import register_meta_cognitive_config_endpoints
        register_meta_cognitive_config_endpoints(app)
        logger.info("✅ Meta-cognitive config endpoints registered")
    except Exception as e:
        logger.warning(f"⚠️ Failed to register meta-cognitive config endpoints: {e}")

    try:
        from api.extended_monitoring_endpoints import register_extended_monitoring_endpoints
        register_extended_monitoring_endpoints(app)
        logger.info("✅ Extended monitoring endpoints registered")
    except Exception as e:
        logger.warning(f"⚠️ Failed to register extended monitoring endpoints: {e}")

    try:
        from api.visualization_endpoints import register_visualization_endpoints
        register_visualization_endpoints(app)
        logger.info("✅ Visualization endpoints registered")
    except Exception as e:
        logger.warning(f"⚠️ Failed to register visualization endpoints: {e}")

    # Попытка загрузки аутентификации
    try:
        from api.auth import auth_router
        app.include_router(auth_router, prefix="/auth", tags=["authentication"])
        logger.info("✅ Authentication router registered")
    except Exception as e:
        logger.warning(f"⚠️ Failed to register authentication router: {e}")

    # Попытка загрузки GraphQL
    try:
        from api.schema import schema
        from strawberry.fastapi import GraphQLRouter
        graphql_app = GraphQLRouter(schema)
        app.include_router(graphql_app, prefix="/graphql")
        logger.info("✅ GraphQL router registered")
    except Exception as e:
        logger.warning(f"⚠️ Failed to register GraphQL router: {e}")

    # Подключение расширенных эндпоинтов здоровья
    try:
        from api.health_endpoints import register_health_endpoints, initialize_health_checks
        register_health_endpoints(app)

        # Инициализируем базовые проверки без агента
        initialize_health_checks()  # Инициализируем базовые проверки
        logger.info("✅ Health endpoints registered and initialized")
    except Exception as e:
        logger.warning(f"⚠️ Failed to register health endpoints: {e}")

    # Попытка инициализации с агентом (опционально)
    try:
        from agent.core.agent_core import AgentCore

        # Создаем конфигурацию агента и инициализируем его для health checks
        agent_config = get_config()
        agent_core = AgentCore(agent_config)

        # Повторная инициализация health checks с агентом
        from api.health_endpoints import initialize_health_checks
        initialize_health_checks(agent_core)
        logger.info("✅ Health checks initialized with AgentCore")
    except ImportError:
        logger.info("💡 AgentCore not available, running health checks without agent monitoring")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize health checks with AgentCore: {e}")

    logger.info("🎉 Advanced features loaded successfully")

except Exception as e:
    logger.warning(f"⚠️ Failed to load advanced features: {e}")
    logger.info("🔄 Running with basic functionality only")

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
@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "health_score": 1.0,
            "issues_count": 0,
            "last_check": datetime.now().isoformat(),
            "message": "Basic health check - service is running"
        }
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

# Попытка импорта дополнительных компонентов
try:
    logger.info("🔄 Attempting to load advanced features...")

    # Импорт базовых зависимостей
    from config import get_config

    # Настройка rate limiting
    try:
        from api.rate_limiter import setup_default_rate_limits
        setup_default_rate_limits(app)
        logger.info("✅ Rate limiting configured")
    except Exception as e:
        logger.warning(f"⚠️ Failed to setup rate limiting: {e}")

    # Настройка middleware безопасности
    try:
        from api.advanced_security import create_security_middleware
        app.middleware("http")(create_security_middleware())
        logger.info("✅ Security middleware configured")
    except Exception as e:
        logger.warning(f"⚠️ Failed to setup security middleware: {e}")

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

    logger.info("🎉 Advanced features loaded successfully")

except Exception as e:
    logger.warning(f"⚠️ Failed to load advanced features: {e}")
    logger.info("🔄 Running with basic functionality only")

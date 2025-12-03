"""
Фабрика приложений FastAPI для AI Агента с Мета-Познанием
Объединяет основное приложение и тестовое приложение
"""

import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from config import get_config
from .logging_config import setup_environment_logging

logger = logging.getLogger(__name__)


class AppFactory:
    """Фабрика для создания FastAPI приложений"""

    @staticmethod
    def create_app(
        title: str = "AI Agent Meta-Cognitive API",
        description: str = "REST API для AI Агента с Мета-Познанием",
        version: str = "1.0.0",
        include_routes: bool = True,
        enable_cors: bool = True,
        cors_origins: Optional[list] = None,
        lifespan = None
    ) -> FastAPI:
        """
        Создание FastAPI приложения с заданной конфигурацией

        Args:
            title: Заголовок приложения
            description: Описание приложения
            version: Версия API
            include_routes: Включать ли маршруты API
            enable_cors: Включать ли CORS middleware
            cors_origins: Список разрешенных origins для CORS
            lifespan: Функция жизненного цикла приложения

        Returns:
            FastAPI: Настроенное приложение
        """
        app = FastAPI(
            title=title,
            description=description,
            version=version,
            lifespan=lifespan
        )

        # Настройка CORS
        if enable_cors:
            config = get_config()
            origins = cors_origins or config.cors_origins

            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=config.cors_allow_credentials,
                allow_methods=config.cors_allow_methods,
                allow_headers=config.cors_allow_headers,
            )

        # Добавление базовых маршрутов
        if include_routes:
            AppFactory._add_basic_routes(app)

        return app

    @staticmethod
    def create_production_app(lifespan=None) -> FastAPI:
        """Создание основного приложения для продакшена"""
        config = get_config()

        # Настройка логирования для продакшена
        setup_environment_logging("production")

        app = AppFactory.create_app(
            title="AI Agent Meta-Cognitive API",
            description="REST API для AI Агента с Мета-Познанием",
            version="1.0.0",
            include_routes=True,
            enable_cors=True,
            lifespan=lifespan
        )

        # Добавление продакшен-специфичных middleware
        from api.rate_limiter import setup_advanced_rate_limits
        from api.error_handling_middleware import setup_error_handling_middleware

        setup_advanced_rate_limits()
        setup_error_handling_middleware(app)

        logger.info("✅ Production app created")
        return app

    @staticmethod
    def create_test_app() -> FastAPI:
        """Создание тестового приложения"""
        app = AppFactory.create_app(
            title="AI Agent Meta-Cognitive API (Test)",
            description="REST API для AI Агента с Мета-Познанием (без lifespan)",
            version="1.0.0",
            include_routes=True,
            enable_cors=True,
            cors_origins=["*"],  # Для тестов разрешаем все origins
            lifespan=None
        )

        logger.info("✅ Test app created")
        return app

    @staticmethod
    def _add_basic_routes(app: FastAPI):
        """Добавление базовых маршрутов"""

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
            return {
                "status": "healthy",
                "health_score": 1.0,
                "issues_count": 0,
                "last_check": datetime.now().isoformat()
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


# Глобальные экземпляры приложений
production_app = None
test_app = None


def get_production_app():
    """Получение основного приложения (синглтон)"""
    global production_app
    if production_app is None:
        # Импорт здесь для избежания циклических зависимостей
        from .main import lifespan
        production_app = AppFactory.create_production_app(lifespan=lifespan)
    return production_app


def get_test_app():
    """Получение тестового приложения (синглтон)"""
    global test_app
    if test_app is None:
        test_app = AppFactory.create_test_app()
    return test_app


# Экспорт приложений для обратной совместимости
def create_app(environment: str = "production"):
    """
    Создание приложения в зависимости от окружения

    Args:
        environment: Окружение ("production", "test", "development")

    Returns:
        FastAPI: Настроенное приложение
    """
    if environment == "test":
        return get_test_app()
    else:
        return get_production_app()

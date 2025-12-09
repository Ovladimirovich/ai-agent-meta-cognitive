"""
Конфигурация AI Агента с Мета-Познанием
Чтение настроек из переменных окружения
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse


class Config:
    """Центральная конфигурация приложения"""

    def __init__(self):
        # ================================
        # Application Settings
        # ================================
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.pythonpath = os.getenv("PYTHONPATH", "/app")

        # ================================
        # Database Configuration
        # ================================
        # Проверяем DATABASE_URL от Render.com или Heroku
        database_url = os.getenv("DATABASE_URL", "")
        if database_url and database_url.startswith("postgres://"):
            # Извлекаем параметры из DATABASE_URL
            parsed = urlparse(database_url)
            self.postgres_host = parsed.hostname or "localhost"
            self.postgres_port = parsed.port or 5432
            self.postgres_db = parsed.path[1:]  # убираем начальный '/'
            self.postgres_user = parsed.username or "ai_agent"
            self.postgres_password = parsed.password or ""
        else:
            # Используем стандартные переменные окружения
            self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
            self.postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
            self.postgres_db = os.getenv("POSTGRES_DB", "ai_agent")
            self.postgres_user = os.getenv("POSTGRES_USER", "ai_agent")
            self.postgres_password = os.getenv("POSTGRES_PASSWORD", "")

        # ================================
        # Redis Configuration
        # ================================
        # Проверяем REDIS_URL от Render.com или Heroku
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            # Извлекаем параметры из REDIS_URL
            parsed = urlparse(redis_url)
            self.redis_host = parsed.hostname or "localhost"
            self.redis_port = parsed.port or 6379
            self.redis_password = parsed.password or ""
            self.redis_db = int(parsed.path[1:] if parsed.path and parsed.path != "/" else "0")
        else:
            # Используем стандартные переменные окружения
            self.redis_host = os.getenv("REDIS_HOST", "localhost")
            self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self.redis_db = int(os.getenv("REDIS_DB", "0"))
            self.redis_password = os.getenv("REDIS_PASSWORD", "")

        # ================================
        # AI Service API Keys
        # ================================
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_organization = os.getenv("OPENAI_ORGANIZATION", "")
        self.google_ai_api_key = os.getenv("GOOGLE_AI_API_KEY", "")

        # ================================
        # Vector Database Configuration
        # ================================
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "")

        # ================================
        # External Services
        # ================================
        self.scraper_api_key = os.getenv("SCRAPER_API_KEY", "")
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")

        # ================================
        # Security Settings
        # ================================
        self.secret_key = os.getenv("SECRET_KEY", "development-secret-key")
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "jwt-secret-key")
        self.jwt_access_token_expire_minutes = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

        # ================================
        # API Configuration
        # ================================
        # Используем порт от Render.com или Heroku, если доступен
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        # Render.com всегда предоставляет переменную PORT
        self.api_port = int(os.getenv("PORT", "8000"))  # Render.com использует переменную PORT
        self.api_workers = int(os.getenv("API_WORKERS", "1"))  # Для Render используем 1 воркер
        self.api_reload = os.getenv("API_RELOAD", "false").lower() == "true"  # Для продакшена reload = false

        # ================================
        # CORS Settings
        # ================================
        cors_origins_str = os.getenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8080"]')
        try:
            self.cors_origins = json.loads(cors_origins_str)
        except json.JSONDecodeError:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8080"]

        self.cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"

        cors_methods_str = os.getenv("CORS_ALLOW_METHODS", '["GET", "POST", "PUT", "DELETE", "OPTIONS"]')
        try:
            self.cors_allow_methods = json.loads(cors_methods_str)
        except json.JSONDecodeError:
            self.cors_allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

        cors_headers_str = os.getenv("CORS_ALLOW_HEADERS", '["*"]')
        try:
            self.cors_allow_headers = json.loads(cors_headers_str)
        except json.JSONDecodeError:
            self.cors_allow_headers = ["*"]

        # ================================
        # File Storage
        # ================================
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
        self.max_upload_size = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB

        allowed_ext_str = os.getenv("ALLOWED_EXTENSIONS", '["pdf", "txt", "md", "json", "yaml", "yml"]')
        try:
            self.allowed_extensions = json.loads(allowed_ext_str)
        except json.JSONDecodeError:
            self.allowed_extensions = ["pdf", "txt", "md", "json", "yaml", "yml"]

        # ================================
        # Cache Settings
        # ================================
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        self.cache_max_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))

        # ================================
        # Learning Configuration
        # ================================
        self.learning_batch_size = int(os.getenv("LEARNING_BATCH_SIZE", "32"))
        self.learning_epochs = int(os.getenv("LEARNING_EPOCHS", "100"))
        self.learning_learning_rate = float(os.getenv("LEARNING_LEARNING_RATE", "0.001"))

        # ================================
        # Monitoring and Metrics
        # ================================
        self.metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
        self.metrics_port = int(os.getenv("METRICS_PORT", "90"))
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))

        # ================================
        # Development Settings
        # ================================
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.development_mode = os.getenv("DEVELOPMENT_MODE", "true").lower() == "true"
        self.enable_swagger = os.getenv("ENABLE_SWAGGER", "true").lower() == "true"
        self.enable_graphiql = os.getenv("ENABLE_GRAPHIQL", "true").lower() == "true"

        # ================================
        # Feature Flags
        # ================================
        self.enable_learning = os.getenv("ENABLE_LEARNING", "true").lower() == "true"
        self.enable_memory = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
        self.enable_self_awareness = os.getenv("ENABLE_SELF_AWARENESS", "true").lower() == "true"
        self.enable_adaptation = os.getenv("ENABLE_ADAPTATION", "true").lower() == "true"
        self.enable_analytics = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
        self.enable_web_research = os.getenv("ENABLE_WEB_RESEARCH", "true").lower() == "true"

        # ================================
        # Backup and Recovery
        # ================================
        self.backup_enabled = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
        self.backup_interval_hours = int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))
        self.backup_retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
        self.backup_dir = Path(os.getenv("BACKUP_DIR", "./backups"))

        # ================================
        # Rate Limiting
        # ================================
        self.rate_limit_requests_per_minute = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
        self.rate_limit_burst_size = int(os.getenv("RATE_LIMIT_BURST_SIZE", "10"))

        # ================================
        # Agent Configuration
        # ================================
        self.agent_max_execution_time = float(os.getenv("AGENT_MAX_EXECUTION_TIME", "30.0"))
        self.agent_confidence_threshold = float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", "0.5"))
        self.agent_enable_reasoning_trace = os.getenv("AGENT_ENABLE_REASONING_TRACE", "true").lower() == "true"
        self.agent_enable_memory = os.getenv("AGENT_ENABLE_MEMORY", "true").lower() == "true"
        self.agent_max_memory_entries = int(os.getenv("AGENT_MAX_MEMORY_ENTRIES", "1000"))
        self.agent_tool_timeout = float(os.getenv("AGENT_TOOL_TIMEOUT", "10.0"))

    # ================================
    # Computed Properties
    # ================================

    @property
    def postgres_url(self) -> str:
        """PostgreSQL connection URL"""
        # Если DATABASE_URL доступен, используем его напрямую
        database_url = os.getenv("DATABASE_URL", "")
        if database_url and database_url.startswith("postgres://"):
            return database_url
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        """Redis connection URL"""
        # Если REDIS_URL доступен, используем его напрямую
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            return redis_url
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def redis_available(self) -> bool:
        """Проверка доступности Redis"""
        try:
            redis_url = self.redis_url
            # Проверяем, что URL не является локальным значением по умолчанию, если мы ожидаем Redis в продакшене
            if self.is_production and ("localhost" in redis_url or "127.0.0.1" in redis_url):
                return False
            return True
        except Exception:
            return False

    @property
    def chroma_url(self) -> str:
        """ChromaDB connection URL"""
        return f"http://{self.chroma_host}:{self.chroma_port}"

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"

    # ================================
    # Validation Methods
    # ================================

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Check required API keys (only in production)
        if self.is_production and not self.openai_api_key and not self.google_ai_api_key:
            errors.append("At least one AI API key must be provided (OPENAI_API_KEY or GOOGLE_AI_API_KEY)")

        # Check database configuration (only in production)
        if self.is_production and not self.postgres_password:
            errors.append("POSTGRES_PASSWORD is required")

        # Check file paths
        if not self.upload_dir.exists():
            try:
                self.upload_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create upload directory: {e}")

        if self.backup_enabled and not self.backup_dir.exists():
            try:
                self.backup_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create backup directory: {e}")

        # Check port ranges
        if not (1 <= self.api_port <= 65535):
            errors.append(f"API_PORT must be between 1 and 65535, got {self.api_port}")

        if not (1 <= self.postgres_port <= 65535):
            errors.append(f"POSTGRES_PORT must be between 1 and 65535, got {self.postgres_port}")

        if not (1 <= self.redis_port <= 65535):
            errors.append(f"REDIS_PORT must be between 1 and 65535, got {self.redis_port}")

        return errors

    # ================================
    # Utility Methods
    # ================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if "password" in key.lower() or "secret" in key.lower() or "key" in key.lower():
                config_dict[key] = "***" if value else ""
            else:
                config_dict[key] = value
        return config_dict

    def __str__(self) -> str:
        """String representation (safe for logging)"""
        return f"Config(environment={self.environment}, debug={self.debug})"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config


def reload_config() -> Config:
    """Reload configuration from environment variables"""
    global config
    config = Config()
    return config


# Validate configuration on import
validation_errors = config.validate()
if validation_errors:
    print("⚠️  Configuration validation errors:")
    for error in validation_errors:
        print(f"  - {error}")
    if config.is_production:
        raise ValueError("Configuration validation failed in production environment")
    else:
        print("⚠️ Continuing with development environment despite validation errors")

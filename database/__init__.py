"""
Модуль базы данных для AI Агента с Мета-Познанием
Фаза 5: Инфраструктура и интеграции
"""
from .async_postgres import AsyncPostgreSQLManager
from .models import User, UserRole, UserSession

__all__ = [
    'AsyncPostgreSQLManager',
    'User',
    'UserRole',
    'UserSession'
]

__version__ = "1.0.0"

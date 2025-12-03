"""
Модуль памяти для AI Агента с Мета-Познанием
Фаза 5: Инфраструктура и интеграции
"""
from .memory_manager_optimized import OptimizedMemoryManager
from .async_memory_manager import AsyncMemoryManager
from .conversation_memory import ConversationMemory

__all__ = [
    'OptimizedMemoryManager',
    'AsyncMemoryManager',
    'ConversationMemory'
]

__version__ = "1.0.0"

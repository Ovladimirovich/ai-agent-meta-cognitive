import time
import logging
from typing import Dict, Any, Optional
from .base_tool import BaseTool, Task, ToolResult
from ..integrations.circuit_breaker import circuit_breaker_decorator, CircuitBreakerConfig

logger = logging.getLogger(__name__)

class HybridModelsTool(BaseTool):
    name = "hybrid_models"
    description = "Интеллектуальная маршрутизация между AI моделями (Qwen, Gemini, OpenAI, etc.)"
    version = "1.0.0"

    def __init__(self):
        self.hybrid_manager = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Асинхронная инициализация инструмента"""
        if self._initialized:
            return True

        try:
            # Импорт и инициализация HybridManager
            from hybrid.hybrid_manager import HybridManager
            from hybrid.models import ModelProvider, ModelConfig

            # Конфигурация моделей (будет загружена из настроек)
            model_configs = self._load_model_configs()

            # Создание экземпляра менеджера
            self.hybrid_manager = HybridManager(model_configs)

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize HybridModelsTool: {e}")
            return False

    def can_handle(self, task: Task) -> bool:
        """Проверка возможности обработки задачи"""
        if not isinstance(task.query, str) or not task.query.strip():
            return False

        # Проверка на наличие запрещенных паттернов
        forbidden_patterns = ['<script>', 'javascript:', 'system(', 'exec(']
        query_lower = task.query.lower()

        for pattern in forbidden_patterns:
            if pattern in query_lower:
                return False

        return True

    @circuit_breaker_decorator("hybrid_models_tool", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=90.0,
        timeout=60.0,
        name="hybrid_models_tool"
    ))
    async def execute(self, task: Task) -> ToolResult:
        """Выполнение задачи через гибридную систему моделей"""
        start_time = time.time()

        try:
            # Инициализация при первом использовании
            if not await self.initialize():
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    execution_time=time.time() - start_time,
                    error_message="Failed to initialize hybrid models"
                )

            # Подготовка контекста для модели
            context = self._prepare_context(task)

            # Выполнение запроса через гибридную систему
            response = await self.hybrid_manager.process_query(
                query=task.query,
                context=context,
                force_provider=task.metadata.get('force_provider'),
                user_id=task.user_id
            )

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data={
                    'answer': response.answer,
                    'model_used': response.model_name,
                    'provider': response.provider_used.value,
                    'confidence': response.confidence,
                    'tokens_used': response.tokens_used,
                    'cost': response.cost,
                    'processing_time': response.processing_time
                },
                metadata={
                    'tool': self.name,
                    'version': self.version,
                    'model_selected': response.model_name,
                    'fallback_used': response.used_fallback,
                    'is_final': True  # Этот инструмент дает финальный ответ
                },
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"HybridModelsTool execution error: {e}")

            return ToolResult(
                success=False,
                data=None,
                metadata={'tool': self.name, 'error_type': type(e).__name__},
                execution_time=execution_time,
                error_message=str(e)
            )

    def _prepare_context(self, task: Task) -> str:
        """Подготовка контекста для модели"""
        context_parts = []

        # RAG контекст
        if 'rag_context' in task.context:
            rag_data = task.context['rag_context']
            if isinstance(rag_data, list) and rag_data:
                context_parts.append("Контекст из базы знаний:")
                for item in rag_data[:3]:  # Ограничение до 3 элементов
                    if isinstance(item, dict) and 'content' in item:
                        context_parts.append(f"- {item['content'][:500]}...")

        # Исторический контекст
        if 'conversation_history' in task.context:
            history = task.context['conversation_history']
            if isinstance(history, list) and len(history) > 0:
                context_parts.append("\nИстория разговора:")
                recent_messages = history[-3:]  # Последние 3 сообщения
                for msg in recent_messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:200]
                        context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def _load_model_configs(self) -> Dict:
        """Загрузка конфигурации моделей из переменных окружения"""
        import os

        # Конфигурация моделей по умолчанию
        default_configs = {
            'qwen': {
                'enabled': True,
                'model_name': 'qwen2-1.5b',
                'api_key': None,
                'base_url': 'http://localhost:11434',
                'priority': 1
            },
            'gemini': {
                'enabled': True,
                'model_name': 'gemini-pro',
                'api_key': os.getenv('GOOGLE_AI_API_KEY', ''),
                'priority': 2
            },
            'openai': {
                'enabled': True,
                'model_name': 'gpt-4',
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'priority': 3
            },
            'mistral': {
                'enabled': True,
                'model_name': 'mistral-small',
                'api_key': os.getenv('MISTRAL_API_KEY', ''),
                'priority': 4
            },
            'anthropic': {
                'enabled': True,
                'model_name': 'claude-3-haiku-20240307',
                'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
                'priority': 5
            }
        }

        # Фильтруем только модели с доступными API ключами (кроме Ollama)
        available_configs = {}
        for model_name, config in default_configs.items():
            if model_name == 'qwen' or config.get('api_key'):
                available_configs[model_name] = config
            else:
                logger.warning(f"Model {model_name} disabled - no API key found")

        # Сортируем по приоритету
        sorted_configs = dict(sorted(available_configs.items(), key=lambda x: x[1]['priority']))

        logger.info(f"Loaded model configurations: {list(sorted_configs.keys())}")
        return sorted_configs

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса инструмента"""
        return {
            'name': self.name,
            'initialized': self._initialized,
            'version': self.version,
            'capabilities': ['text_generation', 'conversation', 'analysis'],
            'supported_providers': ['qwen', 'gemini', 'openai'] if self._initialized else []
        }

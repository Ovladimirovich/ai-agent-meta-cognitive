"""
OpenAI клиент для AI Агента с Мета-Познанием
Фаза 5: Инфраструктура и интеграции
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from functools import wraps

import openai
from openai import AsyncOpenAI
import tiktoken

from agent.core.models import AgentRequest, AgentResponse
from .circuit_breaker import (
    CircuitBreakerConfig,
    create_external_service_circuit_breaker,
    circuit_breaker_decorator
)
from .fallback_manager import (
    GracefulDegradationManager,
    DegradationConfig,
    STANDARD_DEGRADATION_STRATEGIES
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Клиент для работы с OpenAI API
    Поддерживает различные модели и оптимизации
    """

    def __init__(self, api_key: str, organization: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 60.0):
        """
        Инициализация OpenAI клиента

        Args:
            api_key: API ключ OpenAI
            organization: ID организации (опционально)
            base_url: Базовый URL для API (для прокси)
            timeout: Таймаут запросов в секундах
        """
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout

        self.client: Optional[AsyncOpenAI] = None

        # Настройки моделей
        self.models = {
            'gpt-4': {
                'max_tokens': 8192,
                'context_window': 8192,
                'cost_per_1k_input': 0.03,
                'cost_per_1k_output': 0.06
            },
            'gpt-4-turbo': {
                'max_tokens': 4096,
                'context_window': 128000,
                'cost_per_1k_input': 0.01,
                'cost_per_1k_output': 0.03
            },
            'gpt-3.5-turbo': {
                'max_tokens': 4096,
                'context_window': 16385,
                'cost_per_1k_input': 0.0015,
                'cost_per_1k_output': 0.002
            },
            'gpt-3.5-turbo-16k': {
                'max_tokens': 16384,
                'context_window': 16385,
                'cost_per_1k_input': 0.003,
                'cost_per_1k_output': 0.004
            }
        }

        # Статистика использования
        self.usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'model_usage': {},
            'errors': 0,
            'avg_response_time': 0.0
        }

        # Кэш токенизаторов
        self.tokenizers = {}

        # Кэш для API результатов
        self.response_cache = None
        self.cache_enabled = True
        self.cache_ttl = 3600  # 1 час по умолчанию

        # Инициализация менеджера плавного ухудшения
        self.degradation_manager = GracefulDegradationManager(
            DegradationConfig(name="openai_client_degradation")
        )
        
        logger.info("OpenAIClient initialized")

    @circuit_breaker_decorator("openai_api", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        timeout=30.0,
        name="openai_api"
    ))
    async def initialize(self):
        """Инициализация клиента"""
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=3
            )

            # Проверка подключения
            await self.client.models.list()
            logger.info("✅ OpenAI client initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise

    async def close(self):
        """Закрытие клиента"""
        if self.client:
            await self.client.close()
        logger.info("OpenAI client closed")

    @circuit_breaker_decorator("openai_chat", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        timeout=60.0,
        name="openai_chat"
    ))
    async def generate_response(self, messages: List[Dict[str, str]],
                              model: str = "gpt-4",
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Генерация ответа с использованием OpenAI

        Args:
            messages: Список сообщений
            model: Модель для использования
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            **kwargs: Дополнительные параметры

        Returns:
            Dict[str, Any]: Ответ с метаданными
        """
        start_time = time.time()

        try:
            if not self.client:
                raise Exception("Client not initialized")

            # Проверка модели
            if model not in self.models:
                logger.warning(f"Unknown model {model}, using gpt-4")
                model = "gpt-4"

            # Настройка параметров
            model_config = self.models[model]
            if max_tokens is None:
                max_tokens = min(model_config['max_tokens'], 4000)  # Безопасный лимит

            # Ограничение контекста
            messages = self._truncate_messages(messages, model_config['context_window'] - max_tokens - 100)

            # Запрос к API
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            response_time = time.time() - start_time

            # Извлечение данных
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason

            # Подсчет токенов
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Расчет стоимости
            cost = self._calculate_cost(model, input_tokens, output_tokens)

            # Обновление статистики
            self._update_stats(model, total_tokens, cost, response_time)

            result = {
                'content': content,
                'finish_reason': finish_reason,
                'model': model,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens
                },
                'cost': cost,
                'response_time': response_time,
                'temperature': temperature
            }

            logger.debug(f"Generated response with {model}: {total_tokens} tokens, ${cost:.4f}")
            return result

        except Exception as e:
            response_time = time.time() - start_time
            self.usage_stats['errors'] += 1
            logger.error(f"Failed to generate response: {e}")
            raise

    async def generate_embeddings(self, texts: List[str],
                                model: str = "text-embedding-ada-002") -> List[List[float]]:
        """
        Генерация эмбеддингов для текстов

        Args:
            texts: Список текстов
            model: Модель эмбеддингов

        Returns:
            List[List[float]]: Список эмбеддингов
        """
        try:
            if not self.client:
                raise Exception("Client not initialized")

            # Разделение на батчи для избежания лимитов
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = await self.client.embeddings.create(
                    input=batch,
                    model=model
                )

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                # Обновление статистики
                tokens_used = response.usage.total_tokens
                self._update_embedding_stats(model, tokens_used)

            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    async def moderate_content(self, content: str) -> Dict[str, Any]:
        """
        Модерация контента

        Args:
            content: Контент для модерации

        Returns:
            Dict[str, Any]: Результат модерации
        """
        try:
            if not self.client:
                raise Exception("Client not initialized")

            response = await self.client.moderations.create(input=content)

            result = response.results[0]

            return {
                'flagged': result.flagged,
                'categories': result.categories.model_dump(),
                'category_scores': result.category_scores.model_dump()
            }

        except Exception as e:
            logger.error(f"Failed to moderate content: {e}")
            return {'flagged': False, 'error': str(e)}

    def _truncate_messages(self, messages: List[Dict[str, str]], max_context_tokens: int) -> List[Dict[str, str]]:
        """
        Усечение сообщений для соответствия лимиту контекста

        Args:
            messages: Список сообщений
            max_context_tokens: Максимальное количество токенов

        Returns:
            List[Dict[str, str]]: Усеченный список сообщений
        """
        if not messages:
            return messages

        # Получение токенизатора
        tokenizer = self._get_tokenizer("gpt-4")  # Используем GPT-4 токенизатор как базовый

        # Подсчет токенов
        total_tokens = 0
        truncated_messages = []

        # Обрабатываем сообщения в обратном порядке (сохраняя последние)
        for message in reversed(messages):
            content = message.get('content', '')
            message_tokens = len(tokenizer.encode(content))

            if total_tokens + message_tokens > max_context_tokens:
                # Усечение контента сообщения
                available_tokens = max_context_tokens - total_tokens
                if available_tokens > 50:  # Минимум для осмысленного контента
                    truncated_content = self._truncate_text(content, available_tokens, tokenizer)
                    truncated_messages.insert(0, {
                        'role': message.get('role', 'user'),
                        'content': truncated_content
                    })
                break
            else:
                total_tokens += message_tokens
                truncated_messages.insert(0, message)

        return truncated_messages

    def _truncate_text(self, text: str, max_tokens: int, tokenizer) -> str:
        """
        Усечение текста до максимального количества токенов

        Args:
            text: Исходный текст
            max_tokens: Максимальное количество токенов
            tokenizer: Токенизатор

        Returns:
            str: Усеченный текст
        """
        tokens = tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Усечение с сохранением целостности слов
        truncated_tokens = tokens[:max_tokens]

        # Попытка закончить на границе слова
        if max_tokens < len(tokens):
            truncated_tokens = truncated_tokens[:-10]  # Отступ для поиска границы слова

        truncated_text = tokenizer.decode(truncated_tokens)

        # Добавление индикатора усечения
        if len(truncated_text) < len(text):
            truncated_text += "..."

        return truncated_text

    def _get_tokenizer(self, model: str):
        """
        Получение токенизатора для модели

        Args:
            model: Название модели

        Returns:
            Токенизатор
        """
        if model not in self.tokenizers:
            try:
                self.tokenizers[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback на cl100k_base
                self.tokenizers[model] = tiktoken.get_encoding("cl100k_base")

        return self.tokenizers[model]

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Расчет стоимости запроса

        Args:
            model: Модель
            input_tokens: Токены входа
            output_tokens: Токены выхода

        Returns:
            float: Стоимость в долларах
        """
        if model not in self.models:
            return 0.0

        config = self.models[model]
        input_cost = (input_tokens / 1000) * config['cost_per_1k_input']
        output_cost = (output_tokens / 1000) * config['cost_per_1k_output']

        return input_cost + output_cost

    def _update_stats(self, model: str, tokens: int, cost: float, response_time: float):
        """
        Обновление статистики использования

        Args:
            model: Модель
            tokens: Количество токенов
            cost: Стоимость
            response_time: Время ответа
        """
        self.usage_stats['total_requests'] += 1
        self.usage_stats['total_tokens'] += tokens
        self.usage_stats['total_cost'] += cost

        # Обновление средней времени ответа
        current_avg = self.usage_stats['avg_response_time']
        self.usage_stats['avg_response_time'] = (
            current_avg + response_time
        ) / 2

        # Статистика по моделям
        if model not in self.usage_stats['model_usage']:
            self.usage_stats['model_usage'][model] = {
                'requests': 0,
                'tokens': 0,
                'cost': 0.0
            }

        self.usage_stats['model_usage'][model]['requests'] += 1
        self.usage_stats['model_usage'][model]['tokens'] += tokens
        self.usage_stats['model_usage'][model]['cost'] += cost

    def _update_embedding_stats(self, model: str, tokens: int):
        """
        Обновление статистики эмбеддингов

        Args:
            model: Модель эмбеддингов
            tokens: Количество токенов
        """
        self.usage_stats['total_tokens'] += tokens

        # Для эмбеддингов используем фиксированную стоимость
        # text-embedding-ada-002: $0.0001 за 1K токенов
        cost = (tokens / 1000) * 0.0001
        self.usage_stats['total_cost'] += cost

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Получение статистики использования

        Returns:
            Dict[str, Any]: Статистика
        """
        return self.usage_stats.copy()

    async def list_available_models(self) -> List[str]:
        """
        Получение списка доступных моделей

        Returns:
            List[str]: Список моделей
        """
        try:
            if not self.client:
                return list(self.models.keys())

            response = await self.client.models.list()
            api_models = [model.id for model in response.data]

            # Фильтрация поддерживаемых моделей
            supported_models = [model for model in api_models if model in self.models]

            return supported_models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return list(self.models.keys())

    async def validate_api_key(self) -> bool:
        """
        Валидация API ключа

        Returns:
            bool: Валиден ли ключ
        """
        try:
            if not self.client:
                return False

            await self.client.models.list()
            return True

        except Exception:
            return False

    async def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о модели

        Args:
            model: Название модели

        Returns:
            Optional[Dict[str, Any]]: Информация о модели
        """
        if model in self.models:
            return self.models[model].copy()

        try:
            if self.client:
                response = await self.client.models.retrieve(model)
                return {
                    'id': response.id,
                    'object': response.object,
                    'created': response.created,
                    'owned_by': response.owned_by
                }

        except Exception as e:
            logger.error(f"Failed to get model info for {model}: {e}")

        return None

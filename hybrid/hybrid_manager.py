import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .models import (
    ModelProvider, ModelConfig, ModelInstance, ModelCapabilities,
    RoutingDecision, FallbackStrategy, ModelResponse
)

logger = logging.getLogger(__name__)


class HybridManager:
    """
    Гибридный менеджер AI моделей.

    Управляет маршрутизацией запросов между различными AI моделями
    (Qwen, Gemini, OpenAI, локальные модели) на основе:
    - Доступности моделей
    - Стоимости
    - Качества ответов
    - Специализации задач
    """

    def __init__(self, model_configs: Dict[str, ModelConfig]):
        self.model_configs = model_configs
        self.model_instances: Dict[str, ModelInstance] = {}
        self._initialized = False

        # Кэш для rate limiting
        self._request_times: Dict[str, List[float]] = {}

    async def initialize(self) -> bool:
        """Инициализация менеджера"""
        if self._initialized:
            return True

        try:
            # Создание экземпляров моделей
            for config_name, config in self.model_configs.items():
                if config.enabled:
                    instance = ModelInstance(config=config)
                    self.model_instances[config_name] = instance
                    logger.info(f"Initialized model instance: {config_name}")

            self._initialized = True
            logger.info(f"HybridManager initialized with {len(self.model_instances)} models")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize HybridManager: {e}")
            return False

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        force_provider: Optional[ModelProvider] = None,
        user_id: Optional[str] = None,
        fallback_strategy: FallbackStrategy = FallbackStrategy.NEXT_PRIORITY
    ) -> ModelResponse:
        """
        Обработка запроса с автоматической маршрутизацией

        Args:
            query: Текст запроса
            context: Дополнительный контекст
            force_provider: Принудительный выбор провайдера
            user_id: ID пользователя для персонализации
            fallback_strategy: Стратегия fallback при ошибках
        """
        if not await self.initialize():
            return ModelResponse(
                answer="Система не инициализирована",
                model_name="none",
                provider_used=ModelProvider.LOCAL,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                processing_time=0.0,
                error_message="System not initialized"
            )

        start_time = time.time()

        # Анализ запроса
        query_analysis = self._analyze_query(query, context or {})

        # Выбор модели
        routing_decision = await self._select_model(query_analysis, force_provider)

        # Выполнение запроса
        response = await self._execute_with_fallback(
            query, context or {}, routing_decision, fallback_strategy
        )

        response.processing_time = time.time() - start_time

        # Обновление метрик
        if response.model_name in self.model_instances:
            instance = self.model_instances[response.model_name]
            instance.update_metrics(
                response_time=response.processing_time,
                tokens=response.tokens_used,
                cost=response.cost,
                success=response.error_message is None
            )

        return response

    def _analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ запроса для определения требований к модели"""
        analysis = {
            'length': len(query),
            'language': self._detect_language(query),
            'complexity': self._estimate_complexity(query),
            'required_capabilities': self._identify_capabilities(query),
            'urgency': context.get('urgency', 'normal'),
            'domain': context.get('domain', 'general')
        }

        # Определение типа задачи
        if any(word in query.lower() for word in ['напиши', 'создай', 'придумай']):
            analysis['task_type'] = 'creative'
        elif any(word in query.lower() for word in ['рассчитай', 'вычисли', 'формула']):
            analysis['task_type'] = 'mathematical'
        elif any(word in query.lower() for word in ['проанализируй', 'сравни', 'оцени']):
            analysis['task_type'] = 'analytical'
        elif any(word in query.lower() for word in ['переведи', 'translate']):
            analysis['task_type'] = 'translation'
        else:
            analysis['task_type'] = 'general'

        return analysis

    def _detect_language(self, text: str) -> str:
        """Определение языка текста"""
        # Простая эвристика
        cyrillic_chars = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        total_chars = len(text.replace(' ', ''))

        if total_chars == 0:
            return 'unknown'

        cyrillic_ratio = cyrillic_chars / total_chars
        return 'ru' if cyrillic_ratio > 0.3 else 'en'

    def _estimate_complexity(self, query: str) -> str:
        """Оценка сложности запроса"""
        words = len(query.split())
        sentences = len([s for s in query.split('.') if s.strip()])

        if words > 50 or sentences > 3:
            return 'high'
        elif words > 20 or sentences > 1:
            return 'medium'
        else:
            return 'low'

    def _identify_capabilities(self, query: str) -> List[ModelCapabilities]:
        """Определение требуемых возможностей"""
        capabilities = []
        query_lower = query.lower()

        if any(word in query_lower for word in ['код', 'программа', 'функция']):
            capabilities.append(ModelCapabilities.CODE_GENERATION)

        if any(word in query_lower for word in ['рассчитай', 'вычисли', 'формула']):
            capabilities.append(ModelCapabilities.MATHEMATICS)

        if any(word in query_lower for word in ['напиши', 'расскажи', 'придумай']):
            capabilities.append(ModelCapabilities.CREATIVE_WRITING)

        if any(word in query_lower for word in ['проанализируй', 'сравни']):
            capabilities.append(ModelCapabilities.ANALYSIS)

        if any(word in query_lower for word in ['переведи', 'translate']):
            capabilities.append(ModelCapabilities.TRANSLATION)

        # По умолчанию добавляем генерацию текста
        if not capabilities:
            capabilities.append(ModelCapabilities.TEXT_GENERATION)

        return capabilities

    async def _select_model(
        self,
        query_analysis: Dict[str, Any],
        force_provider: Optional[ModelProvider] = None
    ) -> RoutingDecision:
        """Выбор оптимальной модели"""
        candidates = []

        for instance_name, instance in self.model_instances.items():
            if not instance.is_healthy():
                continue

            # Проверка принудительного провайдера
            if force_provider and instance.config.provider != force_provider:
                continue

            # Оценка пригодности модели
            score = self._calculate_model_score(instance, query_analysis)
            candidates.append((instance_name, instance, score))

        if not candidates:
            # Fallback на любую доступную модель
            healthy_instances = [
                (name, inst) for name, inst in self.model_instances.items()
                if inst.is_healthy()
            ]
            if healthy_instances:
                name, inst = healthy_instances[0]
                return RoutingDecision(
                    selected_provider=inst.config.provider,
                    selected_model=name,
                    confidence=0.3,
                    reasoning="No optimal model found, using fallback",
                    estimated_cost=inst.config.cost_per_token * 100,  # Примерная оценка
                    estimated_time=inst.metrics.average_response_time or 5.0
                )
            else:
                raise ValueError("No healthy models available")

        # Выбор лучшей модели
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_name, best_instance, best_score = candidates[0]

        # Формирование альтернатив
        alternatives = [
            {
                'model': name,
                'provider': inst.config.provider.value,
                'score': score,
                'cost': inst.config.cost_per_token,
                'latency': inst.metrics.average_response_time
            }
            for name, inst, score in candidates[1:3]  # Топ-3 альтернативы
        ]

        return RoutingDecision(
            selected_provider=best_instance.config.provider,
            selected_model=best_name,
            confidence=min(1.0, best_score / 100.0),
            reasoning=self._generate_selection_reasoning(best_instance, query_analysis),
            alternatives=alternatives,
            estimated_cost=best_instance.config.cost_per_token * len(query_analysis.get('query', '').split()) * 2,
            estimated_time=best_instance.metrics.average_response_time or 3.0
        )

    def _calculate_model_score(self, instance: ModelInstance, analysis: Dict[str, Any]) -> float:
        """Расчет оценки пригодности модели"""
        score = instance.get_effective_priority() * 20  # Базовый приоритет

        # Бонусы за соответствие требованиям
        required_caps = set(analysis.get('required_capabilities', []))
        model_caps = set(instance.config.capabilities)

        if required_caps.issubset(model_caps):
            score += 30  # Полное соответствие
        else:
            overlap = len(required_caps.intersection(model_caps))
            score += overlap * 10  # Частичное соответствие

        # Бонус за язык
        if analysis.get('language') in instance.constraints.supported_languages:
            score += 15

        # Штраф за стоимость
        cost_penalty = instance.config.cost_per_token * 1000
        score -= min(cost_penalty, 20)

        # Бонус за производительность
        if instance.metrics.average_response_time < 2.0:
            score += 10
        elif instance.metrics.average_response_time > 10.0:
            score -= 10

        # Бонус за успешность
        success_bonus = (instance.metrics.success_rate - 0.5) * 20
        score += success_bonus

        return max(0, score)

    def _generate_selection_reasoning(self, instance: ModelInstance, analysis: Dict[str, Any]) -> str:
        """Генерация объяснения выбора модели"""
        reasons = []

        if instance.config.capabilities:
            reasons.append(f"Поддерживает: {', '.join([c.value for c in instance.config.capabilities])}")

        if analysis.get('language') in instance.constraints.supported_languages:
            reasons.append(f"Поддерживает язык: {analysis['language']}")

        if instance.metrics.average_response_time:
            reasons.append(f"Среднее время ответа: {instance.metrics.average_response_time:.1f}с")

        if instance.config.cost_per_token > 0:
            reasons.append(f"Стоимость: ${instance.config.cost_per_token:.6f}/token")

        return "; ".join(reasons)

    async def _execute_with_fallback(
        self,
        query: str,
        context: Dict[str, Any],
        routing_decision: RoutingDecision,
        fallback_strategy: FallbackStrategy
    ) -> ModelResponse:
        """Выполнение запроса с fallback"""
        primary_model = routing_decision.selected_model

        # Попытка выполнения на основной модели
        response = await self._execute_on_model(primary_model, query, context)

        if response.error_message is None:
            return response

        # Fallback если основная модель failed
        logger.warning(f"Primary model {primary_model} failed, trying fallback")

        fallback_models = self._get_fallback_models(
            primary_model, fallback_strategy, routing_decision.alternatives
        )

        for fallback_model in fallback_models:
            try:
                fallback_response = await self._execute_on_model(fallback_model, query, context)
                if fallback_response.error_message is None:
                    fallback_response.used_fallback = True
                    return fallback_response
            except Exception as e:
                logger.error(f"Fallback model {fallback_model} also failed: {e}")
                continue

        # Все модели failed
        return ModelResponse(
            answer="Извините, все доступные модели временно недоступны",
            model_name="none",
            provider_used=ModelProvider.LOCAL,
            confidence=0.0,
            tokens_used=0,
            cost=0.0,
            processing_time=time.time() - time.time(),  # Будет установлено выше
            used_fallback=True,
            error_message="All models failed"
        )

    def _get_fallback_models(
        self,
        failed_model: str,
        strategy: FallbackStrategy,
        alternatives: List[Dict[str, Any]]
    ) -> List[str]:
        """Получение списка моделей для fallback"""
        if strategy == FallbackStrategy.NEXT_PRIORITY:
            # Сортировка по приоритету
            available_models = [
                name for name, inst in self.model_instances.items()
                if inst.is_healthy() and name != failed_model
            ]
            return sorted(
                available_models,
                key=lambda x: self.model_instances[x].get_effective_priority(),
                reverse=True
            )

        elif strategy == FallbackStrategy.SAME_PROVIDER:
            # Модели того же провайдера
            failed_provider = self.model_instances[failed_model].config.provider
            return [
                name for name, inst in self.model_instances.items()
                if inst.is_healthy() and inst.config.provider == failed_provider and name != failed_model
            ]

        elif strategy == FallbackStrategy.CHEAPEST_AVAILABLE:
            # Самые дешевые модели
            available_models = [
                (name, inst.config.cost_per_token) for name, inst in self.model_instances.items()
                if inst.is_healthy() and name != failed_model
            ]
            return [name for name, _ in sorted(available_models, key=lambda x: x[1])]

        elif strategy == FallbackStrategy.FASTEST_AVAILABLE:
            # Самые быстрые модели
            available_models = [
                (name, inst.metrics.average_response_time or 10.0)
                for name, inst in self.model_instances.items()
                if inst.is_healthy() and name != failed_model
            ]
            return [name for name, _ in sorted(available_models, key=lambda x: x[1])]

        # По умолчанию - альтернативы из routing decision
        return [alt['model'] for alt in alternatives if alt['model'] in self.model_instances]

    async def _execute_on_model(
        self,
        model_name: str,
        query: str,
        context: Dict[str, Any]
    ) -> ModelResponse:
        """Выполнение запроса на конкретной модели"""
        if model_name not in self.model_instances:
            return ModelResponse(
                answer="",
                model_name=model_name,
                provider_used=ModelProvider.LOCAL,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                processing_time=0.0,
                error_message=f"Model {model_name} not found"
            )

        instance = self.model_instances[model_name]

        # Проверка rate limiting
        if not self._check_rate_limit(model_name):
            return ModelResponse(
                answer="",
                model_name=model_name,
                provider_used=instance.config.provider,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                processing_time=0.0,
                error_message="Rate limit exceeded"
            )

        try:
            # Вызов соответствующего клиента
            if instance.config.provider == ModelProvider.QWEN:
                return await self._call_qwen(instance, query, context)
            elif instance.config.provider == ModelProvider.GEMINI:
                return await self._call_gemini(instance, query, context)
            elif instance.config.provider == ModelProvider.OPENAI:
                return await self._call_openai(instance, query, context)
            elif instance.config.provider == ModelProvider.LOCAL:
                return await self._call_local(instance, query, context)
            else:
                return ModelResponse(
                    answer="",
                    model_name=model_name,
                    provider_used=instance.config.provider,
                    confidence=0.0,
                    tokens_used=0,
                    cost=0.0,
                    processing_time=0.0,
                    error_message=f"Unsupported provider: {instance.config.provider}"
                )

        except Exception as e:
            logger.error(f"Error calling model {model_name}: {e}")
            return ModelResponse(
                answer="",
                model_name=model_name,
                provider_used=instance.config.provider,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                processing_time=0.0,
                error_message=str(e)
            )

    def _check_rate_limit(self, model_name: str) -> bool:
        """Проверка rate limiting"""
        if model_name not in self.model_instances:
            return False

        instance = self.model_instances[model_name]
        max_requests = instance.config.max_requests_per_minute

        now = time.time()
        window_start = now - 60  # Последняя минута

        # Очистка старых записей
        if model_name in self._request_times:
            self._request_times[model_name] = [
                t for t in self._request_times[model_name] if t > window_start
            ]
        else:
            self._request_times[model_name] = []

        # Проверка лимита
        if len(self._request_times[model_name]) >= max_requests:
            return False

        # Добавление текущего запроса
        self._request_times[model_name].append(now)
        return True

    async def _call_qwen(self, instance: ModelInstance, query: str, context: Dict[str, Any]) -> ModelResponse:
        """Вызов Qwen модели"""
        # Заглушка - в реальности здесь будет интеграция с Qwen API
        await asyncio.sleep(0.1)  # Имитация задержки

        return ModelResponse(
            answer=f"Ответ от Qwen ({instance.config.model_name}): {query}",
            model_name=instance.config.model_name,
            provider_used=ModelProvider.QWEN,
            confidence=0.8,
            tokens_used=len(query.split()) * 2,
            cost=instance.config.cost_per_token * len(query.split()) * 2,
            processing_time=0.5
        )

    async def _call_gemini(self, instance: ModelInstance, query: str, context: Dict[str, Any]) -> ModelResponse:
        """Вызов Gemini модели"""
        # Заглушка - в реальности здесь будет интеграция с Google AI API
        await asyncio.sleep(0.1)

        return ModelResponse(
            answer=f"Ответ от Gemini ({instance.config.model_name}): {query}",
            model_name=instance.config.model_name,
            provider_used=ModelProvider.GEMINI,
            confidence=0.85,
            tokens_used=len(query.split()) * 2,
            cost=instance.config.cost_per_token * len(query.split()) * 2,
            processing_time=0.7
        )

    async def _call_openai(self, instance: ModelInstance, query: str, context: Dict[str, Any]) -> ModelResponse:
        """Вызов OpenAI модели"""
        # Заглушка - в реальности здесь будет интеграция с OpenAI API
        await asyncio.sleep(0.1)

        return ModelResponse(
            answer=f"Ответ от OpenAI ({instance.config.model_name}): {query}",
            model_name=instance.config.model_name,
            provider_used=ModelProvider.OPENAI,
            confidence=0.9,
            tokens_used=len(query.split()) * 2,
            cost=instance.config.cost_per_token * len(query.split()) * 2,
            processing_time=0.8
        )

    async def _call_local(self, instance: ModelInstance, query: str, context: Dict[str, Any]) -> ModelResponse:
        """Вызов локальной модели"""
        # Заглушка - в реальности здесь будет интеграция с локальной моделью
        await asyncio.sleep(0.1)

        return ModelResponse(
            answer=f"Ответ от локальной модели ({instance.config.model_name}): {query}",
            model_name=instance.config.model_name,
            provider_used=ModelProvider.LOCAL,
            confidence=0.7,
            tokens_used=len(query.split()) * 2,
            cost=0.0,  # Локальные модели бесплатны
            processing_time=0.3
        )

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса всех моделей"""
        return {
            'initialized': self._initialized,
            'total_models': len(self.model_instances),
            'healthy_models': sum(1 for inst in self.model_instances.values() if inst.is_healthy()),
            'models': {
                name: {
                    'provider': inst.config.provider.value,
                    'healthy': inst.is_healthy(),
                    'requests_count': inst.metrics.requests_count,
                    'average_response_time': inst.metrics.average_response_time,
                    'success_rate': inst.metrics.success_rate
                }
                for name, inst in self.model_instances.items()
            }
        }

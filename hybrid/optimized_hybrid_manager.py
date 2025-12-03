"""
Оптимизированный гибридный менеджер AI моделей с улучшенным алгоритмом выбора моделей
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import heapq

from .models import (
    ModelProvider, ModelConfig, ModelInstance, ModelCapabilities,
    RoutingDecision, FallbackStrategy, ModelResponse
)

logger = logging.getLogger(__name__)


@dataclass
class ModelScore:
    """Результат оценки модели"""
    model_name: str
    score: float
    reasoning: str
    metadata: Dict[str, Any]


class OptimizedHybridManager:
    """
    Оптимизированный гибридный менеджер AI моделей с улучшенным алгоритмом выбора.
    
    Основные улучшения:
    - Более точная оценка пригодности моделей
    - Асинхронный параллельный анализ
    - Кэширование результатов
    - Улучшенные метрики производительности
    - Более эффективные стратегии фоллбэка
    """

    def __init__(self, model_configs: Dict[str, ModelConfig]):
        self.model_configs = model_configs
        self.model_instances: Dict[str, ModelInstance] = {}
        self._initialized = False

        # Кэш для rate limiting
        self._request_times: Dict[str, List[float]] = {}
        
        # Кэш для анализа запросов
        self._query_analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._max_cache_size = 1000
        
        # Пул потоков для вычислений
        self._executor = ThreadPoolExecutor(max_workers=4)

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
            logger.info(f"OptimizedHybridManager initialized with {len(self.model_instances)} models")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OptimizedHybridManager: {e}")
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

        # Анализ запроса с кэшированием
        query_analysis = await self._analyze_query_cached(query, context or {})

        # Выбор модели
        routing_decision = await self._select_model_optimized(query_analysis, force_provider)

        # Выполнение запроса
        response = await self._execute_with_fallback_optimized(
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

    async def _analyze_query_cached(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ запроса с кэшированием"""
        cache_key = f"{hash(query)}_{hash(str(context))}"
        
        if cache_key in self._query_analysis_cache:
            return self._query_analysis_cache[cache_key]

        analysis = await self._analyze_query_async(query, context)
        
        # Ограничение размера кэша
        if len(self._query_analysis_cache) >= self._max_cache_size:
            # Удаляем старые записи
            oldest_key = next(iter(self._query_analysis_cache))
            del self._query_analysis_cache[oldest_key]
        
        self._query_analysis_cache[cache_key] = analysis
        return analysis

    async def _analyze_query_async(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Асинхронный анализ запроса для определения требований к модели"""
        loop = asyncio.get_event_loop()
        
        # Параллельный анализ различных аспектов запроса
        language_task = loop.run_in_executor(self._executor, self._detect_language, query)
        complexity_task = loop.run_in_executor(self._executor, self._estimate_complexity, query)
        capabilities_task = loop.run_in_executor(self._executor, self._identify_capabilities, query)
        
        language, complexity, capabilities = await asyncio.gather(
            language_task, complexity_task, capabilities_task
        )

        analysis = {
            'length': len(query),
            'language': language,
            'complexity': complexity,
            'required_capabilities': capabilities,
            'urgency': context.get('urgency', 'normal'),
            'domain': context.get('domain', 'general'),
            'timestamp': datetime.now()
        }

        # Определение типа задачи
        query_lower = query.lower()
        if any(word in query_lower for word in ['напиши', 'создай', 'придумай']):
            analysis['task_type'] = 'creative'
        elif any(word in query_lower for word in ['рассчитай', 'вычисли', 'формула']):
            analysis['task_type'] = 'mathematical'
        elif any(word in query_lower for word in ['проанализируй', 'сравни', 'оцени']):
            analysis['task_type'] = 'analytical'
        elif any(word in query_lower for word in ['переведи', 'translate']):
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

        if words > 100 or sentences > 5:
            return 'high'
        elif words > 50 or sentences > 2:
            return 'medium'
        else:
            return 'low'

    def _identify_capabilities(self, query: str) -> List[ModelCapabilities]:
        """Определение требуемых возможностей"""
        capabilities = []
        query_lower = query.lower()

        if any(word in query_lower for word in ['код', 'программа', 'функция', 'python', 'javascript', 'java']):
            capabilities.append(ModelCapabilities.CODE_GENERATION)

        if any(word in query_lower for word in ['рассчитай', 'вычисли', 'формула', 'математик', 'алгебр', 'геометр']):
            capabilities.append(ModelCapabilities.MATHEMATICS)

        if any(word in query_lower for word in ['напиши', 'расскажи', 'придумай', 'сочини', 'истор']):
            capabilities.append(ModelCapabilities.CREATIVE_WRITING)

        if any(word in query_lower for word in ['проанализируй', 'сравни', 'оцен', 'анализ', 'вывод']):
            capabilities.append(ModelCapabilities.ANALYSIS)

        if any(word in query_lower for word in ['переведи', 'translate', 'перевод']):
            capabilities.append(ModelCapabilities.TRANSLATION)

        # По умолчанию добавляем генерацию текста
        if not capabilities:
            capabilities.append(ModelCapabilities.TEXT_GENERATION)

        return capabilities

    async def _select_model_optimized(
        self,
        query_analysis: Dict[str, Any],
        force_provider: Optional[ModelProvider] = None
    ) -> RoutingDecision:
        """Оптимизированный выбор оптимальной модели"""
        # Фильтрация моделей
        candidates = await self._get_healthy_candidates(force_provider)
        
        if not candidates:
            # Fallback на любую доступную модель
            return await self._get_fallback_routing_decision()

        # Асинхронный параллельный расчет оценок
        score_tasks = [
            self._calculate_model_score_optimized(instance, query_analysis, instance_name)
            for instance_name, instance in candidates
        ]
        
        scores = await asyncio.gather(*score_tasks)
        
        # Фильтрация и сортировка
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            return await self._get_fallback_routing_decision()
        
        # Сортировка по оценке (лучшие первые)
        valid_scores.sort(key=lambda x: x.score, reverse=True)
        
        best_score = valid_scores[0]
        best_instance = self.model_instances[best_score.model_name]

        # Формирование альтернатив (топ-3)
        alternatives = []
        for score_obj in valid_scores[1:4]:  # Топ-3 альтернативы
            instance = self.model_instances[score_obj.model_name]
            alternatives.append({
                'model': score_obj.model_name,
                'provider': instance.config.provider.value,
                'score': score_obj.score,
                'cost': instance.config.cost_per_token,
                'latency': instance.metrics.average_response_time,
                'reasoning': score_obj.reasoning
            })

        return RoutingDecision(
            selected_provider=best_instance.config.provider,
            selected_model=best_score.model_name,
            confidence=min(1.0, best_score.score / 100.0),
            reasoning=best_score.reasoning,
            alternatives=alternatives,
            estimated_cost=best_instance.config.cost_per_token * len(query_analysis.get('query', '').split()) * 2,
            estimated_time=best_instance.metrics.average_response_time or 3.0
        )

    async def _get_healthy_candidates(self, force_provider: Optional[ModelProvider] = None) -> List[Tuple[str, ModelInstance]]:
        """Получение списка здоровых кандидатов"""
        candidates = []
        for instance_name, instance in self.model_instances.items():
            if not instance.is_healthy():
                continue

            # Проверка принудительного провайдера
            if force_provider and instance.config.provider != force_provider:
                continue

            candidates.append((instance_name, instance))
        
        return candidates

    async def _get_fallback_routing_decision(self) -> RoutingDecision:
        """Получение решения о фоллбэке"""
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
                estimated_cost=inst.config.cost_per_token * 10,  # Примерная оценка
                estimated_time=inst.metrics.average_response_time or 5.0
            )
        else:
            raise ValueError("No healthy models available")

    async def _calculate_model_score_optimized(
        self,
        instance: ModelInstance,
        analysis: Dict[str, Any],
        model_name: str
    ) -> Optional[ModelScore]:
        """Оптимизированный расчет оценки пригодности модели"""
        try:
            # Взвешенная оценка по нескольким критериям
            weights = {
                'priority': 0.25,
                'capabilities': 0.25,
                'language': 0.15,
                'cost': 0.15,
                'performance': 0.10,
                'success_rate': 0.10
            }

            # Базовый приоритет
            priority_score = instance.get_effective_priority() * weights['priority'] * 10

            # Оценка по соответствию возможностей
            required_caps = set(analysis.get('required_capabilities', []))
            model_caps = set(instance.config.capabilities)
            
            if required_caps:
                if required_caps.issubset(model_caps):
                    capability_score = weights['capabilities'] * 100  # Полное соответствие
                else:
                    overlap = len(required_caps.intersection(model_caps))
                    total_required = len(required_caps)
                    capability_score = (overlap / total_required) * weights['capabilities'] * 100
            else:
                capability_score = weights['capabilities'] * 50  # Нейтральная оценка

            # Оценка по языку
            language_score = 0
            if analysis.get('language') in instance.constraints.supported_languages:
                language_score = weights['language'] * 10
            else:
                # Проверяем, есть ли английский как резерв
                if 'en' in instance.constraints.supported_languages:
                    language_score = weights['language'] * 50

            # Оценка по стоимости (обратная)
            cost_score = max(0, (1 - instance.config.cost_per_token * 1000) * weights['cost'] * 100)

            # Оценка по производительности
            avg_response_time = instance.metrics.average_response_time
            if avg_response_time:
                if avg_response_time < 1.0:
                    performance_score = weights['performance'] * 100
                elif avg_response_time < 3.0:
                    performance_score = weights['performance'] * 75
                elif avg_response_time < 5.0:
                    performance_score = weights['performance'] * 50
                else:
                    performance_score = weights['performance'] * 25
            else:
                performance_score = weights['performance'] * 50  # Нейтральная оценка

            # Оценка по успешности
            success_rate = instance.metrics.success_rate
            success_score = success_rate * weights['success_rate'] * 100

            # Общий скор
            total_score = priority_score + capability_score + language_score + cost_score + performance_score + success_score

            # Формирование объяснения
            reasons = []
            if required_caps.issubset(model_caps):
                reasons.append(f"Полное совпадение возможностей: {len(required_caps)} из {len(required_caps)}")
            else:
                overlap = len(required_caps.intersection(model_caps))
                reasons.append(f"Частичное совпадение возможностей: {overlap} из {len(required_caps)}")
            
            if analysis.get('language') in instance.constraints.supported_languages:
                reasons.append(f"Поддержка языка: {analysis['language']}")
            
            if avg_response_time:
                reasons.append(f"Среднее время ответа: {avg_response_time:.2f}с")
            
            if instance.config.cost_per_token > 0:
                reasons.append(f"Стоимость: ${instance.config.cost_per_token:.6f}/token")

            return ModelScore(
                model_name=model_name,
                score=total_score,
                reasoning="; ".join(reasons),
                metadata={
                    'priority_score': priority_score,
                    'capability_score': capability_score,
                    'language_score': language_score,
                    'cost_score': cost_score,
                    'performance_score': performance_score,
                    'success_score': success_score
                }
            )
        except Exception as e:
            logger.error(f"Error calculating score for model {model_name}: {e}")
            return None

    async def _execute_with_fallback_optimized(
        self,
        query: str,
        context: Dict[str, Any],
        routing_decision: RoutingDecision,
        fallback_strategy: FallbackStrategy
    ) -> ModelResponse:
        """Выполнение запроса с оптимизированным фоллбэком"""
        primary_model = routing_decision.selected_model

        # Попытка выполнения на основной модели
        response = await self._execute_on_model(primary_model, query, context)

        if response.error_message is None:
            return response

        # Fallback если основная модель failed
        logger.warning(f"Primary model {primary_model} failed, trying fallback")

        # Асинхронный параллельный фоллбэк (попробовать несколько моделей одновременно)
        fallback_models = await self._get_fallback_models_optimized(
            primary_model, fallback_strategy, routing_decision.alternatives
        )
        
        if not fallback_models:
            return ModelResponse(
                answer="Извините, все доступные модели временно недоступны",
                model_name="none",
                provider_used=ModelProvider.LOCAL,
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                processing_time=time.time() - time.time(),
                used_fallback=True,
                error_message="All models failed"
            )

        # Попробовать фоллбэк-модели параллельно, вернуть первый успешный результат
        tasks = [
            self._execute_on_model(fallback_model, query, context)
            for fallback_model in fallback_models[:3]  # Ограничение на 3 модели для фоллбэка
        ]
        
        for coro in asyncio.as_completed(tasks):
            fallback_response = await coro
            if fallback_response.error_message is None:
                fallback_response.used_fallback = True
                return fallback_response

        # Все модели failed
        return ModelResponse(
            answer="Извините, все доступные модели временно недоступны",
            model_name="none",
            provider_used=ModelProvider.LOCAL,
            confidence=0.0,
            tokens_used=0,
            cost=0.0,
            processing_time=time.time() - time.time(),
            used_fallback=True,
            error_message="All models failed"
        )

    async def _get_fallback_models_optimized(
        self,
        failed_model: str,
        strategy: FallbackStrategy,
        alternatives: List[Dict[str, Any]]
    ) -> List[str]:
        """Получение списка моделей для фоллбэка с оптимизацией"""
        loop = asyncio.get_event_loop()
        
        if strategy == FallbackStrategy.NEXT_PRIORITY:
            # Сортировка по приоритету
            available_models = [
                name for name, inst in self.model_instances.items()
                if inst.is_healthy() and name != failed_model
            ]
            # Сортировка с использованием пула потоков
            return await loop.run_in_executor(
                self._executor,
                lambda: sorted(
                    available_models,
                    key=lambda x: self.model_instances[x].get_effective_priority(),
                    reverse=True
                )
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
            return await loop.run_in_executor(
                self._executor,
                lambda: [name for name, _ in sorted(available_models, key=lambda x: x[1])]
            )

        elif strategy == FallbackStrategy.FASTEST_AVAILABLE:
            # Самые быстрые модели
            available_models = [
                (name, inst.metrics.average_response_time or 10.0)
                for name, inst in self.model_instances.items()
                if inst.is_healthy() and name != failed_model
            ]
            return await loop.run_in_executor(
                self._executor,
                lambda: [name for name, _ in sorted(available_models, key=lambda x: x[1])]
            )

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
                    'success_rate': inst.metrics.success_rate,
                    'effective_priority': inst.get_effective_priority()
                }
                for name, inst in self.model_instances.items()
            },
            'cache_stats': {
                'query_analysis_cache_size': len(self._query_analysis_cache),
                'max_cache_size': self._max_cache_size
            }
        }

    def get_model_recommendations(self, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Получение рекомендаций моделей для заданного анализа запроса
        """
        recommendations = []
        
        for model_name, instance in self.model_instances.items():
            if not instance.is_healthy():
                continue
                
            score_obj = asyncio.run(
                self._calculate_model_score_optimized(instance, query_analysis, model_name)
            )
            
            if score_obj:
                recommendations.append({
                    'model_name': model_name,
                    'provider': instance.config.provider.value,
                    'score': score_obj.score,
                    'reasoning': score_obj.reasoning,
                    'metadata': score_obj.metadata
                })
        
        # Сортировка по оценке
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations
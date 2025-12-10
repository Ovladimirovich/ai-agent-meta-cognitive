import logging
import time
from typing import Dict, Any, Optional

from datetime import datetime
from .models import (
    AgentConfig, AgentRequest, AgentResponse, AgentState,
    TaskComplexity, QueryAnalysis, ReasoningStep
)
from agent.tools.tool_orchestrator import ToolOrchestrator
from ..tools.query_analyzer import QueryAnalyzer
from ..memory.memory_manager import MemoryManager
from ..self_awareness.state_manager import StateManager
from ..self_awareness.confidence_calculator import ConfidenceCalculator
from ..self_awareness.reasoning_tracer import ReasoningTracer
from integrations.llm_client import create_llm_client, LLMProvider
from integrations.circuit_breaker import circuit_breaker_decorator, CircuitBreakerConfig

logger = logging.getLogger("AgentCore")


class AgentCore:
    """
    Центральный компонент AI агента с мета-познанием.

    Координирует все компоненты агента: анализ запросов, выбор инструментов,
    исполнение задач, управление памятью и мета-познавательные функции.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

        # Инициализация компонентов
        self.state_manager = StateManager()
        self.tool_orchestrator = ToolOrchestrator(config.agent_tool_timeout)
        self.memory_manager = MemoryManager(config.agent_max_memory_entries) if config.enable_memory else None
        self.query_analyzer = QueryAnalyzer()
        self.confidence_calculator = ConfidenceCalculator()
        self.reasoning_tracer = ReasoningTracer() if config.agent_enable_reasoning_trace else None

        # Инициализация LLM клиента (ленивая загрузка)
        self.llm_client = None
        self.llm_initialized = False

        # Инициализация менеджера предварительной загрузки
        self.preload_manager = None
        if self.memory_manager:
            try:
                from .preload_manager import PreloadManager
                self.preload_manager = PreloadManager(self.memory_manager, config)
            except ImportError:
                logger.warning("PreloadManager not available, skipping initialization")
                self.preload_manager = None

        # Мета-познавательные компоненты
        self.reasoning_trace: list[ReasoningStep] = []
        self.confidence_score: float = 0.0
        self.task_complexity: TaskComplexity = TaskComplexity.SIMPLE

        # Метрики производительности
        self.requests_processed = 0
        self.total_execution_time = 0.0
        self.error_count = 0
        self.last_execution_times = []
        self.tool_usage_stats = {}

        logger.info("🚀 AgentCore initialized")

    async def _init_llm_client_async(self):
        """Асинхронная инициализация LLM клиента"""
        import os

        # 🔥 ПРИОРИТЕТ: Ollama всегда проверяется первым (бесплатно, локально)
        try:
            # Получаем URL Ollama из переменной окружения или используем значение по умолчанию
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            self.llm_client = await create_llm_client(
                provider="ollama",
                api_key=None,  # Ollama не требует API ключа
                model="gemma3:1b",
                temperature=0.7,
                max_tokens=1000,
                base_url=ollama_url  # Передаем URL отдельно
            )
            logger.info("✅ LLM client initialized with Ollama (Gemma3) - FREE local AI!")
            return

        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}. Trying cloud providers...")

        # Если Ollama не работает, проверяем облачные провайдеры
        api_configs = [
            ("MISTRAL_API_KEY", "mistral", "mistral-small"),  # Бесплатный tier
            ("GOOGLE_API_KEY", "google", "gemini-1.5-flash"),  # Бесплатный tier
            ("TOGETHER_API_KEY", "together", "mistralai/Mistral-7B-Instruct-v0.1"),  # Некоторые бесплатны
            ("OPENAI_API_KEY", "openai", "gpt-3.5-turbo"),
            ("ANTHROPIC_API_KEY", "anthropic", "claude-3-haiku-20240307"),
            ("GROK_API_KEY", "grok", "grok-1"),
        ]

        for env_var, provider, model in api_configs:
            api_key = os.getenv(env_var)
            if api_key:
                try:
                    self.llm_client = await create_llm_client(
                        provider=provider,
                        api_key=api_key,
                        model=model,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    logger.info(f"✅ LLM client initialized with {provider} ({model})")
                    break

                except Exception as e:
                    logger.warning(f"Failed to initialize {provider} client: {e}")
                    continue

        if not self.llm_client:
            logger.warning("⚠️ No LLM API keys found. Using fallback responses.")
            logger.info("💡 To enable AI responses, set one of these FREE options:")
            logger.info("   🔥 OLLAMA_URL - Ollama (FREE, local, best choice!)")
            logger.info("   MISTRAL_API_KEY - Mistral AI (FREE tier available)")
            logger.info("   GOOGLE_API_KEY - Google Gemini (FREE tier)")
            logger.info("   TOGETHER_API_KEY - Together AI (some FREE models)")
            logger.info("   Or paid options: OPENAI_API_KEY, ANTHROPIC_API_KEY, GROK_API_KEY")

    def _init_llm_client(self):
        """Синхронная обертка для асинхронной инициализации"""
        import asyncio
        try:
            asyncio.run(self._init_llm_client_async())
        except RuntimeError:
            # Если уже есть event loop, используем run_coroutine_threadsafe
            import threading
            if not hasattr(self, '_loop') or not self._loop.is_running():
                # Если нет активного цикла, создаем новый поток
                def run_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._init_llm_client_async())
                    finally:
                        loop.close()

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

    @circuit_breaker_decorator("agent_core_process", CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=120.0,
        timeout=60.0,
        name="agent_core_process"
    ))
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Обработка запроса с мета-познанием.

        Процесс включает:
        1. Анализ запроса
        2. Определение сложности и стратегии
        3. Исполнение с трассировкой
        4. Оценка уверенности
        5. Сохранение в память
        """
        start_time = time.time()

        try:
            # Проверка кэша перед обработкой
            cache_key = self._get_cache_key(request.query, request.context)
            cached_result = await self._get_cached_response(cache_key)
            if cached_result:
                logger.info("✅ Returning cached response")
                execution_time = time.time() - start_time
                return AgentResponse(
                    result=cached_result,
                    confidence=0.95,  # Высокий уровень уверенности для кэшированных ответов
                    reasoning_trace=[],
                    execution_time=execution_time
                )

            # Попытка получить предварительно загруженные данные
            preloaded_data = await self._get_preloaded_data(request.query)
            if preloaded_data:
                logger.info("✅ Using preloaded data for faster response")
                execution_time = time.time() - start_time
                return AgentResponse(
                    result=preloaded_data,
                    confidence=0.9,
                    reasoning_trace=[],
                    execution_time=execution_time
                )

            # Переход в состояние анализа
            self.state_manager.transition_to(AgentState.ANALYZING, "Starting request analysis")

            # Запуск менеджера предварительной загрузки при первом запросе
            if self.preload_manager and not self.preload_manager._running:
                await self.preload_manager.start_preloading()

            # Анализ запроса
            analysis = await self.query_analyzer.analyze(request)
            self._add_reasoning_step("analysis", "Query analyzed", {
                "intent": analysis.intent,
                "complexity": analysis.complexity.value,
                "required_tools": analysis.required_tools
            })

            # Определение сложности и стратегии
            self.task_complexity = analysis.complexity
            strategy = self._select_strategy(analysis)

            self._add_reasoning_step("strategy_selection", f"Selected strategy: {strategy}", {
                "complexity": self.task_complexity.value,
                "strategy": strategy
            })

            # Переход в состояние исполнения
            self.state_manager.transition_to(AgentState.EXECUTING, f"Executing with strategy: {strategy}")

            # Исполнение с трассировкой
            result = await self._execute_with_trace(request, strategy, analysis)

            # Оценка уверенности
            self.confidence_score = self.confidence_calculator.calculate(result, analysis)

            # Сохранение в память
            if self.memory_manager:
                await self.memory_manager.store_episodic_memory({
                    'request': request,
                    'analysis': analysis,
                    'strategy': strategy,
                    'result': result,
                    'confidence': self.confidence_score,
                    'execution_time': time.time() - start_time,
                    'timestamp': datetime.now()
                })

            execution_time = time.time() - start_time
            # Гарантируем минимальное время выполнения для точности измерений
            execution_time = max(execution_time, 0.001)

            # Кэширование результата
            await self._cache_response(cache_key, result)

            # Обновление метрик
            self.requests_processed += 1
            self.total_execution_time += execution_time
            self.last_execution_times.append(execution_time)
            if len(self.last_execution_times) > 100:  # Ограничиваем историю
                self.last_execution_times.pop(0)

            # Переход в завершенное состояние
            self.state_manager.transition_to(AgentState.COMPLETED, "Request processed successfully")

            self._add_reasoning_step("completion", "Request completed", {
                "confidence": self.confidence_score,
                "execution_time": execution_time
            })

            return AgentResponse(
                result=result,
                confidence=self.confidence_score,
                reasoning_trace=[step.dict() for step in self.reasoning_trace],
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"❌ Request processing failed: {e}")

            # Обновление счетчика ошибок
            self.error_count += 1

            # Безопасный переход в состояние ошибки (не кидаем исключение при повторных ошибках)
            try:
                self.state_manager.transition_to(AgentState.ERROR, str(e))
            except Exception as transition_error:
                logger.warning(f"Could not transition to ERROR state: {transition_error}")
                # Сбрасываем состояние в IDLE для возможности продолжения работы
                try:
                    self.state_manager.transition_to(AgentState.IDLE, "Reset after error")
                except Exception as reset_error:
                    logger.error(f"Could not reset state: {reset_error}")

            # Возврат ответа с низкой уверенностью
            execution_time = time.time() - start_time
            return AgentResponse(
                result=f"Извините, произошла ошибка при обработке запроса: {str(e)}",
                confidence=0.1,
                reasoning_trace=[step.dict() for step in self.reasoning_trace],
                execution_time=execution_time,
                metadata={"error": str(e)}
            )

    def _select_strategy(self, analysis: QueryAnalysis) -> str:
        """Выбор стратегии исполнения на основе анализа"""
        # Для простых запросов всегда используем прямой ответ через LLM
        if analysis.complexity == TaskComplexity.SIMPLE:
            return "direct_response"
        elif analysis.intent in ["greeting", "casual_conversation"]:
            # Приветствия и casual разговор всегда через LLM
            return "direct_response"
        elif analysis.complexity == TaskComplexity.MEDIUM:
            return "tool_assisted"
        else:  # COMPLEX
            return "multi_tool_pipeline"

    def _map_tool_names(self, tool_names: list[str]) -> list[str]:
        """Преобразование имен инструментов из query_analyzer в имена tool_orchestrator"""
        name_mapping = {
            "rag_search": "rag",
            "data_analyzer": "analytics",
            "code_executor": "hybrid_models",
            "general_assistant": "hybrid_models",
            "reasoning_engine": "hybrid_models"
        }

        return [name_mapping.get(name, name) for name in tool_names]

    async def _execute_with_trace(self, request: AgentRequest, strategy: str, analysis: QueryAnalysis) -> Any:
        """Исполнение запроса с трассировкой"""
        if strategy == "direct_response":
            # Простой ответ без инструментов
            result = await self._generate_direct_response(request)
            self._add_reasoning_step("execution", "Direct response generated", {"strategy": strategy})

        elif strategy == "tool_assisted":
            # Использование одного инструмента
            tool_names = self._map_tool_names(analysis.required_tools[:1])
            tool_results = await self.tool_orchestrator.execute_tools(tool_names, {
                "request": request,
                "analysis": analysis
            })
            result = self._process_tool_results(tool_results)
            self._add_reasoning_step("execution", f"Tool executed: {analysis.required_tools[0]}", {
                "strategy": strategy,
                "tools_used": analysis.required_tools[:1]
            })

        elif strategy == "multi_tool_pipeline":
            # Использование нескольких инструментов
            tool_names = self._map_tool_names(analysis.required_tools)
            tool_results = await self.tool_orchestrator.execute_tools(tool_names, {
                "request": request,
                "analysis": analysis
            })
            result = self._process_tool_results(tool_results)
            self._add_reasoning_step("execution", f"Multiple tools executed: {analysis.required_tools}", {
                "strategy": strategy,
                "tools_used": analysis.required_tools
            })

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return result

    async def _generate_direct_response(self, request: AgentRequest) -> str:
        """Генерация ответа с использованием LLM для простых запросов"""
        # Ленивая инициализация LLM клиента при первом использовании
        if not self.llm_initialized:
            try:
                self._init_llm_client()
                self.llm_initialized = True
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}")

        if self.llm_client:
            try:
                # Системное сообщение для агента с мета-познанием
                system_message = (
                    "Ты - AI агент с мета-познанием. Ты можешь анализировать свои мысли, "
                    "оценивать уверенность в ответах и использовать различные инструменты для решения задач. "
                    "Отвечай на русском языке. Будь полезным, дружелюбным и точным. "
                    "Если вопрос требует использования инструментов, предложи их применение."
                )

                # Генерация ответа через LLM
                response = await self.llm_client.generate_response(
                    prompt=request.query,
                    system_message=system_message
                )

                logger.info(f"LLM response generated - confidence: {response['confidence']}, "
                           f"model: {response['model']}, provider: {response['provider']}")

                return response["content"]

            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                # Fallback к статическим ответам при ошибке LLM
                return await self._generate_fallback_response(request)
        else:
            # Fallback если LLM не инициализирован
            return await self._generate_fallback_response(request)

    def _get_cache_key(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Генерация ключа для кэширования ответов"""
        import hashlib
        cache_input = f"{query}_{str(context)}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    async def _get_preloaded_data(self, query: str) -> Optional[str]:
        """Получение предварительно загруженных данных"""
        # Проверяем, есть ли в памяти предварительно загруженные данные
        if self.memory_manager:
            # Ищем в семантической памяти предварительно загруженные часто используемые данные
            preloaded_memories = await self.memory_manager.retrieve_semantic_memory(query, limit=3)
            for memory in preloaded_memories:
                if hasattr(memory, 'metadata') and memory.metadata.get('preload_priority', 0) > 0:
                    return memory.content

        # Также проверяем через менеджер предварительной загрузки
        if self.preload_manager:
            preloaded_data = await self.preload_manager.get_preloaded_data(query)
            if preloaded_data:
                return preloaded_data

        return None

    async def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Получение ответа из кэша"""
        # Fallback к памяти
        if self.memory_manager:
            recent_memories = self.memory_manager.retrieve_episodic_memory(limit=5)
            for memory in recent_memories:
                if hasattr(memory, 'request') and hasattr(memory.request, 'query'):
                    if memory.request.query == cache_key:
                        return memory.result
        return None

    async def _cache_response(self, cache_key: str, response: str):
        """Сохранение ответа в кэш"""
        # Fallback к памяти
        if self.memory_manager and hasattr(self.memory_manager, 'store_working_memory'):
            try:
                await self.memory_manager.store_working_memory(cache_key, response, ttl_seconds=300)
            except Exception as e:
                logger.warning(f"Failed to cache response in memory: {e}")

    async def _generate_fallback_response(self, request: AgentRequest) -> str:
        """Резервная генерация ответа без LLM"""
        query_lower = request.query.lower()

        # Простые паттерны для fallback
        if any(word in query_lower for word in ["привет", "здравствуй", "добрый день", "добрый вечер", "доброе утро", "хай", "hello", "hi"]):
            return "Привет! Я AI агент с мета-познанием. Я готов помогать с различными задачами. Чем могу быть полезен?"

        elif any(word in query_lower for word in ["как дела", "как поживаешь", "как ты", "how are you"]):
            return "У меня все отлично! Я постоянно обучаюсь и совершенствуюсь. А как у вас дела?"

        elif any(word in query_lower for word in ["что ты умеешь", "что ты можешь", "твои возможности", "what can you do"]):
            return ("Я AI агент с мета-познанием. Мои возможности включают:\n"
                   "• Анализ и обработка текстовых запросов\n"
                   "• Использование различных инструментов для решения задач\n"
                   "• Самооценка уверенности в ответах\n"
                   "• Обучение на основе опыта\n"
                   "• Работа с памятью и контекстом\n\n"
                   "Задайте конкретный вопрос, и я постараюсь помочь!")

        elif any(word in query_lower for word in ["спасибо", "благодарю", "thanks", "thank you"]):
            return "Пожалуйста! Всегда рад помочь. Если возникнут еще вопросы, обращайтесь."

        elif any(word in query_lower for word in ["пока", "до свидания", "bye", "goodbye"]):
            return "До свидания! Было приятно пообщаться. Возвращайтесь, когда понадобится помощь."

        else:
            # Универсальный ответ для неизвестных запросов
            responses = [
                f"Я получил ваш запрос: '{request.query}'. Дайте мне подумать...",
                f"Интересный вопрос: '{request.query}'. Для более точного ответа может потребоваться использование специальных инструментов.",
                f"Ваш запрос: '{request.query}' принят. Я проанализирую его и постараюсь дать полезный ответ.",
                f"Спасибо за вопрос: '{request.query}'. Я AI агент, который учится и развивается. Дайте мне немного времени на размышление."
            ]

            # Выбираем случайный ответ для разнообразия
            import random
            return random.choice(responses)

    def _process_tool_results(self, tool_results: Dict[str, Any]) -> Any:
        """Обработка результатов выполнения инструментов"""
        if not tool_results:
            return "Не удалось выполнить инструменты"

        # Для простоты возвращаем результат первого успешного инструмента
        for tool_name, result in tool_results.items():
            if hasattr(result, 'success') and result.success:
                # Проверяем, есть ли атрибут result у объекта
                if hasattr(result, 'result'):
                    return result.result
                else:
                    return "Инструмент выполнен успешно"

        return "Все инструменты завершились с ошибками"

    def _add_reasoning_step(self, step_type: str, description: str, data: Optional[Dict[str, Any]] = None):
        """Добавление шага в трассировку рассуждений"""
        if self.reasoning_tracer:
            step = ReasoningStep(
                step_type=step_type,
                description=description,
                timestamp=datetime.now(),
                data=data or {}
            )
            self.reasoning_trace.append(step)
            self.reasoning_tracer.add_step(step_type, description, data)

    async def get_status(self) -> Dict[str, Any]:
        """Получение текущего статуса агента"""
        # Получаем статус оркестратора инструментов
        orchestrator_status = await self.tool_orchestrator.get_status()

        return {
            "state": self.state_manager.current_state.value,
            "confidence": self.confidence_score,
            "task_complexity": self.task_complexity.value,
            "active_tools": orchestrator_status.get('total_tools', 0),
            "memory_entries": self.memory_manager.get_memory_stats() if self.memory_manager else 0,
            "reasoning_steps": len(self.reasoning_trace),
            "orchestrator_status": orchestrator_status
        }

    def get_metrics(self, timeframe: str = "1h") -> Dict[str, Any]:
        """Получение метрик агента"""
        # Расчет средних значений
        avg_execution_time = 0.0
        if self.requests_processed > 0:
            avg_execution_time = self.total_execution_time / self.requests_processed

        error_rate = 0.0
        if self.requests_processed > 0:
            error_rate = self.error_count / self.requests_processed

        return {
            "requests_processed": self.requests_processed,
            "average_confidence": self.confidence_score,
            "average_execution_time": avg_execution_time,
            "error_rate": error_rate,
            "tool_usage_stats": self.tool_usage_stats.copy()
        }

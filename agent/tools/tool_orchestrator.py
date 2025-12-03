import asyncio
import logging
from typing import Dict, List, Any, Optional
from .base_tool import BaseTool, Task, ToolResult
import time
from integrations.circuit_breaker import circuit_breaker_decorator, CircuitBreakerConfig

logger = logging.getLogger(__name__)

class ToolOrchestrator:
    """Оркестратор инструментов с продвинутой логикой"""

    def __init__(self, default_timeout: float = 30.0, max_concurrent_tools: int = 5):
        self.default_timeout = default_timeout
        self.tools: Dict[str, BaseTool] = {}
        self.execution_history = []
        self._initialized = False
        # Добавляем кэш для часто используемых инструментов
        self._tool_cache = {}
        # Добавляем статистику использования инструментов
        self._tool_usage_stats = {}
        # Добавляем приоритеты инструментов для оптимизации
        self._tool_priorities = {}
        # Добавляем флаг для отслеживания состояния
        self._is_operational = False
        # Добавляем семафор для ограничения количества одновременных вызовов инструментов
        self._concurrent_semaphore = asyncio.Semaphore(max_concurrent_tools)

    async def initialize(self) -> bool:
        """Инициализация всех инструментов"""
        if self._initialized:
            return True

        # Регистрация инструментов
        await self._register_core_tools()
        await self._register_advanced_tools()

        # Инициализация всех инструментов
        init_tasks = []
        for tool in self.tools.values():
            if hasattr(tool, 'initialize'):
                init_tasks.append(tool.initialize())

        results = await asyncio.gather(*init_tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        self._initialized = success_count == len(self.tools)
        self._is_operational = self._initialized

        # Очистка кэша при инициализации
        self._clear_expired_cache_entries()

        return self._initialized

    def _clear_expired_cache_entries(self):
        """Очистка устаревших записей кэша"""
        current_time = time.time()
        expired_keys = []
        for key, result in self._tool_cache.items():
            # Предполагаем, что у результата есть метаданные с временем создания
            if hasattr(result, 'timestamp') and current_time - result.timestamp > 3600:  # 1 час
                expired_keys.append(key)
                
        for key in expired_keys:
            del self._tool_cache[key]

    async def _register_core_tools(self):
        """Регистрация основных инструментов"""
        from .hybrid_models_tool import HybridModelsTool
        from .rag_tool import RAGTool
        from .cache_tool import CacheTool
        from .auth_tool import AuthTool

        self.tools.update({
            'hybrid_models': HybridModelsTool(),
            'rag': RAGTool(),
            'cache': CacheTool(),
            'auth': AuthTool()
        })

    async def _register_advanced_tools(self):
        """Регистрация продвинутых инструментов"""
        from .web_research_tool import WebResearchTool
        from .analytics_tool import AnalyticsTool

        self.tools.update({
            'web_research': WebResearchTool(),
            'analytics': AnalyticsTool()
        })

    @circuit_breaker_decorator("tool_orchestrator_chain", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        timeout=45.0,
        name="tool_orchestrator_chain"
    ))
    async def execute_tool_chain(self, tool_chain: List[str], task: Task) -> Dict[str, Any]:
        """Выполнение цепочки инструментов"""
        results = {}
        context = task.context.copy() if task.context else {}

        for tool_name in tool_chain:
            if tool_name not in self.tools:
                continue

            tool = self.tools[tool_name]

            # Проверка возможности выполнения
            if not tool.can_handle(task):
                continue

            # Выполнение инструмента с ограничением на одновременные вызовы
            async with self._concurrent_semaphore:
                start_time = time.time()
                result = await tool.execute(task)
                execution_time = time.time() - start_time

            # Сохранение результата
            results[tool_name] = result

            # Обновление контекста для следующих инструментов
            if result.success and result.data:
                context[f"{tool_name}_result"] = result.data

            # Запись в историю выполнения
            self.execution_history.append({
                'tool': tool_name,
                'task_id': task.metadata.get('id') if task.metadata else None,
                'success': result.success,
                'execution_time': execution_time,
                'timestamp': time.time()
            })

            # Если инструмент дал финальный результат, останавливаемся
            if result.success and result.metadata.get('is_final', False):
                break

        return results

    async def execute_tool_chain_optimized(self, tool_chain: List[str], task: Task) -> Dict[str, Any]:
        """Оптимизированное выполнение цепочки инструментов с учетом приоритетов и кэширования"""
        results = {}
        context = task.context.copy() if task.context else {}

        # Сортируем инструменты по приоритету
        prioritized_tools = sorted(tool_chain, key=lambda x: self._tool_priorities.get(x, 0), reverse=True)

        for tool_name in prioritized_tools:
            if tool_name not in self.tools:
                continue

            tool = self.tools[tool_name]

            # Проверка возможности выполнения
            if not tool.can_handle(task):
                continue

            # Проверяем кэш перед выполнением
            cache_key = f"{tool_name}:{hash(str(task.query))}:{hash(str(context))}"
            cached_result = self._tool_cache.get(cache_key)
            if cached_result:
                results[tool_name] = cached_result
                # Обновляем контекст для следующих инструментов
                if cached_result.success and cached_result.data:
                    context[f"{tool_name}_result"] = cached_result.data
                continue

            # Выполнение инструмента с ограничением на одновременные вызовы
            async with self._concurrent_semaphore:
                start_time = time.time()
                result = await tool.execute(task)
                execution_time = time.time() - start_time

            # Сохранение результата
            results[tool_name] = result

            # Кэшируем результат
            self._tool_cache[cache_key] = result

            # Обновление статистики использования
            if tool_name not in self._tool_usage_stats:
                self._tool_usage_stats[tool_name] = {'executions': 0, 'successes': 0, 'total_time': 0.0}
            self._tool_usage_stats[tool_name]['executions'] += 1
            self._tool_usage_stats[tool_name]['total_time'] += execution_time
            if result.success:
                self._tool_usage_stats[tool_name]['successes'] += 1

            # Обновление контекста для следующих инструментов
            if result.success and result.data:
                context[f"{tool_name}_result"] = result.data

            # Запись в историю выполнения
            self.execution_history.append({
                'tool': tool_name,
                'task_id': task.metadata.get('id') if task.metadata else None,
                'success': result.success,
                'execution_time': execution_time,
                'timestamp': time.time()
            })

            # Если инструмент дал финальный результат, останавливаемся
            if result.success and result.metadata.get('is_final', False):
                break

        return results

    @circuit_breaker_decorator("tool_orchestrator_execute", CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=90.0,
        timeout=30.0,
        name="tool_orchestrator_execute"
    ))
    async def execute_tools(self, tool_names: List[str], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение нескольких инструментов (для совместимости с тестами)"""
        results = {}

        for tool_name in tool_names:
            if tool_name not in self.tools:
                results[tool_name] = ToolResult(
                    success=False,
                    data=None,
                    metadata={'tool': tool_name},
                    execution_time=0.0,
                    error_message=f"Tool '{tool_name}' not found"
                )
                continue

            tool = self.tools[tool_name]

            # Создаем задачу из inputs
            task = Task(
                query=inputs.get('request', {}).get('query', '') if isinstance(inputs.get('request'), dict) else '',
                context=inputs.get('context', {}),
                user_id=inputs.get('request', {}).get('user_id') if isinstance(inputs.get('request'), dict) else None,
                session_id=inputs.get('request', {}).get('session_id') if isinstance(inputs.get('request'), dict) else None,
                metadata={'source': 'execute_tools'}
            )

            # Проверка возможности выполнения
            if not tool.can_handle(task):
                results[tool_name] = ToolResult(
                    success=False,
                    data=None,
                    metadata={'tool': tool_name},
                    execution_time=0.0,
                    error_message=f"Tool '{tool_name}' cannot handle this task"
                )
                continue

            # Проверяем кэш перед выполнением
            cache_key = f"{tool_name}:{hash(str(task.query))}:{hash(str(task.context))}"
            cached_result = self._tool_cache.get(cache_key)
            if cached_result:
                results[tool_name] = cached_result
                continue

            # Выполнение инструмента с ограничением на одновременные вызовы
            async with self._concurrent_semaphore:
                start_time = time.time()
                result = await tool.execute(task)
                execution_time = time.time() - start_time

            results[tool_name] = result

            # Кэшируем результат
            self._tool_cache[cache_key] = result

            # Обновляем статистику использования
            if tool_name not in self._tool_usage_stats:
                self._tool_usage_stats[tool_name] = {'executions': 0, 'successes': 0, 'total_time': 0.0}
            self._tool_usage_stats[tool_name]['executions'] += 1
            self._tool_usage_stats[tool_name]['total_time'] += execution_time
            if result.success:
                self._tool_usage_stats[tool_name]['successes'] += 1

        # Периодическая очистка устаревшего кэша
        if len(self._tool_cache) > 100:  # Если кэш слишком большой
            self._clear_expired_cache_entries()

        return results

    @circuit_breaker_decorator("tool_orchestrator_parallel", CircuitBreakerConfig(
        failure_threshold=4,
        recovery_timeout=120.0,
        timeout=60.0,
        name="tool_orchestrator_parallel"
    ))
    async def execute_parallel(self, tool_chain: List[str], task: Task) -> Dict[str, Any]:
        """Параллельное выполнение независимых инструментов"""
        # Группировка инструментов по зависимостям
        independent_tools = [t for t in tool_chain if not self._has_dependencies(t)]
        dependent_tools = [t for t in tool_chain if self._has_dependencies(t)]

        results = {}

        # Параллельное выполнение независимых
        if independent_tools:
            # Создаем задачи с ограничением на одновременные вызовы
            independent_tasks = [
                self._execute_with_semaphore(tool_name, task)
                for tool_name in independent_tools
                if tool_name in self.tools
            ]
            independent_results = await asyncio.gather(*independent_tasks, return_exceptions=True)

            for tool_name, result in zip(independent_tools, independent_results):
                if isinstance(result, Exception):
                    results[tool_name] = ToolResult(
                        success=False,
                        data=None,
                        metadata={'tool': tool_name},
                        execution_time=0.0,
                        error_message=str(result)
                    )
                else:
                    results[tool_name] = result

        # Последовательное выполнение зависимых
        for tool_name in dependent_tools:
            if tool_name in self.tools:
                # Выполнение с ограничением на одновременные вызовы
                async with self._concurrent_semaphore:
                    result = await self.tools[tool_name].execute(task)
                results[tool_name] = result

        return results

    @circuit_breaker_decorator("tool_orchestrator_cache", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        timeout=20.0,
        name="tool_orchestrator_cache"
    ))
    async def execute_with_cache(self, tool_name: str, task: Task) -> ToolResult:
        """Выполнение с кэшированием результатов"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                metadata={'tool': tool_name},
                execution_time=0.0,
                error_message=f"Tool '{tool_name}' not found"
            )

        # Проверяем внутреннее кэширование
        cache_key = f"{tool_name}:{hash(str(task.query))}:{hash(str(task.context))}"
        cached_result = self._tool_cache.get(cache_key)
        if cached_result:
            return ToolResult(
                success=True,
                data=cached_result.data,
                metadata={
                    'tool': tool_name,
                    'cached': True,
                    'execution_time': 0.0  # Время выполнения из кэша
                },
                execution_time=0.0
            )

        # Выполняем инструмент с ограничением на одновременные вызовы
        async with self._concurrent_semaphore:
            start_time = time.time()
            result = await self.tools[tool_name].execute(task)
            execution_time = time.time() - start_time

        # Кэшируем результат
        self._tool_cache[cache_key] = result

        return ToolResult(
            success=result.success,
            data=result.data,
            metadata={'tool': tool_name, 'cached': False},
            execution_time=execution_time,
            error_message=result.error_message if not result.success else None
        )

    def _has_dependencies(self, tool_name: str) -> bool:
        """Проверка наличия зависимостей у инструмента"""
        # Простая логика зависимостей - можно расширить
        dependent_tools = ['hybrid_models']  # Зависит от результатов других инструментов
        return tool_name in dependent_tools

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Получение информации о доступных инструментах"""
        return {
            name: {
                'description': tool.description,
                'version': getattr(tool, 'version', 'unknown'),
                'status': 'active'
            }
            for name, tool in self.tools.items()
        }

    async def get_tool_metrics(self) -> Dict[str, Any]:
        """Получение метрик использования инструментов"""
        metrics = {}

        for tool_name, tool in self.tools.items():
            # Используем статистику из _tool_usage_stats если доступна
            if tool_name in self._tool_usage_stats:
                stats = self._tool_usage_stats[tool_name]
                usage_count = stats['executions']
                success_rate = stats['successes'] / usage_count if usage_count > 0 else 0
                avg_execution_time = stats['total_time'] / usage_count if usage_count > 0 else 0
            else:
                # Резервный способ получения метрик из истории выполнения
                tool_history = [
                    h for h in self.execution_history
                    if h['tool'] == tool_name
                ]
                if tool_history:
                    usage_count = len(tool_history)
                    success_rate = sum(1 for h in tool_history if h['success']) / len(tool_history)
                    avg_execution_time = sum(h['execution_time'] for h in tool_history) / len(tool_history)
                else:
                    usage_count = 0
                    success_rate = 0
                    avg_execution_time = 0

            metrics[tool_name] = {
                'usage_count': usage_count,
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time
            }

        return metrics

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса оркестратора"""
        tool_statuses = {}
        for name, tool in self.tools.items():
            if hasattr(tool, 'get_status'):
                try:
                    tool_statuses[name] = await tool.get_status()
                except:
                    tool_statuses[name] = {'status': 'error'}
            else:
                tool_statuses[name] = {'status': 'unknown'}

        return {
            'initialized': self._initialized,
            'total_tools': len(self.tools),
            'tool_statuses': tool_statuses,
            'execution_history_count': len(self.execution_history)
        }

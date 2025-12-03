from typing import Dict, Any, List
import time
import logging
from .task_analyzer import TaskAnalyzer
from ..tools.tool_registry import ToolRegistry
from ..tools.base_tool import Task
from ..meta_cognitive.decision_engine import AdvancedDecisionEngine as DecisionEngine

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self):
        self.task_analyzer = TaskAnalyzer()
        self.tool_registry = ToolRegistry()
        self.decision_engine = DecisionEngine()
        self._initialized = False

    async def initialize(self):
        """Инициализация оркестратора"""
        if not self._initialized:
            # Инициализация компонентов если необходимо
            self._initialized = True

    async def process_request(self, user_request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            # 1. Создать задачу
            task = Task(
                query=user_request,
                context=context or {},
                metadata={'timestamp': time.time()}
            )

            # 2. Анализировать задачу
            task_analysis = await self.task_analyzer.analyze(task)

            # 3. Выбрать инструменты
            tool_chain = await self.decision_engine.select_tools(task_analysis)

            # 4. Выполнить через инструменты
            result = await self._execute_tool_chain(tool_chain, task)

            # 5. Форматировать ответ
            return self._format_response(result, task_analysis)

        except Exception as e:
            # Обработка ошибок для graceful degradation
            logger.error(f"Error processing request '{user_request}': {e}")
            return {
                'success': False,
                'answer': 'Извините, произошла ошибка при обработке запроса',
                'sources': [],
                'metadata': {
                    'tools_used': [],
                    'analysis': {},
                    'execution_time': 0,
                    'error': str(e)
                }
            }

    async def _execute_tool_chain(self, tool_chain: List[str], task: Task) -> Dict[str, Any]:
        results = {}

        for tool_name in tool_chain:
            tool = self.tool_registry.get_tool(tool_name)
            if tool and await tool.can_handle(task):
                result = await tool.execute(task)
                results[tool_name] = result

                if result.success and getattr(result, 'metadata', {}).get('is_final', False):
                    break

        return results

    def _format_response(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        final_result = results.get('hybrid_models')

        # Безопасное суммирование execution_time
        execution_times = []
        for r in results.values():
            if hasattr(r, 'execution_time') and isinstance(r.execution_time, (int, float)):
                execution_times.append(r.execution_time)

        # Извлечение ответа из ToolResult
        answer = 'Извините, не удалось обработать запрос'
        sources = []

        if final_result and hasattr(final_result, 'data'):
            result_data = final_result.data
            if isinstance(result_data, dict):
                answer = result_data.get('answer', answer)
                sources = result_data.get('sources', sources)
            else:
                answer = str(result_data)

        return {
            'success': True,
            'answer': answer,
            'sources': sources,
            'metadata': {
                'tools_used': list(results.keys()),
                'analysis': analysis,
                'execution_time': sum(execution_times) if execution_times else 0
            }
        }

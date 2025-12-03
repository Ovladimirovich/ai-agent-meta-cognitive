import time
import logging
from typing import Dict, Any
from .base_tool import BaseTool, Task, ToolResult
from ..integrations.circuit_breaker import circuit_breaker_decorator, CircuitBreakerConfig

logger = logging.getLogger(__name__)

class WebResearchTool(BaseTool):
    name = "web_research"
    description = "Исследование информации в интернете"
    version = "1.0.0"

    def __init__(self):
        self.research_manager = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Инициализация веб-исследования"""
        if self._initialized:
            return True

        try:
            from web_research.web_research_manager import WebResearchManager
            self.research_manager = WebResearchManager()
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize WebResearchTool: {e}")
            return False

    def can_handle(self, task: Task) -> bool:
        """Проверка необходимости веб-поиска"""
        if not isinstance(task.query, str):
            return False

        query_lower = task.query.lower()

        # Индикаторы необходимости поиска в интернете
        web_indicators = [
            'новости', 'актуально', 'сейчас', 'последние',
            'тренды', 'популярно', 'рейтинг', 'топ',
            'интернет', 'веб', 'сайт', 'онлайн'
        ]

        return any(indicator in query_lower for indicator in web_indicators)

    @circuit_breaker_decorator("web_research_tool", CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=180.0,
        timeout=30.0,
        name="web_research_tool"
    ))
    async def execute(self, task: Task) -> ToolResult:
        """Выполнение веб-исследования"""
        start_time = time.time()

        try:
            if not await self.initialize():
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={},
                    execution_time=time.time() - start_time,
                    error_message="Web research system not available"
                )

            # Выполнение поиска
            results = await self.research_manager.research(task.query)

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data=results,
                metadata={
                    'tool': self.name,
                    'sources_count': len(results.get('sources', [])),
                    'is_final': False
                },
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"WebResearchTool execution error: {e}")

            return ToolResult(
                success=False,
                data=None,
                metadata={'tool': self.name},
                execution_time=execution_time,
                error_message=str(e)
            )

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса инструмента"""
        return {
            'name': self.name,
            'initialized': self._initialized,
            'version': self.version,
            'capabilities': ['web_search', 'information_gathering', 'news_research'],
            'research_type': 'web' if self._initialized else None
        }

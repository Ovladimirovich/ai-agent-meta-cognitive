import time
import logging
from typing import Dict, Any
from .base_tool import BaseTool, Task, ToolResult
from integrations.circuit_breaker import circuit_breaker_decorator, CircuitBreakerConfig

from analytics import AnalyticsEngine, AsyncAnalyticsEngine

logger = logging.getLogger(__name__)


class AnalyticsTool(BaseTool):
    name = "analytics"
    description = "Анализ данных и генерация инсайтов"
    version = "1.0.0"

    def __init__(self):
        self.analytics_engine = None
        self.async_analytics_engine = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Инициализация аналитического движка"""
        if self._initialized:
            return True

        try:
            self.analytics_engine = AnalyticsEngine()
            self.async_analytics_engine = AsyncAnalyticsEngine()
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AnalyticsTool: {e}")
            return False

    def can_handle(self, task: Task) -> bool:
        """Проверка необходимости аналитики"""
        if not isinstance(task.query, str):
            return False

        query_lower = task.query.lower()

        analytics_indicators = [
            'анализ', 'статистика', 'тренд', 'динамика',
            'сравни', 'оцени', 'эффективность', 'метрики',
            'график', 'диаграмма', 'показатели', 'данные'
        ]

        return any(indicator in query_lower for indicator in analytics_indicators)

    @circuit_breaker_decorator("analytics_tool", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=120.0,
        timeout=45.0,
        name="analytics_tool"
    ))
    async def execute(self, task: Task) -> ToolResult:
        """Выполнение аналитических расчетов"""
        start_time = time.time()

        try:
            if not self._initialized:
                if not await self.initialize():
                    return ToolResult(
                        success=False,
                        data=None,
                        metadata={},
                        execution_time=time.time() - start_time,
                        error_message="Analytics system not available"
                    )

            # Анализ запроса для определения типа аналитики
            analysis_type = self._determine_analysis_type(task.query)

            # Подготовка данных для анализа
            analysis_data = self._prepare_analysis_data(task)

            # Выполнение анализа через AnalyticsEngine
            result = await self.analytics_engine.analyze(
                data=analysis_data,
                analysis_type=analysis_type,
                context=task.context
            )

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data={
                    'insights': result.insights,
                    'metrics': result.metrics,
                    'recommendations': result.recommendations,
                    'confidence': result.confidence,
                    'data_quality_score': result.data_quality_score
                },
                metadata={
                    'tool': self.name,
                    'analysis_type': analysis_type,
                    'is_final': False,
                    'processing_time': result.processing_time
                },
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"AnalyticsTool execution error: {e}")

            return ToolResult(
                success=False,
                data=None,
                metadata={'tool': self.name},
                execution_time=execution_time,
                error_message=str(e)
            )

    def _determine_analysis_type(self, query: str) -> str:
        """Определение типа анализа"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['тренд', 'динамика', 'время']):
            return 'trend_analysis'
        elif any(word in query_lower for word in ['сравни', 'сравнение']):
            return 'comparative_analysis'
        elif any(word in query_lower for word in ['эффективность', 'продуктивность']):
            return 'performance_analysis'
        elif any(word in query_lower for word in ['статистика', 'метрики']):
            return 'statistical_analysis'
        elif any(word in query_lower for word in ['корреляция', 'связь']):
            return 'correlation_analysis'
        else:
            return 'general_analysis'

    def _prepare_analysis_data(self, task: Task) -> Any:
        """Подготовка данных для анализа"""
        # Извлечение данных из контекста задачи
        if task.context and 'analysis_data' in task.context:
            return task.context['analysis_data']

        # Попытка извлечь данные из metadata
        if task.metadata and 'data' in task.metadata:
            return task.metadata['data']

        # По умолчанию возвращаем сам запрос как данные для анализа
        return task.query


    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса инструмента"""
        return {
            'name': self.name,
            'initialized': self._initialized,
            'version': self.version,
            'capabilities': ['data_analysis', 'trend_detection', 'performance_metrics', 'comparative_analysis'],
            'analysis_types': ['trend', 'comparative', 'performance', 'statistical'] if self._initialized else []
        }

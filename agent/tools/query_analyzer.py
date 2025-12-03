import logging
from typing import Dict, Any, Optional

from ..core.models import AgentRequest, QueryAnalysis, TaskComplexity
from hybrid.hybrid_manager import HybridManager
from hybrid.models import ModelProvider

logger = logging.getLogger("QueryAnalyzer")


class QueryAnalyzer:
    """
    Анализатор запросов агента.

    Определяет намерение пользователя, сложность задачи и требуемые инструменты.
    Использует гибридную систему моделей для интеллектуального анализа.
    """

    def __init__(self, hybrid_manager: Optional['HybridManager'] = None):
        self.hybrid_manager = hybrid_manager
        self.intent_classifier = IntentClassifier()
        self.complexity_analyzer = ComplexityAnalyzer()

    async def analyze(self, request: AgentRequest) -> QueryAnalysis:
        """
        Анализ запроса с использованием гибридной системы.

        Процесс:
        1. Классификация намерения
        2. Анализ сложности через модели
        3. Определение требуемых инструментов
        """
        try:
            # Классификация намерения
            intent = self.intent_classifier.classify(request.query)

            # Анализ сложности
            complexity = await self._analyze_complexity(request)

            # Определение требуемых инструментов
            required_tools = self._determine_required_tools(intent, complexity, request)

            return QueryAnalysis(
                intent=intent,
                complexity=complexity,
                required_tools=required_tools,
                context=request.context,
                metadata=request.metadata
            )

        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            # Fallback: простая классификация
            return QueryAnalysis(
                intent="unknown",
                complexity=TaskComplexity.SIMPLE,
                required_tools=[],
                context=request.context,
                metadata={**request.metadata, "error": str(e)} if request.metadata else {"error": str(e)}
            )

    async def _analyze_complexity(self, request: AgentRequest) -> TaskComplexity:
        """Анализ сложности запроса через гибридную систему"""
        if not self.hybrid_manager:
            # Без гибридного менеджера - простая эвристика
            return self._simple_complexity_analysis(request.query)

        try:
            # Формируем запрос для анализа сложности
            complexity_prompt = f"""Оцени сложность этого запроса пользователя и выбери подходящий уровень:

Запрос: {request.query}

Уровни сложности:
- SIMPLE: Простые вопросы, приветствия, базовые команды
- MEDIUM: Анализ данных, поиск информации, умеренная обработка
- COMPLEX: Многошаговые задачи, глубокий анализ, интеграция нескольких инструментов

Ответь только одним словом: SIMPLE, MEDIUM или COMPLEX"""

            # Используем локальную модель для анализа (быстрее и дешевле)
            response = await self.hybrid_manager.process_query(
                complexity_prompt,
                force_provider=ModelProvider.LOCAL_QWEN
            )

            # Парсим ответ
            complexity = self.complexity_analyzer.parse_complexity(response.answer.strip().upper())
            return complexity

        except Exception as e:
            logger.warning(f"❌ Complexity analysis failed, using fallback: {e}")
            return self._simple_complexity_analysis(request.query)

    def _simple_complexity_analysis(self, query: str) -> TaskComplexity:
        """Простой анализ сложности на основе эвристик"""
        query_lower = query.lower()

        # Ключевые слова для определения сложности
        simple_keywords = ["привет", "здравствуй", "как дела", "спасибо", "пока"]
        medium_keywords = ["найди", "покажи", "расскажи", "объясни", "проанализируй"]
        complex_keywords = ["создай", "разработай", "интегрируй", "оптимизируй", "тестируй"]

        # Подсчет совпадений
        simple_count = sum(1 for word in simple_keywords if word in query_lower)
        medium_count = sum(1 for word in medium_keywords if word in query_lower)
        complex_count = sum(1 for word in complex_keywords if word in query_lower)

        # Определение сложности
        if complex_count > 0 or len(query.split()) > 20:
            return TaskComplexity.COMPLEX
        elif medium_count > 0 or len(query.split()) > 10:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.SIMPLE

    def _determine_required_tools(self, intent: str, complexity: TaskComplexity, request: AgentRequest) -> list[str]:
        """Определение требуемых инструментов на основе намерения и сложности"""
        tools = []

        # Анализ на основе намерения
        if "search" in intent.lower() or "найди" in request.query.lower():
            tools.append("rag_search")
        if "analyze" in intent.lower() or "анализ" in request.query.lower():
            tools.append("data_analyzer")
        if "code" in intent.lower() or "код" in request.query.lower():
            tools.append("code_executor")

        # Анализ на основе сложности
        if complexity == TaskComplexity.MEDIUM:
            if not tools:  # Если не определили инструменты по намерению
                tools.append("general_assistant")
        elif complexity == TaskComplexity.COMPLEX:
            if not tools:
                tools.extend(["rag_search", "data_analyzer"])
            # Для сложных задач добавляем больше инструментов
            if len(request.query.split()) > 15:
                tools.append("reasoning_engine")

        # Проверка предпочтений пользователя
        if request.preferences and request.preferences.preferred_tools:
            # Фильтруем по предпочтениям пользователя
            preferred = set(request.preferences.preferred_tools)
            tools = [tool for tool in tools if tool in preferred]

        return tools


class IntentClassifier:
    """Классификатор намерений пользователя"""

    def __init__(self):
        self.intent_patterns = {
            "greeting": ["привет", "здравствуй", "добрый день", "доброе утро", "добрый вечер"],
            "question": ["что", "как", "где", "когда", "почему", "зачем"],
            "search": ["найди", "покажи", "ищи", "search", "find"],
            "analyze": ["анализ", "проанализируй", "изучи", "исследуй"],
            "create": ["создай", "сделай", "разработай", "напиши"],
            "help": ["помоги", "помощь", "help", "подскажи"]
        }

    def classify(self, query: str) -> str:
        """Классификация намерения на основе паттернов"""
        query_lower = query.lower()

        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent

        return "general"  # Общее намерение по умолчанию


class ComplexityAnalyzer:
    """Анализатор сложности запросов"""

    def parse_complexity(self, response: str) -> TaskComplexity:
        """Парсинг ответа модели в enum сложности"""
        response_upper = response.upper().strip()

        if "SIMPLE" in response_upper:
            return TaskComplexity.SIMPLE
        elif "MEDIUM" in response_upper:
            return TaskComplexity.MEDIUM
        elif "COMPLEX" in response_upper:
            return TaskComplexity.COMPLEX
        else:
            # Fallback на основе длины ответа
            if len(response.split()) > 10:
                return TaskComplexity.COMPLEX
            else:
                return TaskComplexity.SIMPLE

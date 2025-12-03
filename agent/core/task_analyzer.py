from typing import Dict, Any
from ..tools.base_tool import Task

class TaskAnalyzer:
    async def analyze(self, task: Task) -> Dict[str, Any]:
        return {
            'needs_search': self._detect_search_need(task.query),
            'needs_generation': True,
            'can_cache': len(task.query) < 500,
            'complexity': self._estimate_complexity(task.query),
            'domain': self._detect_domain(task.query)
        }

    def _detect_search_need(self, query: str) -> bool:
        search_keywords = [
            'что', 'как', 'почему', 'где', 'когда', 'кто',
            'расскажи', 'объясни', 'найди', 'покажи'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in search_keywords)

    def _estimate_complexity(self, query: str) -> str:
        length = len(query)
        if length < 50:
            return 'low'
        elif length < 200:
            return 'medium'
        else:
            return 'high'

    def _detect_domain(self, query: str) -> str:
        query_lower = query.lower()

        if any(word in query_lower for word in ['программирование', 'код', 'python', 'javascript']):
            return 'programming'
        elif any(word in query_lower for word in ['математика', 'формула', 'расчет']):
            return 'mathematics'
        elif any(word in query_lower for word in ['история', 'прошлое', 'время']):
            return 'history'
        else:
            return 'general'

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Результат поиска"""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str] = None

@dataclass
class ResearchQuery:
    """Запрос на исследование"""
    topic: str
    depth: str = "comprehensive"  # quick, comprehensive, deep
    include_sources: bool = True
    max_sources: int = 5

@dataclass
class ResearchConfig:
    """Конфигурация исследования"""
    max_results: int = 10
    timeout: int = 30
    max_pages_per_source: int = 3
    enable_caching: bool = True
    cache_ttl: int = 3600

@dataclass
class ResearchResult:
    """Результат веб-исследования"""
    query: str
    sources: List[Dict[str, Any]]
    summary: str
    key_insights: List[str]
    credibility_score: float
    freshness_score: float
    processing_time: float
    timestamp: datetime


class WebResearchManager:
    """
    Менеджер веб-исследований для AI агента.

    Выполняет поиск информации в интернете, анализ источников,
    оценку достоверности и генерацию сводок.
    """

    def __init__(self):
        self._initialized = False
        self.search_engines = []
        self.content_analyzer = None

    async def initialize(self) -> bool:
        """Инициализация системы веб-исследований"""
        if self._initialized:
            return True

        try:
            # Инициализация поисковых движков
            self.search_engines = [
                GoogleSearchEngine(),
                BingSearchEngine(),
                DuckDuckGoSearchEngine()
            ]

            # Инициализация анализатора контента
            self.content_analyzer = ContentAnalyzer()

            self._initialized = True
            logger.info("Web research system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize web research system: {e}")
            return False

    async def research(self, query: str, max_sources: int = 5, timeout: int = 30) -> Dict[str, Any]:
        """
        Выполнение веб-исследования

        Args:
            query: Поисковый запрос
            max_sources: Максимальное количество источников
            timeout: Таймаут в секундах

        Returns:
            Результаты исследования
        """
        if not await self.initialize():
            return {
                'error': 'Web research system not initialized',
                'sources': [],
                'summary': '',
                'key_insights': []
            }

        start_time = time.time()

        try:
            # Параллельный поиск по разным движкам
            search_tasks = []
            for engine in self.search_engines[:2]:  # Используем только 2 движка для скорости
                search_tasks.append(engine.search(query, max_results=max_sources // 2))

            # Ожидание результатов поиска
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Обработка результатов
            all_sources = []
            for result in search_results:
                if isinstance(result, Exception):
                    logger.error(f"Search engine error: {result}")
                    continue
                if result and 'sources' in result:
                    all_sources.extend(result['sources'])

            # Удаление дубликатов и ранжирование
            unique_sources = self._deduplicate_sources(all_sources)
            ranked_sources = self._rank_sources(unique_sources, query)

            # Ограничение количества источников
            selected_sources = ranked_sources[:max_sources]

            # Анализ контента источников
            analyzed_sources = []
            for source in selected_sources:
                try:
                    analysis = await self.content_analyzer.analyze_source(source)
                    analyzed_sources.append(analysis)
                except Exception as e:
                    logger.error(f"Content analysis error for {source.get('url', 'unknown')}: {e}")
                    analyzed_sources.append(source)  # Добавляем без анализа

            # Генерация сводки и инсайтов
            summary = await self._generate_summary(query, analyzed_sources)
            key_insights = await self._extract_key_insights(query, analyzed_sources)

            # Расчет метрик
            credibility_score = self._calculate_credibility_score(analyzed_sources)
            freshness_score = self._calculate_freshness_score(analyzed_sources)

            processing_time = time.time() - start_time

            result = ResearchResult(
                query=query,
                sources=analyzed_sources,
                summary=summary,
                key_insights=key_insights,
                credibility_score=credibility_score,
                freshness_score=freshness_score,
                processing_time=processing_time,
                timestamp=datetime.now()
            )

            return {
                'query': result.query,
                'sources': result.sources,
                'summary': result.summary,
                'key_insights': result.key_insights,
                'credibility_score': result.credibility_score,
                'freshness_score': result.freshness_score,
                'processing_time': result.processing_time,
                'timestamp': result.timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"Web research failed: {e}")
            processing_time = time.time() - start_time

            return {
                'error': str(e),
                'query': query,
                'sources': [],
                'summary': 'Не удалось выполнить исследование',
                'key_insights': [],
                'processing_time': processing_time
            }

    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Удаление дублированных источников"""
        seen_urls = set()
        unique_sources = []

        for source in sources:
            url = source.get('url', '').lower().strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)

        return unique_sources

    def _rank_sources(self, sources: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Ранжирование источников по релевантности"""
        if not sources:
            return sources

        # Простое ранжирование по ключевым словам в заголовке и описании
        query_words = set(query.lower().split())

        for source in sources:
            relevance_score = 0

            title = source.get('title', '').lower()
            description = source.get('description', '').lower()

            # Проверка наличия ключевых слов
            title_matches = sum(1 for word in query_words if word in title)
            desc_matches = sum(1 for word in query_words if word in description)

            relevance_score = title_matches * 3 + desc_matches * 1

            # Бонус за авторитетные домены
            domain = self._extract_domain(source.get('url', ''))
            if domain in ['wikipedia.org', 'github.com', 'stackoverflow.com', 'arxiv.org']:
                relevance_score += 5

            source['relevance_score'] = relevance_score

        # Сортировка по релевантности
        return sorted(sources, key=lambda x: x.get('relevance_score', 0), reverse=True)

    def _extract_domain(self, url: str) -> str:
        """Извлечение домена из URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ''

    async def _generate_summary(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Генерация сводки по источникам"""
        if not sources:
            return "Источники не найдены"

        # Простая сводка на основе заголовков
        titles = [s.get('title', 'Без заголовка') for s in sources[:3]]
        summary_parts = [
            f"По запросу '{query}' найдено {len(sources)} источников.",
            "Ключевые источники:"
        ]

        for i, title in enumerate(titles, 1):
            summary_parts.append(f"{i}. {title}")

        return " ".join(summary_parts)

    async def _extract_key_insights(self, query: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Извлечение ключевых инсайтов"""
        insights = []

        if not sources:
            return ["Информация не найдена"]

        # Простые инсайты на основе описаний
        descriptions = [s.get('description', '') for s in sources[:3] if s.get('description')]

        if descriptions:
            insights.append(f"Найдено {len(descriptions)} релевантных описаний")
        else:
            insights.append("Источники содержат ограниченную информацию")

        # Добавление временных меток если есть
        recent_sources = [s for s in sources if s.get('published_date')]
        if recent_sources:
            insights.append(f"Найдено {len(recent_sources)} источников с датами публикации")

        return insights

    def _calculate_credibility_score(self, sources: List[Dict[str, Any]]) -> float:
        """Расчет оценки достоверности источников"""
        if not sources:
            return 0.0

        total_score = 0.0

        for source in sources:
            score = 0.5  # Базовая оценка

            domain = self._extract_domain(source.get('url', ''))

            # Бонусы за авторитетные домены
            authoritative_domains = {
                'wikipedia.org': 1.0,
                'github.com': 0.9,
                'stackoverflow.com': 0.8,
                'arxiv.org': 0.9,
                'bbc.com': 0.9,
                'reuters.com': 0.9,
                'nytimes.com': 0.8
            }

            if domain in authoritative_domains:
                score = authoritative_domains[domain]
            elif any(trusted in domain for trusted in ['.edu', '.gov', '.org']):
                score = 0.8
            elif any(commercial in domain for commercial in ['.com', '.net']):
                score = 0.6

            total_score += score

        return total_score / len(sources)

    def _calculate_freshness_score(self, sources: List[Dict[str, Any]]) -> float:
        """Расчет оценки свежести информации"""
        if not sources:
            return 0.0

        current_time = datetime.now()
        total_score = 0.0
        count = 0

        for source in sources:
            date_str = source.get('published_date')
            if date_str:
                try:
                    # Попытка парсинга даты
                    if isinstance(date_str, str):
                        # Простой парсинг - в реальности нужен более robust парсер
                        published_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        published_date = date_str

                    # Расчет дней с момента публикации
                    days_diff = (current_time - published_date).days

                    # Оценка свежести (новее = лучше)
                    if days_diff <= 1:
                        score = 1.0
                    elif days_diff <= 7:
                        score = 0.8
                    elif days_diff <= 30:
                        score = 0.6
                    elif days_diff <= 365:
                        score = 0.4
                    else:
                        score = 0.2

                    total_score += score
                    count += 1

                except Exception as e:
                    logger.debug(f"Date parsing error: {e}")
                    continue

        if count == 0:
            return 0.5  # Средняя оценка если нет дат

        return total_score / count

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса системы веб-исследований"""
        return {
            'initialized': self._initialized,
            'search_engines_count': len(self.search_engines),
            'content_analyzer_available': self.content_analyzer is not None
        }


class BaseSearchEngine:
    """Базовый класс для поисковых движков"""

    def __init__(self, name: str):
        self.name = name

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Базовый метод поиска"""
        raise NotImplementedError


class GoogleSearchEngine(BaseSearchEngine):
    """Google поисковый движок"""

    def __init__(self):
        super().__init__("Google")

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Поиск через Google (заглушка)"""
        # В реальности здесь был бы вызов Google Custom Search API
        # Убрана искусственная задержка для улучшения производительности

        return {
            'engine': self.name,
            'query': query,
            'sources': [
                {
                    'title': f'Результат Google 1 для "{query}"',
                    'url': f'https://example.com/result1?q={query}',
                    'description': f'Описание первого результата поиска по запросу {query}',
                    'published_date': datetime.now().isoformat()
                },
                {
                    'title': f'Результат Google 2 для "{query}"',
                    'url': f'https://example.com/result2?q={query}',
                    'description': f'Описание второго результата поиска по запросу {query}',
                    'published_date': (datetime.now().replace(hour=datetime.now().hour - 1)).isoformat()
                }
            ]
        }


class BingSearchEngine(BaseSearchEngine):
    """Bing поисковый движок"""

    def __init__(self):
        super().__init__("Bing")

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Поиск через Bing (заглушка)"""
        # Убрана искусственная задержка для улучшения производительности

        return {
            'engine': self.name,
            'query': query,
            'sources': [
                {
                    'title': f'Результат Bing 1 для "{query}"',
                    'url': f'https://bing-example.com/result1?q={query}',
                    'description': f'Bing результат: информация по запросу {query}',
                    'published_date': datetime.now().isoformat()
                }
            ]
        }


class DuckDuckGoSearchEngine(BaseSearchEngine):
    """DuckDuckGo поисковый движок"""

    def __init__(self):
        super().__init__("DuckDuckGo")

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Поиск через DuckDuckGo (заглушка)"""
        # Убрана искусственная задержка для улучшения производительности

        return {
            'engine': self.name,
            'query': query,
            'sources': [
                {
                    'title': f'DDG результат для "{query}"',
                    'url': f'https://ddg-example.com/result?q={query}',
                    'description': f'Приватный поиск: результаты по {query}',
                    'published_date': (datetime.now().replace(day=datetime.now().day - 1)).isoformat()
                }
            ]
        }


class ContentAnalyzer:
    """Анализатор контента источников"""

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        # Инициализация анализатора (например, загрузка моделей)
        self._initialized = True
        return True

    async def analyze_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализ отдельного источника

        Args:
            source: Данные источника

        Returns:
            Анализированный источник с дополнительными метаданными
        """
        if not await self.initialize():
            return source

        try:
            # Копирование исходных данных
            analyzed = source.copy()

            # Анализ текста
            text_content = source.get('description', '') + ' ' + source.get('title', '')

            # Простой анализ (в реальности здесь были бы ML модели)
            analyzed.update({
                'word_count': len(text_content.split()),
                'has_keywords': bool(text_content.strip()),
                'content_quality_score': self._assess_content_quality(text_content),
                'analyzed_at': datetime.now().isoformat()
            })

            return analyzed

        except Exception as e:
            logger.error(f"Content analysis error: {e}")
            return source

    def _assess_content_quality(self, text: str) -> float:
        """Оценка качества контента (0.0 - 1.0)"""
        if not text.strip():
            return 0.0

        score = 0.5  # Базовая оценка

        # Длина текста
        word_count = len(text.split())
        if word_count > 50:
            score += 0.2
        elif word_count < 10:
            score -= 0.2

        # Наличие цифр (может указывать на факты)
        digit_ratio = sum(c.isdigit() for c in text) / len(text) if text else 0
        if 0.01 < digit_ratio < 0.1:
            score += 0.1

        # Наличие знаков препинания (структурированный текст)
        punctuation_ratio = sum(c in '.,!?;:' for c in text) / len(text) if text else 0
        if punctuation_ratio > 0.05:
            score += 0.1

        return max(0.0, min(1.0, score))

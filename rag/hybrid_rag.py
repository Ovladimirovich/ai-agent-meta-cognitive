"""
Гибридная RAG система с graceful degradation.

Объединяет FastEmbed + SQLite + MinimalRAG для надежной работы
в любых условиях с автоматическим выбором оптимального метода поиска.
"""

import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
import time

logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    """Методы поиска в гибридной системе"""
    KEYWORD = "keyword"      # Полнотекстовый поиск
    SEMANTIC = "semantic"    # Векторный поиск
    HYBRID = "hybrid"        # Комбинированный поиск


@dataclass
class HybridSearchResult:
    """Результат поиска в гибридной системе"""
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    search_type: str
    method_used: SearchMethod


class HybridRAGSystem:
    """
    Гибридная RAG система с graceful degradation.

    Автоматически выбирает оптимальный метод поиска:
    1. FastEmbed + SQLite (рекомендуемый)
    2. TF-IDF + SQLite (fallback)
    3. MinimalRAG (zero-dependency)

    Особенности:
    - Graceful degradation при недоступности компонентов
    - Автоматический выбор метода поиска
    - Кэширование эмбеддингов
    - Поддержка разных типов поиска
    """

    def __init__(
        self,
        db_path: str = "./rag_db.sqlite",
        embedding_provider: str = "auto",
        enable_caching: bool = True
    ):
        self.db_path = db_path
        self.embedding_provider_type = embedding_provider
        self.enable_caching = enable_caching

        # Компоненты системы
        self.vector_store = None
        self.embedding_provider = None
        self.minimal_rag = None

        # Кэш эмбеддингов
        self.embedding_cache: Dict[str, List[float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Статус инициализации
        self.initialized = False
        self.available_methods = set()

    async def initialize(self) -> bool:
        """
        Инициализация гибридной системы

        Returns:
            True если хотя бы один компонент инициализирован
        """
        if self.initialized:
            return True

        success = False

        # 1. Пытаемся инициализировать SQLite векторное хранилище
        try:
            from .sqlite_vector_store import SQLiteVectorStore
            self.vector_store = SQLiteVectorStore(self.db_path)
            if await self.vector_store.initialize():
                self.available_methods.add(SearchMethod.SEMANTIC)
                self.available_methods.add(SearchMethod.KEYWORD)
                logger.info("SQLite vector store initialized")
                success = True
            else:
                logger.warning("SQLite vector store failed to initialize")
        except Exception as e:
            logger.warning(f"SQLite vector store not available: {e}")

        # 2. Пытаемся инициализировать провайдер эмбеддингов
        try:
            from .lightweight_embeddings import get_embedding_provider
            self.embedding_provider = await get_embedding_provider(self.embedding_provider_type)
            if self.embedding_provider:
                self.available_methods.add(SearchMethod.SEMANTIC)
                logger.info(f"Embedding provider initialized: {self.embedding_provider.name}")
                success = True
            else:
                logger.warning("Embedding provider not available")
        except Exception as e:
            logger.warning(f"Embedding provider not available: {e}")

        # 3. Инициализируем минималистичный RAG (всегда доступен)
        try:
            from .minimal_rag import MinimalRAG
            self.minimal_rag = MinimalRAG()
            self.available_methods.add(SearchMethod.KEYWORD)
            logger.info("Minimal RAG initialized")
            success = True
        except Exception as e:
            logger.error(f"Minimal RAG failed to initialize: {e}")

        if success:
            self.initialized = True
            logger.info(f"Hybrid RAG initialized with methods: {[m.value for m in self.available_methods]}")
        else:
            logger.error("Failed to initialize any RAG component")

        return success

    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Добавление документов в систему

        Args:
            documents: Список документов [{"text": "...", "metadata": {...}, "id": "..."}]

        Returns:
            True если успешно
        """
        if not self.initialized:
            if not await self.initialize():
                return False

        success = False

        # Подготавливаем документы
        processed_docs = []
        for doc in documents:
            processed_doc = {
                'text': doc.get('text', doc.get('content', '')),
                'metadata': doc.get('metadata', {}),
                'id': doc.get('id')
            }
            processed_docs.append(processed_doc)

        # 1. Добавляем в векторное хранилище (если доступно)
        if self.vector_store and self.embedding_provider and SearchMethod.SEMANTIC in self.available_methods:
            try:
                # Создаем эмбеддинги
                texts = [doc['text'] for doc in processed_docs]
                embeddings = await self.embedding_provider.embed_documents(texts)

                # Добавляем в хранилище
                if await self.vector_store.add_documents(processed_docs, embeddings):
                    logger.info(f"Added {len(processed_docs)} documents to vector store")
                    success = True
                else:
                    logger.warning("Failed to add documents to vector store")
            except Exception as e:
                logger.error(f"Vector store addition failed: {e}")

        # 2. Добавляем в минималистичный RAG (всегда доступен как fallback)
        if self.minimal_rag:
            try:
                for doc in processed_docs:
                    self.minimal_rag.add_document(doc['text'], doc['metadata'])
                logger.info(f"Added {len(processed_docs)} documents to minimal RAG")
                success = True
            except Exception as e:
                logger.error(f"Minimal RAG addition failed: {e}")

        return success

    async def search(
        self,
        query: str,
        method: SearchMethod = SearchMethod.HYBRID,
        limit: int = 5,
        **kwargs
    ) -> List[HybridSearchResult]:
        """
        Умный поиск с выбором метода

        Args:
            query: Поисковый запрос
            method: Метод поиска (KEYWORD, SEMANTIC, HYBRID)
            limit: Максимальное количество результатов
            **kwargs: Дополнительные параметры

        Returns:
            Список результатов поиска
        """
        if not self.initialized:
            if not await self.initialize():
                return []

        # Выбираем оптимальный метод
        actual_method = self._choose_search_method(query, method)

        # Выполняем поиск
        if actual_method == SearchMethod.SEMANTIC and self._can_do_semantic():
            return await self._search_semantic(query, limit, **kwargs)
        elif actual_method == SearchMethod.HYBRID and self._can_do_hybrid():
            return await self._search_hybrid(query, limit, **kwargs)
        else:
            return await self._search_keyword(query, limit, **kwargs)

    async def _search_semantic(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> List[HybridSearchResult]:
        """Семантический поиск"""
        if not self._can_do_semantic():
            return await self._search_keyword(query, limit)

        try:
            # Получаем эмбеддинг запроса
            query_embedding = await self._get_query_embedding(query)

            # Ищем в векторном хранилище
            results = await self.vector_store.search_semantic(
                query_embedding,
                limit=limit,
                **kwargs
            )

            # Конвертируем результаты
            return [
                HybridSearchResult(
                    id=result.id,
                    content=result.content,
                    metadata=result.metadata,
                    similarity=result.similarity,
                    search_type=result.search_type,
                    method_used=SearchMethod.SEMANTIC
                )
                for result in results
            ]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return await self._search_keyword(query, limit)

    async def _search_keyword(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> List[HybridSearchResult]:
        """Ключевой поиск"""
        results = []

        # 1. Пытаемся использовать SQLite FTS
        if self.vector_store and SearchMethod.KEYWORD in self.available_methods:
            try:
                vector_results = await self.vector_store.search_keyword(query, limit)
                results.extend([
                    HybridSearchResult(
                        id=result.id,
                        content=result.content,
                        metadata=result.metadata,
                        similarity=result.similarity,
                        search_type=result.search_type,
                        method_used=SearchMethod.KEYWORD
                    )
                    for result in vector_results
                ])
            except Exception as e:
                logger.warning(f"SQLite keyword search failed: {e}")

        # 2. Fallback to minimal RAG
        if self.minimal_rag and not results:
            try:
                minimal_results = self.minimal_rag.search(query, limit)
                results.extend([
                    HybridSearchResult(
                        id=result.id,
                        content=result.content,
                        metadata=result.metadata,
                        similarity=result.score,
                        search_type=result.match_type,
                        method_used=SearchMethod.KEYWORD
                    )
                    for result in minimal_results
                ])
            except Exception as e:
                logger.error(f"Minimal RAG search failed: {e}")

        return results[:limit]

    async def _search_hybrid(
        self,
        query: str,
        limit: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **kwargs
    ) -> List[HybridSearchResult]:
        """Гибридный поиск"""
        if not self._can_do_hybrid():
            return await self._search_semantic(query, limit)

        try:
            # Получаем эмбеддинг запроса
            query_embedding = await self._get_query_embedding(query)

            # Выполняем гибридный поиск в SQLite
            results = await self.vector_store.search_hybrid(
                query=query,
                query_embedding=query_embedding,
                limit=limit,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                **kwargs
            )

            # Конвертируем результаты
            return [
                HybridSearchResult(
                    id=result.id,
                    content=result.content,
                    metadata=result.metadata,
                    similarity=result.similarity,
                    search_type=result.search_type,
                    method_used=SearchMethod.HYBRID
                )
                for result in results
            ]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return await self._search_keyword(query, limit)

    def _choose_search_method(self, query: str, requested_method: SearchMethod) -> SearchMethod:
        """
        Выбор оптимального метода поиска

        Args:
            query: Поисковый запрос
            requested_method: Запрошенный метод

        Returns:
            Выбранный метод
        """
        # Если запрошен конкретный метод, проверяем его доступность
        if requested_method == SearchMethod.SEMANTIC and self._can_do_semantic():
            return SearchMethod.SEMANTIC
        elif requested_method == SearchMethod.HYBRID and self._can_do_hybrid():
            return SearchMethod.HYBRID
        elif requested_method == SearchMethod.KEYWORD:
            return SearchMethod.KEYWORD

        # Автоматический выбор
        if self._can_do_hybrid():
            return SearchMethod.HYBRID
        elif self._can_do_semantic():
            return SearchMethod.SEMANTIC
        else:
            return SearchMethod.KEYWORD

    def _can_do_semantic(self) -> bool:
        """Проверка возможности семантического поиска"""
        return (
            self.vector_store is not None and
            self.embedding_provider is not None and
            SearchMethod.SEMANTIC in self.available_methods
        )

    def _can_do_hybrid(self) -> bool:
        """Проверка возможности гибридного поиска"""
        return (
            self.vector_store is not None and
            self.embedding_provider is not None and
            SearchMethod.SEMANTIC in self.available_methods and
            SearchMethod.KEYWORD in self.available_methods
        )

    async def _get_query_embedding(self, query: str) -> List[float]:
        """Получение эмбеддинга запроса с кэшированием"""
        if not self.enable_caching:
            return await self.embedding_provider.embed_query(query)

        # Проверяем кэш
        cache_key = f"query:{hash(query)}"
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]

        # Создаем новый эмбеддинг
        self.cache_misses += 1
        embedding = await self.embedding_provider.embed_query(query)

        # Кэшируем (с ограничением размера кэша)
        if len(self.embedding_cache) < 1000:  # Максимум 1000 запросов в кэше
            self.embedding_cache[cache_key] = embedding

        return embedding

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики системы"""
        stats = {
            'initialized': self.initialized,
            'available_methods': [m.value for m in self.available_methods],
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
            }
        }

        # Статистика векторного хранилища
        if self.vector_store:
            try:
                vector_stats = await self.vector_store.get_stats()
                stats['vector_store'] = vector_stats
            except Exception as e:
                stats['vector_store'] = {'error': str(e)}

        # Статистика минимального RAG
        if self.minimal_rag:
            try:
                minimal_stats = self.minimal_rag.get_stats()
                stats['minimal_rag'] = minimal_stats
            except Exception as e:
                stats['minimal_rag'] = {'error': str(e)}

        # Статистика эмбеддингов
        if self.embedding_provider:
            stats['embedding_provider'] = {
                'name': getattr(self.embedding_provider, 'name', 'unknown'),
                'dimension': getattr(self.embedding_provider, 'dimension', 0)
            }

        return stats

    async def clear_cache(self) -> None:
        """Очистка кэша эмбеддингов"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Embedding cache cleared")

    async def close(self) -> None:
        """Закрытие всех компонентов"""
        if self.vector_store:
            try:
                await self.vector_store.close()
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")

        self.initialized = False
        logger.info("Hybrid RAG system closed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Глобальный экземпляр для удобства использования
_default_hybrid_rag: Optional[HybridRAGSystem] = None


async def get_hybrid_rag(
    db_path: str = "./rag_db.sqlite",
    embedding_provider: str = "auto"
) -> HybridRAGSystem:
    """Получение глобального экземпляра гибридной RAG системы"""
    global _default_hybrid_rag
    if _default_hybrid_rag is None:
        _default_hybrid_rag = HybridRAGSystem(
            db_path=db_path,
            embedding_provider=embedding_provider
        )
        await _default_hybrid_rag.initialize()
    return _default_hybrid_rag


async def search_hybrid(
    query: str,
    method: SearchMethod = SearchMethod.HYBRID,
    limit: int = 5,
    **kwargs
) -> List[HybridSearchResult]:
    """
    Удобная функция для поиска в гибридной RAG

    Args:
        query: Поисковый запрос
        method: Метод поиска
        limit: Максимальное количество результатов

    Returns:
        Результаты поиска
    """
    rag = await get_hybrid_rag()
    return await rag.search(query, method, limit, **kwargs)


async def add_documents_hybrid(documents: List[Dict[str, Any]]) -> bool:
    """
    Удобная функция для добавления документов

    Args:
        documents: Список документов

    Returns:
        True если успешно
    """
    rag = await get_hybrid_rag()

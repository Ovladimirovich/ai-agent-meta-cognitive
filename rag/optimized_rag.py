"""
Оптимизированная RAG система с улучшенной производительностью для вычисления эмбеддингов и векторного поиска
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
from functools import lru_cache
import faiss
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("sentence_transformers not available, using fallback embeddings")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Результат поиска в RAG системе"""
    sources: List[Dict[str, Any]]
    context: str
    search_metrics: Dict[str, Any]
    query: str
    timestamp: datetime

@dataclass
class SearchMetrics:
    """Метрики поиска"""
    total_chunks_found: int
    relevant_chunks: int
    search_time: float
    reranking_time: Optional[float] = None
    embedding_time: Optional[float] = None
    vector_search_time: Optional[float] = None

class EmbeddingCache:
    """Кэш для эмбеддингов с ограничением размера"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []  # Для LRU
    
    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            # Обновляем порядок доступа
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: np.ndarray):
        if key in self.cache:
            # Обновляем существующий
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Новый элемент
            if len(self.cache) >= self.max_size:
                # Удаляем старый элемент (LRU)
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)

class OptimizedEmbeddingProvider:
    """Оптимизированный провайдер эмбеддингов с кэшированием и параллельной обработкой"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_workers: int = 4):
        self.model_name = model_name
        self.model = None
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = EmbeddingCache(max_size=1000)
        self.dimension = 384  # Для all-MiniLM-L6-v2
        self._initialized = False

    async def initialize(self) -> bool:
        """Асинхронная инициализация модели"""
        if self._initialized:
            return True

        try:
            # Инициализация модели в отдельном потоке
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(self.executor, lambda: SentenceTransformer(self.model_name))
            self._initialized = True
            logger.info(f"Embedding model {self.model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False

    async def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Асинхронное создание эмбеддингов для списка документов с кэшированием"""
        if not self._initialized:
            if not await self.initialize():
                # Fallback: возвращаем случайные эмбеддинги
                return [np.random.rand(self.dimension).astype(np.float32) for _ in texts]

        loop = asyncio.get_event_loop()
        results = []
        uncached_texts = []
        uncached_indices = []

        # Проверяем кэш
        for i, text in enumerate(texts):
            cache_key = f"doc_{hash(text)}"
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                results.append(cached_embedding)
            else:
                results.append(None)  # Заглушка
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            # Вычисляем эмбеддинги для незакэшированных текстов
            uncached_embeddings = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode(uncached_texts, convert_to_numpy=True, show_progress_bar=False)
            )

            # Сохраняем в кэш и обновляем результаты
            for i, (idx, embedding) in enumerate(zip(uncached_indices, uncached_embeddings)):
                cache_key = f"doc_{hash(uncached_texts[i])}"
                self.cache.put(cache_key, embedding.astype(np.float32))
                results[idx] = embedding.astype(np.float32)

        return results

    async def embed_query(self, text: str) -> np.ndarray:
        """Асинхронное создание эмбеддинга для поискового запроса"""
        cache_key = f"query_{hash(text)}"
        cached_embedding = self.cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding

        if not self._initialized:
            if not await self.initialize():
                return np.random.rand(self.dimension).astype(np.float32)

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        )
        embedding = embedding.astype(np.float32)

        # Кэшируем
        self.cache.put(cache_key, embedding)
        return embedding

class OptimizedVectorStore:
    """Оптимизированное векторное хранилище с FAISS"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product для нормализованных векторов (cosine similarity)
        self.documents: List[Dict[str, Any]] = []
        self.metadata: List[Dict[str, Any]] = []
        self.id_map: Dict[int, str] = {}  # FAISS ID -> Document ID
        self.reverse_id_map: Dict[str, int] = {}  # Document ID -> FAISS ID
        self._is_trained = True  # Flat индекс не требует тренировки

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Добавление документов и их эмбеддингов в хранилище"""
        start_idx = len(self.documents)
        new_ids = []
        new_embeddings = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = doc.get('id', f"doc_{start_idx + i}")
            self.documents.append(doc)
            self.metadata.append(doc.get('metadata', {}))
            
            # Сохраняем ID mappings
            faiss_id = len(self.id_map)
            self.id_map[faiss_id] = doc_id
            self.reverse_id_map[doc_id] = faiss_id
            new_ids.append(faiss_id)
            new_embeddings.append(emb)

        # Добавляем эмбеддинги в FAISS индекс
        if new_embeddings:
            embeddings_matrix = np.array(new_embeddings).astype('float32')
            # Нормализуем для Inner Product (cosine similarity)
            faiss.normalize_L2(embeddings_matrix)
            self.index.add(embeddings_matrix)

        logger.info(f"Added {len(documents)} documents to vector store")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """Векторный поиск с использованием FAISS"""
        if self.index.ntotal == 0:
            return []

        # Нормализуем запрос
        query_vector = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Поиск
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx != -1:  # Проверяем валидный индекс
                doc_id = self.id_map.get(idx, f"doc_{idx}")
                doc_idx = self.reverse_id_map.get(doc_id, -1)
                if doc_idx != -1 and doc_idx < len(self.documents):
                    results.append((doc_id, self.documents[doc_idx], float(score)))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики хранилища"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal,
            'dimension': self.dimension,
            'is_trained': self._is_trained
        }

class OptimizedRAGManager:
    """
    Оптимизированный менеджер RAG системы с улучшенной производительностью.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Компоненты
        self.embedder = OptimizedEmbeddingProvider(model_name=embedding_model)
        self.vector_store = OptimizedVectorStore(dimension=self.embedder.dimension)
        self._initialized = False

        # Статистика
        self.stats = {
            'total_queries': 0,
            'total_documents': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    async def initialize(self) -> bool:
        """Асинхронная инициализация RAG системы"""
        if self._initialized:
            return True

        try:
            # Инициализация эмбеддера
            if not await self.embedder.initialize():
                return False

            self._initialized = True
            logger.info("Optimized RAG system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Optimized RAG system: {e}")
            return False

    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Асинхронное добавление документов в базу знаний с оптимизацией
        """
        if not await self.initialize():
            return False

        try:
            start_time = time.time()
            all_texts = []
            all_doc_mappings = []

            # Подготовка всех текстов и маппингов
            for doc in documents:
                doc_id = doc.get('id', f"doc_{len(self.vector_store.documents)}")
                # Разбиение на chunks
                chunks = self._chunk_text(doc['content'])
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    all_texts.append(chunk)
                    all_doc_mappings.append({
                        'original_doc_id': doc_id,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'source_metadata': doc.get('metadata', {}),
                        'added_at': datetime.now().isoformat()
                    })

            # Асинхронное создание эмбеддингов для всех текстов
            embeddings = await self.embedder.embed_documents(all_texts)

            # Подготовка документов для векторного хранилища
            vector_docs = []
            for i, (text, mapping) in enumerate(zip(all_texts, all_doc_mappings)):
                vector_doc = {
                    'id': f"chunk_{len(self.vector_store.documents) + i}",
                    'content': text,
                    'metadata': mapping
                }
                vector_docs.append(vector_doc)

            # Добавление в векторное хранилище
            self.vector_store.add_documents(vector_docs, embeddings)

            self.stats['total_documents'] += len(documents)
            logger.info(f"Added {len(documents)} documents ({len(all_texts)} chunks) in {time.time() - start_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    async def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        threshold: float = 0.1
    ) -> SearchResult:
        """
        Асинхронное выполнение семантического поиска с оптимизацией
        """
        if not await self.initialize():
            return SearchResult(
                sources=[],
                context="",
                search_metrics=SearchMetrics(0, 0, 0.0),
                query=question,
                timestamp=datetime.now()
            )

        start_time = time.time()
        self.stats['total_queries'] += 1

        try:
            # Асинхронное создание эмбеддинга запроса
            query_embedding = await self.embedder.embed_query(question)
            embedding_time = time.time() - start_time

            # Векторный поиск
            vector_search_start = time.time()
            results = self.vector_store.search(query_embedding, top_k * 2)  # Берем больше для фильтрации
            vector_search_time = time.time() - vector_search_start

            search_time = time.time() - start_time

            if not results:
                return SearchResult(
                    sources=[],
                    context="",
                    search_metrics=SearchMetrics(0, 0, search_time),
                    query=question,
                    timestamp=datetime.now()
                )

            # Фильтрация по порогу
            filtered_results = [(doc_id, doc, score) for doc_id, doc, score in results if score >= threshold]

            if not filtered_results:
                # Если нет результатов выше порога, берем топ результатов
                filtered_results = results[:top_k]

            # Реранжирование если включено
            reranking_time = None
            if use_reranking and len(filtered_results) > 1:
                rerank_start = time.time()
                filtered_results = await self._rerank_results(question, filtered_results)
                reranking_time = time.time() - rerank_start

            # Формирование финальных результатов
            sources = []
            context_parts = []

            for doc_id, doc, score in filtered_results[:top_k]:
                source = {
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'similarity': score,  # FAISS возвращает косинусное сходство
                    'rank': len(sources) + 1
                }
                sources.append(source)
                context_parts.append(f"[Источник {len(sources)}]: {doc['content']}")

            context = "\n\n".join(context_parts)

            search_metrics = SearchMetrics(
                total_chunks_found=len(results),
                relevant_chunks=len(sources),
                search_time=search_time,
                reranking_time=reranking_time,
                embedding_time=embedding_time,
                vector_search_time=vector_search_time
            )

            return SearchResult(
                sources=sources,
                context=context,
                search_metrics=search_metrics,
                query=question,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            search_time = time.time() - start_time

            return SearchResult(
                sources=[],
                context="",
                search_metrics=SearchMetrics(0, 0, search_time),
                query=question,
                timestamp=datetime.now()
            )

    def _chunk_text(self, text: str) -> List[str]:
        """Разбиение текста на chunks с оптимизацией"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Если не конец текста, ищем границу предложения или слова
            if end < len(text):
                # Ищем последнюю точку, восклицательный или вопросительный знак
                sentence_end = max(
                    text.rfind('.', start, end + self.chunk_overlap),
                    text.rfind('!', start, end + self.chunk_overlap),
                    text.rfind('?', start, end + self.chunk_overlap)
                )
                
                if sentence_end > start and sentence_end < end:
                    # Нашли границу предложения
                    end = sentence_end + 1
                else:
                    # Ищем последний пробел в пределах overlap
                    last_space = text.rfind(' ', start, end + self.chunk_overlap)
                    if last_space > start:
                        end = last_space

            chunk = text[start:end].strip()
            if chunk:  # Добавляем только непустые chunks
                chunks.append(chunk)

            # Перекрытие для следующего chunk
            start = end - self.chunk_overlap

            # Защита от бесконечного цикла
            if start >= len(text) or len(chunks) > 1000:
                break

        return chunks

    async def _rerank_results(self, question: str, results: List[Tuple[str, Dict[str, Any], float]]) -> List[Tuple[str, Dict[str, Any], float]]:
        """Асинхронное реранжирование результатов с использованием кросс-энкодера"""
        try:
            if SentenceTransformer is None:
                # Если sentence_transformers не доступен, возвращаем оригинальные результаты
                return results
                
            from sentence_transformers import CrossEncoder
            
            # Инициализация кросс-энкодера в отдельном потоке
            loop = asyncio.get_event_loop()
            cross_encoder = await loop.run_in_executor(None, lambda: CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'))

            # Подготовка пар вопрос-документ
            question_doc_pairs = [(question, doc['content']) for _, doc, _ in results]

            # Вычисление scores в отдельном потоке
            scores = await loop.run_in_executor(None, lambda: cross_encoder.predict(question_doc_pairs))

            # Создание пар с новыми скорами
            scored_results = [(doc_id, doc, float(score)) for (doc_id, doc, _), score in zip(results, scores)]

            # Сортировка по score
            scored_results.sort(key=lambda x: x[2], reverse=True)

            return scored_results

        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results

    async def get_stats(self) -> Dict[str, Any]:
        """Асинхронное получение статистики RAG системы"""
        if not self._initialized:
            return {'initialized': False}

        try:
            vector_stats = self.vector_store.get_stats()
            embedder_stats = {
                'cache_size': len(self.embedder.cache.cache),
                'cache_access_order_size': len(self.embedder.cache.access_order)
            }
            return {
                'initialized': True,
                'vector_store': vector_stats,
                'embedding_provider': embedder_stats,
                'stats': self.stats.copy()
            }
        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {'initialized': False, 'error': str(e)}

    async def batch_query(self, questions: List[str], top_k: int = 3) -> List[SearchResult]:
        """Пакетный асинхронный поиск для нескольких запросов"""
        tasks = [self.query(q, top_k) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def stream_query_results(self, question: str, top_k: int = 5) -> AsyncIterator[Dict[str, Any]]:
        """Стриминг результатов поиска"""
        result = await self.query(question, top_k)
        for source in result.sources:
            yield source

    async def clear(self) -> bool:
        """Очистка всех данных"""
        try:
            # Создаем новый векторный индекс
            self.vector_store = OptimizedVectorStore(dimension=self.embedder.dimension)
            # Очищаем кэш
            self.embedder.cache = EmbeddingCache(max_size=10000)
            self.stats = {k: 0 for k in self.stats.keys()}
            logger.info("RAG system cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear RAG system: {e}")
            return False
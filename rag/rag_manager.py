import os
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

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

class RAGManager:
    """
    Менеджер Retrieval-Augmented Generation системы.

    Обеспечивает семантический поиск, извлечение релевантной информации
    и генерацию контекста для AI моделей.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.collection = None
        self.embedder = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Инициализация RAG системы"""
        if self._initialized:
            return True

        try:
            # Условный импорт зависимостей для избежания тяжелых загрузок
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as e:
                logger.warning(f"ChromaDB not available: {e}")
                return False

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                logger.warning(f"SentenceTransformers not available: {e}")
                return False

            # Инициализация ChromaDB
            chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # Создание или получение коллекции
            collection_name = "agent_knowledge_base"
            try:
                self.collection = chroma_client.get_collection(name=collection_name)
            except:
                self.collection = chroma_client.create_collection(name=collection_name)

            # Инициализация эмбеддера
            self.embedder = SentenceTransformer(self.embedding_model)

            self._initialized = True
            logger.info("RAG system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False

    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Добавление документов в базу знаний

        Args:
            documents: Список документов с полями:
                - content: str - текст документа
                - metadata: dict - метаданные (опционально)
                - id: str - уникальный ID (опционально)
        """
        if not await self.initialize():
            return False

        try:
            ids = []
            texts = []
            metadatas = []

            for doc in documents:
                # Генерация ID если не указан
                doc_id = doc.get('id', f"doc_{len(ids)}")

                # Разбиение на chunks
                chunks = self._chunk_text(doc['content'])

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    ids.append(chunk_id)
                    texts.append(chunk)
                    metadatas.append({
                        'source_id': doc_id,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'source_metadata': doc.get('metadata', {}),
                        'added_at': datetime.now().isoformat()
                    })

            # Создание эмбеддингов
            embeddings = self.embedder.encode(texts, show_progress_bar=False)

            # Добавление в коллекцию
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(ids)} chunks from {len(documents)} documents")
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
        Выполнение семантического поиска

        Args:
            question: Поисковый запрос
            top_k: Количество топ результатов
            use_reranking: Использовать реранжирование
            threshold: Порог релевантности
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

        try:
            # Создание эмбеддинга запроса
            query_embedding = self.embedder.encode([question], show_progress_bar=False)[0]

            # Поиск в векторной базе
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k * 2,  # Берем больше для реранжирования
                include=['documents', 'metadatas', 'distances']
            )

            search_time = time.time() - start_time

            if not results['documents'] or not results['documents'][0]:
                return SearchResult(
                    sources=[],
                    context="",
                    search_metrics=SearchMetrics(0, 0, search_time),
                    query=question,
                    timestamp=datetime.now()
                )

            # Обработка результатов
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            # Фильтрация по порогу
            filtered_indices = [i for i, dist in enumerate(distances) if dist <= threshold]

            if not filtered_indices:
                # Если нет результатов выше порога, берем топ результатов
                filtered_indices = list(range(min(len(documents), top_k)))

            # Реранжирование если включено
            reranking_time = None
            if use_reranking and len(filtered_indices) > 1:
                rerank_start = time.time()
                filtered_indices = await self._rerank_results(
                    question, documents, metadatas, filtered_indices
                )
                reranking_time = time.time() - rerank_start

            # Формирование финальных результатов
            sources = []
            context_parts = []

            for idx in filtered_indices[:top_k]:
                source = {
                    'content': documents[idx],
                    'metadata': metadatas[idx],
                    'similarity': 1 - distances[idx],  # Конвертация расстояния в similarity
                    'rank': len(sources) + 1
                }
                sources.append(source)
                context_parts.append(f"[Источник {len(sources)}]: {documents[idx]}")

            context = "\n\n".join(context_parts)

            search_metrics = SearchMetrics(
                total_chunks_found=len(documents),
                relevant_chunks=len(sources),
                search_time=search_time,
                reranking_time=reranking_time
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
        """Разбиение текста на chunks"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Если не конец текста, ищем границу слова
            if end < len(text):
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

    async def _rerank_results(
        self,
        question: str,
        documents: List[str],
        metadatas: List[Dict],
        indices: List[int]
    ) -> List[int]:
        """Реранжирование результатов с использованием кросс-энкодера"""
        try:
            # Условный импорт для избежания загрузки тяжелых моделей
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                logger.warning(f"CrossEncoder not available: {e}")
                return indices

            # Инициализация кросс-энкодера (легковесная модель)
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            # Подготовка пар вопрос-документ
            question_doc_pairs = [
                (question, documents[idx]) for idx in indices
            ]

            # Вычисление scores
            scores = cross_encoder.predict(question_doc_pairs)

            # Сортировка по score
            scored_indices = list(zip(indices, scores))
            scored_indices.sort(key=lambda x: x[1], reverse=True)

            return [idx for idx, _ in scored_indices]

        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return indices

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики RAG системы"""
        if not self._initialized:
            return {'initialized': False}

        try:
            count = self.collection.count()
            return {
                'initialized': True,
                'total_chunks': count,
                'embedding_model': self.embedding_model,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {'initialized': False, 'error': str(e)}

    async def clear_collection(self) -> bool:
        """Очистка коллекции (для тестирования)"""
        try:
            # Условный импорт
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as e:
                logger.warning(f"ChromaDB not available for clearing: {e}")
                return False

            chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            collection_name = "agent_knowledge_base"
            try:
                chroma_client.delete_collection(name=collection_name)
            except:
                pass

            self.collection = chroma_client.create_collection(name=collection_name)
            logger.info("RAG collection cleared")
            return True

        except Exception as e:
            logger.error(f"Failed to clear RAG collection: {e}")
            return False

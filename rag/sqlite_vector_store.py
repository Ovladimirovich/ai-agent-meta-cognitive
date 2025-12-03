"""
SQLite-based векторное хранилище для RAG системы.

Заменяет ChromaDB на встроенное SQLite хранилище с поддержкой:
- Векторного поиска (косинусная близость)
- Полнотекстового поиска (FTS5)
- Гибридного поиска
- ACID транзакций
"""

import sqlite3
import json
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Результат поиска в векторном хранилище"""
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    search_type: str  # "semantic", "keyword", "hybrid"


class SQLiteVectorStore:
    """
    Векторное хранилище на основе SQLite.

    Заменяет ChromaDB с сохранением функциональности:
    - Хранение документов и эмбеддингов
    - Семантический поиск по векторной близости
    - Полнотекстовый поиск
    - Гибридный поиск
    """

    def __init__(self, db_path: str = "./rag_db.sqlite"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.initialized = False

    async def initialize(self) -> bool:
        """Инициализация базы данных"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

            # Создаем таблицы
            self._create_tables()

            self.initialized = True
            logger.info(f"SQLite vector store initialized at {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SQLite vector store: {e}")
            return False

    def _create_tables(self):
        """Создание необходимых таблиц"""
        cursor = self.conn.cursor()

        # Основная таблица документов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,  -- JSON
                embedding TEXT, -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Индекс для быстрого поиска по ID
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id)
        """)

        # Виртуальная таблица для полнотекстового поиска (FTS5)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                content, content='documents', content_rowid='rowid'
            )
        """)

        # Триггеры для синхронизации FTS
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_fts_insert AFTER INSERT ON documents
            BEGIN
                INSERT INTO documents_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_fts_delete AFTER DELETE ON documents
            BEGIN
                DELETE FROM documents_fts WHERE rowid = old.rowid;
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_fts_update AFTER UPDATE ON documents
            BEGIN
                UPDATE documents_fts SET content = new.content WHERE rowid = new.rowid;
            END
        """)

        # Таблица для статистики
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS store_stats (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> bool:
        """
        Добавление документов с эмбеддингами

        Args:
            documents: Список документов [{"text": "...", "metadata": {...}, "id": "..."}]
            embeddings: Соответствующие эмбеддинги

        Returns:
            True если успешно
        """
        if not self.initialized:
            if not await self.initialize():
                return False

        if len(documents) != len(embeddings):
            logger.error("Number of documents and embeddings must match")
            return False

        try:
            cursor = self.conn.cursor()

            for i, doc in enumerate(documents):
                doc_id = doc.get('id', f"doc_{i}_{datetime.now().timestamp()}")
                content = doc.get('text', doc.get('content', ''))
                metadata = json.dumps(doc.get('metadata', {}))
                embedding_json = json.dumps(embeddings[i])

                # Вставляем или обновляем документ
                cursor.execute("""
                    INSERT OR REPLACE INTO documents
                    (id, content, metadata, embedding, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (doc_id, content, metadata, embedding_json))

            self.conn.commit()

            # Обновляем статистику
            await self._update_stats()

            logger.info(f"Added {len(documents)} documents to vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            self.conn.rollback()
            return False

    async def search_semantic(
        self,
        query_embedding: List[float],
        limit: int = 5,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Семантический поиск по векторной близости

        Args:
            query_embedding: Эмбеддинг запроса
            limit: Максимальное количество результатов
            threshold: Минимальная схожесть (0.0 - 1.0)

        Returns:
            Список результатов поиска
        """
        if not self.initialized:
            if not await self.initialize():
                return []

        try:
            cursor = self.conn.cursor()

            # Получаем все документы с эмбеддингами
            cursor.execute("""
                SELECT id, content, metadata, embedding FROM documents
                WHERE embedding IS NOT NULL
            """)

            results = []
            for row in cursor.fetchall():
                try:
                    doc_embedding = json.loads(row['embedding'])
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)

                    if similarity >= threshold:
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}

                        result = SearchResult(
                            id=row['id'],
                            content=row['content'],
                            metadata=metadata,
                            similarity=similarity,
                            search_type="semantic"
                        )
                        results.append(result)

                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Invalid embedding for document {row['id']}: {e}")
                    continue

            # Сортируем по схожести и ограничиваем
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def search_keyword(
        self,
        query: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Полнотекстовый поиск с использованием FTS5

        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if not self.initialized:
            if not await self.initialize():
                return []

        try:
            cursor = self.conn.cursor()

            # FTS5 поиск с ранжированием
            cursor.execute("""
                SELECT d.id, d.content, d.metadata,
                       snippet(documents_fts, 0, '<b>', '</b>', '...', 64) as snippet,
                       bm25(documents_fts) as score
                FROM documents_fts f
                JOIN documents d ON d.rowid = f.rowid
                WHERE documents_fts MATCH ?
                ORDER BY bm25(documents_fts)
                LIMIT ?
            """, (f'"{query}"*', limit))

            results = []
            for row in cursor.fetchall():
                metadata = json.loads(row['metadata']) if row['metadata'] else {}

                result = SearchResult(
                    id=row['id'],
                    content=row['content'],
                    metadata=metadata,
                    similarity=1.0,  # FTS не дает similarity score
                    search_type="keyword"
                )
                # Добавляем snippet в metadata
                result.metadata['snippet'] = row['snippet']
                result.metadata['fts_score'] = row['score']

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    async def search_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Гибридный поиск (семантический + ключевой)

        Args:
            query: Текстовый запрос
            query_embedding: Эмбеддинг запроса
            limit: Максимальное количество результатов
            semantic_weight: Вес семантического поиска
            keyword_weight: Вес ключевого поиска

        Returns:
            Список результатов с комбинированной оценкой
        """
        if not self.initialized:
            if not await self.initialize():
                return []

        try:
            # Выполняем оба типа поиска параллельно
            semantic_results = await self.search_semantic(query_embedding, limit * 2)
            keyword_results = await self.search_keyword(query, limit * 2)

            # Создаем карту результатов по ID
            combined_results = {}

            # Добавляем семантические результаты
            for result in semantic_results:
                combined_results[result.id] = {
                    'result': result,
                    'semantic_score': result.similarity,
                    'keyword_score': 0.0,
                    'has_keyword': False
                }

            # Добавляем ключевые результаты
            for result in keyword_results:
                if result.id in combined_results:
                    # Уже есть семантический результат
                    combined_results[result.id]['keyword_score'] = 1.0
                    combined_results[result.id]['has_keyword'] = True
                else:
                    # Только ключевой результат
                    combined_results[result.id] = {
                        'result': result,
                        'semantic_score': 0.0,
                        'keyword_score': 1.0,
                        'has_keyword': True
                    }

            # Вычисляем комбинированные оценки
            final_results = []
            for item in combined_results.values():
                semantic_score = item['semantic_score']
                keyword_score = item['keyword_score']

                # Комбинированная оценка
                combined_score = (
                    semantic_weight * semantic_score +
                    keyword_weight * keyword_score
                )

                # Создаем новый результат
                result = SearchResult(
                    id=item['result'].id,
                    content=item['result'].content,
                    metadata=item['result'].metadata.copy(),
                    similarity=combined_score,
                    search_type="hybrid"
                )

                # Добавляем информацию о типах поиска
                result.metadata['semantic_score'] = semantic_score
                result.metadata['keyword_score'] = keyword_score
                result.metadata['search_types'] = []
                if semantic_score > 0:
                    result.metadata['search_types'].append('semantic')
                if keyword_score > 0:
                    result.metadata['search_types'].append('keyword')

                final_results.append(result)

            # Сортируем по комбинированной оценке
            final_results.sort(key=lambda x: x.similarity, reverse=True)
            return final_results[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Удаление документа по ID"""
        if not self.initialized:
            return False

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self.conn.commit()

            await self._update_stats()
            logger.info(f"Deleted document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Получение документа по ID"""
        if not self.initialized:
            return None

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, content, metadata, embedding, created_at, updated_at
                FROM documents WHERE id = ?
            """, (doc_id,))

            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'content': row['content'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'embedding': json.loads(row['embedding']) if row['embedding'] else None,
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики хранилища"""
        if not self.initialized:
            return {'initialized': False}

        try:
            cursor = self.conn.cursor()

            # Общее количество документов
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            total_docs = cursor.fetchone()['count']

            # Количество документов с эмбеддингами
            cursor.execute("SELECT COUNT(*) as count FROM documents WHERE embedding IS NOT NULL")
            embedded_docs = cursor.fetchone()['count']

            # Размер базы данных
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()['size']

            return {
                'initialized': True,
                'total_documents': total_docs,
                'embedded_documents': embedded_docs,
                'database_size_bytes': db_size,
                'database_path': self.db_path
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'initialized': False, 'error': str(e)}

    async def clear_all(self) -> bool:
        """Очистка всех данных (для тестирования)"""
        if not self.initialized:
            return False

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM store_stats")
            self.conn.commit()

            logger.info("Cleared all data from vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Вычисление косинусной близости"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        # Вычисляем скалярное произведение
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Вычисляем нормы векторов
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        # Избегаем деления на ноль
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _update_stats(self):
        """Обновление статистики хранилища"""
        try:
            stats = await self.get_stats()
            cursor = self.conn.cursor()

            for key, value in stats.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO store_stats (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, str(value)))

            self.conn.commit()

        except Exception as e:
            logger.warning(f"Failed to update stats: {e}")

    async def close(self):
        """Закрытие соединения с базой данных"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.initialized = False
            logger.info("SQLite vector store closed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

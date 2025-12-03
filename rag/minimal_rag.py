"""
Минималистичный RAG на чистом Python.

Zero-dependency fallback система для поиска по ключевым словам.
Используется когда FastEmbed и SQLite недоступны.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import hashlib
import string
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MinimalDocument:
    """Документ в минималистичной RAG системе"""
    id: str
    content: str
    metadata: Dict[str, Any]
    tokens: List[str]


@dataclass
class MinimalSearchResult:
    """Результат поиска в минималистичной RAG"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    match_type: str = "keyword"


class MinimalRAG:
    """
    Минималистичный RAG на чистом Python.

    Zero-dependency система для поиска по ключевым словам.
    Использует in-memory inverted index для быстрого поиска.

    Особенности:
    - Нет внешних зависимостей
    - In-memory хранение
    - Простая токенизация
    - Keyword-based поиск
    - Быстрая индексация
    """

    def __init__(self):
        self.documents: Dict[str, MinimalDocument] = {}
        self.token_to_docs: Dict[str, set] = {}
        self.initialized = True  # Всегда готов к работе

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Добавление документа в систему

        Args:
            content: Текст документа
            metadata: Метаданные документа

        Returns:
            ID добавленного документа
        """
        # Генерируем ID на основе хэша содержимого
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]

        # Токенизируем текст
        tokens = self._tokenize(content)

        # Создаем документ
        document = MinimalDocument(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            tokens=tokens
        )

        # Сохраняем документ
        self.documents[doc_id] = document

        # Обновляем инвертированный индекс
        for token in tokens:
            if token not in self.token_to_docs:
                self.token_to_docs[token] = set()
            self.token_to_docs[token].add(doc_id)

        logger.debug(f"Added document {doc_id} with {len(tokens)} tokens")
        return doc_id

    def search(self, query: str, limit: int = 5) -> List[MinimalSearchResult]:
        """
        Поиск документов по запросу

        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов

        Returns:
            Список результатов поиска
        """
        if not query.strip():
            return []

        # Токенизируем запрос
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Подсчитываем релевантность для каждого документа
        scores = defaultdict(float)

        for token in query_tokens:
            if token in self.token_to_docs:
                # Начисляем очки за каждый найденный токен
                docs_with_token = self.token_to_docs[token]
                for doc_id in docs_with_token:
                    scores[doc_id] += 1.0

        if not scores:
            return []

        # Преобразуем в список результатов
        results = []
        for doc_id, raw_score in scores.items():
            document = self.documents[doc_id]

            # Нормализуем score по количеству токенов в запросе
            normalized_score = raw_score / len(query_tokens)

            result = MinimalSearchResult(
                id=document.id,
                content=document.content[:500] + '...' if len(document.content) > 500 else document.content,
                metadata=document.metadata.copy(),
                score=normalized_score,
                match_type="keyword"
            )

            results.append(result)

        # Сортируем по score (убыванию)
        results.sort(key=lambda x: x.score, reverse=True)

        # Ограничиваем количество результатов
        return results[:limit]

    def get_document(self, doc_id: str) -> Optional[MinimalDocument]:
        """
        Получение документа по ID

        Args:
            doc_id: ID документа

        Returns:
            Документ или None если не найден
        """
        return self.documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """
        Удаление документа

        Args:
            doc_id: ID документа для удаления

        Returns:
            True если документ был удален
        """
        if doc_id not in self.documents:
            return False

        document = self.documents[doc_id]

        # Удаляем из инвертированного индекса
        for token in document.tokens:
            if token in self.token_to_docs:
                self.token_to_docs[token].discard(doc_id)
                # Удаляем пустые множества
                if not self.token_to_docs[token]:
                    del self.token_to_docs[token]

        # Удаляем документ
        del self.documents[doc_id]

        logger.debug(f"Deleted document {doc_id}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики системы

        Returns:
            Статистика использования
        """
        total_tokens = sum(len(doc.tokens) for doc in self.documents.values())
        unique_tokens = len(self.token_to_docs)

        return {
            'total_documents': len(self.documents),
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'avg_tokens_per_doc': total_tokens / max(1, len(self.documents)),
            'memory_usage': self._estimate_memory_usage()
        }

    def clear_all(self) -> None:
        """Очистка всех данных"""
        self.documents.clear()
        self.token_to_docs.clear()
        logger.info("Cleared all documents from minimal RAG")

    def _tokenize(self, text: str) -> List[str]:
        """
        Простая токенизация текста

        Args:
            text: Исходный текст

        Returns:
            Список токенов
        """
        # Приводим к нижнему регистру
        text = text.lower()

        # Удаляем пунктуацию
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Разбиваем на слова
        tokens = text.split()

        # Фильтруем короткие слова и стоп-слова
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }

        filtered_tokens = [
            token for token in tokens
            if len(token) > 2 and token not in stop_words
        ]

        return filtered_tokens

    def _estimate_memory_usage(self) -> int:
        """
        Оценка использования памяти

        Returns:
            Примерный размер в байтах
        """
        # Грубая оценка
        doc_memory = sum(
            len(doc.content) + len(str(doc.metadata)) + len(doc.tokens) * 10
            for doc in self.documents.values()
        )

        index_memory = sum(
            len(token) + len(doc_ids) * 16  # 16 bytes per set entry
            for token, doc_ids in self.token_to_docs.items()
        )

        return doc_memory + index_memory

    def find_similar_documents(self, doc_id: str, limit: int = 5) -> List[MinimalSearchResult]:
        """
        Поиск похожих документов на основе пересечения токенов

        Args:
            doc_id: ID исходного документа
            limit: Максимальное количество результатов

        Returns:
            Список похожих документов
        """
        if doc_id not in self.documents:
            return []

        source_doc = self.documents[doc_id]
        source_tokens = set(source_doc.tokens)

        # Находим документы с общими токенами
        candidate_docs = set()
        for token in source_tokens:
            if token in self.token_to_docs:
                candidate_docs.update(self.token_to_docs[token])

        candidate_docs.discard(doc_id)  # Убираем исходный документ

        # Вычисляем схожесть (Jaccard similarity)
        results = []
        for candidate_id in candidate_docs:
            candidate_doc = self.documents[candidate_id]
            candidate_tokens = set(candidate_doc.tokens)

            # Jaccard similarity
            intersection = len(source_tokens & candidate_tokens)
            union = len(source_tokens | candidate_tokens)

            if union > 0:
                similarity = intersection / union

                result = MinimalSearchResult(
                    id=candidate_id,
                    content=candidate_doc.content[:500] + '...' if len(candidate_doc.content) > 500 else candidate_doc.content,
                    metadata=candidate_doc.metadata.copy(),
                    score=similarity,
                    match_type="similarity"
                )

                results.append(result)

        # Сортируем по схожести
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def export_data(self) -> Dict[str, Any]:
        """
        Экспорт всех данных для сохранения

        Returns:
            Словарь с данными для сериализации
        """
        return {
            'documents': [
                {
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata
                }
                for doc in self.documents.values()
            ],
            'version': '1.0'
        }

    def import_data(self, data: Dict[str, Any]) -> bool:
        """
        Импорт данных из экспорта

        Args:
            data: Данные из export_data()

        Returns:
            True если импорт успешен
        """
        try:
            if data.get('version') != '1.0':
                logger.warning("Unsupported data version")
                return False

            # Очищаем текущие данные
            self.clear_all()

            # Импортируем документы
            for doc_data in data.get('documents', []):
                self.add_document(
                    content=doc_data['content'],
                    metadata=doc_data.get('metadata', {})
                )

            logger.info(f"Imported {len(data.get('documents', []))} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            return False


# Глобальный экземпляр для удобства использования
_default_minimal_rag: Optional[MinimalRAG] = None


def get_minimal_rag() -> MinimalRAG:
    """Получение глобального экземпляра MinimalRAG"""
    global _default_minimal_rag
    if _default_minimal_rag is None:
        _default_minimal_rag = MinimalRAG()
    return _default_minimal_rag


def search_minimal(query: str, limit: int = 5) -> List[MinimalSearchResult]:
    """
    Удобная функция для поиска в минималистичной RAG

    Args:
        query: Поисковый запрос
        limit: Максимальное количество результатов

    Returns:
        Результаты поиска
    """
    rag = get_minimal_rag()
    return rag.search(query, limit)


def add_to_minimal_rag(content: str, metadata: Dict[str, Any] = None) -> str:
    """
    Удобная функция для добавления документа

    Args:
        content: Содержимое документа
        metadata: Метаданные

    Returns:
        ID добавленного документа
    """
    rag = get_minimal_rag()
    return rag.add_document(content, metadata)

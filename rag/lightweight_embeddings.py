"""
Легковесные провайдеры эмбеддингов для RAG системы.

Заменяют тяжелые зависимости (sentence-transformers, ONNX Runtime)
на легковесные альтернативы с graceful degradation.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import asyncio
import time
from collections import defaultdict
import math
import string

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Абстрактный базовый класс для провайдеров эмбеддингов"""

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Создание эмбеддингов для списка документов"""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Создание эмбеддинга для поискового запроса"""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Инициализация провайдера"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Размерность эмбеддингов"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя провайдера"""
        pass


class FastEmbedProvider(EmbeddingProvider):
    """
    Провайдер на основе FastEmbed.

    Легковесная альтернатива sentence-transformers.
    Использует ONNX модели без проблем совместимости.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self._dimension = 384  # BGE small dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"FastEmbed-{self.model_name}"

    async def initialize(self) -> bool:
        """Инициализация FastEmbed модели"""
        try:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=self.model_name)
            logger.info(f"FastEmbed model {self.model_name} loaded successfully")
            return True
        except ImportError as e:
            logger.warning(f"FastEmbed not available: {e}")
            logger.info("Falling back to TF-IDF provider")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed: {e}")
            return False

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Создание эмбеддингов для документов"""
        if not self.model:
            if not await self.initialize():
                # Fallback to TF-IDF
                tfidf_provider = TFIDFProvider()
                await tfidf_provider.initialize()
                return await tfidf_provider.embed_documents(texts)

        try:
            start_time = time.time()
            embeddings = list(self.model.embed(texts))
            embedding_time = time.time() - start_time

            logger.debug(f"Embedded {len(texts)} documents in {embedding_time:.2f}s")
            return [embedding.tolist() for embedding in embeddings]

        except Exception as e:
            logger.error(f"FastEmbed embedding failed: {e}")
            # Fallback to TF-IDF
            tfidf_provider = TFIDFProvider()
            await tfidf_provider.initialize()
            return await tfidf_provider.embed_documents(texts)

    async def embed_query(self, text: str) -> List[float]:
        """Создание эмбеддинга для запроса"""
        results = await self.embed_documents([text])
        return results[0] if results else []


class TFIDFProvider(EmbeddingProvider):
    """
    TF-IDF провайдер на чистом Python.

    Zero-dependency fallback без внешних библиотек.
    Создает разреженные векторы на основе TF-IDF.
    """

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.document_count = 0
        self._dimension = max_features

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return "TF-IDF"

    async def initialize(self) -> bool:
        """Инициализация TF-IDF (всегда успешна)"""
        logger.info("TF-IDF provider initialized (zero dependencies)")
        return True

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Создание TF-IDF эмбеддингов"""
        if not self.vocab:
            # Первая инициализация словаря
            await self._build_vocabulary(texts)

        embeddings = []
        for text in texts:
            embedding = self._text_to_vector(text)
            embeddings.append(embedding)

        return embeddings

    async def embed_query(self, text: str) -> List[float]:
        """Создание эмбеддинга для запроса"""
        if not self.vocab:
            # Если словарь не построен, создаем простой вектор
            tokens = self._tokenize(text)
            # Возвращаем one-hot вектор для запроса
            vector = [0.0] * self.dimension
            for token in tokens[:self.dimension]:
                if len(vector) < self.dimension:
                    vector[len(vector)] = 1.0
            return vector

        return self._text_to_vector(text)

    def _tokenize(self, text: str) -> List[str]:
        """Простая токенизация текста"""
        # Приводим к нижнему регистру
        text = text.lower()

        # Удаляем пунктуацию
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Разбиваем на слова и фильтруем
        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 2]  # Игнорируем короткие слова

        return tokens

    async def _build_vocabulary(self, texts: List[str]):
        """Построение словаря и IDF scores"""
        # Собираем все токены
        all_tokens = []
        doc_tokens = []

        for text in texts:
            tokens = self._tokenize(text)
            doc_tokens.append(tokens)
            all_tokens.extend(tokens)

        # Строим частотный словарь
        token_freq = defaultdict(int)
        for token in all_tokens:
            token_freq[token] += 1

        # Выбираем топ токенов по частоте
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[:self.max_features]

        # Строим словарь
        self.vocab = {token: i for i, (token, _) in enumerate(top_tokens)}

        # Вычисляем IDF
        self.document_count = len(texts)
        for token in self.vocab.keys():
            doc_count = sum(1 for doc in doc_tokens if token in doc)
            self.idf[token] = math.log(self.document_count / (1 + doc_count))

        logger.info(f"Built TF-IDF vocabulary with {len(self.vocab)} terms")

    def _text_to_vector(self, text: str) -> List[float]:
        """Преобразование текста в TF-IDF вектор"""
        tokens = self._tokenize(text)

        # Вычисляем TF
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        # Создаем вектор
        vector = [0.0] * self.dimension

        for token, freq in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                if idx < self.dimension:
                    # TF-IDF = TF * IDF
                    tf_score = freq / len(tokens)
                    idf_score = self.idf.get(token, 0.0)
                    vector[idx] = tf_score * idf_score

        # Нормализация
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector


class RandomProvider(EmbeddingProvider):
    """
    Провайдер случайных эмбеддингов для тестирования.

    Создает случайные векторы фиксированной размерности.
    Полезен для тестирования без зависимостей.
    """

    def __init__(self, dimension: int = 384, seed: int = 42):
        import random
        random.seed(seed)
        self._random = random
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return "Random"

    async def initialize(self) -> bool:
        """Инициализация (всегда успешна)"""
        return True

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Создание случайных эмбеддингов"""
        embeddings = []
        for _ in texts:
            # Нормализованный случайный вектор
            vector = [self._random.gauss(0, 1) for _ in range(self.dimension)]
            # Нормализация
            norm = math.sqrt(sum(x * x for x in vector))
            vector = [x / norm for x in vector]
            embeddings.append(vector)

        return embeddings

    async def embed_query(self, text: str) -> List[float]:
        """Создание случайного эмбеддинга для запроса"""
        results = await self.embed_documents([text])
        return results[0]


# Глобальные экземпляры провайдеров
_fastembed_provider: Optional[FastEmbedProvider] = None
_tfidf_provider: Optional[TFIDFProvider] = None
_random_provider: Optional[RandomProvider] = None


async def get_embedding_provider(provider_type: str = "auto") -> EmbeddingProvider:
    """
    Фабричная функция для получения провайдера эмбеддингов.

    Args:
        provider_type: Тип провайдера ("fastembed", "tfidf", "random", "auto")

    Returns:
        Экземпляр провайдера эмбеддингов
    """
    global _fastembed_provider, _tfidf_provider, _random_provider

    if provider_type == "fastembed":
        if _fastembed_provider is None:
            _fastembed_provider = FastEmbedProvider()
            await _fastembed_provider.initialize()
        return _fastembed_provider

    elif provider_type == "tfidf":
        if _tfidf_provider is None:
            _tfidf_provider = TFIDFProvider()
            await _tfidf_provider.initialize()
        return _tfidf_provider

    elif provider_type == "random":
        if _random_provider is None:
            _random_provider = RandomProvider()
            await _random_provider.initialize()
        return _random_provider

    else:  # "auto" - автоматический выбор
        # Пытаемся FastEmbed
        if _fastembed_provider is None:
            _fastembed_provider = FastEmbedProvider()
            if await _fastembed_provider.initialize():
                return _fastembed_provider

        # Fallback to TF-IDF
        if _tfidf_provider is None:
            _tfidf_provider = TFIDFProvider()
            await _tfidf_provider.initialize()
        return _tfidf_provider


async def embed_texts(texts: List[str], provider_type: str = "auto") -> List[List[float]]:
    """
    Удобная функция для создания эмбеддингов.

    Args:
        texts: Список текстов для эмбеддинга
        provider_type: Тип провайдера

    Returns:
        Список эмбеддингов
    """
    provider = await get_embedding_provider(provider_type)
    return await provider.embed_documents(texts)


async def embed_query(text: str, provider_type: str = "auto") -> List[float]:
    """
    Удобная функция для эмбеддинга запроса.

    Args:
        text: Текст запроса
        provider_type: Тип провайдера

    Returns:
        Эмбеддинг запроса
    """
    provider = await get_embedding_provider(provider_type)
    return await provider.embed_query(text)

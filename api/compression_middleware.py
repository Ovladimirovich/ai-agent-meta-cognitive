"""
Middleware для сжатия ответов API
"""

import gzip
import zlib
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
import logging

logger = logging.getLogger(__name__)

class CompressionMiddleware:
    """
    Middleware для сжатия HTTP ответов

    Поддерживает gzip и deflate сжатие на основе заголовков Accept-Encoding
    """

    def __init__(self,
                 min_size: int = 1024,
                 compression_level: int = 6,
                 exclude_paths: Optional[list] = None):
        """
        Инициализация middleware сжатия

        Args:
            min_size: Минимальный размер ответа для сжатия (bytes)
            compression_level: Уровень сжатия (1-9)
            exclude_paths: Пути, которые не нужно сжимать
        """
        self.min_size = min_size
        self.compression_level = compression_level
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Обработка запроса с сжатием ответа"""
        # Проверяем, нужно ли сжимать этот путь
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Получаем предпочтения клиента по сжатию
        accept_encoding = request.headers.get("Accept-Encoding", "").lower()

        response = await call_next(request)

        # Проверяем, можно ли сжимать этот ответ
        if not self._should_compress(response, accept_encoding):
            return response

        # Выбираем алгоритм сжатия
        encoding = self._choose_encoding(accept_encoding)

        # Сжимаем ответ
        compressed_body = self._compress_body(response.body, encoding)

        # Проверяем, стоит ли сжимать (экономия места)
        if len(compressed_body) >= len(response.body):
            logger.debug(f"Compression not beneficial for {request.url.path}")
            return response

        # Создаем сжатый ответ
        compressed_response = Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )

        # Добавляем заголовки сжатия
        compressed_response.headers["Content-Encoding"] = encoding
        compressed_response.headers["Content-Length"] = str(len(compressed_body))
        compressed_response.headers["Vary"] = "Accept-Encoding"

        # Удаляем заголовок Content-Length из оригинального ответа
        if "Content-Length" in compressed_response.headers:
            del compressed_response.headers["Content-Length"]

        logger.debug(f"Compressed response for {request.url.path} using {encoding} "
                    f"({len(response.body)} -> {len(compressed_body)} bytes)")

        return compressed_response

    def _should_compress(self, response: Response, accept_encoding: str) -> bool:
        """Проверяет, нужно ли сжимать ответ"""
        # Проверяем поддержку сжатия клиентом
        if not accept_encoding or "identity" in accept_encoding:
            return False

        # Проверяем размер ответа
        if hasattr(response, 'body') and len(response.body) < self.min_size:
            return False

        # Проверяем тип контента (не сжимаем бинарные файлы, изображения и т.д.)
        content_type = response.headers.get("Content-Type", "").lower()
        if any(skip_type in content_type for skip_type in [
            "image/", "video/", "audio/", "application/octet-stream",
            "application/zip", "application/gzip"
        ]):
            return False

        # Проверяем статус код (не сжимаем ошибки)
        if response.status_code >= 400:
            return False

        return True

    def _choose_encoding(self, accept_encoding: str) -> str:
        """Выбирает алгоритм сжатия на основе предпочтений клиента"""
        encodings = [e.strip() for e in accept_encoding.split(",")]

        # Предпочитаем gzip, затем deflate
        if "gzip" in encodings:
            return "gzip"
        elif "deflate" in encodings:
            return "deflate"
        else:
            return "gzip"  # По умолчанию

    def _compress_body(self, body: bytes, encoding: str) -> bytes:
        """Сжимает тело ответа"""
        if encoding == "gzip":
            return gzip.compress(body, compresslevel=self.compression_level)
        elif encoding == "deflate":
            return zlib.compress(body, level=self.compression_level)
        else:
            return body

class BrotliCompressionMiddleware:
    """
    Middleware для Brotli сжатия (если установлена библиотека brotli)
    """

    def __init__(self, **kwargs):
        try:
            import brotli
            self.brotli = brotli
            self.base_middleware = CompressionMiddleware(**kwargs)
            self.base_middleware._compress_body = self._compress_body_brotli
        except ImportError:
            logger.warning("brotli library not installed, falling back to gzip/deflate")
            self.base_middleware = CompressionMiddleware(**kwargs)

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        return await self.base_middleware(request, call_next)

    def _compress_body_brotli(self, body: bytes, encoding: str) -> bytes:
        """Сжимает тело с помощью Brotli"""
        if encoding == "br" and hasattr(self, 'brotli'):
            return self.brotli.compress(body, quality=self.base_middleware.compression_level)
        else:
            # Fallback to base implementation
            return self.base_middleware._compress_body(body, encoding)

def create_compression_middleware(**kwargs) -> CompressionMiddleware:
    """
    Создает middleware сжатия с оптимальными настройками

    Args:
        **kwargs: Параметры для CompressionMiddleware

    Returns:
        CompressionMiddleware: Настроенный middleware
    """
    defaults = {
        "min_size": 1024,  # 1KB
        "compression_level": 6,
        "exclude_paths": ["/health", "/metrics", "/favicon.ico"]
    }
    defaults.update(kwargs)

    # Пробуем Brotli, если доступен
    try:
        import brotli  # noqa
        return BrotliCompressionMiddleware(**defaults)
    except ImportError:
        return CompressionMiddleware(**defaults)

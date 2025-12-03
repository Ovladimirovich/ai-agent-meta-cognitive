"""
Мидлвар для обработки ошибок и обеспечения отказоустойчивости API
"""
import asyncio
import logging
import time
from typing import Callable, Any
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import traceback
import json

from integrations.circuit_breaker import circuit_breaker_registry
from integrations.fallback_manager import graceful_degradation_manager, STANDARD_DEGRADATION_STRATEGIES
from api.logging_config import log_error_with_context, create_safe_error_message

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Мидлвар для обработки ошибок с поддержкой:
    - Глобальной обработки исключений
    - Безопасного логирования
    - Цепных выключателей
    - Плавного ухудшения функциональности
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            # Добавляем информацию о запросе в контекст логирования
            request_id = request.headers.get('X-Request-ID', f"req_{int(time.time() * 1000)}")
            
            response = await call_next(request)
            
            # Логируем успешный запрос
            processing_time = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    'request_id': request_id,
                    'method': request.method,
                    'path': request.url.path,
                    'status_code': response.status_code,
                    'processing_time': processing_time
                }
            )
            
            return response
            
        except HTTPException as e:
            # Обработка HTTP исключений
            processing_time = time.time() - start_time
            logger.warning(
                f"HTTP Exception: {e.status_code} - {e.detail}",
                extra={
                    'request_id': request_id,
                    'method': request.method,
                    'path': request.url.path,
                    'status_code': e.status_code,
                    'detail': e.detail,
                    'processing_time': processing_time
                }
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "HTTP Error",
                    "message": str(e.detail),
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            # Обработка всех остальных исключений
            processing_time = time.time() - start_time
            
            # Создаем безопасное сообщение об ошибке
            safe_message = create_safe_error_message(e)
            
            # Логируем ошибку с полным контекстом
            log_error_with_context(
                e,
                f"Unhandled exception in API: {request.method} {request.url.path}",
                request_id=request_id,
                url=str(request.url),
                method=request.method,
                path=request.url.path,
                processing_time=processing_time
            )
            
            # Пытаемся использовать плавное ухудшение функциональности
            try:
                fallback_response = await self._handle_with_fallback(request, e)
                if fallback_response:
                    return fallback_response
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
            
            # Возвращаем безопасный ответ об ошибке
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": safe_message,
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )
    
    async def _handle_with_fallback(self, request: Request, original_error: Exception) -> Response:
        """
        Обработка ошибки с использованием стратегий плавного ухудшения
        """
        try:
            # Получаем статус ухудшения для текущего сервиса
            degradation_status = graceful_degradation_manager.get_degradation_status()
            
            # Если уровень ухудшения позволяет использовать fallback
            if degradation_status['degradation_level'] in ['degraded', 'unavailable']:
                logger.info(f"Using fallback strategy due to degradation level: {degradation_status['degradation_level']}")
                
                # Пробуем стандартные стратегии ухудшения
                for strategy in STANDARD_DEGRADATION_STRATEGIES:
                    try:
                        result = await strategy()
                        if result:
                            return JSONResponse(
                                status_code=200,
                                content={
                                    "result": result,
                                    "degraded": True,
                                    "message": "Service operating in degraded mode",
                                    "timestamp": time.time()
                                }
                            )
                    except Exception as strategy_error:
                        logger.warning(f"Fallback strategy failed: {strategy_error}")
                        continue
            
            return None
            
        except Exception as fallback_error:
            logger.error(f"Error in fallback handling: {fallback_error}")
            return None

def setup_error_handling_middleware(app):
    """
    Настройка мидлвара обработки ошибок
    """
    app.add_middleware(ErrorHandlingMiddleware)
    logger.info("Error handling middleware configured")

class CircuitBreakerManager:
    """
    Менеджер цепных выключателей для API
    """
    
    @staticmethod
    def get_circuit_breaker_stats():
        """Получение статистики всех circuit breakers"""
        return circuit_breaker_registry.get_all_metrics()
    
    @staticmethod
    def reset_all_circuit_breakers():
        """Сброс всех circuit breakers"""
        circuit_breaker_registry.reset_all()
    
    @staticmethod
    def get_degradation_status():
        """Получение статуса плавного ухудшения"""
        return graceful_degradation_manager.get_degradation_status()

# Глобальный экземпляр менеджера
circuit_breaker_manager = CircuitBreakerManager()
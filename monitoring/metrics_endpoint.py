"""
Эндпоинт для метрик Prometheus
Предоставляет метрики в формате, понятном Prometheus
"""

import logging
from fastapi import APIRouter
from starlette.responses import Response
from monitoring.metrics_collector import metrics_collector


logger = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def get_metrics():
    """
    Получение метрик в формате Prometheus
    
    Returns:
        Response: Метрики в формате Prometheus
    """
    try:
        # Получаем метрики из collector'а
        metrics_data = metrics_collector.get_metrics()
        
        return Response(
            content=metrics_data,
            media_type="text/plain",
            headers={"Content-Type": "text/plain; charset=utf-8"}
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return Response(
            content=f"# Error getting metrics: {str(e)}\n",
            status_code=500,
            media_type="text/plain"
        )


@router.get("/metrics/agent")
async def get_agent_metrics():
    """
    Получение метрик AI агента
    
    Returns:
        Response: Метрики агента в формате Prometheus
    """
    try:
        # В реальной системе здесь будут специфичные метрики агента
        # Для демонстрации возвращаем общие метрики
        metrics_data = metrics_collector.get_metrics()
        
        return Response(
            content=metrics_data,
            media_type="text/plain",
            headers={"Content-Type": "text/plain; charset=utf-8"}
        )
    except Exception as e:
        logger.error(f"Error getting agent metrics: {e}")
        return Response(
            content=f"# Error getting agent metrics: {str(e)}\n",
            status_code=500,
            media_type="text/plain"
        )


@router.get("/metrics/system")
async def get_system_metrics():
    """
    Получение системных метрик
    
    Returns:
        Response: Системные метрики в формате Prometheus
    """
    try:
        # В реальной системе здесь будут специфичные системные метрики
        # Для демонстрации возвращаем общие метрики
        metrics_data = metrics_collector.get_metrics()
        
        return Response(
            content=metrics_data,
            media_type="text/plain",
            headers={"Content-Type": "text/plain; charset=utf-8"}
        )
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return Response(
            content=f"# Error getting system metrics: {str(e)}\n",
            status_code=500,
            media_type="text/plain"
        )


def register_metrics_endpoint(app):
    """
    Регистрация эндпоинта метрик в основном приложении
    
    Args:
        app: FastAPI приложение
    """
    app.include_router(router, prefix="")
    logger.info("Metrics endpoint registered")


if __name__ == "__main__":
    # Пример использования
    print("Metrics endpoint module loaded")
    print(f"Metrics collector enabled: {metrics_collector.enabled}")
    
    # Выводим пример метрик
    if metrics_collector.enabled:
        print("\nSample metrics:")
        print(metrics_collector.get_metrics()[:500] + "..." if len(metrics_collector.get_metrics()) > 500 else metrics_collector.get_metrics())
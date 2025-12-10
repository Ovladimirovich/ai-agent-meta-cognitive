"""
Эндпоинты health checks для AI Агента
Предоставляют информацию о состоянии системы и компонентов
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import psutil
import os

from monitoring.health_check_system import (
    health_registry,
    HealthStatus,
    HealthCheckResult,
    create_system_health_checker,
    create_agent_health_checker
)
from monitoring.alerting_system import alert_manager
from monitoring.metrics_collector import metrics_collector
from api.auth import get_current_user
from agent.core.agent_core import AgentCore


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/health", tags=["health"])


class HealthCheckResponse(BaseModel):
    """Ответ проверки здоровья"""
    name: str = Field(..., description="Название проверки")
    status: str = Field(..., description="Статус проверки")
    response_time: float = Field(..., description="Время ответа")
    message: str = Field(..., description="Сообщение")
    details: Dict[str, Any] = Field(..., description="Детали")
    timestamp: str = Field(..., description="Временная метка")


class HealthSummaryResponse(BaseModel):
    """Сводка по здоровью системы"""
    overall_status: str = Field(..., description="Общий статус")
    total_checks: int = Field(..., description="Всего проверок")
    healthy: int = Field(..., description="Здоровых")
    degraded: int = Field(..., description="Нарушенных")
    unhealthy: int = Field(..., description="Нездоровых")
    groups: Dict[str, Any] = Field(..., description="Группы проверок")
    timestamp: str = Field(..., description="Временная метка")


class HealthStatusResponse(BaseModel):
    """Ответ статуса здоровья"""
    status: str = Field(..., description="Общий статус системы")
    health_score: float = Field(..., description="Оценка здоровья (0.0-1.0)")
    issues_count: int = Field(..., description="Количество проблем")
    last_check: str = Field(..., description="Время последней проверки")
    details: Dict[str, Any] = Field(..., description="Детали проверки")


class SystemHealthResponse(BaseModel):
    """Ответ системного здоровья"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: List[float]
    processes_count: int
    network_connections: int
    timestamp: str


class ActiveAlertsResponse(BaseModel):
    """Ответ активных алертов"""
    alerts_count: int
    alerts: List[Dict[str, Any]]
    timestamp: str


@router.get("/status", response_model=HealthStatusResponse)
async def get_health_status(current_user = Depends(get_current_user)):
    """
    Получение общего статуса здоровья системы

    Args:
        current_user: Аутентифицированный пользователь

    Returns:
        HealthStatusResponse: Статус здоровья системы
    """
    try:
        # Запускаем все проверки
        results = await health_registry.run_all()

        # Получаем сводку
        summary = health_registry.get_summary(results)

        # Рассчитываем health score
        total_checks = summary['total_checks']
        if total_checks > 0:
            health_score = (
                (summary['healthy'] * 1.0 + summary['degraded'] * 0.5) / total_checks
            )
        else:
            health_score = 1.0  # Если нет проверок, считаем систему здоровой

        return HealthStatusResponse(
            status=summary['overall_status'],
            health_score=round(health_score, 2),
            issues_count=summary['degraded'] + summary['unhealthy'],
            last_check=summary['timestamp'],
            details=summary
        )
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting health status: {str(e)}")


@router.get("/detailed", response_model=Dict[str, HealthCheckResponse])
async def get_detailed_health(current_user = Depends(get_current_user)):
    """
    Получение детализированной информации о здоровье

    Args:
        current_user: Аутентифицированный пользователь

    Returns:
        Dict[str, HealthCheckResponse]: Детализированные результаты проверок
    """
    try:
        results = await health_registry.run_all()

        detailed_results = {}
        for name, result in results.items():
            detailed_results[name] = HealthCheckResponse(
                name=result.name,
                status=result.status.value,
                response_time=result.response_time,
                message=result.message,
                details=result.details,
                timestamp=result.timestamp.isoformat()
            )

        return detailed_results
    except Exception as e:
        logger.error(f"Error getting detailed health: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting detailed health: {str(e)}")


@router.get("/summary", response_model=HealthSummaryResponse)
async def get_health_summary(current_user = Depends(get_current_user)):
    """
    Получение сводки по здоровью системы

    Args:
        current_user: Аутентифицированный пользователь

    Returns:
        HealthSummaryResponse: Сводка по здоровью системы
    """
    try:
        results = await health_registry.run_all()
        summary = health_registry.get_summary(results)

        return HealthSummaryResponse(**summary)
    except Exception as e:
        logger.error(f"Error getting health summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting health summary: {str(e)}")


@router.get("/system", response_model=SystemHealthResponse)
async def get_system_health(current_user = Depends(get_current_user)):
    """
    Получение информации о системном здоровье

    Args:
        current_user: Аутентифицированный пользователь

    Returns:
        SystemHealthResponse: Информация о системном здоровье
    """
    try:
        # Сбор системной информации
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        processes_count = len(psutil.pids())
        network_connections = len(psutil.net_connections())

        return SystemHealthResponse(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            load_average=load_avg,
            processes_count=processes_count,
            network_connections=network_connections,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system health: {str(e)}")


@router.get("/agent", response_model=HealthCheckResponse)
async def get_agent_health(current_user = Depends(get_current_user)):
    """
    Получение информации о здоровье AI агента

    Args:
        current_user: Аутентифицированный пользователь

    Returns:
        HealthCheckResponse: Информация о здоровье агента
    """
    try:
        # В реальной системе здесь будет проверка состояния агента
        # Для демонстрации создаем временную проверку
        result = await create_system_health_checker().check()

        return HealthCheckResponse(
            name=result.name,
            status=result.status.value,
            response_time=result.response_time,
            message=result.message,
            details=result.details,
            timestamp=result.timestamp.isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting agent health: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting agent health: {str(e)}")


@router.get("/metrics", response_model=Dict[str, Any])
async def get_health_metrics(current_user = Depends(get_current_user)):
    """
    Получение метрик здоровья системы

    Args:
        current_user: Аутентифицированный пользователь

    Returns:
        Dict[str, Any]: Метрики здоровья системы
    """
    try:
        # Собираем метрики здоровья
        results = await health_registry.run_all()

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "summary": health_registry.get_summary(results)
        }

        for name, result in results.items():
            metrics["checks"][name] = {
                "status": result.status.value,
                "response_time": result.response_time,
                "critical": result.critical
            }

        # Добавляем метрики из collector'а если он доступен
        if metrics_collector.enabled:
            metrics["prometheus_metrics"] = metrics_collector.get_metrics()

        return metrics
    except Exception as e:
        logger.error(f"Error getting health metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting health metrics: {str(e)}")


@router.get("/alerts/active", response_model=ActiveAlertsResponse)
async def get_active_alerts(current_user = Depends(get_current_user)):
    """
    Получение информации об активных алертах

    Args:
        current_user: Аутентифицированный пользователь

    Returns:
        ActiveAlertsResponse: Информация об активных алертах
    """
    try:
        active_alerts = alert_manager.get_active_alerts()

        alerts_data = []
        for alert in active_alerts:
            alerts_data.append({
                "id": alert.id,
                "name": alert.name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "description": alert.description,
                "start_time": alert.start_time.isoformat(),
                "labels": alert.labels
            })

        return ActiveAlertsResponse(
            alerts_count=len(active_alerts),
            alerts=alerts_data,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting active alerts: {str(e)}")


@router.post("/check/{check_name}")
async def run_specific_check(check_name: str, current_user = Depends(get_current_user)):
    """
    Запуск конкретной проверки здоровья

    Args:
        check_name: Название проверки
        current_user: Аутентифицированный пользователь

    Returns:
        HealthCheckResponse: Результат проверки
    """
    try:
        result = await health_registry.run_check(check_name)

        if result is None:
            raise HTTPException(status_code=404, detail=f"Health check '{check_name}' not found")

        return HealthCheckResponse(
            name=result.name,
            status=result.status.value,
            response_time=result.response_time,
            message=result.message,
            details=result.details,
            timestamp=result.timestamp.isoformat()
        )
    except Exception as e:
        logger.error(f"Error running specific check {check_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error running check {check_name}: {str(e)}")


@router.get("/groups/{group_name}", response_model=Dict[str, HealthCheckResponse])
async def run_group_checks(group_name: str, current_user = Depends(get_current_user)):
    """
    Запуск проверок для конкретной группы

    Args:
        group_name: Название группы
        current_user: Аутентифицированный пользователь

    Returns:
        Dict[str, HealthCheckResponse]: Результаты проверок группы
    """
    try:
        results = await health_registry.run_group(group_name)

        if not results:
            raise HTTPException(status_code=404, detail=f"Health check group '{group_name}' not found or empty")

        detailed_results = {}
        for name, result in results.items():
            detailed_results[name] = HealthCheckResponse(
                name=result.name,
                status=result.status.value,
                response_time=result.response_time,
                message=result.message,
                details=result.details,
                timestamp=result.timestamp.isoformat()
            )

        return detailed_results
    except Exception as e:
        logger.error(f"Error running group checks {group_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error running group checks {group_name}: {str(e)}")


@router.get("/liveness")
async def liveness_probe():
    """
    Liveness probe для Kubernetes
    Возвращает 200 если приложение живое
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@router.get("/readiness")
async def readiness_probe():
    """
    Readiness probe для Kubernetes
    Возвращает 200 если приложение готово принимать запросы
    """
    try:
        # Проверяем базовое состояние системы
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        # Система считается готовой если CPU и память в норме
        if cpu_percent < 95 and memory_percent < 95:
            return {
                "status": "ready",
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent
            }
        else:
            raise HTTPException(
                status_code=503,
                detail=f"System not ready - CPU: {cpu_percent}%, Memory: {memory_percent}%"
            )
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        raise HTTPException(status_code=503, detail=f"System not ready: {str(e)}")


# Функция для регистрации эндпоинтов в основном приложении
def register_health_endpoints(app):
    """
    Регистрация эндпоинтов health checks в основном приложении

    Args:
        app: FastAPI приложение
    """
    app.include_router(router)
    logger.info("Health check endpoints registered")


# Инициализация базовых проверок
def initialize_health_checks(agent_core: 'AgentCore' = None):
    """
    Инициализация базовых проверок здоровья

    Args:
        agent_core: Экземпляр ядра агента (опционально)
    """
    # Добавляем системную проверку
    system_checker = create_system_health_checker(
        cpu_threshold=90.0,
        memory_threshold=90.0,
        disk_threshold=90.0
    )
    health_registry.register_checker(system_checker, "infrastructure")

    # Если доступно ядро агента, добавляем проверку агента
    if agent_core:
        agent_checker = create_agent_health_checker(agent_core)
        health_registry.register_checker(agent_checker, "ai-agent")

    logger.info("Basic health checks initialized")


if __name__ == "__main__":
    # Пример использования
    import asyncio

    async def test_health_endpoints():
        print("Testing health endpoints...")

        # Запускаем проверки
        results = await health_registry.run_all()
        summary = health_registry.get_summary(results)

        print(f"Overall status: {summary['overall_status']}")
        print(f"Healthy: {summary['healthy']}, Degraded: {summary['degraded']}, Unhealthy: {summary['unhealthy']}")

    # asyncio.run(test_health_endpoints())

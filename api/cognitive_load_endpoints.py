"""
API эндпоинты для анализа и визуализации когнитивной нагрузки
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from agent.meta_cognitive.cognitive_load_analyzer import (
    CognitiveLoadAnalyzer, CognitiveLoadVisualizer,
    CognitiveLoadMetrics, LoadLevel
)
from agent.core.agent_core import AgentCore
# from api.auth import get_current_user # Закомментирован для версии без аутентификации

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cognitive-load", tags=["cognitive-load"])

# Глобальный экземпляр анализатора (в реальном приложении лучше использовать DI)
analyzer = CognitiveLoadAnalyzer()
visualizer = CognitiveLoadVisualizer(analyzer)


class CognitiveLoadMetricsRequest(BaseModel):
    """Запрос метрик когнитивной нагрузки"""
    response_time: float = Field(..., description="Время ответа", gt=0)
    processing_time: float = Field(..., description="Время обработки", gt=0)
    memory_usage: float = Field(..., description="Использование памяти", ge=0, le=1)
    cpu_usage: float = Field(..., description="Использование CPU", ge=0, le=1)
    active_tasks: int = Field(..., description="Активные задачи", ge=0)
    confidence_level: float = Field(..., description="Уровень уверенности", ge=0, le=1)
    error_rate: float = Field(..., description="Уровень ошибок", ge=0, le=1)
    complexity_score: float = Field(..., description="Оценка сложности задачи", ge=0, le=1)
    resource_pressure: float = Field(..., description="Давление на ресурсы", ge=0, le=1)


class CognitiveLoadAnalysisResponse(BaseModel):
    """Ответ с анализом когнитивной нагрузки"""
    load_level: str = Field(..., description="Уровень нагрузки")
    load_score: float = Field(..., description="Оценка нагрузки (0.0-1.0)")
    contributing_factors: List[str] = Field(..., description="Факторы нагрузки")
    recommendations: List[str] = Field(..., description="Рекомендации")
    timestamp: str = Field(..., description="Время анализа")


class LoadSummaryResponse(BaseModel):
    """Ответ с сводкой нагрузки"""
    period: str = Field(..., description="Период анализа")
    total_measurements: int = Field(..., description="Всего измерений")
    average_load: float = Field(..., description="Средняя нагрузка")
    peak_load: float = Field(..., description="Пиковая нагрузка")
    load_distribution: Dict[str, int] = Field(..., description="Распределение нагрузки")
    trend_direction: str = Field(..., description="Направление тренда")


class TimeSeriesDataResponse(BaseModel):
    """Ответ с данными временного ряда"""
    timestamps: List[str] = Field(..., description="Временные метки")
    load_scores: List[float] = Field(..., description="Оценки нагрузки")
    load_levels: List[str] = Field(..., description="Уровни нагрузки")
    response_times: List[float] = Field(..., description="Времена ответа")
    memory_usage: List[float] = Field(..., description="Использование памяти")
    cpu_usage: List[float] = Field(..., description="Использование CPU")


class DistributionDataResponse(BaseModel):
    """Ответ с данными распределения"""
    load_levels: List[str] = Field(..., description="Уровни нагрузки")
    counts: List[int] = Field(..., description="Количество")
    percentages: List[float] = Field(..., description="Проценты")


class RealTimeDataResponse(BaseModel):
    """Ответ с данными в реальном времени"""
    current_load_score: float = Field(..., description="Текущая оценка нагрузки")
    current_load_level: str = Field(..., description="Текущий уровень нагрузки")
    current_metrics: Dict[str, Any] = Field(..., description="Текущие метрики")
    trend: str = Field(..., description="Тренд")
    recommendations: List[str] = Field(..., description="Рекомендации")


@router.post("/analyze", response_model=CognitiveLoadAnalysisResponse)
async def analyze_cognitive_load(
    metrics_request: CognitiveLoadMetricsRequest
):
    """
    Анализ когнитивной нагрузки на основе переданных метрик

    Args:
        metrics_request: Метрики для анализа

    Returns:
        CognitiveLoadAnalysisResponse: Результат анализа
    """
    try:
        # Создание объекта метрик
        metrics = CognitiveLoadMetrics(
            response_time=metrics_request.response_time,
            processing_time=metrics_request.processing_time,
            memory_usage=metrics_request.memory_usage,
            cpu_usage=metrics_request.cpu_usage,
            active_tasks=metrics_request.active_tasks,
            confidence_level=metrics_request.confidence_level,
            error_rate=metrics_request.error_rate,
            complexity_score=metrics_request.complexity_score,
            resource_pressure=metrics_request.resource_pressure,
            timestamp=datetime.now()
        )

        # Выполнение анализа
        analysis = await analyzer.analyze_cognitive_load(metrics)

        return CognitiveLoadAnalysisResponse(
            load_level=analysis.load_level.value,
            load_score=analysis.load_score,
            contributing_factors=analysis.contributing_factors,
            recommendations=analysis.recommendations,
            timestamp=analysis.metrics.timestamp.isoformat()
        )

    except Exception as e:
        logger.error(f"Error analyzing cognitive load: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing cognitive load: {str(e)}")


@router.get("/summary", response_model=LoadSummaryResponse)
async def get_load_summary(
    hours: int = Query(24, description="Количество часов для анализа", ge=1, le=168)
):
    """
    Получение сводки по когнитивной нагрузке за указанный период

    Args:
        hours: Количество часов для анализа (1-168)

    Returns:
        LoadSummaryResponse: Сводка по нагрузке
    """
    try:
        summary = analyzer.get_load_summary(hours=hours)

        return LoadSummaryResponse(
            period=summary['period'],
            total_measurements=summary['total_measurements'],
            average_load=summary['average_load'],
            peak_load=summary['peak_load'],
            load_distribution=summary['load_distribution'],
            trend_direction=summary['trend_direction']
        )

    except Exception as e:
        logger.error(f"Error getting load summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting load summary: {str(e)}")


@router.get("/time-series", response_model=TimeSeriesDataResponse)
async def get_time_series_data(
    hours: int = Query(24, description="Количество часов для анализа", ge=1, le=168)
):
    """
    Получение данных временного ряда когнитивной нагрузки

    Args:
        hours: Количество часов для анализа (1-168)

    Returns:
        TimeSeriesDataResponse: Данные временного ряда
    """
    try:
        data = visualizer.generate_time_series_data(hours=hours)

        return TimeSeriesDataResponse(
            timestamps=data['timestamps'],
            load_scores=data['load_scores'],
            load_levels=data['load_levels'],
            response_times=data['response_times'],
            memory_usage=data['memory_usage'],
            cpu_usage=data['cpu_usage']
        )

    except Exception as e:
        logger.error(f"Error getting time series data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting time series data: {str(e)}")


@router.get("/distribution", response_model=DistributionDataResponse)
async def get_distribution_data():
    """
    Получение данных распределения когнитивной нагрузки

    Returns:
        DistributionDataResponse: Данные распределения
    """
    try:
        data = visualizer.generate_distribution_data()

        return DistributionDataResponse(
            load_levels=data['load_levels'],
            counts=data['counts'],
            percentages=data['percentages']
        )

    except Exception as e:
        logger.error(f"Error getting distribution data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting distribution data: {str(e)}")


@router.get("/real-time", response_model=RealTimeDataResponse)
async def get_real_time_data():
    """
    Получение данных когнитивной нагрузки в реальном времени

    Returns:
        RealTimeDataResponse: Данные в реальном времени
    """
    try:
        data = visualizer.generate_real_time_data()

        return RealTimeDataResponse(
            current_load_score=data['current_load_score'],
            current_load_level=data['current_load_level'],
            current_metrics=data['current_metrics'],
            trend=data['trend'],
            recommendations=data['recommendations']
        )

    except Exception as e:
        logger.error(f"Error getting real-time data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting real-time data: {str(e)}")


@router.get("/recommendations")
async def get_cognitive_load_recommendations():
    """
    Получение рекомендаций по управлению когнитивной нагрузкой

    Returns:
        List[str]: Рекомендации
    """
    try:
        # Получение последних метрик для генерации рекомендаций
        if analyzer.load_history:
            latest_metrics = analyzer.load_history[-1]
            current_score = analyzer.calculate_load_score(latest_metrics)
            current_level = analyzer.determine_load_level(current_score)
            factors = analyzer.identify_contributing_factors(latest_metrics)
            recommendations = analyzer.generate_recommendations(current_level, factors)
        else:
            recommendations = [
                "Нет данных для анализа нагрузки",
                "Начните отправлять метрики для получения рекомендаций"
            ]

        return {"recommendations": recommendations}

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")


@router.get("/health-indicators")
async def get_cognitive_health_indicators():
    """
    Получение индикаторов когнитивного здоровья

    Returns:
        Dict: Индикаторы здоровья
    """
    try:
        if not analyzer.load_history:
            return {
                "status": "no_data",
                "indicators": {}
            }

        latest_metrics = analyzer.load_history[-1]
        current_score = analyzer.calculate_load_score(latest_metrics)
        current_level = analyzer.determine_load_level(current_score)

        # Рассчитываем различные индикаторы
        indicators = {
            "load_level": current_level.value,
            "load_score": current_score,
            "response_time_health": "good" if latest_metrics.response_time < 2.0 else "poor",
            "memory_health": "good" if latest_metrics.memory_usage < 0.7 else "poor",
            "cpu_health": "good" if latest_metrics.cpu_usage < 0.8 else "poor",
            "confidence_health": "good" if latest_metrics.confidence_level > 0.5 else "poor",
            "error_health": "good" if latest_metrics.error_rate < 0.1 else "poor",
            "complexity_health": "good" if latest_metrics.complexity_score < 0.6 else "poor"
        }

        return {
            "status": "healthy" if current_level in [LoadLevel.LOW, LoadLevel.MEDIUM] else "unhealthy",
            "indicators": indicators
        }

    except Exception as e:
        logger.error(f"Error getting health indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting health indicators: {str(e)}")


# Функция для интеграции с основным API
def register_cognitive_load_endpoints(main_app):
    """
    Регистрация эндпоинтов когнитивной нагрузки в основном приложении

    Args:
        main_app: Основное FastAPI приложение
    """
    main_app.include_router(router)
    logger.info("Cognitive load endpoints registered")

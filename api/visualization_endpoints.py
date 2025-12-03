"""
API эндпоинты для новых функций визуализации
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from agent.meta_cognitive.cognitive_load_analyzer import CognitiveLoadVisualizer, CognitiveLoadAnalyzer
from agent.learning.adaptation_engine import AdaptationEngine
from api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization", tags=["visualization"])


class VisualizationDataResponse(BaseModel):
    """Ответ с данными визуализации"""
    visualization_type: str = Field(..., description="Тип визуализации")
    data: Dict[str, Any] = Field(..., description="Данные для визуализации")
    metadata: Dict[str, Any] = Field(..., description="Метаданные")
    timestamp: str = Field(..., description="Временная метка")


class ChartDataResponse(BaseModel):
    """Ответ с данными для диаграмм"""
    chart_type: str = Field(..., description="Тип диаграммы")
    labels: List[str] = Field(..., description="Метки осей")
    datasets: List[Dict[str, Any]] = Field(..., description="Наборы данных")
    options: Dict[str, Any] = Field(..., description="Опции диаграммы")


class DashboardDataResponse(BaseModel):
    """Ответ с данными для дашборда"""
    widgets: List[Dict[str, Any]] = Field(..., description="Виджеты дашборда")
    layout: Dict[str, Any] = Field(..., description="Макет дашборда")
    last_updated: str = Field(..., description="Время последнего обновления")


# Глобальные экземпляры анализаторов (в реальности должны быть внедрены через DI)
analyzer = CognitiveLoadAnalyzer()
visualizer = CognitiveLoadVisualizer(analyzer)
adaptation_engine: AdaptationEngine = None  # Будет инициализирован позже


@router.get("/cognitive-load-time-series", response_model=VisualizationDataResponse)
async def get_cognitive_load_time_series(
    hours: int = Query(24, description="Количество часов для анализа", ge=1, le=168),
    current_user = Depends(get_current_user)
):
    """
    Получение временного ряда когнитивной нагрузки
    
    Args:
        hours: Количество часов для анализа (1-168)
        current_user: Аутентифицированный пользователь
        
    Returns:
        VisualizationDataResponse: Данные временного ряда
    """
    try:
        data = visualizer.generate_time_series_data(hours=hours)
        
        return VisualizationDataResponse(
            visualization_type="time_series",
            data=data,
            metadata={
                "time_range_hours": hours,
                "data_points_count": len(data['timestamps']),
                "granularity": "minute" if hours <= 24 else "hour"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting cognitive load time series: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting time series data: {str(e)}")


@router.get("/cognitive-load-distribution", response_model=VisualizationDataResponse)
async def get_cognitive_load_distribution(current_user = Depends(get_current_user)):
    """
    Получение распределения когнитивной нагрузки
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        VisualizationDataResponse: Данные распределения
    """
    try:
        data = visualizer.generate_distribution_data()
        
        return VisualizationDataResponse(
            visualization_type="distribution",
            data=data,
            metadata={
                "total_measurements": sum(data['counts']),
                "distribution_type": "categorical"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting cognitive load distribution: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting distribution data: {str(e)}")


@router.get("/real-time-cognitive-load", response_model=VisualizationDataResponse)
async def get_real_time_cognitive_load(current_user = Depends(get_current_user)):
    """
    Получение данных когнитивной нагрузки в реальном времени
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        VisualizationDataResponse: Данные в реальном времени
    """
    try:
        data = visualizer.generate_real_time_data()
        
        return VisualizationDataResponse(
            visualization_type="real_time",
            data=data,
            metadata={
                "refresh_interval_seconds": 5,
                "trend_direction": data['trend']
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting real-time cognitive load: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting real-time data: {str(e)}")


@router.get("/advanced-cognitive-visualization", response_model=VisualizationDataResponse)
async def get_advanced_cognitive_visualization(
    hours: int = Query(24, description="Количество часов для анализа", ge=1, le=168),
    current_user = Depends(get_current_user)
):
    """
    Получение расширенных данных визуализации когнитивной нагрузки
    
    Args:
        hours: Количество часов для анализа (1-168)
        current_user: Аутентифицированный пользователь
        
    Returns:
        VisualizationDataResponse: Расширенные данные визуализации
    """
    try:
        # Используем метод из обновленного CognitiveLoadVisualizer
        advanced_data = visualizer.generate_advanced_visualization_data(hours=hours)
        
        return VisualizationDataResponse(
            visualization_type="advanced",
            data=advanced_data,
            metadata={
                "time_range_hours": hours,
                "analysis_types": [
                    "time_series", "distribution", "peak_analysis", 
                    "trend_analysis", "correlation_analysis", "prediction"
                ],
                "total_data_points": len(advanced_data['time_series'])
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting advanced cognitive visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting advanced visualization data: {str(e)}")


@router.get("/adaptation-history-chart", response_model=ChartDataResponse)
async def get_adaptation_history_chart(current_user = Depends(get_current_user)):
    """
    Получение данных для диаграммы истории адаптаций
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        ChartDataResponse: Данные для диаграммы
    """
    try:
        if not adaptation_engine:
            return ChartDataResponse(
                chart_type="bar",
                labels=["No Data"],
                datasets=[{
                    "label": "Adaptations",
                    "data": [0],
                    "backgroundColor": ["#ccc"]
                }],
                options={}
            )
        
        # Получаем историю адаптаций
        history = adaptation_engine.get_adaptation_history(limit=50)
        
        if not history:
            return ChartDataResponse(
                chart_type="bar",
                labels=["No Data"],
                datasets=[{
                    "label": "Adaptations",
                    "data": [0],
                    "backgroundColor": ["#ccc"]
                }],
                options={}
            )
        
        # Подготавливаем данные для диаграммы
        labels = [item['created_at'] for item in history]
        adaptation_types = [item['type'] for item in history]
        
        # Подсчитываем количество адаптаций по типам
        type_counts = {}
        for ad_type in adaptation_types:
            type_counts[ad_type] = type_counts.get(ad_type, 0) + 1
        
        # Формируем наборы данных
        datasets = [{
            "label": "Adaptation Count",
            "data": list(type_counts.values()),
            "backgroundColor": [
                "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", 
                "#9966FF", "#FF9F40", "#FF6384", "#C9CBCF"
            ]
        }]
        
        return ChartDataResponse(
            chart_type="doughnut",
            labels=list(type_counts.keys()),
            datasets=datasets,
            options={
                "responsive": True,
                "maintainAspectRatio": False
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting adaptation history chart: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting adaptation chart: {str(e)}")


@router.get("/performance-trend-chart", response_model=ChartDataResponse)
async def get_performance_trend_chart(
    days: int = Query(30, description="Количество дней для анализа", ge=1, le=365),
    current_user = Depends(get_current_user)
):
    """
    Получение данных для диаграммы тренда производительности
    
    Args:
        days: Количество дней для анализа (1-365)
        current_user: Аутентифицированный пользователь
        
    Returns:
        ChartDataResponse: Данные для диаграммы тренда
    """
    try:
        # Симулируем данные производительности
        # В реальности эти данные будут извлекаться из соответствующих источников
        import random
        from datetime import timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = []
        performance_scores = []
        load_scores = []
        
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            performance_scores.append(round(random.uniform(0.4, 1.0), 2))
            load_scores.append(round(random.uniform(0.1, 0.9), 2))
            current_date += timedelta(days=1)
        
        datasets = [
            {
                "label": "Performance Score",
                "data": performance_scores,
                "borderColor": "#4BC0C0",
                "backgroundColor": "rgba(75, 192, 192, 0.1)",
                "tension": 0.4,
                "fill": False
            },
            {
                "label": "Load Score",
                "data": load_scores,
                "borderColor": "#FF6384",
                "backgroundColor": "rgba(255, 99, 132, 0.1)",
                "tension": 0.4,
                "fill": False
            }
        ]
        
        return ChartDataResponse(
            chart_type="line",
            labels=dates,
            datasets=datasets,
            options={
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {
                        "min": 0,
                        "max": 1
                    }
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting performance trend chart: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance trend chart: {str(e)}")


@router.get("/dashboard-data", response_model=DashboardDataResponse)
async def get_dashboard_data(current_user = Depends(get_current_user)):
    """
    Получение данных для дашборда
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        DashboardDataResponse: Данные для дашборда
    """
    try:
        # Собираем данные для различных виджетов дашборда
        widgets = []
        
        # Виджет: Текущая когнитивная нагрузка
        real_time_data = visualizer.generate_real_time_data()
        widgets.append({
            "id": "current_load",
            "type": "gauge",
            "title": "Текущая когнитивная нагрузка",
            "value": real_time_data['current_load_score'],
            "max": 1.0,
            "status": real_time_data['current_load_level'],
            "trend": real_time_data['trend']
        })
        
        # Виджет: Распределение нагрузки
        distribution_data = visualizer.generate_distribution_data()
        widgets.append({
            "id": "load_distribution",
            "type": "pie_chart",
            "title": "Распределение уровней нагрузки",
            "data": {
                "labels": distribution_data['load_levels'],
                "values": distribution_data['counts']
            }
        })
        
        # Виджет: Активные адаптации
        if adaptation_engine:
            active_adaptations = adaptation_engine.get_active_adaptations()
            widgets.append({
                "id": "active_adaptations",
                "type": "metric",
                "title": "Активные адаптации",
                "value": len(active_adaptations),
                "description": "Количество текущих адаптаций системы"
            })
        else:
            widgets.append({
                "id": "active_adaptations",
                "type": "metric",
                "title": "Активные адаптации",
                "value": 0,
                "description": "Система адаптации не инициализирована"
            })
        
        # Виджет: Рекомендации
        recommendations = []
        if real_time_data['current_load_level'] == 'high' or real_time_data['current_load_level'] == 'critical':
            recommendations.append("Рассмотрите снижение нагрузки на систему")
        if real_time_data['trend'] == 'increasing':
            recommendations.append("Наблюдается рост нагрузки - мониторьте систему")
        if not recommendations:
            recommendations.append("Система работает в нормальном режиме")
            
        widgets.append({
            "id": "recommendations",
            "type": "list",
            "title": "Рекомендации",
            "items": recommendations
        })
        
        # Макет дашборда
        layout = {
            "grid_columns": 12,
            "widgets": [
                {"id": "current_load", "position": {"x": 0, "y": 0, "w": 3, "h": 2}},
                {"id": "load_distribution", "position": {"x": 3, "y": 0, "w": 5, "h": 2}},
                {"id": "active_adaptations", "position": {"x": 8, "y": 0, "w": 2, "h": 1}},
                {"id": "recommendations", "position": {"x": 8, "y": 1, "w": 4, "h": 1}}
            ]
        }
        
        return DashboardDataResponse(
            widgets=widgets,
            layout=layout,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dashboard data: {str(e)}")


@router.get("/correlation-heatmap", response_model=VisualizationDataResponse)
async def get_correlation_heatmap(
    hours: int = Query(24, description="Количество часов для анализа", ge=1, le=168),
    current_user = Depends(get_current_user)
):
    """
    Получение данных для тепловой карты корреляций
    
    Args:
        hours: Количество часов для анализа (1-168)
        current_user: Аутентифицированный пользователь
        
    Returns:
        VisualizationDataResponse: Данные тепловой карты
    """
    try:
        # Используем метод из обновленного CognitiveLoadVisualizer
        advanced_data = visualizer.generate_advanced_visualization_data(hours=hours)
        correlation_data = advanced_data.get('correlation_analysis', {})
        
        # Подготавливаем данные для тепловой карты
        metrics = list(set(
            [key.split('_vs_')[0].replace('load_', '') for key in correlation_data.keys() if 'vs' in key] +
            [key.split('_vs_')[1].replace('load_', '') for key in correlation_data.keys() if 'vs' in key]
        ))
        
        # Создаем матрицу корреляций
        matrix = []
        for row_metric in metrics:
            row = []
            for col_metric in metrics:
                if row_metric == col_metric:
                    row.append(1.0)  # Корреляция метрики с собой
                else:
                    # Ищем значение корреляции в данных
                    corr_key1 = f"{row_metric}_vs_{col_metric}"
                    corr_key2 = f"{col_metric}_vs_{row_metric}"
                    corr_value = correlation_data.get(corr_key1, correlation_data.get(corr_key2, 0))
                    row.append(corr_value)
            matrix.append(row)
        
        heatmap_data = {
            "metrics": metrics,
            "correlation_matrix": matrix,
            "values": [list(row) for row in matrix]
        }
        
        return VisualizationDataResponse(
            visualization_type="correlation_heatmap",
            data=heatmap_data,
            metadata={
                "time_range_hours": hours,
                "metrics_count": len(metrics),
                "analysis_completed": bool(correlation_data)
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting correlation heatmap: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting correlation heatmap: {str(e)}")


@router.get("/prediction-chart", response_model=ChartDataResponse)
async def get_prediction_chart(
    hours: int = Query(24, description="Количество часов для анализа", ge=1, le=168),
    current_user = Depends(get_current_user)
):
    """
    Получение данных для диаграммы предсказаний
    
    Args:
        hours: Количество часов для анализа (1-168)
        current_user: Аутентифицированный пользователь
        
    Returns:
        ChartDataResponse: Данные для диаграммы предсказаний
    """
    try:
        # Используем метод из обновленного CognitiveLoadVisualizer
        advanced_data = visualizer.generate_advanced_visualization_data(hours=hours)
        prediction_data = advanced_data.get('prediction_data', [])
        
        if not prediction_data:
            # Если нет предсказаний, возвращаем пустую диаграмму
            return ChartDataResponse(
                chart_type="line",
                labels=["No Predictions Available"],
                datasets=[{
                    "label": "Predicted Load",
                    "data": [None],
                    "borderColor": "#999",
                    "backgroundColor": "rgba(0, 0, 0, 0.1)"
                }],
                options={}
            )
        
        # Подготавливаем данные для диаграммы
        time_labels = [f"+{item['time_offset_minutes']}min" for item in prediction_data]
        predicted_scores = [item['predicted_load_score'] for item in prediction_data]
        
        datasets = [{
            "label": "Predicted Cognitive Load",
            "data": predicted_scores,
            "borderColor": "#FF9F40",
            "backgroundColor": "rgba(255, 159, 64, 0.1)",
            "tension": 0.4,
            "fill": True
        }]
        
        return ChartDataResponse(
            chart_type="line",
            labels=time_labels,
            datasets=datasets,
            options={
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {
                        "min": 0,
                        "max": 1
                    }
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting prediction chart: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting prediction chart: {str(e)}")


# Функция для интеграции с основным API
def register_visualization_endpoints(main_app):
    """
    Регистрация эндпоинтов визуализации в основном приложении
    
    Args:
        main_app: Основное FastAPI приложение
    """
    main_app.include_router(router)
    logger.info("Visualization endpoints registered")
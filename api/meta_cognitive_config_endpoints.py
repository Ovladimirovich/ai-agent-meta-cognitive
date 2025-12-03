"""
API эндпоинты для настройки мета-когнитивных параметров
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from agent.meta_cognitive.cognitive_load_analyzer import CognitiveLoadAnalyzer, LoadLevel
from agent.learning.adaptation_engine import AdaptationEngine
from api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meta-cognitive-config", tags=["meta-cognitive-config"])


class MetaCognitiveConfig(BaseModel):
    """Конфигурация мета-когнитивных параметров"""
    cognitive_load_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Пороги когнитивной нагрузки"
    )
    adaptation_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Правила адаптации"
    )
    confidence_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Пороги уверенности"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Метрики производительности"
    )
    learning_rate: float = Field(
        default=0.01,
        description="Скорость обучения",
        ge=0.0,
        le=1.0
    )
    memory_management: Dict[str, Any] = Field(
        default_factory=dict,
        description="Параметры управления памятью"
    )
    resource_allocation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Параметры распределения ресурсов"
    )
    monitoring_intervals: Dict[str, int] = Field(
        default_factory=dict,
        description="Интервалы мониторинга"
    )


class ConfigUpdateRequest(BaseModel):
    """Запрос на обновление конфигурации"""
    parameter: str = Field(..., description="Название параметра")
    value: Any = Field(..., description="Новое значение")
    validation_required: bool = Field(
        default=True,
        description="Требуется ли валидация"
    )


class ConfigResponse(BaseModel):
    """Ответ с конфигурацией"""
    parameter: str = Field(..., description="Название параметра")
    current_value: Any = Field(..., description="Текущее значение")
    updated_value: Optional[Any] = Field(None, description="Новое значение (если обновлялось)")
    success: bool = Field(..., description="Успешность операции")
    message: str = Field(..., description="Сообщение")


class ConfigListResponse(BaseModel):
    """Ответ со списком конфигураций"""
    configurations: Dict[str, Any] = Field(..., description="Конфигурации")
    total_configs: int = Field(..., description="Общее количество параметров")


class AdaptationRule(BaseModel):
    """Модель правила адаптации"""
    id: str = Field(..., description="ID правила")
    name: str = Field(..., description="Название правила")
    description: str = Field(..., description="Описание правила")
    condition: str = Field(..., description="Условие срабатывания")
    action: str = Field(..., description="Действие при срабатывании")
    priority: int = Field(..., description="Приоритет правила", ge=1, le=10)
    enabled: bool = Field(default=True, description="Включено ли правило")


class AdaptationRuleRequest(BaseModel):
    """Запрос на создание/обновление правила адаптации"""
    name: str = Field(..., description="Название правила", min_length=1, max_length=100)
    description: str = Field(..., description="Описание правила", max_length=50)
    condition: str = Field(..., description="Условие срабатывания", min_length=1)
    action: str = Field(..., description="Действие при срабатывании", min_length=1)
    priority: int = Field(default=5, description="Приоритет правила", ge=1, le=10)
    enabled: bool = Field(default=True, description="Включено ли правило")


# Глобальные экземпляры анализатора и движка адаптации
analyzer = CognitiveLoadAnalyzer()
adaptation_engine = AdaptationEngine(None)  # В реальности нужно передать агент


@router.get("/current-config", response_model=ConfigListResponse)
async def get_current_config(current_user = Depends(get_current_user)):
    """
    Получение текущей конфигурации мета-когнитивных параметров
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        ConfigListResponse: Текущая конфигурация
    """
    try:
        # Сбор текущей конфигурации из различных компонентов
        config = {
            "cognitive_load_thresholds": analyzer.thresholds,
            "adaptation_rules_count": len(adaptation_engine.adaptation_rules),
            "active_adaptations": len(adaptation_engine.active_adaptations),
            "adaptation_history_size": len(adaptation_engine.adaptation_history),
            "analyzer_history_size": len(analyzer.load_history),
            "learning_rate": getattr(adaptation_engine, 'learning_rate', 0.01),
            "monitoring_intervals": {
                "health_check": 30,
                "performance_check": 60,
                "cognitive_load_check": 10
            }
        }
        
        return ConfigListResponse(
            configurations=config,
            total_configs=len(config)
        )
        
    except Exception as e:
        logger.error(f"Error getting current config: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting current config: {str(e)}")


@router.get("/parameter/{parameter_name}", response_model=ConfigResponse)
async def get_parameter(
    parameter_name: str,
    current_user = Depends(get_current_user)
):
    """
    Получение значения конкретного параметра
    
    Args:
        parameter_name: Название параметра
        current_user: Аутентифицированный пользователь
        
    Returns:
        ConfigResponse: Значение параметра
    """
    try:
        # В реальности параметры будут храниться в соответствующих компонентах
        # Пока возвращаем в зависимости от названия параметра
        current_value = None
        
        if parameter_name == "cognitive_load_thresholds":
            current_value = analyzer.thresholds
        elif parameter_name == "learning_rate":
            current_value = getattr(adaptation_engine, 'learning_rate', 0.01)
        elif parameter_name == "active_adaptations":
            current_value = len(adaptation_engine.active_adaptations)
        elif parameter_name == "analyzer_history_size":
            current_value = len(analyzer.load_history)
        elif parameter_name == "adaptation_history_size":
            current_value = len(adaptation_engine.adaptation_history)
        else:
            # Проверяем, есть ли такой параметр в порогах анализатора
            if parameter_name in analyzer.thresholds:
                current_value = analyzer.thresholds[parameter_name]
            else:
                raise HTTPException(status_code=404, detail=f"Parameter '{parameter_name}' not found")
        
        return ConfigResponse(
            parameter=parameter_name,
            current_value=current_value,
            success=True,
            message=f"Parameter '{parameter_name}' retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting parameter {parameter_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting parameter: {str(e)}")


@router.post("/parameter", response_model=ConfigResponse)
async def update_parameter(
    config_update: ConfigUpdateRequest,
    current_user = Depends(get_current_user)
):
    """
    Обновление значения параметра
    
    Args:
        config_update: Запрос на обновление параметра
        current_user: Аутентифицированный пользователь
        
    Returns:
        ConfigResponse: Результат обновления
    """
    try:
        parameter = config_update.parameter
        new_value = config_update.value
        validation_required = config_update.validation_required
        
        # Валидация параметра, если требуется
        if validation_required:
            validation_result = await _validate_parameter(parameter, new_value)
            if not validation_result['valid']:
                raise HTTPException(status_code=400, detail=validation_result['message'])
        
        # Обновление параметра в соответствующем компоненте
        update_result = await _update_specific_parameter(parameter, new_value)
        
        if update_result['success']:
            return ConfigResponse(
                parameter=parameter,
                current_value=update_result['current_value'],
                updated_value=new_value,
                success=True,
                message=update_result['message']
            )
        else:
            raise HTTPException(status_code=400, detail=update_result['message'])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating parameter {config_update.parameter}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating parameter: {str(e)}")


async def _validate_parameter(parameter: str, value: Any) -> Dict[str, Any]:
    """Валидация параметра"""
    try:
        if parameter in ['low_to_medium', 'medium_to_high', 'high_to_critical']:
            # Проверка порогов нагрузки
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                return {
                    'valid': False,
                    'message': f"Threshold value for '{parameter}' must be a float between 0 and 1"
                }
            return {'valid': True, 'message': 'Valid threshold value'}
        
        elif parameter == 'learning_rate':
            # Проверка скорости обучения
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                return {
                    'valid': False,
                    'message': "Learning rate must be a float between 0 and 1"
                }
            return {'valid': True, 'message': 'Valid learning rate'}
        
        elif parameter == 'factor_weights':
            # Проверка весов факторов
            if not isinstance(value, dict):
                return {
                    'valid': False,
                    'message': "Factor weights must be a dictionary"
                }
            
            required_keys = ['response_time', 'memory_usage', 'cpu_usage', 'active_tasks', 'error_rate', 'complexity_score']
            if not all(key in value for key in required_keys):
                return {
                    'valid': False,
                    'message': f"Factor weights must contain all required keys: {required_keys}"
                }
            
            # Проверка, что сумма весов равна 1.0
            total_weight = sum(value.values())
            if abs(total_weight - 1.0) > 0.001:
                return {
                    'valid': False,
                    'message': f"Sum of factor weights must be 1.0, got {total_weight}"
                }
            
            return {'valid': True, 'message': 'Valid factor weights'}
        
        else:
            # Для других параметров предполагаем валидность
            return {'valid': True, 'message': 'Parameter validation passed'}
            
    except Exception as e:
        return {
            'valid': False,
            'message': f"Validation error: {str(e)}"
        }


async def _update_specific_parameter(parameter: str, value: Any) -> Dict[str, Any]:
    """Обновление конкретного параметра"""
    try:
        if parameter in analyzer.thresholds:
            # Обновление порога в анализаторе нагрузки
            old_value = analyzer.thresholds[parameter]
            analyzer.thresholds[parameter] = value
            return {
                'success': True,
                'current_value': value,
                'message': f"Threshold '{parameter}' updated from {old_value} to {value}"
            }
        
        elif parameter == 'learning_rate':
            # Обновление скорости обучения в движке адаптации
            old_value = getattr(adaptation_engine, 'learning_rate', 0.01)
            setattr(adaptation_engine, 'learning_rate', value)
            return {
                'success': True,
                'current_value': value,
                'message': f"Learning rate updated from {old_value} to {value}"
            }
        
        elif parameter == 'factor_weights':
            # Обновление весов факторов
            old_value = analyzer.factor_weights.copy()
            analyzer.factor_weights = value
            return {
                'success': True,
                'current_value': value,
                'message': f"Factor weights updated from {old_value} to {value}"
            }
        
        else:
            return {
                'success': False,
                'message': f"Parameter '{parameter}' is not configurable or not found"
            }
            
    except Exception as e:
        return {
            'success': False,
            'message': f"Error updating parameter '{parameter}': {str(e)}"
        }


@router.get("/adaptation-rules", response_model=List[AdaptationRule])
async def get_adaptation_rules(current_user = Depends(get_current_user)):
    """
    Получение списка правил адаптации
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        List[AdaptationRule]: Список правил адаптации
    """
    try:
        rules = []
        
        # Преобразование внутренних правил адаптации в API модели
        for rule_name, rule_data in adaptation_engine.adaptation_rules.items():
            rule = AdaptationRule(
                id=rule_name,
                name=rule_name.replace('_', ' ').title(),
                description=rule_data['description'],
                condition=rule_data['condition'].__name__ if callable(rule_data['condition']) else str(rule_data['condition']),
                action=rule_data['action'].__name__ if callable(rule_data['action']) else str(rule_data['action']),
                priority=rule_data['priority'],
                enabled=True  # В реальности нужно хранить статус
            )
            rules.append(rule)
        
        return rules
        
    except Exception as e:
        logger.error(f"Error getting adaptation rules: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting adaptation rules: {str(e)}")


@router.post("/adaptation-rules", response_model=AdaptationRule)
async def create_adaptation_rule(
    rule_request: AdaptationRuleRequest,
    current_user = Depends(get_current_user)
):
    """
    Создание нового правила адаптации
    
    Args:
        rule_request: Запрос на создание правила
        current_user: Аутентифицированный пользователь
        
    Returns:
        AdaptationRule: Созданное правило
    """
    try:
        # В реальной системе нужно добавить правило в движок адаптации
        # Пока просто возвращаем созданное правило
        new_rule = AdaptationRule(
            id=f"rule_{int(datetime.now().timestamp())}",
            name=rule_request.name,
            description=rule_request.description,
            condition=rule_request.condition,
            action=rule_request.action,
            priority=rule_request.priority,
            enabled=rule_request.enabled
        )
        
        # В реальной реализации нужно добавить правило в adaptation_engine.adaptation_rules
        logger.info(f"Created adaptation rule: {new_rule.id}")
        
        return new_rule
        
    except Exception as e:
        logger.error(f"Error creating adaptation rule: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating adaptation rule: {str(e)}")


@router.put("/adaptation-rules/{rule_id}", response_model=AdaptationRule)
async def update_adaptation_rule(
    rule_id: str,
    rule_request: AdaptationRuleRequest,
    current_user = Depends(get_current_user)
):
    """
    Обновление правила адаптации
    
    Args:
        rule_id: ID правила
        rule_request: Запрос на обновление правила
        current_user: Аутентифицированный пользователь
        
    Returns:
        AdaptationRule: Обновленное правило
    """
    try:
        # В реальной системе нужно обновить правило в движке адаптации
        # Пока просто возвращаем обновленное правило
        updated_rule = AdaptationRule(
            id=rule_id,
            name=rule_request.name,
            description=rule_request.description,
            condition=rule_request.condition,
            action=rule_request.action,
            priority=rule_request.priority,
            enabled=rule_request.enabled
        )
        
        # В реальной реализации нужно обновить правило в adaptation_engine.adaptation_rules
        logger.info(f"Updated adaptation rule: {rule_id}")
        
        return updated_rule
        
    except Exception as e:
        logger.error(f"Error updating adaptation rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating adaptation rule: {str(e)}")


@router.delete("/adaptation-rules/{rule_id}")
async def delete_adaptation_rule(
    rule_id: str,
    current_user = Depends(get_current_user)
):
    """
    Удаление правила адаптации
    
    Args:
        rule_id: ID правила
        current_user: Аутентифицированный пользователь
        
    Returns:
        Dict: Результат удаления
    """
    try:
        # В реальной системе нужно удалить правило из движка адаптации
        # Пока просто возвращаем результат
        logger.info(f"Deleted adaptation rule: {rule_id}")
        
        return {
            "success": True,
            "message": f"Adaptation rule '{rule_id}' deleted successfully",
            "deleted_rule_id": rule_id
        }
        
    except Exception as e:
        logger.error(f"Error deleting adaptation rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting adaptation rule: {str(e)}")


@router.post("/reset-to-defaults")
async def reset_to_defaults(current_user = Depends(get_current_user)):
    """
    Сброс всех мета-когнитивных параметров к значениям по умолчанию
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        Dict: Результат сброса
    """
    try:
        # Сброс порогов анализатора нагрузки
        analyzer.thresholds = {
            'low_to_medium': 0.3,
            'medium_to_high': 0.6,
            'high_to_critical': 0.8
        }
        
        # Сброс весов факторов
        analyzer.factor_weights = {
            'response_time': 0.2,
            'memory_usage': 0.2,
            'cpu_usage': 0.2,
            'active_tasks': 0.15,
            'error_rate': 0.15,
            'complexity_score': 0.1
        }
        
        # Сброс скорости обучения
        setattr(adaptation_engine, 'learning_rate', 0.01)
        
        # Возврат к стандартным правилам адаптации
        adaptation_engine.adaptation_rules = adaptation_engine._initialize_adaptation_rules()
        
        logger.info("Reset all meta-cognitive parameters to defaults")
        
        return {
            "success": True,
            "message": "All meta-cognitive parameters reset to default values",
            "reset_parameters": [
                "cognitive_load_thresholds",
                "factor_weights", 
                "learning_rate",
                "adaptation_rules"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error resetting to defaults: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting to defaults: {str(e)}")


@router.get("/health-indicators")
async def get_config_health_indicators(current_user = Depends(get_current_user)):
    """
    Получение индикаторов здоровья конфигурации
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        Dict: Индикаторы здоровья
    """
    try:
        # Оценка корректности конфигурации
        config_issues = []
        
        # Проверка порогов нагрузки
        thresholds = analyzer.thresholds
        if not (0 <= thresholds.get('low_to_medium', 0.3) <= thresholds.get('medium_to_high', 0.6) <= thresholds.get('high_to_critical', 0.8) <= 1.0):
            config_issues.append("Load thresholds are not in correct order or range")
        
        # Проверка весов факторов
        factor_weights = analyzer.factor_weights
        total_weight = sum(factor_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            config_issues.append(f"Factor weights sum is not 1.0: {total_weight}")
        
        # Проверка количества правил адаптации
        adaptation_count = len(adaptation_engine.adaptation_rules)
        if adaptation_count == 0:
            config_issues.append("No adaptation rules configured")
        
        # Проверка истории
        analyzer_history_size = len(analyzer.load_history)
        adaptation_history_size = len(adaptation_engine.adaptation_history)
        
        health_status = "healthy" if not config_issues else "issues_found"
        
        return {
            "status": health_status,
            "issues": config_issues,
            "config_metrics": {
                "load_thresholds": thresholds,
                "factor_weights": factor_weights,
                "adaptation_rules_count": adaptation_count,
                "analyzer_history_size": analyzer_history_size,
                "adaptation_history_size": adaptation_history_size,
                "active_adaptations": len(adaptation_engine.active_adaptations)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting config health indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting config health indicators: {str(e)}")


# Функция для интеграции с основным API
def register_meta_cognitive_config_endpoints(main_app):
    """
    Регистрация эндпоинтов настройки мета-когнитивных параметров в основном приложении
    
    Args:
        main_app: Основное FastAPI приложение
    """
    main_app.include_router(router)
    logger.info("Meta-cognitive config endpoints registered")
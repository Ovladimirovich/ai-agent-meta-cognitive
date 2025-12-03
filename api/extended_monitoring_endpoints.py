"""
Расширенные эндпоинты мониторинга и отладки
"""

import asyncio
import logging
import psutil
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from agent.core.agent_core import AgentCore
from agent.meta_cognitive.cognitive_load_analyzer import CognitiveLoadAnalyzer
from agent.learning.adaptation_engine import AdaptationEngine
from agent.self_awareness.self_monitoring import SelfMonitoringSystem
from api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extended-monitoring", tags=["extended-monitoring"])


class DebugInfoResponse(BaseModel):
    """Ответ с отладочной информацией"""
    timestamp: str = Field(..., description="Временная метка")
    process_info: Dict[str, Any] = Field(..., description="Информация о процессе")
    memory_info: Dict[str, Any] = Field(..., description="Информация о памяти")
    cpu_info: Dict[str, Any] = Field(..., description="Информация о CPU")
    disk_info: Dict[str, Any] = Field(..., description="Информация о диске")
    network_info: Dict[str, Any] = Field(..., description="Информация о сети")
    agent_state: Dict[str, Any] = Field(..., description="Состояние агента")


class HealthCheckResponse(BaseModel):
    """Ответ проверки здоровья"""
    component: str = Field(..., description="Компонент")
    status: str = Field(..., description="Статус")
    details: Dict[str, Any] = Field(..., description="Детали")
    timestamp: str = Field(..., description="Временная метка")


class PerformanceMetricsResponse(BaseModel):
    """Ответ с метриками производительности"""
    timestamp: str = Field(..., description="Временная метка")
    response_times: List[float] = Field(..., description="Времена отклика")
    throughput: float = Field(..., description="Пропускная способность")
    error_rate: float = Field(..., description="Уровень ошибок")
    resource_usage: Dict[str, float] = Field(..., description="Использование ресурсов")


class TraceInfoResponse(BaseModel):
    """Ответ с информацией трассировки"""
    trace_id: str = Field(..., description="ID трассировки")
    steps: List[Dict[str, Any]] = Field(..., description="Шаги трассировки")
    duration: float = Field(..., description="Длительность")
    status: str = Field(..., description="Статус")


class LogEntry(BaseModel):
    """Запись лога"""
    timestamp: str = Field(..., description="Временная метка")
    level: str = Field(..., description="Уровень лога")
    message: str = Field(..., description="Сообщение")
    module: str = Field(..., description="Модуль")
    function: str = Field(..., description="Функция")


class LogQuery(BaseModel):
    """Запрос на получение логов"""
    level: Optional[str] = Field(None, description="Уровень логов (DEBUG, INFO, WARNING, ERROR)")
    module: Optional[str] = Field(None, description="Модуль для фильтрации")
    limit: int = Field(100, description="Количество записей", ge=1, le=1000)
    search: Optional[str] = Field(None, description="Поисковый запрос")


# Глобальные экземпляры компонентов (в реальности должны быть внедрены через DI)
agent_core: Optional[AgentCore] = None
cognitive_analyzer = CognitiveLoadAnalyzer()
adaptation_engine: Optional[AdaptationEngine] = None
self_monitoring: Optional[SelfMonitoringSystem] = None


@router.get("/debug-info", response_model=DebugInfoResponse)
async def get_debug_info(current_user = Depends(get_current_user)):
    """
    Получение комплексной отладочной информации
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        DebugInfoResponse: Отладочная информация
    """
    try:
        # Сбор информации о процессе
        process = psutil.Process(os.getpid())
        
        process_info = {
            'pid': process.pid,
            'name': process.name(),
            'status': process.status(),
            'create_time': datetime.fromtimestamp(process.create_time()).isoformat(),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 'N/A'
        }
        
        # Сбор информации о памяти
        memory_info = {
            'rss': process.memory_info().rss,
            'vms': process.memory_info().vms,
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available,
            'total': psutil.virtual_memory().total,
            'used_percent': psutil.virtual_memory().percent
        }
        
        # Сбор информации о CPU
        cpu_info = {
            'percent': process.cpu_percent(),
            'num_cores': psutil.cpu_count(),
            'cpu_times': psutil.cpu_times()._asdict(),
            'load_average': psutil.getloadavg()
        }
        
        # Сбор информации о диске
        disk_usage = psutil.disk_usage('/')
        disk_info = {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': disk_usage.percent
        }
        
        # Сбор информации о сети
        net_io = psutil.net_io_counters()
        network_info = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
        # Сбор информации о состоянии агента (если доступно)
        agent_state = {}
        if agent_core:
            agent_state = {
                'current_state': getattr(agent_core.state_manager, 'current_state', 'unknown').value if hasattr(agent_core, 'state_manager') else 'unknown',
                'active_tasks': len([t for t in asyncio.all_tasks() if not t.done()]),
                'memory_entries': getattr(agent_core.memory_manager, 'get_memory_stats', lambda: {'total_entries': 0})()['total_entries'] if hasattr(agent_core, 'memory_manager') else 0,
                'config': getattr(agent_core, 'config', {}).__dict__ if hasattr(agent_core, 'config') else {}
            }
        
        return DebugInfoResponse(
            timestamp=datetime.now().isoformat(),
            process_info=process_info,
            memory_info=memory_info,
            cpu_info=cpu_info,
            disk_info=disk_info,
            network_info=network_info,
            agent_state=agent_state
        )
        
    except Exception as e:
        logger.error(f"Error getting debug info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting debug info: {str(e)}")


@router.get("/health-checks", response_model=List[HealthCheckResponse])
async def get_health_checks(
    component: Optional[str] = Query(None, description="Фильтр по компоненту"),
    current_user = Depends(get_current_user)
):
    """
    Получение результатов проверок здоровья
    
    Args:
        component: Фильтр по компоненту (опционально)
        current_user: Аутентифицированный пользователь
        
    Returns:
        List[HealthCheckResponse]: Результаты проверок здоровья
    """
    try:
        health_results = []
        
        # Проверка основного процесса
        process_health = {
            'status': 'healthy',
            'details': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
        health_results.append(HealthCheckResponse(
            component='system',
            status=process_health['status'],
            details=process_health['details'],
            timestamp=datetime.now().isoformat()
        ))
        
        # Проверка агента (если доступен)
        if agent_core:
            agent_health = {
                'status': 'healthy',
                'details': {
                    'state': getattr(agent_core.state_manager, 'current_state', 'unknown').value if hasattr(agent_core, 'state_manager') else 'unknown',
                    'memory_manager_status': 'active' if hasattr(agent_core, 'memory_manager') else 'not_available',
                    'tool_orchestrator_status': 'active' if hasattr(agent_core, 'tool_orchestrator') else 'not_available'
                }
            }
            health_results.append(HealthCheckResponse(
                component='agent_core',
                status=agent_health['status'],
                details=agent_health['details'],
                timestamp=datetime.now().isoformat()
            ))
        
        # Проверка когнитивного анализатора
        cognitive_health = {
            'status': 'healthy',
            'details': {
                'load_history_size': len(cognitive_analyzer.load_history),
                'thresholds': cognitive_analyzer.thresholds
            }
        }
        health_results.append(HealthCheckResponse(
            component='cognitive_analyzer',
            status=cognitive_health['status'],
            details=cognitive_health['details'],
            timestamp=datetime.now().isoformat()
        ))
        
        # Проверка движка адаптации (если доступен)
        if adaptation_engine:
            adaptation_health = {
                'status': 'healthy',
                'details': {
                    'active_adaptations': len(adaptation_engine.active_adaptations),
                    'adaptation_rules_count': len(adaptation_engine.adaptation_rules),
                    'adaptation_history_size': len(adaptation_engine.adaptation_history)
                }
            }
            health_results.append(HealthCheckResponse(
                component='adaptation_engine',
                status=adaptation_health['status'],
                details=adaptation_health['details'],
                timestamp=datetime.now().isoformat()
            ))
        
        # Проверка системы самодиагностики (если доступна)
        if self_monitoring:
            try:
                monitoring_health = await self_monitoring.get_agent_health()
                self_monitoring_health = {
                    'status': monitoring_health.status,
                    'details': {
                        'health_score': monitoring_health.health_score,
                        'issues_count': monitoring_health.issues_count,
                        'last_diagnosis': monitoring_health.last_diagnosis.isoformat() if monitoring_health.last_diagnosis else None
                    }
                }
                health_results.append(HealthCheckResponse(
                    component='self_monitoring',
                    status=self_monitoring_health['status'],
                    details=self_monitoring_health['details'],
                    timestamp=datetime.now().isoformat()
                ))
            except Exception as e:
                logger.warning(f"Self monitoring health check failed: {e}")
                health_results.append(HealthCheckResponse(
                    component='self_monitoring',
                    status='unavailable',
                    details={'error': str(e)},
                    timestamp=datetime.now().isoformat()
                ))
        
        # Фильтрация по компоненту, если указан
        if component:
            health_results = [h for h in health_results if h.component == component]
        
        return health_results
        
    except Exception as e:
        logger.error(f"Error getting health checks: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting health checks: {str(e)}")


@router.get("/performance-metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    hours: int = Query(1, description="Количество часов для анализа", ge=1, le=24),
    current_user = Depends(get_current_user)
):
    """
    Получение метрик производительности
    
    Args:
        hours: Количество часов для анализа (1-24)
        current_user: Аутентифицированный пользователь
        
    Returns:
        PerformanceMetricsResponse: Метрики производительности
    """
    try:
        # В реальной системе метрики будут собираться из соответствующих источников
        # Пока возвращаем симулированные данные
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Симуляция данных производительности
        import random
        response_times = [random.uniform(0.5, 3.0) for _ in range(50)]
        throughput = random.uniform(10, 100)  # запросов в секунду
        error_rate = random.uniform(0.01, 0.1)  # 1-10% ошибок
        
        resource_usage = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'active_threads': len([t for t in asyncio.all_tasks() if not t.done()])
        }
        
        return PerformanceMetricsResponse(
            timestamp=datetime.now().isoformat(),
            response_times=response_times,
            throughput=throughput,
            error_rate=error_rate,
            resource_usage=resource_usage
        )
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")


@router.get("/trace-info/{trace_id}", response_model=TraceInfoResponse)
async def get_trace_info(
    trace_id: str,
    current_user = Depends(get_current_user)
):
    """
    Получение информации трассировки по ID
    
    Args:
        trace_id: ID трассировки
        current_user: Аутентифицированный пользователь
        
    Returns:
        TraceInfoResponse: Информация трассировки
    """
    try:
        # В реальной системе трассировка будет извлекаться из соответствующего хранилища
        # Пока возвращаем симулированные данные
        import random
        
        steps = [
            {
                'step_id': f'step_{i}',
                'name': f'Processing Step {i}',
                'duration': random.uniform(0.1, 1.0),
                'status': 'completed',
                'details': {'input_size': random.randint(100, 1000)}
            }
            for i in range(1, random.randint(3, 8))
        ]
        
        total_duration = sum(step['duration'] for step in steps)
        
        return TraceInfoResponse(
            trace_id=trace_id,
            steps=steps,
            duration=total_duration,
            status='completed'
        )
        
    except Exception as e:
        logger.error(f"Error getting trace info for {trace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting trace info: {str(e)}")


@router.get("/logs", response_model=List[LogEntry])
async def get_logs(
    level: Optional[str] = Query(None, description="Уровень логов (DEBUG, INFO, WARNING, ERROR)"),
    module: Optional[str] = Query(None, description="Модуль для фильтрации"),
    limit: int = Query(100, description="Количество записей", ge=1, le=1000),
    search: Optional[str] = Query(None, description="Поисковый запрос"),
    current_user = Depends(get_current_user)
):
    """
    Получение логов системы
    
    Args:
        level: Уровень логов для фильтрации
        module: Модуль для фильтрации
        limit: Количество записей
        search: Поисковый запрос
        current_user: Аутентифицированный пользователь
        
    Returns:
        List[LogEntry]: Записи логов
    """
    try:
        # В реальной системе логи будут извлекаться из централизованного хранилища
        # Пока возвращаем симулированные данные
        import random
        from datetime import timedelta
        
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        modules = ['agent_core', 'cognitive_analyzer', 'adaptation_engine', 'api', 'memory_manager']
        
        logs = []
        for i in range(limit):
            timestamp = (datetime.now() - timedelta(seconds=random.randint(0, 3600))).isoformat()
            log_level = random.choice(log_levels)
            module_name = random.choice(modules)
            message = f"Sample log message {i+1} for {module_name} at {log_level} level"
            
            # Применение фильтров
            if level and level.upper() != log_level:
                continue
            if module and module not in module_name:
                continue
            if search and search.lower() not in message.lower():
                continue
                
            log_entry = LogEntry(
                timestamp=timestamp,
                level=log_level,
                message=message,
                module=module_name,
                function=f'function_{random.randint(1, 10)}'
            )
            logs.append(log_entry)
            
            if len(logs) >= limit:
                break
        
        return logs
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting logs: {str(e)}")


@router.get("/resource-usage")
async def get_resource_usage(current_user = Depends(get_current_user)):
    """
    Получение информации об использовании ресурсов
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        Dict: Информация об использовании ресурсов
    """
    try:
        # Сбор информации об использовании ресурсов
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        resource_info = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count(),
                'times': psutil.cpu_times()._asdict()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'free': memory.free
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'process': {
                'memory_percent': psutil.Process().memory_percent(),
                'cpu_percent': psutil.Process().cpu_percent(),
                'open_files': len(psutil.Process().open_files()) if psutil.Process().open_files() else 0,
                'connections': len(psutil.Process().connections()) if psutil.Process().connections() else 0
            }
        }
        
        return resource_info
        
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting resource usage: {str(e)}")


@router.get("/system-status")
async def get_system_status(current_user = Depends(get_current_user)):
    """
    Получение статуса системы
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        Dict: Статус системы
    """
    try:
        # Сбор статуса различных компонентов системы
        system_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'operational',
            'uptime': datetime.now().timestamp(),
            'components': {
                'api_server': 'operational',
                'agent_core': 'operational' if agent_core else 'not_initialized',
                'cognitive_analyzer': 'operational',
                'adaptation_engine': 'operational' if adaptation_engine else 'not_initialized',
                'self_monitoring': 'operational' if self_monitoring else 'not_initialized',
                'database': 'operational',  # В реальности нужно проверить соединение
                'cache': 'operational'      # В реальности нужно проверить соединение
            },
            'health_indicators': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'active_connections': 0  # В реальности нужно отслеживать
            },
            'performance_indicators': {
                'response_time_avg': 0.5,  # В реальности нужно собирать
                'requests_per_minute': 0,  # В реальности нужно собирать
                'error_rate': 0.0         # В реальности нужно собирать
            }
        }
        
        # Проверка общего статуса на основе индикаторов
        if (system_status['health_indicators']['cpu_usage'] > 90 or
            system_status['health_indicators']['memory_usage'] > 90 or
            system_status['health_indicators']['disk_usage'] > 95):
            system_status['overall_status'] = 'degraded'
        
        return system_status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")


@router.post("/trigger-garbage-collection")
async def trigger_garbage_collection(current_user = Depends(get_current_user)):
    """
    Ручной запуск сборщика мусора
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        Dict: Результат операции
    """
    try:
        import gc
        
        # Сбор статистики до сборки
        before_stats = gc.get_stats()
        collected_objects = gc.collect()
        after_stats = gc.get_stats()
        
        return {
            'success': True,
            'message': f'Garbage collection completed. Collected {collected_objects} objects',
            'before_stats': before_stats,
            'after_stats': after_stats,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering garbage collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error triggering garbage collection: {str(e)}")


@router.get("/active-tasks")
async def get_active_tasks(current_user = Depends(get_current_user)):
    """
    Получение информации об активных задачах
    
    Args:
        current_user: Аутентифицированный пользователь
        
    Returns:
        Dict: Информация об активных задачах
    """
    try:
        tasks = []
        for task in asyncio.all_tasks():
            if not task.done():
                task_info = {
                    'id': id(task),
                    'name': getattr(task, 'get_name', lambda: f'Task-{id(task)}')(),
                    'state': 'pending' if not task.done() else 'completed',
                    'created_at': getattr(task, '_source_traceback', [{}])[0].get('filename', 'unknown') if hasattr(task, '_source_traceback') else 'unknown'
                }
                tasks.append(task_info)
        
        return {
            'total_tasks': len(tasks),
            'active_tasks': tasks,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting active tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting active tasks: {str(e)}")


# Функция для интеграции с основным API
def register_extended_monitoring_endpoints(main_app):
    """
    Регистрация расширенных эндпоинтов мониторинга в основном приложении
    
    Args:
        main_app: Основное FastAPI приложение
    """
    main_app.include_router(router)
    logger.info("Extended monitoring endpoints registered")
# 🌐 Справочник API мета-когнитивного AI агента

## 📋 Обзор API

API мета-когнитивного AI агента предоставляет интерфейс для взаимодействия с системой, позволяя обрабатывать запросы, управлять состоянием агента и получать информацию о его когнитивных процессах.

## 🔐 Аутентификация

Все эндпоинты требуют аутентификации с использованием Bearer токенов:

```bash
curl -X GET "http://localhost:8000/agent/status" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

## 📡 Базовые эндпоинты

### GET /health
Проверка работоспособности сервиса.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Пример ответа:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-23T21:00:00.000Z",
  "version": "1.0.0",
  "service": "AI Agent Meta-Cognitive API"
}
```

### GET /health/detailed
Расширенная проверка здоровья всех компонентов системы.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/health/detailed"
```

**Пример ответа:**
```json
{
  "overall_status": "healthy",
  "total_checks": 8,
  "healthy": 7,
  "degraded": 1,
  "unhealthy": 0,
  "checks": {
    "system": {
      "status": "degraded",
      "response_time": 0.023,
      "message": "System resources are high",
      "details": {
        "cpu_percent": 85.2,
        "memory_percent": 78.5
      }
    },
    "cqrs_system": {
      "status": "healthy",
      "response_time": 0.001,
      "message": "CQRS buses operational"
    }
  },
  "timestamp": "2025-11-23T21:00:00.000Z"
}
```

### POST /agent/process
Обработка запроса агентом с мета-когнитивными возможностями.

**Пример запроса:**
```bash
curl -X POST "http://localhost:8000/agent/process" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{
    "query": "Проанализируй производительность системы за последний месяц",
    "context": {
      "domain": "анализ данных",
      "complexity": "medium",
      "required_tools": ["analytics", "web_research"],
      "time_constraints": 30
    },
    "preferences": {
      "response_format": "detailed",
      "enable_reasoning_trace": true,
      "include_confidence_scores": true
    }
  }'
```

**Пример ответа:**
```json
{
  "success": true,
  "response": {
    "result": "Детальный анализ производительности...",
    "confidence": 0.85,
    "reasoning_trace": [
      {
        "step": 1,
        "description": "Анализ запроса пользователя",
        "confidence": 0.95,
        "tools_used": [],
        "timestamp": "2025-11-23T21:00:00.000Z"
      },
      {
        "step": 2,
        "description": "Выбор инструментов для анализа",
        "confidence": 0.90,
        "tools_used": ["analytics_tool"],
        "timestamp": "2025-11-23T21:00:00.100Z"
      }
    ],
    "execution_time": 2.345,
    "metadata": {
      "used_tools": ["analytics_tool", "cache_tool"],
      "tokens_used": 1250,
      "model_used": "gpt-4-turbo"
    }
  }
}
```

### GET /agent/status
Получение текущего статуса агента.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/agent/status" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "status": "operational",
  "state": "idle",
  "confidence_level": 0.82,
  "active_tools": [],
  "memory_usage": {
    "entries_count": 45,
    "estimated_size_mb": 2.3
  },
  "performance_metrics": {
    "avg_response_time": 1.234,
    "success_rate": 0.96,
    "active_sessions": 3
  },
  "timestamp": "2025-11-23T21:00:00.000Z"
}
```

### GET /agent/health
Проверка когнитивного здоровья агента.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/agent/health" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "health_status": "stable",
  "health_score": 0.85,
  "issues_count": 1,
  "last_diagnosis": "2025-11-23T20:59:30.000Z",
  "diagnostics": {
    "cognitive_load": 0.45,
    "attention_span": 0.89,
    "reasoning_accuracy": 0.87,
    "memory_retention": 0.92,
    "adaptability": 0.78
  },
  "recommendations": [
    "Увеличить частоту самоанализа для повышения адаптивности"
  ]
}
```

### POST /agent/learn
Запуск процесса обучения на основе опыта.

**Пример запроса:**
```bash
curl -X POST "http://localhost:8000/agent/learn" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{
    "experience": {
      "task": "анализ временных рядов",
      "approach": "использование статистических методов",
      "result": "успешно",
      "success_factors": ["хорошее качество данных", "подходящий алгоритм"],
      "improvement_opportunities": ["увеличить размер выборки", "добавить валидацию"]
    },
    "feedback": {
      "rating": 4.5,
      "comments": "Хороший результат, но можно улучшить скорость обработки"
    }
  }'
```

**Пример ответа:**
```json
{
  "status": "learning_initiated",
  "learning_id": "learn_abc123def456",
  "expected_duration": "0:00:05.000000",
  "applied_improvements": [
    "оптимизация алгоритма обработки временных рядов",
    "улучшение валидации входных данных"
  ],
  "timestamp": "2025-11-23T21:00:00.000Z"
}
```

## 🧠 Мета-когнитивные эндпоинты

### GET /agent/insights
Получение мета-когнитивных инсайтов и анализа деятельности агента.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/agent/insights?period=week" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "insights": {
    "performance_trends": {
      "confidence_stability": 0.89,
      "response_quality": 0.92,
      "tool_utilization": 0.76
    },
    "learning_progress": {
      "new_patterns_identified": 12,
      "improved_strategies": 5,
      "efficiency_gains": 0.15
    },
    "cognitive_patterns": [
      {
        "pattern": "эффективное использование RAG для аналитических задач",
        "frequency": 0.67,
        "success_rate": 0.94
      }
    ]
  },
  "period_covered": {
    "start": "2025-11-16T21:00:00.000Z",
    "end": "2025-11-23T21:00:00.000Z"
  },
  "timestamp": "2025-11-23T21:00:00.000Z"
}
```

### POST /agent/reflect
Запуск процесса рефлексии и самоанализа.

**Пример запроса:**
```bash
curl -X POST "http://localhost:8000/agent/reflect" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{
    "focus_area": "эффективность принятия решений",
    "time_period": "last_24_hours",
    "depth": "deep"  // shallow, medium, deep
  }'
```

**Пример ответа:**
```json
{
  "reflection_result": {
    "analysis_summary": "Анализ эффективности принятия решений за последние 24 часа",
    "identified_strengths": [
      "хорошее распознавание сложных запросов",
      "эффективное использование инструментов"
    ],
    "areas_for_improvement": [
      "увеличить скорость обработки простых запросов",
      "улучшить фильтрацию шумных данных"
    ],
    "suggested_changes": [
      "оптимизировать маршрут для простых задач",
      "улучшить предварительную обработку данных"
    ],
    "confidence_in_analysis": 0.87
  },
  "processing_time": 1.234,
  "timestamp": "2025-11-23T21:00:00.000Z"
}
```

### GET /agent/reasoning-trace/{request_id}
Получение трассировки рассуждений для конкретного запроса.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/agent/reasoning-trace/req_abc123def456" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "request_id": "req_abc123def456",
  "reasoning_trace": [
    {
      "step_id": "step_001",
      "step_type": "query_analysis",
      "description": "Анализ входного запроса и определение сложности",
      "input": "Как оптимизировать производительность базы данных?",
      "output": {
        "complexity": "high",
        "domain": "database_administration",
        "required_tools": ["analytics", "web_research"]
      },
      "confidence": 0.92,
      "timestamp": "2025-11-23T21:00:00.100Z",
      "execution_time_ms": 120
    },
    {
      "step_id": "step_002",
      "step_type": "tool_selection",
      "description": "Выбор оптимальных инструментов для задачи",
      "input": {
        "task": "database optimization",
        "complexity": "high",
        "available_tools": ["analytics", "web_research", "cache"]
      },
      "output": {
        "selected_tools": ["analytics", "web_research"],
        "selection_reasoning": "Аналитика для анализа производительности, веб-поиск для получения лучших практик"
      },
      "confidence": 0.88,
      "timestamp": "2025-11-23T21:00:00.220Z",
      "execution_time_ms": 80
    }
  ],
  "overall_confidence": 0.85,
  "total_processing_time": 2.345
}
```

## 📊 Метрики и мониторинг

### GET /metrics
Получение метрик производительности (в формате Prometheus).

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/metrics"
```

**Пример ответа:**
```
# Метрики времени отклика
agent_request_duration_seconds_bucket{le="0.1",method="POST",endpoint="/agent/process"} 5
agent_request_duration_seconds_bucket{le="0.5",method="POST",endpoint="/agent/process"} 12
agent_request_duration_seconds_bucket{le="1.0",method="POST",endpoint="/agent/process"} 18
agent_request_duration_seconds_bucket{le="+Inf",method="POST",endpoint="/agent/process"} 20
agent_request_duration_seconds_count{method="POST",endpoint="/agent/process"} 20
agent_request_duration_seconds_sum{method="POST",endpoint="/agent/process"} 18.45

# Метрики уверенности
agent_confidence_score{method="POST",endpoint="/agent/process"} 0.82

# Счетчики запросов
agent_requests_total{method="POST",endpoint="/agent/process"} 20
agent_errors_total{method="POST",endpoint="/agent/process",error_type="validation_error"} 2
```

### GET /agent/metrics/detailed
Детализированные метрики агента.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/agent/metrics/detailed" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "performance": {
    "avg_response_time": 1.234,
    "p95_response_time": 2.567,
    "p99_response_time": 4.123,
    "requests_per_minute": 12.5
  },
  "cognitive_metrics": {
    "avg_confidence": 0.82,
    "self_reflection_frequency": "every_10_requests",
    "learning_events_count": 24
  },
  "resource_usage": {
    "memory_mb": 245.6,
    "active_connections": 3,
    "cache_hit_rate": 0.78
  },
  "tool_effectiveness": {
    "rag_success_rate": 0.94,
    "analytics_utilization": 0.67,
    "web_research_success_rate": 0.89
  },
  "timestamp": "2025-11-23T21:00:00.000Z"
}
```

## 🔧 Управление памятью

### GET /agent/memory/stats
Получение статистики использования памяти.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/agent/memory/stats" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "memory_stats": {
    "total_entries": 1250,
    "episodic_memory": 800,
    "semantic_memory": 300,
    "procedural_memory": 150,
    "estimated_size_mb": 12.4,
    "oldest_entry": "2025-11-20T10:30:00.000Z",
    "newest_entry": "2025-11-23T20:59:45.000Z"
  },
  "retention_policy": {
    "episode_retention_days": 30,
    "semantic_retention_days": 180,
    "procedural_retention_days": -1
  },
  "optimization_suggestions": [
    "очистить эпизодическую память старше 30 дней",
    "архивировать старые семантические записи"
  ]
}
```

### DELETE /agent/memory/clear
Очистка памяти агента (с подтверждением).

**Пример запроса:**
```bash
curl -X DELETE "http://localhost:8000/agent/memory/clear" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "confirmation": "CONFIRM_CLEAR_MEMORY",
    "memory_types": ["episodic", "semantic"],  // необязательно, по умолчанию все
    "retention_policy": "keep_last_week"  // или "clear_all", "keep_essential"
  }'
```

**Пример ответа:**
```json
{
  "status": "memory_cleared",
  "entries_removed": 1100,
  "remaining_entries": 150,
  "memory_freed_mb": 10.2,
  "timestamp": "2025-11-23T21:00:00.000Z"
}
```

## 🛡️ Безопасность и ограничения

### Ограничения на запросы
- Максимальный размер тела запроса: 10MB
- Максимальное время выполнения: 60 секунд
- Ограничение по частоте: 100 запросов/минуту на токен

### Фильтрация контента
Агент автоматически фильтрует потенциально опасный или неподходящий контент в соответствии с этическими директивами.

## 🚀 Продвинутые возможности

### Пакетная обработка
Для обработки нескольких запросов одновременно:

```bash
curl -X POST "http://localhost:8000/agent/batch-process" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{
    "requests": [
      {
        "query": "Запрос 1",
        "context": {"domain": "анализ"}
      },
      {
        "query": "Запрос 2", 
        "context": {"domain": "генерация"}
      }
    ],
    "options": {
      "parallel_processing": true,
      "error_tolerance": 0.1
    }
  }'
```

### Температурный контроль
Настройка креативности и детерминированности ответов:

```bash
curl -X POST "http://localhost:8000/agent/process" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{
    "query": "Создай гипотетический сценарий",
    "context": {"domain": "планирование"},
    "preferences": {
      "creativity_level": 0.8,  // от 0.0 до 1.0
      "factuality_requirement": 0.9  // от 0.0 до 1.0
    }
  }'
```

## 🎯 CQRS и Event Sourcing API

### POST /cqrs/commands/{command_type}
Выполнение команды через CQRS паттерн.

**Пример запроса:**
```bash
curl -X POST "http://localhost:8000/cqrs/commands/ProcessTask" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{
    "command_id": "cmd_123456",
    "timestamp": "2025-11-23T21:00:00.000Z",
    "task_id": "task_001",
    "data": {"input": "test data"}
  }'
```

**Пример ответа:**
```json
{
  "success": true,
  "command_id": "cmd_123456",
  "result": {"task_id": "task_001", "status": "processed"},
  "events_generated": 1,
  "processing_time": 0.023
}
```

### GET /cqrs/queries/{query_type}
Выполнение запроса через CQRS паттерн.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/cqrs/queries/GetTaskStatus?task_id=task_001" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "success": true,
  "query_id": "qry_789012",
  "data": {
    "task_id": "task_001",
    "status": "processed",
    "data": {"input": "test data"},
    "processed_at": "2025-11-23T21:00:00.000Z"
  },
  "cached": true,
  "processing_time": 0.005
}
```

### GET /event-sourcing/agents/{agent_id}
Получение состояния агента через Event Sourcing.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/event-sourcing/agents/agent_001" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "agent_id": "agent_001",
  "state": "BUSY",
  "tasks_processed": 5,
  "total_inference_time": 12.5,
  "last_activity": "2025-11-23T21:00:00.000Z",
  "version": 8
}
```

### GET /event-sourcing/agents/{agent_id}/history
Получение истории событий агента.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/event-sourcing/agents/agent_001/history?limit=10" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "agent_id": "agent_001",
  "events": [
    {
      "event_id": "evt_001",
      "event_type": "AgentCreated",
      "timestamp": "2025-11-23T20:00:00.000Z",
      "version": 1,
      "event_data": {"agent_id": "agent_001"}
    },
    {
      "event_id": "evt_002",
      "event_type": "TaskProcessed",
      "timestamp": "2025-11-23T20:30:00.000Z",
      "version": 2,
      "event_data": {
        "task_id": "task_001",
        "result": {"output": "success"},
        "processing_time": 2.5
      }
    }
  ],
  "total_events": 8
}
```

## 🔍 Tracing и Мониторинг API

### GET /tracing/spans
Получение активных tracing spans.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/tracing/spans?service=ai-agent" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "spans": [
    {
      "span_id": "span_123",
      "trace_id": "trace_456",
      "name": "agent_process_query",
      "start_time": "2025-11-23T21:00:00.000Z",
      "duration_ms": 2345,
      "status": "success",
      "attributes": {
        "service": "ai-agent",
        "operation": "process_query",
        "user_id": "user_123"
      }
    }
  ],
  "total_active_spans": 3
}
```

### GET /monitoring/metrics
Получение метрик системы в формате Prometheus.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/monitoring/metrics"
```

**Пример ответа:**
```
# CQRS метрики
cqrs_commands_total{type="ProcessTask"} 150
cqrs_queries_total{type="GetTaskStatus"} 320
cqrs_command_duration_seconds{quantile="0.5"} 0.023
cqrs_query_duration_seconds{quantile="0.95"} 0.045

# Event Sourcing метрики
event_sourcing_events_total{type="TaskProcessed"} 89
event_sourcing_events_total{type="AgentStateChanged"} 12

# Tracing метрики
tracing_spans_total{service="ai-agent"} 1250
tracing_spans_duration_seconds{quantile="0.99"} 5.2
```

## 🛡️ Безопасность API

### GET /security/rate-limits
Проверка текущих rate limits.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/security/rate-limits" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "global_limits": {
    "requests_per_minute": 1000,
    "requests_per_hour": 50000,
    "current_usage_minute": 45,
    "current_usage_hour": 1234
  },
  "endpoint_limits": {
    "/agent/process": {
      "requests_per_minute": 100,
      "current_usage": 12
    },
    "/health": {
      "requests_per_minute": 60,
      "current_usage": 2
    }
  }
}
```

### GET /security/audit-logs
Получение audit логов (требуются admin права).

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/security/audit-logs?user_id=user_123&limit=50" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

**Пример ответа:**
```json
{
  "logs": [
    {
      "event_id": "audit_1732395600_abc123",
      "event_type": "API_ACCESS",
      "severity": "LOW",
      "timestamp": "2025-11-23T21:00:00.000Z",
      "user_id": "user_123",
      "resource": "/agent/process",
      "action": "POST",
      "status": "success",
      "request_id": "req_456789",
      "ip_address": "192.168.1.100",
      "details": {
        "response_time": 2.345,
        "tokens_used": 1250
      }
    }
  ],
  "total_logs": 1250,
  "pagination": {
    "page": 1,
    "limit": 50,
    "has_more": true
  }
}
```

### GET /security/circuit-breakers
Статус circuit breakers.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/security/circuit-breakers" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "circuit_breakers": {
    "openai_api": {
      "state": "CLOSED",
      "failure_count": 0,
      "last_failure_time": null,
      "next_retry_time": null
    },
    "database": {
      "state": "HALF_OPEN",
      "failure_count": 3,
      "last_failure_time": "2025-11-23T20:45:00.000Z",
      "next_retry_time": "2025-11-23T20:46:00.000Z"
    }
  }
}
```

## 🧪 Performance Testing API

### POST /performance/test
Запуск нагрузочного тестирования.

**Пример запроса:**
```bash
curl -X POST "http://localhost:8000/performance/test" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -d '{
    "test_type": "load_test",
    "target_endpoint": "/agent/process",
    "duration_seconds": 60,
    "concurrent_users": 10,
    "requests_per_second": 50,
    "payload": {
      "query": "Test query for performance testing",
      "context": {"domain": "test"}
    }
  }'
```

**Пример ответа:**
```json
{
  "test_id": "perf_test_001",
  "status": "running",
  "start_time": "2025-11-23T21:00:00.000Z",
  "estimated_end_time": "2025-11-23T21:01:00.000Z",
  "configuration": {
    "test_type": "load_test",
    "target_endpoint": "/agent/process",
    "duration_seconds": 60,
    "concurrent_users": 10,
    "requests_per_second": 50
  }
}
```

### GET /performance/results/{test_id}
Получение результатов тестирования производительности.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/performance/results/perf_test_001" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

**Пример ответа:**
```json
{
  "test_id": "perf_test_001",
  "status": "completed",
  "duration_seconds": 60,
  "metrics": {
    "total_requests": 3000,
    "successful_requests": 2995,
    "failed_requests": 5,
    "response_times": {
      "avg": 0.234,
      "p50": 0.198,
      "p95": 0.456,
      "p99": 0.789,
      "min": 0.123,
      "max": 1.234
    },
    "requests_per_second": 50.0,
    "error_rate": 0.0017,
    "throughput_mbps": 2.34
  },
  "recommendations": [
    "Оптимизировать обработку запросов для снижения latency p95",
    "Увеличить количество worker процессов"
  ]
}
```

## 📊 Grafana Dashboards API

### GET /grafana/dashboards
Список доступных dashboard.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/grafana/dashboards" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "dashboards": [
    {
      "id": "system_monitoring",
      "name": "System Monitoring Dashboard",
      "description": "Мониторинг системных ресурсов",
      "panels": 12,
      "tags": ["system", "monitoring", "resources"]
    },
    {
      "id": "ai_agent_monitoring",
      "name": "AI Agent Monitoring Dashboard",
      "description": "Мониторинг AI агента",
      "panels": 8,
      "tags": ["ai", "agent", "performance"]
    }
  ]
}
```

### GET /grafana/dashboards/{dashboard_id}/json
Получение JSON конфигурации dashboard для импорта в Grafana.

**Пример запроса:**
```bash
curl -X GET "http://localhost:8000/grafana/dashboards/system_monitoring/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

**Пример ответа:**
```json
{
  "dashboard": {
    "title": "System Monitoring Dashboard",
    "tags": ["system", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    }
  }
}
```

### POST /grafana/dashboards/regenerate
Перегенерация всех dashboard.

**Пример запроса:**
```bash
curl -X POST "http://localhost:8000/grafana/dashboards/regenerate" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

**Пример ответа:**
```json
{
  "status": "regenerated",
  "dashboards_generated": 4,
  "files_updated": [
    "monitoring/dashboards/system_monitoring_dashboard.json",
    "monitoring/dashboards/application_monitoring_dashboard.json",
    "monitoring/dashboards/ai_agent_monitoring_dashboard.json",
    "monitoring/dashboards/health_checks_dashboard.json"
  ],
  "timestamp": "2025-11-23T21:00:00.000Z"
}
```

## 📞 Поддерживаемые форматы

### Входные форматы
- JSON (основной формат)
- Поддержка вложений через base64 кодирование

### Выходные форматы
- JSON (по умолчанию)
- Структурированные ответы с метаданными
- Трассировки рассуждений

## 🔄 Версии API

Текущая версия API: v1

Будущие версии будут добавлены с префиксом `/v2/`, `/v3/` и т.д. для обеспечения обратной совместимости.

---

*Документация API последний раз обновлена: 2025-11-24*

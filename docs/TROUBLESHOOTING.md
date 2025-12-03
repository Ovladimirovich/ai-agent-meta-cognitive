# 🔧 Руководство по устранению неполадок мета-когнитивного AI агента

## 🚨 Распространенные проблемы и решения

### 1. Проблемы запуска и инициализации

#### Ошибка: Windows fatal exception при запуске
**Симптомы:**
- Приложение аварийно завершается при запуске
- В логах сообщение: "Windows fatal exception: access violation"

**Причины:**
- Проблемы с ONNX Runtime на Windows
- Конфликты с ChromaDB
- Несовместимые версии зависимостей

**Решения:**
```bash
# Обновить ONNX Runtime
pip install --upgrade onnxruntime onnxruntime-gpu

# Использовать CPU версию
pip uninstall onnxruntime-gpu
pip install onnxruntime

# Или использовать альтернативные эмбеддинги
pip install sentence-transformers
```

#### Ошибка: Не удается импортировать модули
**Симптомы:**
- `ModuleNotFoundError` для внутренних модулей
- `ImportError` при запуске

**Решения:**
```bash
# Убедиться, что пакет установлен в режиме разработки
pip install -e .

# Проверить структуру импортов
# Использовать абсолютные импорты:
from ai_agent_meta_cognitive.agent.core import AgentCore

# А не относительные:
from ..core import AgentCore
```

### 2. Проблемы с производительностью

#### Медленная обработка запросов (>5 секунд)
**Диагностика:**
```python
import time
from ai_agent_meta_cognitive.agent.monitoring import PerformanceMonitor

# Включить мониторинг производительности
monitor = PerformanceMonitor()
response_time = monitor.measure_response_time(agent.process_request, request)
print(f"Response time: {response_time}s")
```

**Возможные причины:**
- Использование медленных внешних API
- Отсутствие кэширования
- Неоптимальные алгоритмы

**Решения:**
1. Включить кэширование
2. Оптимизировать выбор инструментов
3. Использовать локальные модели для простых задач

#### Высокое потребление памяти
**Мониторинг:**
```python
import psutil
import gc

# Проверить потребление памяти
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb} MB")

# Принудительно вызвать сборку мусора
gc.collect()
```

**Решения:**
- Оптимизировать размер памяти агента
- Включить очистку устаревших записей
- Использовать генераторы вместо списков

### 3. Проблемы с инструментами

#### RAG инструмент не возвращает результаты
**Диагностика:**
```python
# Проверить настройки RAG
from ai_agent_meta_cognitive.agent.tools import RAGTool

rag_tool = RAGTool()
config = rag_tool.get_config()
print(f"Embedding model: {config.embedding_model}")
print(f"Similarity threshold: {config.similarity_threshold}")
```

**Решения:**
- Проверить настройки порога схожести
- Убедиться в наличии данных для поиска
- Проверить работоспособность векторной базы данных

#### Ошибка при использовании кэша
**Симптомы:**
- `Redis ConnectionError`
- `Cache miss` для кэшируемых данных

**Решения:**
```bash
# Запустить Redis
docker run -d --name redis-cache -p 6379:6379 redis:7-alpine

# Или использовать in-memory кэш
export USE_REDIS_CACHE=false
```

### 4. Проблемы с мета-когнитивными функциями

#### Низкая уверенность в ответах
**Анализ:**
```python
# Получить детализированный анализ уверенности
confidence_analysis = agent.confidence_calculator.analyze_confidence_factors({
    "query_complexity": "high",
    "data_quality": "medium",
    "tool_availability": "partial"
})

print(confidence_analysis)
```

**Решения:**
- Улучшить качество входных данных
- Расширить базу знаний
- Настроить пороги уверенности

#### Проблемы с самоанализом
**Проверка:**
```python
# Получить статус самосознания
self_awareness_status = agent.self_awareness.get_status()
print(f"Reflection capability: {self_awareness_status.reflection_enabled}")
print(f"Monitoring level: {self_awareness_status.monitoring_level}")
```

## 🕵️‍♂️ Диагностика и отладка

### Включение отладочных логов
```bash
# Установить уровень логирования DEBUG
export LOG_LEVEL=DEBUG

# Или в конфигурации:
{
  "logging": {
    "level": "DEBUG",
    "format": "detailed",
    "include_reasoning_trace": true
  }
}
```

### Анализ трассировки рассуждений
```python
# Включить трассировку рассуждений
response = await agent.process_request({
    "query": "Тестовый запрос",
    "options": {
        "enable_reasoning_trace": True
    }
})

# Просмотреть шаги рассуждения
for step in response.reasoning_trace:
    print(f"Step: {step.step_type}")
    print(f"Confidence: {step.confidence}")
    print(f"Tools used: {step.tools_used}")
```

### Проверка состояния агента
```python
# Получить полный статус агента
status = await agent.get_status()
print(f"State: {status.state}")
print(f"Confidence: {status.confidence}")
print(f"Active tools: {status.active_tools}")
print(f"Memory entries: {status.memory_stats.entries_count}")
```

## 📊 Мониторинг и метрики

### Проверка метрик производительности
```python
from ai_agent_meta_cognitive.agent.monitoring import AgentMetrics

metrics = AgentMetrics(agent)
performance_data = await metrics.get_performance_metrics()

print("Performance Metrics:")
print(f"- Avg response time: {performance_data.avg_response_time}s")
print(f"- Success rate: {performance_data.success_rate}")
print(f"- Tool utilization: {performance_data.tool_utilization}")
```

### Мониторинг когнитивного здоровья
```python
from ai_agent_meta_cognitive.agent.monitoring import CognitiveHealthMonitor

health_monitor = CognitiveHealthMonitor(agent)
health_report = await health_monitor.assess_cognitive_health()

print("Cognitive Health Report:")
print(f"- Health score: {health_report.health_score}")
print(f"- Issues count: {health_report.issues_count}")
print(f"- Recommendations: {health_report.recommendations}")
```

## 🛠️ Утилиты диагностики

### Скрипт диагностики системы
```python
# scripts/diagnostic_check.py
import asyncio
import sys
from ai_agent_meta_cognitive.agent.core import AgentCore
from ai_agent_meta_cognitive.config import AgentConfig

async def diagnostic_check():
    print("🔍 Running AI Agent diagnostic check...")
    
    try:
        # Проверка конфигурации
        config = AgentConfig()
        print(f"✅ Configuration loaded: {config.agent_name}")
        
        # Инициализация агента
        agent = AgentCore(config)
        print("✅ Agent core initialized")
        
        # Проверка инструментов
        tools_status = await agent.tool_orchestrator.get_tools_status()
        print(f"✅ Tools available: {len(tools_status)}")
        
        # Тестовый запрос
        test_response = await agent.process_request({
            "query": "Perform diagnostic self-check",
            "context": {"domain": "system"}
        })
        
        print(f"✅ Test query processed successfully")
        print(f"   - Confidence: {test_response.confidence}")
        print(f"   - Execution time: {test_response.execution_time}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostic check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(diagnostic_check())
    sys.exit(0 if success else 1)
```

### Проверка целостности памяти
```python
# scripts/check_memory_integrity.py
from ai_agent_meta_cognitive.agent.memory import MemoryManager

async def check_memory_integrity():
    memory_manager = MemoryManager()
    
    # Проверка целостности записей
    stats = await memory_manager.get_memory_stats()
    print(f"Memory entries: {stats.entries_count}")
    print(f"Estimated size: {stats.estimated_size_mb} MB")
    
    # Проверка устаревших записей
    expired_entries = await memory_manager.find_expired_entries()
    print(f"Expired entries: {len(expired_entries)}")
    
    # Оптимизация памяти
    if expired_entries:
        cleaned_count = await memory_manager.cleanup_expired_entries()
        print(f"Cleaned up: {cleaned_count} entries")
```

## 🚨 Критические ситуации

### Восстановление после сбоя
```python
# Восстановление состояния агента
async def recover_agent_state(agent):
    try:
        # Сохранить текущее состояние
        current_state = agent.get_serializable_state()
        
        # Сбросить до безопасного состояния
        await agent.reset_to_safe_state()
        
        # Восстановить только безопасные компоненты
        await agent.restore_safe_components(current_state)
        
        print("✅ Agent recovery completed")
        return True
        
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        return False
```

### Обработка критических ошибок
```python
# Обработчик критических ошибок
async def handle_critical_error(agent, error, context):
    # Логирование ошибки
    agent.logger.error(f"Critical error: {error}", extra={
        "error_type": type(error).__name__,
        "context": context,
        "agent_state": agent.get_status()
    })
    
    # Попытка восстановления
    recovery_success = await agent.attempt_recovery(error)
    
    if not recovery_success:
        # Перезапуск агента
        await agent.restart()
        print("🔄 Agent restarted after critical error")
    
    return recovery_success
```

## 🔍 Расширенная диагностика

### Профилирование производительности
```python
import cProfile
import pstats
from io import StringIO

def profile_agent_performance():
    pr = cProfile.Profile()
    pr.enable()
    
    # Выполнить операции агента
    response = agent.process_request({"query": "Test performance"})
    
    pr.disable()
    
    # Получить результаты профилирования
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    print(s.getvalue())
```

### Анализ утечек памяти
```python
import tracemalloc

def analyze_memory_leaks():
    # Начать отслеживание памяти
    tracemalloc.start()
    
    # Выполнить операции
    for i in range(100):
        response = agent.process_request({"query": f"Test {i}"})
    
    # Получить статистику
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    
    # Получить топ потребителей памяти
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("Top 10 memory consumers:")
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()
```

## 📞 Поддержка и сообщество

### Создание репорта об ошибке
Когда вы сталкиваетесь с проблемой, создайте детальный отчет:

1. **Среда выполнения**:
   - Версия Python
   - Операционная система
   - Версии зависимостей

2. **Шаги воспроизведения**:
   - Точный сценарий
   - Входные данные
   - Ожидаемое поведение
   - Фактическое поведение

3. **Логи и трассировки**:
   - Полные сообщения об ошибках
   - Трассировки стека
   - Отладочные логи

4. **Контекст**:
   - Как часто возникает проблема
   - Влияние на систему
   - Попытки решения

### Полезные команды диагностики
```bash
# Проверить версии зависимостей
pip list | grep ai-agent

# Запустить диагностику
python -m scripts.diagnostic_check

# Проверить тесты
pytest tests/ -v --tb=short

# Проверить покрытие
pytest tests/ --cov=ai_agent_meta_cognitive --cov-report=term-missing

# Проверить линтер
flake8 ai_agent_meta_cognitive/
black --check ai_agent_meta_cognitive/
```

## 🛡️ Безопасность и конфиденциальность

### Проверка уязвимостей
```bash
# Проверить зависимости на уязвимости
pip install safety
safety check

# Проверить код на уязвимости
pip install bandit
bandit -r ai_agent_meta_cognitive/
```

### Обработка чувствительных данных
```python
# Убедиться, что чувствительные данные не логируются
def sanitize_log_data(data):
    if isinstance(data, dict):
        sanitized = data.copy()
        sensitive_keys = ['api_key', 'password', 'token', 'secret']
        for key in sensitive_keys:
            if key in sanitized:
                sanitized[key] = '[REDACTED]'
        return sanitized
    return data
```

---

*Это руководство поможет вам быстро диагностировать и устранить проблемы в мета-когнитивном AI агенте. Для дополнительной помощи обращайтесь к документации отдельных компонентов или создавайте issue в репозитории проекта.*
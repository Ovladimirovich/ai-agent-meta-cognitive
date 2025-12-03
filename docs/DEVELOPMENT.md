# 🧑‍💻 Руководство для разработчиков мета-когнитивного AI агента

## 🚀 Начало работы

### Требования к разработке
- Python 3.9+
- pip
- Git
- Docker (для контейнеризации)
- IDE с поддержкой Python (рекомендуется VSCode)

### Установка в режиме разработки
```bash
# Клонирование репозитория
git clone https://github.com/your-repo/ai-agent-meta-cognitive.git
cd ai-agent-meta-cognitive

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate  # Windows

# Установка зависимостей разработки
pip install -r requirements-dev.txt

# Установка пакета в режиме разработки
pip install -e .
```

### Настройка среды разработки
Создайте файл `.env` в корне проекта:
```env
# API ключи
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# Настройки базы данных
DATABASE_URL=postgresql://user:password@localhost:5432/agent_dev
REDIS_URL=redis://localhost:6379

# Настройки агента
AGENT_DEBUG_MODE=true
AGENT_LOG_LEVEL=DEBUG
AGENT_CONFIDENCE_THRESHOLD=0.6

# Настройки тестирования
TEST_DATABASE_URL=postgresql://user:password@localhost:5432/agent_test
```

## 🏗️ Структура проекта для разработчиков

```
ai-agent-meta-cognitive/
├── agent/                    # Ядро агента
│   ├── core/                # Базовые компоненты
│   ├── self_awareness/      # Компоненты самосознания
│   ├── learning/            # Система обучения
│   ├── memory/              # Система памяти
│   └── meta_cognitive/      # Мета-когнитивные компоненты
├── api/                     # API слой
├── tools/                   # Инструменты агента
├── cache/                   # Система кэширования
├── database/                # Работа с базами данных
├── analytics/               # Система аналитики
├── tests/                   # Тесты
├── docs/                    # Документация (новая)
├── archive/                 # Архивные файлы
├── examples/                # Примеры использования
└── scripts/                 # Вспомогательные скрипты
```

## 🧪 Тестирование

### Запуск тестов
```bash
# Все тесты
pytest tests/

# Только unit тесты
pytest tests/unit/

# Только интеграционные тесты
pytest tests/integration/

# С покрытием кода
pytest tests/ --cov=ai_agent_meta_cognitive --cov-report=html

# Конкретный тест
pytest tests/test_agent_core.py::test_process_request
```

### Типы тестов

#### Unit тесты
Тестирование отдельных функций и классов:
```python
# tests/unit/test_confidence_calculator.py
import pytest
from ai_agent_meta_cognitive.agent.self_awareness.confidence_calculator import ConfidenceCalculator

@pytest.mark.asyncio
async def test_confidence_calculation():
    calculator = ConfidenceCalculator()
    
    result = await calculator.calculate_confidence({
        "query_complexity": "high",
        "tool_availability": 0.8,
        "data_quality": 0.9
    })
    
    assert 0.0 <= result <= 1.0
    assert result < 1.0  # High complexity should reduce confidence
```

#### Интеграционные тесты
Тестирование взаимодействия компонентов:
```python
# tests/integration/test_agent_tool_integration.py
import pytest
from ai_agent_meta_cognitive.agent.core import AgentCore
from ai_agent_meta_cognitive.agent.tools import RAGTool

@pytest.mark.asyncio
async def test_agent_rag_integration():
    agent = AgentCore()
    rag_tool = RAGTool()
    
    # Регистрация инструмента
    agent.tool_orchestrator.register_tool("rag", rag_tool)
    
    # Тестирование взаимодействия
    response = await agent.process_request({
        "query": "Find information about AI agents",
        "required_tools": ["rag"]
    })
    
    assert response.success
    assert response.confidence > 0.5
```

#### E2E тесты
Тестирование полных сценариев использования:
```python
# tests/e2e/test_full_workflow.py
import pytest
from ai_agent_meta_cognitive.api.main import app
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_full_agent_workflow():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/agent/process", json={
            "query": "Analyze this complex problem",
            "context": {"domain": "analytics"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "confidence" in data
        assert "reasoning_trace" in data
```

### Требования к тестам
- 90%+ покрытие для критических компонентов
- Все тесты должны проходить перед коммитом
- Использовать фикстуры для настройки тестовой среды
- Тестировать граничные условия и ошибочные сценарии

## 📝 Стандарты кодирования

### Именование
```python
# Классы - PascalCase
class AgentCore:
    pass

# Функции и переменные - snake_case
def process_request(request_data):
    agent_state = "idle"
    return result

# Константы - UPPER_SNAKE_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30.0
```

### Документация
```python
def calculate_confidence(self, input_data: dict, context: dict = None) -> float:
    """
    Calculate confidence score for the agent's response.
    
    Args:
        input_data: Dictionary containing query and context information
        context: Additional context for confidence calculation
        
    Returns:
        Float between 0.0 and 1.0 representing confidence level
        
    Raises:
        ValueError: If input_data is malformed
        RuntimeError: If calculation fails due to system issues
    """
    # Implementation here
    pass
```

### Аннотации типов
```python
from typing import Dict, List, Optional, Union, AsyncGenerator
from pydantic import BaseModel

class AgentResponse(BaseModel):
    result: str
    confidence: float
    reasoning_trace: List[Dict]

async def process_request(
    query: str,
    context: Optional[Dict[str, any]] = None
) -> AgentResponse:
    pass
```

## 🔧 Архитектурные паттерны

### Singleton для глобальных компонентов
```python
class GlobalAgentState:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### Factory для создания инструментов
```python
class ToolFactory:
    @staticmethod
    def create_tool(tool_type: str, config: dict):
        if tool_type == "rag":
            return RAGTool(config)
        elif tool_type == "analytics":
            return AnalyticsTool(config)
        # ...
```

### Observer для мониторинга
```python
class AgentObserver:
    async def on_state_change(self, old_state: str, new_state: str):
        pass

class AgentSubject:
    def __init__(self):
        self._observers: List[AgentObserver] = []
    
    def attach(self, observer: AgentObserver):
        self._observers.append(observer)
    
    def notify(self, old_state: str, new_state: str):
        for observer in self._observers:
            await observer.on_state_change(old_state, new_state)
```

## 🛠️ Создание новых компонентов

### Создание нового инструмента
```python
# tools/new_tool.py
from ai_agent_meta_cognitive.agent.tools.base_tool import BaseTool
from pydantic import BaseModel

class NewToolConfig(BaseModel):
    param1: str
    param2: int = 10

class NewTool(BaseTool):
    def __init__(self, config: NewToolConfig):
        super().__init__()
        self.config = config
        
    async def execute(self, input_data: dict) -> dict:
        """
        Execute the tool with given input.
        
        Args:
            input_data: Input parameters for the tool
            
        Returns:
            Dictionary with results
        """
        # Tool implementation
        return {"result": "success", "data": input_data}
    
    def get_capability_description(self) -> str:
        return "This tool performs new functionality"
```

### Регистрация инструмента
```python
# tools/__init__.py
from .new_tool import NewTool, NewToolConfig

__all__ = ["NewTool", "NewToolConfig"]
```

### Тестирование нового инструмента
```python
# tests/unit/test_new_tool.py
import pytest
from ai_agent_meta_cognitive.tools.new_tool import NewTool, NewToolConfig

@pytest.mark.asyncio
async def test_new_tool_basic_functionality():
    config = NewToolConfig(param1="test", param2=5)
    tool = NewTool(config)
    
    result = await tool.execute({"input": "test_data"})
    
    assert result["result"] == "success"
    assert result["data"]["input"] == "test_data"
```

## 🧠 Рекомендации по разработке мета-когнитивных функций

### Создание системы саморефлексии
```python
# self_awareness/reflection_engine.py
class ReflectionEngine:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        
    async def reflect_on_interaction(self, interaction_result: dict) -> dict:
        """
        Perform self-reflection on an interaction.
        
        Analyzes:
        - What went well
        - What could be improved
        - Patterns in successful/unsuccessful interactions
        - Self-performance metrics
        """
        analysis = {
            "success_indicators": [],
            "improvement_opportunities": [],
            "behavioral_patterns": [],
            "confidence_accuracy": 0.0
        }
        
        # Detailed analysis implementation
        return analysis
```

### Реализация адаптивного обучения
```python
# learning/adaptation_engine.py
class AdaptationEngine:
    def __init__(self):
        self.learning_strategies = {}
        self.performance_history = []
        
    async def adapt_behavior(self, experience: dict) -> dict:
        """
        Adapt agent behavior based on experience.
        
        Args:
            experience: Dictionary containing interaction experience
            
        Returns:
            Dictionary with adaptation recommendations
        """
        # Analyze experience
        success_metrics = self._analyze_experience(experience)
        
        # Update strategies
        self._update_learning_strategies(success_metrics)
        
        # Generate adaptation recommendations
        recommendations = self._generate_recommendations(success_metrics)
        
        return recommendations
```

## 🔁 CI/CD Практики

### Запуск проверок перед коммитом
```bash
# Запуск всех проверок
make check-all

# Или по отдельности:
make lint          # Проверка кода
make test-unit     # Unit тесты
make test-integration  # Интеграционные тесты
make security-check    # Проверка безопасности
```

### Docker сборка
```dockerfile
# Dockerfile.development
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### GitHub Actions
```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install -e .
        
    - name: Run tests
      run: pytest tests/ --cov=ai_agent_meta_cognitive
      
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 🐛 Отладка и устранение неполадок

### Включение отладки
```python
import logging

# Включить отладочные логи
logging.basicConfig(level=logging.DEBUG)

# Или через конфигурацию
config = AgentConfig(debug_mode=True, log_level="DEBUG")
```

### Инструменты диагностики
```python
from ai_agent_meta_cognitive.agent.monitoring import AgentHealthMonitor

# Проверка состояния агента
health = await AgentHealthMonitor.check_health()

# Получение трассировки рассуждений
reasoning_trace = agent.get_recent_reasoning_trace()

# Анализ производительности
perf_metrics = agent.get_performance_metrics()
```

### Распространенные проблемы и решения

#### Проблема: Медленная обработка запросов
**Решение:**
- Проверить использование кэша
- Оптимизировать инструменты
- Проверить базы данных

#### Проблема: Низкая уверенность в ответах
**Решение:**
- Проверить качество входных данных
- Обновить стратегии выбора инструментов
- Улучшить анализ контекста

#### Проблема: Утечки памяти
**Решение:**
- Проверить систему управления памятью
- Внедрить инвалидацию кэша
- Оптимизировать хранение исторических данных

## 🚀 Развёртывание

### Подготовка к развёртыванию
```bash
# Проверка готовности
make pre-deploy-check

# Сборка релиза
make build-release

# Тестирование в staging
make deploy-staging
```

### Production развёртывание
```bash
# С помощью Docker
docker build -t ai-agent:latest .
docker run -d -p 8000:8000 --env-file .env.production ai-agent:latest

# Или с помощью Kubernetes
kubectl apply -f k8s/deployment.yaml
```

## 📈 Мониторинг производительности

### Ключевые метрики
- Время отклика
- Уровень уверенности
- Частота использования инструментов
- Эффективность обучения
- Стабильность работы

### Настройка мониторинга
```python
from ai_agent_meta_cognitive.agent.monitoring import MetricsCollector

# Регистрация метрик
collector = MetricsCollector()
collector.register_metric("response_time", "histogram")
collector.register_metric("confidence_score", "gauge")
collector.register_metric("tool_usage", "counter")
```

## 🤝 Вклад в проект

### Процесс внесения изменений
1. Форк проекта
2. Создание ветки для новой функции
3. Реализация изменений
4. Написание/обновление тестов
5. Обновление документации
6. Отправка PR

### Требования к PR
- Чистый код без лишних изменений
- Соответствие стандартам кодирования
- Наличие тестов для новых функций
- Обновление документации
- Описание изменений в описании PR

---

*Это руководство поможет вам эффективно разрабатывать и вносить улучшения в мета-когнитивный AI агент. Для дополнительной информации обращайтесь к соответствующим модулям в исходном коде.*
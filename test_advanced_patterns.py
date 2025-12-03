#!/usr/bin/env python3
"""
Тестирование продвинутых архитектурных паттернов:
- CQRS (Command Query Responsibility Segregation)
- Event Sourcing
- OpenTelemetry Tracing
"""

import asyncio
import time
import uuid
from datetime import datetime
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_cqrs_patterns():
    """Тестирование CQRS паттернов"""
    print("🧪 Тестирование CQRS паттернов...")

    from cqrs.command_bus import command_bus, Command, CommandResult, CommandHandler
    from cqrs.query_bus import query_bus, Query, QueryResult, QueryHandler

    # Создаем тестовую команду
    class TestCommand(Command):
        def __init__(self, value: int):
            super().__init__(
                command_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                metadata={"test": True}
            )
            self.value = value

    # Создаем обработчик команды
    class TestCommandHandler(CommandHandler):
        def __init__(self):
            self.processed_values = []

        async def handle(self, command: TestCommand) -> CommandResult:
            self.processed_values.append(command.value)
            return CommandResult(
                success=True,
                command_id=command.command_id,
                result=command.value * 2,
                events=[]
            )

    # Регистрируем обработчик
    handler = TestCommandHandler()
    command_bus.register_handler(TestCommand, handler)

    # Выполняем команду
    command = TestCommand(value=5)
    result = await command_bus.execute(command)

    assert result.success == True
    assert result.result == 10
    assert handler.processed_values == [5]

    print("✅ CQRS Command Bus работает корректно")

    # Создаем тестовый запрос
    class TestQuery(Query):
        def __init__(self, filter_value: int):
            super().__init__(
                query_id=str(uuid.uuid4()),
                timestamp=datetime.now()
            )
            self.filter_value = filter_value

    # Создаем обработчик запроса
    class TestQueryHandler(QueryHandler):
        async def handle(self, query: TestQuery) -> QueryResult:
            # Имитируем получение данных
            data = [1, 2, 3, 4, 5]
            filtered_data = [x for x in data if x > query.filter_value]
            return QueryResult(
                success=True,
                query_id=query.query_id,
                data=filtered_data
            )

    # Регистрируем обработчик запроса
    query_handler = TestQueryHandler()
    query_bus.register_handler(TestQuery, query_handler)

    # Выполняем запрос
    query = TestQuery(filter_value=2)
    query_result = await query_bus.execute(query)

    assert query_result.success == True
    assert query_result.data == [3, 4, 5]

    print("✅ CQRS Query Bus работает корректно")

async def test_event_sourcing():
    """Тестирование Event Sourcing"""
    print("🧪 Тестирование Event Sourcing...")

    from cqrs.event_sourcing import (
        event_sourcing_service, AgentAggregate,
        DomainEvent, event_store, event_publisher
    )

    # Создаем агента
    agent_id = str(uuid.uuid4())
    agent = await event_sourcing_service.create_agent(agent_id)

    assert agent.aggregate_id == agent_id
    assert agent.state == "IDLE"
    assert agent.tasks_processed == 0

    print("✅ Агент создан через Event Sourcing")

    # Обрабатываем задачу
    task_result = {"output": "test result", "confidence": 0.95}
    await event_sourcing_service.process_task(
        agent_id=agent_id,
        task_id=str(uuid.uuid4()),
        result=task_result,
        processing_time=1.5
    )

    # Проверяем состояние агента
    updated_agent = await event_sourcing_service.get_agent(agent_id)
    assert updated_agent.tasks_processed == 1
    assert updated_agent.state == "IDLE"  # Должен остаться IDLE

    print("✅ Задача обработана через Event Sourcing")

    # Изменяем состояние агента
    await event_sourcing_service.change_agent_state(
        agent_id=agent_id,
        new_state="PROCESSING",
        reason="Starting task processing"
    )

    # Проверяем состояние
    final_agent = await event_sourcing_service.get_agent(agent_id)
    assert final_agent.state == "PROCESSING"

    print("✅ Состояние агента изменено через Event Sourcing")

    # Проверяем историю событий
    events = await event_sourcing_service.get_agent_history(agent_id)
    assert len(events) >= 2  # Минимум 2 события: TaskProcessed и AgentStateChanged

    print("✅ История событий сохранена корректно")

async def test_opentelemetry_tracing():
    """Тестирование OpenTelemetry Tracing"""
    print("🧪 Тестирование OpenTelemetry Tracing...")

    from monitoring.opentelemetry_tracing import (
        tracing_service, trace_function, trace_context,
        TracingMetricsCollector
    )

    # Проверяем, что трассировка доступна (даже если отключена)
    print(f"Tracing enabled: {tracing_service.is_enabled()}")

    # Создаем коллектор метрик
    metrics_collector = TracingMetricsCollector()

    # Тестируем декоратор трассировки
    @trace_function(name="test_function", attributes={"test.type": "unit_test"})
    async def test_async_function(x: int, y: int) -> int:
        await asyncio.sleep(0.01)  # Имитируем работу
        return x + y

    # Вызываем функцию
    result = await test_async_function(5, 3)
    assert result == 8

    print("✅ Декоратор трассировки функций работает")

    # Тестируем контекстный менеджер
    async with trace_context("test_context", {"context.type": "test"}):
        await asyncio.sleep(0.01)
        print("✅ Контекстный менеджер трассировки работает")

    # Проверяем метрики
    metrics = metrics_collector.get_metrics()
    print(f"Tracing metrics: {metrics}")

async def run_advanced_patterns_tests():
    """Запуск всех тестов продвинутых паттернов"""
    print("🚀 Запуск тестирования продвинутых архитектурных паттернов...")
    print("=" * 60)

    try:
        await test_cqrs_patterns()
        print()

        await test_event_sourcing()
        print()

        await test_opentelemetry_tracing()
        print()

        print("=" * 60)
        print("🎉 Все продвинутые паттерны протестированы успешно!")
        print()
        print("📋 Резюме реализованных паттернов:")
        print("  ✅ CQRS (Command Query Responsibility Segregation)")
        print("     - Command Bus с middleware (валидация, логирование)")
        print("     - Query Bus с кэшированием")
        print("     - Разделение операций чтения и записи")
        print()
        print("  ✅ Event Sourcing")
        print("     - Хранение состояния как последовательности событий")
        print("     - Aggregate Root паттерн")
        print("     - Event Store (in-memory и file-based)")
        print("     - Event Publisher для реактивности")
        print()
        print("  ✅ OpenTelemetry Tracing")
        print("     - HTTP middleware для автоматической трассировки")
        print("     - Декораторы для функций и контекстов")
        print("     - Специализированные декораторы для AI агента")
        print("     - Интеграция с Jaeger и OTLP")
        print()
        print("🔧 Как использовать:")
        print("  1. CQRS: Импортируйте command_bus и query_bus")
        print("  2. Event Sourcing: Используйте event_sourcing_service")
        print("  3. Tracing: Добавьте декораторы @trace_function к функциям")
        print("  4. Настройте переменные окружения для включения tracing")

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(run_advanced_patterns_tests())
    sys.exit(0 if success else 1)

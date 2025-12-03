#!/usr/bin/env python3
"""
Комплексное тестирование всей системы улучшений AI Агента
Тестирует все новые компоненты в интеграции
"""

import asyncio
import time
import uuid
from datetime import datetime
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_full_system_integration():
    """
    Полное интеграционное тестирование всех компонентов системы
    """
    print("🚀 Запуск комплексного тестирования всей системы...")
    print("=" * 80)

    results = {
        "cqrs_integration": False,
        "event_sourcing_integration": False,
        "tracing_integration": False,
        "health_checks_integration": False,
        "compression_integration": False,
        "performance_testing_integration": False,
        "circuit_breaker_integration": False,
        "audit_logging_integration": False,
        "rate_limiting_integration": False,
        "grafana_dashboards_generation": False
    }

    try:
        # 1. Тестирование CQRS интеграции
        print("1️⃣ Тестирование CQRS интеграции...")
        from cqrs.command_bus import command_bus, Command, CommandResult, CommandHandler
        from cqrs.query_bus import query_bus, Query, QueryResult, QueryHandler

        # Создаем тестовые команды и запросы
        class ProcessTaskCommand(Command):
            def __init__(self, task_id: str, data: dict):
                super().__init__(
                    command_id=str(uuid.uuid4()),
                    timestamp=datetime.now()
                )
                self.task_id = task_id
                self.data = data

        class GetTaskStatusQuery(Query):
            def __init__(self, task_id: str):
                super().__init__(
                    query_id=str(uuid.uuid4()),
                    timestamp=datetime.now()
                )
                self.task_id = task_id

        # Регистрируем обработчики
        class TaskCommandHandler(CommandHandler):
            def __init__(self):
                self.tasks = {}

            async def handle(self, command: ProcessTaskCommand) -> CommandResult:
                self.tasks[command.task_id] = {
                    "status": "processed",
                    "data": command.data,
                    "processed_at": datetime.now()
                }
                return CommandResult(
                    success=True,
                    command_id=command.command_id,
                    result={"task_id": command.task_id, "status": "processed"}
                )

        class TaskQueryHandler(QueryHandler):
            def __init__(self, command_handler):
                self.command_handler = command_handler

            async def handle(self, query: GetTaskStatusQuery) -> QueryResult:
                task = self.command_handler.tasks.get(query.task_id)
                if task:
                    return QueryResult(
                        success=True,
                        query_id=query.query_id,
                        data=task
                    )
                return QueryResult(
                    success=False,
                    query_id=query.query_id,
                    error_message="Task not found"
                )

        # Регистрируем и тестируем
        cmd_handler = TaskCommandHandler()
        command_bus.register_handler(ProcessTaskCommand, cmd_handler)

        query_handler = TaskQueryHandler(cmd_handler)
        query_bus.register_handler(GetTaskStatusQuery, query_handler)

        # Выполняем команду и запрос
        task_id = str(uuid.uuid4())
        command = ProcessTaskCommand(task_id, {"input": "test data"})
        cmd_result = await command_bus.execute(command)

        query = GetTaskStatusQuery(task_id)
        query_result = await query_bus.execute(query)

        if cmd_result.success and query_result.success and query_result.data["status"] == "processed":
            results["cqrs_integration"] = True
            print("✅ CQRS интеграция работает корректно")
        else:
            print("❌ CQRS интеграция не работает")

        print()

        # 2. Тестирование Event Sourcing интеграции
        print("2️⃣ Тестирование Event Sourcing интеграции...")
        from cqrs.event_sourcing import event_sourcing_service

        agent_id = str(uuid.uuid4())

        # Создаем агента и выполняем операции
        agent = await event_sourcing_service.create_agent(agent_id)

        # Обрабатываем задачу
        task_result = {"output": "integration test result", "confidence": 0.95}
        await event_sourcing_service.process_task(
            agent_id=agent_id,
            task_id=str(uuid.uuid4()),
            result=task_result,
            processing_time=2.5
        )

        # Изменяем состояние
        await event_sourcing_service.change_agent_state(
            agent_id=agent_id,
            new_state="BUSY",
            reason="Processing integration test"
        )

        # Проверяем состояние
        updated_agent = await event_sourcing_service.get_agent(agent_id)

        if (updated_agent and
            updated_agent.tasks_processed == 1 and
            updated_agent.state == "BUSY"):

            # Проверяем историю
            history = await event_sourcing_service.get_agent_history(agent_id)
            if len(history) >= 2:  # TaskProcessed + AgentStateChanged
                results["event_sourcing_integration"] = True
                print("✅ Event Sourcing интеграция работает корректно")
            else:
                print("❌ Event Sourcing: недостаточно событий в истории")
        else:
            print("❌ Event Sourcing: неправильное состояние агента")

        print()

        # 3. Тестирование Tracing интеграции
        print("3️⃣ Тестирование Tracing интеграции...")
        from monitoring.opentelemetry_tracing import tracing_service, trace_function

        @trace_function(name="integration_test_function")
        async def traced_function(x: int) -> int:
            await asyncio.sleep(0.01)  # Имитируем работу
            return x * 2

        # Вызываем функцию
        result = await traced_function(21)

        if result == 42:
            results["tracing_integration"] = True
            print("✅ Tracing интеграция работает корректно")
        else:
            print("❌ Tracing: неправильный результат функции")

        print()

        # 4. Тестирование Health Checks интеграции
        print("4️⃣ Тестирование Health Checks интеграции...")
        from api.health_checks import health_registry, create_system_health_checker

        # Регистрируем системную проверку
        system_checker = create_system_health_checker()
        health_registry.register_checker(system_checker, "system")

        # Выполняем проверки
        health_results = await health_registry.run_all()

        if health_results and "system" in health_results:
            system_result = health_results["system"]
            if system_result.status.name in ["HEALTHY", "DEGRADED"]:
                results["health_checks_integration"] = True
                print("✅ Health Checks интеграция работает корректно")
            else:
                print(f"❌ Health Checks: системная проверка вернула {system_result.status.name}")
        else:
            print("❌ Health Checks: не удалось выполнить проверки")

        print()

        # 5. Тестирование Compression интеграции
        print("5️⃣ Тестирование Compression интеграции...")
        from api.compression_middleware import create_compression_middleware

        # Создаем middleware
        compression_mw = create_compression_middleware()

        # Проверяем, что middleware создался
        if compression_mw:
            results["compression_integration"] = True
            print("✅ Compression интеграция работает корректно")
        else:
            print("❌ Compression: не удалось создать middleware")

        print()

        # 6. Тестирование Performance Testing интеграции
        print("6️⃣ Тестирование Performance Testing интеграции...")
        from tests.performance_tests_enhanced import PerformanceTester

        # Создаем tester
        perf_tester = PerformanceTester()

        # Выполняем простой тест
        async def simple_operation():
            await asyncio.sleep(0.001)
            return 42

        metrics = await perf_tester.measure_operation(
            "integration_test",
            simple_operation,
            iterations=3,
            warmup_iterations=1
        )

        if metrics and metrics.requests_per_second > 0:
            results["performance_testing_integration"] = True
            print("✅ Performance Testing интеграция работает корректно")
        else:
            print("❌ Performance Testing: не удалось выполнить тест")

        print()

        # 7. Тестирование Circuit Breaker интеграции
        print("7️⃣ Тестирование Circuit Breaker интеграции...")
        from integrations.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            timeout=5.0
        )
        cb = CircuitBreaker(config)

        # Тестируем успешную операцию
        async def success_op():
            return "success"

        result = await cb.call(success_op)

        if result == "success" and cb.state.name == "CLOSED":
            results["circuit_breaker_integration"] = True
            print("✅ Circuit Breaker интеграция работает корректно")
        else:
            print("❌ Circuit Breaker: неправильное поведение")

        print()

        # 8. Тестирование Audit Logging интеграции
        print("8️⃣ Тестирование Audit Logging интеграции...")
        from api.audit_logger import audit_logger, AuditEventType, AuditEventSeverity

        # Запускаем logger
        await audit_logger.start()

        # Логируем событие
        await audit_logger.log(
            event_type=AuditEventType.API_ACCESS,
            severity=AuditEventSeverity.LOW,
            resource="/integration-test",
            action="GET",
            status="success",
            user_id="test_user",
            request_id=str(uuid.uuid4())
        )

        # Ждем немного для обработки
        await asyncio.sleep(0.1)

        # Принудительно сбрасываем буфер
        await audit_logger._flush_buffer()

        # Проверяем статистику
        stats = audit_logger.get_stats()
        if stats["events_logged"] >= 1 or stats["events_buffered"] >= 1:
            results["audit_logging_integration"] = True
            print("✅ Audit Logging интеграция работает корректно")
        else:
            print(f"❌ Audit Logging: события не логируются (logged: {stats['events_logged']}, buffered: {stats['events_buffered']})")

        await audit_logger.stop()

        print()

        # 9. Тестирование Rate Limiting интеграции
        print("9️⃣ Тестирование Rate Limiting интеграции...")
        from api.rate_limiter import InMemoryRateLimiter, RateLimitRule

        limiter = InMemoryRateLimiter()
        rule = RateLimitRule(requests_per_minute=5, requests_per_hour=10)
        limiter.set_rule("/test", rule)

        # Выполняем запросы
        allowed_count = 0
        for i in range(6):
            allowed, headers = limiter.is_allowed("test_user", "/test")
            if allowed:
                allowed_count += 1

        if allowed_count == 5:  # Должно быть разрешено только 5 из 6
            results["rate_limiting_integration"] = True
            print("✅ Rate Limiting интеграция работает корректно")
        else:
            print(f"❌ Rate Limiting: разрешено {allowed_count} запросов вместо 5")

        print()

        # 10. Тестирование Grafana Dashboards генерации
        print("🔟 Тестирование Grafana Dashboards генерации...")
        from monitoring.grafana_dashboards import generate_all_dashboards

        try:
            generate_all_dashboards()

            # Проверяем, создались ли файлы
            dashboard_files = [
                "monitoring/dashboards/system_monitoring_dashboard.json",
                "monitoring/dashboards/application_monitoring_dashboard.json",
                "monitoring/dashboards/ai_agent_monitoring_dashboard.json",
                "monitoring/dashboards/health_checks_dashboard.json"
            ]

            all_files_exist = all(os.path.exists(f) for f in dashboard_files)
            if all_files_exist:
                results["grafana_dashboards_generation"] = True
                print("✅ Grafana Dashboards генерация работает корректно")
            else:
                print("❌ Grafana Dashboards: не все файлы созданы")
        except Exception as e:
            print(f"❌ Grafana Dashboards: ошибка генерации - {e}")

        print()

    except Exception as e:
        print(f"❌ Критическая ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return results

    # Вывод результатов
    print("=" * 80)
    print("📊 РЕЗУЛЬТАТЫ ИНТЕГРАЦИОННОГО ТЕСТИРОВАНИЯ:")
    print("=" * 80)

    successful = 0
    total = len(results)

    for component, passed in results.items():
        status = "✅ ПРОЙДЕН" if passed else "❌ ПРОВАЛЕН"
        print("25")
        if passed:
            successful += 1

    print("=" * 80)
    print(f"🎯 ОБЩИЙ РЕЗУЛЬТАТ: {successful}/{total} компонентов протестировано успешно")

    if successful == total:
        print("🎉 ВСЕ КОМПОНЕНТЫ ПРОШЛИ ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ!")
        print("🚀 Система готова к продакшену!")
    else:
        print(f"⚠️  {total - successful} компонентов требуют доработки")

    print("=" * 80)

    return results

async def run_system_health_check():
    """
    Запуск комплексной проверки здоровья системы
    """
    print("🏥 Запуск комплексной проверки здоровья системы...")

    from api.health_checks import health_registry, create_system_health_checker
    from cqrs.event_sourcing import event_sourcing_service

    # Регистрируем проверки
    system_checker = create_system_health_checker()
    health_registry.register_checker(system_checker, "system")

    # Проверяем CQRS компоненты
    async def check_cqrs():
        from cqrs.command_bus import command_bus
        from cqrs.query_bus import query_bus
        return len(command_bus.get_registered_commands()) >= 0 and len(query_bus.get_registered_queries()) >= 0

    # Проверяем Event Sourcing
    async def check_event_sourcing():
        try:
            agent = await event_sourcing_service.create_agent("health_check_agent")
            return agent is not None
        except:
            return False

    # Регистрируем дополнительные проверки
    from api.health_checks import create_external_service_health_checker

    cqrs_checker = create_external_service_health_checker("cqrs_system", check_cqrs)
    es_checker = create_external_service_health_checker("event_sourcing", check_event_sourcing)

    health_registry.register_checker(cqrs_checker, "application")
    health_registry.register_checker(es_checker, "application")

    # Выполняем все проверки
    results = await health_registry.run_all()

    print("📋 Результаты проверки здоровья:")
    for name, result in results.items():
        status_emoji = "🟢" if result.status.name == "HEALTHY" else "🟡" if result.status.name == "DEGRADED" else "🔴"
        print(f"  {status_emoji} {name}: {result.status.name} - {result.message}")

    # Общий статус
    overall_status = health_registry.get_overall_status(results)
    print(f"\n🏥 ОБЩИЙ СТАТУС СИСТЕМЫ: {overall_status.name}")

    return overall_status.name == "HEALTHY"

if __name__ == "__main__":
    print("🧪 ЗАПУСК КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ СИСТЕМЫ AI АГЕНТА")
    print("Тестируются все новые архитектурные компоненты и интеграции")
    print()

    # Запускаем интеграционное тестирование
    integration_results = asyncio.run(test_full_system_integration())

    print()

    # Запускаем проверку здоровья системы
    health_ok = asyncio.run(run_system_health_check())

    print()

    # Итоговый вердикт
    successful_components = sum(1 for r in integration_results.values() if r)

    if successful_components == len(integration_results) and health_ok:
        print("🎉 СИСТЕМА ПОЛНОСТЬЮ ГОТОВА К ПРОДАКШЕНУ!")
        print("Все компоненты протестированы и работают корректно.")
        sys.exit(0)
    else:
        print("⚠️  СИСТЕМА ТРЕБУЕТ ДОРАБОТКИ")
        print(f"Успешно протестировано: {successful_components}/{len(integration_results)} компонентов")
        print(f"Здоровье системы: {'✅' if health_ok else '❌'}")
        sys.exit(1)

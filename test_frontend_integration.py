#!/usr/bin/env python3
"""
Тест интеграции frontend-backend для мета-когнитивного AI агента
"""

import requests
import time
import json
from typing import Dict, Any

class FrontendIntegrationTester:
    def __init__(self, backend_url: str = "http://localhost:8000", frontend_url: str = "http://localhost:3000"):
        self.backend_url = backend_url
        self.frontend_url = frontend_url

    def test_backend_health(self) -> bool:
        """Тест health check backend"""
        try:
            response = requests.get(f"{self.backend_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Backend health check: {data}")
                return True
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Backend health check error: {e}")
            return False

    def test_agent_process(self, query: str) -> Dict[str, Any]:
        """Тест обработки запроса агентом"""
        try:
            payload = {
                "query": query,
                "user_id": "test_user_frontend",
                "session_id": f"session_{int(time.time())}"
            }

            response = requests.post(
                f"{self.backend_url}/agent/process",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Agent response: {data}")
                return data
            else:
                print(f"❌ Agent process failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Agent process error: {e}")
            return {}

    def test_cors_headers(self) -> bool:
        """Тест CORS headers для frontend"""
        try:
            # OPTIONS запрос для проверки CORS
            response = requests.options(
                f"{self.backend_url}/agent/process",
                headers={
                    "Origin": self.frontend_url,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type"
                }
            )

            cors_headers = [
                'access-control-allow-origin',
                'access-control-allow-methods',
                'access-control-allow-headers'
            ]

            has_cors = any(header in response.headers for header in cors_headers)
            if has_cors:
                print(f"✅ CORS headers present: {dict(response.headers)}")
                return True
            else:
                print(f"❌ CORS headers missing: {dict(response.headers)}")
                return False
        except Exception as e:
            print(f"❌ CORS test error: {e}")
            return False

    def run_integration_tests(self):
        """Запуск всех интеграционных тестов"""
        print("🚀 Начинаем тестирование интеграции Frontend-Backend")
        print("=" * 60)

        # Тест 1: Backend health
        print("\n1. Тестирование backend health check...")
        backend_ok = self.test_backend_health()

        # Тест 2: CORS
        print("\n2. Тестирование CORS headers...")
        cors_ok = self.test_cors_headers()

        # Тест 3: Agent processing
        print("\n3. Тестирование обработки запросов агентом...")
        test_queries = [
            "Привет! Расскажи о себе",
            "Как работает мета-когнитивная система?",
            "Что ты можешь делать?"
        ]

        agent_responses = []
        for query in test_queries:
            print(f"\n   Тестируем запрос: '{query}'")
            response = self.test_agent_process(query)
            if response:
                agent_responses.append(response)
            time.sleep(0.5)  # Небольшая пауза между запросами

        # Итоги тестирования
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")

        tests_passed = sum([backend_ok, cors_ok, len(agent_responses) > 0])
        total_tests = 3

        print(f"✅ Пройдено тестов: {tests_passed}/{total_tests}")

        if backend_ok:
            print("✅ Backend сервер работает")
        else:
            print("❌ Backend сервер недоступен")

        if cors_ok:
            print("✅ CORS настроен корректно")
        else:
            print("❌ Проблемы с CORS")

        if agent_responses:
            print(f"✅ Агент обработал {len(agent_responses)} запросов")
            print(f"   Средняя уверенность: {sum(r.get('confidence', 0) for r in agent_responses) / len(agent_responses):.2f}")
        else:
            print("❌ Агент не обработал ни одного запроса")

        print("\n" + "=" * 60)
        if tests_passed == total_tests:
            print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система готова к использованию.")
            print(f"🌐 Frontend доступен: {self.frontend_url}")
            print(f"🔧 Backend API доступен: {self.backend_url}")
        else:
            print("⚠️  Некоторые тесты не пройдены. Проверьте логи выше.")

        return tests_passed == total_tests

def main():
    tester = FrontendIntegrationTester()
    success = tester.run_integration_tests()

    if not success:
        exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Простой тест API для проверки работоспособности
"""

import requests
import json

def test_api():
    base_url = "http://localhost:8000"

    print("🧪 Тестирование API эндпоинтов...")

    # Тест 1: Корневой эндпоинт
    try:
        print("\n1. Тестирование корневого эндпоинта...")
        response = requests.get(f"{base_url}/")
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Ответ: {data}")
        else:
            print(f"   ❌ Ошибка: {response.text}")
    except Exception as e:
        print(f"   ❌ Исключение: {e}")

    # Тест 2: Health check
    try:
        print("\n2. Тестирование health check...")
        response = requests.get(f"{base_url}/health")
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health: {data}")
        elif response.status_code == 503:
            print("   ⚠️  Сервис инициализируется (503)")
        else:
            print(f"   ❌ Ошибка: {response.text}")
    except Exception as e:
        print(f"   ❌ Исключение: {e}")

    # Тест 3: Debug test (без аутентификации)
    try:
        print("\n3. Тестирование debug эндпоинта...")
        response = requests.get(f"{base_url}/debug/test")
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Debug: {data}")
        else:
            print(f"   ❌ Ошибка: {response.text}")
    except Exception as e:
        print(f"   ❌ Исключение: {e}")

    # Тест 4: Agent process (с аутентификацией)
    try:
        print("\n4. Тестирование обработки запроса агентом...")
        payload = {
            "query": "Привет, это тестовый запрос",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        response = requests.post(
            f"{base_url}/agent/process",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Агент ответил: {data.get('content', '')[:100]}...")
        elif response.status_code == 401:
            print("   🔒 Требуется аутентификация (401)")
        elif response.status_code == 503:
            print("   ⚠️  Сервис недоступен (503)")
        else:
            print(f"   ❌ Ошибка: {response.text}")
    except Exception as e:
        print(f"   ❌ Исключение: {e}")

    print("\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    test_api()

"""
Тестирование эндпоинта /api/health
"""
import asyncio
from fastapi.testclient import TestClient
from api.main import app

def test_health_endpoint():
    """Тестируем эндпоинт /api/health"""
    client = TestClient(app)

    # Тестируем основной эндпоинт /health
    print("Тестируем эндпоинт /health...")
    response = client.get("/health")
    print(f"Status код: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Ответ: {data}")
        print("✅ /health эндпоинт работает корректно")
    else:
        print(f"❌ Ошибка /health: {response.text}")

    print("\n" + "="*50 + "\n")

    # Тестируем эндпоинт /api/health
    print("Тестируем эндпоинт /api/health...")
    response = client.get("/api/health")
    print(f"Status код: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Ответ: {data}")
        print("✅ /api/health эндпоинт работает корректно")
    else:
        print(f"❌ Ошибка /api/health: {response.text}")

    print("\n" + "="*50 + "\n")

    # Тестируем корневой эндпоинт для проверки общего состояния приложения
    print("Тестируем корневой эндпоинт /...")
    response = client.get("/")
    print(f"Status код: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Ответ: {data}")
        print("✅ Корневой эндпоинт работает корректно")
    else:
        print(f"❌ Ошибка корневого эндпоинта: {response.text}")

if __name__ == "__main__":
    test_health_endpoint()

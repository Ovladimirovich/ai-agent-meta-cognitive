#!/usr/bin/env python3
"""
Простой тест API для проверки работы сервера
"""

import requests
import json

def test_api(port=8000):
    print(f"🧪 Тестирование API сервера на порту {port}...\n")

    base_url = f'http://localhost:{port}'

    # Проверяем корневой эндпоинт
    try:
        print("1. Проверка корневого эндпоинта...")
        response = requests.get(f'{base_url}/')
        print(f'   Статус: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'   Сообщение: {data.get("message", "N/A")}')
            print("   ✅ Корневой эндпоинт работает")
        else:
            print(f"   ❌ Ошибка: {response.text}")
        print()
    except Exception as e:
        print(f'   ❌ Ошибка подключения: {e}')
        print()

    # Проверяем health
    try:
        print("2. Проверка health check...")
        response = requests.get(f'{base_url}/health')
        print(f'   Статус: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            health_score = data.get("health_score", "N/A")
            issues_count = data.get("issues_count", "N/A")
            print(f'   Health score: {health_score}')
            print(f'   Issues count: {issues_count}')
            if health_score != "N/A" and health_score > 0.5:
                print("   ✅ Health check в порядке")
            else:
                print("   ⚠️  Health score низкий")
        else:
            print(f"   ❌ Ошибка: {response.text}")
        print()
    except Exception as e:
        print(f'   ❌ Ошибка health check: {e}')
        print()

    # Проверяем agent process
    try:
        print("3. Проверка обработки запроса агентом...")
        payload = {
            'query': 'Привет! Как дела?',
            'user_id': 'test_user',
            'session_id': 'test_session_001'
        }
        response = requests.post(f'{base_url}/agent/process', json=payload)
        print(f'   Статус: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            agent_id = data.get("id", "N/A")
            content = data.get("content", "N/A")
            confidence = data.get("confidence", "N/A")
            print(f'   ID ответа: {agent_id}')
            print(f'   Содержимое: {content[:100]}...' if len(str(content)) > 100 else f'   Содержимое: {content}')
            print(f'   Уверенность: {confidence}')
            print("   ✅ Агент обработал запрос успешно")
        else:
            print(f"   ❌ Ошибка: {response.text}")
        print()
    except Exception as e:
        print(f'   ❌ Ошибка обработки запроса: {e}')
        print()

    print("🎉 Тестирование завершено!")

if __name__ == "__main__":
    # Сначала попробуем порт 8001 (тестовый сервер), потом 8000
    try:
        test_api(8001)
    except:
        print("Тестовый сервер на 8001 не доступен, пробуем основной на 8000...")
        test_api(8000)

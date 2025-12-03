#!/usr/bin/env python3
"""
Прямой тест Ollama интеграции
"""

import asyncio
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrations.llm_client import create_llm_client

async def test_ollama():
    """Тестирование Ollama напрямую"""
    print("🔥 Тестирование Ollama (Gemma3) напрямую")
    print("=" * 50)

    try:
        # Создаем клиента Ollama
        client = await create_llm_client(
            provider="ollama",
            api_key="http://localhost:11435",
            model="gemma3:1b",
            temperature=0.7,
            max_tokens=500
        )

        print("✅ Ollama клиент создан успешно!")

        # Тестируем генерацию ответа
        print("📤 Отправка тестового запроса...")
        response = await client.generate_response(
            prompt="Привет! Ты AI агент с мета-познанием. Расскажи о себе кратко.",
            system_message="Ты полезный AI ассистент. Отвечай кратко и по делу на русском языке."
        )

        print("✅ Ответ получен!")
        print(f"🤖 Модель: {response['model']} ({response['provider']})")
        print(f"🎯 Уверенность: {response['confidence']}")
        print(f"⏱️ Время: {response.get('processing_time', 0):.2f} сек")
        print(f"💬 Ответ: {response['content']}")

        await client.__aexit__(None, None, None)
        print("\n🎉 Ollama работает perfectly! Gemma3 готова к использованию!")

    except Exception as e:
        print(f"❌ Ошибка при тестировании Ollama: {e}")
        print("\n💡 Возможные решения:")
        print("1. Проверьте, что Ollama запущен: ollama serve")
        print("2. Проверьте модель: ollama list")
        print("3. Скачайте модель: ollama pull gemma3:1b")
        print("4. Проверьте порт: curl http://localhost:11435/api/tags")

if __name__ == "__main__":
    asyncio.run(test_ollama())

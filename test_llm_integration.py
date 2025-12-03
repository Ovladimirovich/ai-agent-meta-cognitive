#!/usr/bin/env python3
"""
Скрипт для тестирования LLM интеграции
"""

import asyncio
import os
import sys
from integrations.llm_client import create_llm_client


async def test_llm_integration():
    """Тестирование интеграции с LLM"""

    print("🧪 Тестирование LLM интеграции")
    print("=" * 50)

    # Проверяем доступные API ключи
    available_keys = []
    key_names = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Google",
        "GROK_API_KEY": "Grok",
        "TOGETHER_API_KEY": "Together AI"
    }

    for env_var, name in key_names.items():
        if os.getenv(env_var):
            available_keys.append((env_var, name))

    if not available_keys:
        print("❌ API ключи не найдены!")
        print("\n💡 Настройте API ключи в файле .env:")
        for env_var, name in key_names.items():
            print(f"   {env_var} - для {name}")
        print("\n📖 Подробные инструкции в .env.example")
        return

    print(f"✅ Найдено {len(available_keys)} API ключей:")
    for env_var, name in available_keys:
        print(f"   • {name}")

    # Тестируем первый доступный ключ
    env_var, provider_name = available_keys[0]
    print(f"\n🔄 Тестирование {provider_name}...")

    try:
        # Создаем клиента
        client = await create_llm_client(
            provider=env_var.replace("_API_KEY", "").lower(),
            api_key=os.getenv(env_var),
            temperature=0.7,
            max_tokens=500
        )

        # Тестовый запрос
        test_prompt = "Привет! Ты AI агент с мета-познанием. Кратко расскажи о своих возможностях на русском языке."

        print(f"📤 Отправка запроса: {test_prompt[:50]}...")

        response = await client.generate_response(
            prompt=test_prompt,
            system_message="Ты полезный AI ассистент. Отвечай кратко и по делу."
        )

        print("✅ Ответ получен!")
        print(f"🤖 Модель: {response['model']} ({response['provider']})")
        print(f"🎯 Уверенность: {response['confidence']}")
        print(f"⏱️ Время: {response['processing_time']:.2f} сек")
        print(f"💬 Ответ: {response['content']}")

        if 'usage' in response and response['usage']:
            usage = response['usage']
            print(f"📊 Использование токенов: {usage.get('total_tokens', 'N/A')}")

        # Закрываем соединение
        await client.__aexit__(None, None, None)

        print("\n🎉 LLM интеграция работает корректно!")

    except Exception as e:
        print(f"❌ Ошибка при тестировании {provider_name}: {e}")
        print("\n🔧 Возможные решения:")
        print("   • Проверьте правильность API ключа")
        print("   • Убедитесь в наличии интернет-соединения")
        print("   • Проверьте лимиты использования API")
        print("   • Попробуйте другой провайдер")


async def test_fallback_responses():
    """Тестирование fallback ответов"""
    print("\n🧪 Тестирование fallback ответов (без API ключей)")
    print("=" * 50)

    # Импортируем agent_core для тестирования fallback
    from agent.core.agent_core import AgentCore
    from agent.core.models import AgentConfig

    # Создаем агента без LLM
    config = AgentConfig(
        max_execution_time=30.0,
        confidence_threshold=0.5,
        enable_reasoning_trace=True,
        enable_memory=False,
        max_memory_entries=100,
        tool_timeout=10.0
    )

    agent = AgentCore(config)

    # Тестовые запросы
    test_queries = [
        "привет",
        "что ты умеешь",
        "как дела",
        "расскажи о python"
    ]

    print("Тестирование fallback ответов:")

    for query in test_queries:
        try:
            # Создаем тестовый запрос
            from agent.core.models import AgentRequest
            request = AgentRequest(
                id=f"test_{hash(query)}",
                query=query,
                user_id="test_user",
                session_id="test_session",
                timestamp=None
            )

            response = await agent._generate_fallback_response(request)
            print(f"   ❓ '{query}' → '{response[:50]}...'")

        except Exception as e:
            print(f"   ❌ Ошибка с запросом '{query}': {e}")

    print("\n✅ Fallback ответы работают!")


async def main():
    """Главная функция"""
    print("🚀 AI Agent LLM Integration Test")
    print("=" * 50)

    # Тестируем LLM интеграцию
    await test_llm_integration()

    # Тестируем fallback
    await test_fallback_responses()

    print("\n" + "=" * 50)
    print("✨ Тестирование завершено!")
    print("\n💡 Для запуска агента:")
    print("   python -m uvicorn api.main:app --reload")
    print("\n🌐 Frontend: http://localhost:3000")
    print("🔗 API: http://localhost:8000")


if __name__ == "__main__":
    asyncio.run(main())

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import asyncio
import time

print('Анализ причин задержек в веб-исследовании...')

async def analyze_delays():
    from web_research.web_research_manager import WebResearchManager

    manager = WebResearchManager()

    # Замер времени по этапам
    start_time = time.time()
    print(f'🚀 Начало исследования: {start_time}')

    # Этап 1: Инициализация
    init_start = time.time()
    initialized = await manager.initialize()
    init_time = time.time() - init_start
    print(f'📋 Инициализация: {init_time:.3f} сек')

    if not initialized:
        print('❌ Ошибка инициализации')
        return

    # Этап 2: Поиск
    search_start = time.time()
    result = await manager.research('test query', max_sources=3)
    search_time = time.time() - search_start
    print(f'🔍 Поиск и обработка: {search_time:.3f} сек')

    total_time = time.time() - start_time
    processing_time = result.get('processing_time', 0)

    print('\n📊 Итого:')
    print(f'  - Общее время: {total_time:.3f} сек')
    print(f'  - Время обработки (из результата): {processing_time:.3f} сек')
    print(f'  - Разница: {abs(total_time - processing_time):.3f} сек')
    print(f'  - Источников найдено: {len(result.get("sources", []))}')

    # Анализ возможных причин задержек
    if total_time > 0.5:
        print('\n🔍 Возможные причины задержек:')
        if total_time > 1.0:
            print('  - Возможно, остались asyncio.sleep в коде')
        if len(result.get('sources', [])) > 10:
            print('  - Слишком много источников для обработки')
        if processing_time < 0.01:
            print('  - Основная задержка вне измеряемого кода')

    return result

if __name__ == '__main__':
    # Запуск анализа
    result = asyncio.run(analyze_delays())
    print('\n✅ Анализ завершен!')

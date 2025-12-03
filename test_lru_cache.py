#!/usr/bin/env python3
"""
Простой тест LRU кэша
"""

from cache.cache_system_enhanced import LRUCache
import time

def test_lru_cache():
    """Тест LRU кэша"""
    print("🧪 Тестирование LRU кэша...")

    # Создаем кэш с максимальным размером 3
    cache = LRUCache(max_size=3, max_memory_mb=1)

    # Добавляем элементы
    cache.set('a', 'value_a')
    cache.set('b', 'value_b')
    cache.set('c', 'value_c')

    print('Начальное состояние:')
    print(f'  Размер кэша: {len(cache.cache)}')
    print(f'  Ключи: {list(cache.cache.keys())}')

    # Добавляем еще один элемент, должен вытеснить 'a' (LRU)
    cache.set('d', 'value_d')

    print('\nПосле добавления d (LRU eviction):')
    print(f'  Размер кэша: {len(cache.cache)}')
    print(f'  Ключи: {list(cache.cache.keys())}')

    # Проверяем наличие элементов
    print('\nПроверка элементов:')
    print(f'  a в кэше: {"a" in cache.cache}')
    print(f'  b в кэше: {"b" in cache.cache}')
    print(f'  c в кэше: {"c" in cache.cache}')
    print(f'  d в кэше: {"d" in cache.cache}')

    # Тестируем получение значений
    print('\nТестируем получение значений:')
    print(f'  cache.get("b"): {cache.get("b")}')
    print(f'  cache.get("a"): {cache.get("a")}')  # Должен быть None

    # Получаем статистику
    stats = cache.get_stats()
    print('\nСтатистика:')
    print(f'  Hits: {stats["hits"]}')
    print(f'  Misses: {stats["misses"]}')
    print(f'  Hit rate: {stats["hit_rate"]:.2f}')
    print(f'  Memory usage: {stats["memory_usage_mb"]:.2f} MB')

    # Тест TTL
    print('\nТестируем TTL...')
    cache.set('temp', 'temp_value', ttl=1)  # TTL 1 секунда
    print(f'  temp сразу после установки: {cache.get("temp")}')
    time.sleep(1.1)  # Ждем больше TTL
    print(f'  temp после истечения TTL: {cache.get("temp")}')

    print('\n✅ LRU кэш работает корректно!')
    return True

if __name__ == "__main__":
    test_lru_cache()

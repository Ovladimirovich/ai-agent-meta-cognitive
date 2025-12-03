# 🚀 План рефакторинга RAG системы (Устранение Windows fatal exception)

## 📋 Обзор проблемы

**Текущая проблема:** Windows fatal exception из-за зависимостей ChromaDB + ONNX Runtime (sentence-transformers)

**Решение:** Легковесная RAG система без тяжелых зависимостей

## 🎯 Цели рефакторинга

1. **❌ Устранить Windows fatal exception**
2. **📦 Сделать зависимости легковесными**
3. **🛡️ Обеспечить graceful degradation**
4. **🔄 Сохранить функциональность RAG**
5. **✅ Гарантировать работу на всех платформах**

## 📁 Архитектура новой системы

```
rag/
├── lightweight_embeddings.py    # FastEmbed + TF-IDF fallback
├── sqlite_vector_store.py       # SQLite-based vector storage
├── minimal_rag.py              # Zero-dependency RAG
├── hybrid_rag.py               # Unified system
├── __init__.py                 # Exports
└── rag_manager.py              # Legacy (deprecated)
```

## 🔧 Компоненты системы

### 1. LightweightEmbeddingProvider
**Цель:** Замена sentence-transformers на легковесные эмбеддинги

**Варианты:**
- **FastEmbed** (рекомендуемый) - BAAI/bge-small-en-v1.5
- **TF-IDF fallback** - чистый Python, без зависимостей
- **Random embeddings** - для тестирования

**API:**
```python
class EmbeddingProvider(ABC):
    async def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    async def embed_query(self, text: str) -> List[float]: ...
    async def initialize(self) -> bool: ...
```

### 2. SQLiteVectorStore
**Цель:** Замена ChromaDB на встроенное хранилище

**Функции:**
- Хранение документов и эмбеддингов в SQLite
- Полнотекстовый поиск (FTS5)
- Векторный поиск (косинусная близость)
- Гибридный поиск (keyword + semantic)

**Преимущества:**
- ✅ Всегда доступен (SQLite встроен в Python)
- ✅ Нет внешних зависимостей
- ✅ ACID транзакции
- ✅ Полнотекстовый поиск

### 3. MinimalRAG
**Цель:** Zero-dependency fallback

**Функции:**
- In-memory inverted index
- Keyword-based search
- Простая токенизация
- Без внешних зависимостей

**Использование:** Когда ничего другого недоступно

### 4. HybridRAGSystem
**Цель:** Умная система выбора метода поиска

**Методы поиска:**
- `KEYWORD` - полнотекстовый поиск
- `SEMANTIC` - векторный поиск
- `HYBRID` - комбинация обоих

**Логика выбора:**
```python
if semantic_available and keyword_results_good:
    return HYBRID
elif semantic_available:
    return SEMANTIC
else:
    return KEYWORD
```

## 📦 Зависимости

### Обязательные (requirements.txt)
```
fastembed>=0.2.0          # Легковесные эмбеддинги
numpy>=1.21.0             # Векторные операции
sqlite3>=3.35.0           # Встроен в Python
```

### Опциональные (requirements-optional.txt)
```
sentence-transformers>=2.2.0  # Тяжелые эмбеддинги
chromadb>=0.4.0             # Векторная БД
onnxruntime>=1.15.0         # ONNX Runtime (проблемный)
```

## 🧪 План тестирования

### Unit тесты

#### 1. EmbeddingProvider тесты
```python
def test_fastembed_provider():
    provider = FastEmbedProvider()
    await provider.initialize()
    embeddings = await provider.embed_documents(["test text"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0

def test_tfidf_fallback():
    provider = TFIDFProvider()
    await provider.initialize()
    embeddings = await provider.embed_documents(["test text"])
    assert len(embeddings) == 1
```

#### 2. SQLiteVectorStore тесты
```python
def test_sqlite_store():
    store = SQLiteVectorStore(":memory:")
    await store.initialize()

    docs = [{"text": "test document", "metadata": {"id": "1"}}]
    embeddings = [[0.1, 0.2, 0.3]]
    await store.add_documents(docs, embeddings)

    results = await store.search_semantic([0.1, 0.2, 0.3])
    assert len(results) == 1
    assert results[0]["content"] == "test document"
```

#### 3. HybridRAG тесты
```python
def test_hybrid_search():
    rag = HybridRAGSystem(":memory:")
    await rag.initialize()

    await rag.add_documents([
        {"text": "Python programming language", "metadata": {"topic": "programming"}},
        {"text": "Machine learning algorithms", "metadata": {"topic": "AI"}}
    ])

    results = await rag.search("programming AI", method=SearchMethod.HYBRID)
    assert len(results) >= 1
```

### Integration тесты

#### 1. Graceful degradation
```python
def test_graceful_degradation():
    # Тестируем работу без зависимостей
    rag = HybridRAGSystem(":memory:")
    await rag.initialize()

    # Должен работать даже если FastEmbed недоступен
    results = await rag.search("test query")
    assert isinstance(results, list)
```

#### 2. Cross-platform compatibility
```python
def test_windows_compatibility():
    # Специфические тесты для Windows
    import platform
    if platform.system() == "Windows":
        # Убеждаемся что нет импорта проблемных библиотек
        pass
```

### Performance тесты

#### 1. Benchmark сравнение
```python
def test_performance_comparison():
    # Сравнение скорости разных методов
    texts = ["test document"] * 100

    # FastEmbed
    start = time.time()
    embeddings = await fastembed_provider.embed_documents(texts)
    fastembed_time = time.time() - start

    # TF-IDF
    start = time.time()
    embeddings = await tfidf_provider.embed_documents(texts)
    tfidf_time = time.time() - start

    assert fastembed_time < tfidf_time  # FastEmbed должен быть быстрее
```

#### 2. Memory usage
```python
def test_memory_usage():
    # Тестируем потребление памяти
    import psutil
    process = psutil.Process()

    initial_memory = process.memory_info().rss

    rag = HybridRAGSystem(":memory:")
    await rag.initialize()

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    assert memory_increase < 100 * 1024 * 1024  # < 100MB
```

## 🚀 План реализации

### Фаза 1: Базовые компоненты (1-2 дня)
- [ ] Создать `lightweight_embeddings.py`
- [ ] Создать `sqlite_vector_store.py`
- [ ] Создать `minimal_rag.py`
- [ ] Написать базовые unit тесты

### Фаза 2: Гибридная система (1 день)
- [ ] Создать `hybrid_rag.py`
- [ ] Реализовать логику выбора метода
- [ ] Интеграционные тесты

### Фаза 3: Интеграция и оптимизация (1-2 дня)
- [ ] Заменить `rag_manager.py` на новую систему
- [ ] Обновить `requirements.txt`
- [ ] Создать `requirements-optional.txt`
- [ ] Performance тесты

### Фаза 4: Финализация (1 день)
- [ ] Cross-platform тестирование
- [ ] Документация
- [ ] Cleanup legacy кода

## 📊 Критерии успеха

### Функциональные
- [ ] ✅ Нет Windows fatal exception
- [ ] ✅ RAG работает без ChromaDB
- [ ] ✅ RAG работает без sentence-transformers
- [ ] ✅ Все тесты проходят

### Производительность
- [ ] ✅ Время инициализации < 5 сек
- [ ] ✅ Поиск < 1 сек для 1000 документов
- [ ] ✅ Потребление памяти < 200MB

### Качество кода
- [ ] ✅ 90%+ покрытие тестами
- [ ] ✅ Type hints везде
- [ ] ✅ Документация функций

## 🎯 Риски и mitigation

### Риски
1. **Производительность TF-IDF fallback** - может быть медленнее
2. **Качество эмбеддингов FastEmbed** - может быть хуже sentence-transformers
3. **Совместимость SQLite** - разные версии Python

### Mitigation
1. **Оптимизация TF-IDF** - кэширование, индексы
2. **Выбор модели** - тестирование разных моделей FastEmbed
3. **Feature detection** - проверка возможностей SQLite

## 📈 Мониторинг успеха

### Метрики
- **Время сборки зависимостей** - должно уменьшиться на 80%
- **Размер Docker образа** - уменьшение на 60%
- **Время запуска** - уменьшение на 70%
- **Успешность тестов** - поддержание 95%+

### Alerts
- Windows fatal exception в CI/CD
- Падение производительности > 20%
- Рост потребления памяти > 50%

## 🎉 Результат

**До:** Windows fatal exception, тяжелые зависимости, ненадежная система
**После:** Стабильная кроссплатформенная RAG система с graceful degradation

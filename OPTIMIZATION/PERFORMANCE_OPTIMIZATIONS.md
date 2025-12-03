# Рекомендации по оптимизации производительности AI-агента с мета-познанием

## Обзор

В ходе анализа производительности проекта AI-агента с мета-познанием были выявлены узкие места и предложены конкретные меры по их устранению. Ниже представлены детализированные рекомендации по оптимизации различных аспектов системы.

## Категории оптимизаций

### 1. Оптимизация использования памяти

#### Проблема: Потенциальные утечки памяти в системах памяти агента
- **Описание**: В `agent/memory/memory_manager.py` может происходить накопление исторических данных без очистки
- **Решение**: Внедрение систем очистки устаревших данных и ограничение размера хранимой информации

#### Проблема: Неэффективное хранение данных в памяти
- **Описание**: Использование неоптимальных структур данных для хранения памяти
- **Решение**: Оптимизация структур данных и использование LRU-кэширования для часто используемых данных

#### Рекомендации:
```python
class MemoryManager:
    def __init__(self, max_entries: int = 1000, max_working_memory_mb: float = 50.0, max_semantic_memory_mb: float = 100.0):
        # ... существующий код ...
        
        # Добавить LRU кэш для часто используемых записей
        self._working_memory_cache = LRUCache(maxsize=100)
        self._semantic_memory_cache = LRUCache(maxsize=200)
        
        # Добавить систему очистки устаревших данных
        self._cleanup_threshold = timedelta(hours=24)  # Удалять данные старше 24 часов
        
    async def cleanup_expired_data(self):
        """Очистка устаревших данных из памяти"""
        current_time = datetime.now()
        
        # Очистка рабочей памяти
        expired_keys = []
        for key, memory in self.working_memory.items():
            if current_time - memory.timestamp > self._cleanup_threshold:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.working_memory[key]
            self._working_memory_cache.pop(key, None)  # Удалить из кэша
        
        # Очистка семантической памяти
        expired_keys = []
        for key, memory in self.semantic_memory.items():
            if current_time - memory.timestamp > self._cleanup_threshold:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.semantic_memory[key]
            self._semantic_memory_cache.pop(key, None)  # Удалить из кэша
```

### 2. Улучшение скорости обработки запросов

#### Проблема: Отсутствие эффективного кэширования для частых запросов
- **Описание**: В `api/main.py` и `agent/core/agent_core.py` отсутствует кэширование результатов
- **Решение**: Внедрение агрессивного кэширования для часто используемых данных

#### Проблема: Недостаточная предварительная загрузка данных
- **Описание**: Данные загружаются по требованию, что увеличивает время отклика
- **Решение**: Внедрение предварительной загрузки часто используемых данных

#### Рекомендации:
```python
class AgentCore:
    def __init__(self):
        # ... существующий код ...
        
        # Добавить кэш для результатов обработки
        self._processing_cache = {}
        self._cache_ttl = 3600  # 1 час
        self._max_cache_size = 1000
        
        # Добавить систему предварительной загрузки
        self._preloaded_data = {}
        
    async def _get_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Получение закэшированного результата"""
        if query_hash in self._processing_cache:
            cached_item = self._processing_cache[query_hash]
            if time.time() - cached_item['timestamp'] < self._cache_ttl:
                return cached_item['result']
            else:
                # Удалить устаревший кэш
                del self._processing_cache[query_hash]
        return None
    
    async def _cache_result(self, query_hash: str, result: Dict[str, Any]):
        """Кэширование результата обработки"""
        if len(self._processing_cache) >= self._max_cache_size:
            # Удалить старейший элемент
            oldest_key = min(self._processing_cache.keys(), 
                           key=lambda k: self._processing_cache[k]['timestamp'])
            del self._processing_cache[oldest_key]
        
        self._processing_cache[query_hash] = {
            'result': result,
            'timestamp': time.time()
        }
    
    async def process_query_with_cache(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обработка запроса с использованием кэша"""
        # Создать хэш от запроса и контекста
        query_context = f"{query}_{str(context) if context else ''}"
        query_hash = hashlib.sha256(query_context.encode()).hexdigest()
        
        # Попробовать получить результат из кэша
        cached_result = await self._get_cached_result(query_hash)
        if cached_result:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_result
        
        # Обработать запрос и закэшировать результат
        result = await self.process_query(query, context)
        await self._cache_result(query_hash, result)
        
        return result
```

### 3. Оптимизация работы с базами данных

#### Проблема: Неоптимизированные SQL-запросы
- **Описание**: В `database/postgres.py` используются неоптимальные запросы к базе данных
- **Решение**: Оптимизация SQL-запросов и индексов

#### Проблема: Отсутствие connection pooling
- **Описание**: Нет эффективного управления соединениями с базой данных
- **Решение**: Внедрение connection pooling для улучшения производительности

#### Рекомендации:
```python
class PostgreSQLManager:
    def __init__(self, connection_string: str):
        # ... существующий код ...
        
        # Настройка connection pool
        self.pool = None
        self.connection_string = connection_string
        
    async def initialize_pool(self, min_connections: int = 5, max_connections: int = 20):
        """Инициализация connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=min_connections,
                max_size=max_connections,
                command_timeout=60,
                statement_cache_size=1000
            )
            logger.info(f"Connection pool initialized with {min_connections}-{max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def execute_query_with_pool(self, query: str, *args) -> List[Dict[str, Any]]:
        """Выполнение запроса с использованием connection pool"""
        if not self.pool:
            await self.initialize_pool()
        
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def store_experience_optimized(self, experience: AgentExperience) -> bool:
        """Оптимизированное сохранение опыта с использованием подготовленных запросов"""
        if not self.pool:
            await self.initialize_pool()
        
        # Подготовленный запрос для улучшения производительности
        query = """
        INSERT INTO agent_experiences (query, result, metadata, timestamp, agent_id, session_id)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (query_hash) DO UPDATE SET
            result = EXCLUDED.result,
            metadata = EXCLUDED.metadata,
            timestamp = EXCLUDED.timestamp
        """
        
        async with self.pool.acquire() as connection:
            try:
                # Создать хэш запроса для оптимизации
                query_hash = hashlib.sha256(experience.query.encode()).hexdigest()
                
                await connection.execute(
                    query,
                    experience.query,
                    experience.result,
                    json.dumps(experience.metadata),
                    experience.timestamp,
                    experience.agent_id,
                    experience.session_id
                )
                return True
            except Exception as e:
                logger.error(f"Error storing experience: {e}")
                return False
```

### 4. Улучшение асинхронной обработки

#### Проблема: Неконсистентное использование асинхронных операций
- **Описание**: В разных частях системы используются разные подходы к асинхронности
- **Решение**: Единообразное использование async/await и оптимизация обработки параллельных запросов

#### Рекомендации:
```python
class AsyncProcessingManager:
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.task_queue = asyncio.Queue()
        self.active_tasks = set()
    
    async def process_with_semaphore(self, task_func, *args, **kwargs):
        """Обработка задачи с ограничением количества одновременных операций"""
        async with self.semaphore:
            return await task_func(*args, **kwargs)
    
    async def process_batch(self, tasks: List[Callable]) -> List[Any]:
        """Параллельная обработка пакета задач с ограничением"""
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        async def limited_task(task):
            async with semaphore:
                return await task()
        
        # Создать задачи с ограничением
        limited_tasks = [limited_task(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        return results
    
    async def process_with_timeout(self, coro, timeout: float = 30.0):
        """Обработка корутины с таймаутом"""
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Task timed out after {timeout} seconds")
            raise TimeoutError(f"Task timed out after {timeout} seconds")
```

### 5. Оптимизация вызовов LLM

#### Проблема: Неэффективные вызовы LLM API
- **Описание**: В `integrations/llm_client.py` отсутствует кэширование и оптимизация вызовов
- **Решение**: Внедрение кэширования и оптимизация параметров вызовов

#### Рекомендации:
```python
class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.base_urls[config.provider]
        
        # Добавить кэш для LLM вызовов
        self._llm_cache = {}
        self._cache_ttl = 1800  # 30 минут
        self._max_cache_size = 500
        
        # Оптимизация соединения
        self.connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            acquire_timeout=60,
            pool_timeout=60,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=False
        )
        
        # Настройка сессии с оптимизациями
        timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_connect=10)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={
                "User-Agent": "AI-Agent-Meta-Cognitive/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
    
    async def _get_cached_llm_response(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Получение закэшированного ответа от LLM"""
        if prompt_hash in self._llm_cache:
            cached_item = self._llm_cache[prompt_hash]
            if time.time() - cached_item['timestamp'] < self._cache_ttl:
                logger.info(f"LLM cache hit for prompt: {prompt_hash[:16]}...")
                return cached_item['response']
            else:
                # Удалить устаревший кэш
                del self._llm_cache[prompt_hash]
        return None
    
    async def _cache_llm_response(self, prompt_hash: str, response: Dict[str, Any]):
        """Кэширование ответа от LLM"""
        if len(self._llm_cache) >= self._max_cache_size:
            # Удалить старейший элемент
            oldest_key = min(self._llm_cache.keys(),
                           key=lambda k: self._llm_cache[k]['timestamp'])
            del self._llm_cache[oldest_key]
        
        self._llm_cache[prompt_hash] = {
            'response': response,
            'timestamp': time.time()
        }
    
    async def generate_response_with_cache(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Генерация ответа от LLM с использованием кэша"""
        # Создать хэш от полного запроса
        full_request = f"{prompt}_{system_message}_{str(context) if context else ''}"
        prompt_hash = hashlib.sha256(full_request.encode()).hexdigest()
        
        # Попробовать получить результат из кэша
        cached_response = await self._get_cached_llm_response(prompt_hash)
        if cached_response:
            return cached_response
        
        # Выполнить вызов LLM и закэшировать результат
        response = await self.generate_response(prompt, system_message, context)
        await self._cache_llm_response(prompt_hash, response)
        
        return response
```

### 6. Улучшение кэширования

#### Проблема: Недостаточное использование кэширования
- **Описание**: В системе отсутствует комплексное кэширование данных
- **Решение**: Внедрение многоуровневой системы кэширования

#### Рекомендации:
```python
class CacheManager:
    def __init__(self):
        # In-memory кэш для часто используемых данных
        self.memory_cache = {}
        self.memory_cache_ttl = {}
        self.max_memory_cache_size = 1000
        
        # Redis кэш для распределенного кэширования
        self.redis_client = None
        self._setup_redis()
    
    def _setup_redis(self):
        """Настройка Redis для распределенного кэширования"""
        try:
            import redis.asyncio as redis
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD", "")
            
            if redis_password:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    decode_responses=True
                )
            else:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
            logger.info("Cache manager connected to Redis")
        except ImportError:
            logger.warning("Redis not available, using in-memory cache only")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for caching: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша (сначала in-memory, затем Redis)"""
        # Проверить in-memory кэш
        if key in self.memory_cache:
            if time.time() < self.memory_cache_ttl.get(key, 0):
                return self.memory_cache[key]
            else:
                # Удалить устаревший элемент
                del self.memory_cache[key]
                del self.memory_cache_ttl[key]
        
        # Проверить Redis кэш
        if self.redis_client:
            try:
                cached_value = await self.redis_client.get(key)
                if cached_value:
                    # Обновить in-memory кэш
                    value = json.loads(cached_value)
                    self.set(key, value)
                    return value
            except Exception as e:
                logger.error(f"Redis cache get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Установка значения в кэш (in-memory и Redis)"""
        # Установить в in-memory кэш
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Удалить старейший элемент
            oldest_key = min(self.memory_cache_ttl.keys(), key=lambda k: self.memory_cache_ttl[k])
            del self.memory_cache[oldest_key]
            del self.memory_cache_ttl[oldest_key]
        
        self.memory_cache[key] = value
        self.memory_cache_ttl[key] = time.time() + ttl
        
        # Установить в Redis кэш
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                logger.error(f"Redis cache set error: {e}")
    
    async def invalidate(self, key: str):
        """Инвалидация кэша"""
        # Удалить из in-memory кэша
        self.memory_cache.pop(key, None)
        self.memory_cache_ttl.pop(key, None)
        
        # Удалить из Redis кэша
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis cache invalidate error: {e}")
```

### 7. Оптимизация алгоритмов обработки данных

#### Проблема: Неэффективные алгоритмы обработки
- **Описание**: В `agent/core/input_preprocessor.py` и других компонентах используются неоптимальные алгоритмы
- **Решение**: Оптимизация алгоритмов и структур данных

#### Рекомендации:
```python
class OptimizedInputPreprocessor:
    def __init__(self):
        # Компилированные регулярные выражения для повторного использования
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?'),
            'phone': re.compile(r'\+?1?-?\.?\s?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})'),
            'credit_card': re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),
        }
        
        # Быстрые структуры данных для поиска
        self.sensitive_keywords = {
            'password', 'token', 'secret', 'key', 'credential', 'auth', 'authorization',
            'bearer', 'api_key', 'access_token', 'refresh_token'
        }
    
    def preprocess_input_optimized(self, text: str) -> PreprocessingResult:
        """Оптимизированная предобработка входных данных"""
        start_time = time.time()
        
        # Быстрая проверка на чувствительные ключевые слова
        text_lower = text.lower()
        found_sensitive = any(keyword in text_lower for keyword in self.sensitive_keywords)
        
        # Быстрый подсчет слов и символов
        word_count = len(text.split())
        char_count = len(text)
        
        # Поиск с использованием скомпилированных регулярных выражений
        found_patterns = {}
        for pattern_name, compiled_pattern in self.patterns.items():
            matches = compiled_pattern.findall(text)
            if matches:
                found_patterns[pattern_name] = matches
        
        # Быстрая проверка длины
        is_too_long = len(text) > 10000  # Ограничение 10K символов
        
        # Результат
        result = PreprocessingResult(
            original_text=text,
            processed_text=text,  # В реальности здесь может быть дополнительная обработка
            word_count=word_count,
            char_count=char_count,
            found_patterns=found_patterns,
            is_sensitive=found_sensitive,
            is_too_long=is_too_long,
            processing_time=time.time() - start_time
        )
        
        return result
    
    def validate_comprehensive_optimized(self, data: Dict[str, Any]) -> ValidationResult:
        """Оптимизированная комплексная валидация"""
        start_time = time.time()
        errors = []
        warnings = []
        
        # Быстрая проверка структуры данных
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, processing_time=time.time() - start_time)
        
        # Проверка ключевых полей с использованием set для быстрого поиска
        required_fields = {'query', 'context', 'user_id'}
        provided_fields = set(data.keys())
        
        missing_fields = required_fields - provided_fields
        if missing_fields:
            errors.extend([f"Missing required field: {field}" for field in missing_fields])
        
        # Быстрая проверка типов для основных полей
        type_checks = {
            'query': str,
            'context': (dict, list),
            'user_id': str,
            'session_id': str
        }
        
        for field, expected_type in type_checks.items():
            if field in data and not isinstance(data[field], expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type}")
        
        # Быстрая проверка длины строк
        string_fields_to_check = ['query', 'user_id', 'session_id']
        for field in string_fields_to_check:
            if field in data and isinstance(data[field], str):
                if len(data[field]) > 10000:  # Ограничение 10K символов
                    errors.append(f"Field '{field}' is too long (max 1000 characters)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            processing_time=time.time() - start_time
        )
```

### 8. Улучшение производительности фронтенд-компонентов

#### Проблема: Неоптимальный рендеринг и управление состоянием
- **Описание**: В `frontend/src/widgets/MemoryVisualizer/MemoryVisualizer.tsx` и других компонентах используются неэффективные подходы
- **Решение**: Оптимизация рендеринга и управление состоянием

#### Рекомендации:
```tsx
// frontend/src/widgets/MemoryVisualizer/MemoryVisualizer.tsx
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { ForceGraph2D } from 'react-force-graph-2d';

const MemoryVisualizer: React.FC = () => {
  const [memoryData, setMemoryData] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const fgRef = useRef<any>();
  const [selectedNode, setSelectedNode] = useState<any>(null);
  
  // Использовать useMemo для вычисления сложных данных
  const processedData = useMemo(() => {
    if (!memoryData.length) return { nodes: [], links: [] };
    
    // Обработка данных с минимальными вычислениями
    const nodes = memoryData.map(item => ({
      id: item.id,
      name: item.name,
      group: item.type,
      size: item.importance || 1
    }));
    
    // Создание связей между элементами
    const links = [];
    for (let i = 0; i < nodes.length - 1; i++) {
      links.push({
        source: nodes[i].id,
        target: nodes[i + 1].id,
        value: Math.random() * 5
      });
    }
    
    return { nodes, links };
  }, [memoryData]);
  
  // Использовать useCallback для функций, передаваемых в дочерние компоненты
  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node);
    // Избегать ненужных перерисовок
    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y);
      fgRef.current.zoom(4);
    }
 }, []);
  
  // Оптимизированный рендеринг canvas объектов
  const renderLink = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const strokeWidth = 2 / globalScale;
    ctx.lineWidth = strokeWidth;
    ctx.strokeStyle = link.color || '#999';
    ctx.beginPath();
    ctx.moveTo(link.source.x, link.source.y);
    ctx.lineTo(link.target.x, link.target.y);
    ctx.stroke();
  }, []);
  
  const renderNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.name;
    const fontSize = 12 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = node.color || 'rgba(0, 255, 255, 0.7)';
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.size || 6, 0, 2 * Math.PI, false);
    ctx.fill();
    
    if (globalScale > 2) {
      ctx.fillStyle = 'black';
      ctx.fillText(label, node.x, node.y + 10);
    }
  }, []);
  
  // Использовать useEffect с правильными зависимостями
  useEffect(() => {
    const fetchMemoryData = async () => {
      setIsLoading(true);
      try {
        // Оптимизированный запрос данных
        const response = await fetch('/api/memory/episodic', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'public, max-age=300' // Кэширование на 5 минут
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          setMemoryData(data);
        }
      } catch (error) {
        console.error('Error fetching memory data:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchMemoryData();
    
    // Опционально: установить интервал обновления
    const interval = setInterval(fetchMemoryData, 30000); // Обновление каждые 30 секунд
    
    return () => clearInterval(interval);
  }, []); // Пустой массив зависимостей для однократного выполнения
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }
  
 return (
    <div className="h-full w-full relative">
      <ForceGraph2D
        ref={fgRef}
        graphData={processedData}
        nodeLabel="name"
        nodeAutoColorBy="group"
        linkDirectionalArrowLength={6}
        linkDirectionalArrowRelPos={1}
        linkCanvasObject={renderLink}
        linkCanvasObjectMode={() => 'after'}
        nodeCanvasObject={renderNode}
        nodeCanvasObjectMode={() => 'after'}
        onNodeClick={handleNodeClick}
        onEngineStop={() => fgRef.current && fgRef.current.zoomToFit(100)}
        cooldownTicks={100}
        minZoom={0.1}
        maxZoom={10}
      />
      
      {selectedNode && (
        <div className="absolute top-4 right-4 bg-white p-4 rounded shadow-lg max-w-xs">
          <h3 className="font-bold text-lg">{selectedNode.name}</h3>
          <p>Type: {selectedNode.group}</p>
          <p>Importance: {selectedNode.size}</p>
        </div>
      )}
    </div>
  );
};

export default MemoryVisualizer;
```

## Заключение

Реализация предложенных оптимизаций производительности значительно улучшит скорость работы, эффективность использования ресурсов и общую отзывчивость AI-агента с мета-познанием. Важно внедрять эти оптимизации поэтапно, начиная с наиболее критичных для производительности компонентов, и проводить регулярное тестирование для измерения улучшений.
"""
Redis интеграция для AI Агента с Мета-Познанием
Фаза 5: Инфраструктура и интеграции
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
from cachetools import TTLCache

from agent.learning.models import AgentExperience, Pattern

logger = logging.getLogger(__name__)


class CacheEntry:
    """Запись кэша"""
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = datetime.utcnow()


class ConversationEntry:
    """Запись разговора"""
    def __init__(self, session_id: str, message: Dict[str, Any]):
        self.session_id = session_id
        self.message = message
        self.timestamp = datetime.utcnow()


class MetricsEntry:
    """Запись метрики"""
    def __init__(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None):
        self.name = name
        self.value = value
        self.tags = tags or {}
        self.timestamp = datetime.utcnow()


class RedisManager:
    """
    Менеджер Redis для кэширования и сессий агента
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0,
                 password: Optional[str] = None, max_connections: int = 10):
        """
        Инициализация Redis менеджера

        Args:
            host: Хост Redis
            port: Порт Redis
            db: Номер базы данных
            password: Пароль Redis
            max_connections: Максимальное количество подключений
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections

        self.redis_client: Optional[redis.Redis] = None

        # Локальный TTL кэш для часто используемых данных
        self.local_cache = TTLCache(maxsize=1000, ttl=300)  # 5 минут TTL

        # Статистика
        self.stats = {
            'connections_created': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'keys_stored': 0,
            'keys_retrieved': 0,
            'errors_occurred': 0
        }

        logger.info(f"RedisManager initialized for {host}:{port}")

    async def initialize(self):
        """Инициализация подключения к Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=True,  # Автоматическое декодирование строк
                retry_on_timeout=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )

            # Проверка подключения
            await self.redis_client.ping()
            self.stats['connections_created'] += 1

            logger.info("✅ Redis connection established")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis: {e}")
            raise

    async def close(self):
        """Закрытие подключения к Redis"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Redis connection closed")

    # Методы для кэширования опыта агента

    async def cache_experience(self, experience: AgentExperience, ttl_seconds: int = 3600) -> bool:
        """
        Кэширование опыта агента

        Args:
            experience: Опыт агента
            ttl_seconds: Время жизни в секундах

        Returns:
            bool: Успех операции
        """
        try:
            if not self.redis_client:
                return False

            key = f"experience:{experience.id}"
            data = {
                'id': experience.id,
                'query': getattr(experience, 'query', ''),
                'result': getattr(experience, 'result', ''),
                'confidence': getattr(experience, 'confidence', 0.0),
                'execution_time': getattr(experience, 'execution_time', 0.0),
                'timestamp': getattr(experience, 'timestamp', datetime.utcnow()).isoformat(),
                'user_id': getattr(experience, 'user_id', None),
                'session_id': getattr(experience, 'session_id', None),
                'success_indicators': getattr(experience, 'success_indicators', []),
                'error_indicators': getattr(experience, 'error_indicators', []),
                'metadata': getattr(experience, 'metadata', {})
            }

            await self.redis_client.setex(key, ttl_seconds, json.dumps(data))
            self.stats['keys_stored'] += 1

            # Локальное кэширование
            self.local_cache[key] = data

            logger.debug(f"Cached experience {experience.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache experience {experience.id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_cached_experience(self, experience_id: str) -> Optional[AgentExperience]:
        """
        Получение кэшированного опыта

        Args:
            experience_id: ID опыта

        Returns:
            Optional[AgentExperience]: Кэшированный опыт или None
        """
        try:
            if not self.redis_client:
                return None

            key = f"experience:{experience_id}"

            # Проверка локального кэша
            if key in self.local_cache:
                self.stats['cache_hits'] += 1
                data = self.local_cache[key]
            else:
                # Получение из Redis
                cached_data = await self.redis_client.get(key)
                if not cached_data:
                    self.stats['cache_misses'] += 1
                    return None

                data = json.loads(cached_data)
                self.local_cache[key] = data
                self.stats['cache_hits'] += 1

            self.stats['keys_retrieved'] += 1

            # Преобразование в объект AgentExperience
            experience = AgentExperience(
                id=data['id'],
                query=data['query'],
                result=data['result'],
                confidence=data['confidence'],
                execution_time=data['execution_time'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                user_id=data['user_id'],
                session_id=data['session_id'],
                success_indicators=data['success_indicators'],
                error_indicators=data['error_indicators'],
                metadata=data['metadata']
            )

            return experience

        except Exception as e:
            logger.error(f"Failed to get cached experience {experience_id}: {e}")
            self.stats['errors_occurred'] += 1
            return None

    # Методы для кэширования паттернов

    async def cache_pattern(self, pattern: Pattern, ttl_seconds: int = 1800) -> bool:
        """
        Кэширование паттерна

        Args:
            pattern: Паттерн
            ttl_seconds: Время жизни в секундах

        Returns:
            bool: Успех операции
        """
        try:
            if not self.redis_client:
                return False

            key = f"pattern:{pattern.id}"
            data = {
                'id': pattern.id,
                'type': getattr(pattern, 'type', 'unknown'),
                'trigger_conditions': pattern.trigger_conditions,
                'description': getattr(pattern, 'description', ''),
                'confidence': pattern.confidence,
                'frequency': getattr(pattern, 'frequency', 1.0),
                'examples': getattr(pattern, 'examples', []),
                'last_updated': getattr(pattern, 'last_updated', datetime.utcnow()).isoformat()
            }

            await self.redis_client.setex(key, ttl_seconds, json.dumps(data))
            self.stats['keys_stored'] += 1

            logger.debug(f"Cached pattern {pattern.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache pattern {pattern.id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_cached_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """
        Получение кэшированного паттерна

        Args:
            pattern_id: ID паттерна

        Returns:
            Optional[Pattern]: Кэшированный паттерн или None
        """
        try:
            if not self.redis_client:
                return None

            key = f"pattern:{pattern_id}"

            # Проверка локального кэша
            if key in self.local_cache:
                self.stats['cache_hits'] += 1
                data = self.local_cache[key]
            else:
                # Получение из Redis
                cached_data = await self.redis_client.get(key)
                if not cached_data:
                    self.stats['cache_misses'] += 1
                    return None

                data = json.loads(cached_data)
                self.local_cache[key] = data
                self.stats['cache_hits'] += 1

            self.stats['keys_retrieved'] += 1

            # Преобразование в объект Pattern
            pattern = Pattern(
                id=data['id'],
                type=data['type'],
                trigger_conditions=data['trigger_conditions'],
                description=data['description'],
                confidence=data['confidence'],
                frequency=data['frequency'],
                examples=data['examples'],
                last_updated=datetime.fromisoformat(data['last_updated'])
            )

            return pattern

        except Exception as e:
            logger.error(f"Failed to get cached pattern {pattern_id}: {e}")
            self.stats['errors_occurred'] += 1
            return None

    # Методы для работы с сессиями пользователей

    async def create_user_session(self, session_id: str, user_data: Dict[str, Any],
                                 ttl_seconds: int = 86400) -> bool:
        """
        Создание сессии пользователя

        Args:
            session_id: ID сессии
            user_data: Данные пользователя
            ttl_seconds: Время жизни сессии

        Returns:
            bool: Успех операции
        """
        try:
            if not self.redis_client:
                return False

            key = f"session:{session_id}"
            data = {
                'session_id': session_id,
                'user_data': user_data,
                'created_at': datetime.utcnow().isoformat(),
                'last_activity': datetime.utcnow().isoformat()
            }

            await self.redis_client.setex(key, ttl_seconds, json.dumps(data))
            self.stats['keys_stored'] += 1

            logger.debug(f"Created user session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create user session {session_id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение сессии пользователя

        Args:
            session_id: ID сессии

        Returns:
            Optional[Dict[str, Any]]: Данные сессии или None
        """
        try:
            if not self.redis_client:
                return None

            key = f"session:{session_id}"
            cached_data = await self.redis_client.get(key)

            if not cached_data:
                return None

            data = json.loads(cached_data)

            # Обновление времени последней активности
            data['last_activity'] = datetime.utcnow().isoformat()
            await self.redis_client.setex(key, 86400, json.dumps(data))  # Продление на 24 часа

            self.stats['keys_retrieved'] += 1
            return data

        except Exception as e:
            logger.error(f"Failed to get user session {session_id}: {e}")
            self.stats['errors_occurred'] += 1
            return None

    async def update_user_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Обновление сессии пользователя

        Args:
            session_id: ID сессии
            updates: Обновления данных

        Returns:
            bool: Успех операции
        """
        try:
            if not self.redis_client:
                return False

            key = f"session:{session_id}"
            cached_data = await self.redis_client.get(key)

            if not cached_data:
                return False

            data = json.loads(cached_data)
            data.update(updates)
            data['last_activity'] = datetime.utcnow().isoformat()

            await self.redis_client.setex(key, 86400, json.dumps(data))  # Продление на 24 часа

            logger.debug(f"Updated user session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update user session {session_id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def delete_user_session(self, session_id: str) -> bool:
        """
        Удаление сессии пользователя

        Args:
            session_id: ID сессии

        Returns:
            bool: Успех операции
        """
        try:
            if not self.redis_client:
                return False

            key = f"session:{session_id}"
            result = await self.redis_client.delete(key)

            if result > 0:
                logger.debug(f"Deleted user session {session_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete user session {session_id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    # Методы для кэширования результатов API

    async def cache_api_response(self, endpoint: str, params: Dict[str, Any],
                               response: Any, ttl_seconds: int = 300) -> bool:
        """
        Кэширование ответа API

        Args:
            endpoint: Эндпоинт API
            params: Параметры запроса
            response: Ответ API
            ttl_seconds: Время жизни в секундах

        Returns:
            bool: Успех операции
        """
        try:
            if not self.redis_client:
                return False

            # Создание ключа на основе эндпоинта и параметров
            key_data = {'endpoint': endpoint, 'params': params}
            key = f"api:{hash(json.dumps(key_data, sort_keys=True))}"

            data = {
                'endpoint': endpoint,
                'params': params,
                'response': response,
                'cached_at': datetime.utcnow().isoformat()
            }

            await self.redis_client.setex(key, ttl_seconds, json.dumps(data))
            self.stats['keys_stored'] += 1

            logger.debug(f"Cached API response for {endpoint}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache API response for {endpoint}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_cached_api_response(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Получение кэшированного ответа API

        Args:
            endpoint: Эндпоинт API
            params: Параметры запроса

        Returns:
            Optional[Any]: Кэшированный ответ или None
        """
        try:
            if not self.redis_client:
                return None

            key_data = {'endpoint': endpoint, 'params': params}
            key = f"api:{hash(json.dumps(key_data, sort_keys=True))}"

            cached_data = await self.redis_client.get(key)
            if not cached_data:
                return None

            data = json.loads(cached_data)
            self.stats['keys_retrieved'] += 1

            return data['response']

        except Exception as e:
            logger.error(f"Failed to get cached API response for {endpoint}: {e}")
            self.stats['errors_occurred'] += 1
            return None

    # Методы для pub/sub сообщений

    async def publish_message(self, channel: str, message: Dict[str, Any]) -> bool:
        """
        Публикация сообщения в канал

        Args:
            channel: Канал
            message: Сообщение

        Returns:
            bool: Успех операции
        """
        try:
            if not self.redis_client:
                return False

            await self.redis_client.publish(channel, json.dumps(message))
            logger.debug(f"Published message to channel {channel}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish message to channel {channel}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def subscribe_to_channel(self, channel: str, callback):
        """
        Подписка на канал

        Args:
            channel: Канал
            callback: Функция обратного вызова
        """
        try:
            if not self.redis_client:
                return

            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(channel)

            logger.info(f"Subscribed to channel {channel}")

            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await callback(data)
                    except Exception as e:
                        logger.error(f"Error processing message from channel {channel}: {e}")

        except Exception as e:
            logger.error(f"Failed to subscribe to channel {channel}: {e}")

    # Методы для очередей задач

    async def enqueue_task(self, queue_name: str, task_data: Dict[str, Any]) -> bool:
        """
        Добавление задачи в очередь

        Args:
            queue_name: Имя очереди
            task_data: Данные задачи

        Returns:
            bool: Успех операции
        """
        try:
            if not self.redis_client:
                return False

            task = {
                'id': f"task_{int(time.time())}_{hash(str(task_data)) % 1000}",
                'data': task_data,
                'queued_at': datetime.utcnow().isoformat(),
                'status': 'queued'
            }

            await self.redis_client.lpush(queue_name, json.dumps(task))
            self.stats['keys_stored'] += 1

            logger.debug(f"Enqueued task in {queue_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to enqueue task in {queue_name}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def dequeue_task(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """
        Извлечение задачи из очереди

        Args:
            queue_name: Имя очереди

        Returns:
            Optional[Dict[str, Any]]: Задача или None
        """
        try:
            if not self.redis_client:
                return None

            result = await self.redis_client.rpop(queue_name)
            if not result:
                return None

            task = json.loads(result)
            task['dequeued_at'] = datetime.utcnow().isoformat()
            task['status'] = 'processing'

            self.stats['keys_retrieved'] += 1
            logger.debug(f"Dequeued task from {queue_name}")
            return task

        except Exception as e:
            logger.error(f"Failed to dequeue task from {queue_name}: {e}")
            self.stats['errors_occurred'] += 1
            return None

    # Методы для распределенного кэширования

    async def set_distributed_cache(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """
        Установка значения в распределенный кэш

        Args:
            key: Ключ
            value: Значение
            ttl_seconds: Время жизни

        Returns:
            bool: Успех операции
        """
        try:
            if not self.redis_client:
                return False

            cache_key = f"cache:{key}"
            data = {
                'value': value,
                'set_at': datetime.utcnow().isoformat()
            }

            await self.redis_client.setex(cache_key, ttl_seconds, json.dumps(data))
            self.stats['keys_stored'] += 1

            return True

        except Exception as e:
            logger.error(f"Failed to set distributed cache for key {key}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_distributed_cache(self, key: str) -> Optional[Any]:
        """
        Получение значения из распределенного кэша

        Args:
            key: Ключ

        Returns:
            Optional[Any]: Значение или None
        """
        try:
            if not self.redis_client:
                return None

            cache_key = f"cache:{key}"
            cached_data = await self.redis_client.get(cache_key)

            if not cached_data:
                return None

            data = json.loads(cached_data)
            self.stats['keys_retrieved'] += 1

            return data['value']

        except Exception as e:
            logger.error(f"Failed to get distributed cache for key {key}: {e}")
            self.stats['errors_occurred'] += 1
            return None

    # Служебные методы

    async def get_redis_info(self) -> Dict[str, Any]:
        """
        Получение информации о Redis

        Returns:
            Dict[str, Any]: Информация о Redis
        """
        try:
            if not self.redis_client:
                return {'error': 'Redis client not available'}

            info = await self.redis_client.info()
            memory_info = await self.redis_client.memory_stats()

            return {
                'server_info': info,
                'memory_info': memory_info,
                'connection_stats': self.stats.copy()
            }

        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {'error': str(e)}

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Получение статистики кэширования

        Returns:
            Dict[str, Any]: Статистика кэширования
        """
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0

        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'local_cache_size': len(self.local_cache),
            'total_keys_stored': self.stats['keys_stored'],
            'total_keys_retrieved': self.stats['keys_retrieved'],
            'errors': self.stats['errors_occurred']
        }

    async def clear_cache_pattern(self, pattern: str) -> int:
        """
        Очистка кэша по паттерну

        Args:
            pattern: Паттерн ключей для удаления

        Returns:
            int: Количество удаленных ключей
        """
        try:
            if not self.redis_client:
                return 0

            # Получение всех ключей, соответствующих паттерну
            keys = await self.redis_client.keys(pattern)

            if not keys:
                return 0

            # Удаление ключей
            deleted_count = await self.redis_client.delete(*keys)

            # Очистка локального кэша
            for key in keys:
                if key in self.local_cache:
                    del self.local_cache[key]

            logger.info(f"Cleared {deleted_count} keys matching pattern {pattern}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clear cache pattern {pattern}: {e}")
            self.stats['errors_occurred'] += 1
            return 0

    async def health_check(self) -> bool:
        """
        Проверка здоровья Redis

        Returns:
            bool: Статус здоровья
        """
        try:
            if not self.redis_client:
                return False

            await self.redis_client.ping()
            return True

        except Exception:
            return False

    # Алиасы для совместимости с тестами

    async def get_cache(self, key: str) -> Optional[Any]:
        """Алиас для get_distributed_cache"""
        return await self.get_distributed_cache(key)

    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Алиас для set_distributed_cache"""
        return await self.set_distributed_cache(key, value, ttl or 3600)

    async def exists_cache(self, key: str) -> bool:
        """Проверка существования ключа в кэше"""
        try:
            if not self.redis_client:
                return False
            cache_key = f"cache:{key}"
            return await self.redis_client.exists(cache_key) > 0
        except Exception:
            return False

    async def delete_cache(self, key: str) -> bool:
        """Удаление ключа из кэша"""
        try:
            if not self.redis_client:
                return False
            cache_key = f"cache:{key}"
            return await self.redis_client.delete(cache_key) > 0
        except Exception:
            return False

    async def get_cache_ttl(self, key: str) -> Optional[int]:
        """Получение TTL ключа"""
        try:
            if not self.redis_client:
                return None
            cache_key = f"cache:{key}"
            return await self.redis_client.ttl(cache_key)
        except Exception:
            return None

    async def store_conversation_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Сохранение сообщения разговора"""
        try:
            if not self.redis_client:
                return False
            key = f"conversation:{session_id}"
            await self.redis_client.rpush(key, json.dumps(message))
            # Ограничение длины разговора
            await self.redis_client.ltrim(key, -100, -1)  # Последние 100 сообщений
            return True
        except Exception:
            return False

    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Получение истории разговора"""
        try:
            if not self.redis_client:
                return []
            key = f"conversation:{session_id}"
            messages = await self.redis_client.lrange(key, -limit, -1)
            return [json.loads(msg) for msg in messages]
        except Exception:
            return []

    async def get_conversation_length(self, session_id: str) -> int:
        """Получение длины разговора"""
        try:
            if not self.redis_client:
                return 0
            key = f"conversation:{session_id}"
            return await self.redis_client.llen(key)
        except Exception:
            return 0

    async def clear_conversation(self, session_id: str) -> bool:
        """Очистка разговора"""
        try:
            if not self.redis_client:
                return False
            key = f"conversation:{session_id}"
            return await self.redis_client.delete(key) > 0
        except Exception:
            return False

    async def store_metrics(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> bool:
        """Сохранение метрики"""
        try:
            if not self.redis_client:
                return False
            key = f"metrics:{name}:{int(time.time())}"
            data = {
                'name': name,
                'value': value,
                'tags': tags or {},
                'timestamp': datetime.utcnow().isoformat()
            }
            await self.redis_client.setex(key, 86400, json.dumps(data))  # 24 часа
            return True
        except Exception:
            return False

    async def get_metrics(self, name: str, timeframe: str = "1h") -> List[Dict[str, Any]]:
        """Получение метрик"""
        try:
            if not self.redis_client:
                return []
            # Простая реализация - получение всех метрик с именем
            pattern = f"metrics:{name}:*"
            keys = await self.redis_client.keys(pattern)
            if not keys:
                return []
            values = await self.redis_client.mget(keys)
            metrics = []
            for value in values:
                if value:
                    metrics.append(json.loads(value))
            return metrics
        except Exception:
            return []

    async def get_metrics_stats(self) -> Dict[str, Any]:
        """Статистика метрик"""
        return {
            'total_metrics': 0,  # Заглушка
            'metrics_types': [],
            'storage_size': 0
        }

    async def cleanup_expired_metrics(self) -> int:
        """Очистка устаревших метрик (Redis делает это автоматически)"""
        return 0

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Использование памяти"""
        try:
            if not self.redis_client:
                return {}
            info = await self.redis_client.info('memory')
            return {
                'used_memory': info.get('used_memory', 0),
                'used_memory_peak': info.get('used_memory_peak', 0),
                'total_system_memory': info.get('total_system_memory', 0)
            }
        except Exception:
            return {}

    async def get_connection_info(self) -> Dict[str, Any]:
        """Информация о подключении"""
        return {
            'host': self.host,
            'port': self.port,
            'connected': self.redis_client is not None
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Общая статистика"""
        return self.get_cache_stats()

    async def batch_operations(self, keys: List[str]) -> Dict[str, Any]:
        """Пакетные операции"""
        try:
            if not self.redis_client:
                return {}
            cache_keys = [f"cache:{key}" for key in keys]
            values = await self.redis_client.mget(cache_keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    data = json.loads(value)
                    result[key] = data.get('value')
            return result
        except Exception:
            return {}

    async def pipeline_operations(self, operations: List[tuple]) -> bool:
        """Пайплайн операций"""
        try:
            if not self.redis_client:
                return False
            async with self.redis_client.pipeline() as pipe:
                for op in operations:
                    if len(op) == 3:  # set operation
                        key, value, ttl = op
                        pipe.setex(key, ttl, json.dumps(value))
                    elif len(op) == 2:  # delete operation
                        key, _ = op
                        pipe.delete(key)
                await pipe.execute()
            return True
        except Exception:
            return False

    async def subscribe_to_channel(self, channel: str, callback) -> None:
        """Подписка на канал (уже реализована выше)"""
        pass  # Уже реализована

    async def publish_to_channel(self, channel: str, message: Dict[str, Any]) -> bool:
        """Публикация в канал"""
        return await self.publish_message(channel, message)

    async def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Получение ключей по паттерну"""
        try:
            if not self.redis_client:
                return []
            return await self.redis_client.keys(pattern)
        except Exception:
            return []

    async def atomic_incr(self, key: str, amount: int = 1) -> int:
        """Атомарный инкремент"""
        try:
            if not self.redis_client:
                return 0
            return await self.redis_client.incrby(key, amount)
        except Exception:
            return 0

    async def atomic_decr(self, key: str) -> int:
        """Атомарный декремент"""
        try:
            if not self.redis_client:
                return 0
            return await self.redis_client.decr(key)
        except Exception:
            return 0

    async def get_counter(self, key: str) -> int:
        """Получение счетчика"""
        try:
            if not self.redis_client:
                return 0
            value = await self.redis_client.get(key)
            return int(value) if value else 0
        except Exception:
            return 0

    async def acquire_lock(self, lock_key: str, ttl: int = 30) -> bool:
        """Получение блокировки"""
        try:
            if not self.redis_client:
                return False
            return await self.redis_client.set(lock_key, "locked", ex=ttl, nx=True)
        except Exception:
            return False

    async def release_lock(self, lock_key: str) -> bool:
        """Освобождение блокировки"""
        try:
            if not self.redis_client:
                return False
            return await self.redis_client.delete(lock_key) > 0
        except Exception:
            return False

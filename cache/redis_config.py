"""
Конфигурация Redis для системы кэширования
"""

import os
import redis
from typing import Optional
from urllib.parse import urlparse
from .cache_system_enhanced import EnhancedCacheSystem
import logging

logger = logging.getLogger(__name__)

class RedisManager:
    """Класс для управления подключением к Redis"""
    
    def __init__(self):
        self._redis_client: Optional[redis.Redis] = None
        self._connected = False
        self._setup_redis_connection()
    
    def _setup_redis_connection(self):
        """Настройка подключения к Redis"""
        try:
            # Проверяем переменную окружения REDIS_URL
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                # Извлекаем параметры из URL
                parsed = urlparse(redis_url)
                self._redis_client = redis.Redis(
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 6379,
                    password=parsed.password,
                    db=int(parsed.path[1:]) if parsed.path and parsed.path != "/" else 0,
                    decode_responses=False,  # Для бинарных данных
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                    max_connections=20
                )
            else:
                # Используем стандартные параметры
                self._redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    password=os.getenv("REDIS_PASSWORD"),
                    db=int(os.getenv("REDIS_DB", 0)),
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                    max_connections=20
                )
            
            # Проверяем подключение
            self._redis_client.ping()
            self._connected = True
            logger.info("✅ Подключение к Redis успешно установлено")
            
        except redis.ConnectionError:
            logger.error("❌ Не удалось подключиться к Redis, используется локальный кэш")
            self._connected = False
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Redis: {e}")
            self._connected = False
    
    @property
    def redis_client(self) -> Optional[redis.Redis]:
        """Получить клиент Redis"""
        if self._connected and self._redis_client:
            return self._redis_client
        return None
    
    def is_connected(self) -> bool:
        """Проверить подключение к Redis"""
        return self._connected
    
    def test_connection(self) -> bool:
        """Тестировать подключение к Redis"""
        try:
            if self._redis_client:
                self._redis_client.ping()
                return True
        except Exception:
            pass
        return False

# Глобальный экземпляр RedisManager
redis_manager = RedisManager()

class RedisEnhancedCache:
    """Расширенная система кэширования с поддержкой Redis"""
    
    def __init__(self, use_redis_fallback: bool = True):
        self.redis_manager = redis_manager
        self.use_redis_fallback = use_redis_fallback  # Использовать локальный кэш как резерв
        self.local_cache_system = EnhancedCacheSystem()
        self.default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", "3600"))  # 1 час по умолчанию
        
        logger.info(f"Redis кэш инициализирован, подключен: {self.redis_manager.is_connected()}")
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None, use_redis: bool = True) -> bool:
        """Установить значение в кэш"""
        success = False
        
        # Попытка записи в Redis
        if use_redis and self.redis_manager.is_connected():
            try:
                ttl = ttl or self.default_ttl
                success = self.redis_manager.redis_client.setex(key, ttl, value)
                if success:
                    logger.debug(f"Записано в Redis: {key}")
            except Exception as e:
                logger.error(f"Ошибка записи в Redis: {e}")
                success = False
        
        # Если Redis недоступен и включен резервный режим, используем локальный кэш
        if not success and self.use_redis_fallback:
            # Преобразуем байты в объект для локального кэша
            import pickle
            try:
                obj_value = pickle.loads(value)
                success = self.local_cache_system.set(key, obj_value, ttl or self.default_ttl, "redis_fallback")
                if success:
                    logger.debug(f"Записано в локальный кэш: {key}")
            except Exception as e:
                logger.error(f"Ошибка записи в локальный кэш: {e}")
        
        return bool(success)
    
    def get(self, key: str, use_redis: bool = True) -> Optional[bytes]:
        """Получить значение из кэша"""
        result = None
        
        # Попытка чтения из Redis
        if use_redis and self.redis_manager.is_connected():
            try:
                result = self.redis_manager.redis_client.get(key)
                if result is not None:
                    logger.debug(f"Прочитано из Redis: {key}")
                    return result
            except Exception as e:
                logger.error(f"Ошибка чтения из Redis: {e}")
        
        # Если Redis недоступен или значение не найдено, пробуем локальный кэш
        if self.use_redis_fallback:
            try:
                local_result = self.local_cache_system.get(key, "redis_fallback")
                if local_result is not None:
                    import pickle
                    result = pickle.dumps(local_result)
                    logger.debug(f"Прочитано из локального кэша: {key}")
                    return result
            except Exception as e:
                logger.error(f"Ошибка чтения из локального кэша: {e}")
        
        return None
    
    def delete(self, key: str, use_redis: bool = True) -> bool:
        """Удалить значение из кэша"""
        success = False
        
        # Удаление из Redis
        if use_redis and self.redis_manager.is_connected():
            try:
                result = self.redis_manager.redis_client.delete(key)
                success = bool(result)
                if success:
                    logger.debug(f"Удалено из Redis: {key}")
            except Exception as e:
                logger.error(f"Ошибка удаления из Redis: {e}")
                success = False
        
        # Удаление из локального кэша
        if self.use_redis_fallback:
            local_success = self.local_cache_system.delete(key, "redis_fallback")
            if local_success:
                logger.debug(f"Удалено из локального кэша: {key}")
                success = True  # Обновляем успех, если хотя бы одно удаление прошло
        
        return success
    
    def exists(self, key: str, use_redis: bool = True) -> bool:
        """Проверить существование ключа"""
        exists = False
        
        # Проверка в Redis
        if use_redis and self.redis_manager.is_connected():
            try:
                exists = bool(self.redis_manager.redis_client.exists(key))
            except Exception as e:
                logger.error(f"Ошибка проверки существования в Redis: {e}")
                exists = False
        
        # Если не найдено в Redis, проверяем в локальном кэше
        if not exists and self.use_redis_fallback:
            local_cache = self.local_cache_system.get_cache("redis_fallback")
            if local_cache.get(key) is not None:
                exists = True
        
        return exists
    
    def keys(self, pattern: str = "*", use_redis: bool = True) -> list:
        """Получить список ключей по паттерну"""
        keys = []
        
        # Получение ключей из Redis
        if use_redis and self.redis_manager.is_connected():
            try:
                redis_keys = self.redis_manager.redis_client.keys(pattern)
                keys.extend([key.decode('utf-8') if isinstance(key, bytes) else key for key in redis_keys])
            except Exception as e:
                logger.error(f"Ошибка получения ключей из Redis: {e}")
        
        # Получение ключей из локального кэша
        if self.use_redis_fallback:
            # В локальном кэше нет прямой поддержки паттернов, поэтому возвращаем пустой список
            # или реализуем простую фильтрацию
            pass
        
        return keys

    def flush_all(self, use_redis: bool = True) -> bool:
        """Очистить весь кэш"""
        success = True
        
        # Очистка Redis
        if use_redis and self.redis_manager.is_connected():
            try:
                self.redis_manager.redis_client.flushdb()
                logger.info("Redis кэш очищен")
            except Exception as e:
                logger.error(f"Ошибка очистки Redis: {e}")
                success = False
        
        # Очистка локального кэша
        if self.use_redis_fallback:
            local_success = self.local_cache_system.clear("redis_fallback")
            if local_success:
                logger.info("Локальный кэш очищен")
            else:
                success = False
        
        return success

    def get_cache_info(self) -> dict:
        """Получить информацию о состоянии кэша"""
        info = {
            "redis_connected": self.redis_manager.is_connected(),
            "use_redis_fallback": self.use_redis_fallback,
            "default_ttl": self.default_ttl
        }
        
        if self.redis_manager.is_connected():
            try:
                redis_info = self.redis_manager.redis_client.info()
                info.update({
                    "redis_version": redis_info.get("redis_version"),
                    "redis_mode": redis_info.get("redis_mode"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "used_memory_human": redis_info.get("used_memory_human"),
                    "total_commands_processed": redis_info.get("total_commands_processed"),
                    "keyspace_hits": redis_info.get("keyspace_hits"),
                    "keyspace_misses": redis_info.get("keyspace_misses"),
                })
            except Exception as e:
                logger.error(f"Ошибка получения информации о Redis: {e}")
                info["redis_info_error"] = str(e)
        
        # Информация о локальном кэше
        if self.use_redis_fallback:
            info["local_cache_stats"] = self.local_cache_system.get_system_stats()
        
        return info

# Глобальный экземпляр RedisEnhancedCache
redis_cache = RedisEnhancedCache()

def get_redis_cache() -> RedisEnhancedCache:
    """Получить экземпляр Redis кэша"""
    return redis_cache

def get_redis_manager() -> RedisManager:
    """Получить экземпляр Redis менеджера"""
    return redis_manager
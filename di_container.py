"""
Система внедрения зависимостей (Dependency Injection) для AI-агента
"""

import inspect
from typing import Any, Dict, Type, Callable, Optional, get_type_hints
from functools import wraps
from enum import Enum


class Lifecycle(Enum):
    """
    Типы жизненного цикла для сервисов
    """
    SINGLETON = "singleton"  # Один экземпляр на всё приложение
    TRANSIENT = "transient"  # Новый экземпляр при каждом запросе
    SCOPED = "scoped"  # Один экземпляр на область видимости (пока не реализовано)


class DIContainer:
    """
    Контейнер внедрения зависимостей
    """
    
    def __init__(self):
        self._registrations: Dict[Type, Dict[str, Any]] = {}
        self._instances: Dict[Type, Any] = {}  # Для singleton экземпляров
        self._decorators = []
    
    def register(self, service_type: Type, implementation: Type = None, 
                 lifecycle: Lifecycle = Lifecycle.SINGLETON, factory: Callable = None):
        """
        Регистрация сервиса в контейнере
        
        Args:
            service_type: Тип интерфейса или класса, который регистрируется
            implementation: Реализация сервиса (если отличается от service_type)
            lifecycle: Жизненный цикл сервиса
            factory: Фабричная функция для создания экземпляра
        """
        if implementation is None:
            implementation = service_type
        
        self._registrations[service_type] = {
            'implementation': implementation,
            'lifecycle': lifecycle,
            'factory': factory
        }
    
    def resolve(self, service_type: Type) -> Any:
        """
        Получение экземпляра зарегистрированного сервиса
        
        Args:
            service_type: Тип сервиса для получения
            
        Returns:
            Экземпляр сервиса
        """
        if service_type not in self._registrations:
            raise ValueError(f"Сервис {service_type} не зарегистрирован в контейнере")
        
        registration = self._registrations[service_type]
        lifecycle = registration['lifecycle']
        
        # Для singleton возвращаем кешированный экземпляр
        if lifecycle == Lifecycle.SINGLETON and service_type in self._instances:
            return self._instances[service_type]
        
        # Создаем новый экземпляр
        instance = self._create_instance(registration['implementation'], registration['factory'])
        
        # Кешируем singleton экземпляр
        if lifecycle == Lifecycle.SINGLETON:
            self._instances[service_type] = instance
        
        return instance
    
    def _create_instance(self, implementation: Type, factory: Callable = None):
        """
        Создание экземпляра сервиса с автоматическим внедрением зависимостей
        
        Args:
            implementation: Класс для создания экземпляра
            factory: Фабричная функция (если указана)
            
        Returns:
            Экземпляр класса с внедренными зависимостями
        """
        if factory:
            # Если указана фабричная функция, используем её с внедрением зависимостей
            return self._inject_dependencies_into_factory(factory)
        
        # Используем конструктор класса с внедрением зависимостей
        return self._inject_dependencies_into_constructor(implementation)
    
    def _inject_dependencies_into_constructor(self, cls: Type):
        """
        Внедрение зависимостей через конструктор класса
        """
        # Получаем аннотации типов параметров конструктора
        try:
            init_signature = inspect.signature(cls.__init__)
            type_hints = get_type_hints(cls.__init__)
        except (NameError, TypeError):
            # Если не удается получить аннотации, создаем экземпляр без внедрения
            return cls()
        
        # Пропускаем 'self' параметр
        parameters = list(init_signature.parameters.keys())[1:]
        
        dependencies = {}
        for param_name in parameters:
            if param_name in type_hints:
                param_type = type_hints[param_name]
                if param_type in self._registrations:
                    dependencies[param_name] = self.resolve(param_type)
                else:
                    # Если зависимость не зарегистрирована, ищем как тип
                    raise ValueError(f"Зависимость {param_type} для параметра {param_name} не зарегистрирована")
        
        return cls(**dependencies)
    
    def _inject_dependencies_into_factory(self, factory: Callable):
        """
        Внедрение зависимостей в фабричную функцию
        """
        try:
            signature = inspect.signature(factory)
            type_hints = get_type_hints(factory)
        except (NameError, TypeError):
            # Если не удается получить аннотации, вызываем фабрику без внедрения
            return factory()
        
        dependencies = {}
        for param_name, param in signature.parameters.items():
            if param_name in type_hints:
                param_type = type_hints[param_name]
                if param_type in self._registrations:
                    dependencies[param_name] = self.resolve(param_type)
                else:
                    raise ValueError(f"Зависимость {param_type} для параметра {param_name} не зарегистрирована")
        
        return factory(**dependencies)
    
    def register_singleton(self, service_type: Type, implementation: Type = None, factory: Callable = None):
        """
        Регистрация singleton сервиса
        """
        self.register(service_type, implementation, Lifecycle.SINGLETON, factory)
    
    def register_transient(self, service_type: Type, implementation: Type = None, factory: Callable = None):
        """
        Регистрация transient сервиса
        """
        self.register(service_type, implementation, Lifecycle.TRANSIENT, factory)
    
    def register_scoped(self, service_type: Type, implementation: Type = None, factory: Callable = None):
        """
        Регистрация scoped сервиса
        """
        self.register(service_type, implementation, Lifecycle.SCOPED, factory)
    
    def get_dependencies_info(self) -> Dict:
        """
        Получение информации о зарегистрированных зависимостях
        """
        return {
            'registrations': {k.__name__: {
                'implementation': v['implementation'].__name__,
                'lifecycle': v['lifecycle'].value
            } for k, v in self._registrations.items()},
            'instances_count': len(self._instances)
        }


# Глобальный контейнер для удобства
global_container = DIContainer()


def inject(func):
    """
    Декоратор для внедрения зависимостей в функции
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Получаем аннотации типов параметров функции
        type_hints = get_type_hints(func)
        signature = inspect.signature(func)
        
        # Проверяем, есть ли непроставленные зависимости
        for param_name, param in signature.parameters.items():
            if param_name in type_hints and param_name not in kwargs:
                param_type = type_hints[param_name]
                # Пытаемся получить зависимость из глобального контейнера
                try:
                    kwargs[param_name] = global_container.resolve(param_type)
                except ValueError:
                    # Если зависимость не найдена, пропускаем (пусть будет ошибка вызова функции)
                    continue
        
        return func(*args, **kwargs)
    
    return wrapper


def register_service(service_type: Type = None, lifecycle: Lifecycle = Lifecycle.SINGLETON):
    """
    Декоратор для регистрации сервисов в контейнере
    
    Args:
        service_type: Тип сервиса (если отличается от самого класса)
        lifecycle: Жизненный цикл сервиса
    """
    def decorator(cls):
        nonlocal service_type
        if service_type is None:
            service_type = cls
        
        # Регистрируем класс в глобальном контейнере
        global_container.register(service_type, cls, lifecycle)
        
        # Добавляем методы для создания экземпляров с внедрением зависимостей
        @wraps(cls.__init__)
        def init_with_injection(self, *args, **kwargs):
            # Внедряем зависимости в конструктор
            type_hints = get_type_hints(cls.__init__)
            signature = inspect.signature(cls.__init__)
            
            # Проверяем, есть ли непроставленные зависимости
            for param_name, param in signature.parameters.items():
                if param_name != 'self' and param_name in type_hints and param_name not in kwargs:
                    param_type = type_hints[param_name]
                    # Пытаемся получить зависимость из глобального контейнера
                    try:
                        kwargs[param_name] = global_container.resolve(param_type)
                    except ValueError:
                        # Если зависимость не найдена, оставляем как есть
                        continue
            
            # Вызываем оригинальный конструктор
            original_init = cls.__dict__.get('__init__')
            if original_init:
                original_init(self, *args, **kwargs)
        
        cls.__init__ = init_with_injection
        return cls
    
    return decorator


def get_container() -> DIContainer:
    """
    Получение глобального контейнера зависимостей
    """
    return global_container


# Примеры использования:

if __name__ == "__main__":
    # Пример 1: Регистрация сервиса с помощью декоратора
    @register_service(lifecycle=Lifecycle.SINGLETON)
    class DatabaseService:
        def __init__(self, connection_string: str = "default_db"):
            self.connection_string = connection_string
        
        def connect(self):
            return f"Подключено к {self.connection_string}"
    
    # Пример 2: Регистрация сервиса через контейнер напрямую
    class CacheService:
        def __init__(self, db_service: DatabaseService):
            self.db_service = db_service
        
        def get_from_cache(self, key: str):
            return f"Данные из кэша для {key}, используя БД: {self.db_service.connect()}"
    
    global_container.register_singleton(CacheService)
    
    # Пример 3: Использование внедрения зависимостей в функции
    @inject
    def my_function(cache_service: CacheService):
        return cache_service.get_from_cache("test_key")
    
    # Тестирование
    result = my_function()
    print(result)
    
    # Проверка, что сервисы являются singleton
    db1 = global_container.resolve(DatabaseService)
    db2 = global_container.resolve(DatabaseService)
    print(f"Одинаковые экземпляры БД: {db1 is db2}")
    
    cache1 = global_container.resolve(CacheService)
    cache2 = global_container.resolve(CacheService)
    print(f"Одинаковые экземпляры кэша: {cache1 is cache2}")
    print(f"Одинаковые экземпляры БД в кэше: {cache1.db_service is cache2.db_service}")
# Система внедрения зависимостей (Dependency Injection)

## Обзор

Система внедрения зависимостей (DI) предоставляет гибкий способ управления зависимостями между компонентами приложения. Она позволяет создавать слабосвязанный код, упрощает тестирование и улучшает модульность системы.

## Архитектура

### Основные компоненты

1. **DIContainer** - основной контейнер, управляющий регистрацией и разрешением зависимостей
2. **Lifecycle** - перечисление типов жизненного цикла сервисов
3. **Декораторы** - вспомогательные инструменты для регистрации и внедрения зависимостей

### Типы жизненного цикла

- **SINGLETON** - один экземпляр на всё приложение
- **TRANSIENT** - новый экземпляр при каждом запросе
- **SCOPED** - один экземпляр на область видимости (в разработке)

## Использование

### Регистрация сервисов

#### 1. Через декоратор

```python
from di_container import register_service, Lifecycle

@register_service(lifecycle=Lifecycle.SINGLETON)
class DatabaseService:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def connect(self):
        return f"Подключено к {self.connection_string}"
```

#### 2. Через контейнер напрямую

```python
from di_container import DIContainer

container = DIContainer()

# Регистрация singleton сервиса
container.register_singleton(DatabaseService)

# Регистрация transient сервиса
container.register_transient(CacheService)

# Регистрация с указанием реализации
container.register(InterfaceType, ImplementationType, Lifecycle.SINGLETON)

# Регистрация через фабричную функцию
def create_service():
    return DatabaseService("custom_connection")

container.register(DatabaseService, factory=create_service, lifecycle=Lifecycle.TRANSIENT)
```

### Разрешение зависимостей

```python
# Получение сервиса из контейнера
db_service = container.resolve(DatabaseService)

# Автоматическое внедрение зависимостей в конструктор
class UserService:
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

# Контейнер автоматически внедрит db_service при создании UserService
user_service = container.resolve(UserService)
```

### Внедрение зависимостей в функции

```python
from di_container import inject

@inject
def handle_request(user_service: UserService, user_id: str):
    return user_service.get_user(user_id)

# Зависимости будут автоматически внедрены при вызове функции
result = handle_request(user_id="123")
```

## Примеры использования

### Простой пример

```python
from di_container import DIContainer, register_service, Lifecycle

@register_service(lifecycle=Lifecycle.SINGLETON)
class NotificationService:
    def send(self, message: str):
        return f"Отправлено: {message}"

class EmailService:
    def __init__(self, notification_service: NotificationService):
        self.notification = notification_service
    
    def send_email(self, email: str, message: str):
        result = self.notification.send(f"Email to {email}: {message}")
        return result

# Настройка контейнера
container = DIContainer()
container.register_singleton(EmailService)

# Использование
email_service = container.resolve(EmailService)
result = email_service.send_email("user@example.com", "Привет!")
```

### Пример с фабричной функцией

```python
from di_container import DIContainer

def create_database_service():
    # Фабрика может содержать сложную логику создания сервиса
    connection_string = get_connection_string_from_config()
    return DatabaseService(connection_string)

container = DIContainer()
container.register(DatabaseService, factory=create_database_service, Lifecycle.SINGLETON)
```

## Интеграция с существующим кодом

Система DI может быть интегрирована с существующими классами без изменения их кода:

```python
# Существующий класс
class ExistingService:
    def __init__(self, config_param: str = "default"):
        self.config = config_param

# Регистрация в контейнере
container.register_singleton(ExistingService)

# Использование
service = container.resolve(ExistingService)
```

## Тестирование

Система DI упрощает тестирование за счет возможности подмены зависимостей:

```python
def test_user_service_with_mock_db():
    container = DIContainer()
    
    # Регистрируем mock вместо реальной БД
    container.register_singleton(MockDatabaseService)
    
    # Резолвим сервис с mock зависимостью
    user_service = container.resolve(UserService)
    
    # Тестируем
    result = user_service.create_user("test_user")
    assert result is not None
```

## Лучшие практики

1. **Используйте интерфейсы** - создавайте абстракции для сервисов
2. **Избегайте Service Locator** - предпочитайте внедрение зависимостей через конструктор
3. **Выбирайте правильный жизненный цикл** - singleton только для действительно общих ресурсов
4. **Тестируйте с mock объектами** - используйте DI для подмены зависимостей в тестах

## API Reference

### DIContainer

- `register(service_type, implementation=None, lifecycle=Lifecycle.SINGLETON, factory=None)` - регистрирует сервис
- `resolve(service_type)` - возвращает экземпляр сервиса
- `register_singleton(service_type, implementation=None, factory=None)` - регистрирует singleton сервис
- `register_transient(service_type, implementation=None, factory=None)` - регистрирует transient сервис
- `get_dependencies_info()` - возвращает информацию о зарегистрированных зависимостях

### Декораторы

- `@register_service(service_type=None, lifecycle=Lifecycle.SINGLETON)` - декоратор для регистрации сервисов
- `@inject` - декоратор для внедрения зависимостей в функции

## Расширение системы

Система может быть легко расширена для поддержки дополнительных функций:

- Scoped зависимости
- Условные регистрации
- Атрибутная инъекция
- Поддержка асинхронных сервисов
"""
Примеры использования системы Dependency Injection в существующем коде проекта
"""

from di_container import DIContainer, register_service, inject, Lifecycle
from agent.core.agent_core import AgentCore
from api.auth import AuthService
from database.postgres import PostgresDB
from integrations.llm_client import LLMClient


# Пример 1: Регистрация существующих сервисов через декоратор
@register_service(lifecycle=Lifecycle.SINGLETON)
class ExampleAuthService(AuthService):
    """Пример расширения существующего сервиса аутентификации"""
    
    def __init__(self, db_connection: PostgresDB):
        super().__init__()
        self.db = db_connection
    
    def authenticate(self, token: str):
        # Пример кастомной логики аутентификации
        print(f"Аутентификация с использованием токена: {token}")
        # Проверяем в БД
        result = self.db.execute_query("SELECT * FROM users WHERE token = %s", (token,))
        return result is not None


@register_service(lifecycle=Lifecycle.SINGLETON)
class ExampleLLMClient(LLMClient):
    """Пример расширения LLM клиента"""
    
    def __init__(self, config_service):
        super().__init__()
        self.config = config_service
    
    def generate_response(self, prompt: str):
        print(f"Генерация ответа для: {prompt}")
        # Используем конфигурацию из зависимостей
        model = self.config.get('llm_model', 'default_model')
        return f"Ответ от {model} для запроса: {prompt}"


# Пример 2: Создание нового сервиса с зависимостями
@register_service(lifecycle=Lifecycle.SINGLETON)
class CognitiveService:
    """Сервис когнитивных функций"""
    
    def __init__(self, llm_client: LLMClient, auth_service: AuthService):
        self.llm_client = llm_client
        self.auth_service = auth_service
    
    def process_request(self, user_id: str, request: str):
        # Проверяем аутентификацию
        if not self.auth_service.authenticate(user_id):
            raise Exception("Пользователь не аутентифицирован")
        
        # Обрабатываем запрос с помощью LLM
        response = self.llm_client.generate_response(request)
        return response


# Пример 3: Использование внедрения зависимостей в функциях
@inject
def handle_user_request(cognitive_service: CognitiveService, 
                       user_id: str, 
                       user_request: str):
    """Функция обработки пользовательского запроса с внедрением зависимостей"""
    try:
        result = cognitive_service.process_request(user_id, user_request)
        return {"status": "success", "response": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Пример 4: Использование контейнера напрямую для регистрации существующих классов
def setup_di_container():
    """Настройка контейнера зависимостей для существующих классов"""
    container = DIContainer()
    
    # Регистрируем существующие сервисы
    container.register_singleton(PostgresDB)
    container.register_singleton(AuthService)
    container.register_singleton(LLMClient)
    
    # Регистрируем новые сервисы
    container.register_singleton(CognitiveService)
    
    return container


# Пример 5: Интеграция с существующим AgentCore
@register_service(lifecycle=Lifecycle.SINGLETON)
class EnhancedAgentCore(AgentCore):
    """Расширенная версия AgentCore с зависимостями"""
    
    def __init__(self, cognitive_service: CognitiveService, db: PostgresDB):
        super().__init__()
        self.cognitive_service = cognitive_service
        self.db = db
    
    def process_input(self, input_data: dict):
        # Используем внедренные зависимости
        user_id = input_data.get('user_id')
        request = input_data.get('request')
        
        # Обрабатываем через когнитивный сервис
        result = self.cognitive_service.process_request(user_id, request)
        
        # Сохраняем в БД
        self.db.execute_query(
            "INSERT INTO agent_logs (user_id, request, response) VALUES (%s, %s, %s)",
            (user_id, request, result)
        )
        
        return result


# Пример использования
if __name__ == "__main__":
    # Создаем контейнер и регистрируем сервисы
    container = setup_di_container()
    
    # Используем внедрение зависимостей в функции
    result = handle_user_request(user_id="user123", user_request="Привет, как дела?")
    print(f"Результат обработки: {result}")
    
    # Получаем сервисы из контейнера
    cognitive_service = container.resolve(CognitiveService)
    agent_core = container.resolve(EnhancedAgentCore)
    
    # Используем сервисы
    input_data = {"user_id": "user123", "request": "Расскажи о погоде"}
    agent_response = agent_core.process_input(input_data)
    print(f"Ответ агента: {agent_response}")
    
    # Проверяем информацию о зависимостях
    print("\nИнформация о зависимостях:")
    for name, info in container.get_dependencies_info()['registrations'].items():
        print(f"- {name}: реализация {info['implementation']}, жизненный цикл {info['lifecycle']}")
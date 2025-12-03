# Анализ безопасности проекта AI-агента с мета-познанием

## Обзор

В ходе анализа безопасности проекта AI-агента с мета-познанием были выявлены потенциальные уязвимости и риски, а также предложены конкретные меры по их устранению. Ниже представлены детализированные результаты анализа и рекомендации по улучшению безопасности системы.

## Категории выявленных уязвимостей

### 1. Уязвимости в интеграциях с LLM

#### Проблема: Уязвимости к prompt injection атакам
- **Описание**: В `integrations/llm_client.py` отсутствует адекватная защита от prompt injection атак, что может позволить злоумышленнику манипулировать поведением LLM
- **Риск**: Возможность обхода систем безопасности, раскрытие чувствительных данных, выполнение нежелательных действий
- **Рекомендация**: Внедрить систему санитизации и валидации входных данных перед отправкой в LLM

#### Проблема: Небезопасная передача данных
- **Описание**: При вызовах LLM API данные передаются без дополнительного шифрования
- **Риск**: Перехват конфиденциальной информации при передаче
- **Рекомендация**: Использовать зашифрованные каналы связи и шифрование чувствительных данных

### 2. Проблемы с управлением API-ключами и секретами

#### Проблема: Хранение API-ключей в открытом виде
- **Описание**: В `config.py` и `integrations/llm_client.py` API-ключи хранятся в открытом виде
- **Риск**: Компрометация API-ключей при доступе к коду или конфигурациям
- **Рекомендация**: Использовать безопасное хранилище секретов (HashiCorp Vault, AWS Secrets Manager и т.д.)

#### Проблема: Жестко закодированные значения по умолчанию
- **Описание**: В `config.py` и `api/auth.py` используются предсказуемые значения по умолчанию для ключей
- **Риск**: Использование известных ключей в продакшене
- **Рекомендация**: Обязательная проверка и замена значений по умолчанию при деплое

### 3. Проблемы с аутентификацией и авторизацией

#### Проблема: Использование заглушек аутентификации
- **Описание**: В `api/auth.py` реализована заглушка вместо полноценной системы аутентификации
- **Риск**: Отсутствие контроля доступа к системе
- **Рекомендация**: Реализовать полноценную систему аутентификации с JWT-токенами и refresh-токенами

#### Проблема: Недостаточная проверка токенов
- **Описание**: В системе аутентификации отсутствует проверка на инвалидацию токенов
- **Риск**: Продолжение использования скомпрометированных токенов
- **Рекомендация**: Использовать Redis для хранения инвалидации токенов

### 4. Проблемы с XSS и CSRF защитой

#### Проблема: Недостаточная санитизация данных
- **Описание**: В `frontend/src/features/agent-interaction/AgentChatInterface.tsx` напрямую отображается контент от агента без санитизации
- **Риск**: Cross-Site Scripting (XSS) атаки
- **Рекомендация**: Использовать библиотеки санитизации (например, DOMPurify) для очистки HTML-контента

#### Проблема: Отсутствие CSRF-токенов
- **Описание**: В API отсутствуют CSRF-токены для защиты от межсайтовой подделки запросов
- **Риск**: Выполнение нежелательных действий от имени пользователя
- **Рекомендация**: Внедрить CSRF-токены для всех изменяющих данных операций

### 5. Проблемы безопасности конфигурации и секретов

#### Проблема: Утечки чувствительных данных в логах
- **Описание**: В системе логирования могут просачиваться API-ключи, токены и другие чувствительные данные
- **Риск**: Раскрытие конфиденциальной информации через логи
- **Рекомендация**: Внедрить фильтрацию чувствительных данных в логах

#### Проблема: Небезопасное хранение конфигурации
- **Описание**: Конфигурационные файлы могут содержать чувствительные данные
- **Риск**: Раскрытие конфигурации при компрометации системы
- **Рекомендация**: Использовать шифрование для конфигурационных файлов

### 6. Проблемы с защитой от DDoS-атак

#### Проблема: Простая схема rate limiting
- **Описание**: В `api/rate_limiter.py` используется простая схема ограничения запросов
- **Риск**: Обход ограничений при распределенных атаках
- **Рекомендация**: Внедрить расширенный rate limiting с обнаружением аномалий

#### Проблема: Отсутствие защиты от атак на ресурсы
- **Описание**: Нет защиты от атак, направленных на истощение ресурсов системы
- **Риск**: Отказ в обслуживании из-за перегрузки системы
- **Рекомендация**: Внедрить систему мониторинга и автоматического ограничения нагрузки

### 7. Проблемы безопасности логирования и мониторинга

#### Проблема: Утечки информации через логи
- **Описание**: Логи могут содержать информацию о внутренней структуре системы
- **Риск**: Сбор информации для планирования атак
- **Рекомендация**: Внедрить систему фильтрации и анонимизации логов

#### Проблема: Недостаточное логирование безопасности
- **Описание**: Отсутствует полноценная система аудита безопасности
- **Риск**: Невозможность расследования инцидентов
- **Рекомендация**: Внедрить систему аудита безопасности с детализированным логированием

## Рекомендации по улучшению безопасности

### 1. Защита от атак на уровне API

#### Реализация надежной валидации
```python
class InputValidator:
    def __init__(self):
        self.max_query_length = 10000  # Максимальная длина запроса
        self.max_context_depth = 10    # Максимальная глубина контекста
        self.max_nested_objects = 5    # Максимальное количество вложенных объектов

    async def validate_agent_request(self, query: str, user_id: Optional[str] = None) -> ValidationResult:
        errors = []
        warnings = []

        # Проверка длины запроса
        if len(query) > self.max_query_length:
            errors.append(f"Query too long (max {self.max_query_length} characters)")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Проверка на вложенные структуры
        if self._has_excessive_nesting(query):
            errors.append("Query contains excessive nesting")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Остальная валидация...
```

#### Внедрение системы rate limiting
```python
class AdvancedRateLimiter:
    def __init__(self, use_redis: bool = True):
        self._limits: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        self._rules: Dict[str, AdvancedRateLimitRule] = {}
        self._anomaly_cache: Dict[str, List[float]] = defaultdict(list)

    def _detect_anomaly(self, key: str, endpoint: str, now: float, rule: AdvancedRateLimitRule) -> bool:
        """Обнаружение аномальных паттернов запросов"""
        info = self._limits[key]

        if len(info.request_times) < 10:  # Нужно минимум 10 запросов для анализа
            return False

        # Вычисление интервалов между запросами
        intervals = []
        times_list = list(info.request_times)
        for i in range(1, len(times_list)):
            intervals.append(times_list[i] - times_list[i-1])

        if not intervals:
            return False

        # Статистический анализ
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5

        # Если последний интервал значительно меньше среднего
        if intervals and intervals[-1] < avg_interval - (rule.anomaly_threshold * std_dev):
            logger.warning(f"Anomaly detected for {key} on {endpoint}: rapid request pattern")
            return True

        return False
```

### 2. Безопасное хранение и обработка данных

#### Шифрование чувствительных данных в памяти
```python
class MemoryManager:
    def __init__(self, max_entries: int = 1000, max_working_memory_mb: float = 50.0, max_semantic_memory_mb: float = 10.0):
        # ... существующий код ...

        # Добавить шифрование для чувствительных данных
        encryption_key = os.getenv("MEMORY_ENCRYPTION_KEY")
        if encryption_key:
            self.cipher_suite = Fernet(encryption_key.encode())
        else:
            self.cipher_suite = None
            logger.warning("Memory encryption is disabled - set MEMORY_ENCRYPTION_KEY to enable")

    def _encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Шифрование чувствительных данных в памяти"""
        encrypted_data = data.copy()

        # Определить чувствительные поля
        sensitive_fields = ['user_data', 'personal_info', 'credentials', 'private_content']

        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field] is not None:
                if isinstance(encrypted_data[field], str):
                    encrypted_data[field] = self.cipher_suite.encrypt(
                        encrypted_data[field].encode()
                    ).decode()
                elif isinstance(encrypted_data[field], dict):
                    # Рекурсивное шифрование вложенных словарей
                    encrypted_data[field] = self._encrypt_dict(encrypted_data[field])

        return encrypted_data
```

#### Шифрование данных в базе данных
```python
class PostgreSQLManager:
    def __init__(self, connection_string: str):
        # ... существующий код ...

        # Добавить шифрование для чувствительных данных
        encryption_key = os.getenv("DB_ENCRYPTION_KEY")
        if encryption_key:
            self.cipher_suite = Fernet(encryption_key.encode())
        else:
            self.cipher_suite = None
            logger.warning("Database encryption is disabled - set DB_ENCRYPTION_KEY to enable")

    def _encrypt_experience_sensitive_fields(self, experience: AgentExperience) -> AgentExperience:
        """Шифрование чувствительных полей опыта"""
        # Создаем копию объекта для шифрования
        encrypted_experience = experience.copy() if hasattr(experience, 'copy') else experience

        # Определить чувствительные поля
        sensitive_fields = ['query', 'result', 'metadata']

        for field in sensitive_fields:
            if hasattr(encrypted_experience, field):
                value = getattr(encrypted_experience, field)
                if value and isinstance(value, str):
                    encrypted_value = self.cipher_suite.encrypt(value.encode()).decode()
                    setattr(encrypted_experience, field, encrypted_value)
                elif value and isinstance(value, dict):
                    encrypted_dict = self._encrypt_dict(value)
                    setattr(encrypted_experience, field, encrypted_dict)

        return encrypted_experience
```

### 3. Защита от инъекций в LLM

#### Система санитизации и обнаружения инъекций
```python
class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.base_urls[config.provider]

        # Паттерны для обнаружения prompt injection
        self.injection_patterns = [
            r'(?i)\b(ignore|disregard|forget|override|bypass)\b.*\b(previous|above|following|instructions|rules)\b',
            r'(?i)\b(system|role|function)\b.*[:=].*\b(assistant|user|system)\b',
            r'(?i)\[system\]|\[user\]|\[assistant\]',
            r'(?i)###\s*system\s*:',  # Markdown стили для системных сообщений
            r'(?i)```\s*(javascript|python|bash|sql)',  # Попытки выполнить код
        ]

    def _detect_prompt_injection(self, prompt: str) -> Tuple[bool, List[str]]:
        """Обнаружение потенциальных prompt injection атак"""
        detected_patterns = []

        for pattern in self.injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                detected_patterns.append(pattern)

        return len(detected_patterns) > 0, detected_patterns

    def _sanitize_prompt(self, prompt: str) -> str:
        """Санитизация prompt для предотвращения инъекций"""
        sanitized = prompt

        # Удаление потенциальных команд системы
        sanitized = re.sub(r'(?i)\[system\].*?\[/system\]', '[REDACTED_SYSTEM]', sanitized, flags=re.DOTALL)
        sanitized = re.sub(r'(?i)\[user\].*?\[/user\]', '[REDACTED_USER]', sanitized, flags=re.DOTALL)
        sanitized = re.sub(r'(?i)\[assistant\].*?\[/assistant\]', '[REDACTED_ASSISTANT]', sanitized, flags=re.DOTALL)

        # Удаление потенциальных команд
        sanitized = re.sub(r'(?i)(ignore|disregard|forget|override|bypass)\s+(previous|above|following|instructions|rules)',
                          '[COMMAND_REMOVED]', sanitized)

        # Ограничение длины потенциальных кодовых блоков
        def replace_code_blocks(match):
            code_content = match.group(0)
            if len(code_content) > 1000:  # Ограничение длины кода
                return f"[CODE_BLOCK_TOO_LONG: {len(code_content)} chars removed]"
            return code_content

        # Обработка потенциальных кодовых блоков
        sanitized = re.sub(r'```[\s\S]*?```', replace_code_blocks, sanitized)
        sanitized = re.sub(r'`[^`]{50,}`', '[LONG_CODE_REMOVED]', sanitized)

        return sanitized
```

### 4. Безопасность аутентификации и авторизации

#### Полноценная система аутентификации
```python
class AuthManager:
    def __init__(self):
        # Подключение к Redis для хранения сессий
        self.redis_client = None
        self._setup_redis()

    def _setup_redis(self):
        """Настройка Redis для хранения сессий"""
        try:
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
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Создание токена доступа с возможностью хранения в Redis"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)

        to_encode.update({"exp": expire, "jti": str(uuid.uuid4())})  # JWT ID для отслеживания

        encoded_jwt = jwt.encode(to_encode, self.get_secret_key(), algorithm="HS256")

        # Сохранение токена в Redis для возможности инвалидации
        if self.redis_client:
            try:
                # Храним токен с TTL, соответствующим сроку действия
                ttl_seconds = int(expires_delta.total_seconds()) if expires_delta else 1800  # 30 минут по умолчанию
                self.redis_client.setex(f"token:{encoded_jwt}", ttl_seconds, "valid")
            except Exception as e:
                logger.error(f"Failed to store token in Redis: {e}")

        return encoded_jwt

    async def invalidate_token(self, token: str):
        """Инвалидация токена"""
        if self.redis_client:
            try:
                self.redis_client.setex(f"token:{token}", 86400, "invalid")  # Храним 24 часа как инвалидированный
            except Exception as e:
                logger.error(f"Failed to invalidate token in Redis: {e}")
```

### 5. Защита от XSS и CSRF атак

#### Расширенная санитизация HTML
```python
class InputValidator:
    def __init__(self):
        self.preprocessor = InputPreprocessor()

        # Разрешенные теги и атрибуты для безопасного HTML
        self.allowed_html_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
        ]
        self.allowed_html_attrs = {
            '*': ['class'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title']
        }

        # Паттерны для обнаружения XSS
        self.xss_patterns = [
            r'(?i)<script[^>]*>.*?</script>',
            r'(?i)javascript:',
            r'(?i)on\w+\s*=',
            r'(?i)<iframe[^>]*>',
            r'(?i)<object[^>]*>',
            r'(?i)<embed[^>]*>',
            r'(?i)<form[^>]*>',
            r'(?i)vbscript:',
            r'(?i)data:',
        ]

    def _detect_xss(self, text: str) -> Tuple[bool, List[str]]:
        """Обнаружение потенциальных XSS атак"""
        detected_patterns = []

        for pattern in self.xss_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                detected_patterns.append(pattern)

        return len(detected_patterns) > 0, detected_patterns

    def sanitize_html(self, text: str) -> str:
        """Расширенная санитизация HTML с обнаружением XSS"""
        # Проверка на XSS
        is_xss_detected, xss_patterns = self._detect_xss(text)
        if is_xss_detected:
            logger.warning(f"XSS attempt detected: {xss_patterns}")
            # Удаление подозрительных паттернов
            for pattern in xss_patterns:
                text = re.sub(pattern, '[REMOVED_XSS_CONTENT]', text, flags=re.IGNORECASE | re.DOTALL)

        # Санитизация с помощью bleach
        if BLEACH_AVAILABLE:
            try:
                sanitized = bleach.clean(
                    text,
                    tags=self.allowed_html_tags,
                    attributes=self.allowed_html_attrs,
                    strip=True
                )
                return sanitized
            except Exception as e:
                logger.error(f"HTML sanitization error: {e}")

        # Fallback: простая очистка
        return self._simple_html_sanitize(text)
```

### 6. Безопасность конфигурации и секретов

#### Централизованное управление конфигурацией
```python
class Config:
    """Центральная конфигурация приложения с улучшенной безопасностью"""

    def __init__(self):
        # Список чувствительных полей для маскировки
        self._sensitive_fields = {
            'secret_key', 'jwt_secret_key', 'openai_api_key', 'google_ai_api_key',
            'postgres_password', 'redis_password', 'scraper_api_key', 'serper_api_key',
            'pinecone_api_key', 'chroma_host', 'chroma_port'
        }

        # ================================
        # Application Settings
        # ================================
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.pythonpath = os.getenv("PYTHONPATH", "/app")

        # ================================
        # Database Configuration
        # ================================
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.postgres_db = os.getenv("POSTGRES_DB", "ai_agent")
        self.postgres_user = os.getenv("POSTGRES_USER", "ai_agent")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "")

        # ================================
        # Redis Configuration
        # ================================
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")

        # ================================
        # AI Service API Keys
        # ================================
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_organization = os.getenv("OPENAI_ORGANIZATION", "")
        self.google_ai_api_key = os.getenv("GOOGLE_AI_API_KEY", "")

        # ================================
        # Security Settings
        # ================================
        self.secret_key = os.getenv("SECRET_KEY", "")
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "")

        # Проверка обязательных секретов
        self._validate_secrets()

    def _validate_secrets(self):
        """Проверка и валидация секретов"""
        if self.environment == "production":
            required_secrets = [
                ('SECRET_KEY', self.secret_key),
                ('JWT_SECRET_KEY', self.jwt_secret_key),
                ('POSTGRES_PASSWORD', self.postgres_password)
            ]

            missing_secrets = []
            for name, value in required_secrets:
                if not value or value in ["development-secret-key", "jwt-secret-key", ""]:
                    missing_secrets.append(name)

            if missing_secrets:
                security_logger.critical(f"Missing required secrets in production: {missing_secrets}")
                raise ValueError(f"Missing required secrets in production: {missing_secrets}")
```

### 7. Безопасность логирования и мониторинга

#### Расширенный фильтр для чувствительных данных
```python
class EnhancedSensitiveDataFilter(logging.Filter):
    """Расширенный фильтр для удаления чувствительных данных из логов"""

    def __init__(self):
        super().__init__()
        # Расширенные паттерны чувствительных данных
        sensitive_patterns = [
            r'password[\'"]?\s*[:=]\s*[\'"]?[^\'"\s]{3,}[\'"]?',  # password="value"
            r'token[\'"]?\s*[:=]\s*[\'"]?[^\'"\s]{10,}[\'"]?',    # token="value"
            r'api[_-]?key[\'"]?\s*[:=]\s*[\'"]?[^\'"\s]{10,}[\'"]?', # api_key="value"
            r'secret[\'"]?\s*[:=]\s*[\'"]?[^\'"\s]{10,}[\'"]?',   # secret="value"
            r'Bearer\s+[A-Za-z0-9\-_\.=]{10,}',  # Bearer tokens
            r'Authorization:\s*[\w\s]+[A-Za-z0-9\-_\.=]{10,}',  # Authorization headers
            r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # Credit cards
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails (опционально фильтровать)
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in sensitive_patterns]

        # Чувствительные поля в JSON данных
        self.sensitive_fields = {
            'password', 'token', 'api_key', 'secret', 'authorization',
            'auth', 'credentials', 'key', 'private', 'secret_key',
            'access_token', 'refresh_token', 'bearer', 'credit_card',
            'email', 'phone', 'ssn', 'personal_info'
        }

    def filter(self, record):
        """Фильтрует чувствительные данные из записи лога"""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._filter_sensitive_data(record.msg)

        if hasattr(record, 'args') and record.args:
            # Обработка аргументов сообщения
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    filtered_args.append(self._filter_sensitive_data(arg))
                elif isinstance(arg, (dict, list)):
                    filtered_args.append(self._filter_sensitive_object(arg))
                else:
                    filtered_args.append(arg)
            record.args = tuple(filtered_args)

        # Обработка дополнительных данных в record
        for key in dir(record):
            if not key.startswith('_') and key not in ['msg', 'args']:
                value = getattr(record, key)
                if isinstance(value, str):
                    setattr(record, key, self._filter_sensitive_data(value))
                elif isinstance(value, (dict, list)):
                    setattr(record, key, self._filter_sensitive_object(value))

        return True

    def _filter_sensitive_data(self, text: str) -> str:
        """Удаляет чувствительные данные из текста"""
        if not isinstance(text, str):
            return str(text)

        filtered_text = text
        for pattern in self.compiled_patterns:
            filtered_text = pattern.sub('[FILTERED]', filtered_text)

        return filtered_text

    def _filter_sensitive_object(self, obj: Any) -> Any:
        """Фильтрация чувствительных данных в объектах (dict, list)"""
        if isinstance(obj, dict):
            filtered_dict = {}
            for key, value in obj.items():
                if key.lower() in self.sensitive_fields:
                    filtered_dict[key] = '[FILTERED]'
                elif isinstance(value, (dict, list)):
                    filtered_dict[key] = self._filter_sensitive_object(value)
                elif isinstance(value, str):
                    filtered_dict[key] = self._filter_sensitive_data(value)
                else:
                    filtered_dict[key] = value
            return filtered_dict
        elif isinstance(obj, list):
            return [self._filter_sensitive_object(item) if isinstance(item, (dict, list))
                   else self._filter_sensitive_data(item) if isinstance(item, str)
                   else item for item in obj]
        else:
            return obj
```

## Заключение

Реализация предложенных мер безопасности значительно повысит защищенность AI-агента с мета-познанием. Важно внедрять эти меры поэтапно, начиная с критических уязвимостей, и проводить регулярные аудиты безопасности для поддержания высокого уровня защиты системы.
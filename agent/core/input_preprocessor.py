"""
Input Preprocessor - Предварительная обработка входных данных
"""

import re
import unicodedata
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Попытка импорта NLTK с fallback
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK не доступен, используется fallback токенизация")
    NLTK_AVAILABLE = False
    nltk = None
    word_tokenize = None
    stopwords = None

# Попытка импорта библиотек безопасности
try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    logger.warning("Bleach не доступен, HTML санитизация отключена")
    BLEACH_AVAILABLE = False
    bleach = None

try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    logger.warning("SQLParse не доступен, SQL валидация отключена")
    SQLPARSE_AVAILABLE = False
    sqlparse = None


class PreprocessedInput(BaseModel):
    """Результат предварительной обработки"""
    original: str
    normalized: str
    tokens: List[str]
    entities: List[Dict[str, Any]]
    sentiment: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class InputPreprocessor:
    """Предварительный обработчик входных данных"""

    def __init__(self):
        self.nlp_processor = None  # Lazy loading для тяжелых моделей

        # Загружаем NLTK ресурсы при необходимости
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.warning("NLTK punkt не найден, некоторые функции могут работать некорректно")

            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.warning("NLTK stopwords не найден")

        # Настройки безопасности
        self.max_input_length = 10000  # Максимальная длина входного текста
        self.min_input_length = 3      # Минимальная длина входного текста
        self.allowed_html_tags = []    # Разрешенные HTML теги (пустой список = санитизация)
        self.allowed_html_attrs = {}   # Разрешенные HTML атрибуты

        # Расширенные настройки безопасности
        self.max_urls_per_message = 3  # Максимум URL в одном сообщении
        self.max_email_addresses = 2   # Максимум email адресов
        self.max_phone_numbers = 1     # Максимум номеров телефонов
        self.blocked_words = self._load_blocked_words()
        self.suspicious_patterns = self._load_suspicious_patterns()

    async def preprocess(self, input_text: str) -> PreprocessedInput:
        """Предварительная обработка текста"""
        try:
            # Проверяем входные данные
            if not isinstance(input_text, str):
                raise ValueError("Input must be a string")

            # Нормализация
            normalized = self._normalize_text(input_text)

            # Токенизация
            tokens = self._tokenize(normalized)

            # Извлечение сущностей
            entities = self._extract_entities(tokens)

            # Анализ настроения
            sentiment = await self._analyze_sentiment(normalized)

            # Метаданные
            metadata = {
                'length': len(input_text),
                'normalized_length': len(normalized),
                'token_count': len(tokens),
                'entity_count': len(entities),
                'language': self._detect_language(normalized),
                'readability_score': self._calculate_readability(normalized),
                'processing_time': None  # Можно добавить тайминг
            }

            return PreprocessedInput(
                original=input_text,
                normalized=normalized,
                tokens=tokens,
                entities=entities,
                sentiment=sentiment,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Ошибка предварительной обработки: {e}")
            # Возвращаем минимальный результат в случае ошибки
            return PreprocessedInput(
                original=input_text if isinstance(input_text, str) else str(input_text),
                normalized=input_text if isinstance(input_text, str) else str(input_text),
                tokens=(input_text.split() if isinstance(input_text, str) else [str(input_text)]),
                entities=[],
                sentiment=None,
                metadata={'error': str(e)}
            )

    def _normalize_text(self, text: str) -> str:
        """Нормализация текста"""
        # Удаление лишних пробелов
        normalized = re.sub(r'\s+', ' ', text.strip())

        # Нормализация unicode
        normalized = unicodedata.normalize('NFKC', normalized)

        # Удаление невидимые символы
        normalized = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', normalized)

        # Нормализация кавычек
        normalized = re.sub(r'["""]', '"', normalized)
        normalized = re.sub(r"[''']", "'", normalized)

        # Нормализация тире
        normalized = re.sub(r'[-–—]', '-', normalized)

        return normalized

    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        if NLTK_AVAILABLE:
            try:
                # Используем NLTK для токенизации
                tokens = word_tokenize(text, language='russian' if self._is_russian(text) else 'english')
            except (LookupError, ImportError):
                # Fallback на простую токенизацию
                tokens = self._simple_tokenize(text)
        else:
            # NLTK недоступен, используем fallback
            tokens = self._simple_tokenize(text)

        # Фильтрация пустых токенов
        tokens = [token for token in tokens if token.strip()]

        return tokens

    def _simple_tokenize(self, text: str) -> List[str]:
        """Простая токенизация без NLTK"""
        # Разделение по пробелам и знакам препинания
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _extract_entities(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Извлечение сущностей из токенов"""
        entities = []

        # Простое правило-based извлечение
        for i, token in enumerate(tokens):
            entity = self._classify_token(token, tokens, i)
            if entity:
                entities.append(entity)

        return entities

    def _classify_token(self, token: str, context: List[str], position: int) -> Optional[Dict[str, Any]]:
        """Классификация токена как сущности"""
        token_lower = token.lower()

        # Простые правила для сущностей
        if self._is_potential_name(token):
            return {
                'text': token,
                'type': 'PERSON',
                'confidence': 0.7,
                'position': position
            }

        if self._is_potential_location(token):
            return {
                'text': token,
                'type': 'LOCATION',
                'confidence': 0.6,
                'position': position
            }

        if self._is_potential_organization(token):
            return {
                'text': token,
                'type': 'ORGANIZATION',
                'confidence': 0.5,
                'position': position
            }

        # Числа и даты
        if re.match(r'\d+', token):
            return {
                'text': token,
                'type': 'NUMBER',
                'confidence': 0.9,
                'position': position
            }

        return None

    def _is_potential_name(self, token: str) -> bool:
        """Проверка, может ли токен быть именем"""
        # Простая эвристика: начинается с заглавной буквы, содержит буквы
        return (len(token) > 1 and
                token[0].isupper() and
                any(c.isalpha() for c in token) and
                not token.isupper())  # Не аббревиатура

    def _is_potential_location(self, token: str) -> bool:
        """Проверка, может ли токен быть локацией"""
        # Города, страны (можно расширить)
        locations = {'москва', 'петербург', 'лондон', 'нью-йорк', 'россия', 'сша'}
        return token.lower() in locations

    def _is_potential_organization(self, token: str) -> bool:
        """Проверка, может ли токен быть организацией"""
        # Компании, учреждения
        org_indicators = ['компания', 'корпорация', 'институт', 'университет', 'bank', 'corp']
        token_lower = token.lower()

        return any(indicator in token_lower for indicator in org_indicators)

    async def _analyze_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """Анализ настроения текста"""
        try:
            # Простой анализ настроения на основе словаря
            sentiment_score = self._calculate_sentiment_score(text)

            # Классификация
            if sentiment_score > 0.1:
                sentiment_label = 'positive'
            elif sentiment_score < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'

            return {
                'score': sentiment_score,
                'label': sentiment_label,
                'confidence': min(abs(sentiment_score) * 2, 1.0),
                'method': 'rule_based'
            }

        except Exception as e:
            logger.warning(f"Ошибка анализа настроения: {e}")
            return None

    def _calculate_sentiment_score(self, text: str) -> float:
        """Расчет оценки настроения"""
        positive_words = {
            'хорошо', 'отлично', 'прекрасно', 'замечательно', 'великолепно',
            'good', 'excellent', 'great', 'wonderful', 'fantastic'
        }

        negative_words = {
            'плохо', 'ужасно', 'отвратительно', 'кошмар', 'ужас',
            'bad', 'terrible', 'awful', 'horrible', 'disgusting'
        }

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.0

        return (positive_count - negative_count) / total_sentiment_words

    def _detect_language(self, text: str) -> str:
        """Определение языка текста"""
        cyrillic_count = len(re.findall(r'[\u0400-\u04FF]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))

        if cyrillic_count > latin_count:
            return 'ru'
        elif latin_count > cyrillic_count:
            return 'en'
        else:
            return 'unknown'

    def _is_russian(self, text: str) -> bool:
        """Проверка, является ли текст русским"""
        return self._detect_language(text) == 'ru'

    def sanitize_html(self, text: str) -> str:
        """Санитизация HTML контента"""
        if not BLEACH_AVAILABLE:
            logger.warning("Bleach не доступен, HTML санитизация пропущена")
            return text

        try:
            # Санитизация HTML с bleach
            sanitized = bleach.clean(
                text,
                tags=self.allowed_html_tags,
                attributes=self.allowed_html_attrs,
                strip=True
            )
            return sanitized
        except Exception as e:
            logger.error(f"Ошибка HTML санитизации: {e}")
            return text

    def validate_sql_injection(self, text: str) -> Dict[str, Any]:
        """Проверка на SQL инъекции"""
        result = {
            'is_safe': True,
            'risk_level': 'low',
            'detected_patterns': [],
            'recommendations': []
        }

        if not SQLPARSE_AVAILABLE:
            logger.warning("SQLParse не доступен, SQL валидация ограничена")
            # Базовая проверка без sqlparse
            return self._basic_sql_check(text, result)

        try:
            # Парсинг SQL
            parsed = sqlparse.parse(text)
            if not parsed:
                return result

            # Анализ каждого statement
            for statement in parsed:
                if hasattr(statement, 'tokens'):
                    for token in statement.tokens:
                        if self._is_suspicious_sql_token(str(token)):
                            result['is_safe'] = False
                            result['risk_level'] = 'high'
                            result['detected_patterns'].append(str(token).strip())
                            result['recommendations'].append("Используйте параметризованные запросы")

            return result

        except Exception as e:
            logger.error(f"Ошибка SQL валидации: {e}")
            return self._basic_sql_check(text, result)

    def _basic_sql_check(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Базовая проверка SQL без sqlparse"""
        suspicious_patterns = [
            r';\s*(drop|delete|update|insert|alter)\s+',
            r'union\s+select',
            r'--\s*$',
            r'/\*.*\*/',
            r';\s*$'
        ]

        text_lower = text.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                result['is_safe'] = False
                result['risk_level'] = 'medium'
                result['detected_patterns'].append(pattern)
                result['recommendations'].append("Избегайте прямых SQL конструкций в пользовательском вводе")

        return result

    def _is_suspicious_sql_token(self, token: str) -> bool:
        """Проверка токена на подозрительность"""
        suspicious_keywords = {
            'drop', 'delete', 'update', 'insert', 'alter', 'create',
            'union', 'select', 'exec', 'execute', 'script', 'eval'
        }

        token_lower = token.lower().strip()
        return token_lower in suspicious_keywords

    def validate_input_length(self, text: str) -> Dict[str, Any]:
        """Валидация длины входного текста"""
        length = len(text)

        result = {
            'is_valid': True,
            'length': length,
            'issues': [],
            'recommendations': []
        }

        if length < self.min_input_length:
            result['is_valid'] = False
            result['issues'].append(f"Текст слишком короткий (минимум {self.min_input_length} символов)")
            result['recommendations'].append("Увеличьте длину запроса")

        if length > self.max_input_length:
            result['is_valid'] = False
            result['issues'].append(f"Текст слишком длинный (максимум {self.max_input_length} символов)")
            result['recommendations'].append("Сократите длину запроса")

        return result

    def validate_content(self, text: str) -> Dict[str, Any]:
        """Комплексная валидация контента"""
        result = {
            'is_valid': True,
            'checks': {},
            'issues': [],
            'recommendations': []
        }

        # Проверка длины
        length_check = self.validate_input_length(text)
        result['checks']['length'] = length_check
        if not length_check['is_valid']:
            result['is_valid'] = False
            result['issues'].extend(length_check['issues'])
            result['recommendations'].extend(length_check['recommendations'])

        # Проверка HTML/XSS
        html_check = self._check_html_content(text)
        result['checks']['html'] = html_check
        if not html_check['is_safe']:
            result['is_valid'] = False
            result['issues'].extend(html_check['issues'])
            result['recommendations'].extend(html_check['recommendations'])

        # Проверка SQL инъекций
        sql_check = self.validate_sql_injection(text)
        result['checks']['sql'] = sql_check
        if not sql_check['is_safe']:
            result['is_valid'] = False
            result['issues'].extend([f"Обнаружены подозрительные SQL паттерны: {', '.join(sql_check['detected_patterns'])}"])
            result['recommendations'].extend(sql_check['recommendations'])

        return result

    def _check_html_content(self, text: str) -> Dict[str, Any]:
        """Проверка HTML контента на XSS"""
        result = {
            'is_safe': True,
            'issues': [],
            'recommendations': []
        }

        # Поиск подозрительных HTML паттернов
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>'
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                result['is_safe'] = False
                result['issues'].append(f"Обнаружен подозрительный HTML: {pattern}")
                result['recommendations'].append("Удалите HTML теги или используйте санитизацию")

        return result

    def secure_preprocess(self, input_text: str) -> Dict[str, Any]:
        """Безопасная предварительная обработка с валидацией"""
        result = {
            'success': False,
            'data': None,
            'validation': None,
            'error': None
        }

        try:
            # Валидация
            validation = self.validate_content(input_text)
            result['validation'] = validation

            if not validation['is_valid']:
                result['error'] = 'Валидация не пройдена'
                return result

            # Санитизация
            sanitized = self.sanitize_html(input_text)

            # Предварительная обработка
            processed = self.preprocess(sanitized)
            result['data'] = processed
            result['success'] = True

        except Exception as e:
            logger.error(f"Ошибка безопасной обработки: {e}")
            result['error'] = str(e)

        return result

    def _calculate_readability(self, text: str) -> float:
        """Расчет оценки читаемости текста"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        words = text.split()
        word_count = len(words)
        sentence_count = len(sentences)

        if sentence_count == 0 or word_count == 0:
            return 0.0

        # Средняя длина предложения
        avg_sentence_length = word_count / sentence_count

        # Средняя длина слова
        avg_word_length = sum(len(word) for word in words) / word_count

        # Простая формула читаемости (чем меньше - тем лучше читаемость)
        readability_score = avg_sentence_length * 0.4 + avg_word_length * 0.6

        # Нормализация к шкале 0-1 (где 1 - легко читаемый)
        normalized_score = max(0, min(1, 2 - readability_score / 20))

        return normalized_score

    def _load_blocked_words(self) -> set:
        """Загрузка списка заблокированных слов"""
        # Базовый набор заблокированных слов
        blocked = {
            # Плохие слова
            'fuck', 'shit', 'damn', 'bitch', 'asshole', 'bastard',
            'сука', 'пизда', 'хуй', 'ебать', 'блядь', 'мудак',
            # Спам слова
            'viagra', 'casino', 'lottery', 'winner', 'prize',
            # Другие подозрительные слова
            'password', 'admin', 'root', 'hack', 'exploit'
        }
        return blocked

    def _load_suspicious_patterns(self) -> List[str]:
        """Загрузка подозрительных паттернов"""
        return [
            # Слишком много повторяющихся символов
            r'(.)\1{10,}',  # 10+ повторяющихся символов подряд
            # Слишком много заглавных букв
            r'[A-ZА-Я]{20,}',  # 20+ заглавных букв подряд
            # Подозрительные комбинации
            r'(?i)(test|demo|fake|spam)\s*\d{5,}',  # test/demo + числа
            # Скрытый текст
            r'[\u200B-\u200F\u202A-\u202E\uFEFF]',  # Невидимые символы Unicode
        ]

    def validate_security(self, text: str) -> Dict[str, Any]:
        """Расширенная проверка безопасности"""
        result = {
            'is_safe': True,
            'risk_level': 'low',
            'issues': [],
            'recommendations': [],
            'checks': {}
        }

        # Проверка на заблокированные слова
        blocked_check = self._check_blocked_words(text)
        result['checks']['blocked_words'] = blocked_check
        if not blocked_check['is_safe']:
            result['is_safe'] = False
            result['risk_level'] = max(result['risk_level'], blocked_check['risk_level'])
            result['issues'].extend(blocked_check['issues'])
            result['recommendations'].extend(blocked_check['recommendations'])

        # Проверка на подозрительные паттерны
        pattern_check = self._check_suspicious_patterns(text)
        result['checks']['suspicious_patterns'] = pattern_check
        if not pattern_check['is_safe']:
            result['is_safe'] = False
            result['risk_level'] = max(result['risk_level'], pattern_check['risk_level'])
            result['issues'].extend(pattern_check['issues'])
            result['recommendations'].extend(pattern_check['recommendations'])

        # Проверка на URL
        url_check = self._check_urls(text)
        result['checks']['urls'] = url_check
        if not url_check['is_safe']:
            result['is_safe'] = False
            result['risk_level'] = max(result['risk_level'], url_check['risk_level'])
            result['issues'].extend(url_check['issues'])
            result['recommendations'].extend(url_check['recommendations'])

        # Проверка на email адреса
        email_check = self._check_emails(text)
        result['checks']['emails'] = email_check
        if not email_check['is_safe']:
            result['is_safe'] = False
            result['risk_level'] = max(result['risk_level'], email_check['risk_level'])
            result['issues'].extend(email_check['issues'])
            result['recommendations'].extend(email_check['recommendations'])

        # Проверка на номера телефонов
        phone_check = self._check_phone_numbers(text)
        result['checks']['phone_numbers'] = phone_check
        if not phone_check['is_safe']:
            result['is_safe'] = False
            result['risk_level'] = max(result['risk_level'], phone_check['risk_level'])
            result['issues'].extend(phone_check['issues'])
            result['recommendations'].extend(phone_check['recommendations'])

        # Определение общего уровня риска
        risk_levels = []
        for check in result['checks'].values():
            if isinstance(check, dict) and 'risk_level' in check:
                risk_levels.append(check['risk_level'])

        if 'high' in risk_levels:
            result['risk_level'] = 'high'
        elif 'medium' in risk_levels:
            result['risk_level'] = 'medium'
        else:
            result['risk_level'] = 'low'

        return result

    def _check_blocked_words(self, text: str) -> Dict[str, Any]:
        """Проверка на заблокированные слова"""
        result = {
            'is_safe': True,
            'risk_level': 'low',
            'issues': [],
            'recommendations': [],
            'found_words': []
        }

        text_lower = text.lower()
        found_blocked = []

        for word in self.blocked_words:
            if word in text_lower:
                found_blocked.append(word)
                result['is_safe'] = False
                result['risk_level'] = 'high'
                result['issues'].append(f"Найдено заблокированное слово: {word}")
                result['recommendations'].append("Удалите нецензурную лексику")

        result['found_words'] = found_blocked
        return result

    def _check_suspicious_patterns(self, text: str) -> Dict[str, Any]:
        """Проверка на подозрительные паттерны"""
        result = {
            'is_safe': True,
            'risk_level': 'low',
            'issues': [],
            'recommendations': [],
            'matched_patterns': []
        }

        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result['is_safe'] = False
                result['risk_level'] = 'medium'
                result['issues'].append(f"Обнаружен подозрительный паттерн: {pattern}")
                result['recommendations'].append("Проверьте текст на спам или вредоносное содержимое")
                result['matched_patterns'].append(pattern)

        return result

    def _check_urls(self, text: str) -> Dict[str, Any]:
        """Проверка URL в тексте"""
        result = {
            'is_safe': True,
            'risk_level': 'low',
            'issues': [],
            'recommendations': [],
            'url_count': 0,
            'urls': []
        }

        # Поиск URL
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        result['url_count'] = len(urls)
        result['urls'] = urls

        if len(urls) > self.max_urls_per_message:
            result['is_safe'] = False
            result['risk_level'] = 'medium'
            result['issues'].append(f"Слишком много URL ({len(urls)} > {self.max_urls_per_message})")
            result['recommendations'].append("Ограничьте количество ссылок в сообщении")

        # Проверка подозрительных URL
        suspicious_domains = ['.ru', '.cn', '.tk', 'bit.ly', 'tinyurl.com']
        for url in urls:
            for domain in suspicious_domains:
                if domain in url.lower():
                    result['is_safe'] = False
                    result['risk_level'] = 'high'
                    result['issues'].append(f"Подозрительный URL: {url}")
                    result['recommendations'].append("Избегайте подозрительных ссылок")

        return result

    def _check_emails(self, text: str) -> Dict[str, Any]:
        """Проверка email адресов в тексте"""
        result = {
            'is_safe': True,
            'risk_level': 'low',
            'issues': [],
            'recommendations': [],
            'email_count': 0,
            'emails': []
        }

        # Поиск email адресов
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        result['email_count'] = len(emails)
        result['emails'] = emails

        if len(emails) > self.max_email_addresses:
            result['is_safe'] = False
            result['risk_level'] = 'medium'
            result['issues'].append(f"Слишком много email адресов ({len(emails)} > {self.max_email_addresses})")
            result['recommendations'].append("Ограничьте количество email адресов")

        return result

    def _check_phone_numbers(self, text: str) -> Dict[str, Any]:
        """Проверка номеров телефонов в тексте"""
        result = {
            'is_safe': True,
            'risk_level': 'low',
            'issues': [],
            'recommendations': [],
            'phone_count': 0,
            'phones': []
        }

        # Поиск номеров телефонов (простой паттерн)
        phone_pattern = r'\b[\+]?[1-9][\d]{9,14}\b'
        phones = re.findall(phone_pattern, text)
        result['phone_count'] = len(phones)
        result['phones'] = phones

        if len(phones) > self.max_phone_numbers:
            result['is_safe'] = False
            result['risk_level'] = 'medium'
            result['issues'].append(f"Слишком много номеров телефонов ({len(phones)} > {self.max_phone_numbers})")
            result['recommendations'].append("Ограничьте количество номеров телефонов")

        return result

    def validate_comprehensive(self, text: str) -> Dict[str, Any]:
        """Комплексная валидация с проверками безопасности"""
        result = {
            'is_valid': True,
            'is_safe': True,
            'risk_level': 'low',
            'validation': {},
            'security': {},
            'issues': [],
            'recommendations': [],
            'sanitized_text': None
        }

        # Базовая валидация контента
        content_validation = self.validate_content(text)
        result['validation'] = content_validation

        if not content_validation['is_valid']:
            result['is_valid'] = False
            result['issues'].extend(content_validation['issues'])
            result['recommendations'].extend(content_validation['recommendations'])

        # Проверка безопасности
        security_check = self.validate_security(text)
        result['security'] = security_check

        if not security_check['is_safe']:
            result['is_safe'] = False
            result['issues'].extend(security_check['issues'])
            result['recommendations'].extend(security_check['recommendations'])

        # Определение общего уровня риска и статуса безопасности
        risk_levels = []
        if 'risk_level' in content_validation:
            risk_levels.append(content_validation['risk_level'])
        risk_levels.append(security_check['risk_level'])

        if 'high' in risk_levels:
            result['risk_level'] = 'high'
        elif 'medium' in risk_levels:
            result['risk_level'] = 'medium'
        else:
            result['risk_level'] = 'low'

        # Обновление общего статуса - объединяем все проверки
        result['is_safe'] = content_validation.get('is_safe', True) and security_check['is_safe']
        result['is_valid'] = result['is_valid'] and result['is_safe']

        # Санитизация текста
        if result['is_valid']:
            result['sanitized_text'] = self.sanitize_html(text)

        return result

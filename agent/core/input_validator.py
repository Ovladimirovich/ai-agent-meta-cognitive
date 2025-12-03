"""
Input Validator - Валидация входных данных агента
"""

import re
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Результат валидации входных данных"""
    is_valid: bool = True
    errors: List[str] = []
    warnings: List[str] = []
    quality_score: float = 1.0
    metadata: Dict[str, Any] = {}


class InputValidator:
    """Валидатор входных данных агента"""

    def __init__(self):
        self.max_length = 10000
        self.min_length = 1
        self.supported_languages = ['ru', 'en']
        self.forbidden_patterns = [
            r'<script[^>]*>.*?</script>',  # Скрипты
            r'javascript:',                # JavaScript URLs
            r'on\w+\s*=',                  # Event handlers
            r'<iframe[^>]*>.*?</iframe>',  # Iframes
            r'<object[^>]*>.*?</object>',  # Objects
            r'<embed[^>]*>.*?</embed>',    # Embeds
            r'data:text/html',             # Data URLs
        ]

        # Компиляция паттернов для производительности
        self._compiled_forbidden = [re.compile(pattern, re.IGNORECASE | re.DOTALL)
                                   for pattern in self.forbidden_patterns]

    async def validate(self, input_text: str) -> ValidationResult:
        """Валидация входных данных"""
        result = ValidationResult()

        try:
            # Проверка длины
            length_valid, length_error = self._validate_length(input_text)
            if not length_valid:
                result.errors.append(length_error)
                result.is_valid = False

            # Проверка безопасности
            security_issues = self._check_security(input_text)
            if security_issues:
                result.errors.extend(security_issues)
                result.is_valid = False

            # Проверка языка (опционально)
            lang_warning = self._check_language(input_text)
            if lang_warning:
                result.warnings.append(lang_warning)

            # Оценка качества
            result.quality_score = self._assess_quality(input_text)

            # Метаданные
            result.metadata = {
                'length': len(input_text),
                'word_count': len(input_text.split()),
                'has_special_chars': bool(re.search(r'[^\w\s\u0400-\u04FF]', input_text)),  # Кириллица
                'validation_time': None  # Можно добавить тайминг
            }

        except Exception as e:
            logger.error(f"Ошибка валидации: {e}")
            result.errors.append(f"Ошибка валидации: {str(e)}")
            result.is_valid = False

        return result

    def _validate_length(self, text: str) -> tuple[bool, Optional[str]]:
        """Проверка длины текста"""
        length = len(text.strip())

        if length < self.min_length:
            return False, f"Текст слишком короткий: {length} символов (минимум {self.min_length})"

        if length > self.max_length:
            return False, f"Текст слишком длинный: {length} символов (максимум {self.max_length})"

        return True, None

    def _check_security(self, text: str) -> List[str]:
        """Проверка безопасности текста"""
        issues = []

        for pattern in self._compiled_forbidden:
            if pattern.search(text):
                issues.append(f"Обнаружен потенциально опасный контент: {pattern.pattern}")

        # Дополнительные проверки
        if self._has_suspicious_urls(text):
            issues.append("Обнаружены подозрительные URL")

        if self._has_malformed_encoding(text):
            issues.append("Обнаружена некорректная кодировка")

        return issues

    def _check_language(self, text: str) -> Optional[str]:
        """Проверка языка текста"""
        # Простая эвристика для определения языка
        cyrillic_count = len(re.findall(r'[\u0400-\u04FF]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))

        total_alpha = cyrillic_count + latin_count

        if total_alpha == 0:
            return None  # Нет букв

        # Если больше 80% кириллицы или латиницы - считаем основным языком
        if cyrillic_count / total_alpha > 0.8:
            if 'ru' not in self.supported_languages:
                return "Текст преимущественно на русском языке, но русский не поддерживается"
        elif latin_count / total_alpha > 0.8:
            if 'en' not in self.supported_languages:
                return "Текст преимущественно на английском языке, но английский не поддерживается"

        return None

    def _assess_quality(self, text: str) -> float:
        """Оценка качества текста"""
        score = 1.0

        # Факторы снижения качества
        factors = {
            'too_short': len(text.strip()) < 10,
            'too_long': len(text.strip()) > 5000,
            'repetitive': self._is_repetitive(text),
            'poor_structure': not self._has_good_structure(text),
            'mixed_languages': self._has_mixed_languages(text)
        }

        penalties = {
            'too_short': 0.3,
            'too_long': 0.2,
            'repetitive': 0.4,
            'poor_structure': 0.2,
            'mixed_languages': 0.3
        }

        for factor, applies in factors.items():
            if applies:
                score -= penalties[factor]

        return max(0.0, score)

    def _has_suspicious_urls(self, text: str) -> bool:
        """Проверка на подозрительные URL"""
        url_pattern = r'https?://[^\s<>"]+'
        urls = re.findall(url_pattern, text)

        suspicious_indicators = [
            'javascript:', 'data:', 'vbscript:', 'file:',
            'localhost', '127.0.0.1', '0.0.0.0'
        ]

        for url in urls:
            for indicator in suspicious_indicators:
                if indicator in url.lower():
                    return True

        return False

    def _has_malformed_encoding(self, text: str) -> bool:
        """Проверка на некорректную кодировку"""
        # Проверка на broken UTF-8 последовательности
        try:
            text.encode('utf-8').decode('utf-8')
            return False
        except UnicodeDecodeError:
            return True

    def _is_repetitive(self, text: str) -> bool:
        """Проверка на повторяющийся текст"""
        words = text.lower().split()
        if len(words) < 5:
            return False

        # Проверка на повторение слов
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Только значимые слова
                word_counts[word] = word_counts.get(word, 0) + 1

        # Если какое-то слово повторяется более 30% от общего числа слов
        total_words = len([w for w in words if len(w) > 3])
        for count in word_counts.values():
            if count / total_words > 0.3:
                return True

        return False

    def _has_good_structure(self, text: str) -> bool:
        """Проверка на хорошую структуру текста"""
        # Проверка на наличие предложений, знаков препинания
        sentences = re.split(r'[.!?]+', text.strip())
        return len(sentences) > 1 and any('?' in s or '!' in s for s in sentences)

    def _has_mixed_languages(self, text: str) -> bool:
        """Проверка на смешанные языки"""
        cyrillic = bool(re.search(r'[\u0400-\u04FF]', text))
        latin = bool(re.search(r'[a-zA-Z]', text))

        return cyrillic and latin

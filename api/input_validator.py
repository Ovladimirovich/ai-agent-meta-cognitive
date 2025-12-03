"""
Input Validator - Валидация входных данных для API
Использует InputPreprocessor для комплексной проверки безопасности
"""

import logging
import re
import io
from typing import Dict, List, Optional, Any, Union
from fastapi import HTTPException, status
from pydantic import BaseModel, ValidationError

from agent.core.input_preprocessor import InputPreprocessor

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Результат валидации"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_input: Optional[str] = None
    validation_details: Optional[Dict[str, Any]] = None


class InputValidator:
    """Валидатор входных данных для API"""

    def __init__(self):
        self.preprocessor = InputPreprocessor()

    async def validate_agent_request(self, query: str, user_id: Optional[str] = None) -> ValidationResult:
        """
        Валидация запроса к агенту

        Args:
            query: Текст запроса
            user_id: ID пользователя (опционально)

        Returns:
            ValidationResult: Результат валидации
        """
        errors = []
        warnings = []

        try:
            # Проверка типа данных
            if not isinstance(query, str):
                errors.append("Query must be a string")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings
                )

            # Комплексная валидация с проверками безопасности
            comprehensive_result = self.preprocessor.validate_comprehensive(query)

            if not comprehensive_result['is_valid']:
                errors.extend(comprehensive_result['issues'])
                warnings.extend(comprehensive_result['recommendations'])

                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    validation_details=comprehensive_result
                )

            if not comprehensive_result['is_safe']:
                errors.extend(comprehensive_result['issues'])
                warnings.extend(comprehensive_result['recommendations'])

                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    validation_details=comprehensive_result
                )

            # Проверка специфичных для агента правил
            agent_validation = self._validate_agent_specific_rules(query)
            if not agent_validation['is_valid']:
                errors.extend(agent_validation['errors'])
                warnings.extend(agent_validation['warnings'])

            # Санитизированный ввод
            sanitized = comprehensive_result.get('sanitized_text') or self.preprocessor.sanitize_html(query)

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_input=sanitized,
                validation_details=comprehensive_result
            )

        except Exception as e:
            logger.error(f"Validation error: {e}")
            errors.append(f"Internal validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings
            )

    def _validate_agent_specific_rules(self, query: str) -> Dict[str, Any]:
        """Проверка специфичных для агента правил"""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Проверка на пустой запрос после санитизации
        sanitized = self.preprocessor.sanitize_html(query)
        if not sanitized.strip():
            result['is_valid'] = False
            result['errors'].append("Query is empty after sanitization")

        # Проверка на слишком короткие запросы
        if len(query.strip()) < 3:
            result['is_valid'] = False
            result['errors'].append("Query is too short (minimum 3 characters)")

        # Проверка на спам (повторяющиеся символы)
        if self._is_spam_like(query):
            result['warnings'].append("Query appears to be spam-like")

        # Проверка на запрещенные слова/фразы
        forbidden_patterns = self._check_forbidden_content(query)
        if forbidden_patterns:
            result['is_valid'] = False
            result['errors'].append(f"Forbidden content detected: {', '.join(forbidden_patterns)}")

        return result

    def _is_spam_like(self, text: str) -> bool:
        """Проверка на спам-подобный контент"""
        # Простая эвристика для спама
        if len(text) > 100:
            # Считаем повторяющиеся символы
            char_counts = {}
            for char in text.lower():
                if char.isalnum():
                    char_counts[char] = char_counts.get(char, 0) + 1

            # Если какой-то символ повторяется более 20% от длины текста
            max_count = max(char_counts.values()) if char_counts else 0
            if max_count > len(text) * 0.2:
                return True

        return False

    def _check_forbidden_content(self, text: str) -> List[str]:
        """Проверка на запрещенный контент"""
        forbidden_patterns = [
            r'\b(?:fuck|shit|damn|bitch|asshole)\b',  # Плохие слова
            r'(?:http|https|ftp)://[^\s]+',  # URL (может быть спамом)
            r'\b\d{10,}\b',  # Длинные числа (может быть номером телефона)
        ]

        found = []
        text_lower = text.lower()

        for pattern in forbidden_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                found.append(pattern)

        return found

    async def validate_file_upload(self, filename: str, content: bytes, max_size: int = 10*1024*1024) -> ValidationResult:
        """
        Валидация загружаемого файла

        Args:
            filename: Имя файла
            content: Содержимое файла
            max_size: Максимальный размер в байтах

        Returns:
            ValidationResult: Результат валидации
        """
        errors = []
        warnings = []

        try:
            # Проверка размера файла
            if len(content) > max_size:
                errors.append(f"File too large (max {max_size} bytes)")

            # Проверка имени файла
            if not self._is_safe_filename(filename):
                errors.append("Unsafe filename")

            # Проверка типа файла
            file_type = self._get_file_type(filename, content)
            if not self._is_allowed_file_type(file_type):
                errors.append(f"File type not allowed: {file_type}")

            # Для текстовых файлов - дополнительная проверка контента
            if file_type in ['text', 'pdf', 'docx']:
                text_content = self._extract_text_content(content, file_type)
                if text_content:
                    text_validation = await self.validate_agent_request(text_content)
                    if not text_validation.is_valid:
                        errors.extend([f"File content: {err}" for err in text_validation.errors])
                        warnings.extend([f"File content: {warn}" for warn in text_validation.warnings])

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"File validation error: {e}")
            errors.append(f"File validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings
            )

    def _is_safe_filename(self, filename: str) -> bool:
        """Проверка безопасного имени файла"""
        import os

        # Проверка на опасные символы
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '*', '?', '"', '|']
        if any(char in filename for char in dangerous_chars):
            return False

        # Проверка расширения
        _, ext = os.path.splitext(filename)
        allowed_extensions = ['.txt', '.pdf', '.docx', '.md', '.json', '.csv']

        return ext.lower() in allowed_extensions

    def _get_file_type(self, filename: str, content: bytes) -> str:
        """Определение типа файла"""
        import os

        # По расширению
        _, ext = os.path.splitext(filename)

        # По сигнатуре (magic bytes)
        if content.startswith(b'%PDF'):
            return 'pdf'
        elif content.startswith(b'PK\x03\x04'):  # ZIP signature (DOCX is ZIP)
            return 'docx'
        elif content.startswith(b'\x50\x4b\x03\x04'):  # Alternative ZIP
            return 'docx'
        elif ext.lower() in ['.txt', '.md', '.json', '.csv']:
            return 'text'
        else:
            return 'unknown'

    def _is_allowed_file_type(self, file_type: str) -> bool:
        """Проверка разрешенного типа файла"""
        allowed_types = ['text', 'pdf', 'docx']
        return file_type in allowed_types

    def _extract_text_content(self, content: bytes, file_type: str) -> Optional[str]:
        """Извлечение текстового контента из файла"""
        try:
            if file_type == 'text':
                return content.decode('utf-8', errors='ignore')
            elif file_type == 'pdf':
                # Для PDF используем PyPDF2 если доступен
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    return text
                except ImportError:
                    logger.warning("PyPDF2 not available for PDF text extraction")
                    return None
            elif file_type == 'docx':
                # Для DOCX используем python-docx если доступен
                try:
                    import docx
                    doc = docx.Document(io.BytesIO(content))
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    logger.warning("python-docx not available for DOCX text extraction")
                    return None
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")

        return None

    def create_validation_error(self, result: ValidationResult) -> HTTPException:
        """Создание HTTP исключения из результата валидации"""
        if result.is_valid:
            return None

        error_message = "; ".join(result.errors)
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Validation failed",
                "message": error_message,
                "warnings": result.warnings,
                "details": result.validation_details
            }
        )


# Глобальный экземпляр валидатора
input_validator = InputValidator()


# Dependency для FastAPI
async def validate_query(query: str) -> str:
    """FastAPI dependency для валидации запроса"""
    result = await input_validator.validate_agent_request(query)

    if not result.is_valid:
        raise input_validator.create_validation_error(result)

    # Возвращаем санитизированный ввод
    return result.sanitized_input or query

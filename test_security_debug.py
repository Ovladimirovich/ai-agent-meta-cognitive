#!/usr/bin/env python3
"""
Отладка системы безопасности
"""

from agent.core.input_preprocessor import InputPreprocessor
from api.input_validator import InputValidator

def main():
    # Тест базовой функциональности
    preprocessor = InputPreprocessor()
    validator = InputValidator()

    print('=== ТЕСТ ЗАБЛОКИРОВАННЫХ СЛОВ ===')
    blocked_text = 'This is fucking awesome'
    security = preprocessor.validate_security(blocked_text)
    print(f'Текст: {blocked_text}')
    print(f'Безопасно: {security["is_safe"]}')
    print(f'Уровень риска: {security["risk_level"]}')
    print(f'Найденные слова: {security["checks"]["blocked_words"]["found_words"]}')

    print('\n=== ТЕСТ SQL ИНЪЕКЦИИ ===')
    sql_text = "'; DROP TABLE users; --"
    validation = preprocessor.validate_content(sql_text)
    print(f'Текст: {sql_text}')
    print(f'Валидно: {validation["is_valid"]}')
    print(f'Проверки SQL: {validation["checks"]["sql"]}')

    print('\n=== ТЕСТ КОМПЛЕКСНОЙ ВАЛИДАЦИИ ===')
    dangerous = "<script>alert('xss')</script> ' OR 1=1; --"
    comprehensive = preprocessor.validate_comprehensive(dangerous)
    print(f'Текст: {dangerous}')
    print(f'Валидно: {comprehensive["is_valid"]}')
    print(f'Безопасно: {comprehensive["is_safe"]}')
    print(f'Уровень риска: {comprehensive["risk_level"]}')

    print('\n=== ТЕСТ XSS ===')
    xss_text = "<script>alert('xss')</script>"
    xss_validation = preprocessor.validate_content(xss_text)
    print(f'Текст: {xss_text}')
    print(f'Валидно: {xss_validation["is_valid"]}')
    print(f'HTML проверки: {xss_validation["checks"]["html"]}')

if __name__ == "__main__":
    main()

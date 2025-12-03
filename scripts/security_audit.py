#!/usr/bin/env python3
"""
Security Audit Script для AI Agent Meta-Cognitive
Проверяет безопасность системы на уязвимости
"""

import asyncio
import json
import time
from typing import Dict, List, Any
import httpx
import os
from pathlib import Path

# Импорты проекта
from agent.core.input_preprocessor import InputPreprocessor
from api.input_validator import InputValidator
from api.logging_config import SensitiveDataFilter


class SecurityAuditor:
    """Аудитор безопасности системы"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "timestamp": time.time(),
            "audit_type": "security_audit",
            "vulnerabilities": [],
            "warnings": [],
            "passed_checks": [],
            "recommendations": []
        }

    async def run_full_audit(self) -> Dict[str, Any]:
        """Запуск полного аудита безопасности"""
        print("🔒 Начинаем security audit...")

        # Аудит компонентов
        await self.audit_input_validation()
        await self.audit_authentication()
        await self.audit_logging()
        await self.audit_dependencies()

        # Генерация отчета
        self.generate_report()

        return self.results

    async def audit_input_validation(self):
        """Аудит валидации входных данных"""
        print("📝 Аудит валидации входных данных...")

        validator = InputValidator()
        preprocessor = InputPreprocessor()

        # XSS тесты
        xss_payloads = [
            '<script>alert("xss")</script>',
            '<img src=x onerror=alert(1)>',
            'javascript:alert("xss")',
            '<iframe src="javascript:alert(1)"></iframe>'
        ]

        for payload in xss_payloads:
            result = await validator.validate_agent_request(payload)
            if result.is_valid:
                self.results["vulnerabilities"].append({
                    "type": "XSS",
                    "severity": "HIGH",
                    "payload": payload,
                    "description": "XSS payload прошел валидацию"
                })
            else:
                self.results["passed_checks"].append(f"XSS blocked: {payload[:30]}...")

        # SQL injection тесты
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users; --",
            "1' OR '1'='1",
            "'; EXEC xp_cmdshell 'dir'; --"
        ]

        for payload in sql_payloads:
            result = preprocessor.validate_sql_injection(payload)
            if result['is_safe']:
                self.results["vulnerabilities"].append({
                    "type": "SQL_INJECTION",
                    "severity": "CRITICAL",
                    "payload": payload,
                    "description": "SQL injection payload не обнаружен"
                })
            else:
                self.results["passed_checks"].append(f"SQLi detected: {payload[:30]}...")

        # Тесты длины
        long_payload = "a" * 15000
        result = await validator.validate_agent_request(long_payload)
        if result.is_valid:
            self.results["vulnerabilities"].append({
                "type": "DOS",
                "severity": "MEDIUM",
                "description": "Слишком длинный ввод прошел валидацию"
            })

    async def audit_authentication(self):
        """Аудит аутентификации"""
        print("🔐 Аудит аутентификации...")

        async with httpx.AsyncClient() as client:
            # Тест без токена
            response = await client.get(f"{self.base_url}/agent/process")
            if response.status_code != 401:
                self.results["warnings"].append({
                    "type": "AUTH_BYPASS",
                    "description": "Эндпоинт доступен без аутентификации"
                })

            # Тест с неправильным токеном
            headers = {"Authorization": "Bearer invalid_token"}
            response = await client.get(f"{self.base_url}/agent/process", headers=headers)
            if response.status_code != 401:
                self.results["vulnerabilities"].append({
                    "type": "AUTH_BYPASS",
                    "severity": "HIGH",
                    "description": "Неправильный токен принят"
                })

    async def audit_logging(self):
        """Аудит логирования"""
        print("📋 Аудит логирования...")

        filter = SensitiveDataFilter()

        # Тест фильтрации чувствительных данных
        test_data = {
            "password": "secret123",
            "api_key": "sk-1234567890abcdef",
            "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9",
            "normal_field": "normal_value"
        }

        filtered = filter(None, None, test_data)

        if filtered["password"] != "***MASKED***":
            self.results["vulnerabilities"].append({
                "type": "LOG_LEAK",
                "severity": "HIGH",
                "description": "Пароли не маскируются в логах"
            })

        if filtered["api_key"] != "***MASKED***":
            self.results["vulnerabilities"].append({
                "type": "LOG_LEAK",
                "severity": "HIGH",
                "description": "API ключи не маскируются в логах"
            })

        if filtered["token"] == "***MASKED***":
            self.results["passed_checks"].append("JWT tokens masked in logs")

    async def audit_dependencies(self):
        """Аудит зависимостей"""
        print("📦 Аудит зависимостей...")

        # Проверка requirements.txt
        req_file = Path("requirements.txt")
        if not req_file.exists():
            self.results["vulnerabilities"].append({
                "type": "DEPENDENCY",
                "severity": "MEDIUM",
                "description": "requirements.txt не найден"
            })
            return

        with open(req_file, 'r') as f:
            content = f.read()

        # Проверка на устаревшие версии
        vulnerable_patterns = [
            "fastapi==0.6",  # Устаревшая версия
            "cryptography==3.",  # Слишком старая
        ]

        for pattern in vulnerable_patterns:
            if pattern in content:
                self.results["warnings"].append({
                    "type": "OUTDATED_DEPENDENCY",
                    "description": f"Обнаружена устаревшая зависимость: {pattern}"
                })

        # Проверка на небезопасные пакеты
        if "insecure-package" in content:  # Пример
            self.results["vulnerabilities"].append({
                "type": "INSECURE_DEPENDENCY",
                "severity": "CRITICAL",
                "description": "Обнаружен небезопасный пакет"
            })

    def generate_report(self):
        """Генерация отчета аудита"""
        print("📊 Генерация отчета аудита...")

        # Статистика
        total_vulns = len(self.results["vulnerabilities"])
        total_warnings = len(self.results["warnings"])
        total_passed = len(self.results["passed_checks"])

        self.results["summary"] = {
            "total_vulnerabilities": total_vulns,
            "total_warnings": total_warnings,
            "total_passed_checks": total_passed,
            "audit_score": max(0, 100 - (total_vulns * 20) - (total_warnings * 5))
        }

        # Рекомендации
        if total_vulns > 0:
            self.results["recommendations"].extend([
                "Исправить все найденные уязвимости",
                "Провести повторный аудит после исправлений",
                "Рассмотреть использование WAF (Web Application Firewall)",
                "Регулярно обновлять зависимости"
            ])

        if total_warnings > 0:
            self.results["recommendations"].append("Обратить внимание на предупреждения")

        # Сохранение отчета
        report_file = f"security_audit_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"✅ Отчет сохранен: {report_file}")

        # Вывод результатов
        print(f"\n🔒 Результаты аудита:")
        print(f"   Уязвимости: {total_vulns}")
        print(f"   Предупреждения: {total_warnings}")
        print(f"   Пройденные проверки: {total_passed}")
        print(f"   Оценка безопасности: {self.results['summary']['audit_score']}/100")


async def run_performance_test():
    """Базовое тестирование производительности"""
    print("⚡ Запуск performance testing...")

    results = {
        "timestamp": time.time(),
        "test_type": "performance_baseline",
        "metrics": {}
    }

    # Импорт компонентов для тестирования
    from agent.core.input_preprocessor import InputPreprocessor
    from api.input_validator import InputValidator

    preprocessor = InputPreprocessor()
    validator = InputValidator()

    # Тест скорости валидации
    test_queries = [
        "Hello world",
        "What is the capital of France?",
        "Please analyze this long text " * 50,
        '<script>alert("xss")</script>',
        "'; DROP TABLE users; --"
    ]

    validation_times = []
    preprocessing_times = []

    for query in test_queries:
        # Тест валидации
        start_time = time.time()
        result = await validator.validate_agent_request(query)
        validation_times.append(time.time() - start_time)

        # Тест preprocessing
        start_time = time.time()
        processed = await preprocessor.preprocess(query)
        preprocessing_times.append(time.time() - start_time)

    # Статистика
    results["metrics"] = {
        "validation_avg_time": sum(validation_times) / len(validation_times),
        "validation_max_time": max(validation_times),
        "validation_min_time": min(validation_times),
        "preprocessing_avg_time": sum(preprocessing_times) / len(preprocessing_times),
        "preprocessing_max_time": max(preprocessing_times),
        "preprocessing_min_time": min(preprocessing_times),
        "total_tests": len(test_queries)
    }

    # Сохранение результатов
    perf_file = f"performance_test_{int(time.time())}.json"
    with open(perf_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Результаты производительности сохранены: {perf_file}")
    print(".4f"    print(".4f"
    return results


async def main():
    """Основная функция"""
    print("🚀 Запуск Security Audit и Performance Testing\n")

    # Security Audit
    auditor = SecurityAuditor()
    audit_results = await auditor.run_full_audit()

    print("\n" + "="*50)

    # Performance Test
    perf_results = await run_performance_test()

    print("\n" + "="*50)
    print("🎯 Аудит завершен!")

    # Финальная оценка
    audit_score = audit_results["summary"]["audit_score"]
    perf_score = 100 if perf_results["metrics"]["validation_avg_time"] < 0.1 else 80

    overall_score = (audit_score + perf_score) / 2

    print(".1f"    print("   Безопасность: {audit_score}/100")
    print("   Производительность: {perf_score}/100")

    if overall_score >= 90:
        print("   Статус: 🟢 ОТЛИЧНО")
    elif overall_score >= 70:
        print("   Статус: 🟡 ХОРОШО")

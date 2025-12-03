"""
Этическое обучение и безопасность
Фаза 4: Продвинутые функции
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import hashlib

from ..learning.models import AgentExperience, LearningResult, Pattern, ProcessedExperience
from ..learning.learning_engine import LearningEngine
from ..core.models import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


class EthicalCategory:
    """Категории этических проблем"""
    HARM_TO_HUMANS = "harm_to_humans"
    DISCRIMINATION = "discrimination"
    PRIVACY_VIOLATION = "privacy_violation"
    MISINFORMATION = "misinformation"
    UNFAIRNESS = "unfairness"
    MANIPULATION = "manipulation"
    AUTONOMY_VIOLATION = "autonomy_violation"
    TRANSPARENCY_LACK = "transparency_lack"


class SafetyRisk:
    """Уровни рисков безопасности"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EthicalChecker:
    """
    Проверяющий этичности решений и безопасности обучения
    """

    def __init__(self):
        # Этические правила и паттерны
        self.ethical_rules = self._load_ethical_rules()
        self.safety_patterns = self._load_safety_patterns()

        # История проверок
        self.check_history: List[Dict[str, Any]] = []

        # Метрики этичности
        self.ethical_metrics = {
            'total_checks': 0,
            'ethical_violations': 0,
            'safety_incidents': 0,
            'blocked_decisions': 0,
            'warnings_issued': 0
        }

        logger.info("EthicalChecker initialized")

    async def check_decision_ethics(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверка этичности решения

        Args:
            decision: Принимаемое решение
            context: Контекст решения

        Returns:
            Dict[str, Any]: Результат проверки
        """
        self.ethical_metrics['total_checks'] += 1

        violations = []
        warnings = []
        risk_level = SafetyRisk.LOW

        # Проверка на вред людям
        harm_check = await self._check_harm_to_humans(decision, context)
        if harm_check['violates']:
            violations.append(harm_check)
            risk_level = max(risk_level, SafetyRisk.HIGH)

        # Проверка на дискриминацию
        discrimination_check = await self._check_discrimination(decision, context)
        if discrimination_check['violates']:
            violations.append(discrimination_check)
            risk_level = max(risk_level, SafetyRisk.MEDIUM)

        # Проверка на нарушение приватности
        privacy_check = await self._check_privacy_violation(decision, context)
        if privacy_check['violates']:
            violations.append(privacy_check)
            risk_level = max(risk_level, SafetyRisk.HIGH)

        # Проверка на дезинформацию
        misinformation_check = await self._check_misinformation(decision, context)
        if misinformation_check['violates']:
            violations.append(misinformation_check)
            risk_level = max(risk_level, SafetyRisk.MEDIUM)

        # Проверка на манипуляцию
        manipulation_check = await self._check_manipulation(decision, context)
        if manipulation_check['warning']:
            warnings.append(manipulation_check)

        # Сохранение результата проверки
        check_result = {
            'decision_id': decision.get('id', 'unknown'),
            'timestamp': datetime.now(),
            'violations': violations,
            'warnings': warnings,
            'risk_level': risk_level,
            'approved': len(violations) == 0,
            'requires_review': risk_level in [SafetyRisk.HIGH, SafetyRisk.CRITICAL]
        }

        self.check_history.append(check_result)

        if violations:
            self.ethical_metrics['ethical_violations'] += 1
            if not check_result['approved']:
                self.ethical_metrics['blocked_decisions'] += 1

        if warnings:
            self.ethical_metrics['warnings_issued'] += len(warnings)

        logger.info(f"Ethical check completed for decision {decision.get('id', 'unknown')}: "
                   f"approved={check_result['approved']}, violations={len(violations)}")

        return check_result

    async def check_learning_safety(self, experience: AgentExperience) -> Dict[str, Any]:
        """
        Проверка безопасности обучения на опыте

        Args:
            experience: Опыт для обучения

        Returns:
            Dict[str, Any]: Результат проверки безопасности
        """
        safety_issues = []
        risk_level = SafetyRisk.LOW

        # Проверка на вредные паттерны в обучении
        harmful_patterns = await self._detect_harmful_patterns(experience)
        if harmful_patterns:
            safety_issues.extend(harmful_patterns)
            risk_level = max(risk_level, SafetyRisk.MEDIUM)

        # Проверка на bias в данных
        bias_check = await self._check_learning_bias(experience)
        if bias_check['detected']:
            safety_issues.append(bias_check)
            risk_level = max(risk_level, SafetyRisk.MEDIUM)

        # Проверка на потенциал злоупотребления
        abuse_check = await self._check_abuse_potential(experience)
        if abuse_check['high_risk']:
            safety_issues.append(abuse_check)
            risk_level = max(risk_level, SafetyRisk.HIGH)

        result = {
            'experience_id': experience.id,
            'safe_to_learn': risk_level != SafetyRisk.CRITICAL,
            'safety_issues': safety_issues,
            'risk_level': risk_level,
            'recommendations': self._generate_safety_recommendations(safety_issues),
            'timestamp': datetime.now()
        }

        if safety_issues:
            self.ethical_metrics['safety_incidents'] += 1

        return result

    async def generate_ethical_guidelines(self, domain: str) -> Dict[str, Any]:
        """
        Генерация этических рекомендаций для домена

        Args:
            domain: Область применения

        Returns:
            Dict[str, Any]: Этические рекомендации
        """
        guidelines = {
            'domain': domain,
            'general_principles': [
                'respect_human_autonomy',
                'ensure_fairness',
                'protect_privacy',
                'promote_transparency',
                'prevent_harm'
            ],
            'domain_specific_rules': self._get_domain_rules(domain),
            'decision_framework': self._create_decision_framework(domain),
            'monitoring_requirements': self._get_monitoring_requirements(domain),
            'generated_at': datetime.now()
        }

        return guidelines

    def _load_ethical_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Загрузка этических правил

        Returns:
            Dict[str, List[Dict[str, Any]]]: Этические правила
        """
        return {
            EthicalCategory.HARM_TO_HUMANS: [
                {
                    'pattern': r'(?i)(kill|hurt|damage|injure|destroy).*human',
                    'severity': SafetyRisk.CRITICAL,
                    'description': 'Potential harm to humans'
                },
                {
                    'pattern': r'(?i)weapon|bomb|explosive|poison',
                    'severity': SafetyRisk.HIGH,
                    'description': 'Dangerous weapons or substances'
                }
            ],
            EthicalCategory.DISCRIMINATION: [
                {
                    'pattern': r'(?i)(discriminat|bias|prejudice).*race|gender|religion|ethnicity',
                    'severity': SafetyRisk.HIGH,
                    'description': 'Discriminatory content'
                }
            ],
            EthicalCategory.PRIVACY_VIOLATION: [
                {
                    'pattern': r'(?i)personal.*data|private.*information|confidential',
                    'severity': SafetyRisk.MEDIUM,
                    'description': 'Privacy-sensitive information'
                }
            ],
            EthicalCategory.MISINFORMATION: [
                {
                    'pattern': r'(?i)false.*information|mislead|deceive|lie',
                    'severity': SafetyRisk.MEDIUM,
                    'description': 'Potential misinformation'
                }
            ]
        }

    def _load_safety_patterns(self) -> List[Dict[str, Any]]:
        """
        Загрузка паттернов безопасности

        Returns:
            List[Dict[str, Any]]: Паттерны безопасности
        """
        return [
            {
                'name': 'harmful_instructions',
                'patterns': [r'(?i)how to.*(hack|steal|forge|fraud)'],
                'risk_level': SafetyRisk.HIGH
            },
            {
                'name': 'sensitive_data_exposure',
                'patterns': [r'\b\d{3}-\d{2}-\d{4}\b', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
                'risk_level': SafetyRisk.MEDIUM
            },
            {
                'name': 'malicious_code',
                'patterns': [r'(?i)(rm -rf|format|delete.*system|drop.*table)'],
                'risk_level': SafetyRisk.CRITICAL
            }
        ]

    async def _check_harm_to_humans(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверка на вред людям

        Args:
            decision: Решение
            context: Контекст

        Returns:
            Dict[str, Any]: Результат проверки
        """
        content = self._extract_content_from_decision(decision)

        for rule in self.ethical_rules[EthicalCategory.HARM_TO_HUMANS]:
            if re.search(rule['pattern'], content):
                return {
                    'violates': True,
                    'category': EthicalCategory.HARM_TO_HUMANS,
                    'severity': rule['severity'],
                    'description': rule['description'],
                    'matched_pattern': rule['pattern']
                }

        return {'violates': False}

    async def _check_discrimination(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверка на дискриминацию

        Args:
            decision: Решение
            context: Контекст

        Returns:
            Dict[str, Any]: Результат проверки
        """
        content = self._extract_content_from_decision(decision)

        for rule in self.ethical_rules[EthicalCategory.DISCRIMINATION]:
            if re.search(rule['pattern'], content):
                return {
                    'violates': True,
                    'category': EthicalCategory.DISCRIMINATION,
                    'severity': rule['severity'],
                    'description': rule['description'],
                    'matched_pattern': rule['pattern']
                }

        return {'violates': False}

    async def _check_privacy_violation(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверка на нарушение приватности

        Args:
            decision: Решение
            context: Контекст

        Returns:
            Dict[str, Any]: Результат проверки
        """
        content = self._extract_content_from_decision(decision)

        for rule in self.ethical_rules[EthicalCategory.PRIVACY_VIOLATION]:
            if re.search(rule['pattern'], content):
                return {
                    'violates': True,
                    'category': EthicalCategory.PRIVACY_VIOLATION,
                    'severity': rule['severity'],
                    'description': rule['description'],
                    'matched_pattern': rule['pattern']
                }

        return {'violates': False}

    async def _check_misinformation(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверка на дезинформацию

        Args:
            decision: Решение
            context: Контекст

        Returns:
            Dict[str, Any]: Результат проверки
        """
        content = self._extract_content_from_decision(decision)

        # Проверка на уверенные заявления о спорных фактах
        confidence_indicators = ['definitely', 'absolutely', 'certainly', 'undoubtedly']
        controversial_topics = ['conspiracy', 'hoax', 'fake news']

        has_high_confidence = any(word in content.lower() for word in confidence_indicators)
        has_controversial = any(topic in content.lower() for topic in controversial_topics)

        if has_high_confidence and has_controversial:
            return {
                'violates': True,
                'category': EthicalCategory.MISINFORMATION,
                'severity': SafetyRisk.MEDIUM,
                'description': 'High confidence claims about controversial topics'
            }

        return {'violates': False}

    async def _check_manipulation(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверка на манипуляцию

        Args:
            decision: Решение
            context: Контекст

        Returns:
            Dict[str, Any]: Результат проверки
        """
        content = self._extract_content_from_decision(decision)

        manipulation_patterns = [
            r'(?i)you must|you have to|you should',
            r'(?i)everyone knows|everyone agrees',
            r'(?i)trust me|believe me'
        ]

        for pattern in manipulation_patterns:
            if re.search(pattern, content):
                return {
                    'warning': True,
                    'category': EthicalCategory.MANIPULATION,
                    'severity': SafetyRisk.LOW,
                    'description': 'Potential manipulative language',
                    'matched_pattern': pattern
                }

        return {'warning': False}

    async def _detect_harmful_patterns(self, experience: AgentExperience) -> List[Dict[str, Any]]:
        """
        Обнаружение вредных паттернов в обучении

        Args:
            experience: Опыт

        Returns:
            List[Dict[str, Any]]: Найденные вредные паттерны
        """
        harmful_patterns = []
        content = f"{getattr(experience, 'query', '')} {getattr(experience, 'response', '')}"

        for safety_pattern in self.safety_patterns:
            for pattern in safety_pattern['patterns']:
                if re.search(pattern, content):
                    harmful_patterns.append({
                        'pattern_name': safety_pattern['name'],
                        'matched_pattern': pattern,
                        'risk_level': safety_pattern['risk_level'],
                        'description': f"Matched harmful pattern: {safety_pattern['name']}"
                    })

        return harmful_patterns

    async def _check_learning_bias(self, experience: AgentExperience) -> Dict[str, Any]:
        """
        Проверка на bias в данных обучения

        Args:
            experience: Опыт

        Returns:
            Dict[str, Any]: Результат проверки на bias
        """
        # Простая проверка на стереотипы
        content = f"{getattr(experience, 'query', '')} {getattr(experience, 'response', '')}"

        bias_indicators = [
            'all men are', 'all women are', 'typical behavior',
            'naturally', 'instinctively', 'by nature'
        ]

        bias_detected = any(indicator in content.lower() for indicator in bias_indicators)

        return {
            'detected': bias_detected,
            'bias_type': 'stereotyping' if bias_detected else None,
            'description': 'Potential stereotypical assumptions' if bias_detected else 'No bias detected'
        }

    async def _check_abuse_potential(self, experience: AgentExperience) -> Dict[str, Any]:
        """
        Проверка на потенциал злоупотребления

        Args:
            experience: Опыт

        Returns:
            Dict[str, Any]: Оценка потенциала злоупотребления
        """
        content = f"{getattr(experience, 'query', '')} {getattr(experience, 'response', '')}"

        # Проверка на запросы опасных инструкций
        dangerous_queries = [
            'how to hack', 'how to make', 'how to build.*bomb',
            'how to poison', 'how to steal', 'how to forge'
        ]

        high_risk = any(query in content.lower() for query in dangerous_queries)

        return {
            'high_risk': high_risk,
            'risk_factors': dangerous_queries if high_risk else [],
            'description': 'High abuse potential detected' if high_risk else 'Low abuse potential'
        }

    def _extract_content_from_decision(self, decision: Dict[str, Any]) -> str:
        """
        Извлечение текстового содержимого из решения

        Args:
            decision: Решение

        Returns:
            str: Текстовое содержимое
        """
        content_parts = []

        for key, value in decision.items():
            if isinstance(value, str):
                content_parts.append(value)
            elif isinstance(value, dict):
                content_parts.extend(self._extract_text_from_dict(value))
            elif isinstance(value, list):
                content_parts.extend(str(item) for item in value if isinstance(item, (str, int, float)))

        return ' '.join(content_parts)

    def _extract_text_from_dict(self, data: Dict[str, Any]) -> List[str]:
        """
        Рекурсивное извлечение текста из словаря

        Args:
            data: Словарь

        Returns:
            List[str]: Текстовые значения
        """
        texts = []
        for value in data.values():
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict):
                texts.extend(self._extract_text_from_dict(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        texts.append(item)
        return texts

    def _generate_safety_recommendations(self, safety_issues: List[Dict[str, Any]]) -> List[str]:
        """
        Генерация рекомендаций по безопасности

        Args:
            safety_issues: Проблемы безопасности

        Returns:
            List[str]: Рекомендации
        """
        recommendations = []

        for issue in safety_issues:
            if issue.get('risk_level') == SafetyRisk.CRITICAL:
                recommendations.append("Block this learning experience immediately")
            elif issue.get('risk_level') == SafetyRisk.HIGH:
                recommendations.append("Require human review before learning")
            elif issue.get('risk_level') == SafetyRisk.MEDIUM:
                recommendations.append("Add additional safety checks")
            else:
                recommendations.append("Monitor this pattern closely")

        return list(set(recommendations))  # Удаление дубликатов

    def _get_domain_rules(self, domain: str) -> List[Dict[str, Any]]:
        """
        Получение правил для конкретного домена

        Args:
            domain: Домен

        Returns:
            List[Dict[str, Any]]: Правила домена
        """
        domain_rules = {
            'healthcare': [
                {'rule': 'never_diagnose_without_doctor', 'priority': 'critical'},
                {'rule': 'protect_patient_privacy', 'priority': 'high'}
            ],
            'finance': [
                {'rule': 'prevent_fraud_assistance', 'priority': 'critical'},
                {'rule': 'ensure_transparency', 'priority': 'high'}
            ],
            'education': [
                {'rule': 'provide_accurate_information', 'priority': 'high'},
                {'rule': 'avoid_political_bias', 'priority': 'medium'}
            ]
        }

        return domain_rules.get(domain, [])

    def _create_decision_framework(self, domain: str) -> Dict[str, Any]:
        """
        Создание фреймворка принятия решений

        Args:
            domain: Домен

        Returns:
            Dict[str, Any]: Фреймворк решений
        """
        return {
            'steps': [
                'assess_harm_potential',
                'check_fairness',
                'verify_accuracy',
                'ensure_transparency',
                'get_human_review_if_needed'
            ],
            'checkpoints': [
                {'step': 'assess_harm_potential', 'required': True},
                {'step': 'check_fairness', 'required': True},
                {'step': 'verify_accuracy', 'domain_specific': True},
                {'step': 'ensure_transparency', 'required': True}
            ]
        }

    def _get_monitoring_requirements(self, domain: str) -> List[str]:
        """
        Получение требований к мониторингу

        Args:
            domain: Домен

        Returns:
            List[str]: Требования мониторинга
        """
        base_requirements = [
            'log_all_decisions',
            'track_user_feedback',
            'monitor_for_bias',
            'regular_audits'
        ]

        domain_specific = {
            'healthcare': ['medical_accuracy_checks', 'patient_outcome_tracking'],
            'finance': ['transaction_monitoring', 'fraud_detection'],
            'education': ['learning_outcome_assessment', 'content_accuracy_verification']
        }

        return base_requirements + domain_specific.get(domain, [])

    def get_ethical_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик этичности

        Returns:
            Dict[str, Any]: Метрики
        """
        total_checks = self.ethical_metrics['total_checks']
        return {
            **self.ethical_metrics,
            'violation_rate': self.ethical_metrics['ethical_violations'] / total_checks if total_checks > 0 else 0,
            'safety_incident_rate': self.ethical_metrics['safety_incidents'] / total_checks if total_checks > 0 else 0,
            'approval_rate': (total_checks - self.ethical_metrics['blocked_decisions']) / total_checks if total_checks > 0 else 1.0,
            'recent_checks': len([c for c in self.check_history if (datetime.now() - c['timestamp']).days < 7])
        }


class EthicalLearningEngine:
    """
    Двигатель этического обучения с проверками безопасности
    """

    def __init__(self, learning_engine: LearningEngine):
        """
        Инициализация этического двигателя обучения

        Args:
            learning_engine: Базовый двигатель обучения
        """
        self.learning_engine = learning_engine
        self.ethical_checker = EthicalChecker()

        # Этические паттерны обучения
        self.ethical_patterns: List[Pattern] = []
        self.blocked_experiences: List[str] = []

        # Метрики этического обучения
        self.ethical_learning_metrics = {
            'ethical_learning_sessions': 0,
            'experiences_filtered': 0,
            'decisions_reviewed': 0,
            'ethical_improvements': 0
        }

        logger.info("EthicalLearningEngine initialized")

    async def learn_ethically(self, experience: AgentExperience) -> LearningResult:
        """
        Этическое обучение на опыте

        Args:
            experience: Опыт агента

        Returns:
            LearningResult: Результат обучения
        """
        self.ethical_learning_metrics['ethical_learning_sessions'] += 1

        # Шаг 1: Проверка безопасности обучения
        safety_check = await self.ethical_checker.check_learning_safety(experience)

        if not safety_check['safe_to_learn']:
            self.blocked_experiences.append(experience.id)
            self.ethical_learning_metrics['experiences_filtered'] += 1

            logger.warning(f"Experience {experience.id} blocked for safety reasons: {safety_check['risk_level']}")

            # Возврат пустого результата
            return LearningResult(
                experience_processed=ProcessedExperience(
                    original_experience=experience,
                    key_elements=[],
                    significance_score=0.0,
                    categories=['blocked'],
                    lessons=['Experience blocked for ethical/safety reasons'],
                    processing_timestamp=datetime.now()
                ),
                patterns_extracted=0,
                cognitive_updates=0,
                skills_developed=0,
                adaptation_applied=None,
                learning_effectiveness=0.0,
                learning_time=0.0,
                timestamp=datetime.now()
            )

        # Шаг 2: Этическая фильтрация и улучшение опыта
        ethical_experience = await self._apply_ethical_filtering(experience)

        # Шаг 3: Обучение с этическими ограничениями
        result = await self.learning_engine.learn_from_experience(ethical_experience)

        # Шаг 4: Пост-обработка для этического улучшения
        ethical_result = await self._enhance_with_ethical_patterns(result)

        logger.info(f"Ethical learning completed for experience {experience.id}")

        return ethical_result

    async def review_decision(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверка решения на этичность

        Args:
            decision: Решение для проверки
            context: Контекст решения

        Returns:
            Dict[str, Any]: Результат проверки
        """
        self.ethical_learning_metrics['decisions_reviewed'] += 1

        ethical_check = await self.ethical_checker.check_decision_ethics(decision, context)

        # Дополнительные рекомендации
        if not ethical_check['approved']:
            ethical_check['remediation_suggestions'] = await self._suggest_remediation(decision, ethical_check['violations'])

        return ethical_check

    async def generate_ethical_training_data(self, domain: str) -> List[AgentExperience]:
        """
        Генерация этических тренировочных данных

        Args:
            domain: Домен

        Returns:
            List[AgentExperience]: Этические тренировочные примеры
        """
        ethical_scenarios = self._get_ethical_scenarios(domain)

        training_data = []
        for scenario in ethical_scenarios:
            experience = AgentExperience(
                id=f"ethical_training_{domain}_{len(training_data)}",
                query=scenario['query'],
                response=scenario['ethical_response'],
                timestamp=datetime.now(),
                significance_score=0.9,
                metadata={
                    'ethical_training': True,
                    'domain': domain,
                    'scenario_type': scenario['type'],
                    'ethical_principles': scenario['principles']
                }
            )
            training_data.append(experience)

        return training_data

    async def _apply_ethical_filtering(self, experience: AgentExperience) -> AgentExperience:
        """
        Применение этической фильтрации к опыту

        Args:
            experience: Исходный опыт

        Returns:
            AgentExperience: Отфильтрованный опыт
        """
        # Удаление потенциально вредного контента
        filtered_query = self._filter_content(experience.query)
        filtered_response = self._filter_content(experience.response)

        # Добавление этических метаданных
        ethical_metadata = getattr(experience, 'metadata', {}).copy()
        ethical_metadata['ethical_filtering_applied'] = True
        ethical_metadata['content_filtered'] = (filtered_query != experience.query) or (filtered_response != experience.response)

        filtered_experience = AgentExperience(
            id=experience.id,
            query=filtered_query,
            response=filtered_response,
            timestamp=experience.timestamp,
            significance_score=experience.significance_score,
            metadata=ethical_metadata
        )

        return filtered_experience

    async def _enhance_with_ethical_patterns(self, result: LearningResult) -> LearningResult:
        """
        Улучшение результата этическими паттернами

        Args:
            result: Результат обучения

        Returns:
            LearningResult: Улучшенный результат
        """
        # Добавление этических уроков
        ethical_lessons = [
            "Always consider potential harm",
            "Ensure fairness and avoid bias",
            "Protect user privacy",
            "Be transparent about limitations"
        ]

        if hasattr(result.experience_processed, 'lessons'):
            result.experience_processed.lessons.extend(ethical_lessons)

        # Увеличение этической эффективности
        ethical_bonus = 0.1  # +10% за этическое обучение
        result.learning_effectiveness = min(result.learning_effectiveness + ethical_bonus, 1.0)

        self.ethical_learning_metrics['ethical_improvements'] += 1

        return result

    async def _suggest_remediation(self, decision: Dict[str, Any], violations: List[Dict[str, Any]]) -> List[str]:
        """
        Предложение исправлений для нарушений

        Args:
            decision: Решение
            violations: Нарушения

        Returns:
            List[str]: Предложения по исправлению
        """
        suggestions = []

        for violation in violations:
            category = violation.get('category')

            if category == EthicalCategory.HARM_TO_HUMANS:
                suggestions.append("Remove or rephrase content that could cause harm")
                suggestions.append("Add disclaimers about responsible use")

            elif category == EthicalCategory.DISCRIMINATION:
                suggestions.append("Review and remove biased assumptions")
                suggestions.append("Ensure inclusive language")

            elif category == EthicalCategory.PRIVACY_VIOLATION:
                suggestions.append("Remove personal information")
                suggestions.append("Use anonymized examples")

            elif category == EthicalCategory.MISINFORMATION:
                suggestions.append("Add uncertainty markers")
                suggestions.append("Provide sources for claims")

        return list(set(suggestions))  # Удаление дубликатов

    def _filter_content(self, content: str) -> str:
        """
        Фильтрация контента от потенциально вредных элементов

        Args:
            content: Исходный контент

        Returns:
            str: Отфильтрованный контент
        """
        if not content:
            return content

        # Простая фильтрация - можно расширить
        filtered = content

        # Удаление потенциально опасных инструкций
        dangerous_patterns = [
            r'(?i)how to.*hack',
            r'(?i)how to.*steal',
            r'(?i)how to.*forge'
        ]

        for pattern in dangerous_patterns:
            filtered = re.sub(pattern, '[FILTERED: Potentially harmful instruction]', filtered)

        return filtered

    def _get_ethical_scenarios(self, domain: str) -> List[Dict[str, Any]]:
        """
        Получение этических сценариев для домена

        Args:
            domain: Домен

        Returns:
            List[Dict[str, Any]]: Этические сценарии
        """
        base_scenarios = [
            {
                'type': 'privacy_protection',
                'query': 'How can I find someone\'s personal information?',
                'ethical_response': 'I cannot assist with finding personal information about individuals without their consent, as this violates privacy laws and ethical guidelines.',
                'principles': ['privacy', 'consent', 'lawfulness']
            },
            {
                'type': 'harm_prevention',
                'query': 'Give me instructions for making explosives.',
                'ethical_response': 'I cannot provide instructions for creating explosives or any harmful substances, as this could cause serious harm to people.',
                'principles': ['harm_prevention', 'safety', 'responsibility']
            },
            {
                'type': 'fairness',
                'query': 'Why are certain groups better than others?',
                'ethical_response': 'All individuals and groups should be treated with equal respect and dignity. Claims of superiority based on group membership are not supported by ethical principles.',
                'principles': ['fairness', 'equality', 'respect']
            }
        ]

        # Домен-специфичные сценарии
        domain_scenarios = {
            'healthcare': [
                {
                    'type': 'medical_advice',
                    'query': 'I have chest pain, what should I do?',
                    'ethical_response': 'I am not a medical professional and cannot provide medical advice. Please seek immediate medical attention from a qualified healthcare provider.',
                    'principles': ['competence', 'safety', 'professionalism']
                }
            ],
            'finance': [
                {
                    'type': 'investment_advice',
                    'query': 'What\'s the best stock to invest in?',
                    'ethical_response': 'I cannot provide personalized investment advice. Investment decisions should be made with qualified financial advisors based on your individual circumstances.',
                    'principles': ['competence', 'transparency', 'responsibility']
                }
            ]
        }

        return base_scenarios + domain_scenarios.get(domain, [])

    def get_ethical_learning_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик этического обучения

        Returns:
            Dict[str, Any]: Метрики
        """
        total_sessions = self.ethical_learning_metrics['ethical_learning_sessions']
        return {
            **self.ethical_learning_metrics,
            'filtering_rate': self.ethical_learning_metrics['experiences_filtered'] / total_sessions if total_sessions > 0 else 0,
            'review_rate': self.ethical_learning_metrics['decisions_reviewed'] / total_sessions if total_sessions > 0 else 0,
            'ethical_improvement_rate': self.ethical_learning_metrics['ethical_improvements'] / total_sessions if total_sessions > 0 else 0,
            'blocked_experiences_count': len(self.blocked_experiences)
        }

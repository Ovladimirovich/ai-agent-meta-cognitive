import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.models import (
    ErrorAnalysis, ErrorInstance, ErrorPattern, ErrorSeverity,
    AgentInteraction
)

logger = logging.getLogger(__name__)


class ErrorClassifier:
    """Классификатор ошибок агента"""

    def __init__(self):
        self.error_patterns = {}
        self.error_categories = {
            'factual_error': {
                'description': 'Фактическая ошибка',
                'severity': ErrorSeverity.MEDIUM,
                'indicators': ['incorrect', 'wrong', 'false', 'factually']
            },
            'logical_error': {
                'description': 'Логическая ошибка',
                'severity': ErrorSeverity.HIGH,
                'indicators': ['contradiction', 'illogical', 'inconsistent', 'paradox']
            },
            'execution_error': {
                'description': 'Ошибка выполнения',
                'severity': ErrorSeverity.MEDIUM,
                'indicators': ['failed', 'error', 'exception', 'timeout']
            },
            'timeout_error': {
                'description': 'Превышение времени ожидания',
                'severity': ErrorSeverity.MEDIUM,
                'indicators': ['timeout', 'timed out', 'too long', 'slow']
            },
            'resource_error': {
                'description': 'Ошибка ресурсов',
                'severity': ErrorSeverity.HIGH,
                'indicators': ['memory', 'disk', 'cpu', 'resource', 'limit']
            },
            'understanding_error': {
                'description': 'Ошибка понимания запроса',
                'severity': ErrorSeverity.LOW,
                'indicators': ['misunderstood', 'unclear', 'ambiguous', 'confused']
            },
            'tool_error': {
                'description': 'Ошибка инструмента',
                'severity': ErrorSeverity.MEDIUM,
                'indicators': ['tool failed', 'instrument error', 'api error']
            },
            'validation_error': {
                'description': 'Ошибка валидации',
                'severity': ErrorSeverity.LOW,
                'indicators': ['invalid', 'validation', 'format', 'schema']
            }
        }

    async def classify_errors(self, interaction: AgentInteraction) -> ErrorAnalysis:
        """Классификация ошибок во взаимодействии"""
        errors = []

        # Анализ результата на ошибки
        if interaction.response and hasattr(interaction.response, 'result'):
            result_errors = self._analyze_result_for_errors(interaction.response.result)
            errors.extend(result_errors)

        # Анализ трассировки рассуждений на ошибки
        trace_source = interaction.reasoning_trace or (interaction.response.reasoning_trace if interaction.response else [])
        trace_errors = self._analyze_trace_for_errors(trace_source)
        errors.extend(trace_errors)

        # Анализ списка ошибок взаимодействия
        if interaction.errors:
            explicit_errors = self._classify_explicit_errors(interaction.errors)
            errors.extend(explicit_errors)

        # Группировка ошибок по паттернам
        patterns = self._group_errors_into_patterns(errors)

        # Расчет метрик ошибок
        metrics = self._calculate_error_metrics(errors, patterns)

        # Оценка серьезности
        severity_assessment = self._assess_severity(errors)

        return ErrorAnalysis(
            errors=errors,
            patterns=patterns,
            metrics=metrics,
            severity_assessment=severity_assessment
        )

    def _analyze_result_for_errors(self, result: Any) -> List[ErrorInstance]:
        """Анализ результата на наличие ошибок"""
        errors = []
        result_text = str(result).lower()

        for category, config in self.error_categories.items():
            if any(indicator in result_text for indicator in config['indicators']):
                error = ErrorInstance(
                    category=category,
                    description=config['description'],
                    message=f"Обнаружена {config['description']} в результате",
                    severity=config['severity'],
                    context={'source': 'result', 'result_snippet': result_text[:200]},
                    timestamp=datetime.now()
                )
                errors.append(error)

        return errors

    def _analyze_trace_for_errors(self, trace: List[Dict]) -> List[ErrorInstance]:
        """Анализ трассировки рассуждений на ошибки"""
        errors = []

        for i, step in enumerate(trace):
            step_text = str(step.get('description', '')).lower()
            step_type = step.get('step_type', '')  # Исправлено: step_type вместо type

            # Специфические проверки для типов шагов (сначала!)
            if step_type == 'tool_execution' and 'failed' in step_text:
                error = ErrorInstance(
                    category='tool_error',
                    description='Ошибка выполнения инструмента',
                    message=f"Инструмент не смог выполниться в шаге {i+1}",
                    severity=ErrorSeverity.MEDIUM,
                    context={'source': 'trace', 'step_index': i},
                    timestamp=datetime.now()
                )
                errors.append(error)
                continue  # Пропускаем общую проверку, если нашли специфическую

            # Проверка на явные ошибки в шаге (общие индикаторы)
            for category, config in self.error_categories.items():
                if any(indicator in step_text for indicator in config['indicators']):
                    error = ErrorInstance(
                        category=category,
                        description=config['description'],
                        message=f"Ошибка в шаге {i+1}: {step.get('description', '')}",
                        severity=config['severity'],
                        context={
                            'source': 'trace',
                            'step_index': i,
                            'step_type': step_type
                        },
                        timestamp=datetime.now()
                    )
                    errors.append(error)
                    break  # Нашли категорию, выходим

        return errors

    def _classify_explicit_errors(self, error_list: List[str]) -> List[ErrorInstance]:
        """Классификация явных ошибок из списка"""
        errors = []

        for error_msg in error_list:
            error_text = str(error_msg).lower()
            classified = False

            # Специфические проверки типов Python ошибок (сначала!)
            if 'valueerror' in error_text:
                error = ErrorInstance(
                    category='validation_error',
                    description='Ошибка валидации',
                    message=error_msg,
                    severity=ErrorSeverity.LOW,
                    context={'source': 'explicit'},
                    timestamp=datetime.now()
                )
                errors.append(error)
                continue
            elif 'connectionerror' in error_text or 'network' in error_text:
                error = ErrorInstance(
                    category='timeout_error',
                    description='Ошибка сети/таймаута',
                    message=error_msg,
                    severity=ErrorSeverity.MEDIUM,
                    context={'source': 'explicit'},
                    timestamp=datetime.now()
                )
                errors.append(error)
                continue
            elif 'runtimeerror' in error_text:
                error = ErrorInstance(
                    category='execution_error',
                    description='Ошибка выполнения',
                    message=error_msg,
                    severity=ErrorSeverity.MEDIUM,
                    context={'source': 'explicit'},
                    timestamp=datetime.now()
                )
                errors.append(error)
                continue

            # Общие индикаторы (если не подошли специфические)
            for category, config in self.error_categories.items():
                if any(indicator in error_text for indicator in config['indicators']):
                    error = ErrorInstance(
                        category=category,
                        description=config['description'],
                        message=error_msg,
                        severity=config['severity'],
                        context={'source': 'explicit'},
                        timestamp=datetime.now()
                    )
                    errors.append(error)
                    classified = True
                    break

            # Если не удалось классифицировать, создаем общую ошибку
            if not classified:
                error = ErrorInstance(
                    category='unknown_error',
                    description='Неизвестная ошибка',
                    message=error_msg,
                    severity=ErrorSeverity.MEDIUM,
                    context={'source': 'explicit'},
                    timestamp=datetime.now()
                )
                errors.append(error)

        return errors

    def _group_errors_into_patterns(self, errors: List[ErrorInstance]) -> List[ErrorPattern]:
        """Группировка ошибок в паттерны"""
        if not errors:
            return []

        # Группировка по категориям
        category_groups = {}
        for error in errors:
            if error.category not in category_groups:
                category_groups[error.category] = []
            category_groups[error.category].append(error)

        patterns = []
        for category, category_errors in category_groups.items():
            if len(category_errors) >= 2:  # Минимум 2 ошибки для паттерна
                pattern = ErrorPattern(
                    pattern_id=f"{category}_pattern_{len(patterns)}",
                    description=f"Повторяющаяся ошибка типа '{category}': {len(category_errors)} случаев",
                    confidence=min(len(category_errors) / 10.0, 1.0),  # Нормализация
                    examples=category_errors[:5],  # Ограничим примерами
                    recommendation=self._generate_pattern_recommendation(category, category_errors)
                )
                patterns.append(pattern)

        return patterns

    def _generate_pattern_recommendation(self, category: str, errors: List[ErrorInstance]) -> str:
        """Генерация рекомендации для паттерна ошибок"""
        recommendations = {
            'factual_error': "Улучшить систему проверки фактов и верификации информации",
            'logical_error': "Добавить дополнительные проверки логической一致ности",
            'execution_error': "Улучшить обработку исключений и восстановление после ошибок",
            'timeout_error': "Оптимизировать время выполнения и добавить таймауты",
            'resource_error': "Мониторить использование ресурсов и добавить ограничения",
            'understanding_error': "Улучшить анализ и понимание запросов пользователя",
            'tool_error': "Добавить проверки доступности инструментов и fallback механизмы",
            'validation_error': "Улучшить валидацию входных данных и форматов"
        }

        return recommendations.get(category, "Проанализировать и устранить причину повторяющихся ошибок")

    def _calculate_error_metrics(self, errors: List[ErrorInstance],
                               patterns: List[ErrorPattern]) -> Dict[str, Any]:
        """Расчет метрик ошибок"""
        if not errors:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'severity_distribution': {},
                'most_common_category': None,
                'pattern_count': 0
            }

        total_errors = len(errors)

        # Распределение по серьезности
        severity_dist = {}
        for error in errors:
            severity = error.severity.value
            severity_dist[severity] = severity_dist.get(severity, 0) + 1

        # Самая частая категория
        category_counts = {}
        for error in errors:
            category_counts[error.category] = category_counts.get(error.category, 0) + 1

        most_common_category = max(category_counts.items(), key=lambda x: x[1])[0]

        return {
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_errors + 10, 1),  # Нормализованная ставка
            'severity_distribution': severity_dist,
            'most_common_category': most_common_category,
            'pattern_count': len(patterns),
            'categories_count': len(category_counts)
        }

    def _assess_severity(self, errors: List[ErrorInstance]) -> Dict[str, int]:
        """Оценка общей серьезности ошибок"""
        severity_counts = {
            'low': 0,
            'medium': 0,
            'high': 0,
            'critical': 0
        }

        for error in errors:
            severity_counts[error.severity.value] += 1

        return severity_counts

    def _calculate_severity(self, category: str) -> ErrorSeverity:
        """Расчет серьезности ошибки по категории"""
        return self.error_categories.get(category, {}).get('severity', ErrorSeverity.MEDIUM)

    async def analyze_error_trends(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Анализ трендов ошибок по множеству взаимодействий"""
        all_errors = []

        for interaction in interactions:
            analysis = await self.classify_errors(interaction)
            all_errors.extend(analysis.errors)

        if not all_errors:
            return {'trend': 'no_errors', 'description': 'Ошибок не обнаружено'}

        # Анализ трендов по времени
        errors_by_time = {}
        for error in all_errors:
            if error.timestamp:
                time_key = error.timestamp.strftime('%Y-%m-%d')
                if time_key not in errors_by_time:
                    errors_by_time[time_key] = []
                errors_by_time[time_key].append(error)

        # Определение тренда
        recent_days = sorted(errors_by_time.keys())[-7:]  # Последние 7 дней
        recent_error_counts = [len(errors_by_time.get(day, [])) for day in recent_days]

        if len(recent_error_counts) >= 2:
            trend = 'stable'
            if recent_error_counts[-1] > recent_error_counts[0] * 1.5:
                trend = 'increasing'
            elif recent_error_counts[-1] < recent_error_counts[0] * 0.7:
                trend = 'decreasing'

            return {
                'trend': trend,
                'recent_counts': recent_error_counts,
                'description': f"Тренд ошибок: {trend}"
            }

        return {
            'trend': 'insufficient_data',
            'description': 'Недостаточно данных для анализа тренда'
        }

    async def generate_error_report(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Генерация автоматического отчета об ошибках"""
        # Сбор всех ошибок
        all_errors = []
        error_analyses = []

        for interaction in interactions:
            analysis = await self.classify_errors(interaction)
            all_errors.extend(analysis.errors)
            error_analyses.append(analysis)

        # Генерация сводки
        report = {
            'generated_at': datetime.now(),
            'period': {
                'start': min((i.timestamp for i in interactions), default=None),
                'end': max((i.timestamp for i in interactions), default=None),
                'total_interactions': len(interactions)
            },
            'summary': {
                'total_errors': len(all_errors),
                'error_rate': len(all_errors) / max(len(interactions), 1),
                'interactions_with_errors': len([a for a in error_analyses if a.errors])
            }
        }

        if all_errors:
            # Распределение по категориям
            category_distribution = {}
            severity_distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

            for error in all_errors:
                category_distribution[error.category] = category_distribution.get(error.category, 0) + 1
                severity_distribution[error.severity.value] += 1

            report['distributions'] = {
                'by_category': category_distribution,
                'by_severity': severity_distribution
            }

            # Топ паттернов
            all_patterns = []
            for analysis in error_analyses:
                all_patterns.extend(analysis.patterns)

            top_patterns = sorted(all_patterns, key=lambda p: p.confidence, reverse=True)[:5]
            report['top_patterns'] = [
                {
                    'description': p.description,
                    'confidence': p.confidence,
                    'recommendation': p.recommendation,
                    'examples_count': len(p.examples)
                }
                for p in top_patterns
            ]

            # Критические ошибки
            critical_errors = [e for e in all_errors if e.severity == ErrorSeverity.CRITICAL]
            report['critical_errors'] = [
                {
                    'category': e.category,
                    'message': e.message,
                    'timestamp': e.timestamp.isoformat() if e.timestamp else None
                }
                for e in critical_errors[:10]  # Ограничим 10 самыми свежими
            ]

        # Рекомендации
        report['recommendations'] = self._generate_report_recommendations(report)

        return report

    def _generate_report_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе отчета"""
        recommendations = []

        summary = report.get('summary', {})
        distributions = report.get('distributions', {})

        # Анализ уровня ошибок
        error_rate = summary.get('error_rate', 0)
        if error_rate > 0.5:
            recommendations.append("Критический уровень ошибок! Требуется немедленное вмешательство.")
        elif error_rate > 0.2:
            recommendations.append("Высокий уровень ошибок. Рекомендуется анализ и оптимизация.")
        elif error_rate > 0.1:
            recommendations.append("Умеренный уровень ошибок. Следует мониторить тенденции.")

        # Анализ распределения по категориям
        category_dist = distributions.get('by_category', {})
        if category_dist:
            most_common = max(category_dist.items(), key=lambda x: x[1])
            total_interactions = report.get('period', {}).get('total_interactions', 1)
            if most_common[1] > total_interactions * 0.3:
                recommendations.append(f"Основная проблема: {most_common[0]}. Фокус на устранении этой категории ошибок.")

        # Анализ серьезности
        severity_dist = distributions.get('by_severity', {})
        critical_count = severity_dist.get('critical', 0)
        if critical_count > 0:
            recommendations.append(f"Обнаружено {critical_count} критических ошибок. Требуется срочное исправление.")

        # Общие рекомендации
        if not recommendations:
            recommendations.append("Уровень ошибок в норме. Продолжить мониторинг.")

        return recommendations

    async def send_error_alert(self, error_analysis: ErrorAnalysis, alert_config: Dict[str, Any]) -> bool:
        """Отправка автоматического алерта об ошибках"""
        try:
            # Проверка условий для алерта
            if not self._should_send_alert(error_analysis, alert_config):
                return False

            # Формирование сообщения алерта
            alert_message = self._format_alert_message(error_analysis)

            # Отправка алерта (заглушка - в реальности интеграция с системами уведомлений)
            logger.warning(f"ERROR ALERT: {alert_message}")

            # Здесь можно добавить интеграцию с:
            # - Email
            # - Slack/Discord
            # - PagerDuty
            # - SMS
            # - и т.д.

            return True

        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")
            return False

    def _should_send_alert(self, error_analysis: ErrorAnalysis, alert_config: Dict[str, Any]) -> bool:
        """Проверка необходимости отправки алерта"""
        # Проверка на критические ошибки
        critical_errors = [e for e in error_analysis.errors if e.severity == ErrorSeverity.CRITICAL]
        if critical_errors and alert_config.get('alert_on_critical', True):
            return True

        # Проверка на высокую частоту ошибок
        error_threshold = alert_config.get('error_rate_threshold', 0.3)
        total_errors = len(error_analysis.errors)
        if total_errors > error_threshold * 10:  # Примерная оценка
            return True

        # Проверка на новые паттерны ошибок
        if error_analysis.patterns and alert_config.get('alert_on_new_patterns', True):
            return True

        return False

    def _format_alert_message(self, error_analysis: ErrorAnalysis) -> str:
        """Форматирование сообщения алерта"""
        lines = ["🚨 ОБНАРУЖЕНЫ ОШИБКИ АГЕНТА", ""]

        # Сводка
        lines.append(f"Всего ошибок: {len(error_analysis.errors)}")
        lines.append(f"Паттернов: {len(error_analysis.patterns)}")

        # Критические ошибки
        critical = [e for e in error_analysis.errors if e.severity == ErrorSeverity.CRITICAL]
        if critical:
            lines.append(f"Критических: {len(critical)}")

        # Топ категорий
        if error_analysis.errors:
            from collections import Counter
            categories = Counter(e.category for e in error_analysis.errors)
            top_category = categories.most_common(1)[0]
            lines.append(f"Основная категория: {top_category[0]} ({top_category[1]} случаев)")

        return "\n".join(lines)

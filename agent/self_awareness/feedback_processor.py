import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.models import (
    FeedbackAnalysis, FeedbackIssue, Sentiment, Insight, InsightType,
    AgentInteraction
)

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Анализатор sentiment обратной связи"""

    def __init__(self):
        self.sentiment_keywords = {
            'positive': [
                'хорошо', 'отлично', 'прекрасно', 'замечательно', 'великолепно',
                'правильно', 'верно', 'точно', 'спасибо', 'полезно',
                'good', 'excellent', 'great', 'perfect', 'awesome',
                'correct', 'right', 'accurate', 'helpful', 'useful'
            ],
            'negative': [
                'плохо', 'ужасно', 'отвратительно', 'неправильно', 'ошибка',
                'неверно', 'ложь', 'бесполезно', 'бессмысленно', 'разочарование',
                'bad', 'terrible', 'awful', 'wrong', 'error',
                'incorrect', 'false', 'useless', 'disappointing', 'fail'
            ],
            'neutral': [
                'нормально', 'обычно', 'стандартно', 'приемлемо', 'ок',
                'fine', 'okay', 'average', 'standard', 'acceptable'
            ]
        }

    async def analyze(self, text: str) -> Sentiment:
        """Анализ sentiment текста"""
        if not text or not text.strip():
            return Sentiment.NEUTRAL

        text_lower = text.lower()
        scores = {'positive': 0, 'negative': 0, 'neutral': 0}

        # Подсчет ключевых слов
        for sentiment, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[sentiment] += 1

        # Определение преобладающего sentiment
        max_sentiment = max(scores.items(), key=lambda x: x[1])

        # Если нет явных ключевых слов, возвращаем neutral
        if max_sentiment[1] == 0:
            return Sentiment.NEUTRAL

        # Проверка на смешанные чувства (близкие scores)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1][1] > 0:
            ratio = sorted_scores[0][1] / max(sorted_scores[1][1], 1)
            if ratio < 2.0:  # Если разница менее чем в 2 раза
                return Sentiment.NEUTRAL

        sentiment_map = {
            'positive': Sentiment.POSITIVE,
            'negative': Sentiment.NEGATIVE,
            'neutral': Sentiment.NEUTRAL
        }

        return sentiment_map[max_sentiment[0]]


class FeedbackClassifier:
    """Классификатор типов обратной связи"""

    def __init__(self):
        self.feedback_types = {
            'accuracy': {
                'name': 'Точность ответа',
                'indicators': ['точность', 'правильность', 'верность', 'accuracy', 'correctness']
            },
            'completeness': {
                'name': 'Полнота ответа',
                'indicators': ['полнота', 'завершенность', 'complete', 'comprehensive']
            },
            'relevance': {
                'name': 'Релевантность',
                'indicators': ['релевантность', 'отношение', 'relevance', 'pertinent']
            },
            'usefulness': {
                'name': 'Полезность',
                'indicators': ['полезность', 'пригодность', 'useful', 'helpful']
            },
            'performance': {
                'name': 'Производительность',
                'indicators': ['скорость', 'быстро', 'медленно', 'performance', 'speed']
            },
            'interface': {
                'name': 'Интерфейс',
                'indicators': ['интерфейс', 'дизайн', 'ui', 'ux', 'interface']
            },
            'general': {
                'name': 'Общая обратная связь',
                'indicators': []  # По умолчанию
            }
        }

    async def classify(self, feedback: 'FeedbackAnalysis') -> str:
        """Классификация типа обратной связи"""
        text = feedback.__dict__.get('text', '') if hasattr(feedback, 'text') else str(feedback)

        for feedback_type, config in self.feedback_types.items():
            for indicator in config['indicators']:
                if indicator.lower() in text.lower():
                    return feedback_type

        return 'general'


class FeedbackProcessor:
    """Обработчик обратной связи"""

    def __init__(self, hybrid_manager=None):
        self.hybrid_manager = hybrid_manager
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feedback_classifier = FeedbackClassifier()

    async def process_feedback(self, feedback: FeedbackAnalysis) -> FeedbackAnalysis:
        """Обработка обратной связи пользователя"""
        # Анализ sentiment
        sentiment = await self.sentiment_analyzer.analyze(feedback.__dict__.get('text', ''))

        # Классификация типа обратной связи
        feedback_type = await self.feedback_classifier.classify(feedback)

        # Извлечение конкретных проблем
        issues = await self._extract_issues(feedback, sentiment, feedback_type)

        # Генерация insights
        insights = await self._generate_feedback_insights(feedback, issues, sentiment)

        # Оценка критичности
        criticality = self._assess_criticality(sentiment, feedback_type, issues)

        # Обновление объекта feedback
        feedback.sentiment = sentiment
        feedback.feedback_type = feedback_type
        feedback.issues = issues
        feedback.insights = insights
        feedback.criticality = criticality
        feedback.processed_at = datetime.now()

        return feedback

    async def _extract_issues(self, feedback: FeedbackAnalysis,
                            sentiment: Sentiment,
                            feedback_type: str) -> List[FeedbackIssue]:
        """Извлечение конкретных проблем из обратной связи"""
        issues = []
        text = feedback.__dict__.get('text', '')

        # Базовый анализ текста
        basic_issues = self._extract_basic_issues(text, sentiment, feedback_type)
        issues.extend(basic_issues)

        # Если доступен hybrid_manager, используем AI для глубокого анализа
        if self.hybrid_manager and text:
            ai_issues = await self._extract_issues_with_ai(text, sentiment, feedback_type)
            issues.extend(ai_issues)

        return issues

    def _extract_basic_issues(self, text: str, sentiment: Sentiment,
                            feedback_type: str) -> List[FeedbackIssue]:
        """Базовое извлечение проблем из текста"""
        issues = []

        if sentiment == Sentiment.NEGATIVE:
            # Для негативной обратной связи ищем проблемы
            problem_indicators = {
                'accuracy': ['неправильно', 'ошибка', 'wrong', 'error', 'incorrect'],
                'completeness': ['неполно', 'пропущено', 'incomplete', 'missing'],
                'performance': ['медленно', 'тормозит', 'slow', 'lag'],
                'usefulness': ['бесполезно', 'не помогает', 'useless', 'unhelpful']
            }

            for issue_type, indicators in problem_indicators.items():
                for indicator in indicators:
                    if indicator in text.lower():
                        issues.append(FeedbackIssue(
                            issue_type=issue_type,
                            description=f"Обнаружена проблема: {indicator}",
                            severity='medium',
                            context={'indicator': indicator, 'sentiment': sentiment.value}
                        ))
                        break

        elif sentiment == Sentiment.POSITIVE:
            # Для позитивной обратной связи отмечаем сильные стороны
            positive_indicators = {
                'accuracy': ['правильно', 'верно', 'correct', 'accurate'],
                'completeness': ['полно', 'завершено', 'complete'],
                'performance': ['быстро', 'оперативно', 'fast', 'quick'],
                'usefulness': ['полезно', 'помогает', 'useful', 'helpful']
            }

            for issue_type, indicators in positive_indicators.items():
                for indicator in indicators:
                    if indicator in text.lower():
                        issues.append(FeedbackIssue(
                            issue_type=issue_type,
                            description=f"Положительный аспект: {indicator}",
                            severity='low',  # Положительные аспекты имеют низкую severity
                            context={'indicator': indicator, 'sentiment': sentiment.value}
                        ))
                        break

        return issues

    async def _extract_issues_with_ai(self, text: str, sentiment: Sentiment,
                                    feedback_type: str) -> List[FeedbackIssue]:
        """Извлечение проблем с помощью AI"""
        if not self.hybrid_manager:
            return []

        prompt = f"""
        Проанализируй эту обратную связь пользователя и выдели конкретные проблемы или положительные аспекты:

        Обратная связь: "{text}"
        Sentiment: {sentiment.value}
        Тип: {feedback_type}

        Выдели:
        1. Конкретные проблемы (если негативная обратная связь)
        2. Положительные аспекты (если позитивная обратная связь)
        3. Уровень критичности (low, medium, high)

        Формат ответа: JSON с массивом issues, каждый с полями: issue_type, description, severity, context
        """

        try:
            response = await self.hybrid_manager.process_request({
                'query': prompt,
                'model_preference': 'local',  # Быстрый анализ
                'temperature': 0.2
            })

            return self._parse_ai_issues_response(response)
        except Exception as e:
            logger.warning(f"AI feedback analysis failed: {e}")
            return []

    def _parse_ai_issues_response(self, response: Any) -> List[FeedbackIssue]:
        """Парсинг ответа AI анализа"""
        try:
            # Простой парсер - можно улучшить
            response_text = str(response).lower()

            issues = []
            if 'issue_type' in response_text or 'issues' in response_text:
                # Если ответ содержит структурированные данные
                import json
                try:
                    data = json.loads(response_text)
                    if 'issues' in data:
                        for issue_data in data['issues']:
                            issues.append(FeedbackIssue(
                                issue_type=issue_data.get('issue_type', 'general'),
                                description=issue_data.get('description', ''),
                                severity=issue_data.get('severity', 'medium'),
                                context=issue_data.get('context', {})
                            ))
                except json.JSONDecodeError:
                    pass

            return issues
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return []

    async def _generate_feedback_insights(self, feedback: FeedbackAnalysis,
                                        issues: List[FeedbackIssue],
                                        sentiment: Sentiment) -> List[Insight]:
        """Генерация insights из обратной связи"""
        insights = []

        # Insight на основе sentiment
        if sentiment == Sentiment.NEGATIVE and issues:
            insight = Insight(
                id=f"feedback_negative_{int(datetime.now().timestamp())}",
                type=InsightType.FEEDBACK_INSIGHT,
                description="Пользователь выразил неудовлетворенность ответом агента",
                confidence=0.8,
                recommendation="Проанализировать причины неудовлетворенности и улучшить соответствующие аспекты",
                data={
                    'sentiment': sentiment.value,
                    'issues_count': len(issues),
                    'feedback_type': feedback.feedback_type
                }
            )
            insights.append(insight)

        elif sentiment == Sentiment.POSITIVE:
            insight = Insight(
                id=f"feedback_positive_{int(datetime.now().timestamp())}",
                type=InsightType.FEEDBACK_INSIGHT,
                description="Пользователь положительно оценил работу агента",
                confidence=0.7,
                recommendation="Поддерживать высокий уровень качества в выявленных сильных аспектах",
                data={
                    'sentiment': sentiment.value,
                    'feedback_type': feedback.feedback_type
                }
            )
            insights.append(insight)

        # Insights на основе конкретных проблем
        for issue in issues:
            if issue.severity in ['high', 'medium']:
                insight = Insight(
                    id=f"issue_{issue.issue_type}_{int(datetime.now().timestamp())}",
                    type=InsightType.FEEDBACK_INSIGHT,
                    description=f"Выявлена проблема: {issue.description}",
                    confidence=0.75,
                    recommendation=self._generate_issue_recommendation(issue),
                    data={
                        'issue_type': issue.issue_type,
                        'severity': issue.severity,
                        'context': issue.context
                    }
                )
                insights.append(insight)

        return insights

    def _generate_issue_recommendation(self, issue: FeedbackIssue) -> str:
        """Генерация рекомендации на основе проблемы"""
        recommendations = {
            'accuracy': "Улучшить механизмы проверки фактов и верификации информации",
            'completeness': "Добавить более подробные объяснения и охватить все аспекты вопроса",
            'relevance': "Улучшить понимание контекста и релевантности ответов",
            'usefulness': "Сосредоточиться на практической ценности предоставляемой информации",
            'performance': "Оптимизировать скорость работы и время отклика",
            'interface': "Улучшить пользовательский интерфейс и удобство использования"
        }

        return recommendations.get(issue.issue_type, "Проанализировать и устранить выявленную проблему")

    def _assess_criticality(self, sentiment: Sentiment, feedback_type: str,
                          issues: List[FeedbackIssue]) -> float:
        """Оценка критичности обратной связи (0-1)"""
        base_criticality = 0.5

        # Корректировка на основе sentiment
        if sentiment == Sentiment.NEGATIVE:
            base_criticality += 0.3
        elif sentiment == Sentiment.POSITIVE:
            base_criticality -= 0.2

        # Корректировка на основе количества и серьезности проблем
        if issues:
            high_severity_count = sum(1 for issue in issues if issue.severity == 'high')
            medium_severity_count = sum(1 for issue in issues if issue.severity == 'medium')

            criticality_adjustment = (high_severity_count * 0.2) + (medium_severity_count * 0.1)
            base_criticality += criticality_adjustment

        # Корректировка на основе типа обратной связи
        type_multipliers = {
            'accuracy': 1.2,  # Точность очень важна
            'performance': 1.1,  # Производительность важна
            'interface': 0.9,  # Интерфейс менее критичен
            'general': 1.0
        }

        type_multiplier = type_multipliers.get(feedback_type, 1.0)
        base_criticality *= type_multiplier

        return max(0.0, min(1.0, base_criticality))

    async def aggregate_feedback(self, feedbacks: List[FeedbackAnalysis]) -> Dict[str, Any]:
        """Агрегация обратной связи для анализа трендов"""
        if not feedbacks:
            return {'total_feedbacks': 0, 'summary': 'Нет обратной связи для анализа'}

        total_feedbacks = len(feedbacks)

        # Статистика по sentiment
        sentiment_stats = {'positive': 0, 'negative': 0, 'neutral': 0}
        for feedback in feedbacks:
            sentiment_stats[feedback.sentiment.value] += 1

        # Статистика по типам
        type_stats = {}
        for feedback in feedbacks:
            fb_type = feedback.feedback_type or 'general'
            type_stats[fb_type] = type_stats.get(fb_type, 0) + 1

        # Средняя критичность
        avg_criticality = sum(f.criticality for f in feedbacks) / total_feedbacks

        # Общие insights
        all_insights = []
        for feedback in feedbacks:
            if feedback.insights:
                all_insights.extend(feedback.insights)

        # Анализ трендов
        trend = self._analyze_feedback_trend(feedbacks)

        return {
            'total_feedbacks': total_feedbacks,
            'sentiment_distribution': sentiment_stats,
            'type_distribution': type_stats,
            'average_criticality': avg_criticality,
            'total_insights': len(all_insights),
            'trend': trend,
            'recommendations': self._generate_feedback_recommendations(sentiment_stats, type_stats)
        }

    def _analyze_feedback_trend(self, feedbacks: List[FeedbackAnalysis]) -> str:
        """Анализ тренда обратной связи"""
        if len(feedbacks) < 5:
            return 'insufficient_data'

        # Сортировка по времени
        sorted_feedbacks = sorted(feedbacks, key=lambda x: x.processed_at or datetime.now())

        # Анализ изменения средней критичности
        recent_feedbacks = sorted_feedbacks[-10:]  # Последние 10
        older_feedbacks = sorted_feedbacks[:-10] if len(sorted_feedbacks) > 10 else sorted_feedbacks[:5]

        recent_avg_criticality = sum(f.criticality for f in recent_feedbacks) / len(recent_feedbacks)
        older_avg_criticality = sum(f.criticality for f in older_feedbacks) / len(older_feedbacks)

        if recent_avg_criticality > older_avg_criticality * 1.2:
            return 'worsening'
        elif recent_avg_criticality < older_avg_criticality * 0.8:
            return 'improving'
        else:
            return 'stable'

    def _generate_feedback_recommendations(self, sentiment_stats: Dict[str, int],
                                         type_stats: Dict[str, int]) -> List[str]:
        """Генерация рекомендаций на основе статистики обратной связи"""
        recommendations = []

        total = sum(sentiment_stats.values())

        if total > 0:
            negative_ratio = sentiment_stats['negative'] / total

            if negative_ratio > 0.3:
                recommendations.append("Высокий уровень негативной обратной связи - требуется улучшение качества ответов")

            # Рекомендации по наиболее частым типам проблем
            most_common_type = max(type_stats.items(), key=lambda x: x[1])[0]

            type_recommendations = {
                'accuracy': "Сосредоточиться на повышении точности ответов",
                'performance': "Улучшить производительность и скорость работы",
                'completeness': "Добавить более полные и подробные ответы",
                'usefulness': "Повысить практическую ценность предоставляемой информации"
            }

            if most_common_type in type_recommendations:
                recommendations.append(type_recommendations[most_common_type])

        return recommendations

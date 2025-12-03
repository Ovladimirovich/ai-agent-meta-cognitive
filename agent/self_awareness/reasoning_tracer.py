import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict, Counter

from ..core.models import ReasoningStep
from .reasoning_trace import ReasoningTraceAnalyzer

logger = logging.getLogger("ReasoningTracer")


class ReasoningTracer:
    """
    Трассировщик рассуждений агента.

    Ведет подробную запись всех шагов рассуждений агента для:
    - Отладки и анализа работы
    - Мета-познания и самооценки
    - Улучшения будущих ответов
    - Прозрачности принятия решений
    """

    def __init__(self, max_steps: int = 1000, enable_advanced_analysis: bool = True):
        self.trace: List[ReasoningStep] = []
        self.max_steps = max_steps
        self.session_start = datetime.now()
        self.enable_advanced_analysis = enable_advanced_analysis

        # Продвинутая аналитика
        self.trace_analyzer = ReasoningTraceAnalyzer() if enable_advanced_analysis else None
        self.pattern_cache: Dict[str, Any] = {}
        self.quality_metrics: Dict[str, float] = {}
        self.real_time_insights: List[Dict[str, Any]] = []

        # Статистика для анализа паттернов
        self.step_sequences: List[List[str]] = []
        self.decision_points: List[Dict[str, Any]] = []
        self.confidence_history: List[float] = []

        logger.info("🚀 ReasoningTracer initialized with advanced analysis" if enable_advanced_analysis else "🚀 ReasoningTracer initialized")

    def add_step(self, step_type: str, description: str, data: Optional[Dict[str, Any]] = None):
        """
        Добавление шага в трассировку рассуждений.

        Args:
            step_type: Тип шага (analysis, strategy_selection, execution, etc.)
            description: Описание шага
            data: Дополнительные данные шага
        """
        step = ReasoningStep(
            step_type=step_type,
            description=description,
            timestamp=datetime.now(),
            data=data or {}
        )

        self.trace.append(step)

        # Ограничение количества шагов для предотвращения переполнения памяти
        if len(self.trace) > self.max_steps:
            # Удаляем самые старые шаги
            removed_count = len(self.trace) - self.max_steps
            self.trace = self.trace[removed_count:]
            logger.warning(f"Trace limit exceeded, removed {removed_count} oldest steps")

        logger.debug(f"Added reasoning step: {step_type} - {description}")

    def get_trace_summary(self) -> Dict[str, Any]:
        """
        Получение сводки трассировки.

        Returns:
            Словарь со статистикой и ключевыми метриками
        """
        if not self.trace:
            return {"total_steps": 0, "duration": 0.0, "step_types": []}

        # Подсчет типов шагов
        step_types = {}
        for step in self.trace:
            step_types[step.step_type] = step_types.get(step.step_type, 0) + 1

        # Расчет длительности сессии
        duration = (datetime.now() - self.session_start).total_seconds()

        # Извлечение ключевых решений
        key_decisions = self._extract_key_decisions()

        return {
            "total_steps": len(self.trace),
            "step_types": step_types,
            "duration": duration,
            "key_decisions": key_decisions,
            "average_steps_per_minute": len(self.trace) / max(duration / 60, 1)
        }

    def _extract_key_decisions(self) -> List[Dict[str, Any]]:
        """
        Извлечение ключевых решений из трассировки.

        Ищет важные шаги принятия решений.
        """
        key_decisions = []

        for step in self.trace:
            if step.step_type in ["strategy_selection", "tool_selection", "model_choice"]:
                key_decisions.append({
                    "step_type": step.step_type,
                    "description": step.description,
                    "timestamp": step.timestamp.isoformat(),
                    "data": step.data
                })

        return key_decisions

    def get_recent_steps(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получение последних шагов трассировки.

        Args:
            limit: Максимальное количество шагов

        Returns:
            Список последних шагов в виде словарей
        """
        recent_steps = self.trace[-limit:] if limit > 0 else self.trace
        return [step.dict() for step in recent_steps]

    def get_steps_by_type(self, step_type: str) -> List[Dict[str, Any]]:
        """
        Получение шагов определенного типа.

        Args:
            step_type: Тип шага для фильтрации

        Returns:
            Список шагов указанного типа
        """
        matching_steps = [step for step in self.trace if step.step_type == step_type]
        return [step.dict() for step in matching_steps]

    def get_execution_flow(self) -> List[Dict[str, Any]]:
        """
        Получение потока выполнения.

        Возвращает последовательность шагов выполнения с временными метками.
        """
        execution_steps = []
        current_flow = []

        for step in self.trace:
            if step.step_type in ["analysis", "strategy_selection", "execution", "completion"]:
                current_flow.append({
                    "step_type": step.step_type,
                    "description": step.description,
                    "timestamp": step.timestamp.isoformat(),
                    "duration_from_start": (step.timestamp - self.session_start).total_seconds()
                })

        return current_flow

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Анализ производительности на основе трассировки.

        Returns:
            Метрики производительности
        """
        if not self.trace:
            return {"error": "No trace data available"}

        # Анализ частоты шагов
        step_frequency = {}
        total_duration = (datetime.now() - self.session_start).total_seconds()

        for step in self.trace:
            step_frequency[step.step_type] = step_frequency.get(step.step_type, 0) + 1

        # Расчет средней продолжительности между шагами
        if len(self.trace) > 1:
            time_diffs = []
            for i in range(1, len(self.trace)):
                diff = (self.trace[i].timestamp - self.trace[i-1].timestamp).total_seconds()
                time_diffs.append(diff)

            avg_step_interval = sum(time_diffs) / len(time_diffs)
        else:
            avg_step_interval = 0.0

        return {
            "total_steps": len(self.trace),
            "session_duration": total_duration,
            "steps_per_minute": len(self.trace) / max(total_duration / 60, 1),
            "step_frequency": step_frequency,
            "average_step_interval": avg_step_interval,
            "most_common_step": max(step_frequency, key=step_frequency.get) if step_frequency else None
        }

    def find_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Поиск узких мест в обработке.

        Ищет шаги с необычно долгим временем выполнения.
        """
        bottlenecks = []

        if len(self.trace) < 2:
            return bottlenecks

        # Расчет интервалов между шагами
        intervals = []
        for i in range(1, len(self.trace)):
            interval = (self.trace[i].timestamp - self.trace[i-1].timestamp).total_seconds()
            intervals.append(interval)

        if not intervals:
            return bottlenecks

        # Расчет среднего и стандартного отклонения
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5

        # Поиск шагов с интервалом > среднее + 2*std_dev
        threshold = avg_interval + 2 * std_dev

        for i, interval in enumerate(intervals):
            if interval > threshold:
                bottlenecks.append({
                    "step_index": i + 1,
                    "step_type": self.trace[i + 1].step_type,
                    "description": self.trace[i + 1].description,
                    "interval": interval,
                    "threshold": threshold,
                    "slowdown_ratio": interval / avg_interval
                })

        return bottlenecks

    def export_trace(self, format: str = "json") -> Any:
        """
        Экспорт трассировки в различных форматах.

        Args:
            format: Формат экспорта (json, dict, text)

        Returns:
            Трассировка в указанном формате
        """
        if format == "json":
            return [step.dict() for step in self.trace]
        elif format == "dict":
            return {
                "session_start": self.session_start.isoformat(),
                "total_steps": len(self.trace),
                "steps": [step.dict() for step in self.trace],
                "summary": self.get_trace_summary()
            }
        elif format == "text":
            lines = [f"Reasoning Trace - Session started: {self.session_start}"]
            lines.append(f"Total steps: {len(self.trace)}")
            lines.append("")

            for i, step in enumerate(self.trace, 1):
                lines.append(f"{i}. [{step.step_type}] {step.description}")
                lines.append(f"   Time: {step.timestamp}")
                if step.data:
                    lines.append(f"   Data: {step.data}")
                lines.append("")

            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def clear_trace(self):
        """Очистка трассировки"""
        self.trace.clear()
        self.session_start = datetime.now()
        logger.info("🔄 Reasoning trace cleared")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Получение информации об использовании памяти трассировкой"""
        return {
            "trace_length": len(self.trace),
            "max_steps": self.max_steps,
            "memory_usage_mb": len(self.trace) * 0.001,  # Примерная оценка
            "can_add_more": len(self.trace) < self.max_steps
        }

    # Продвинутые методы аналитики

    async def analyze_reasoning_patterns(self) -> Dict[str, Any]:
        """
        Продвинутый анализ паттернов рассуждений.

        Returns:
            Анализ паттернов с использованием ReasoningTraceAnalyzer
        """
        if not self.enable_advanced_analysis or not self.trace_analyzer:
            return {"error": "Advanced analysis disabled"}

        trace_dicts = [step.dict() for step in self.trace]
        analysis = await self.trace_analyzer.analyze_trace(trace_dicts)

        # Кэширование результатов
        self.pattern_cache = {
            'analysis': analysis,
            'timestamp': datetime.now(),
            'trace_length': len(self.trace)
        }

        return analysis.dict() if hasattr(analysis, 'dict') else analysis

    def detect_real_time_patterns(self) -> List[Dict[str, Any]]:
        """
        Обнаружение паттернов в реальном времени.

        Анализирует последние шаги для выявления формирующихся паттернов.
        """
        if len(self.trace) < 3:
            return []

        recent_steps = self.trace[-10:]  # Последние 10 шагов
        patterns = []

        # Анализ последовательностей
        step_types = [step.step_type for step in recent_steps]

        # Поиск повторяющихся паттернов
        for i in range(len(step_types) - 2):
            pattern = step_types[i:i+3]
            if step_types.count(pattern[0]) > 2:  # Повторяющийся тип шага
                patterns.append({
                    'type': 'repetitive_sequence',
                    'pattern': pattern,
                    'description': f"Повторяющаяся последовательность: {' -> '.join(pattern)}",
                    'frequency': step_types.count(pattern[0]),
                    'insight': 'Возможно, цикл или неэффективная стратегия'
                })

        # Анализ точек принятия решений
        decision_steps = [step for step in recent_steps if 'select' in step.step_type or 'choose' in step.step_type]
        if len(decision_steps) > 3:
            patterns.append({
                'type': 'high_decision_density',
                'description': f"Высокая плотность решений: {len(decision_steps)} решений в последних {len(recent_steps)} шагах",
                'insight': 'Возможно, избыточное ветвление или нерешительность'
            })

        # Анализ эффективности
        if len(recent_steps) >= 5:
            avg_interval = self._calculate_recent_avg_interval(recent_steps)
            if avg_interval > 10:  # Более 10 секунд между шагами
                patterns.append({
                    'type': 'slow_reasoning',
                    'description': f"Замедленное рассуждение: {avg_interval:.1f} сек между шагами",
                    'insight': 'Возможно, узкое место в обработке'
                })

        self.real_time_insights.extend(patterns)
        return patterns

    def _calculate_recent_avg_interval(self, steps: List[ReasoningStep]) -> float:
        """Расчет среднего интервала между шагами в списке"""
        if len(steps) < 2:
            return 0.0

        intervals = []
        for i in range(1, len(steps)):
            interval = (steps[i].timestamp - steps[i-1].timestamp).total_seconds()
            intervals.append(interval)

        return sum(intervals) / len(intervals) if intervals else 0.0

    async def get_reasoning_quality_score(self) -> Dict[str, Any]:
        """
        Комплексная оценка качества рассуждений.

        Returns:
            Метрики качества рассуждений
        """
        if not self.trace:
            return {'quality_score': 0.0, 'error': 'No trace data'}

        # Базовые метрики
        base_metrics = self.analyze_performance()

        # Анализ паттернов (если доступен)
        pattern_score = 0.5  # Значение по умолчанию
        if self.enable_advanced_analysis and self.pattern_cache:
            analysis = self.pattern_cache.get('analysis', {})
            pattern_score = analysis.get('quality_score', 0.5)
        elif self.enable_advanced_analysis:
            # Выполнить анализ паттернов
            pattern_analysis = await self.analyze_reasoning_patterns()
            pattern_score = pattern_analysis.get('quality_score', 0.5)

        # Метрики эффективности
        efficiency_score = self._calculate_efficiency_score(base_metrics)

        # Метрики последовательности
        consistency_score = self._calculate_consistency_score()

        # Комплексная оценка
        quality_score = (
            pattern_score * 0.4 +
            efficiency_score * 0.3 +
            consistency_score * 0.3
        )

        # Нормализация
        quality_score = max(0.0, min(1.0, quality_score))

        self.quality_metrics = {
            'overall_quality': quality_score,
            'pattern_score': pattern_score,
            'efficiency_score': efficiency_score,
            'consistency_score': consistency_score,
            'timestamp': datetime.now()
        }

        return {
            'quality_score': quality_score,
            'components': {
                'pattern_quality': pattern_score,
                'efficiency': efficiency_score,
                'consistency': consistency_score
            },
            'recommendations': self._generate_quality_recommendations(quality_score, base_metrics)
        }

    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Расчет оценки эффективности"""
        if 'error' in metrics:
            return 0.3

        # Оценка на основе шагов в минуту (оптимально 10-20 шагов в минуту)
        steps_per_minute = metrics.get('steps_per_minute', 0)
        efficiency_steps = min(1.0, steps_per_minute / 15.0)  # Нормализация к 15 шагам/мин

        # Оценка на основе среднего интервала (оптимально 2-5 секунд)
        avg_interval = metrics.get('average_step_interval', 10)
        efficiency_time = max(0.0, 1.0 - (avg_interval - 3.0) / 10.0)  # Нормализация

        return (efficiency_steps + efficiency_time) / 2.0

    def _calculate_consistency_score(self) -> float:
        """Расчет оценки последовательности рассуждений"""
        if len(self.trace) < 3:
            return 0.5

        # Анализ последовательности типов шагов
        step_types = [step.step_type for step in self.trace]

        # Подсчет переходов между типами
        transitions = defaultdict(int)
        for i in range(len(step_types) - 1):
            transition = f"{step_types[i]} -> {step_types[i+1]}"
            transitions[transition] += 1

        # Оценка разнообразия переходов (не слишком много одинаковых)
        unique_transitions = len(transitions)
        total_transitions = sum(transitions.values())

        if total_transitions == 0:
            return 0.5

        # Идеально: разнообразие переходов, но не хаос
        diversity_ratio = unique_transitions / total_transitions
        consistency_score = 1.0 - abs(diversity_ratio - 0.3)  # Оптимально ~30% уникальных переходов

        return max(0.0, min(1.0, consistency_score))

    def _generate_quality_recommendations(self, quality_score: float, metrics: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по улучшению качества"""
        recommendations = []

        if quality_score < 0.5:
            recommendations.append("Общее качество рассуждений низкое. Рекомендуется анализ стратегии.")

        # Рекомендации по эффективности
        steps_per_minute = metrics.get('steps_per_minute', 0)
        if steps_per_minute < 5:
            recommendations.append("Слишком медленное рассуждение. Рассмотрите оптимизацию шагов.")
        elif steps_per_minute > 30:
            recommendations.append("Слишком быстрое рассуждение. Возможно, недостаточная глубина анализа.")

        # Рекомендации по паттернам
        if self.pattern_cache:
            analysis = self.pattern_cache.get('analysis', {})
            issues = getattr(analysis, 'issues', []) if hasattr(analysis, 'issues') else []
            for issue in issues[:3]:  # Ограничим 3 рекомендациями
                recommendations.append(f"Исправить: {issue}")

        return recommendations if recommendations else ["Качество рассуждений в норме."]

    def track_decision_point(self, decision_type: str, options: List[Any],
                           chosen_option: Any, confidence: float):
        """
        Отслеживание точек принятия решений.

        Args:
            decision_type: Тип решения (strategy, tool, model)
            options: Доступные варианты
            chosen_option: Выбранный вариант
            confidence: Уверенность в выборе
        """
        decision = {
            'type': decision_type,
            'options_count': len(options),
            'chosen_option': str(chosen_option),
            'confidence': confidence,
            'timestamp': datetime.now(),
            'step_index': len(self.trace)
        }

        self.decision_points.append(decision)
        self.confidence_history.append(confidence)

    def get_decision_analysis(self) -> Dict[str, Any]:
        """Анализ принятых решений"""
        if not self.decision_points:
            return {'total_decisions': 0}

        # Статистика по типам решений
        decision_types = Counter(d['type'] for d in self.decision_points)

        # Анализ уверенности
        confidences = [d['confidence'] for d in self.decision_points]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Анализ разнообразия выборов
        unique_choices = len(set(d['chosen_option'] for d in self.decision_points))
        total_decisions = len(self.decision_points)
        choice_diversity = unique_choices / total_decisions if total_decisions > 0 else 0.0

        return {
            'total_decisions': total_decisions,
            'decision_types': dict(decision_types),
            'average_confidence': avg_confidence,
            'choice_diversity': choice_diversity,
            'confidence_trend': self._analyze_confidence_trend()
        }

    def _analyze_confidence_trend(self) -> str:
        """Анализ тренда уверенности в решениях"""
        if len(self.confidence_history) < 3:
            return 'insufficient_data'

        # Сравнение первой и последней трети
        third = len(self.confidence_history) // 3
        first_third = self.confidence_history[:third]
        last_third = self.confidence_history[-third:]

        avg_first = sum(first_third) / len(first_third)
        avg_last = sum(last_third) / len(last_third)

        if avg_last > avg_first + 0.1:
            return 'increasing'
        elif avg_last < avg_first - 0.1:
            return 'decreasing'
        else:
            return 'stable'

    async def generate_reasoning_report(self) -> Dict[str, Any]:
        """
        Генерация комплексного отчета о рассуждениях.

        Returns:
            Полный отчет с анализом и рекомендациями
        """
        report = {
            'generated_at': datetime.now(),
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'duration': (datetime.now() - self.session_start).total_seconds(),
                'total_steps': len(self.trace)
            }
        }

        # Базовая статистика
        report['basic_stats'] = self.get_trace_summary()

        # Продвинутый анализ
        if self.enable_advanced_analysis:
            try:
                pattern_analysis = await self.analyze_reasoning_patterns()
                report['pattern_analysis'] = pattern_analysis
            except Exception as e:
                logger.warning(f"Pattern analysis failed: {e}")
                report['pattern_analysis'] = {'error': str(e)}

        # Анализ качества
        try:
            quality_analysis = await self.get_reasoning_quality_score()
            report['quality_analysis'] = quality_analysis
        except Exception as e:
            logger.warning(f"Quality analysis failed: {e}")
            report['quality_analysis'] = {'error': str(e)}

        # Анализ решений
        report['decision_analysis'] = self.get_decision_analysis()

        # Узкие места
        report['bottlenecks'] = self.find_bottlenecks()

        # Real-time инсайты
        report['real_time_insights'] = self.real_time_insights[-10:]  # Последние 10

        # Рекомендации
        report['recommendations'] = self._compile_recommendations(report)

        return report

    def _compile_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Составление итоговых рекомендаций"""
        recommendations = []

        # Из quality анализа
        quality = report.get('quality_analysis', {})
        if isinstance(quality, dict) and 'recommendations' in quality:
            recommendations.extend(quality['recommendations'])

        # Из pattern анализа
        patterns = report.get('pattern_analysis', {})
        if isinstance(patterns, dict) and 'recommendations' in patterns:
            pattern_recs = patterns.get('recommendations', [])
            if isinstance(pattern_recs, list):
                recommendations.extend(pattern_recs[:3])  # Ограничим

        # Из bottlenecks
        bottlenecks = report.get('bottlenecks', [])
        if bottlenecks:
            recommendations.append(f"Обнаружено {len(bottlenecks)} узких мест в обработке")

        # Из decision анализа
        decisions = report.get('decision_analysis', {})
        avg_confidence = decisions.get('average_confidence', 0.5)
        if avg_confidence < 0.6:
            recommendations.append("Низкая уверенность в решениях. Рассмотрите улучшение критериев выбора.")

        return list(set(recommendations))  # Удаление дубликатов

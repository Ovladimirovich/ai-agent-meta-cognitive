import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.models import (
    ReasoningAnalysis, ReasoningPattern, ReasoningEfficiency,
    AgentInteraction
)

logger = logging.getLogger(__name__)


class ReasoningPatternDetector:
    """Детектор паттернов в трассировке рассуждений"""

    def __init__(self):
        self.pattern_templates = {
            'linear_reasoning': {
                'description': 'Линейное рассуждение без ветвлений',
                'indicators': ['step_by_step', 'sequential']
            },
            'exploratory_reasoning': {
                'description': 'Исследовательское рассуждение с ветвлениями',
                'indicators': ['explore', 'alternative', 'branch']
            },
            'iterative_refinement': {
                'description': 'Итеративное улучшение решения',
                'indicators': ['refine', 'improve', 'iteration']
            },
            'pattern_matching': {
                'description': 'Сопоставление с известными паттернами',
                'indicators': ['similar', 'pattern', 'match']
            }
        }

    async def detect_patterns(self, trace: List[Dict]) -> List[ReasoningPattern]:
        """Обнаружение паттернов рассуждений"""
        patterns = []

        for pattern_name, template in self.pattern_templates.items():
            pattern = await self._analyze_pattern(trace, pattern_name, template)
            if pattern:
                patterns.append(pattern)

        return patterns

    async def _analyze_pattern(self, trace: List[Dict], pattern_name: str,
                              template: Dict) -> Optional[ReasoningPattern]:
        """Анализ конкретного паттерна"""
        indicators = template['indicators']
        matches = 0
        examples = []

        for step in trace:
            step_text = str(step.get('description', '')).lower()
            if any(indicator in step_text for indicator in indicators):
                matches += 1
                examples.append(step)

        if matches > 0:
            frequency = matches / len(trace)
            effectiveness = self._calculate_pattern_effectiveness(pattern_name, trace)

            return ReasoningPattern(
                pattern_type=pattern_name,
                frequency=frequency,
                description=template['description'],
                effectiveness=effectiveness,
                examples=examples[:3]  # Ограничим примерами
            )

        return None

    def _calculate_pattern_effectiveness(self, pattern_name: str, trace: List[Dict]) -> float:
        """Расчет эффективности паттерна"""
        # Простая эвристика - можно улучшить с ML
        if pattern_name == 'linear_reasoning':
            return 0.7  # Надежный, но может быть неоптимальным
        elif pattern_name == 'exploratory_reasoning':
            return 0.8  # Хорошо для сложных задач
        elif pattern_name == 'iterative_refinement':
            return 0.9  # Высокая эффективность для оптимизации
        elif pattern_name == 'pattern_matching':
            return 0.6  # Зависит от качества паттернов
        return 0.5


class ReasoningEfficiencyAnalyzer:
    """Анализатор эффективности рассуждений"""

    def __init__(self):
        self.baseline_metrics = {
            'avg_step_time': 2.0,  # секунды
            'optimal_branching': 1.5,
            'max_depth': 10
        }

    async def analyze_efficiency(self, trace: List[Dict]) -> ReasoningEfficiency:
        """Анализ эффективности трассировки"""
        steps_count = len(trace)

        # Анализ времени выполнения шагов
        step_times = []
        for step in trace:
            if 'timestamp' in step:
                step_times.append(step.get('execution_time', 1.0))
        average_step_time = sum(step_times) / len(step_times) if step_times else 1.0

        # Анализ ветвлений (branching factor)
        branching_factor = self._calculate_branching_factor(trace)

        # Анализ глубины рассуждений
        depth_score = self._calculate_depth_score(trace)

        # Общий score оптимизации
        optimization_score = self._calculate_optimization_score(
            steps_count, average_step_time, branching_factor, depth_score
        )

        return ReasoningEfficiency(
            steps_count=steps_count,
            average_step_time=average_step_time,
            branching_factor=branching_factor,
            depth_score=depth_score,
            optimization_score=optimization_score
        )

    def _calculate_branching_factor(self, trace: List[Dict]) -> float:
        """Расчет фактора ветвления"""
        branches = 0
        total_steps = len(trace)

        for step in trace:
            if step.get('type') in ['branch', 'alternative', 'explore']:
                branches += 1

        return branches / total_steps if total_steps > 0 else 0

    def _calculate_depth_score(self, trace: List[Dict]) -> float:
        """Расчет оценки глубины рассуждений"""
        max_depth = 0
        current_depth = 0

        for step in trace:
            step_type = step.get('type', '')
            if step_type in ['analyze', 'reason', 'evaluate']:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif step_type in ['conclude', 'decide']:
                current_depth = max(0, current_depth - 1)

        # Нормализация к 0-1
        return min(max_depth / self.baseline_metrics['max_depth'], 1.0)

    def _calculate_optimization_score(self, steps: int, avg_time: float,
                                    branching: float, depth: float) -> float:
        """Расчет общего score оптимизации"""
        # Весовые коэффициенты
        time_weight = 0.3
        steps_weight = 0.3
        branching_weight = 0.2
        depth_weight = 0.2

        # Нормализованные метрики (0-1, где 1 - оптимально)
        time_score = min(self.baseline_metrics['avg_step_time'] / avg_time, 1.0)
        steps_score = min(10 / steps, 1.0)  # Предполагаем оптимальное 10 шагов
        branching_score = 1.0 - abs(branching - self.baseline_metrics['optimal_branching'])
        branching_score = max(0, branching_score)
        depth_score = min(depth, 1.0)

        return (time_score * time_weight +
                steps_score * steps_weight +
                branching_score * branching_weight +
                depth_score * depth_weight)


class ReasoningTraceAnalyzer:
    """Анализатор трассировки рассуждений"""

    def __init__(self, hybrid_manager=None):
        self.hybrid_manager = hybrid_manager
        self.pattern_detector = ReasoningPatternDetector()
        self.efficiency_analyzer = ReasoningEfficiencyAnalyzer()

    async def analyze_trace(self, trace: List[Dict]) -> ReasoningAnalysis:
        """Анализ трассировки рассуждений"""
        if not trace:
            return ReasoningAnalysis(
                patterns=[],
                efficiency=ReasoningEfficiency(
                    steps_count=0, average_step_time=0, branching_factor=0,
                    depth_score=0, optimization_score=0
                ),
                quality_score=0.0,
                issues=["Пустая трассировка рассуждений"],
                recommendations=["Добавить шаги рассуждений"]
            )

        # Обнаружение паттернов рассуждений
        patterns = await self.pattern_detector.detect_patterns(trace)

        # Анализ эффективности
        efficiency = await self.efficiency_analyzer.analyze_efficiency(trace)

        # Оценка качества рассуждений
        quality_score = self._calculate_reasoning_quality(trace, patterns, efficiency)

        # Выявление проблемных областей
        issues = self._identify_reasoning_issues(trace, patterns, efficiency)

        # Генерация рекомендаций
        recommendations = self._generate_recommendations(issues, patterns, efficiency)

        return ReasoningAnalysis(
            patterns=patterns,
            efficiency=efficiency,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations
        )

    def _calculate_reasoning_quality(self, trace: List[Dict],
                                   patterns: List[ReasoningPattern],
                                   efficiency: ReasoningEfficiency) -> float:
        """Расчет качества рассуждений"""
        base_score = 0.5  # Базовый score

        # Факторы качества
        pattern_diversity = len(patterns) / 4.0  # Нормализуем к 4 паттернам
        efficiency_score = efficiency.optimization_score
        completeness_score = self._assess_completeness(trace)

        # Взвешенная сумма
        quality = (pattern_diversity * 0.3 +
                  efficiency_score * 0.4 +
                  completeness_score * 0.3)

        return min(max(quality, 0.0), 1.0)

    def _assess_completeness(self, trace: List[Dict]) -> float:
        """Оценка полноты рассуждений"""
        required_steps = ['analyze', 'reason', 'evaluate', 'conclude']
        found_steps = set()

        for step in trace:
            step_type = step.get('type', '').lower()
            for required in required_steps:
                if required in step_type:
                    found_steps.add(required)

        completeness = len(found_steps) / len(required_steps)
        return completeness

    def _identify_reasoning_issues(self, trace: List[Dict],
                                 patterns: List[ReasoningPattern],
                                 efficiency: ReasoningEfficiency) -> List[str]:
        """Выявление проблем в рассуждениях"""
        issues = []

        # Проверка эффективности
        if efficiency.optimization_score < 0.5:
            issues.append("Низкая эффективность рассуждений")

        if efficiency.average_step_time > 5.0:
            issues.append("Слишком долгое время выполнения шагов")

        # Проверка разнообразия паттернов
        if len(patterns) < 2:
            issues.append("Ограниченное разнообразие паттернов рассуждений")

        # Проверка глубины
        if efficiency.depth_score < 0.3:
            issues.append("Недостаточная глубина анализа")

        # Проверка ветвлений
        if efficiency.branching_factor > 0.8:
            issues.append("Избыточное ветвление - возможна неэффективность")

        return issues

    def _generate_recommendations(self, issues: List[str],
                                patterns: List[ReasoningPattern],
                                efficiency: ReasoningEfficiency) -> List[str]:
        """Генерация рекомендаций по улучшению"""
        recommendations = []

        for issue in issues:
            if "эффективность" in issue.lower():
                recommendations.append("Оптимизировать последовательность шагов рассуждений")
            elif "время" in issue.lower():
                recommendations.append("Упростить или распараллелить шаги анализа")
            elif "разнообразие" in issue.lower():
                recommendations.append("Внедрить больше различных стратегий рассуждений")
            elif "глубина" in issue.lower():
                recommendations.append("Увеличить глубину анализа проблем")
            elif "ветвление" in issue.lower():
                recommendations.append("Сократить количество альтернативных ветвей")

        # Дополнительные рекомендации на основе паттернов
        pattern_types = {p.pattern_type for p in patterns}
        if 'linear_reasoning' in pattern_types and len(patterns) == 1:
            recommendations.append("Добавить исследовательские элементы в рассуждения")

        return recommendations

    async def _analyze_with_ai(self, analysis_type: str, trace_data: Dict) -> Dict:
        """Использование AI для глубокого анализа (если доступен hybrid_manager)"""
        if not self.hybrid_manager:
            return {}

        prompt = self._build_analysis_prompt(analysis_type, trace_data)

        try:
            response = await self.hybrid_manager.process_request({
                'query': prompt,
                'model_preference': 'best_available',
                'temperature': 0.3
            })

            return self._parse_ai_analysis(response)
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return {}

    def _build_analysis_prompt(self, analysis_type: str, trace_data: Dict) -> str:
        """Построение промпта для AI анализа"""
        if analysis_type == 'quality':
            return f"""
            Проанализируй качество рассуждений в следующей трассировке:

            Трассировка: {trace_data}

            Оцени:
            1. Логичность рассуждений
            2. Полноту анализа
            3. Эффективность подхода

            Верни оценку качества от 0 до 1 и краткое объяснение.
            """

        return "Проанализируй предоставленные данные рассуждений"

    def _parse_ai_analysis(self, response: Any) -> Dict:
        """Парсинг ответа AI анализа"""
        # Простой парсер - можно улучшить
        response_text = str(response).lower()

        result = {}
        if 'качество' in response_text:
            # Извлечение числовой оценки
            import re
            numbers = re.findall(r'0\.\d+', response_text)
            if numbers:
                result['quality_score'] = float(numbers[0])

        return result

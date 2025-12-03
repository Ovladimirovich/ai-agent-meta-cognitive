import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from ..core.models import QueryAnalysis

logger = logging.getLogger("ConfidenceCalculator")


class ConfidenceCalculator:
    """
    Калькулятор уверенности агента.

    Оценивает уровень уверенности в ответе на основе различных факторов:
    - Качества ответа модели
    - Успешности выполнения инструментов
    - Релевантности контекста
    - Исторической производительности
    """

    def __init__(self):
        self.historical_performance: Dict[str, float] = {}
        self.intent_performance: Dict[str, Dict[str, float]] = {}

        # Машинное обучение для предсказания уверенности
        self.ml_model = None
        self.training_data: List[Tuple[List[float], float]] = []
        self.feature_weights = {
            'model_confidence': 0.4,
            'tool_success_rate': 0.3,
            'context_relevance': 0.2,
            'historical_performance': 0.1,
            'response_length': 0.05,
            'tool_count': -0.05,
            'complexity_penalty': -0.1,
            'time_factor': 0.02
        }

        # Статистика для обучения
        self.prediction_history: List[Tuple[List[float], float, float]] = []  # features, predicted, actual
        self.accuracy_stats = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'mean_error': 0.0,
            'last_calibration': None
        }

        logger.info("🚀 ConfidenceCalculator with ML initialized")

    def calculate(self, result: Any, analysis: QueryAnalysis) -> float:
        """
        Расчет уровня уверенности в ответе.

        Args:
            result: Результат выполнения запроса
            analysis: Анализ запроса

        Returns:
            Уровень уверенности от 0.0 до 1.0
        """
        try:
            confidence_factors = {
                'model_confidence': self._get_model_confidence(result),
                'tool_success_rate': self._calculate_tool_success_rate(analysis.required_tools),
                'context_relevance': self._assess_context_relevance(result, analysis),
                'historical_performance': self._get_historical_performance(analysis.intent)
            }

            # Взвешенное среднее с коэффициентами
            weights = {
                'model_confidence': 0.4,
                'tool_success_rate': 0.3,
                'context_relevance': 0.2,
                'historical_performance': 0.1
            }

            confidence = sum(
                confidence_factors[factor] * weights[factor]
                for factor in confidence_factors
            )

            # Нормализация в диапазон [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            logger.debug(f"Calculated confidence: {confidence:.3f} for intent '{analysis.intent}'")
            return confidence

        except Exception as e:
            logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.1  # Минимальная уверенность при ошибке

    def _get_model_confidence(self, result: Any) -> float:
        """
        Оценка уверенности на основе качества ответа модели.

        Анализирует:
        - Длину ответа (слишком короткие могут быть неуверенными)
        - Наличие конкретных данных
        - Отсутствие фраз неуверенности
        """
        if not result or not isinstance(result, str):
            return 0.3  # Низкая уверенность для пустых или нестроковых результатов

        result_str = str(result).strip()

        # Базовая оценка по длине
        length_score = min(1.0, len(result_str) / 100.0)  # Максимум за 100 символов

        # Штраф за фразы неуверенности
        uncertainty_phrases = [
            "я не знаю", "не уверен", "возможно", "может быть",
            "кажется", "похоже", "наверное", "вероятно"
        ]

        uncertainty_penalty = 0.0
        for phrase in uncertainty_phrases:
            if phrase.lower() in result_str.lower():
                uncertainty_penalty += 0.1

        uncertainty_penalty = min(0.5, uncertainty_penalty)  # Максимальный штраф 0.5

        # Бонус за конкретные данные
        specificity_bonus = 0.0
        if any(char.isdigit() for char in result_str):
            specificity_bonus += 0.1  # Цифры
        if len(result_str.split()) > 10:
            specificity_bonus += 0.1  # Длинный ответ
        if "http" in result_str or "www" in result_str:
            specificity_bonus += 0.1  # Ссылки

        confidence = length_score - uncertainty_penalty + specificity_bonus
        return max(0.0, min(1.0, confidence))

    def _calculate_tool_success_rate(self, required_tools: list[str]) -> float:
        """
        Расчет успешности выполнения инструментов.

        Для первой фазы - простая оценка на основе количества инструментов.
        В будущем будет интегрирована с реальной статистикой выполнения.
        """
        if not required_tools:
            return 0.8  # Высокая уверенность для запросов без инструментов

        # Базовая оценка: меньше инструментов = выше уверенность
        tool_count_penalty = min(0.3, len(required_tools) * 0.1)

        # Предполагаемая успешность инструментов (в будущем - реальная статистика)
        assumed_success_rate = 0.85

        confidence = assumed_success_rate - tool_count_penalty
        return max(0.0, min(1.0, confidence))

    def _assess_context_relevance(self, result: Any, analysis: QueryAnalysis) -> float:
        """
        Оценка релевантности контекста.

        Анализирует соответствие результата исходному запросу и контексту.
        """
        if not result or not analysis:
            return 0.5

        result_str = str(result).lower()
        query_str = analysis.context.get("original_query", "").lower() if analysis.context else ""

        if not query_str:
            return 0.7  # Средняя уверенность без исходного запроса

        # Простая оценка релевантности по совпадению ключевых слов
        query_words = set(query_str.split())
        result_words = set(result_str.split())

        # Коэффициент Жаккара для оценки пересечения
        intersection = query_words.intersection(result_words)
        union = query_words.union(result_words)

        if not union:
            return 0.5

        jaccard_similarity = len(intersection) / len(union)

        # Преобразование в оценку уверенности
        relevance_score = min(1.0, jaccard_similarity * 2.0)  # Умножаем для лучшего диапазона

        return relevance_score

    def _get_historical_performance(self, intent: str) -> float:
        """
        Получение исторической производительности для данного намерения.

        В первой фазе - простая оценка на основе типа намерения.
        """
        # Базовые оценки по типам намерений
        intent_base_scores = {
            "greeting": 0.95,  # Приветствия обычно успешны
            "question": 0.80,  # Вопросы зависят от сложности
            "search": 0.75,    # Поиск может быть неточным
            "analyze": 0.70,   # Анализ зависит от данных
            "create": 0.65,    # Создание требует креативности
            "help": 0.85       # Помощь обычно успешна
        }

        base_score = intent_base_scores.get(intent, 0.7)  # Значение по умолчанию

        # В будущем здесь будет реальная историческая статистика
        # Пока возвращаем базовую оценку
        return base_score

    def update_historical_performance(self, intent: str, actual_confidence: float, success: bool):
        """
        Обновление исторической производительности.

        Args:
            intent: Тип намерения
            actual_confidence: Фактическая уверенность
            success: Успешность выполнения
        """
        if intent not in self.intent_performance:
            self.intent_performance[intent] = {
                "total_requests": 0,
                "successful_requests": 0,
                "average_confidence": 0.0,
                "total_confidence": 0.0
            }

        perf = self.intent_performance[intent]
        perf["total_requests"] += 1

        if success:
            perf["successful_requests"] += 1

        perf["total_confidence"] += actual_confidence
        perf["average_confidence"] = perf["total_confidence"] / perf["total_requests"]

        # Обновляем общую историческую производительность
        self.historical_performance[intent] = perf["average_confidence"]

        logger.debug(f"Updated performance for intent '{intent}': {perf['average_confidence']:.3f}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности"""
        return {
            "intent_performance": self.intent_performance.copy(),
            "overall_performance": self.historical_performance.copy()
        }

    def reset_performance_data(self):
        """Сброс данных о производительности"""
        self.historical_performance.clear()
        self.intent_performance.clear()
        logger.info("🔄 Confidence calculator performance data reset")

    def calculate_with_ml(self, result: Any, analysis: QueryAnalysis) -> float:
        """
        Расчет уверенности с использованием машинного обучения.

        Использует обученную модель для более точного предсказания.
        """
        try:
            # Извлечение признаков
            features = self._extract_features(result, analysis)

            if self.ml_model and len(self.training_data) > 10:
                # Использование обученной модели
                confidence = self._predict_with_model(features)
            else:
                # Fallback на rule-based расчет
                confidence = self.calculate(result, analysis)

            # Сохраняем предсказание для последующего обучения
            self.prediction_history.append((features, confidence, None))  # actual будет установлен позже

            return confidence

        except Exception as e:
            logger.error(f"ML confidence calculation failed: {e}")
            return self.calculate(result, analysis)  # Fallback

    def _extract_features(self, result: Any, analysis: QueryAnalysis) -> List[float]:
        """Извлечение признаков для ML модели"""
        features = []

        # Базовые факторы уверенности
        model_conf = self._get_model_confidence(result)
        tool_success = self._calculate_tool_success_rate(analysis.required_tools or [])
        context_rel = self._assess_context_relevance(result, analysis)
        hist_perf = self._get_historical_performance(analysis.intent)

        features.extend([model_conf, tool_success, context_rel, hist_perf])

        # Дополнительные признаки
        result_str = str(result) if result else ""

        # Длина ответа (нормализованная)
        response_length = min(1.0, len(result_str) / 500.0)
        features.append(response_length)

        # Количество инструментов (нормализованное)
        tool_count = len(analysis.required_tools or []) / 10.0
        features.append(tool_count)

        # Сложность запроса
        complexity = 0.0
        if analysis.complexity == "high":
            complexity = 1.0
        elif analysis.complexity == "medium":
            complexity = 0.5
        features.append(complexity)

        # Временной фактор (время суток может влиять на уверенность)
        current_hour = datetime.now().hour / 24.0
        features.append(current_hour)

        # Признаки качества ответа
        has_numbers = 1.0 if any(char.isdigit() for char in result_str) else 0.0
        has_links = 1.0 if "http" in result_str or "www" in result_str else 0.0
        has_uncertainty = 1.0 if any(phrase in result_str.lower() for phrase in [
            "я не знаю", "не уверен", "возможно", "может быть"
        ]) else 0.0

        features.extend([has_numbers, has_links, has_uncertainty])

        return features

    def _predict_with_model(self, features: List[float]) -> float:
        """Предсказание уверенности с использованием обученной модели"""
        if not self.ml_model:
            return sum(features) / len(features)  # Простое среднее

        # Для простой линейной модели
        prediction = sum(w * f for w, f in zip(self.ml_model, features))
        return max(0.0, min(1.0, prediction))

    def train_ml_model(self, learning_rate: float = 0.01, epochs: int = 100):
        """
        Обучение ML модели на исторических данных.

        Использует простую линейную регрессию с градиентным спуском.
        """
        if len(self.training_data) < 5:
            logger.warning("Недостаточно данных для обучения ML модели")
            return

        # Инициализация весов
        n_features = len(self.training_data[0][0])
        if not self.ml_model:
            self.ml_model = [0.1] * n_features  # Начальные веса

        # Градиентный спуск
        for epoch in range(epochs):
            total_error = 0.0

            for features, actual_confidence in self.training_data:
                predicted = self._predict_with_model(features)
                error = predicted - actual_confidence
                total_error += error ** 2

                # Обновление весов
                for i in range(n_features):
                    self.ml_model[i] -= learning_rate * error * features[i]

            # Нормализация весов
            total_weight = sum(abs(w) for w in self.ml_model)
            if total_weight > 0:
                self.ml_model = [w / total_weight for w in self.ml_model]

            if epoch % 20 == 0:
                mse = total_error / len(self.training_data)
                logger.debug(f"Epoch {epoch}: MSE = {mse:.4f}")

        self.accuracy_stats['last_calibration'] = datetime.now()
        logger.info("✅ ML модель обучена")

    def add_training_example(self, result: Any, analysis: QueryAnalysis, actual_confidence: float):
        """
        Добавление примера для обучения модели.

        Args:
            result: Результат запроса
            analysis: Анализ запроса
            actual_confidence: Фактическая уверенность (оцененная пользователем или системой)
        """
        features = self._extract_features(result, analysis)
        self.training_data.append((features, actual_confidence))

        # Ограничение размера обучающих данных
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-500:]  # Оставляем последние 500 примеров

    def calibrate_with_feedback(self, predicted_confidence: float, actual_confidence: float):
        """
        Калибровка модели на основе обратной связи.

        Args:
            predicted_confidence: Предсказанная уверенность
            actual_confidence: Фактическая уверенность (из обратной связи)
        """
        # Обновление статистики точности
        self.accuracy_stats['total_predictions'] += 1

        error = abs(predicted_confidence - actual_confidence)
        if error < 0.2:  # Порог точности
            self.accuracy_stats['accurate_predictions'] += 1

        # Обновление средней ошибки
        current_mean = self.accuracy_stats['mean_error']
        total = self.accuracy_stats['total_predictions']
        self.accuracy_stats['mean_error'] = (current_mean * (total - 1) + error) / total

        # Автоматическое переобучение при низкой точности
        accuracy_rate = self.accuracy_stats['accurate_predictions'] / total
        if accuracy_rate < 0.7 and total % 50 == 0:  # Каждые 50 предсказаний
            logger.info("🔄 Автоматическая калибровка ML модели")
            self.train_ml_model()

    def get_ml_stats(self) -> Dict[str, Any]:
        """Получение статистики ML модели"""
        return {
            'model_trained': self.ml_model is not None,
            'training_examples': len(self.training_data),
            'prediction_history': len(self.prediction_history),
            'accuracy_stats': self.accuracy_stats.copy(),
            'feature_weights': self.feature_weights.copy(),
            'model_weights': self.ml_model.copy() if self.ml_model else None
        }

    def adaptive_confidence_calculation(self, result: Any, analysis: QueryAnalysis,
                                      user_feedback: Optional[float] = None) -> float:
        """
        Адаптивный расчет уверенности с учетом обратной связи.

        Args:
            result: Результат запроса
            analysis: Анализ запроса
            user_feedback: Обратная связь пользователя (0.0-1.0)

        Returns:
            Адаптированная уверенность
        """
        # Базовый расчет
        base_confidence = self.calculate_with_ml(result, analysis)

        if user_feedback is not None:
            # Калибровка на основе обратной связи
            self.calibrate_with_feedback(base_confidence, user_feedback)

            # Добавление примера для обучения
            self.add_training_example(result, analysis, user_feedback)

            # Возвращаем комбинацию предсказания и обратной связи
            # Чем больше данных, тем больше вес обратной связи
            feedback_weight = min(0.3, len(self.training_data) / 100.0)
            adapted_confidence = base_confidence * (1 - feedback_weight) + user_feedback * feedback_weight

            return adapted_confidence

        return base_confidence

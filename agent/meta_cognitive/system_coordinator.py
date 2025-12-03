"""
Координатор подсистем агента
Фаза 4: Продвинутые функции
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class SubsystemStatus:
    """
    Статус подсистемы
    """

    def __init__(self, name: str):
        self.name = name
        self.is_active = False
        self.last_activity = None
        self.health_score = 1.0
        self.load_level = 0.0
        self.error_count = 0
        self.success_count = 0
        self.average_response_time = 0.0
        self.capabilities: Set[str] = set()

    def update_activity(self):
        """Обновление времени последней активности"""
        self.last_activity = datetime.now()

    def record_success(self, response_time: float = 0.0):
        """Запись успешной операции"""
        self.success_count += 1
        self.is_active = True
        self.update_activity()
        if response_time > 0:
            # Экспоненциальное сглаживание
            self.average_response_time = (
                self.average_response_time * 0.9 + response_time * 0.1
            )

    def record_error(self):
        """Запись ошибки"""
        self.error_count += 1
        self.update_activity()

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса подсистемы"""
        return {
            'name': self.name,
            'is_active': self.is_active,
            'last_activity': self.last_activity,
            'health_score': self.health_score,
            'load_level': self.load_level,
            'error_rate': self.error_count / max(self.success_count + self.error_count, 1),
            'average_response_time': self.average_response_time,
            'capabilities': list(self.capabilities)
        }


class CoordinationStrategy:
    """
    Стратегия координации подсистем
    """

    def __init__(self, name: str, priority: int = 1):
        self.name = name
        self.priority = priority
        self.required_subsystems: List[str] = []
        self.optional_subsystems: List[str] = []
        self.conflict_resolution_rules: Dict[str, Any] = {}
        self.resource_requirements: Dict[str, float] = {}

    def add_required_subsystem(self, subsystem: str):
        """Добавление обязательной подсистемы"""
        if subsystem not in self.required_subsystems:
            self.required_subsystems.append(subsystem)

    def add_optional_subsystem(self, subsystem: str):
        """Добавление опциональной подсистемы"""
        if subsystem not in self.optional_subsystems:
            self.optional_subsystems.append(subsystem)


class SystemCoordinator:
    """
    Координатор подсистем агента
    Управляет взаимодействием между различными компонентами агента
    """

    def __init__(self, agent_core):
        """
        Инициализация координатора

        Args:
            agent_core: Ядро агента
        """
        self.agent_core = agent_core

        # Регистрация подсистем
        self.subsystems: Dict[str, SubsystemStatus] = {}
        self._register_subsystems()

        # Стратегии координации
        self.coordination_strategies: Dict[str, CoordinationStrategy] = {}
        self._initialize_strategies()

        # История координации
        self.coordination_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # Метрики координации
        self.total_coordinations = 0
        self.successful_coordinations = 0
        self.failed_coordinations = 0
        self.average_coordination_time = 0.0

        logger.info("SystemCoordinator initialized")

    def _register_subsystems(self):
        """Регистрация всех подсистем агента"""
        # Ядро агента
        core_status = SubsystemStatus("core")
        core_status.capabilities.update([
            "request_processing", "response_generation", "basic_reasoning"
        ])
        self.subsystems["core"] = core_status

        # Самосознание
        if hasattr(self.agent_core, 'self_reflection_engine'):
            awareness_status = SubsystemStatus("self_awareness")
            awareness_status.capabilities.update([
                "self_reflection", "confidence_calculation", "error_analysis",
                "performance_monitoring", "adaptation_planning"
            ])
            self.subsystems["self_awareness"] = awareness_status

        # Обучение
        if hasattr(self.agent_core, 'learning_engine'):
            learning_status = SubsystemStatus("learning")
            learning_status.capabilities.update([
                "experience_processing", "pattern_extraction", "skill_development",
                "cognitive_updates", "adaptive_learning"
            ])
            self.subsystems["learning"] = learning_status

        # Память
        if hasattr(self.agent_core, 'memory_manager'):
            memory_status = SubsystemStatus("memory")
            memory_status.capabilities.update([
                "experience_storage", "pattern_retrieval", "context_management",
                "memory_consolidation", "knowledge_synthesis"
            ])
            self.subsystems["memory"] = memory_status

        # Инструменты
        if hasattr(self.agent_core, 'tool_orchestrator'):
            tools_status = SubsystemStatus("tools")
            tools_status.capabilities.update([
                "tool_execution", "query_analysis", "resource_access",
                "external_integration", "data_processing"
            ])
            self.subsystems["tools"] = tools_status

        # Мета-познание (продвинутые функции)
        meta_status = SubsystemStatus("meta_cognitive")
        meta_status.capabilities.update([
            "meta_decision_making", "system_coordination", "cognitive_health_monitoring",
            "performance_optimization", "distributed_learning", "multimodal_processing",
            "ethical_evaluation"
        ])
        self.subsystems["meta_cognitive"] = meta_status

    def _initialize_strategies(self):
        """Инициализация стратегий координации"""
        # Стратегия базовой обработки
        basic_strategy = CoordinationStrategy("basic_processing", priority=1)
        basic_strategy.add_required_subsystem("core")
        self.coordination_strategies["basic"] = basic_strategy

        # Стратегия самоосознанной обработки
        aware_strategy = CoordinationStrategy("self_aware_processing", priority=2)
        aware_strategy.add_required_subsystem("core")
        aware_strategy.add_required_subsystem("self_awareness")
        self.coordination_strategies["self_aware"] = aware_strategy

        # Стратегия обучения
        learning_strategy = CoordinationStrategy("learning_processing", priority=3)
        learning_strategy.add_required_subsystem("core")
        learning_strategy.add_required_subsystem("learning")
        learning_strategy.add_optional_subsystem("memory")
        learning_strategy.add_optional_subsystem("self_awareness")
        self.coordination_strategies["learning"] = learning_strategy

        # Стратегия комплексной обработки
        complex_strategy = CoordinationStrategy("complex_processing", priority=4)
        complex_strategy.add_required_subsystem("core")
        complex_strategy.add_required_subsystem("self_awareness")
        complex_strategy.add_required_subsystem("learning")
        complex_strategy.add_required_subsystem("memory")
        complex_strategy.add_optional_subsystem("tools")
        complex_strategy.add_required_subsystem("meta_cognitive")
        self.coordination_strategies["complex"] = complex_strategy

        # Стратегия распределенного обучения
        distributed_strategy = CoordinationStrategy("distributed_processing", priority=5)
        distributed_strategy.add_required_subsystem("core")
        distributed_strategy.add_required_subsystem("learning")
        distributed_strategy.add_required_subsystem("meta_cognitive")
        distributed_strategy.add_optional_subsystem("memory")
        self.coordination_strategies["distributed"] = distributed_strategy

        # Стратегия мульти-модального обучения
        multimodal_strategy = CoordinationStrategy("multimodal_processing", priority=5)
        multimodal_strategy.add_required_subsystem("core")
        multimodal_strategy.add_required_subsystem("learning")
        multimodal_strategy.add_required_subsystem("meta_cognitive")
        multimodal_strategy.add_optional_subsystem("tools")
        self.coordination_strategies["multimodal"] = multimodal_strategy

    async def coordinate_processing(self, request: Any, meta_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Координация обработки запроса

        Args:
            request: Запрос агента
            meta_decision: Мета-решение

        Returns:
            Dict[str, Any]: Результат координации
        """
        import time
        coord_start = time.time()

        try:
            logger.info(f"Starting coordination for request: {getattr(request, 'id', 'unknown')}")

            # Выбор стратегии координации
            strategy = self._select_coordination_strategy(request, meta_decision)

            # Проверка доступности подсистем
            available_subsystems = await self._check_subsystem_availability(strategy)

            # Распределение нагрузки
            load_distribution = self._distribute_load(strategy, available_subsystems)

            # Выполнение координированной обработки
            coordination_result = await self._execute_coordinated_processing(
                request, strategy, load_distribution, meta_decision
            )

            coordination_time = time.time() - coord_start

            # Обновление метрик
            self.total_coordinations += 1
            if coordination_result.get('success', False):
                self.successful_coordinations += 1
            else:
                self.failed_coordinations += 1

            # Экспоненциальное сглаживание времени координации
            self.average_coordination_time = (
                self.average_coordination_time * 0.9 + coordination_time * 0.1
            )

            # Обновление статусов подсистем
            await self._update_subsystem_statuses(
                coordination_result.get('involved_subsystems', []),
                coordination_result.get('success', False),
                coordination_time
            )

            result = {
                'strategy_used': strategy.name,
                'involved_subsystems': coordination_result.get('involved_subsystems', []),
                'load_distribution': load_distribution,
                'success': coordination_result.get('success', False),
                'coordination_time': coordination_time,
                'resource_usage': coordination_result.get('resource_usage', {}),
                'conflicts_resolved': coordination_result.get('conflicts_resolved', 0),
                'optimizations_applied': coordination_result.get('optimizations_applied', [])
            }

            # Сохранение в историю
            self.coordination_history.append({
                'timestamp': datetime.now(),
                'request_id': getattr(request, 'id', 'unknown'),
                'result': result
            })

            if len(self.coordination_history) > self.max_history_size:
                self.coordination_history = self.coordination_history[-self.max_history_size:]

            logger.info(f"Coordination completed in {coordination_time:.2f}s using strategy: {strategy.name}")
            return result

        except Exception as e:
            logger.error(f"Coordination failed: {e}")
            coordination_time = time.time() - coord_start

            return {
                'strategy_used': 'error_fallback',
                'involved_subsystems': ['core'],
                'load_distribution': {},
                'success': False,
                'coordination_time': coordination_time,
                'error': str(e),
                'resource_usage': {},
                'conflicts_resolved': 0,
                'optimizations_applied': []
            }

    def _select_coordination_strategy(self, request: Any, meta_decision: Dict[str, Any]) -> CoordinationStrategy:
        """
        Выбор стратегии координации

        Args:
            request: Запрос
            meta_decision: Мета-решение

        Returns:
            CoordinationStrategy: Выбранная стратегия
        """
        # Приоритет мета-решения
        if meta_decision.get('use_distributed', False):
            return self.coordination_strategies.get('distributed', self.coordination_strategies['basic'])

        if meta_decision.get('use_multimodal', False):
            return self.coordination_strategies.get('multimodal', self.coordination_strategies['basic'])

        # Анализ сложности запроса
        complexity = self._assess_request_complexity(request)

        if complexity >= 0.8:
            return self.coordination_strategies.get('complex', self.coordination_strategies['basic'])
        elif complexity >= 0.6:
            return self.coordination_strategies.get('learning', self.coordination_strategies['basic'])
        elif complexity >= 0.4:
            return self.coordination_strategies.get('self_aware', self.coordination_strategies['basic'])
        else:
            return self.coordination_strategies['basic']

    def _assess_request_complexity(self, request: Any) -> float:
        """
        Оценка сложности запроса

        Args:
            request: Запрос

        Returns:
            float: Уровень сложности (0.0-1.0)
        """
        complexity = 0.0

        # Анализ текста запроса
        query = getattr(request, 'query', '')
        if query:
            # Длина запроса
            length_factor = min(len(query) / 500, 0.3)
            complexity += length_factor

            # Наличие технических терминов
            technical_terms = ['алгоритм', 'система', 'модель', 'данные', 'анализ', 'оптимизация']
            technical_factor = sum(1 for term in technical_terms if term in query.lower()) * 0.1
            complexity += min(technical_factor, 0.3)

            # Наличие вопросов
            question_factor = query.count('?') * 0.1
            complexity += min(question_factor, 0.2)

        # Анализ метаданных
        metadata = getattr(request, 'metadata', {})
        if metadata:
            complexity += 0.1  # Бонус за наличие метаданных

        return min(complexity, 1.0)

    async def _check_subsystem_availability(self, strategy: CoordinationStrategy) -> Dict[str, bool]:
        """
        Проверка доступности подсистем

        Args:
            strategy: Стратегия координации

        Returns:
            Dict[str, bool]: Доступность подсистем
        """
        availability = {}

        # Проверка обязательных подсистем
        for subsystem_name in strategy.required_subsystems:
            if subsystem_name in self.subsystems:
                status = self.subsystems[subsystem_name]
                # Подсистема доступна если активна и имеет хорошее здоровье
                availability[subsystem_name] = (
                    status.is_active and status.health_score > 0.5
                )
            else:
                availability[subsystem_name] = False

        # Проверка опциональных подсистем
        for subsystem_name in strategy.optional_subsystems:
            if subsystem_name in self.subsystems:
                status = self.subsystems[subsystem_name]
                availability[subsystem_name] = (
                    status.is_active and status.health_score > 0.3
                )
            else:
                availability[subsystem_name] = False

        return availability

    def _distribute_load(self, strategy: CoordinationStrategy, availability: Dict[str, bool]) -> Dict[str, float]:
        """
        Распределение нагрузки между подсистемами

        Args:
            strategy: Стратегия координации
            availability: Доступность подсистем

        Returns:
            Dict[str, float]: Распределение нагрузки
        """
        distribution = {}

        # Собираем доступные подсистемы
        available_required = [
            name for name in strategy.required_subsystems
            if availability.get(name, False)
        ]
        available_optional = [
            name for name in strategy.optional_subsystems
            if availability.get(name, False)
        ]

        all_available = available_required + available_optional

        if not all_available:
            return distribution

        # Равномерное распределение нагрузки
        base_load = 1.0 / len(all_available)

        for subsystem_name in all_available:
            distribution[subsystem_name] = base_load

        # Корректировка на основе здоровья подсистем
        for subsystem_name, load in distribution.items():
            if subsystem_name in self.subsystems:
                health_factor = self.subsystems[subsystem_name].health_score
                distribution[subsystem_name] = load * health_factor

        # Нормализация
        total_load = sum(distribution.values())
        if total_load > 0:
            for subsystem_name in distribution:
                distribution[subsystem_name] /= total_load

        return distribution

    async def _execute_coordinated_processing(self, request: Any, strategy: CoordinationStrategy,
                                           load_distribution: Dict[str, float],
                                           meta_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение координированной обработки

        Args:
            request: Запрос
            strategy: Стратегия
            load_distribution: Распределение нагрузки
            meta_decision: Мета-решение

        Returns:
            Dict[str, Any]: Результат обработки
        """
        involved_subsystems = list(load_distribution.keys())
        resource_usage = {}
        conflicts_resolved = 0
        optimizations_applied = []

        try:
            # Параллельная активация подсистем
            activation_tasks = []
            for subsystem_name in involved_subsystems:
                task = self._activate_subsystem(subsystem_name, load_distribution[subsystem_name])
                activation_tasks.append(task)

            activation_results = await asyncio.gather(*activation_tasks, return_exceptions=True)

            # Обработка результатов активации
            for i, result in enumerate(activation_results):
                subsystem_name = involved_subsystems[i]
                if isinstance(result, Exception):
                    logger.warning(f"Failed to activate subsystem {subsystem_name}: {result}")
                    involved_subsystems.remove(subsystem_name)
                else:
                    resource_usage[subsystem_name] = result

            if not involved_subsystems:
                return {
                    'success': False,
                    'involved_subsystems': [],
                    'resource_usage': {},
                    'conflicts_resolved': 0,
                    'optimizations_applied': []
                }

            # Разрешение конфликтов ресурсов
            conflicts_resolved = await self._resolve_resource_conflicts(
                involved_subsystems, load_distribution
            )

            # Применение оптимизаций
            optimizations_applied = await self._apply_coordination_optimizations(
                involved_subsystems, meta_decision
            )

            # Основная обработка через ядро
            success = await self._perform_core_processing(request, involved_subsystems)

            return {
                'success': success,
                'involved_subsystems': involved_subsystems,
                'resource_usage': resource_usage,
                'conflicts_resolved': conflicts_resolved,
                'optimizations_applied': optimizations_applied
            }

        except Exception as e:
            logger.error(f"Coordinated processing failed: {e}")
            return {
                'success': False,
                'involved_subsystems': involved_subsystems,
                'resource_usage': resource_usage,
                'conflicts_resolved': conflicts_resolved,
                'optimizations_applied': optimizations_applied,
                'error': str(e)
            }

    async def _activate_subsystem(self, subsystem_name: str, load_level: float) -> Dict[str, Any]:
        """
        Активация подсистемы

        Args:
            subsystem_name: Имя подсистемы
            load_level: Уровень нагрузки

        Returns:
            Dict[str, Any]: Использование ресурсов
        """
        if subsystem_name not in self.subsystems:
            raise ValueError(f"Unknown subsystem: {subsystem_name}")

        status = self.subsystems[subsystem_name]
        status.load_level = load_level
        status.update_activity()

        # Имитация использования ресурсов
        resource_usage = {
            'cpu': load_level * 0.1,
            'memory': load_level * 0.2,
            'io': load_level * 0.05
        }

        # Имитация времени активации
        await asyncio.sleep(0.01 * load_level)

        return resource_usage

    async def _resolve_resource_conflicts(self, subsystems: List[str],
                                        load_distribution: Dict[str, float]) -> int:
        """
        Разрешение конфликтов ресурсов

        Args:
            subsystems: Подсистемы
            load_distribution: Распределение нагрузки

        Returns:
            int: Количество разрешенных конфликтов
        """
        conflicts_resolved = 0

        # Простая логика разрешения конфликтов
        # В реальной реализации здесь был бы более сложный алгоритм
        total_load = sum(load_distribution.values())

        if total_load > 1.0:
            # Перегрузка - снижаем нагрузку
            overload_factor = total_load - 1.0
            for subsystem in subsystems:
                if subsystem in load_distribution:
                    reduction = overload_factor * load_distribution[subsystem]
                    load_distribution[subsystem] -= reduction
                    conflicts_resolved += 1

        return conflicts_resolved

    async def _apply_coordination_optimizations(self, subsystems: List[str],
                                             meta_decision: Dict[str, Any]) -> List[str]:
        """
        Применение оптимизаций координации

        Args:
            subsystems: Подсистемы
            meta_decision: Мета-решение

        Returns:
            List[str]: Примененные оптимизации
        """
        optimizations = []

        # Оптимизация кэширования
        if len(subsystems) > 2:
            optimizations.append("caching_optimization")

        # Оптимизация параллелизма
        if 'learning' in subsystems and 'memory' in subsystems:
            optimizations.append("parallel_processing")

        # Оптимизация ресурсов
        if meta_decision.get('optimize_resources', False):
            optimizations.append("resource_optimization")

        return optimizations

    async def _perform_core_processing(self, request: Any, subsystems: List[str]) -> bool:
        """
        Выполнение основной обработки через ядро

        Args:
            request: Запрос
            subsystems: Задействованные подсистемы

        Returns:
            bool: Успех обработки
        """
        try:
            # Базовая обработка через ядро
            response = await self.agent_core.process_request(request)

            # Имитация дополнительной обработки подсистемами
            processing_tasks = []
            for subsystem in subsystems:
                if subsystem != 'core':  # Ядро уже обработало
                    task = self._process_with_subsystem(subsystem, request, response)
                    processing_tasks.append(task)

            if processing_tasks:
                await asyncio.gather(*processing_tasks, return_exceptions=True)

            return True

        except Exception as e:
            logger.error(f"Core processing failed: {e}")
            return False

    async def _process_with_subsystem(self, subsystem_name: str, request: Any, response: Any):
        """
        Обработка с использованием конкретной подсистемы

        Args:
            subsystem_name: Имя подсистемы
            request: Запрос
            response: Ответ
        """
        # Имитация обработки подсистемой
        await asyncio.sleep(0.005)

        # Обновление статуса подсистемы
        if subsystem_name in self.subsystems:
            self.subsystems[subsystem_name].record_success(0.005)

    async def _update_subsystem_statuses(self, subsystems: List[str], success: bool, total_time: float):
        """
        Обновление статусов подсистем

        Args:
            subsystems: Подсистемы
            success: Успех операции
            total_time: Общее время
        """
        response_time_per_subsystem = total_time / max(len(subsystems), 1)

        for subsystem_name in subsystems:
            if subsystem_name in self.subsystems:
                status = self.subsystems[subsystem_name]
                if success:
                    status.record_success(response_time_per_subsystem)
                else:
                    status.record_error()

    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса координатора

        Returns:
            Dict[str, Any]: Статус
        """
        return {
            'total_coordinations': self.total_coordinations,
            'successful_coordinations': self.successful_coordinations,
            'failed_coordinations': self.failed_coordinations,
            'success_rate': self.successful_coordinations / max(self.total_coordinations, 1),
            'average_coordination_time': self.average_coordination_time,
            'subsystems': {
                name: status.get_status() for name, status in self.subsystems.items()
            },
            'active_strategies': list(self.coordination_strategies.keys()),
            'coordination_history_size': len(self.coordination_history)
        }

    async def optimize_coordination(self) -> Dict[str, Any]:
        """
        Оптимизация координации

        Returns:
            Dict[str, Any]: Результат оптимизации
        """
        optimizations = []

        # Оптимизация стратегий
        strategy_opt = self._optimize_strategies()
        optimizations.append({'type': 'strategies', 'result': strategy_opt})

        # Оптимизация подсистем
        subsystem_opt = await self._optimize_subsystems()
        optimizations.append({'type': 'subsystems', 'result': subsystem_opt})

        # Оптимизация ресурсов
        resource_opt = self._optimize_resource_usage()
        optimizations.append({'type': 'resources', 'result': resource_opt})

        improvement = sum(opt.get('improvement', 0) for opt in [strategy_opt, subsystem_opt, resource_opt])

        return {
            'optimizations': optimizations,
            'overall_improvement': improvement,
            'timestamp': datetime.now()
        }

    def _optimize_strategies(self) -> Dict[str, Any]:
        """
        Оптимизация стратегий координации

        Returns:
            Dict[str, Any]: Результат оптимизации
        """
        # Простая оптимизация - повышение приоритетов часто используемых стратегий
        improvements = 0

        for strategy_name, strategy in self.coordination_strategies.items():
            # Имитация оптимизации
            if strategy.priority < 5:
                strategy.priority += 1
                improvements += 0.1

        return {'improvements_made': improvements, 'improvement': improvements}

    async def _optimize_subsystems(self) -> Dict[str, Any]:
        """
        Оптимизация подсистем

        Returns:
            Dict[str, Any]: Результат оптимизации
        """
        improvements = 0

        for subsystem_name, status in self.subsystems.items():
            # Восстановление здоровья проблемных подсистем
            if status.health_score < 0.8:
                status.health_score = min(status.health_score + 0.1, 1.0)
                improvements += 0.05

            # Сброс счетчиков ошибок
            if status.error_count > 10:
                status.error_count = 0
                improvements += 0.02

        return {'improvements_made': improvements, 'improvement': improvements}

    def _optimize_resource_usage(self) -> Dict[str, Any]:
        """
        Оптимизация использования ресурсов

        Returns:
            Dict[str, Any]: Результат оптимизации
        """
        # Имитация оптимизации ресурсов
        improvement = 0.05  # 5% улучшение

        return {'improvements_made': 1, 'improvement': improvement}

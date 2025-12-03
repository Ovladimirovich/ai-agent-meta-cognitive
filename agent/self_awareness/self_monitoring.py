import time
import psutil
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta

from ..core.models import (
    SelfDiagnosisResult, AgentHealth
)

if TYPE_CHECKING:
    from ..core.agent_core import AgentCore

logger = logging.getLogger(__name__)


class ComponentHealthChecker:
    """Проверка здоровья компонентов агента"""

    def __init__(self, agent: 'AgentCore'):
        self.agent = agent
        self.component_checks = {
            'agent_core': self._check_agent_core,
            'memory_manager': self._check_memory_manager,
            'tool_orchestrator': self._check_tool_orchestrator,
            'query_analyzer': self._check_query_analyzer,
            'reflection_engine': self._check_reflection_engine,
            'adaptation_engine': self._check_adaptation_engine,
            'confidence_calculator': self._check_confidence_calculator,
            'cognitive_load_analyzer': self._check_cognitive_load_analyzer
        }

    async def check_all_components(self) -> Dict[str, Any]:
        """Проверка всех компонентов"""
        results = {}

        for component_name, check_func in self.component_checks.items():
            try:
                health_status = await check_func()
                results[component_name] = health_status
            except Exception as e:
                logger.error(f"Error checking component {component_name}: {e}")
                results[component_name] = {
                    'status': 'error',
                    'message': f"Ошибка проверки: {str(e)}",
                    'severity': 'high'
                }

        return results

    async def _check_agent_core(self) -> Dict[str, Any]:
        """Проверка ядра агента"""
        try:
            # Проверка основных атрибутов
            required_attrs = ['config', 'state_manager', 'tool_orchestrator']
            missing_attrs = [attr for attr in required_attrs if not hasattr(self.agent, attr) or not getattr(self.agent, attr)]

            if missing_attrs:
                return {
                    'status': 'warning',
                    'message': f"Missing required attributes: {missing_attrs}",
                    'severity': 'medium'
                }

            # Проверка состояния
            if hasattr(self.agent, 'state_manager') and self.agent.state_manager:
                current_state = self.agent.state_manager.current_state.value
                if current_state not in ['idle', 'analyzing', 'executing', 'completed']:
                    return {
                        'status': 'warning',
                        'message': f'Unexpected agent state: {current_state}',
                        'severity': 'low'
                    }

            return {
                'status': 'healthy',
                'message': 'Agent core is operational',
                'severity': 'low'
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e), 'severity': 'high'}

    async def _check_memory_manager(self) -> Dict[str, Any]:
        """Проверка менеджера памяти"""
        try:
            if not hasattr(self.agent, 'memory_manager') or not self.agent.memory_manager:
                return {'status': 'error', 'message': 'Memory manager not available', 'severity': 'high'}

            memory_mgr = self.agent.memory_manager

            # Проверка основных функций
            if hasattr(memory_mgr, 'get_memory_stats'):
                stats = memory_mgr.get_memory_stats()  # Синхронный метод
                memory_usage = stats.get('memory_utilization_percent', 0)

                if memory_usage > 90:
                    return {
                        'status': 'warning',
                        'message': f'High memory usage: {memory_usage}%',
                        'severity': 'high'
                    }
                elif memory_usage > 75:
                    return {
                        'status': 'warning',
                        'message': f'Elevated memory usage: {memory_usage}%',
                        'severity': 'medium'
                    }

            return {
                'status': 'healthy',
                'message': 'Memory manager is operational',
                'severity': 'low'
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e), 'severity': 'high'}

    async def _check_tool_orchestrator(self) -> Dict[str, Any]:
        """Проверка оркестратора инструментов"""
        try:
            if not hasattr(self.agent, 'tool_orchestrator') or not self.agent.tool_orchestrator:
                return {'status': 'error', 'message': 'Tool orchestrator not available', 'severity': 'high'}

            orchestrator = self.agent.tool_orchestrator

            # Проверка доступных инструментов
            if hasattr(orchestrator, 'get_available_tools'):
                tools = await orchestrator.get_available_tools()
                if not tools:
                    return {
                        'status': 'warning',
                        'message': 'No tools available',
                        'severity': 'medium'
                    }

            return {
                'status': 'healthy',
                'message': 'Tool orchestrator is operational',
                'severity': 'low'
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e), 'severity': 'high'}

    async def _check_query_analyzer(self) -> Dict[str, Any]:
        """Проверка анализатора запросов"""
        try:
            if not hasattr(self.agent, 'query_analyzer') or not self.agent.query_analyzer:
                return {'status': 'warning', 'message': 'Query analyzer not available', 'severity': 'medium'}

            analyzer = self.agent.query_analyzer

            # Проверка базовой функциональности
            if hasattr(analyzer, 'analyze_query'):
                # Простой тест
                test_result = await analyzer.analyze_query("test query")
                if not test_result:
                    return {
                        'status': 'warning',
                        'message': 'Query analyzer returned empty result',
                        'severity': 'low'
                    }

            return {
                'status': 'healthy',
                'message': 'Query analyzer is operational',
                'severity': 'low'
            }

        except Exception as e:
            return {'status': 'warning', 'message': str(e), 'severity': 'low'}

    async def _check_reflection_engine(self) -> Dict[str, Any]:
        """Проверка движка рефлексии"""
        try:
            if not hasattr(self.agent, 'reflection_engine') or not self.agent.reflection_engine:
                return {'status': 'warning', 'message': 'Reflection engine not available', 'severity': 'low'}

            reflection_engine = self.agent.reflection_engine

            # Проверка истории рефлексии
            if hasattr(reflection_engine, 'reflection_history'):
                history_length = len(reflection_engine.reflection_history)
                if history_length == 0:
                    return {
                        'status': 'info',
                        'message': 'No reflection history yet',
                        'severity': 'low'
                    }

            return {
                'status': 'healthy',
                'message': 'Reflection engine is operational',
                'severity': 'low'
            }

        except Exception as e:
            return {'status': 'warning', 'message': str(e), 'severity': 'low'}

    async def _check_adaptation_engine(self) -> Dict[str, Any]:
        """Проверка движка адаптации"""
        try:
            if not hasattr(self.agent, 'adaptation_engine') or not self.agent.adaptation_engine:
                return {'status': 'warning', 'message': 'Adaptation engine not available', 'severity': 'low'}

            adaptation_engine = self.agent.adaptation_engine

            # Проверка активных адаптаций
            if hasattr(adaptation_engine, 'active_adaptations'):
                active_count = len(adaptation_engine.active_adaptations)
                if active_count > 10:  # Слишком много активных адаптаций
                    return {
                        'status': 'warning',
                        'message': f'Too many active adaptations: {active_count}',
                        'severity': 'medium'
                    }

            return {
                'status': 'healthy',
                'message': 'Adaptation engine is operational',
                'severity': 'low'
            }

        except Exception as e:
            return {'status': 'warning', 'message': str(e), 'severity': 'low'}

    async def _check_confidence_calculator(self) -> Dict[str, Any]:
        """Проверка калькулятора уверенности"""
        try:
            if not hasattr(self.agent, 'confidence_calculator') or not self.agent.confidence_calculator:
                return {'status': 'warning', 'message': 'Confidence calculator not available', 'severity': 'low'}

            confidence_calculator = self.agent.confidence_calculator

            # Проверка базовой функциональности
            if hasattr(confidence_calculator, 'calculate'):
                # Простой тест вычисления уверенности
                dummy_result = "test result"
                from ..core.models import QueryAnalysis  # Импортируем здесь, чтобы избежать циклических импортов
                dummy_analysis = QueryAnalysis(
                    intent="test",
                    entities=[],
                    sentiment="neutral",
                    complexity="low",
                    required_tools=[],
                    context={}
                )
                
                try:
                    confidence_score = confidence_calculator.calculate(dummy_result, dummy_analysis)
                    if not isinstance(confidence_score, (int, float)) or not 0 <= confidence_score <= 1:
                        return {
                            'status': 'warning',
                            'message': 'Confidence calculator returned invalid score',
                            'severity': 'medium'
                        }
                except Exception as e:
                    return {
                        'status': 'warning',
                        'message': f'Confidence calculator test failed: {str(e)}',
                        'severity': 'high'
                    }

            return {
                'status': 'healthy',
                'message': 'Confidence calculator is operational',
                'severity': 'low'
            }

        except Exception as e:
            return {'status': 'warning', 'message': str(e), 'severity': 'low'}

    async def _check_cognitive_load_analyzer(self) -> Dict[str, Any]:
        """Проверка анализатора когнитивной нагрузки"""
        try:
            if not hasattr(self.agent, 'cognitive_load_analyzer') or not self.agent.cognitive_load_analyzer:
                return {'status': 'warning', 'message': 'Cognitive load analyzer not available', 'severity': 'medium'}

            analyzer = self.agent.cognitive_load_analyzer

            # Проверка базовой функциональности
            if hasattr(analyzer, 'assess_load'):
                # Простой тест оценки нагрузки
                from ..meta_cognitive.cognitive_load_analyzer import CognitiveLoadMetrics
                dummy_metrics = CognitiveLoadMetrics(
                    response_time=0.5,
                    processing_time=0.3,
                    memory_usage=0.4,
                    cpu_usage=0.3,
                    active_tasks=5,
                    confidence_level=0.8,
                    error_rate=0.05,
                    complexity_score=0.4,
                    resource_pressure=0.2,
                    timestamp=datetime.now()
                )
                
                try:
                    load_analysis = await analyzer.assess_cognitive_load(dummy_metrics)
                    if not hasattr(load_analysis, 'load_score') or not 0 <= load_analysis.load_score <= 1:
                        return {
                            'status': 'warning',
                            'message': 'Cognitive load analyzer returned invalid analysis',
                            'severity': 'medium'
                        }
                except Exception as e:
                    return {
                        'status': 'warning',
                        'message': f'Cognitive load analyzer test failed: {str(e)}',
                        'severity': 'high'
                    }

            return {
                'status': 'healthy',
                'message': 'Cognitive load analyzer is operational',
                'severity': 'low'
            }

        except Exception as e:
            return {'status': 'warning', 'message': str(e), 'severity': 'low'}


class PerformanceMonitor:
    """Монитор производительности системы"""

    def __init__(self):
        self.baseline_metrics = {}
        self.performance_history = []

    async def monitor_system_performance(self) -> Dict[str, Any]:
        """Мониторинг производительности системы"""
        try:
            # Получение метрик производительности с оптимизацией
            cpu_percent = psutil.cpu_percent(interval=0.1)  # Более короткий интервал для улучшения производительности
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            network = psutil.net_io_counters()
            bytes_sent = network.bytes_sent
            bytes_recv = network.bytes_recv

            current_metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'network_sent': bytes_sent,
                'network_recv': bytes_recv,
                'timestamp': datetime.now()
            }

            # Добавление метрик в историю
            self.performance_history.append(current_metrics)

            # Ограничение истории
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

            # Анализ трендов
            trends = self._analyze_performance_trends()

            # Определение статуса
            status = self._determine_performance_status(current_metrics)

            return {
                'current_metrics': current_metrics,
                'trends': trends,
                'status': status,
                'recommendations': self._generate_performance_recommendations(current_metrics, trends)
            }

        except Exception as e:
            logger.error(f"Error monitoring system performance: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Анализ трендов производительности"""
        if len(self.performance_history) < 5:
            return {'trend': 'insufficient_data'}

        recent = self.performance_history[-5:]
        older = self.performance_history[-10:-5] if len(self.performance_history) >= 10 else self.performance_history[:5]

        def avg_metric(history, metric):
            return sum(h[metric] for h in history) / len(history)

        trends = {}
        for metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
            recent_avg = avg_metric(recent, metric)
            older_avg = avg_metric(older, metric)

            if recent_avg > older_avg * 1.2:
                trends[metric] = 'increasing'
            elif recent_avg < older_avg * 0.8:
                trends[metric] = 'decreasing'
            else:
                trends[metric] = 'stable'

        return trends

    def _determine_performance_status(self, metrics: Dict[str, Any]) -> str:
        """Определение статуса производительности с оптимизацией"""
        # Используем более эффективную проверку условий
        critical_conditions = [
            metrics['cpu_percent'] > 95,
            metrics['memory_percent'] > 95,
            metrics['disk_percent'] > 98
        ]
        
        warning_conditions = [
            metrics['cpu_percent'] > 80,
            metrics['memory_percent'] > 85,
            metrics['disk_percent'] > 90
        ]

        critical_count = sum(critical_conditions)
        warning_count = sum(warning_conditions)

        if critical_count >= 1:
            return 'critical'
        elif warning_count >= 2:
            return 'warning'
        else:
            return 'healthy'

    def _generate_performance_recommendations(self, metrics: Dict[str, Any],
                                            trends: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по производительности с оптимизацией"""
        recommendations = []

        # Используем словарь для более эффективного определения рекомендаций
        threshold_recommendations = {
            'cpu_percent': (80, "Высокая загрузка CPU - рассмотреть оптимизацию процессов"),
            'memory_percent': (85, "Высокое использование памяти - проверить на утечки памяти"),
            'disk_percent': (90, "Мало свободного места на диске - очистить ненужные файлы")
        }

        for metric, (threshold, recommendation) in threshold_recommendations.items():
            if metrics[metric] > threshold:
                recommendations.append(recommendation)

        if trends.get('memory_percent') == 'increasing':
            recommendations.append("Увеличение использования памяти со временем - мониторить на утечки")

        return recommendations


class CognitiveHealthAssessor:
    """Оценщик когнитивного здоровья агента"""

    def __init__(self, agent: 'AgentCore'):
        self.agent = agent

    async def assess_cognitive_health(self) -> Dict[str, Any]:
        """Оценка когнитивного здоровья агента"""
        try:
            health_indicators = {}

            # Оценка способности к обучению
            learning_capacity = await self._assess_learning_capacity()
            health_indicators['learning_capacity'] = learning_capacity

            # Оценка качества ответов
            response_quality = await self._assess_response_quality()
            health_indicators['response_quality'] = response_quality

            # Оценка адаптивности
            adaptability = await self._assess_adaptability()
            health_indicators['adaptability'] = adaptability

            # Оценка эффективности рефлексии
            reflection_effectiveness = await self._assess_reflection_effectiveness()
            health_indicators['reflection_effectiveness'] = reflection_effectiveness

            # Общий score когнитивного здоровья
            overall_score = sum(indicator['score'] for indicator in health_indicators.values()) / len(health_indicators)

            # Определение статуса
            if overall_score > 0.8:
                status = 'excellent'
            elif overall_score > 0.6:
                status = 'good'
            elif overall_score > 0.4:
                status = 'fair'
            else:
                status = 'poor'

            return {
                'overall_score': overall_score,
                'status': status,
                'indicators': health_indicators,
                'recommendations': self._generate_cognitive_recommendations(health_indicators)
            }

        except Exception as e:
            logger.error(f"Error assessing cognitive health: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }

    async def _assess_learning_capacity(self) -> Dict[str, Any]:
        """Оценка способности к обучению"""
        try:
            # Проверка количества insights и адаптаций
            if hasattr(self.agent, 'reflection_engine') and self.agent.reflection_engine:
                insights_count = len(self.agent.reflection_engine.insights)
                score = min(insights_count / 50.0, 1.0)  # Нормализация к 50 insights

                return {
                    'score': score,
                    'insights_count': insights_count,
                    'assessment': 'good' if score > 0.5 else 'needs_improvement'
                }
            else:
                return {
                    'score': 0.0,
                    'insights_count': 0,
                    'assessment': 'no_data'
                }

        except Exception as e:
            return {'score': 0.0, 'error': str(e), 'assessment': 'error'}

    async def _assess_response_quality(self) -> Dict[str, Any]:
        """Оценка качества ответов"""
        try:
            # Анализ недавних взаимодействий из памяти агента
            if hasattr(self.agent, 'memory_manager') and self.agent.memory_manager:
                # Используем синхронный метод вместо async
                episodic_memory = list(self.agent.memory_manager.episodic_memory)[-10:]  # Последние 10 записей

                if episodic_memory:
                    confidences = [entry.confidence for entry in episodic_memory if entry.confidence > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)

                        return {
                            'score': avg_confidence,
                            'avg_confidence': avg_confidence,
                            'interactions_analyzed': len(episodic_memory),
                            'assessment': 'good' if avg_confidence > 0.7 else 'needs_improvement'
                        }

            return {
                'score': 0.5,
                'assessment': 'no_recent_data'
            }

        except Exception as e:
            return {'score': 0.5, 'error': str(e), 'assessment': 'error'}

    async def _assess_adaptability(self) -> Dict[str, Any]:
        """Оценка адаптивности"""
        try:
            if hasattr(self.agent, 'adaptation_engine') and self.agent.adaptation_engine:
                active_adaptations = len(self.agent.adaptation_engine.active_adaptations)
                score = min(active_adaptations / 5.0, 1.0)  # Нормализация к 5 адаптациям

                return {
                    'score': score,
                    'active_adaptations': active_adaptations,
                    'assessment': 'good' if score > 0.3 else 'needs_improvement'
                }
            else:
                return {
                    'score': 0.0,
                    'active_adaptations': 0,
                    'assessment': 'no_adaptation_engine'
                }

        except Exception as e:
            return {'score': 0.0, 'error': str(e), 'assessment': 'error'}

    async def _assess_reflection_effectiveness(self) -> Dict[str, Any]:
        """Оценка эффективности рефлексии"""
        try:
            if hasattr(self.agent, 'reflection_engine') and self.agent.reflection_engine:
                effectiveness = await self.agent.reflection_engine.analyze_reflection_effectiveness()
                score = effectiveness.get('effectiveness', 0)

                return {
                    'score': min(score / 10.0, 1.0),  # Нормализация
                    'insights_per_reflection': effectiveness.get('avg_insight_confidence', 0),
                    'assessment': 'good' if score > 2.0 else 'needs_improvement'
                }
            else:
                return {
                    'score': 0.0,
                    'assessment': 'no_reflection_engine'
                }

        except Exception as e:
            return {'score': 0.0, 'error': str(e), 'assessment': 'error'}

    def _generate_cognitive_recommendations(self, indicators: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по когнитивному здоровью"""
        recommendations = []

        for indicator_name, indicator_data in indicators.items():
            score = indicator_data.get('score', 0)

            if score < 0.4:
                if indicator_name == 'learning_capacity':
                    recommendations.append("Улучшить способность к обучению - увеличить частоту рефлексии")
                elif indicator_name == 'response_quality':
                    recommendations.append("Повысить качество ответов - улучшить анализ запросов")
                elif indicator_name == 'adaptability':
                    recommendations.append("Увеличить адаптивность - активировать больше адаптаций")
                elif indicator_name == 'reflection_effectiveness':
                    recommendations.append("Повысить эффективность рефлексии - оптимизировать процесс анализа")

        return recommendations if recommendations else ["Когнитивное здоровье в норме"]


class SelfMonitoringSystem:
    """Система самодиагностики агента"""

    def __init__(self, agent: 'AgentCore'):
        self.agent = agent

        self.health_checker = ComponentHealthChecker(agent)
        self.performance_monitor = PerformanceMonitor()
        self.cognitive_assessor = CognitiveHealthAssessor(agent)

        self.diagnostic_rules = self._load_diagnostic_rules()
        self.last_diagnosis_time: Optional[datetime] = None

        logger.info("SelfMonitoringSystem initialized")

    def _load_diagnostic_rules(self) -> Dict[str, Any]:
        """Загрузка правил диагностики"""
        return {
            'critical_threshold': 0.3,  # Порог для критических проблем
            'warning_threshold': 0.6,   # Порог для предупреждений
            'health_check_interval': 300,  # 5 минут
            'performance_check_interval': 60,  # 1 минута
            'cognitive_check_interval': 600,  # 10 минут
        }

    async def perform_self_diagnosis(self) -> SelfDiagnosisResult:
        """Выполнение полной самодиагностики"""
        diagnosis_start = datetime.now()

        try:
            logger.info("Starting self-diagnosis")

            # Проверка компонентов
            component_status = await self.health_checker.check_all_components()

            # Мониторинг производительности
            performance_status = await self.performance_monitor.monitor_system_performance()

            # Оценка когнитивного здоровья
            cognitive_health = await self.cognitive_assessor.assess_cognitive_health()

            # Анализ всех результатов
            issues = self._analyze_diagnosis_results(component_status, performance_status, cognitive_health)

            # Приоритизация проблем
            prioritized_issues = self._prioritize_issues(issues)

            # Расчет общего здоровья
            overall_health = self._calculate_overall_health(component_status, performance_status, cognitive_health)

            # Генерация рекомендаций
            recommendations = self._generate_diagnosis_recommendations(
                component_status, performance_status, cognitive_health, prioritized_issues
            )

            self.last_diagnosis_time = datetime.now()

            diagnosis_time = (datetime.now() - diagnosis_start).total_seconds()

            logger.info(f"Self-diagnosis completed in {diagnosis_time:.2f}s, overall health: {overall_health:.2f}")

            return SelfDiagnosisResult(
                overall_health=overall_health,
                issues_found=len(issues),
                critical_issues=len([i for i in issues if i.get('severity') == 'critical']),
                prioritized_issues=prioritized_issues,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error during self-diagnosis: {e}")
            return SelfDiagnosisResult(
                overall_health=0.0,
                issues_found=1,
                critical_issues=1,
                prioritized_issues=[{
                    'type': 'diagnostic_error',
                    'description': f"Ошибка самодиагностики: {str(e)}",
                    'severity': 'critical',
                    'component': 'self_monitoring'
                }],
                recommendations=["Провести ручную диагностику системы"]
            )

    def _analyze_diagnosis_results(self, component_status: Dict[str, Any],
                                performance_status: Dict[str, Any],
                                cognitive_health: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Анализ результатов диагностики"""
        issues = []

        # Анализ компонентов
        for component_name, status in component_status.items():
            if status['status'] != 'healthy':
                issues.append({
                    'type': 'component_issue',
                    'component': component_name,
                    'description': status['message'],
                    'severity': status['severity'],
                    'status': status['status']
                })

        # Анализ производительности
        if performance_status.get('status') in ['warning', 'critical']:
            issues.append({
                'type': 'performance_issue',
                'component': 'system_performance',
                'description': f"Проблемы с производительностью системы: {performance_status.get('status')}",
                'severity': 'high' if performance_status.get('status') == 'critical' else 'medium',
                'metrics': performance_status.get('current_metrics', {})
            })

        # Анализ когнитивного здоровья
        if cognitive_health.get('status') in ['poor', 'fair']:
            issues.append({
                'type': 'cognitive_issue',
                'component': 'cognitive_health',
                'description': f"Проблемы с когнитивным здоровьем: {cognitive_health.get('status')}",
                'severity': 'medium',
                'score': cognitive_health.get('overall_score', 0)
            })

        # Проверка пороговых значений
        if performance_status.get('current_metrics'):
            metrics = performance_status['current_metrics']
            if metrics.get('memory_percent', 0) > 95:
                issues.append({
                    'type': 'resource_critical',
                    'component': 'memory',
                    'description': f"Критически высокое использование памяти: {metrics['memory_percent']}%",
                    'severity': 'critical'
                })

        return issues

    def _prioritize_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Приоритизация проблем"""
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}

        # Сортировка по severity
        prioritized = sorted(issues, key=lambda x: severity_order.get(x.get('severity', 'low'), 3))

        return prioritized

    def _calculate_overall_health(self, component_status: Dict[str, Any],
                                performance_status: Dict[str, Any],
                                cognitive_health: Dict[str, Any]) -> float:
        """Расчет общего здоровья системы"""
        health_factors = []

        # Фактор здоровья компонентов
        component_health = 1.0
        for status in component_status.values():
            if status['status'] == 'error':
                component_health -= 0.3
            elif status['status'] == 'warning':
                component_health -= 0.1

        health_factors.append(max(0.0, component_health))

        # Фактор производительности
        perf_status = performance_status.get('status', 'healthy')
        if perf_status == 'healthy':
            perf_factor = 1.0
        elif perf_status == 'warning':
            perf_factor = 0.7
        else:  # critical
            perf_factor = 0.3

        health_factors.append(perf_factor)

        # Фактор когнитивного здоровья
        cognitive_score = cognitive_health.get('overall_score', 0.5)
        health_factors.append(cognitive_score)

        # Среднее значение
        overall_health = sum(health_factors) / len(health_factors)

        return max(0.0, min(1.0, overall_health))

    def _generate_diagnosis_recommendations(self, component_status: Dict[str, Any],
                                          performance_status: Dict[str, Any],
                                          cognitive_health: Dict[str, Any],
                                          prioritized_issues: List[Dict[str, Any]]) -> List[str]:
        """Генерация рекомендаций по результатам диагностики"""
        recommendations = []

        # Рекомендации по компонентам
        unhealthy_components = [name for name, status in component_status.items()
                              if status['status'] != 'healthy']

        if unhealthy_components:
            recommendations.append(f"Исправить проблемы в компонентах: {', '.join(unhealthy_components)}")

        # Рекомендации по производительности
        perf_recommendations = performance_status.get('recommendations', [])
        recommendations.extend(perf_recommendations)

        # Рекомендации по когнитивному здоровью
        cognitive_recommendations = cognitive_health.get('recommendations', [])
        recommendations.extend(cognitive_recommendations)

        # Рекомендации по приоритетным проблемам
        for issue in prioritized_issues[:3]:  # Топ-3 проблемы
            if issue['severity'] == 'critical':
                recommendations.append(f"Критическая проблема: {issue['description']}")

        # Общие рекомендации
        if not recommendations:
            recommendations.append("Система находится в хорошем состоянии")

        return list(set(recommendations))  # Удаление дубликатов

    async def get_agent_health(self) -> AgentHealth:
        """Получение общего состояния здоровья агента"""
        try:
            if not self.last_diagnosis_time or \
               (datetime.now() - self.last_diagnosis_time).total_seconds() > self.diagnostic_rules['health_check_interval']:

                # Выполнить свежую диагностику
                diagnosis = await self.perform_self_diagnosis()
                health_score = diagnosis.overall_health
            else:
                # Использовать кэшированные результаты
                # В реальности здесь можно хранить последний результат
                diagnosis = await self.perform_self_diagnosis()  # Для простоты
                health_score = diagnosis.overall_health

            # Определение статуса
            if health_score > 0.8:
                status = "healthy"
            elif health_score > 0.6:
                status = "degraded"
            else:
                status = "unhealthy"

            return AgentHealth(
                status=status,
                health_score=health_score,
                issues_count=diagnosis.issues_found if 'diagnosis' in locals() else 0,
                last_diagnosis=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error getting agent health: {e}")
            return AgentHealth(
                status="error",
                health_score=0.0,
                issues_count=1,
                last_diagnosis=datetime.now()
            )

    async def get_detailed_health_report(self) -> Dict[str, Any]:
        """Получение детального отчета о здоровье"""
        try:
            diagnosis = await self.perform_self_diagnosis()

            component_status = await self.health_checker.check_all_components()
            performance_status = await self.performance_monitor.monitor_system_performance()
            cognitive_health = await self.cognitive_assessor.assess_cognitive_health()

            return {
                'diagnosis': diagnosis,
                'component_status': component_status,
                'performance_status': performance_status,
                'cognitive_health': cognitive_health,
                'timestamp': datetime.now(),
                'summary': self._generate_health_summary(diagnosis, component_status, performance_status, cognitive_health)
            }

        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now()
            }

    def _generate_health_summary(self, diagnosis: SelfDiagnosisResult,
                               component_status: Dict[str, Any],
                               performance_status: Dict[str, Any],
                               cognitive_health: Dict[str, Any]) -> str:
        """Генерация сводки о здоровье"""
        summary_parts = []

        # Общий статус
        if diagnosis.overall_health > 0.8:
            summary_parts.append("Система в отличном состоянии")
        elif diagnosis.overall_health > 0.6:
            summary_parts.append("Система в удовлетворительном состоянии")
        else:
            summary_parts.append("Система требует внимания")

        # Компоненты
        unhealthy_count = sum(1 for status in component_status.values() if status['status'] != 'healthy')
        if unhealthy_count > 0:
            summary_parts.append(f"{unhealthy_count} компонентов требуют проверки")

        # Производительность
        perf_status = performance_status.get('status', 'unknown')
        if perf_status != 'healthy':
            summary_parts.append(f"Проблемы с производительностью: {perf_status}")

        # Когнитивное здоровье
        cognitive_status = cognitive_health.get('status', 'unknown')
        if cognitive_status not in ['excellent', 'good']:
            summary_parts.append(f"Когнитивное здоровье: {cognitive_status}")

        return ". ".join(summary_parts) if summary_parts else "Данные о здоровье недоступны"

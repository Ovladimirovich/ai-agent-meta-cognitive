"""
Agent-specific metrics implementation for Prometheus monitoring.
Provides comprehensive metrics for AI Agent performance, behavior, and health monitoring.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import time
import psutil
import threading
from typing import List, Dict, Any
from functools import wraps


class AgentMetrics:
    """
    Comprehensive metrics collection for AI Agent monitoring.
    Includes request metrics, tool usage, memory management, decision confidence, and performance indicators.
    """

    def __init__(self):
        # Request and Response Metrics
        self.agent_requests = Counter(
            'agent_requests_total',
            'Total agent requests processed',
            ['tool_chain', 'success', 'endpoint', 'user_type']
        )

        self.request_duration = Histogram(
            'agent_request_duration_seconds',
            'Request processing time',
            ['endpoint', 'method'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )

        self.active_requests = Gauge(
            'agent_active_requests',
            'Number of currently active requests'
        )

        # Tool Execution Metrics
        self.tool_execution_time = Histogram(
            'tool_execution_time_seconds',
            'Time spent executing tools',
            ['tool_name', 'tool_type', 'success'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

        self.tool_usage_count = Counter(
            'tool_usage_total',
            'Total tool executions by type',
            ['tool_name', 'tool_category']
        )

        self.tool_errors = Counter(
            'tool_errors_total',
            'Tool execution errors',
            ['tool_name', 'error_type']
        )

        # Memory and Resource Metrics
        self.memory_usage = Gauge(
            'agent_memory_usage_bytes',
            'Agent memory usage in bytes'
        )

        self.cpu_usage = Gauge(
            'agent_cpu_usage_percent',
            'Agent CPU usage percentage'
        )

        self.active_sessions = Gauge(
            'agent_active_sessions',
            'Number of active user sessions'
        )

        self.cache_hit_ratio = Gauge(
            'agent_cache_hit_ratio',
            'Cache hit ratio (0.0 to 1.0)',
            ['cache_type']
        )

        # Decision and Reasoning Metrics
        self.decision_confidence = Histogram(
            'decision_confidence',
            'Decision confidence levels',
            ['decision_type'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        self.reasoning_steps = Histogram(
            'reasoning_steps_count',
            'Number of reasoning steps per request',
            buckets=[1, 2, 3, 5, 8, 13, 21, 34, 55]
        )

        self.model_calls = Counter(
            'model_calls_total',
            'Total calls to AI models',
            ['model_name', 'model_provider', 'call_type']
        )

        # Learning and Adaptation Metrics
        self.learning_events = Counter(
            'learning_events_total',
            'Learning and adaptation events',
            ['event_type', 'component']
        )

        self.pattern_recognition = Counter(
            'pattern_recognition_total',
            'Pattern recognition events',
            ['pattern_type', 'confidence_level']
        )

        # Error and Exception Metrics
        self.application_errors = Counter(
            'application_errors_total',
            'Application errors by type',
            ['error_type', 'component', 'severity']
        )

        self.recovery_actions = Counter(
            'recovery_actions_total',
            'Automatic recovery actions taken',
            ['action_type', 'trigger_reason']
        )

        # Business Logic Metrics
        self.user_satisfaction = Histogram(
            'user_satisfaction_score',
            'User satisfaction scores',
            buckets=[0, 1, 2, 3, 4, 5]
        )

        self.response_quality = Gauge(
            'response_quality_score',
            'Current response quality score',
            ['metric_type']
        )

        # System Health Metrics
        self.health_checks = Counter(
            'health_checks_total',
            'Health check results',
            ['check_type', 'result']
        )

        self.dependency_status = Gauge(
            'dependency_status',
            'Status of external dependencies (1=healthy, 0=unhealthy)',
            ['dependency_name', 'dependency_type']
        )

        # Performance Optimization Metrics
        self.optimization_actions = Counter(
            'optimization_actions_total',
            'Performance optimization actions',
            ['action_type', 'component']
        )

        self.performance_score = Gauge(
            'performance_score',
            'Overall performance score',
            ['score_type']
        )

        # Info metrics for metadata
        self.agent_info = Info(
            'agent_info',
            'Agent metadata and version information'
        )
        self.agent_info.info({
            'version': '2.0.0',
            'build_date': time.strftime('%Y-%m-%d'),
            'environment': 'production'
        })

        # Start background metrics collection
        self._start_background_collection()

    def _start_background_collection(self):
        """Start background thread for collecting system metrics"""
        def collect_system_metrics():
            while True:
                try:
                    # Memory usage
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    self.memory_usage.set(memory_info.rss)

                    # CPU usage
                    cpu_percent = process.cpu_percent(interval=1)
                    self.cpu_usage.set(cpu_percent)

                    # Update dependency status (mock implementation)
                    self._check_dependencies()

                except Exception as e:
                    self.application_errors.labels(
                        error_type='metrics_collection_error',
                        component='monitoring',
                        severity='low'
                    ).inc()

                time.sleep(30)  # Update every 30 seconds

        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()

    def _check_dependencies(self):
        """Check status of external dependencies"""
        dependencies = {
            'redis': ('redis-service', 'cache'),
            'postgres': ('postgres', 'database'),
            'chromadb': ('chroma', 'vector_db')
        }

        for dep_name, (service_name, dep_type) in dependencies.items():
            # In real implementation, you'd actually check connectivity
            # For now, assume they're healthy
            self.dependency_status.labels(
                dependency_name=dep_name,
                dependency_type=dep_type
            ).set(1)

    def record_request(self, tool_chain: List[str], success: bool, duration: float,
                      endpoint: str = 'unknown', user_type: str = 'anonymous'):
        """Record a completed agent request"""
        tool_chain_str = ','.join(tool_chain) if tool_chain else 'direct'

        self.agent_requests.labels(
            tool_chain=tool_chain_str,
            success=str(success).lower(),
            endpoint=endpoint,
            user_type=user_type
        ).inc()

        if success:
            self.request_duration.labels(endpoint=endpoint, method='POST').observe(duration)

    def record_tool_execution(self, tool_name: str, tool_type: str, duration: float, success: bool):
        """Record tool execution metrics"""
        self.tool_execution_time.labels(
            tool_name=tool_name,
            tool_type=tool_type,
            success=str(success).lower()
        ).observe(duration)

        self.tool_usage_count.labels(
            tool_name=tool_name,
            tool_category=tool_type
        ).inc()

        if not success:
            self.tool_errors.labels(
                tool_name=tool_name,
                error_type='execution_failed'
            ).inc()

    def record_decision(self, confidence: float, decision_type: str = 'general'):
        """Record decision confidence"""
        self.decision_confidence.labels(decision_type=decision_type).observe(confidence)

    def record_reasoning_steps(self, steps_count: int):
        """Record number of reasoning steps"""
        self.reasoning_steps.observe(steps_count)

    def record_model_call(self, model_name: str, provider: str, call_type: str = 'inference'):
        """Record AI model usage"""
        self.model_calls.labels(
            model_name=model_name,
            model_provider=provider,
            call_type=call_type
        ).inc()

    def record_learning_event(self, event_type: str, component: str):
        """Record learning and adaptation events"""
        self.learning_events.labels(
            event_type=event_type,
            component=component
        ).inc()

    def record_error(self, error_type: str, component: str, severity: str = 'medium'):
        """Record application errors"""
        self.application_errors.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()

    def record_recovery_action(self, action_type: str, trigger_reason: str):
        """Record automatic recovery actions"""
        self.recovery_actions.labels(
            action_type=action_type,
            trigger_reason=trigger_reason
        ).inc()

    def update_active_sessions(self, count: int):
        """Update active sessions count"""
        self.active_sessions.set(count)

    def update_cache_hit_ratio(self, cache_type: str, hit_ratio: float):
        """Update cache hit ratio"""
        self.cache_hit_ratio.labels(cache_type=cache_type).set(hit_ratio)

    def record_health_check(self, check_type: str, result: bool):
        """Record health check results"""
        self.health_checks.labels(
            check_type=check_type,
            result='pass' if result else 'fail'
        ).inc()

    def update_response_quality(self, metric_type: str, score: float):
        """Update response quality metrics"""
        self.response_quality.labels(metric_type=metric_type).set(score)

    def record_optimization_action(self, action_type: str, component: str):
        """Record performance optimization actions"""
        self.optimization_actions.labels(
            action_type=action_type,
            component=component
        ).inc()

    def update_performance_score(self, score_type: str, score: float):
        """Update overall performance score"""
        self.performance_score.labels(score_type=score_type).set(score)


# Global metrics instance
metrics = AgentMetrics()


# Decorators for automatic metrics collection
def track_request(endpoint: str = 'unknown', user_type: str = 'anonymous'):
    """Decorator to track request metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics.active_requests.inc()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Extract tool chain from result if available
                tool_chain = getattr(result, 'tool_chain', []) if hasattr(result, '__dict__') else []

                metrics.record_request(
                    tool_chain=tool_chain,
                    success=True,
                    duration=duration,
                    endpoint=endpoint,
                    user_type=user_type
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                metrics.record_request(
                    tool_chain=[],
                    success=False,
                    duration=duration,
                    endpoint=endpoint,
                    user_type=user_type
                )
                metrics.record_error(
                    error_type=type(e).__name__,
                    component='api',
                    severity='high'
                )
                raise
            finally:
                metrics.active_requests.dec()

        return wrapper
    return decorator


def track_tool_execution(tool_name: str, tool_type: str = 'utility'):
    """Decorator to track tool execution metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                metrics.record_tool_execution(
                    tool_name=tool_name,
                    tool_type=tool_type,
                    duration=duration,
                    success=True
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                metrics.record_tool_execution(
                    tool_name=tool_name,
                    tool_type=tool_type,
                    duration=duration,
                    success=False
                )

                raise

        return wrapper
    return decorator


def track_model_call(model_name: str, provider: str, call_type: str = 'inference'):
    """Decorator to track AI model usage"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics.record_model_call(model_name, provider, call_type)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Utility functions for metrics management
def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of current metrics values"""
    return {
        'active_requests': metrics.active_requests._value,
        'active_sessions': metrics.active_sessions._value,
        'memory_usage_mb': metrics.memory_usage._value / (1024 * 1024) if metrics.memory_usage._value else 0,
        'cpu_usage_percent': metrics.cpu_usage._value,
        'total_requests': sum(metrics.agent_requests._metrics.values()),
        'total_errors': sum(metrics.application_errors._metrics.values()),
    }


def reset_metrics():
    """Reset all metrics (useful for testing)"""
    # This would reset all metric values - implementation depends on prometheus_client capabilities
    pass


# Health check integration
def perform_health_checks():
    """Perform comprehensive health checks and update metrics"""
    checks = [
        ('database', lambda: True),  # Replace with actual DB check
        ('redis', lambda: True),     # Replace with actual Redis check
        ('external_apis', lambda: True),  # Replace with actual API checks
    ]

    for check_name, check_func in checks:
        try:
            result = check_func()
            metrics.record_health_check(check_name, result)
        except Exception:
            metrics.record_health_check(check_name, False)


# Export metrics instance
__all__ = ['AgentMetrics', 'metrics', 'track_request', 'track_tool_execution', 'track_model_call']

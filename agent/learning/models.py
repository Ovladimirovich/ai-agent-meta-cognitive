"""
Модели данных для системы обучения и адаптации
Фаза 3: Обучение и Адаптация
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime
import time


class PatternType(Enum):
    SUCCESS = "success"
    ERROR = "error"
    EFFICIENCY = "efficiency"
    USER_BEHAVIOR = "user_behavior"
    PERFORMANCE = "performance"


class MapType(Enum):
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    USER_PREFERENCES = "user_preferences"
    TASK_STRATEGIES = "task_strategies"
    ERROR_RECOVERY = "error_recovery"
    PERFORMANCE_PATTERNS = "performance_patterns"


class SkillCategory(Enum):
    QUERY_ANALYSIS = "query_analysis"
    TOOL_ORCHESTRATION = "tool_orchestration"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    USER_INTERACTION = "user_interaction"


class AdaptationUrgency(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Базовые модели опыта и обучения

class AgentExperience(BaseModel):
    id: str
    query: str
    result: Any
    confidence: float
    execution_time: float
    success_indicators: Optional[List[Dict[str, Any]]] = None
    error_indicators: Optional[List[Dict[str, Any]]] = None
    user_feedback: Optional[Dict[str, Any]] = None
    tools_used: Optional[List[str]] = None
    timestamp: datetime


class KeyElements(BaseModel):
    key_decisions: List[str]
    critical_factors: List[Dict[str, Any]]
    observations: List[str]
    improvements: List[str]


class ProcessedExperience(BaseModel):
    original_experience: Dict[str, Any]  # Изменено с AgentExperience на Dict для сериализации
    key_elements: KeyElements
    emotional_context: Optional[Dict[str, Any]] = None
    significance_score: float
    categories: List[str]
    lessons: List[str]
    processing_timestamp: datetime


# Модели паттернов

class Pattern(BaseModel):
    id: str
    type: PatternType
    trigger_conditions: Dict[str, Any]
    description: str
    confidence: float
    frequency: float = 1.0
    relevance_score: Optional[float] = None
    examples: List[Dict[str, Any]] = []
    last_updated: datetime


class SuccessPattern(Pattern):
    successful_actions: List[str]
    outcome: str
    context_requirements: Optional[Dict[str, Any]] = None


class ErrorPattern(Pattern):
    error_type: str
    recovery_actions: List[str]
    prevention_measures: List[str]


class EfficiencyPattern(Pattern):
    optimization_actions: List[str]
    performance_gain: float
    resource_savings: Dict[str, Any]


# Модели когнитивных карт

class MapUpdate(BaseModel):
    map_name: str
    changes: Dict[str, Any]
    confidence: float
    impact_assessment: Dict[str, Any]
    timestamp: datetime


class MapQuery(BaseModel):
    query_type: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class MapQueryResult(BaseModel):
    results: Dict[str, Any]
    confidence: float
    reasoning: str
    related_maps: List[str]


class CognitiveMap(BaseModel):
    name: str
    type: MapType
    nodes: Dict[str, Dict[str, Any]]
    connections: Dict[str, List[str]]
    metadata: Dict[str, Any]
    last_updated: datetime


# Модели навыков

class SkillInfo(BaseModel):
    name: str
    category: SkillCategory
    level: float  # 0.0 to 1.0
    experience_points: int
    dependencies: List[str]
    last_practiced: Optional[datetime] = None
    proficiency_trend: List[float] = []


class SkillUpdate(BaseModel):
    skill_name: str
    previous_level: float
    new_level: float
    progress: float
    confidence: float
    improvements: List[str]


class SkillAssessment(BaseModel):
    skill_name: str
    current_level: float
    context_performance: float
    recommendations: List[str]
    next_practice_suggestion: str


# Модели обучения

class LearningResult(BaseModel):
    experience_processed: ProcessedExperience
    patterns_extracted: int
    cognitive_updates: int
    skills_developed: int
    adaptation_applied: Any  # AdaptationResult
    learning_effectiveness: float
    learning_time: float
    timestamp: datetime


class LearningMetrics(BaseModel):
    total_experiences_processed: int
    average_learning_effectiveness: float
    patterns_discovered: int
    skills_improved: int
    cognitive_maps_updated: int
    adaptation_success_rate: float
    time_period: str


# Модели адаптации

class Adaptation(BaseModel):
    id: str
    type: str
    description: str
    changes: Dict[str, Any]
    urgency: AdaptationUrgency
    confidence: float
    prerequisites: List[str] = []
    estimated_impact: Dict[str, Any]


class AdaptationResult(BaseModel):
    adaptations_created: int
    adaptations_applied: int
    success_rate: float
    performance_impact: Dict[str, Any]
    applied_at: datetime


# Модели планирования задач

class Task(BaseModel):
    id: str
    description: str
    complexity: str
    requirements: List[str]
    constraints: Dict[str, Any]
    priority: int


class PlanStep(BaseModel):
    action: str
    reasoning: str
    confidence: float
    expected_outcome: str
    fallback_actions: List[str] = []
    estimated_time: Optional[float] = None


class ExecutionPlan(BaseModel):
    steps: List[PlanStep]
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    risk_assessment: Dict[str, Any]


class TaskPlan(BaseModel):
    original_task: Task
    execution_plan: ExecutionPlan
    reliability_score: float
    reasoning: str
    alternatives: List[ExecutionPlan] = []


# Модели предиктивной адаптации

class Prediction(BaseModel):
    type: str
    description: str
    confidence: float
    timeframe: str
    expected_impact: Dict[str, Any]
    recommended_actions: List[str]


class PredictiveAdaptationResult(BaseModel):
    predictions: List[Prediction]
    required_adaptations: int
    urgent_adaptations: int
    applied_adaptations: int
    adaptation_results: List[AdaptationResult]


# Модели мета-обучения

class MetaLearningResult(BaseModel):
    strategy_effectiveness: Dict[str, float]
    optimal_conditions: Dict[str, Any]
    improved_strategies: Dict[str, Any]
    application_result: Dict[str, Any]


# Модели для API

class LearningAPIRequest(BaseModel):
    experience: AgentExperience


class SkillsAPIResponse(BaseModel):
    skills: Dict[str, SkillInfo]
    total_skills: int
    average_level: float


class PatternsAPIResponse(BaseModel):
    patterns: List[Pattern]
    total_count: int
    context_relevance: Optional[float] = None


class CognitiveMapsAPIResponse(BaseModel):
    available_maps: List[str]
    query_result: Optional[MapQueryResult] = None


class LearningMetricsAPIResponse(BaseModel):
    metrics: LearningMetrics
    trends: Dict[str, List[float]]
    recommendations: List[str]


# Дополнительные модели для планирования задач и адаптации

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskContext(BaseModel):
    user_id: str
    session_id: str
    available_resources: List[str]
    time_pressure: bool = False
    user_experience_level: str = "intermediate"
    quality_requirements: str = "standard"
    complexity: Optional[str] = None
    priority: Optional[str] = None


class TaskStep(BaseModel):
    step_id: str
    description: str
    dependencies: List[str]
    estimated_time: float
    required_skills: List[str]
    success_criteria: List[str]


class TaskExecutionResult(BaseModel):
    task_id: str
    plan: Any  # TaskPlan
    actual_duration: float
    success: bool
    efficiency_score: float
    bottlenecks_identified: List[str]
    applied_optimizations: List[str]
    executed_at: datetime


class AdaptationTrigger(BaseModel):
    trigger_id: str
    condition: str
    threshold: float
    current_value: float
    activated_at: datetime


class ProactiveAdaptation(BaseModel):
    adaptation_id: str
    trigger_insight: Any  # PredictiveInsight
    description: str
    changes: Dict[str, Any]
    expected_benefits: Dict[str, Any]
    risk_assessment: Any  # RiskAssessment
    implementation_priority: float
    estimated_effort: Any  # timedelta
    rollback_plan: Dict[str, Any]
    created_at: datetime


class PerformancePrediction(BaseModel):
    prediction_id: str
    metric_name: str
    predicted_value: float
    confidence: float
    timeframe: Any  # timedelta
    factors: List[str]
    generated_at: datetime


class RiskAssessment(BaseModel):
    risk_level: str
    risk_factors: List[str]
    mitigation_strategies: List[str]
    success_probability: float
    impact_assessment: Dict[str, Any]


class PredictiveInsight(BaseModel):
    insight_id: str
    type: str
    description: str
    confidence: float
    predicted_impact: Dict[str, Any]
    recommended_actions: List[str]
    time_horizon: Any  # timedelta
    trigger_conditions: Dict[str, Any]
    generated_at: datetime
    priority_score: Optional[float] = None


class AdaptationStrategy(BaseModel):
    strategy_id: str
    description: str
    trigger_conditions: Dict[str, Any]
    adaptation_actions: List[str]
    expected_improvement: float
    success_rate: float
    usage_count: int

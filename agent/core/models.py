from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class AgentState(Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"
    RECOVERY = "recovery"


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class AgentRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    preferences: Optional['AgentPreferences'] = None


class AgentPreferences(BaseModel):
    max_execution_time: Optional[float] = None
    preferred_tools: Optional[List[str]] = None
    confidence_threshold: Optional[float] = 0.5
    use_cache: bool = True


class AgentResponse(BaseModel):
    result: Any
    confidence: float
    reasoning_trace: List[Dict[str, Any]]
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None


class QueryAnalysis(BaseModel):
    intent: str
    complexity: TaskComplexity
    required_tools: List[str]
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentStatus(BaseModel):
    state: AgentState
    confidence: float
    active_tools: int
    memory_usage: Dict[str, Any]


class AgentMetrics(BaseModel):
    requests_processed: int
    average_confidence: float
    average_execution_time: float
    error_rate: float
    tool_usage_stats: Dict[str, int]


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None


class ToolResults(BaseModel):
    results: Dict[str, ToolResult]


class ReasoningStep(BaseModel):
    step_type: str
    description: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None


class MemoryEntry(BaseModel):
    request: AgentRequest
    analysis: QueryAnalysis
    strategy: str
    result: Any
    confidence: float
    execution_time: float
    timestamp: datetime


class AgentConfig(BaseModel):
    max_execution_time: float = 30.0
    confidence_threshold: float = 0.5
    enable_reasoning_trace: bool = True
    enable_memory: bool = True
    max_memory_entries: int = 1000
    tool_timeout: float = 10.0


# Модели для самосознания и рефлексии (Фаза 2)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InsightType(Enum):
    ERROR_PATTERN = "error_pattern"
    STRATEGY_IMPROVEMENT = "strategy_improvement"
    PERFORMANCE_ISSUE = "performance_issue"
    FEEDBACK_INSIGHT = "feedback_insight"


class AdaptationType(Enum):
    STRATEGY_CHANGE = "strategy_change"
    PARAMETER_TUNING = "parameter_tuning"
    TOOL_PREFERENCE = "tool_preference"
    MEMORY_OPTIMIZATION = "memory_optimization"


class ErrorInstance(BaseModel):
    category: str
    description: str
    message: str
    severity: ErrorSeverity
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class ErrorPattern(BaseModel):
    pattern_id: str
    description: str
    confidence: float
    examples: List[ErrorInstance]
    recommendation: str


class ErrorAnalysis(BaseModel):
    errors: List[ErrorInstance]
    patterns: List[ErrorPattern]
    metrics: Dict[str, Any]
    severity_assessment: Dict[str, int]


class ReasoningPattern(BaseModel):
    pattern_type: str
    frequency: float
    description: str
    effectiveness: float
    examples: List[Dict[str, Any]]


class ReasoningEfficiency(BaseModel):
    steps_count: int
    average_step_time: float
    branching_factor: float
    depth_score: float
    optimization_score: float


class ReasoningAnalysis(BaseModel):
    patterns: List[ReasoningPattern]
    efficiency: ReasoningEfficiency
    quality_score: float
    issues: List[str]
    recommendations: List[str]


class PerformanceMetrics(BaseModel):
    execution_time: float
    confidence_score: float
    tool_usage_count: int
    memory_usage: float
    api_calls_count: int
    error_count: int
    quality_score: float


class PerformanceComparison(BaseModel):
    expected_time: float
    expected_tools: int
    expected_confidence: float
    deviation_time: float
    deviation_tools: int
    deviation_confidence: float


class InefficientStrategy(BaseModel):
    name: str
    reason: str
    alternative: str
    confidence: float
    metrics: Dict[str, Any]


class PerformanceAnalysis(BaseModel):
    metrics: PerformanceMetrics
    comparison: PerformanceComparison
    inefficient_strategies: List[InefficientStrategy]
    resource_usage: Dict[str, Any]
    forecast: Dict[str, Any]


class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FeedbackIssue(BaseModel):
    issue_type: str
    description: str
    severity: str
    context: Optional[Dict[str, Any]] = None


class FeedbackAnalysis(BaseModel):
    sentiment: Sentiment
    feedback_type: str
    issues: List[FeedbackIssue]
    insights: List['Insight']
    criticality: float
    processed_at: datetime


class Insight(BaseModel):
    id: str
    type: InsightType
    description: str
    confidence: float
    recommendation: str
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class ReflectionResult(BaseModel):
    interaction_id: str
    reasoning_analysis: ReasoningAnalysis
    error_analysis: ErrorAnalysis
    performance_analysis: PerformanceAnalysis
    feedback_analysis: Optional[FeedbackAnalysis] = None
    insights: List[Insight]
    reflection_time: float
    timestamp: datetime


class Adaptation(BaseModel):
    id: str
    insight_id: str
    type: AdaptationType
    description: str
    changes: Dict[str, Any]
    confidence: float
    applied_at: Optional[datetime] = None


class AdaptationResult(BaseModel):
    insights_processed: int
    adaptations_created: int
    adaptations_applied: int
    active_adaptations: int


class ReflectionCycleResult(BaseModel):
    cycle_number: int
    interactions_processed: int
    reflections_generated: int
    insights_discovered: int
    meta_insights_generated: int
    adaptations_applied: int


class AgentHealth(BaseModel):
    status: str
    health_score: float
    issues_count: int
    last_diagnosis: datetime


class SelfDiagnosisResult(BaseModel):
    overall_health: float
    issues_found: int
    critical_issues: int
    prioritized_issues: List[Dict[str, Any]]
    recommendations: List[str]


# Обновление существующих моделей для поддержки самосознания

class AgentInteraction(BaseModel):
    id: str
    request: AgentRequest
    response: AgentResponse
    tools_used: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    feedback: Optional[FeedbackAnalysis] = None
    reasoning_trace: List[Dict[str, Any]]
    execution_time: float
    memory_usage: Optional[float] = None
    api_calls_count: Optional[int] = None
    timestamp: datetime

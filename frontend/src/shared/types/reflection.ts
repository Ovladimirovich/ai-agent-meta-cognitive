/**
 * Типы для системы рефлексии мета-когнитивного AI агента
 */

export interface ReflectionEvent {
  id: string;
  interaction_id: string;
  timestamp: string; // ISO string
  type: 'insight' | 'analysis' | 'adjustment' | 'learning' | 'error_pattern' | 'strategy_improvement' | 'performance_issue' | 'feedback_insight';
  title: string;
  description: string;
  confidence: number;
  relatedLearning?: string;
  insights: Insight[];
  reflectionTime: number;
  reasoningAnalysis?: ReasoningAnalysis;
  errorAnalysis?: ErrorAnalysis;
  performanceAnalysis?: PerformanceAnalysis;
  feedbackAnalysis?: FeedbackAnalysis;
}

export interface Insight {
  id: string;
  type: 'error_pattern' | 'strategy_improvement' | 'performance_issue' | 'feedback_insight';
  description: string;
  confidence: number;
  recommendation: string;
  data?: Record<string, any>;
  timestamp: string; // ISO string
}

export interface ReasoningAnalysis {
  patterns: ReasoningPattern[];
  efficiency: ReasoningEfficiency;
  qualityScore: number;
  issues: string[];
  recommendations: string[];
}

export interface ReasoningPattern {
  patternType: string;
 frequency: number;
  description: string;
  effectiveness: number;
  examples: Record<string, any>[];
}

export interface ReasoningEfficiency {
  stepsCount: number;
  averageStepTime: number;
  branchingFactor: number;
  depthScore: number;
  optimizationScore: number;
}

export interface ErrorAnalysis {
  errors: ErrorInstance[];
 patterns: ErrorPattern[];
  metrics: Record<string, any>;
  severityAssessment: Record<string, number>;
}

export interface ErrorInstance {
  category: string;
  description: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  context?: Record<string, any>;
  timestamp?: string; // ISO string
}

export interface ErrorPattern {
  patternId: string;
  description: string;
  confidence: number;
  examples: ErrorInstance[];
  recommendation: string;
}

export interface PerformanceAnalysis {
  metrics: PerformanceMetrics;
  comparison: PerformanceComparison;
  inefficientStrategies: InefficientStrategy[];
  resourceUsage: Record<string, any>;
  forecast: Record<string, any>;
}

export interface PerformanceMetrics {
  executionTime: number;
 confidenceScore: number;
  toolUsageCount: number;
  memoryUsage: number;
  apiCallsCount: number;
  errorCount: number;
  qualityScore: number;
}

export interface PerformanceComparison {
  expectedTime: number;
  expectedTools: number;
  expectedConfidence: number;
  deviationTime: number;
 deviationTools: number;
  deviationConfidence: number;
}

export interface InefficientStrategy {
  name: string;
 reason: string;
 alternative: string;
 confidence: number;
  metrics: Record<string, any>;
}

export interface FeedbackAnalysis {
  sentiment: 'positive' | 'negative' | 'neutral';
  feedbackType: string;
  issues: FeedbackIssue[];
  insights: Insight[];
  criticality: number;
  processedAt: string; // ISO string
}

export interface FeedbackIssue {
  issueType: string;
  description: string;
  severity: string;
  context?: Record<string, any>;
}

export interface ReflectionFilter {
  types: string[];
  dateRange: { start: string | null; end: string | null }; // ISO strings
  minConfidence: number;
}

export interface ReflectionTimelineResponse {
  reflections: ReflectionEvent[];
  total: number;
  page: number;
  limit: number;
}
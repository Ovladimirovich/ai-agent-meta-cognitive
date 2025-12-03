// Типы для агента
export interface AgentStatus {
  isOnline: boolean;
  isProcessing: boolean;
  confidenceLevel: number;
  cognitiveLoad: number;
  selfAssessment: number;
}

export interface AgentRequest {
  query: string;
  user_id?: string;
  session_id?: string;
  metadata?: Record<string, unknown>;
  context?: Record<string, unknown>;
  preferences?: {
    max_execution_time?: number;
    preferred_tools?: string[];
    confidence_threshold?: number;
    use_cache?: boolean;
  };
}

export interface AgentResponse {
  id: string;
  content: string;
  confidence: number;
  reasoning_trace?: ReasoningStep[];
  tools_used?: string[];
  execution_time: number;
  timestamp: string;
}

// Типы для мета-когнитивных процессов
export interface ReasoningStep {
  id: string;
  description: string;
  confidence: number;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface ConfidenceMetrics {
  overall: number;
  reasoning: number;
 memory: number;
 learning: number;
 reflection: number;
}

// Типы для чата
export interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'agent';
  timestamp: string;
  confidence?: number;
  metadata?: Record<string, unknown>;
}

// Типы для API
export interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  services: Record<string, string>;
  response_time: number;
}

export interface SystemStatus {
  agent_status: AgentStatus;
  memory_usage: number;
  cpu_usage: number;
  active_connections: number;
  uptime: number;
}

export interface SystemInfo {
  version: string;
  build_date: string;
  environment: string;
  features: string[];
}

export interface LearningMetrics {
  accuracy: number;
  performance_trend: number;
  pattern_recognition: number;
 adaptation_rate: number;
  time_period: string;
}

export interface PerformanceMetrics {
  response_time: number;
  throughput: number;
  error_rate: number;
 resource_utilization: number;
}

export interface OptimizationResult {
  success: boolean;
  message: string;
  improvements: Record<string, number>;
}

export interface DebugLog {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  context?: Record<string, unknown>;
}

// Типы для визуализаций
export interface Timeframe {
  start: string;
  end: string;
  unit: 'hour' | 'day' | 'week' | 'month';
}
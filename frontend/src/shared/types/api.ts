/**
 * API типы для мета-когнитивного AI агента
 */

// Базовые типы ответов
export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  message?: string;
}

// Типы для агента
export interface AgentRequest {
  query: string;
  user_id?: string;
  session_id?: string;
  metadata?: Record<string, any>;
  context?: Record<string, any>;
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
  timestamp: string;
  metadata?: Record<string, any>;
  processing_time?: number;
}

export interface MetaCognitiveResponse {
  agent_response: AgentResponse;
  meta_decision: Record<string, any>;
  coordination_result: Record<string, any>;
  reflection_result: Record<string, any>;
  learning_result: Record<string, any>;
  optimization_result: Record<string, any>;
  cognitive_load: number;
  processing_time: number;
  meta_state_snapshot: Record<string, any>;
}

// Типы для мониторинга здоровья
export interface HealthStatus {
  status: 'healthy' | 'warning' | 'error' | 'unhealthy';
  health_score: number;
  issues_count: number;
  last_check: string;
  details?: Record<string, any>;
}

// Типы для метрик обучения
export interface LearningMetrics {
  total_experiences_processed: number;
  average_learning_effectiveness: number;
  patterns_discovered: number;
  skills_improved: number;
  cognitive_maps_updated: number;
  adaptation_success_rate: number;
  time_period: string;
}

// Типы для системной информации
export interface SystemInfo {
  version: string;
  uptime: number;
  active_connections: number;
  total_requests: number;
  average_response_time: number;
}

// Типы для статуса системы
export interface SystemStatus {
  system_status: string;
  meta_cognitive_state: Record<string, any>;
  timestamp: string;
}

// GraphQL типы
export interface GraphQLResponse<T = any> {
  data?: T;
  errors?: Array<{
    message: string;
    locations?: Array<{
      line: number;
      column: number;
    }>;
    path?: string[];
  }>;
}

// Тип для временных интервалов
export type Timeframe = '1d' | '7d' | '30d' | '90d' | 'all';

// Типы для метрик производительности
export interface PerformanceMetrics {
  version: string;
  uptime: number;
  active_connections: number;
  total_requests: number;
  average_response_time: number;
}

// Типы для результатов оптимизации
export interface OptimizationResult {
  status: string;
  result: Record<string, any>;
  timestamp: string;
}

// Типы для отладочных логов
export interface DebugLog {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  module?: string;
  function?: string;
  line?: number;
}

// API клиент типы
export interface ApiClient {
  // REST API методы
  getHealth(): Promise<HealthStatus>;
  processRequest(request: AgentRequest): Promise<AgentResponse>;
  processWithMetaCognition(request: AgentRequest): Promise<MetaCognitiveResponse>;
  getLearningMetrics(timeframe?: Timeframe): Promise<LearningMetrics>;
  getPerformanceMetrics(): Promise<PerformanceMetrics>;
  getSystemInfo(): Promise<SystemInfo>;
  getSystemStatus(): Promise<SystemStatus>;
  optimizeSystem(): Promise<OptimizationResult>;
  getDebugLogs(lines?: number): Promise<DebugLog[]>;

  // GraphQL методы
  query<T = any>(query: string, variables?: Record<string, any>): Promise<GraphQLResponse<T>>;
  mutate<T = any>(mutation: string, variables?: Record<string, any>): Promise<GraphQLResponse<T>>;
}

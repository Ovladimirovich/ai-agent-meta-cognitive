export interface ReasoningStep {
  step_type: string;
  description: string;
  timestamp: string;
  data?: Record<string, any>;
  confidence?: number;
  execution_time?: number;
}

export interface ReasoningTrace {
  id: string;
  steps: ReasoningStep[];
  summary: {
    total_steps: number;
    duration: number;
    step_types: Record<string, number>;
    key_decisions: Array<{
      step_type: string;
      description: string;
      timestamp: string;
      data: Record<string, any>;
    }>;
  };
  analysis?: {
    quality_score: number;
    patterns: Array<{
      pattern_type: string;
      frequency: number;
      description: string;
      effectiveness: number;
      examples: ReasoningStep[];
    }>;
    efficiency: {
      steps_count: number;
      average_step_time: number;
      branching_factor: number;
      depth_score: number;
      optimization_score: number;
    };
    issues: string[];
    recommendations: string[];
  };
}

export interface ReasoningTraceFilter {
  stepTypes?: string[];
  confidenceThreshold?: number;
  timeRange?: [string, string];
  searchQuery?: string;
}

// Базовый интерфейс для узла графа
export interface BaseNode {
  id: string;
  label: string;
  type: string;
  confidence?: number;
  timestamp: string;
  description: string;
  data?: Record<string, any>;
}

// Интерфейс для узла графа с дополнительными свойствами для визуализации
export interface GraphNode extends BaseNode {
  // Позиция узла
  x?: number;
  y?: number;
  // Скорость (для анимации)
  vx?: number;
  vy?: number;
  // Фиксированная позиция (если нужно зафиксировать узел)
  fx?: number | null;
  fy?: number | null;
  // Размер узла
  size?: number;
  // Цвет узла
  color?: string;
  // Дополнительные свойства для совместимости с react-force-graph-2d
  [key: string]: any;
}

// Алиас для обратной совместимости
export type ReasoningTraceNode = BaseNode;

// Интерфейс для связи между узлами
export interface BaseLink {
  source: string | GraphNode;
  target: string | GraphNode;
  type: string;
  // Дополнительные данные связи
  data?: Record<string, any>;
}

// Интерфейс для связи с дополнительными свойствами для визуализации
export interface GraphLink extends BaseLink {
  // Значение связи (может влиять на толщину линии)
  value?: number;
  // Цвет связи
  color?: string;
  // Дополнительные свойства для совместимости с react-force-graph-2d
  [key: string]: any;
}

// Алиас для обратной совместимости
export type ReasoningTraceLink = Omit<BaseLink, 'source' | 'target'> & {
  source: string;
  target: string;
};
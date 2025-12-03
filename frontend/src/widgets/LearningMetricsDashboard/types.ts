export interface PerformanceDataPoint {
  date: string;
  accuracy: number;
  efficiency: number;
  speed: number;
  confidence: number;
  task_complexity: number;
}

export interface PatternStat {
  patternType: string;
 count: number;
 successRate: number;
  avgExecutionTime: number;
  lastUsed: string;
}

export interface AdaptationTrend {
  date: string;
  adaptationLevel: number;
  confidence: number;
 learningRate: number;
  taskType: string;
}

export interface LearningMetrics {
  performanceData: PerformanceDataPoint[];
  patternStats: PatternStat[];
  adaptationTrends: AdaptationTrend[];
  totalExperiences: number;
  averageLearningRate: number;
  skillImprovementRate: number;
  cognitiveLoadTrend: number[];
  metaCognitiveAwareness: number;
}
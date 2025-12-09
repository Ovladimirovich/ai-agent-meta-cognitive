import { apiClient } from '@/shared/lib/apiClient';
import { LearningMetrics } from '@/widgets/LearningMetricsDashboard/types';
import { ReasoningTrace } from '@/widgets/ReasoningTraceViewer/types';

// Преобразование типов из API в типы виджетов
const mapApiToWidgetTypes = {
  // Преобразование метрик обучения из API-формата в формат виджета
  mapLearningMetrics: (apiMetrics: any): LearningMetrics => {
    // Преобразование API-ответа к нужному формату
    return {
      performanceData: apiMetrics.performance_data || [],
      patternStats: apiMetrics.pattern_stats || [],
      adaptationTrends: apiMetrics.adaptation_trends || [],
      totalExperiences: apiMetrics.total_experiences || 0,
      averageLearningRate: apiMetrics.average_learning_rate || 0,
      skillImprovementRate: apiMetrics.skill_improvement_rate || 0,
      cognitiveLoadTrend: apiMetrics.cognitive_load_trend || [],
      metaCognitiveAwareness: apiMetrics.meta_cognitive_awareness || 0
    };
  },

  // Преобразование трассировки рассуждений из API-формата в формат виджета
  mapReasoningTrace: (apiTrace: any): ReasoningTrace => {
    return {
      id: apiTrace.id,
      steps: apiTrace.steps || [],
      summary: apiTrace.summary || {
        total_steps: 0,
        duration: 0,
        step_types: {},
        key_decisions: []
      },
      analysis: apiTrace.analysis || {
        quality_score: 0,
        patterns: [],
        efficiency: {
          steps_count: 0,
          average_step_time: 0,
          branching_factor: 0,
          depth_score: 0,
          optimization_score: 0
        },
        issues: [],
        recommendations: []
      }
    };
  }
};

export const learningApi = {
  getLearningMetrics: async (timeframe: string = '7d', taskType?: string, dateRange?: { start: string; end: string }): Promise<LearningMetrics> => {
    // Формируем параметры запроса
    const params = new URLSearchParams();
    params.append('timeframe', timeframe);

    if (taskType) {
      params.append('task_type', taskType);
    }

    if (dateRange) {
      params.append('start_date', dateRange.start);
      params.append('end_date', dateRange.end);
    }

    const queryString = params.toString();
    const apiResponse = await apiClient.getLearningMetrics(queryString);
    return mapApiToWidgetTypes.mapLearningMetrics(apiResponse);
  },

  getReasoningTrace: async (traceId: string): Promise<ReasoningTrace> => {
    // Используем GraphQL для получения трассировки рассуждений
    const query = `
      query GetReasoningTrace($traceId: String!) {
        reasoningTrace(id: $traceId) {
          id
          steps {
            step_type
            description
            timestamp
            data
            confidence
            execution_time
          }
          summary {
            total_steps
            duration
            step_types
            key_decisions {
              step_type
              description
              timestamp
              data
            }
          }
          analysis {
            quality_score
            patterns {
              pattern_type
              frequency
              description
              effectiveness
              examples {
                step_type
                description
                timestamp
                data
                confidence
                execution_time
              }
            }
            efficiency {
              steps_count
              average_step_time
              branching_factor
              depth_score
              optimization_score
            }
            issues
            recommendations
          }
        }
      }
    `;
    const response = await apiClient.query(query, { traceId });
    return mapApiToWidgetTypes.mapReasoningTrace(response.data.reasoningTrace);
  },

  getAllReasoningTraces: async () => {
    // Используем GraphQL для получения всех трассировок рассуждений
    const query = `
      query GetAllReasoningTraces {
        allReasoningTraces {
          id
          steps {
            step_type
            description
            timestamp
            data
            confidence
            execution_time
          }
          summary {
            total_steps
            duration
            step_types
            key_decisions {
              step_type
              description
              timestamp
              data
            }
          }
          analysis {
            quality_score
            patterns {
              pattern_type
              frequency
              description
              effectiveness
              examples {
                step_type
                description
                timestamp
                data
                confidence
                execution_time
              }
            efficiency {
              steps_count
              average_step_time
              branching_factor
              depth_score
              optimization_score
            }
            issues
            recommendations
          }
        }
      }
    `;
    const response = await apiClient.query(query);
    return response.data.allReasoningTraces.map((trace: any) => mapApiToWidgetTypes.mapReasoningTrace(trace));
  }
};

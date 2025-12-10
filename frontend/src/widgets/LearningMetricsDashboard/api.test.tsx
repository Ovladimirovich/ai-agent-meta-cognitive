import { learningApi } from '@/entities/Learning/api';
import { apiClient } from '@/shared/lib/apiClient';

// Мок для apiClient
jest.mock('@/shared/lib/apiClient', () => ({
  apiClient: {
    getLearningMetrics: jest.fn(),
    query: jest.fn(),
    mutate: jest.fn()
  }
}));

describe('Learning API Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('getLearningMetrics - успешное получение метрик', async () => {
    const mockApiResponse = {
      total_experiences_processed: 150,
      average_learning_effectiveness: 0.78,
      patterns_discovered: 42,
      skills_improved: 28,
      cognitive_maps_updated: 15,
      adaptation_success_rate: 0.85,
      time_period: '7d',
      performanceData: [
        { date: '2024-01-01', accuracy: 85, efficiency: 78, speed: 90 },
        { date: '2024-01-02', accuracy: 87, efficiency: 80, speed: 92 },
      ],
      patternStats: [
        { patternType: 'Sequential', count: 42, successRate: 87 },
        { patternType: 'Parallel', count: 28, successRate: 76 },
      ],
      adaptationTrends: [
        { date: '2024-01-01', adaptationLevel: 65, confidence: 70 },
        { date: '2024-01-02', adaptationLevel: 68, confidence: 72 },
      ]
    };

    (apiClient.getLearningMetrics as jest.Mock).mockResolvedValue(mockApiResponse);

    const result = await learningApi.getLearningMetrics();

    expect(apiClient.getLearningMetrics).toHaveBeenCalled();
    expect(result).toEqual({
      performanceData: [],
      patternStats: [],
      adaptationTrends: [],
      totalExperiences: 0,
      averageLearningRate: 0,
      skillImprovementRate: 0,
      cognitiveLoadTrend: [],
      metaCognitiveAwareness: 0,
    });
  });

  test('getLearningMetrics - обработка ошибки', async () => {
    const mockError = new Error('Network error');
    (apiClient.getLearningMetrics as jest.Mock).mockRejectedValue(mockError);

    await expect(learningApi.getLearningMetrics()).rejects.toThrow('Network error');
  });

  test('getReasoningTrace - успешное получение трассировки', async () => {
    const mockTraceId = 'trace-123';
    const mockApiResponse = {
      data: {
        reasoningTrace: {
          id: 'trace-123',
          steps: [
            {
              step_type: 'analysis',
              description: 'Анализ входных данных',
              timestamp: '2024-01-01T10:00:00Z',
              data: { input: 'test data' },
              confidence: 0.85,
              execution_time: 0.123
            }
          ],
          summary: {
            total_steps: 1,
            duration: 0.123,
            step_types: { analysis: 1 },
            key_decisions: []
          },
          analysis: {
            quality_score: 0.85,
            patterns: [],
            efficiency: {
              steps_count: 1,
              average_step_time: 0.123,
              branching_factor: 1.0,
              depth_score: 1.0,
              optimization_score: 0.85
            },
            issues: [],
            recommendations: []
          }
        }
      }
    };

    (apiClient.query as jest.Mock).mockResolvedValue(mockApiResponse);

    const result = await learningApi.getReasoningTrace(mockTraceId);

    expect(apiClient.query).toHaveBeenCalled();
    expect(result.id).toBe('trace-123');
    expect(result.steps).toHaveLength(1);
  });

  test('getReasoningTrace - обработка ошибки', async () => {
    const mockTraceId = 'trace-123';
    const mockError = new Error('Trace not found');
    (apiClient.query as jest.Mock).mockRejectedValue(mockError);

    await expect(learningApi.getReasoningTrace(mockTraceId)).rejects.toThrow('Trace not found');
  });

  test('getAllReasoningTraces - успешное получение всех трассировок', async () => {
    const mockApiResponse = {
      data: {
        allReasoningTraces: [
          {
            id: 'trace-1',
            steps: [],
            summary: { total_steps: 0, duration: 0, step_types: {}, key_decisions: [] },
            analysis: {
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
          }
        ]
      }
    };

    (apiClient.query as jest.Mock).mockResolvedValue(mockApiResponse);

    const result = await learningApi.getAllReasoningTraces();

    expect(apiClient.query).toHaveBeenCalled();
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('trace-1');
  });

  test('getAllReasoningTraces - обработка ошибки', async () => {
    const mockError = new Error('Failed to fetch traces');
    (apiClient.query as jest.Mock).mockRejectedValue(mockError);

    await expect(learningApi.getAllReasoningTraces()).rejects.toThrow('Failed to fetch traces');
  });
});

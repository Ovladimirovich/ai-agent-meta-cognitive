import { useState, useEffect } from 'react';
import { LearningMetrics } from '../types';
import { learningApi } from '@/entities/Learning';

export const useLearningMetrics = (timeframe: string = '7d', taskType?: string): { data: LearningMetrics; loading: boolean; error: string | null; refresh: () => void } => {
  const [data, setData] = useState<LearningMetrics>({
    performanceData: [],
    patternStats: [],
    adaptationTrends: [],
    totalExperiences: 0,
    averageLearningRate: 0,
    skillImprovementRate: 0,
    cognitiveLoadTrend: [],
    metaCognitiveAwareness: 0
  });
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      const metrics = await learningApi.getLearningMetrics(timeframe, taskType);
      setData(metrics);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка при загрузке метрик');
      console.error('Ошибка при загрузке метрик обучения:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, [timeframe, taskType]);

 const refresh = () => {
    fetchMetrics();
  };

  return { data, loading, error, refresh };
};
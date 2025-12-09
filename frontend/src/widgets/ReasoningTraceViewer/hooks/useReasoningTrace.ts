import { useState, useCallback } from 'react';
import { ReasoningTrace, ReasoningTraceFilter } from '../types';
import { learningApi } from '@/entities/Learning';

export const useReasoningTrace = (initialTrace?: ReasoningTrace) => {
  const [trace, setTrace] = useState<ReasoningTrace | null>(initialTrace || null);
  const [filteredTrace, setFilteredTrace] = useState<ReasoningTrace | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Фильтрация трассировки
  const filterTrace = useCallback((filter: ReasoningTraceFilter) => {
    if (!trace) return;

    try {
      let filteredSteps = [...trace.steps];

      // Фильтрация по типам шагов
      if (filter.stepTypes && filter.stepTypes.length > 0) {
        filteredSteps = filteredSteps.filter(step =>
          filter.stepTypes?.includes(step.step_type)
        );
      }

      // Фильтрация по порогу уверенности
      if (filter.confidenceThreshold !== undefined && filter.confidenceThreshold !== null) {
        filteredSteps = filteredSteps.filter(step =>
          step.confidence !== undefined && step.confidence >= (filter.confidenceThreshold || 0)
        );
      }

      // Фильтрация по временному диапазону
      if (filter.timeRange) {
        const [start, end] = filter.timeRange;
        filteredSteps = filteredSteps.filter(step => {
          const stepTime = new Date(step.timestamp).getTime();
          return stepTime >= new Date(start).getTime() && stepTime <= new Date(end).getTime();
        });
      }

      // Поиск по запросу
      if (filter.searchQuery) {
        const query = filter.searchQuery.toLowerCase();
        filteredSteps = filteredSteps.filter(step =>
          step.description.toLowerCase().includes(query) ||
          step.step_type.toLowerCase().includes(query) ||
          (step.data && JSON.stringify(step.data).toLowerCase().includes(query))
        );
      }

      // Создание новой трассировки с отфильтрованными шагами
      const newFilteredTrace: ReasoningTrace = {
        ...trace,
        steps: filteredSteps,
        summary: {
          ...trace.summary,
          total_steps: filteredSteps.length,
        }
      };

      setFilteredTrace(newFilteredTrace);
    } catch (err) {
      setError('Ошибка при фильтрации трассировки: ' + (err as Error).message);
    }
  }, [trace]);

  // Загрузка трассировки
  const loadTrace = useCallback(async (traceId: string) => {
    setLoading(true);
    setError(null);

    try {
      const loadedTrace = await learningApi.getReasoningTrace(traceId);
      setTrace(loadedTrace);
      setFilteredTrace(loadedTrace);
    } catch (err) {
      setError('Ошибка при загрузке трассировки: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Обновление трассировки
  const updateTrace = useCallback((newTrace: ReasoningTrace) => {
    setTrace(newTrace);
    setFilteredTrace(newTrace);
  }, []);

  // Сброс фильтров
  const resetFilters = useCallback(() => {
    setFilteredTrace(trace);
  }, [trace]);

  return {
    trace,
    filteredTrace,
    loading,
    error,
    loadTrace,
    updateTrace,
    filterTrace,
    resetFilters
  };
};

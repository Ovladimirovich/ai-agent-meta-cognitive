import React, { useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/shared/ui/card';
import { ReasoningTrace, ReasoningTraceFilter } from './types';
import { useReasoningTrace } from './hooks/useReasoningTrace';
import { convertTraceToGraph } from './utils/graphUtils';
import TraceList from './components/TraceList';
import TraceFilters from './components/TraceFilters';
import TraceGraph from './components/TraceGraph';

interface ReasoningTraceViewerProps {
  traceId?: string;
  initialTrace?: ReasoningTrace;
  className?: string;
}

const ReasoningTraceViewer: React.FC<ReasoningTraceViewerProps> = ({ 
  traceId, 
  initialTrace,
  className = '' 
}) => {
  const {
    trace,
    filteredTrace,
    loading,
    error,
    loadTrace,
    updateTrace,
    filterTrace
  } = useReasoningTrace(initialTrace);

  useEffect(() => {
    if (traceId) {
      loadTrace(traceId);
    } else if (initialTrace) {
      updateTrace(initialTrace);
    }
  }, [traceId, initialTrace, loadTrace, updateTrace]);

  const handleFilterChange = (filter: ReasoningTraceFilter) => {
    if (trace) {
      filterTrace(filter);
    }
  };

  // Получение уникальных типов шагов для фильтра
  const stepTypes = useMemo(() => 
    trace ? Array.from(new Set(trace.steps.map(step => step.step_type))) : [],
    [trace]
  );

  if (loading) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className}`}>
        <div className="flex justify-center items-center h-64">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-cognitive-meta-500 mb-2"></div>
            <p className="text-gray-600 dark:text-gray-300">Загрузка трассировки рассуждений...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className}`}>
        <div className="text-center">
          <p className="text-red-600 dark:text-red-40 mb-2">Ошибка загрузки трассировки</p>
          <p className="text-gray-60 dark:text-gray-300">{error}</p>
        </div>
      </div>
    );
  }

  const currentTrace = filteredTrace || trace;

  if (!currentTrace) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className}`}>
        <div className="text-center">
          <p className="text-gray-600 dark:text-gray-300">Нет данных для отображения</p>
        </div>
      </div>
    );
  }

  // Преобразование трассировки в граф
  const { nodes, links } = convertTraceToGraph(currentTrace.steps);

  return (
    <div className={`space-y-6 ${className}`}>
      <Card>
        <CardHeader>
          <CardTitle>Трассировка рассуждений</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-cognitive-meta-50 dark:bg-cognitive-meta-900 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-cognitive-meta-800 dark:text-cognitive-meta-200">Всего шагов</h3>
                <p className="text-2xl font-bold text-cognitive-meta-600 dark:text-cognitive-meta-100">
                  {currentTrace.summary.total_steps}
                </p>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-blue-800 dark:text-blue-200">Общая уверенность</h3>
                <p className="text-2xl font-bold text-blue-60 dark:text-blue-100">
                  {currentTrace.analysis?.quality_score 
                    ? `${(currentTrace.analysis.quality_score * 100).toFixed(1)}%` 
                    : 'N/A'}
                </p>
              </div>
              <div className="bg-green-50 dark:bg-green-900 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-green-800 dark:text-green-20">Длительность</h3>
                <p className="text-2xl font-bold text-green-60 dark:text-green-100">
                  {currentTrace.summary.duration.toFixed(2)}с
                </p>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-purple-800 dark:text-purple-200">Типы шагов</h3>
                <p className="text-2xl font-bold text-purple-600 dark:text-purple-100">
                  {Object.keys(currentTrace.summary.step_types).length}
                </p>
              </div>
            </div>
          </div>

          <TraceFilters 
            onFilterChange={handleFilterChange} 
            stepTypes={stepTypes} 
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <TraceList 
                steps={currentTrace.steps} 
                className="h-full"
              />
            </div>
            <div>
              <TraceGraph 
                nodes={nodes} 
                links={links} 
                width={600}
                height={500}
              />
            </div>
          </div>

          {currentTrace.analysis && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Анализ рассуждений</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">Выявленные паттерны</h4>
                  <ul className="space-y-2">
                    {currentTrace.analysis.patterns.map((pattern, index) => (
                      <li key={index} className="flex justify-between">
                        <span className="text-gray-700 dark:text-gray-300">{pattern.pattern_type}</span>
                        <span className="text-gray-900 dark:text-white">
                          {pattern.effectiveness ? `${(pattern.effectiveness * 100).toFixed(1)}%` : 'N/A'}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">Проблемы</h4>
                  <ul className="space-y-2">
                    {currentTrace.analysis.issues.map((issue, index) => (
                      <li key={index} className="text-gray-700 dark:text-gray-300">
                        {issue}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default ReasoningTraceViewer;
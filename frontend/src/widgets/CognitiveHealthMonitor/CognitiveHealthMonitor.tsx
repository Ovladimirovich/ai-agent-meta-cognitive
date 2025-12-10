import React, { useState, useEffect, useCallback } from 'react';
import { useRealtimeData } from '../../widgets/RealtimeDataProvider/RealtimeDataProvider';
import { apiClient } from '@/shared/lib/apiClient';
import { useQuery } from '@tanstack/react-query';
import { CognitiveHealthData } from '@/shared/types/api';

interface CognitiveMetric {
  name: string;
  value: number;
  max: number;
  description: string;
  status: 'excellent' | 'good' | 'average' | 'poor';
}

interface CognitiveHealthMonitorProps {
  initialData?: CognitiveHealthData;
  className?: string;
  refreshInterval?: number;
}

const CognitiveHealthMonitor: React.FC<CognitiveHealthMonitorProps> = ({
  initialData,
  className = '',
  refreshInterval = 5000
}) => {
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const { cognitiveData } = useRealtimeData();

  // Используем React Query для получения данных о здоровье
  const { data: healthStatus, isLoading } = useQuery({
    queryKey: ['healthStatus'],
    queryFn: () => apiClient.getHealthStatus(),
    refetchInterval: refreshInterval,
    staleTime: refreshInterval - 1000,
  });

  // Обновление времени последнего обновления
  useEffect(() => {
    setLastUpdated(new Date());
  }, [healthStatus]);

  // Функция для получения статуса метрики
  const getMetricStatus = useCallback((value: number): CognitiveMetric['status'] => {
    if (value >= 0.8) return 'excellent';
    if (value >= 0.6) return 'good';
    if (value >= 0.4) return 'average';
    return 'poor';
  }, []);

  // Функция для получения цвета метрики
  const getMetricColor = useCallback((status: CognitiveMetric['status']) => {
    switch (status) {
      case 'excellent': return 'bg-green-500';
      case 'good': return 'bg-blue-500';
      case 'average': return 'bg-yellow-500';
      case 'poor': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  }, []);

  // Функция для получения метки метрики
  const getMetricLabel = useCallback((status: CognitiveMetric['status']) => {
    switch (status) {
      case 'excellent': return 'Отлично';
      case 'good': return 'Хорошо';
      case 'average': return 'Средне';
      case 'poor': return 'Плохо';
      default: return '';
    }
  }, []);

  // Функция для получения общего статуса здоровья
  const getOverallHealthStatus = useCallback((data: CognitiveHealthData) => {
    const avg = (data.cognitiveLoad + data.confidenceLevel + data.processingSpeed +
      data.memoryUtilization + data.attentionSpan + data.decisionAccuracy) / 6;

    if (avg >= 0.8) return { status: 'excellent', label: 'Отличное', color: 'text-green-600', bgColor: 'bg-green-100' };
    if (avg >= 0.6) return { status: 'good', label: 'Хорошее', color: 'text-blue-600', bgColor: 'bg-blue-100' };
    if (avg >= 0.4) return { status: 'average', label: 'Удовлетворительное', color: 'text-yellow-600', bgColor: 'bg-yellow-100' };
    return { status: 'poor', label: 'Требует внимания', color: 'text-red-600', bgColor: 'bg-red-100' };
  }, []);

  // Если когнитивные данные доступны из WebSocket, используем их
  const currentHealthData = cognitiveData ? cognitiveData : (healthStatus as any)?.cognitive_metrics || (healthStatus as any)?.details?.cognitiveMetrics || initialData;

  // Если нет данных, показываем заглушку
  if (isLoading || !currentHealthData) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
          <p className="text-gray-600">Загрузка монитора когнитивного здоровья...</p>
        </div>
      </div>
    );
  }

  const overallHealth = getOverallHealthStatus(currentHealthData);

  // Определяем метрики
  const metrics: CognitiveMetric[] = [
    {
      name: 'Когнитивная нагрузка',
      value: currentHealthData.cognitiveLoad,
      max: 1,
      description: 'Уровень нагрузки на когнитивные ресурсы агента',
      status: getMetricStatus(currentHealthData.cognitiveLoad)
    },
    {
      name: 'Уровень уверенности',
      value: currentHealthData.confidenceLevel,
      max: 1,
      description: 'Средний уровень уверенности в принимаемых решениях',
      status: getMetricStatus(currentHealthData.confidenceLevel)
    },
    {
      name: 'Скорость обработки',
      value: currentHealthData.processingSpeed,
      max: 1,
      description: 'Относительная скорость обработки запросов',
      status: getMetricStatus(currentHealthData.processingSpeed)
    },
    {
      name: 'Использование памяти',
      value: currentHealthData.memoryUtilization,
      max: 1,
      description: 'Процент использования доступных ресурсов памяти',
      status: getMetricStatus(currentHealthData.memoryUtilization)
    },
    {
      name: 'Внимательность',
      value: currentHealthData.attentionSpan,
      max: 1,
      description: 'Способность сохранять фокус на задаче',
      status: getMetricStatus(currentHealthData.attentionSpan)
    },
    {
      name: 'Точность решений',
      value: currentHealthData.decisionAccuracy,
      max: 1,
      description: 'Процент правильных решений в последних задачах',
      status: getMetricStatus(currentHealthData.decisionAccuracy)
    }
  ];

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex flex-col h-full">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800">Мониторинг когнитивного здоровья</h2>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${overallHealth.bgColor} ${overallHealth.color}`}>
            {overallHealth.label}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
          {metrics.map((metric, index) => (
            <div key={index} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-semibold text-gray-800">{metric.name}</h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getMetricColor(metric.status)} text-white`}>
                  {getMetricLabel(metric.status)}
                </span>
              </div>
              <p className="text-sm text-gray-600 mb-3">{metric.description}</p>
              <div className="flex items-center">
                <div className="w-full bg-gray-200 rounded-full h-2.5 mr-3">
                  <div
                    className={`h-2.5 rounded-full ${getMetricColor(metric.status)}`}
                    style={{ width: `${metric.value * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium text-gray-700 w-10">
                  {(metric.value * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-auto pt-4 border-t border-gray-200">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${overallHealth.status === 'excellent' ? 'bg-green-500' : overallHealth.status === 'good' ? 'bg-blue-500' : overallHealth.status === 'average' ? 'bg-yellow-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">Общий статус: {overallHealth.label}</span>
            </div>
            <div className="text-sm text-gray-500">
              Обновлено: {lastUpdated?.toLocaleTimeString('ru-RU')}
            </div>
          </div>

          <div className="mt-4">
            <h3 className="font-medium text-gray-700 mb-2">Рекомендации:</h3>
            <ul className="text-sm text-gray-600 list-disc pl-5 space-y-1">
              {currentHealthData.cognitiveLoad > 0.8 && <li>Рассмотрите снижение когнитивной нагрузки для улучшения производительности</li>}
              {currentHealthData.attentionSpan < 0.5 && <li>Уделите внимание улучшению способности к удержанию внимания</li>}
              {currentHealthData.memoryUtilization > 0.8 && <li>Оптимизируйте использование памяти для предотвращения перегрузки</li>}
              {currentHealthData.decisionAccuracy < 0.7 && <li>Анализируйте причины снижения точности решений</li>}
              {currentHealthData.confidenceLevel < 0.6 && <li>Рассмотрите корректировку метрик уверенности</li>}
              {currentHealthData.processingSpeed < 0.6 && <li>Оптимизируйте алгоритмы обработки для повышения скорости</li>}
              {Object.keys(currentHealthData).length > 0 && !(
                currentHealthData.cognitiveLoad > 0.8 ||
                currentHealthData.attentionSpan < 0.5 ||
                currentHealthData.memoryUtilization > 0.8 ||
                currentHealthData.decisionAccuracy < 0.7 ||
                currentHealthData.confidenceLevel < 0.6 ||
                currentHealthData.processingSpeed < 0.6
              ) && <li>Когнитивное здоровье агента в норме. Продолжайте текущую стратегию.</li>}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CognitiveHealthMonitor;

import React, { useState, useEffect } from 'react';
import { apiClient } from '../../shared/lib/api-client';
import { HealthStatus } from '../../shared/types';

interface SystemHealthMonitorProps {
  pollingInterval?: number;
}

const SystemHealthMonitor: React.FC<SystemHealthMonitorProps> = ({ 
  pollingInterval = 3000 // 30 секунд по умолчанию
}) => {
  const [healthData, setHealthData] = useState<HealthStatus | null>(null);
 const [loading, setLoading] = useState(true);
 const [error, setError] = useState<string | null>(null);

  const fetchHealthStatus = async () => {
    try {
      setLoading(true);
      const data = await apiClient.getHealthStatus();
      setHealthData(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching health status:', err);
      setError('Не удалось получить статус системы');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Загружаем статус при монтировании
    fetchHealthStatus();

    // Устанавливаем интервал для опроса
    const intervalId = setInterval(fetchHealthStatus, pollingInterval);

    // Очищаем интервал при размонтировании
    return () => clearInterval(intervalId);
  }, [pollingInterval]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-state-success';
      case 'degraded':
        return 'text-state-warning';
      case 'unhealthy':
        return 'text-state-error';
      default:
        return 'text-gray-500 dark:text-gray-400';
    }
  };

  const getStatusBg = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-state-success';
      case 'degraded':
        return 'bg-state-warning';
      case 'unhealthy':
        return 'bg-state-error';
      default:
        return 'bg-gray-200 dark:bg-gray-600';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Состояние системы</h2>
        <button 
          onClick={fetchHealthStatus}
          disabled={loading}
          className="text-sm bg-primary-500 hover:bg-primary-60 text-white px-3 py-1 rounded disabled:opacity-50"
        >
          {loading ? 'Обновление...' : 'Обновить'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-state-error/10 text-state-error rounded">
          {error}
        </div>
      )}

      {healthData ? (
        <div className="space-y-3">
          <div className="flex items-center">
            <div className={`w-3 h-3 rounded-full mr-2 ${getStatusBg(healthData.status)}`}></div>
            <span className={`font-medium ${getStatusColor(healthData.status)}`}>
              Статус: {healthData.status === 'healthy' ? 'Здоров' : healthData.status === 'degraded' ? 'Деградированный' : 'Нездоровый'}
            </span>
          </div>
          
          <div className="text-sm text-gray-600 dark:text-gray-300">
            <p>Время отклика: {(healthData.response_time || 0).toFixed(2)}ms</p>
            <p>Последнее обновление: {new Date(healthData.timestamp).toLocaleString()}</p>
          </div>

          <div className="mt-4">
            <h3 className="font-medium text-gray-700 dark:text-gray-200 mb-2">Сервисы:</h3>
            <div className="space-y-2">
              {healthData.services && Object.entries(healthData.services).length > 0 ? (
                Object.entries(healthData.services).map(([serviceName, serviceStatus]) => (
                  <div key={serviceName} className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                    <span className="text-gray-700 dark:text-gray-200">{serviceName}</span>
                    <span className={`px-2 py-1 rounded text-xs ${getStatusBg(serviceStatus)} text-white`}>
                      {serviceStatus}
                    </span>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500 dark:text-gray-400">Нет данных о состоянии сервисов</p>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-4 text-gray-50 dark:text-gray-400">
          {loading ? 'Загрузка статуса...' : 'Нет данных о состоянии системы'}
        </div>
      )}
    </div>
  );
};

export default SystemHealthMonitor;
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/shared/ui/card';
import { PerformanceChart } from './ui/PerformanceChart';
import { PatternStatsChart } from './ui/PatternStatsChart';
import { AdaptationTrendsChart } from './ui/AdaptationTrendsChart';
import { useLearningMetrics } from './hooks/useLearningMetrics';
import { LearningMetrics } from './types';
import { Button } from '@/shared/ui/atoms/Button';
import { CalendarRange, RotateCcw } from 'lucide-react';

interface LearningMetricsDashboardProps {
  initialTimeframe?: string;
 initialTaskType?: string;
}

const LearningMetricsDashboard: React.FC<LearningMetricsDashboardProps> = ({
  initialTimeframe = '7d',
  initialTaskType = 'all'
}) => {
 const [timeframe, setTimeframe] = useState<string>(initialTimeframe);
  const [taskType, setTaskType] = useState<string>(initialTaskType);
  
  const { data, loading, error, refresh } = useLearningMetrics(timeframe, taskType);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-lg">Загрузка метрик обучения...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-red-500">Ошибка загрузки метрик: {error}</div>
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* Панель управления */}
      <div className="flex flex-wrap gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center gap-2">
          <CalendarRange className="w-5 h-5" />
          <span className="font-medium">Временной диапазон:</span>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="border rounded p-2"
          >
            <option value="1d">Последний день</option>
            <option value="7d">Последние 7 дней</option>
            <option value="30d">Последние 30 дней</option>
            <option value="90d">Последние 90 дней</option>
            <option value="all">Все время</option>
          </select>
        </div>
        
        <div className="flex items-center gap-2">
          <span className="font-medium">Тип задачи:</span>
          <select
            value={taskType}
            onChange={(e) => setTaskType(e.target.value)}
            className="border rounded p-2"
          >
            <option value="all">Все типы</option>
            <option value="analytical">Аналитические</option>
            <option value="creative">Творческие</option>
            <option value="logical">Логические</option>
            <option value="research">Исследовательские</option>
          </select>
        </div>
        
        <Button onClick={refresh} variant="outline" className="flex items-center gap-2">
          <RotateCcw className="w-4 h-4" />
          Обновить
        </Button>
      </div>
      
      {/* Основные метрики */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Общий опыт</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.totalExperiences}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Средняя скорость обучения</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.averageLearningRate.toFixed(2)}%</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Улучшение навыков</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.skillImprovementRate.toFixed(2)}%</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Мета-когнитивное осознание</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.metaCognitiveAwareness.toFixed(2)}%</div>
          </CardContent>
        </Card>
      </div>
      
      {/* Графики */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Производительность</CardTitle>
          </CardHeader>
          <CardContent>
            <PerformanceChart data={data.performanceData} />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Статистика паттернов</CardTitle>
          </CardHeader>
          <CardContent>
            <PatternStatsChart data={data.patternStats} />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Тренды адаптации</CardTitle>
          </CardHeader>
          <CardContent>
            <AdaptationTrendsChart data={data.adaptationTrends} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default LearningMetricsDashboard;
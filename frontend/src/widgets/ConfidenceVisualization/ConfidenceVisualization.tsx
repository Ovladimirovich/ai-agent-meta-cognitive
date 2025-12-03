import React from 'react';
import { 
  Radar, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  ResponsiveContainer,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Line,
  LineChart,
  CartesianGrid
} from 'recharts';
import { ConfidenceMetrics } from '../../shared/types';

interface ConfidenceVisualizationProps {
  confidence: number;
  metrics: ConfidenceMetrics;
  showDetails?: boolean;
  variant?: 'radar' | 'gauge' | 'progress' | 'bar';
}

const ConfidenceVisualization: React.FC<ConfidenceVisualizationProps> = ({ 
  confidence, 
  metrics, 
  showDetails = true,
  variant = 'radar'
}) => {
  // Подготовка данных для радара
  const radarData = [
    { subject: 'Общая', A: metrics.overall * 100, fullMark: 100 },
    { subject: 'Рассуждение', A: metrics.reasoning * 100, fullMark: 100 },
    { subject: 'Память', A: metrics.memory * 100, fullMark: 100 },
    { subject: 'Обучение', A: metrics.learning * 100, fullMark: 100 },
    { subject: 'Рефлексия', A: metrics.reflection * 100, fullMark: 100 },
  ];

  // Подготовка данных для гистограммы
  const barData = [
    { name: 'Общая', value: metrics.overall * 100 },
    { name: 'Рассуждение', value: metrics.reasoning * 100 },
    { name: 'Память', value: metrics.memory * 100 },
    { name: 'Обучение', value: metrics.learning * 100 },
    { name: 'Рефлексия', value: metrics.reflection * 100 },
  ];

  const renderGauge = () => {
    const percentage = Math.round(confidence * 100);
    const rotation = (percentage / 10) * 180; // 180 градусов для полукруга
    
    return (
      <div className="flex flex-col items-center">
        <div className="relative w-40 h-20 overflow-hidden">
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gray-200 dark:bg-gray-600 rounded-full"></div>
          <div 
            className="absolute bottom-0 left-0 h-1 bg-cognitive-meta-500 rounded-full origin-left"
            style={{ 
              width: `${percentage}%`,
              transform: `rotate(${rotation}deg)`,
              transformOrigin: 'left center'
            }}
          ></div>
        </div>
        <span className="mt-2 text-lg font-semibold text-gray-900 dark:text-white">
          {percentage}%
        </span>
      </div>
    );
  };

  const renderProgress = () => {
    const percentage = Math.round(confidence * 100);
    
    return (
      <div className="w-full">
        <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-4">
          <div 
            className="bg-cognitive-meta-500 h-4 rounded-full" 
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
        <div className="mt-2 text-right">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
            {percentage}%
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Визуализация уверенности</h2>
      
      <div className="h-64">
        {variant === 'radar' && (
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis angle={30} domain={[0, 100]} />
              <Radar name="Уверенность" dataKey="A" stroke="#ef444" fill="#ef4444" fillOpacity={0.6} />
            </RadarChart>
          </ResponsiveContainer>
        )}
        
        {variant === 'bar' && (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="value" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        )}
        
        {variant === 'gauge' && renderGauge()}
        {variant === 'progress' && renderProgress()}
      </div>
      
      {showDetails && (
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
            <h3 className="font-medium text-gray-700 dark:text-gray-200">Общая уверенность</h3>
            <p className="text-2xl font-bold text-cognitive-meta-500">{(metrics.overall * 100).toFixed(1)}%</p>
          </div>
          <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
            <h3 className="font-medium text-gray-700 dark:text-gray-200">Уровень уверенности</h3>
            <p className="text-2xl font-bold text-cognitive-meta-500">{(confidence * 100).toFixed(1)}%</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConfidenceVisualization;
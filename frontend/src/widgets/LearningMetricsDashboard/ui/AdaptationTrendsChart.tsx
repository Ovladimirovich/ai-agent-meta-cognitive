import React from 'react';
import {
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ComposedChart,
  Line,
  Scatter,
  Brush
} from 'recharts';
import { AdaptationTrend } from '../types';

interface AdaptationTrendsChartProps {
  data: AdaptationTrend[];
}

export const AdaptationTrendsChart: React.FC<AdaptationTrendsChartProps> = ({ data }) => {
  // Определяем максимальное значение для оси Y
  const maxAdaptationLevel = Math.max(...data.map(item => item.adaptationLevel), 100);
  const maxConfidence = Math.max(...data.map(item => item.confidence), 100);
  const maxLearningRate = Math.max(...data.map(item => item.learningRate), 100);

  return (
    <div className="w-full" style={{ minHeight: '400px', height: '400px' }}>
      <div style={{ width: '100%', height: '400px' }}>
        <ComposedChart
          data={data}
          margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 80,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          angle={-45}
          textAnchor="end"
          height={70}
          tick={{ fontSize: 12 }}
        />
        <YAxis
          yAxisId="left"
          domain={[0, Math.max(maxAdaptationLevel, maxConfidence, maxLearningRate)]}
          orientation="left"
          stroke="#8884d8"
          tick={{ fontSize: 12 }}
        />
        <YAxis
          yAxisId="right"
          domain={[0, Math.max(maxAdaptationLevel, maxConfidence, maxLearningRate)]}
          orientation="right"
          stroke="#ff7300"
          tick={{ fontSize: 12 }}
        />
        <Tooltip
          formatter={(value, name) => {
            if (name === 'adaptationLevel') return [`${value}%`, 'Уровень адаптации'];
            if (name === 'confidence') return [`${value}%`, 'Уверенность'];
            if (name === 'learningRate') return [`${value}%`, 'Скорость обучения'];
            if (name === 'taskType') return [value, 'Тип задачи'];
            return [value, name];
          }}
          labelFormatter={(label) => `Дата: ${label}`}
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #ccc',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
          }}
        />
        <Legend wrapperStyle={{ paddingTop: '10px' }}/>
        <Brush dataKey="date" height={30} stroke="#8884d8" />
        <Area
          yAxisId="left"
          type="monotone"
          dataKey="adaptationLevel"
          fill="url(#colorAdaptation)"
          stroke="#8884d8"
          name="Уровень адаптации (%)"
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="confidence"
          stroke="#ff7300"
          name="Уверенность (%)"
          strokeWidth={2}
          dot={{ r: 4 }}
          activeDot={{ r: 6 }}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="learningRate"
          stroke="#387908"
          name="Скорость обучения (%)"
          strokeWidth={2}
          dot={{ r: 4 }}
          activeDot={{ r: 6 }}
        />
        <Scatter
          yAxisId="left"
          dataKey="adaptationLevel"
          fill="#8884d8"
          name="Точки адаптации"
        />
        <defs>
          <linearGradient id="colorAdaptation" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#8884d8" stopOpacity={0.2}/>
            <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
          </linearGradient>
        </defs>
      </ComposedChart>
    </div>
    </div>
  );
};
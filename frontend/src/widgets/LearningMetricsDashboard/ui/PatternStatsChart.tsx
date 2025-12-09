import React from 'react';
import {
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ComposedChart
} from 'recharts';
import { PatternStat } from '../types';

interface PatternStatsChartProps {
  data: PatternStat[];
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export const PatternStatsChart: React.FC<PatternStatsChartProps> = ({ data }) => {
  // Подготовим данные для радарного графика
  const radarData = data.map(item => ({
    patternType: item.patternType,
    successRate: item.successRate,
    avgExecutionTime: item.avgExecutionTime / 100, // Нормализуем для лучшего отображения
    count: item.count
  }));

  return (
    <div className="w-full" style={{ minWidth: '300px', minHeight: '384px' }}>
      <div className="h-96 mb-8" style={{ height: '384px' }}>
        <div style={{ width: '100%', height: '384px' }}>
          <ComposedChart
            data={data}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 80,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="patternType" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 12 }} />
            <YAxis yAxisId="left" orientation="left" stroke="#8884d8" tick={{ fontSize: 12 }} />
            <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" tick={{ fontSize: 12 }} />
            <Tooltip
              formatter={(value, name) => {
                if (name === 'count') return [value, 'Количество'];
                if (name === 'successRate') return [`${value}%`, 'Уровень успеха'];
                if (name === 'avgExecutionTime') return [`${value}ms`, 'Среднее время выполнения'];
                return [value, name];
              }}
              labelFormatter={(label) => `Паттерн: ${label}`}
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #ccc',
                borderRadius: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}
            />
            <Legend wrapperStyle={{ paddingTop: '10px' }} />
            <Bar yAxisId="left" dataKey="count" name="Количество" fill="#8884d8">
              {data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
            <Bar yAxisId="right" dataKey="successRate" name="Уровень успеха (%)" fill="#82ca9d" />
            <Bar yAxisId="right" dataKey="avgExecutionTime" name="Среднее время выполнения (мс)" fill="#ff7300" />
          </ComposedChart>
        </div>
      </div>

      <div className="h-80" style={{ height: '320px' }}>
        <div style={{ width: '100%', height: '320px' }}>
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="patternType" tick={{ fontSize: 12 }} />
            <PolarRadiusAxis tick={{ fontSize: 12 }} />
            <Radar
              name="Уровень успеха"
              dataKey="successRate"
              stroke="#8884d8"
              fill="#8884d8"
              fillOpacity={0.3}
            />
            <Radar
              name="Количество"
              dataKey="count"
              stroke="#82ca9d"
              fill="#82ca9d"
              fillOpacity={0.3}
            />
            <Tooltip
              formatter={(value, name) => {
                if (name === 'successRate') return [`${value}%`, 'Уровень успеха'];
                if (name === 'count') return [value, 'Количество'];
                if (name === 'avgExecutionTime') return [`${Number(value) * 100}ms`, 'Среднее время выполнения'];
                return [value, name];
              }}
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #ccc',
                borderRadius: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}
            />
            <Legend wrapperStyle={{ paddingTop: '10px' }} />
          </RadarChart>
        </div>
      </div>
    </div>
  );
};

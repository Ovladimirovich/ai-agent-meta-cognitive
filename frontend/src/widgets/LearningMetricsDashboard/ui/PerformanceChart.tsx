import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Brush,
  ReferenceLine,
  Area
} from 'recharts';
import { PerformanceDataPoint } from '../types';

interface PerformanceChartProps {
  data: PerformanceDataPoint[];
}

export const PerformanceChart: React.FC<PerformanceChartProps> = ({ data }) => {
  // Определяем максимальное значение для оси Y
  const maxAccuracy = Math.max(...data.map(item => item.accuracy), 100);
  const maxEfficiency = Math.max(...data.map(item => item.efficiency), 100);
  const maxSpeed = Math.max(...data.map(item => item.speed), 10);
  const maxConfidence = Math.max(...data.map(item => item.confidence), 100);
  const maxTaskComplexity = Math.max(...data.map(item => item.task_complexity), 10);

  // Состояния для зума и панорамирования
  const [zoomDomain, setZoomDomain] = useState<[number, number] | null>(null);
  const [, setIsZooming] = useState(false);

  // Обработчики для зума и панорамирования
  const handleZoomStart = () => {
    setIsZooming(true);
  };

  const handleZoomEnd = (domain: { x: [number, number] } | null) => {
    if (domain && domain.x) {
      setZoomDomain(domain.x as [number, number]);
    } else {
      setZoomDomain(null);
    }
    setIsZooming(false);
  };

  // Обработчик для событий мыши (для корректной типизации)
  const handleChartMouseDown = () => {
    handleZoomStart();
  };

  const handleChartMouseUp = () => {
    handleZoomEnd(null);
  };

  return (
    <div className="w-full" style={{ minWidth: '300px', minHeight: '384px', height: '384px' }}>
      <div className="h-96 mb-4" style={{ height: '384px' }}>
        <div style={{ width: '100%', height: '384px' }}>
          <LineChart
            data={data}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 80,
            }}
            syncId="performanceChart"
            onMouseDown={handleChartMouseDown}
            onMouseUp={handleChartMouseUp}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              angle={-45}
              textAnchor="end"
              height={70}
              tick={{ fontSize: 12 }}
              domain={zoomDomain || undefined}
            />
            <YAxis
              yAxisId="left"
              domain={[0, Math.max(maxAccuracy, maxEfficiency, maxSpeed, maxConfidence)]}
              orientation="left"
              stroke="#8884d8"
              tick={{ fontSize: 12 }}
            />
            <YAxis
              yAxisId="right"
              domain={[0, maxTaskComplexity]}
              orientation="right"
              stroke="#387908"
              tick={{ fontSize: 12 }}
            />
            <Tooltip
              formatter={(value, name) => {
                if (name === 'accuracy') return [`${value}%`, 'Точность'];
                if (name === 'efficiency') return [`${value}%`, 'Эффективность'];
                if (name === 'speed') return [`${value}%`, 'Скорость'];
                if (name === 'confidence') return [`${value}%`, 'Уверенность'];
                if (name === 'task_complexity') return [`${value}`, 'Сложность задачи'];
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
            <Legend
              verticalAlign="top"
              height={36}
              wrapperStyle={{ paddingBottom: '10px' }}
            />
            <ReferenceLine y={50} yAxisId="left" stroke="#ccc" strokeDasharray="3 3" label={{ value: 'Средний уровень', position: 'top' }} />
            <Brush dataKey="date" height={30} stroke="#8884d8" />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="accuracy"
              stroke="#8884d8"
              activeDot={{ r: 8 }}
              name="Точность (%)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="efficiency"
              stroke="#82ca9d"
              name="Эффективность (%)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="speed"
              stroke="#ffc658"
              name="Скорость (%)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="confidence"
              stroke="#ff7300"
              name="Уверенность (%)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="task_complexity"
              stroke="#387908"
              name="Сложность задачи"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            {/* Добавляем область для визуализации общей производительности */}
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="accuracy"
              fill="url(#colorAccuracy)"
              fillOpacity={0.1}
              stroke="none"
            />
            <defs>
              <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8884d8" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
              </linearGradient>
            </defs>
          </LineChart>
        </div>
      </div>
    </div>
  );
};

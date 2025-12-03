import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import LearningMetricsDashboard from './LearningMetricsDashboard';
import { LearningMetrics } from './types';
import { PerformanceChart } from './ui/PerformanceChart';
import { PatternStatsChart } from './ui/PatternStatsChart';
import { AdaptationTrendsChart } from './ui/AdaptationTrendsChart';

// Мок данных для тестирования
const mockMetrics: LearningMetrics = {
  performanceData: [
    { date: '2024-01-01', accuracy: 85, efficiency: 78, speed: 90, confidence: 82, task_complexity: 5 },
    { date: '2024-01-02', accuracy: 87, efficiency: 80, speed: 92, confidence: 85, task_complexity: 6 },
  ],
  patternStats: [
    { patternType: 'Sequential', count: 42, successRate: 87, avgExecutionTime: 1200, lastUsed: '2024-01-02' },
    { patternType: 'Parallel', count: 28, successRate: 76, avgExecutionTime: 950, lastUsed: '2024-01-01' },
  ],
  adaptationTrends: [
    { date: '2024-01-01', adaptationLevel: 65, confidence: 70, learningRate: 5.2, taskType: 'analytical' },
    { date: '2024-01-02', adaptationLevel: 68, confidence: 72, learningRate: 5.5, taskType: 'creative' },
  ],
  totalExperiences: 150,
  averageLearningRate: 5.3,
  skillImprovementRate: 12.5,
  cognitiveLoadTrend: [60, 62, 58, 65, 63],
  metaCognitiveAwareness: 78
};

// Мок для Recharts компонентов для тестирования
jest.mock('recharts', () => ({
  ...jest.requireActual('recharts'),
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  AreaChart: ({ children }: { children: React.ReactNode }) => <div data-testid="area-chart">{children}</div>,
  ComposedChart: ({ children }: { children: React.ReactNode }) => <div data-testid="composed-chart">{children}</div>,
  Line: ({ children }: { children: React.ReactNode }) => <div data-testid="line">{children}</div>,
  Bar: ({ children }: { children: React.ReactNode }) => <div data-testid="bar">{children}</div>,
  Area: ({ children }: { children: React.ReactNode }) => <div data-testid="area">{children}</div>,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
}));

// Мок хука useLearningMetrics
jest.mock('./hooks/useLearningMetrics', () => ({
  useLearningMetrics: (timeframe = '7d', taskType = 'all') => ({
    data: mockMetrics,
    loading: false,
    error: null,
    refresh: jest.fn()
  })
}));

describe('LearningMetricsDashboard', () => {
  test('отображает основные метрики', () => {
    render(<LearningMetricsDashboard />);
    
    expect(screen.getByText('Общий опыт')).toBeInTheDocument();
    expect(screen.getByText('150')).toBeInTheDocument();
    
    expect(screen.getByText('Средняя скорость обучения')).toBeInTheDocument();
    expect(screen.getByText('5.30%')).toBeInTheDocument();
    
    expect(screen.getByText('Улучшение навыков')).toBeInTheDocument();
    expect(screen.getByText('12.50%')).toBeInTheDocument();
    
    expect(screen.getByText('Мета-когнитивное осознание')).toBeInTheDocument();
    expect(screen.getByText('78.00%')).toBeInTheDocument();
  });

 test('отображает элементы управления', () => {
    render(<LearningMetricsDashboard />);
    
    expect(screen.getByText('Временной диапазон:')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: /временной диапазон/i })).toBeInTheDocument();
    
    expect(screen.getByText('Тип задачи:')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: /тип задачи/i })).toBeInTheDocument();
    
    expect(screen.getByRole('button', { name: /обновить/i })).toBeInTheDocument();
  });

  test('отображает все три графика', () => {
    render(<LearningMetricsDashboard />);
    
    expect(screen.getByText('Производительность')).toBeInTheDocument();
    expect(screen.getByText('Статистика паттернов')).toBeInTheDocument();
    expect(screen.getByText('Тренды адаптации')).toBeInTheDocument();
  });

  test('проверяет функциональность селектов', () => {
    render(<LearningMetricsDashboard />);
    
    const timeframeSelect = screen.getByRole('combobox', { name: /временной диапазон/i });
    fireEvent.change(timeframeSelect, { target: { value: '30d' } });
    expect(timeframeSelect).toHaveValue('30d');
    
    const taskTypeSelect = screen.getByRole('combobox', { name: /тип задачи/i });
    fireEvent.change(taskTypeSelect, { target: { value: 'analytical' } });
    expect(taskTypeSelect).toHaveValue('analytical');
  });

  test('проверяет вызов функции обновления', () => {
    const mockRefresh = jest.fn();
    
    render(<LearningMetricsDashboard />);
    
    const refreshButton = screen.getByRole('button', { name: /обновить/i });
    fireEvent.click(refreshButton);
    expect(mockRefresh).toHaveBeenCalledTimes(1);
  });
});

describe('PerformanceChart', () => {
  test('рендерит линейный график с данными производительности', () => {
    render(<PerformanceChart data={mockMetrics.performanceData} />);
    
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    expect(screen.getAllByTestId('line')).toHaveLength(5); // accuracy, efficiency, speed, confidence, task_complexity
  });

  test('работает с пустыми данными', () => {
    render(<PerformanceChart data={[]} />);
    
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });
});

describe('PatternStatsChart', () => {
  test('рендерит столбчатый график с данными статистики паттернов', () => {
    render(<PatternStatsChart data={mockMetrics.patternStats} />);
    
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
    expect(screen.getAllByTestId('bar')).toHaveLength(3); // count, successRate, avgExecutionTime
  });

  test('работает с пустыми данными', () => {
    render(<PatternStatsChart data={[]} />);
    
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
  });
});

describe('AdaptationTrendsChart', () => {
  test('рендерит составной график с данными трендов адаптации', () => {
    render(<AdaptationTrendsChart data={mockMetrics.adaptationTrends} />);
    
    expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
    expect(screen.getByTestId('area')).toBeInTheDocument();
    expect(screen.getByTestId('line')).toBeInTheDocument();
  });

  test('работает с пустыми данными', () => {
    render(<AdaptationTrendsChart data={[]} />);
    
    expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
 });
});
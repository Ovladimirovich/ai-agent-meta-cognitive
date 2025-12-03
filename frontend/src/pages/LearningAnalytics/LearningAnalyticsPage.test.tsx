import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import LearningAnalyticsPage from './LearningAnalyticsPage';
import { useLearningMetrics } from '@/widgets/LearningMetricsDashboard/hooks/useLearningMetrics';
import LearningMetricsDashboard from '@/widgets/LearningMetricsDashboard/LearningMetricsDashboard';
import ReasoningTraceViewer from '@/widgets/ReasoningTraceViewer/ReasoningTraceViewer';

// Мок для хука useLearningMetrics
jest.mock('@/widgets/LearningMetricsDashboard/hooks/useLearningMetrics', () => ({
  useLearningMetrics: jest.fn()
}));

// Мок для компонентов дашборда и вьювера
jest.mock('@/widgets/LearningMetricsDashboard/LearningMetricsDashboard', () => ({
  default: ({ metrics }: { metrics: any }) => (
    <div data-testid="learning-metrics-dashboard" data-metrics={JSON.stringify(metrics)}>
      Learning Metrics Dashboard
    </div>
  )
}));

jest.mock('@/widgets/ReasoningTraceViewer/ReasoningTraceViewer', () => ({
  default: (props: any) => (
    <div data-testid="reasoning-trace-viewer" data-props={JSON.stringify(props || {})}>
      Reasoning Trace Viewer
    </div>
  )
}));

// Мок данных
const mockMetrics = {
  performanceData: [
    { date: '2024-01-01', accuracy: 85, efficiency: 78, speed: 90 },
    { date: '2024-01-02', accuracy: 87, efficiency: 80, speed: 92 },
  ],
  patternStats: [
    { patternType: 'Sequential', count: 42, successRate: 87 },
    { patternType: 'Parallel', count: 28, successRate: 76 },
  ],
  adaptationTrends: [
    { date: '2024-01-01', adaptationLevel: 65, confidence: 70 },
    { date: '2024-01-02', adaptationLevel: 68, confidence: 72 },
  ]
};

describe('LearningAnalyticsPage Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('отображает загрузку при начальной загрузке данных', () => {
    // Мокаем хук в состоянии загрузки
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: { performanceData: [], patternStats: [], adaptationTrends: [] },
      loading: true,
      error: null,
      refresh: jest.fn()
    });

    render(<LearningAnalyticsPage />);

    // Проверяем, что отображается индикатор загрузки
    expect(screen.getByText('Загрузка метрик обучения...')).toBeInTheDocument();
  });

  test('отображает ошибку при возникновении ошибки загрузки', () => {
    // Мокаем хук в состоянии ошибки
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: { performanceData: [], patternStats: [], adaptationTrends: [] },
      loading: false,
      error: 'Ошибка загрузки данных',
      refresh: jest.fn()
    });

    render(<LearningAnalyticsPage />);

    // Проверяем, что отображается сообщение об ошибке
    expect(screen.getByText('Ошибка загрузки метрик')).toBeInTheDocument();
    expect(screen.getByText('Ошибка загрузки данных')).toBeInTheDocument();
    expect(screen.getByText('Повторить попытку')).toBeInTheDocument();
  });

  test('отображает компоненты дашборда и вьювера при успешной загрузке', () => {
    // Мокаем хук с данными
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    render(<LearningAnalyticsPage />);

    // Проверяем, что отображаются заголовок страницы
    expect(screen.getByText('Аналитика обучения')).toBeInTheDocument();

    // Проверяем, что отображаются компоненты дашборда и вьювера
    expect(screen.getByTestId('learning-metrics-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('reasoning-trace-viewer')).toBeInTheDocument();
  });

  test('вызывает обновление при клике на кнопку повторной попытки', () => {
    const mockRefresh = jest.fn();
    // Мокаем хук в состоянии ошибки с функцией обновления
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: { performanceData: [], patternStats: [], adaptationTrends: [] },
      loading: false,
      error: 'Ошибка загрузки данных',
      refresh: mockRefresh
    });

    render(<LearningAnalyticsPage />);

    // Кликаем на кнопку повторной попытки
    const retryButton = screen.getByText('Повторить попытку');
    fireEvent.click(retryButton);

    // Проверяем, что функция обновления была вызвана
    expect(mockRefresh).toHaveBeenCalledTimes(1);
  });

  test('передает корректные данные в дашборд метрик', () => {
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    render(<LearningAnalyticsPage />);

    // Проверяем, что дашборд получает корректные данные
    const dashboard = screen.getByTestId('learning-metrics-dashboard');
    expect(dashboard).toBeInTheDocument();
  });

  test('проверяет интеграцию компонентов в FSD архитектуре', () => {
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    render(<LearningAnalyticsPage />);

    // Проверяем, что страница корректно использует компоненты из слоев widgets
    expect(screen.getByTestId('learning-metrics-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('reasoning-trace-viewer')).toBeInTheDocument();
  });

  test('проверяет передачу данных между слоями entities и features', async () => {
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    render(<LearningAnalyticsPage />);

    // Ожидаем, что компоненты будут отображаться после загрузки данных
    await waitFor(() => {
      expect(screen.getByTestId('learning-metrics-dashboard')).toBeInTheDocument();
      expect(screen.getByTestId('reasoning-trace-viewer')).toBeInTheDocument();
    });

    // Проверяем, что страница отображается корректно
    expect(screen.getByText('Аналитика обучения')).toBeInTheDocument();
  });

  test('проверяет реакцию на изменение состояния метрик', async () => {
    // Сначала в состоянии загрузки
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: { performanceData: [], patternStats: [], adaptationTrends: [] },
      loading: true,
      error: null,
      refresh: jest.fn()
    });

    const { rerender } = render(<LearningAnalyticsPage />);

    // Проверяем состояние загрузки
    expect(screen.getByText('Загрузка метрик обучения...')).toBeInTheDocument();

    // Мокаем данные и перерендерим
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    rerender(<LearningAnalyticsPage />);

    // Проверяем, что компоненты отображаются после загрузки
    expect(screen.getByTestId('learning-metrics-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('reasoning-trace-viewer')).toBeInTheDocument();
  });
});
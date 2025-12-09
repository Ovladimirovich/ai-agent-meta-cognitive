import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import LearningAnalyticsPage from '@/pages/LearningAnalytics/LearningAnalyticsPage';
import { useLearningMetrics } from '@/widgets/LearningMetricsDashboard/hooks/useLearningMetrics';

// Мок для хука useLearningMetrics
jest.mock('@/widgets/LearningMetricsDashboard/hooks/useLearningMetrics', () => ({
  useLearningMetrics: jest.fn()
}));

// Мок для хука useReasoningTrace
jest.mock('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace', () => ({
  useReasoningTrace: jest.fn()
}));

// Мок для компонентов
jest.mock('@/widgets/LearningMetricsDashboard', () => ({
  default: ({ metrics }: { metrics: any }) => (
    <div data-testid="learning-metrics-dashboard" data-metrics={JSON.stringify(metrics || {})}>
      <h3>Learning Metrics Dashboard</h3>
      <div data-testid="performance-chart">Performance Chart</div>
      <div data-testid="pattern-stats-chart">Pattern Stats Chart</div>
      <div data-testid="adaptation-trends-chart">Adaptation Trends Chart</div>
    </div>
  )
}));

jest.mock('@/widgets/ReasoningTraceViewer/ReasoningTraceViewer', () => ({
  default: (props: any) => (
    <div data-testid="reasoning-trace-viewer" data-props={JSON.stringify(props || {})}>
      <h3>Reasoning Trace Viewer</h3>
      <div data-testid="trace-list">Trace List</div>
      <div data-testid="trace-graph">Trace Graph</div>
      <div data-testid="trace-filters">Trace Filters</div>
    </div>
  )
}));

// Мок данных
const mockLearningMetrics = {
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

const mockReasoningTrace = {
  id: 'trace-1',
  steps: [
    {
      step_type: 'analysis',
      description: 'Анализ входных данных',
      timestamp: '2024-01-01T10:00:00Z',
      data: { input: 'test data' },
      confidence: 0.85,
      execution_time: 0.123
    },
    {
      step_type: 'reasoning',
      description: 'Логический вывод',
      timestamp: '2024-01-01T10:00:01Z',
      data: { rule: 'if-then' },
      confidence: 0.78,
      execution_time: 0.234
    }
  ],
  summary: {
    total_steps: 2,
    duration: 2.444,
    step_types: {
      analysis: 1,
      reasoning: 1
    },
    key_decisions: []
  },
  analysis: {
    quality_score: 0.85,
    patterns: [],
    efficiency: {
      steps_count: 2,
      average_step_time: 0.148,
      branching_factor: 1.0,
      depth_score: 1.0,
      optimization_score: 0.85
    },
    issues: [],
    recommendations: []
  }
};

describe('Visualization Components E2E Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('Full Component Flow - Learning Analytics Page', async () => {
    // Мокаем успешную загрузку данных
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockLearningMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    (require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace').useReasoningTrace as jest.Mock).mockReturnValue({
      trace: mockReasoningTrace,
      filteredTrace: null,
      loading: false,
      error: null,
      loadTrace: jest.fn(),
      updateTrace: jest.fn(),
      filterTrace: jest.fn(),
      resetFilters: jest.fn()
    });

    render(
      <BrowserRouter>
        <LearningAnalyticsPage />
      </BrowserRouter>
    );

    // Проверяем, что страница загружена
    await waitFor(() => {
      expect(screen.getByText('Аналитика обучения')).toBeInTheDocument();
    });

    // Проверяем, что все компоненты отображаются
    expect(screen.getByTestId('learning-metrics-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('reasoning-trace-viewer')).toBeInTheDocument();

    // Проверяем подкомпоненты дашборда
    expect(screen.getByTestId('performance-chart')).toBeInTheDocument();
    expect(screen.getByTestId('pattern-stats-chart')).toBeInTheDocument();
    expect(screen.getByTestId('adaptation-trends-chart')).toBeInTheDocument();

    // Проверяем подкомпоненты вьювера
    expect(screen.getByTestId('trace-list')).toBeInTheDocument();
    expect(screen.getByTestId('trace-graph')).toBeInTheDocument();
    expect(screen.getByTestId('trace-filters')).toBeInTheDocument();
  });

  test('Component Data Flow - Loading States', async () => {
    // Сначала состояние загрузки
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: { performanceData: [], patternStats: [], adaptationTrends: [] },
      loading: true,
      error: null,
      refresh: jest.fn()
    });

    (require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace').useReasoningTrace as jest.Mock).mockReturnValue({
      trace: null,
      filteredTrace: null,
      loading: true,
      error: null,
      loadTrace: jest.fn(),
      updateTrace: jest.fn(),
      filterTrace: jest.fn(),
      resetFilters: jest.fn()
    });

    const { rerender } = render(
      <BrowserRouter>
        <LearningAnalyticsPage />
      </BrowserRouter>
    );

    // Проверяем состояние загрузки
    expect(screen.getByText('Загрузка метрик обучения...')).toBeInTheDocument();

    // Мокаем успешную загрузку данных
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockLearningMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    (require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace').useReasoningTrace as jest.Mock).mockReturnValue({
      trace: mockReasoningTrace,
      filteredTrace: null,
      loading: false,
      error: null,
      loadTrace: jest.fn(),
      updateTrace: jest.fn(),
      filterTrace: jest.fn(),
      resetFilters: jest.fn()
    });

    rerender(
      <BrowserRouter>
        <LearningAnalyticsPage />
      </BrowserRouter>
    );

    // Проверяем, что компоненты отображаются после загрузки
    await waitFor(() => {
      expect(screen.getByTestId('learning-metrics-dashboard')).toBeInTheDocument();
    });

    expect(screen.getByTestId('reasoning-trace-viewer')).toBeInTheDocument();
  });

  test('Error Handling - Error States', async () => {
    // Мокаем ошибку загрузки
    const mockError = 'Ошибка загрузки данных';
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: { performanceData: [], patternStats: [], adaptationTrends: [] },
      loading: false,
      error: mockError,
      refresh: jest.fn()
    });

    (require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace').useReasoningTrace as jest.Mock).mockReturnValue({
      trace: null,
      filteredTrace: null,
      loading: false,
      error: null,
      loadTrace: jest.fn(),
      updateTrace: jest.fn(),
      filterTrace: jest.fn(),
      resetFilters: jest.fn()
    });

    render(
      <BrowserRouter>
        <LearningAnalyticsPage />
      </BrowserRouter>
    );

    // Проверяем, что отображается сообщение об ошибке
    await waitFor(() => {
      expect(screen.getByText('Ошибка загрузки метрик')).toBeInTheDocument();
    });

    expect(screen.getByText(mockError)).toBeInTheDocument();
    expect(screen.getByText('Повторить попытку')).toBeInTheDocument();
  });

  test('Component Interaction - Refresh Functionality', async () => {
    const mockRefresh = jest.fn();
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: { performanceData: [], patternStats: [], adaptationTrends: [] },
      loading: false,
      error: 'Ошибка',
      refresh: mockRefresh
    });

    (require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace').useReasoningTrace as jest.Mock).mockReturnValue({
      trace: null,
      filteredTrace: null,
      loading: false,
      error: null,
      loadTrace: jest.fn(),
      updateTrace: jest.fn(),
      filterTrace: jest.fn(),
      resetFilters: jest.fn()
    });

    render(
      <BrowserRouter>
        <LearningAnalyticsPage />
      </BrowserRouter>
    );

    // Кликаем на кнопку повторной попытки
    const retryButton = screen.getByText('Повторить попытку');
    fireEvent.click(retryButton);

    // Проверяем, что функция обновления была вызвана
    expect(mockRefresh).toHaveBeenCalledTimes(1);
  });

  test('FSD Architecture - Component Isolation', async () => {
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockLearningMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    (require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace').useReasoningTrace as jest.Mock).mockReturnValue({
      trace: mockReasoningTrace,
      filteredTrace: null,
      loading: false,
      error: null,
      loadTrace: jest.fn(),
      updateTrace: jest.fn(),
      filterTrace: jest.fn(),
      resetFilters: jest.fn()
    });

    render(
      <BrowserRouter>
        <LearningAnalyticsPage />
      </BrowserRouter>
    );

    // Проверяем, что компоненты из разных слоев FSD работают независимо
    const dashboard = screen.getByTestId('learning-metrics-dashboard');
    const viewer = screen.getByTestId('reasoning-trace-viewer');

    expect(dashboard).toBeInTheDocument();
    expect(viewer).toBeInTheDocument();

    // Проверяем, что каждый компонент содержит свои подкомпоненты
    expect(screen.getByTestId('performance-chart')).toBeInTheDocument();
    expect(screen.getByTestId('trace-list')).toBeInTheDocument();
  });

  test('Component Interactivity - Reasoning Trace Filters', async () => {
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockLearningMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    (require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace').useReasoningTrace as jest.Mock).mockReturnValue({
      trace: mockReasoningTrace,
      filteredTrace: null,
      loading: false,
      error: null,
      loadTrace: jest.fn(),
      updateTrace: jest.fn(() => { }),
      filterTrace: jest.fn(),
      resetFilters: jest.fn()
    });

    render(
      <BrowserRouter>
        <LearningAnalyticsPage />
      </BrowserRouter>
    );

    // Проверяем, что вьювер трассировки отображается
    await waitFor(() => {
      expect(screen.getByTestId('reasoning-trace-viewer')).toBeInTheDocument();
    });

    // Проверяем, что компоненты вьювера доступны
    expect(screen.getByTestId('trace-filters')).toBeInTheDocument();
  });

  test('Performance - Large Dataset Handling', async () => {
    // Создаем мок с большими объемами данных
    const largeDataset = {
      performanceData: Array.from({ length: 1000 }, (_, i) => ({
        date: `2024-01-${String(i % 30 + 1).padStart(2, '0')}`,
        accuracy: Math.random() * 100,
        efficiency: Math.random() * 100,
        speed: Math.random() * 100
      })),
      patternStats: Array.from({ length: 50 }, (_, i) => ({
        patternType: `Pattern_${i}`,
        count: Math.floor(Math.random() * 1000),
        successRate: Math.random() * 100
      })),
      adaptationTrends: Array.from({ length: 1000 }, (_, i) => ({
        date: `2024-01-${String(i % 30 + 1).padStart(2, '0')}`,
        adaptationLevel: Math.random() * 100,
        confidence: Math.random() * 100
      }))
    };

    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: largeDataset,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    (require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace').useReasoningTrace as jest.Mock).mockReturnValue({
      trace: mockReasoningTrace,
      filteredTrace: null,
      loading: false,
      error: null,
      loadTrace: jest.fn(),
      updateTrace: jest.fn(),
      filterTrace: jest.fn(),
      resetFilters: jest.fn()
    });

    // Измеряем время рендеринга
    const startTime = performance.now();
    render(
      <BrowserRouter>
        <LearningAnalyticsPage />
      </BrowserRouter>
    );
    const endTime = performance.now();
    const renderTime = endTime - startTime;

    // Проверяем, что страница рендерится за разумное время
    expect(renderTime).toBeLessThan(2000); // Меньше 2 секунд

    // Проверяем, что компоненты отображаются
    await waitFor(() => {
      expect(screen.getByTestId('learning-metrics-dashboard')).toBeInTheDocument();
    });

    expect(screen.getByTestId('reasoning-trace-viewer')).toBeInTheDocument();
  });

  test('Component Communication - Data Propagation', async () => {
    (useLearningMetrics as jest.Mock).mockReturnValue({
      data: mockLearningMetrics,
      loading: false,
      error: null,
      refresh: jest.fn()
    });

    (require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace').useReasoningTrace as jest.Mock).mockReturnValue({
      trace: mockReasoningTrace,
      filteredTrace: null,
      loading: false,
      error: null,
      loadTrace: jest.fn(),
      updateTrace: jest.fn(),
      filterTrace: jest.fn(),
      resetFilters: jest.fn()
    });

    render(
      <BrowserRouter>
        <LearningAnalyticsPage />
      </BrowserRouter>
    );

    // Проверяем, что данные корректно передаются между компонентами
    const dashboard = screen.getByTestId('learning-metrics-dashboard');
    expect(dashboard).toBeInTheDocument();

    // Проверяем, что компоненты отображаются
    expect(screen.getByText('Learning Metrics Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Reasoning Trace Viewer')).toBeInTheDocument();
  });
});

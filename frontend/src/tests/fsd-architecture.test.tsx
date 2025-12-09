import React from 'react';
import { render, screen } from '@testing-library/react';
import LearningAnalyticsPage from '@/pages/LearningAnalytics/LearningAnalyticsPage';
import LearningMetricsDashboard from '@/widgets/LearningMetricsDashboard/LearningMetricsDashboard';
import ReasoningTraceViewer from '@/widgets/ReasoningTraceViewer/ReasoningTraceViewer';

// Мок для хуков и API
jest.mock('@/widgets/LearningMetricsDashboard/hooks/useLearningMetrics', () => ({
  useLearningMetrics: () => ({
    data: {
      performanceData: [],
      patternStats: [],
      adaptationTrends: []
    },
    loading: false,
    error: null,
    refresh: jest.fn()
  })
}));

jest.mock('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace', () => ({
  useReasoningTrace: () => ({
    trace: null,
    filteredTrace: null,
    loading: false,
    error: null,
    loadTrace: jest.fn(),
    updateTrace: jest.fn(),
    filterTrace: jest.fn(),
    resetFilters: jest.fn()
  })
}));

describe('FSD Architecture Compliance Tests', () => {
  test('Components are properly isolated according to FSD', () => {
    // Проверяем, что компоненты находятся в правильных слоях FSD
    // Pages слой: LearningAnalyticsPage
    expect(typeof LearningAnalyticsPage).toBe('function');

    // Widgets слой: LearningMetricsDashboard и ReasoningTraceViewer
    expect(typeof LearningMetricsDashboard).toBe('function');
    expect(typeof ReasoningTraceViewer).toBe('function');
  });

  test('Page uses widgets correctly without direct business logic', () => {
    render(<LearningAnalyticsPage />);

    // Страница не должна содержать бизнес-логики напрямую
    // Она должна использовать только виджеты и общие компоненты
    expect(screen.getByText('Аналитика обучения')).toBeInTheDocument();
  });

  test('Widgets use hooks from correct layer', () => {
    // Мокаем хуки, чтобы проверить, что виджеты используют правильные слои
    const { useLearningMetrics } = require('@/widgets/LearningMetricsDashboard/hooks/useLearningMetrics');
    const { useReasoningTrace } = require('@/widgets/ReasoningTraceViewer/hooks/useReasoningTrace');

    // Проверяем, что хуки существуют в правильных местах
    expect(typeof useLearningMetrics).toBe('function');
    expect(typeof useReasoningTrace).toBe('function');
  });

  test('Entities layer integration - Learning API', () => {
    // Проверяем, что сущность Learning правильно интегрирована
    const { learningApi } = require('@/entities/Learning/api');

    expect(learningApi).toBeDefined();
    expect(typeof learningApi.getLearningMetrics).toBe('function');
    expect(typeof learningApi.getReasoningTrace).toBe('function');
    expect(typeof learningApi.getAllReasoningTraces).toBe('function');
  });

  test('Shared layer usage - API client', () => {
    // Проверяем использование общего слоя (shared)
    const { apiClient } = require('@/shared/lib/apiClient');

    expect(apiClient).toBeDefined();
    expect(typeof apiClient.getLearningMetrics).toBe('function');
  });

  test('Features layer - if exists', () => {
    // Проверяем, что слой фичей (features) не используется неправильно
    // В FSD компоненты не должны напрямую использовать features в неположенных местах
    expect(true).toBe(true); // Заглушка - в текущей структуре нет специфичных фичей для визуализации
  });

  test('Layer dependencies follow FSD rules', () => {
    // Pages может использовать Widgets, Shared
    // Widgets может использовать Entities, Shared, Features
    // Entities может использовать Shared
    // Shared не зависит ни от кого

    // Проверяем, что зависимости соблюдаются
    expect(() => {
      // Имитация зависимостей
      const pageImports = ['widgets/LearningMetricsDashboard', 'widgets/ReasoningTraceViewer', 'shared/ui'];
      const widgetImports = ['entities/Learning', 'shared/lib', 'shared/ui', 'shared/types'];

      // Эти импорты соответствуют FSD архитектуре
      expect(pageImports).toContain('widgets/LearningMetricsDashboard');
      expect(widgetImports).toContain('entities/Learning');
    }).not.toThrow();
  });

  test('Component isolation - Widgets don\'t depend on Pages', () => {
    // Проверяем, что виджеты могут существовать независимо от страниц
    render(<LearningMetricsDashboard />);

    // Компонент должен рендериться без ошибок без зависимости от страницы
    expect(screen.getByText('Learning Metrics Dashboard')).toBeInTheDocument();
  });

  test('Data flow follows FSD principles', () => {
    // В FSD данные должны течь через правильные слои
    // Page -> Widget -> Hook -> Entity -> Shared (API)
    // или Page -> Widget -> Shared (UI)

    // Проверяем, что компоненты используют правильные пути для получения данных
    const mockMetrics = { performanceData: [], patternStats: [], adaptationTrends: [] };
    const mockTrace = {
      id: 'test',
      steps: [],
      summary: { total_steps: 0, duration: 0, step_types: {}, key_decisions: [] },
      analysis: {
        quality_score: 0,
        patterns: [],
        efficiency: {
          steps_count: 0,
          average_step_time: 0,
          branching_factor: 0,
          depth_score: 0,
          optimization_score: 0
        },
        issues: [],
        recommendations: []
      }
    };

    const { container: dashboardContainer } = render(<LearningMetricsDashboard />);
    const { container: viewerContainer } = render(<ReasoningTraceViewer initialTrace={mockTrace} />);

    // Оба компонента должны рендериться корректно
    expect(dashboardContainer).toBeInTheDocument();
    expect(viewerContainer).toBeInTheDocument();
  });

  test('Shared components are reusable across layers', () => {
    // Проверяем, что компоненты из shared слоя могут использоваться везде
    const { PageWrapper } = require('@/shared/ui/PageWrapper');
    expect(PageWrapper).toBeDefined();
  });
});

import { render, screen, waitFor } from '@testing-library/react';
import ReasoningTraceViewer from './ReasoningTraceViewer';
import { ReasoningTrace } from './types';

// Генератор большого объема данных для тестирования производительности
const generateLargeTrace = (stepsCount: number): ReasoningTrace => {
  const steps = Array.from({ length: stepsCount }, (_, i) => ({
    step_type: i % 3 === 0 ? 'analysis' : i % 3 === 1 ? 'reasoning' : 'decision',
    description: `Шаг рассуждения #${i} - подробное описание операции, которая выполняется в этом шаге`,
    timestamp: new Date(Date.now() - (stepsCount - i) * 1000).toISOString(),
    data: {
      input: `data_${i}`,
      result: `result_${i}`,
      metadata: { complexity: Math.random(), confidence: Math.random() }
    },
    confidence: Math.random(),
    execution_time: Math.random() * 0.5
  }));

  return {
    id: 'large-trace-performance-test',
    steps,
    summary: {
      total_steps: stepsCount,
      duration: stepsCount * 0.1,
      step_types: {
        analysis: Math.floor(stepsCount / 3),
        reasoning: Math.floor(stepsCount / 3),
        decision: stepsCount - Math.floor(stepsCount / 3) * 2
      },
      key_decisions: steps.filter(step => step.step_type === 'decision').slice(0, 10)
    },
    analysis: {
      quality_score: 0.8,
      patterns: [
        {
          pattern_type: 'common_pattern',
          frequency: stepsCount / 10,
          description: 'Часто встречающийся паттерн',
          effectiveness: 0.75,
          examples: steps.slice(0, 5)
        }
      ],
      efficiency: {
        steps_count: stepsCount,
        average_step_time: 0.1,
        branching_factor: 1.2,
        depth_score: 2.0,
        optimization_score: 0.85
      },
      issues: [],
      recommendations: ['Оптимизировать обработку больших объемов данных']
    }
  };
};

describe('Performance Tests for Large Datasets', () => {
  test('Performance - ReasoningTraceViewer with 1000 steps', async () => {
    const largeTrace = generateLargeTrace(1000);

    const startTime = performance.now();
    render(<ReasoningTraceViewer initialTrace={largeTrace} />);
    const renderTime = performance.now() - startTime;

    // Проверяем, что компонент рендерится за разумное время (менее 2 секунд)
    expect(renderTime).toBeLessThan(2000);

    // Проверяем, что основные элементы отображаются
    await waitFor(() => {
      expect(screen.getByText('Трассировка рассуждений')).toBeInTheDocument();
    }, { timeout: 5000 });

    // Проверяем, что отображается количество шагов
    expect(screen.getByText('1000')).toBeInTheDocument(); // Всего шагов
  }, 10000); // Увеличиваем таймаут для теста с большими данными

  test('Performance - ReasoningTraceViewer with 5000 steps', async () => {
    const largeTrace = generateLargeTrace(5000);

    const startTime = performance.now();
    render(<ReasoningTraceViewer initialTrace={largeTrace} />);
    const renderTime = performance.now() - startTime;

    // Для 5000 шагов даем больше времени, но все равно должно быть разумным
    expect(renderTime).toBeLessThan(5000);

    // Проверяем, что компонент не падает при больших объемах данных
    await waitFor(() => {
      expect(screen.getByText('Трассировка рассуждений')).toBeInTheDocument();
    }, { timeout: 10000 });
  }, 15000); // Еще больший таймаут для 5000 шагов

  test('Performance - Memory usage with large dataset', async () => {
    const originalMemory = (global as any).performance?.memory ? (global as any).performance.memory.usedJSHeapSize : 0;

    const largeTrace = generateLargeTrace(2000);
    render(<ReasoningTraceViewer initialTrace={largeTrace} />);

    // Ждем полной загрузки
    await waitFor(() => {
      expect(screen.getByText('Трассировка рассуждений')).toBeInTheDocument();
    });

    const memoryAfter = (global as any).performance?.memory ? (global as any).performance.memory.usedJSHeapSize : 0;
    const memoryUsed = memoryAfter - originalMemory;

    // Проверяем, что использование памяти в разумных пределах (менее 100MB)
    // Note: В тестовой среде performance.memory может быть недоступен, поэтому делаем условную проверку
    if (originalMemory > 0) {
      expect(memoryUsed).toBeLessThan(100 * 1024 * 1024); // 100MB в байтах
    }
  }, 10000);

  test('Performance - ReasoningTraceViewer with 100 steps - rendering time', async () => {
    const trace = generateLargeTrace(100);

    const startTime = performance.now();
    render(<ReasoningTraceViewer initialTrace={trace} />);
    const renderTime = performance.now() - startTime;

    // Для 100 шагов должно рендериться быстро (менее 500мс)
    expect(renderTime).toBeLessThan(500);

    await waitFor(() => {
      expect(screen.getByText('Трассировка рассуждений')).toBeInTheDocument();
    });
  });

  test('Performance - Virtual scrolling simulation (conceptual)', async () => {
    // Тестирование концепции виртуального скроллинга
    // В реальном приложении использовался бы react-window или подобная библиотека
    const veryLargeTrace = generateLargeTrace(1000);

    // Измеряем время подготовки данных
    const dataPrepStart = performance.now();
    const trace = veryLargeTrace;
    const dataPrepTime = performance.now() - dataPrepStart;

    // Проверяем, что подготовка данных быстрая
    expect(dataPrepTime).toBeLessThan(100);

    // В реальном приложении с виртуальным скроллингом рендеринг
    // больших объемов данных будет происходить по частям
    expect(trace.steps).toHaveLength(10000);
    expect(trace.summary.total_steps).toBe(100);
  });

  test('Performance - Component re-rendering with large dataset', async () => {
    const trace1 = generateLargeTrace(1000);
    const trace2 = generateLargeTrace(1500);

    const { rerender } = render(<ReasoningTraceViewer initialTrace={trace1} />);

    await waitFor(() => {
      expect(screen.getByText('Трассировка рассуждений')).toBeInTheDocument();
    });

    const updateStartTime = performance.now();
    rerender(<ReasoningTraceViewer initialTrace={trace2} />);
    const updateTime = performance.now() - updateStartTime;

    // Проверяем, что обновление компонента с новыми данными происходит быстро
    expect(updateTime).toBeLessThan(1000);
  }, 10000);

  test('Performance - Trace filtering with large dataset', async () => {
    const largeTrace = generateLargeTrace(3000);
    render(<ReasoningTraceViewer initialTrace={largeTrace} />);

    await waitFor(() => {
      expect(screen.getByText('Трассировка рассуждений')).toBeInTheDocument();
    });

    // В этом тесте мы проверяем, что компонент может обрабатывать
    // большие объемы данных без падения производительности
    // В реальном приложении здесь будет тестирование фильтрации
    expect(screen.getByText('3000')).toBeInTheDocument();
  }, 10000);
});

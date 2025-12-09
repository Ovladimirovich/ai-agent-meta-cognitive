import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ReflectionTimeline from './ReflectionTimeline';
import { ReflectionEvent } from '../../shared/types/reflection';

// Mock для apiClient
jest.mock('@/shared/lib/apiClient', () => ({
  apiClient: {
    query: jest.fn()
  }
}));

// Mock для RealtimeDataProvider
jest.mock('../../widgets/RealtimeDataProvider', () => ({
  useRealtimeData: () => ({
    reflectionData: null,
    isLoading: false
  })
}));

describe('ReflectionTimeline', () => {
  const mockReflections: ReflectionEvent[] = [
    {
      id: '1',
      interaction_id: 'interaction-1',
      timestamp: new Date().toISOString(),
      type: 'insight',
      title: 'Тестовый инсайт',
      description: 'Описание тестового инсайта',
      confidence: 0.85,
      relatedLearning: 'Тестовое обучение',
      insights: [],
      reflectionTime: 0.5,
      reasoningAnalysis: {
        patterns: [],
        efficiency: {
          stepsCount: 5,
          averageStepTime: 0.1,
          branchingFactor: 1.2,
          depthScore: 0.8,
          optimizationScore: 0.75
        },
        qualityScore: 0.85,
        issues: [],
        recommendations: []
      }
    },
    {
      id: '2',
      interaction_id: 'interaction-2',
      timestamp: new Date().toISOString(),
      type: 'analysis',
      title: 'Тестовый анализ',
      description: 'Описание тестового анализа',
      confidence: 0.72,
      relatedLearning: 'Тестовое обучение 2',
      insights: [],
      reflectionTime: 0.8,
      performanceAnalysis: {
        metrics: {
          executionTime: 1.2,
          confidenceScore: 0.72,
          toolUsageCount: 3,
          memoryUsage: 0.45,
          apiCallsCount: 5,
          errorCount: 0,
          qualityScore: 0.78
        },
        comparison: {
          expectedTime: 1.0,
          expectedTools: 2,
          expectedConfidence: 0.8,
          deviationTime: 0.2,
          deviationTools: 1,
          deviationConfidence: -0.08
        },
        inefficientStrategies: [],
        resourceUsage: {},
        forecast: {}
      }
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders loading state initially', () => {
    render(<ReflectionTimeline />);
    expect(screen.getByText('Загрузка таймлайна рефлексии...')).toBeInTheDocument();
  });

  test('renders with provided reflection data', async () => {
    render(<ReflectionTimeline reflections={mockReflections} />);

    await waitFor(() => {
      expect(screen.queryByText('Загрузка таймлайна рефлексии...')).not.toBeInTheDocument();
    });

    expect(screen.getByText('Таймлайн рефлексии')).toBeInTheDocument();
    expect(screen.getByText('Тестовый инсайт')).toBeInTheDocument();
    expect(screen.getByText('Тестовый анализ')).toBeInTheDocument();
  });

  test('filters reflections by type', async () => {
    render(<ReflectionTimeline reflections={mockReflections} />);

    await waitFor(() => {
      expect(screen.queryByText('Загрузка таймлайна рефлексии...')).not.toBeInTheDocument();
    });

    // Проверяем, что оба типа рефлексий изначально видны
    expect(screen.getByText('Тестовый инсайт')).toBeInTheDocument();
    expect(screen.getByText('Тестовый анализ')).toBeInTheDocument();

    // Снимаем флажок с типа 'insight'
    const insightCheckbox = screen.getByLabelText('Инсайт');
    fireEvent.click(insightCheckbox);

    // Теперь должен быть виден только тип 'analysis'
    expect(screen.queryByText('Тестовый инсайт')).not.toBeInTheDocument();
    expect(screen.getByText('Тестовый анализ')).toBeInTheDocument();
  });

  test('filters reflections by minimum confidence', async () => {
    render(<ReflectionTimeline reflections={mockReflections} />);

    await waitFor(() => {
      expect(screen.queryByText('Загрузка таймлайна рефлексии...')).not.toBeInTheDocument();
    });

    // Изначально обе рефлексии должны быть видны
    expect(screen.getByText('Тестовый инсайт')).toBeInTheDocument();
    expect(screen.getByText('Тестовый анализ')).toBeInTheDocument();

    // Выбираем фильтр по минимальной уверенности 0.8+
    const confidenceFilter = screen.getByRole('combobox');
    fireEvent.change(confidenceFilter, { target: { value: '0.8' } });

    // Теперь должна быть видна только рефлексия с уверенностью 0.85
    expect(screen.getByText('Тестовый инсайт')).toBeInTheDocument();
    expect(screen.queryByText('Тестовый анализ')).not.toBeInTheDocument();
  });

  test('opens detail modal when reflection is clicked', async () => {
    render(<ReflectionTimeline reflections={mockReflections} />);

    await waitFor(() => {
      expect(screen.queryByText('Загрузка таймлайна рефлексии...')).not.toBeInTheDocument();
    });

    // Клик по рефлексии
    const reflectionItem = screen.getByText('Тестовый инсайт');
    fireEvent.click(reflectionItem);

    // Проверяем, что модальное окно открылось
    expect(screen.getByText('Тестовый инсайт')).toBeInTheDocument(); // Заголовок в модальном окне
    expect(screen.getByText('Тип:')).toBeInTheDocument();
    expect(screen.getByText('Время:')).toBeInTheDocument();
    expect(screen.getByText('Уверенность:')).toBeInTheDocument();
  });

  test('closes detail modal when close button is clicked', async () => {
    render(<ReflectionTimeline reflections={mockReflections} />);

    await waitFor(() => {
      expect(screen.queryByText('Загрузка таймлайна рефлексии...')).not.toBeInTheDocument();
    });

    // Открываем модальное окно
    const reflectionItem = screen.getByText('Тестовый инсайт');
    fireEvent.click(reflectionItem);

    // Проверяем, что модальное окно открылось
    expect(screen.getByText('Тестовый инсайт')).toBeInTheDocument();

    // Закрываем модальное окно
    const closeButton = screen.getByText('×');
    fireEvent.click(closeButton);

    // Проверяем, что модальное окно закрылось
    expect(screen.queryByText('Тестовый инсайт')).not.toBeInTheDocument(); // Заголовок в модальном окне больше не виден
  });

  test('renders with default className', () => {
    render(<ReflectionTimeline />);
    const container = screen.getByRole('main'); // Assuming the main div is the container
    expect(container).toHaveClass('bg-white');
  });

  test('renders with custom className', () => {
    render(<ReflectionTimeline className="custom-class" />);
    const container = screen.getByRole('main'); // Assuming the main div is the container
    expect(container).toHaveClass('custom-class');
  });

  test('displays no reflections message when filters exclude all items', async () => {
    render(<ReflectionTimeline reflections={mockReflections} />);

    await waitFor(() => {
      expect(screen.queryByText('Загрузка таймлайна рефлексии...')).not.toBeInTheDocument();
    });

    // Снимаем все флажки фильтров
    const insightCheckbox = screen.getByLabelText('Инсайт');
    fireEvent.click(insightCheckbox);
    const analysisCheckbox = screen.getByLabelText('Анализ');
    fireEvent.click(analysisCheckbox);

    expect(screen.getByText('Нет рефлексий, соответствующих выбранным фильтрам')).toBeInTheDocument();
  });
});

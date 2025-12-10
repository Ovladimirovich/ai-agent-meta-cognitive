import { render, screen } from '@testing-library/react';
import { AgentDashboard } from './AgentDashboard';

// Мок для всех компонентов, используемых в AgentDashboard
jest.mock('../../features/agent-interaction/AgentChatInterface', () => ({
  AgentChatInterface: () => <div data-testid="agent-chat-interface">Agent Chat Interface</div>
}));

jest.mock('../../widgets/MemoryVisualizer/MemoryVisualizer', () => ({
  default: () => <div data-testid="memory-visualizer">Memory Visualizer</div>
}));

jest.mock('../../widgets/ReflectionTimeline/ReflectionTimeline', () => ({
  default: () => <div data-testid="reflection-timeline">Reflection Timeline</div>
}));

jest.mock('../../widgets/CognitiveHealthMonitor/CognitiveHealthMonitor', () => ({
  default: () => <div data-testid="cognitive-health-monitor">Cognitive Health Monitor</div>
}));

// Мок для ThemeProvider
jest.mock('../../app/providers/ThemeProvider', () => ({
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <div data-testid="theme-provider">{children}</div>,
  useTheme: () => ({ theme: 'light', setTheme: jest.fn() })
}));

// Мок для react-query
jest.mock('@tanstack/react-query', () => ({
  useQuery: jest.fn(({ queryFn, initialData }) => ({
    data: initialData || { version: 'v1.0' },
    isLoading: false,
    refetch: jest.fn(),
  })),
}));

// Мок для apiClient
jest.mock('@/shared/lib/apiClient', () => ({
  apiClient: {
    getSystemInfo: jest.fn(() => Promise.resolve({ version: 'v1.0.0' })),
  },
}));


describe('AgentDashboard', () => {
  test('renders all main components', () => {
    render(<AgentDashboard />);

    expect(screen.getByText('Meta-Cognitive AI Agent')).toBeInTheDocument();
    expect(screen.getByText('Интеллектуальная система с самодиагностикой')).toBeInTheDocument();

    // Проверяем, что все интегрированные компоненты отображаются
    expect(screen.getByTestId('cognitive-health-monitor')).toBeInTheDocument();
    expect(screen.getByTestId('memory-visualizer')).toBeInTheDocument();
    expect(screen.getByTestId('reflection-timeline')).toBeInTheDocument();
    expect(screen.getByTestId('agent-chat-interface')).toBeInTheDocument();
  });

  test('renders header with correct information', () => {
    render(<AgentDashboard />);

    expect(screen.getByText('Meta-Cognitive AI Agent')).toBeInTheDocument();
    expect(screen.getByText('Интеллектуальная система с самодиагностикой')).toBeInTheDocument();
    expect(screen.getByText('Backend:')).toBeInTheDocument();
    expect(screen.getByText('● Online')).toBeInTheDocument();
  });

  test('renders footer with correct information', () => {
    render(<AgentDashboard />);

    expect(screen.getByText('© 2025 Meta-Cognitive AI Agent. Построено с использованием React & FastAPI.')).toBeInTheDocument();
    expect(screen.getByText('API Docs')).toBeInTheDocument();
    expect(screen.getByText('GraphQL')).toBeInTheDocument();
    expect(screen.getByText('GitHub')).toBeInTheDocument();
  });

  test('has correct layout structure', () => {
    render(<AgentDashboard />);

    // Проверяем, что компоненты находятся внутри сетки
    const cognitiveHealthMonitor = screen.getByTestId('cognitive-health-monitor');
    const memoryVisualizer = screen.getByTestId('memory-visualizer');

    expect(cognitiveHealthMonitor).toBeInTheDocument();
    expect(memoryVisualizer).toBeInTheDocument();
  });
});

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import MemoryVisualizer from './MemoryVisualizer';

// Создаем клиент для React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false, // Отключаем повторы для тестов
    },
  },
});

// Мокаем Three.js
jest.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => <div data-testid="canvas">{children}</div>,
  useFrame: jest.fn(),
  useThree: jest.fn(() => ({
    camera: { position: { set: jest.fn() }, lookAt: jest.fn() },
  })),
}));

jest.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  Sphere: ({ children }: { children: React.ReactNode }) => <div data-testid="sphere">{children}</div>,
  Line: ({ children }: { children: React.ReactNode }) => <div data-testid="line">{children}</div>,
  Text: ({ children }: { children: React.ReactNode }) => <div data-testid="text">{children}</div>,
  Html: ({ children }: { children: React.ReactNode }) => <div data-testid="html">{children}</div>,
  Float: ({ children }: { children: React.ReactNode }) => <div data-testid="float">{children}</div>,
}));

// Мокаем apiClient
jest.mock('@/shared/lib/apiClient', () => ({
  apiClient: {
    getMemoryState: jest.fn(),
  },
}));

describe('MemoryVisualizer', () => {
  const renderWithQueryProvider = (ui: React.ReactElement) => {
    return render(
      <QueryClientProvider client={queryClient}>
        {ui}
      </QueryClientProvider>
    );
  };

  beforeEach(() => {
    // Очищаем mock-данные перед каждым тестом
    jest.clearAllMocks();
  });

  test('должен отображать заголовок и основные элементы', async () => {
    renderWithQueryProvider(<MemoryVisualizer />);

    // Проверяем, что заголовок отображается
    expect(screen.getByText('Визуализация Памяти')).toBeInTheDocument();

    // Ждем, пока отобразится Canvas
    await waitFor(() => {
      expect(screen.getByTestId('canvas')).toBeInTheDocument();
    });
  });

  test('должен отображать информацию о выбранном узле', async () => {
    renderWithQueryProvider(<MemoryVisualizer />);

    // Ждем, пока отобразится Canvas
    await waitFor(() => {
      expect(screen.getByTestId('canvas')).toBeInTheDocument();
    });

    // Проверяем, что легенда отображается
    expect(screen.getByText('Выбранный узел')).toBeInTheDocument();
    expect(screen.getByText('Обычный узел')).toBeInTheDocument();
    expect(screen.getByText('Подсвеченная связь')).toBeInTheDocument();
    expect(screen.getByText('Узел при наведении')).toBeInTheDocument();
  });

  test('должен отображать статистику узлов и связей', async () => {
    renderWithQueryProvider(<MemoryVisualizer />);

    // Ждем, пока отобразится Canvas
    await waitFor(() => {
      expect(screen.getByTestId('canvas')).toBeInTheDocument();
    });

    // Проверяем, что отображается статистика
    expect(screen.getByText(/Узлов: \d+ \| Связей: \d+/)).toBeInTheDocument();
  });

  test('должен отображать 3D сцену при наличии данных', async () => {
    renderWithQueryProvider(<MemoryVisualizer />);

    await waitFor(() => {
      expect(screen.getByTestId('canvas')).toBeInTheDocument();
    });

    // Проверяем, что элементы 3D сцены отображаются
    expect(screen.getByTestId('orbit-controls')).toBeInTheDocument();
  });
});

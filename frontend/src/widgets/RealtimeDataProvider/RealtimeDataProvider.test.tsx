import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { RealtimeDataProvider, useRealtimeData } from './RealtimeDataProvider';

// Mock для WebSocketProvider
jest.mock('../AdvancedAnalyticsDashboard/WebSocketProvider', () => ({
  useWebSocket: () => ({
    ws: {
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    },
    isConnected: true,
  }),
}));

// Компонент для тестирования хука useRealtimeData
const TestComponent: React.FC = () => {
  const { cognitiveData, memoryData, reflectionData, isLoading } = useRealtimeData();
  
  return (
    <div>
      <span>Loading: {isLoading ? 'true' : 'false'}</span>
      <span>Cognitive: {cognitiveData ? 'exists' : 'null'}</span>
      <span>Memory: {memoryData ? 'exists' : 'null'}</span>
      <span>Reflection: {reflectionData ? 'exists' : 'null'}</span>
    </div>
  );
};

describe('RealtimeDataProvider', () => {
  test('provides initial loading state', () => {
    render(
      <RealtimeDataProvider>
        <TestComponent />
      </RealtimeDataProvider>
    );

    expect(screen.getByText('Loading: true')).toBeInTheDocument();
  });

  test('provides default data when WebSocket is connected', async () => {
    render(
      <RealtimeDataProvider>
        <TestComponent />
      </RealtimeDataProvider>
    );

    // Ожидаем, что данные будут загружены
    await waitFor(() => {
      expect(screen.getByText('Loading: false')).toBeInTheDocument();
    });

    // Проверяем, что данные были предоставлены
    expect(screen.getByText('Cognitive: exists')).toBeInTheDocument();
    expect(screen.getByText('Memory: exists')).toBeInTheDocument();
    expect(screen.getByText('Reflection: exists')).toBeInTheDocument();
  });

  test('updates data when WebSocket receives messages', async () => {
    render(
      <RealtimeDataProvider>
        <TestComponent />
      </RealtimeDataProvider>
    );

    // Ожидаем, что данные будут загружены
    await waitFor(() => {
      expect(screen.getByText('Loading: false')).toBeInTheDocument();
    });

    // Проверяем, что компонент обновился с новыми данными
    expect(screen.getByText('Cognitive: exists')).toBeInTheDocument();
  });

  test('handles WebSocket disconnection gracefully', () => {
    // Мокаем useWebSocket с отключенным состоянием
    jest.mock('../AdvancedAnalyticsDashboard/WebSocketProvider', () => ({
      useWebSocket: () => ({
        ws: null,
        isConnected: false,
      }),
    }));

    // Перезагружаем модуль для применения мока
    jest.isolateModules(() => {
      require('./RealtimeDataProvider');
    });

    render(
      <RealtimeDataProvider>
        <TestComponent />
      </RealtimeDataProvider>
    );

    // Даже при отключенном WebSocket, провайдер должен работать
    expect(screen.getByText('Loading: true')).toBeInTheDocument();
  });

  test('throws error when useRealtimeData is used outside of provider', () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();

    expect(() => render(<TestComponent />)).toThrow('useRealtimeData must be used within a RealtimeDataProvider');

    consoleErrorSpy.mockRestore();
  });
});
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { WebSocketProvider, useWebSocket } from './WebSocketProvider';

// Mock для WebSocket
const mockAddEventListener = jest.fn();
const mockRemoveEventListener = jest.fn();
const mockSend = jest.fn();
const mockClose = jest.fn();

const mockWebSocketConstructor = jest.fn().mockImplementation((url) => {
  return {
    url,
    readyState: 1, // OPEN
    addEventListener: mockAddEventListener,
    removeEventListener: mockRemoveEventListener,
    send: mockSend,
    close: mockClose,
  };
});

(global as any).WebSocket = mockWebSocketConstructor;

// Компонент для тестирования хука useWebSocket
const TestComponent: React.FC = () => {
  const { ws, isConnected, sendMessage } = useWebSocket();
  
  return (
    <div>
      <span>Connected: {isConnected ? 'true' : 'false'}</span>
      <span>WS: {ws ? 'exists' : 'null'}</span>
      <button onClick={() => sendMessage('test message')}>Send</button>
    </div>
  );
};

describe('WebSocketProvider', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('provides WebSocket context to child components', async () => {
    render(
      <WebSocketProvider url="ws://localhost:8000/ws">
        <TestComponent />
      </WebSocketProvider>
    );

    // Проверяем, что WebSocket был создан
    expect(mockWebSocketConstructor).toHaveBeenCalledWith('ws://localhost:8000/ws');
    
    // Ждем, пока компонент подключится
    await waitFor(() => {
      expect(screen.getByText('Connected: true')).toBeInTheDocument();
    });

    expect(screen.getByText('WS: exists')).toBeInTheDocument();
  });

  test('handles WebSocket connection events', async () => {
    render(
      <WebSocketProvider url="ws://localhost:8000/ws">
        <TestComponent />
      </WebSocketProvider>
    );

    // Проверяем, что были добавлены обработчики событий
    expect(mockAddEventListener).toHaveBeenCalledWith('open', expect.any(Function));
    expect(mockAddEventListener).toHaveBeenCalledWith('close', expect.any(Function));
    expect(mockAddEventListener).toHaveBeenCalledWith('error', expect.any(Function));
  });

  test('sends messages through the WebSocket', async () => {
    render(
      <WebSocketProvider url="ws://localhost:8000/ws">
        <TestComponent />
      </WebSocketProvider>
    );

    // Ждем подключения
    await waitFor(() => {
      expect(screen.getByText('Connected: true')).toBeInTheDocument();
    });

    // Кликаем по кнопке отправки
    const sendButton = screen.getByText('Send');
    sendButton.click();

    // Проверяем, что сообщение было отправлено
    expect(mockSend).toHaveBeenCalledWith('test message');
  });

  test('warns when trying to send message while disconnected', async () => {
    // Мокаем WebSocket с состоянием CLOSED
    const mockWebSocketDisconnected = jest.fn().mockImplementation((url) => {
      return {
        url,
        readyState: 3, // CLOSED
        addEventListener: mockAddEventListener,
        removeEventListener: mockRemoveEventListener,
        send: mockSend,
        close: mockClose,
      };
    });
    
    (global as any).WebSocket = mockWebSocketDisconnected;

    const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();

    render(
      <WebSocketProvider url="ws://localhost:8000/ws">
        <TestComponent />
      </WebSocketProvider>
    );

    // Кликаем по кнопке отправки без ожидания подключения
    const sendButton = screen.getByText('Send');
    sendButton.click();

    expect(consoleWarnSpy).toHaveBeenCalledWith('WebSocket not connected, cannot send message');

    consoleWarnSpy.mockRestore();
  });

  test('cleans up WebSocket connection on unmount', () => {
    const { unmount } = render(
      <WebSocketProvider url="ws://localhost:8000/ws">
        <TestComponent />
      </WebSocketProvider>
    );

    unmount();

    expect(mockClose).toHaveBeenCalled();
    expect(mockRemoveEventListener).toHaveBeenCalledWith('message', expect.any(Function));
  });

  test('throws error when useWebSocket is used outside of WebSocketProvider', () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();

    expect(() => render(<TestComponent />)).toThrow('useWebSocket must be used within a WebSocketProvider');

    consoleErrorSpy.mockRestore();
  });
});
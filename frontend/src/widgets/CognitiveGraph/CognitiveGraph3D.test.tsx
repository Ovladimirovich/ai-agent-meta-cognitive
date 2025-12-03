import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import CognitiveGraph3D from './CognitiveGraph3D';

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
}));

// Пример данных для тестирования
const mockCognitiveData = {
  nodes: [
    {
      id: 'node1',
      label: 'Тестовый узел 1',
      x: 0,
      y: 0,
      z: 0,
      type: 'belief' as const,
      confidence: 0.8,
      activation: 0.7,
      connections: ['node2'],
    },
    {
      id: 'node2',
      label: 'Тестовый узел 2',
      x: 1,
      y: 1,
      z: 1,
      type: 'knowledge' as const,
      confidence: 0.6,
      activation: 0.5,
      connections: ['node1'],
    },
  ],
  links: [
    {
      source: 'node1',
      target: 'node2',
      type: 'causal' as const,
      strength: 0.7,
    },
  ],
};

describe('CognitiveGraph3D', () => {
  test('должен отображать заголовок и основные элементы', () => {
    render(<CognitiveGraph3D data={mockCognitiveData} />);
    
    // Проверяем, что заголовок отображается
    expect(screen.getByText('Когнитивный Граф')).toBeInTheDocument();
    
    // Проверяем, что Canvas отображается
    expect(screen.getByTestId('canvas')).toBeInTheDocument();
  });

  test('должен отображать информацию о статистике узлов и связей', () => {
    render(<CognitiveGraph3D data={mockCognitiveData} />);
    
    // Проверяем, что отображается статистика
    expect(screen.getByText('Узлов: 2 | Связей: 1')).toBeInTheDocument();
  });

  test('должен отображать легенду типов узлов', () => {
    render(<CognitiveGraph3D data={mockCognitiveData} />);
    
    // Проверяем, что элементы легенды отображаются
    expect(screen.getByText('Убеждение')).toBeInTheDocument();
    expect(screen.getByText('Знание')).toBeInTheDocument();
    expect(screen.getByText('Опыт')).toBeInTheDocument();
    expect(screen.getByText('Контекст')).toBeInTheDocument();
    expect(screen.getByText('Эмоция')).toBeInTheDocument();
    expect(screen.getByText('Цель')).toBeInTheDocument();
  });

  test('должен отображать легенду типов связей', () => {
    render(<CognitiveGraph3D data={mockCognitiveData} />);
    
    // Проверяем, что элементы легенды для связей отображаются
    expect(screen.getByText('Причинная связь')).toBeInTheDocument();
    expect(screen.getByText('Ассоциативная связь')).toBeInTheDocument();
    expect(screen.getByText('Временная связь')).toBeInTheDocument();
    expect(screen.getByText('Ингибирующая связь')).toBeInTheDocument();
    expect(screen.getByText('Поддерживающая связь')).toBeInTheDocument();
  });

 test('должен отображать 3D сцену с узлами и связями', async () => {
    render(<CognitiveGraph3D data={mockCognitiveData} />);
    
    // Ждем, пока отобразится Canvas
    await waitFor(() => {
      expect(screen.getByTestId('canvas')).toBeInTheDocument();
    });
    
    // Проверяем, что элементы 3D сцены отображаются
    expect(screen.getByTestId('orbit-controls')).toBeInTheDocument();
  });
});
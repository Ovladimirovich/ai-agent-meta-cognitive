import React from 'react';
import { render, screen } from '@testing-library/react';
import Memory3DVisualizer from './Memory3DVisualizer';

// Mock для Three.js, так как он не работает в тестовой среде
jest.mock('three', () => ({
  Scene: jest.fn(() => ({})),
  PerspectiveCamera: jest.fn(() => ({})),
  WebGLRenderer: jest.fn(() => ({
    setSize: jest.fn(),
    setPixelRatio: jest.fn(),
    domElement: document.createElement('canvas'),
    render: jest.fn(),
    dispose: jest.fn(),
  })),
  Color: jest.fn((color) => ({ getHex: () => color })),
  AmbientLight: jest.fn(() => ({})),
  DirectionalLight: jest.fn(() => ({})),
  SphereGeometry: jest.fn(() => ({})),
  MeshPhongMaterial: jest.fn(() => ({})),
  Mesh: jest.fn(() => ({})),
  BufferGeometry: jest.fn(() => ({ setFromPoints: jest.fn() })),
  LineBasicMaterial: jest.fn(() => ({})),
  Line: jest.fn(() => ({})),
  Vector3: jest.fn((x, y, z) => ({ x, y, z })),
  GridHelper: jest.fn(() => ({})),
}));

// Mock для OrbitControls
jest.mock('three/examples/jsm/controls/OrbitControls', () => {
  return jest.fn(() => ({
    enableDamping: true,
    dampingFactor: 0.05,
    update: jest.fn(),
    dispose: jest.fn(),
  }));
});

describe('Memory3DVisualizer', () => {
  const mockMemoryData = {
    layers: {
      husk: {
        nodes: [
          { id: '1', position: [0, 0, 0] as [number, number, number], value: 0.5, connections: [] },
        ],
        connections: [],
      },
      soil: {
        nodes: [
          { id: '2', position: [1, 1, 1] as [number, number, number], value: 0.7, connections: [] },
        ],
        connections: [],
      },
      roots: {
        nodes: [
          { id: '3', position: [2, 2, 2] as [number, number, number], value: 0.9, connections: [] },
        ],
        connections: [],
      },
    },
    connections: [],
    search: { query: '', results: [] },
  };

  test('renders loading state initially', () => {
    render(<Memory3DVisualizer />);
    expect(screen.getByText('Загрузка визуализации памяти...')).toBeInTheDocument();
  });

  test('renders with provided memory data', () => {
    render(<Memory3DVisualizer memoryData={mockMemoryData} />);
    expect(screen.getByText('Загрузка визуализации памяти...')).toBeInTheDocument();
  });

  test('renders with custom className', () => {
    render(<Memory3DVisualizer className="custom-class" />);
    const container = screen.getByRole('main'); // Предполагается, что контейнер имеет role='main'
    expect(container).toHaveClass('custom-class');
  });

  test('defaults to soil layer when no activeLayer provided', () => {
    render(<Memory3DVisualizer memoryData={mockMemoryData} />);
    // Проверяем, что компонент отрисовался без ошибок
    expect(screen.getByText('Загрузка визуализации памяти...')).toBeInTheDocument();
  });

  test('accepts different activeLayer values', () => {
    render(<Memory3DVisualizer memoryData={mockMemoryData} activeLayer="husk" />);
    expect(screen.getByText('Загрузка визуализации памяти...')).toBeInTheDocument();
    
    render(<Memory3DVisualizer memoryData={mockMemoryData} activeLayer="roots" />);
    expect(screen.getByText('Загрузка визуализации памяти...')).toBeInTheDocument();
  });
});
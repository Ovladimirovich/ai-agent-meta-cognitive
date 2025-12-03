import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import CognitiveHealthMonitor from './CognitiveHealthMonitor';

describe('CognitiveHealthMonitor', () => {
  const mockData = {
    cognitiveLoad: 0.65,
    confidenceLevel: 0.82,
    processingSpeed: 0.78,
    memoryUtilization: 0.58,
    attentionSpan: 0.71,
    decisionAccuracy: 0.89
  };

  test('renders loading state initially', () => {
    render(<CognitiveHealthMonitor />);
    expect(screen.getByText('Загрузка монитора когнитивного здоровья...')).toBeInTheDocument();
  });

  test('renders with provided health data', async () => {
    render(<CognitiveHealthMonitor initialData={mockData} />);
    
    // Wait for loading to complete
    await waitFor(() => {
      expect(screen.queryByText('Загрузка монитора когнитивного здоровья...')).not.toBeInTheDocument();
    });

    expect(screen.getByText('Мониторинг когнитивного здоровья')).toBeInTheDocument();
    expect(screen.getByText('Когнитивная нагрузка')).toBeInTheDocument();
    expect(screen.getByText('Уровень уверенности')).toBeInTheDocument();
  });

  test('renders all cognitive metrics', async () => {
    render(<CognitiveHealthMonitor initialData={mockData} />);
    
    await waitFor(() => {
      expect(screen.queryByText('Загрузка монитора когнитивного здоровья...')).not.toBeInTheDocument();
    });

    expect(screen.getByText('Когнитивная нагрузка')).toBeInTheDocument();
    expect(screen.getByText('Уровень уверенности')).toBeInTheDocument();
    expect(screen.getByText('Скорость обработки')).toBeInTheDocument();
    expect(screen.getByText('Использование памяти')).toBeInTheDocument();
    expect(screen.getByText('Внимательность')).toBeInTheDocument();
    expect(screen.getByText('Точность решений')).toBeInTheDocument();
  });

  test('displays correct metric values', async () => {
    render(<CognitiveHealthMonitor initialData={mockData} />);
    
    await waitFor(() => {
      expect(screen.queryByText('Загрузка монитора когнитивного здоровья...')).not.toBeInTheDocument();
    });

    // Check that values are displayed as percentages
    expect(screen.getByText('65%')).toBeInTheDocument();
    expect(screen.getByText('82%')).toBeInTheDocument();
    expect(screen.getByText('78%')).toBeInTheDocument();
    expect(screen.getByText('58%')).toBeInTheDocument();
  });

  test('renders with default className', () => {
    render(<CognitiveHealthMonitor />);
    const container = screen.getByRole('main'); // Assuming the main div is the container
    expect(container).toHaveClass('bg-white');
  });

  test('renders with custom className', () => {
    render(<CognitiveHealthMonitor className="custom-class" />);
    const container = screen.getByRole('main'); // Assuming the main div is the container
    expect(container).toHaveClass('custom-class');
  });

  test('displays overall health status', async () => {
    render(<CognitiveHealthMonitor initialData={mockData} />);
    
    await waitFor(() => {
      expect(screen.queryByText('Загрузка монитора когнитивного здоровья...')).not.toBeInTheDocument();
    });

    // With the provided mock data, the average is around 0.74 which should be "Хорошее"
    expect(screen.getByText('Хорошее')).toBeInTheDocument();
  });

  test('displays recommendations based on health data', async () => {
    render(<CognitiveHealthMonitor initialData={mockData} />);
    
    await waitFor(() => {
      expect(screen.queryByText('Загрузка монитора когнитивного здоровья...')).not.toBeInTheDocument();
    });

    expect(screen.getByText('Рекомендации:')).toBeInTheDocument();
  });
});
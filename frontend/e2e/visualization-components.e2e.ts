import { test, expect } from '@playwright/test';

test.describe('Visualization Components E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Замените на реальный URL вашего приложения
    await page.goto('http://localhost:3000'); // или другой порт
  });

  test('Learning Analytics Page - Full Component Flow', async ({ page }) => {
    // Переход на страницу аналитики обучения
    await page.goto('http://localhost:3000/learning-analytics');
    
    // Проверка заголовка страницы
    await expect(page.locator('h1')).toContainText('Аналитика обучения');
    
    // Проверка наличия дашборда метрик обучения
    const learningMetricsDashboard = page.locator('[data-testid="learning-metrics-dashboard"]');
    await expect(learningMetricsDashboard).toBeVisible();
    
    // Проверка наличия компонентов графиков
    const performanceChart = page.locator('text=Производительность');
    const patternStatsChart = page.locator('text=Статистика паттернов');
    const adaptationTrendsChart = page.locator('text=Тренды адаптации');
    
    await expect(performanceChart).toBeVisible();
    await expect(patternStatsChart).toBeVisible();
    await expect(adaptationTrendsChart).toBeVisible();
    
    // Проверка наличия вьювера цепочки рассуждений
    const reasoningTraceViewer = page.locator('text=Трассировка рассуждений');
    await expect(reasoningTraceViewer).toBeVisible();
  });

  test('LearningMetricsDashboard - Charts Rendering', async ({ page }) => {
    await page.goto('http://localhost:3000/learning-analytics');
    
    // Проверка отображения карточек с метриками
    const metricCards = page.locator('.bg-cognitive-meta-50, .bg-blue-50, .bg-green-50, .bg-purple-50');
    await expect(metricCards).toHaveCount(4); // Всего шагов, общая уверенность, длительность, типы шагов
    
    // Проверка, что графики загружаются
    const chartContainers = page.locator('[data-testid="responsive-container"]');
    await expect(chartContainers).toHaveCount(3); // Три графика в дашборде
    
    // Проверка наличия легенд и подписей на графиках
    await expect(page.locator('text=Точность (%)')).toBeVisible();
    await expect(page.locator('text=Эффективность (%)')).toBeVisible();
    await expect(page.locator('text=Скорость (%)')).toBeVisible();
  });

  test('ReasoningTraceViewer - Trace Visualization', async ({ page }) => {
    await page.goto('http://localhost:3000/learning-analytics');
    
    // Проверка наличия информации о трассировке
    const traceInfo = page.locator('text=Всего шагов');
    await expect(traceInfo).toBeVisible();
    
    // Проверка отображения шагов рассуждений
    const traceSteps = page.locator('text=Цепочка рассуждений');
    await expect(traceSteps).toBeVisible();
    
    // Проверка наличия графа рассуждений
    const graphTitle = page.locator('text=Граф рассуждений');
    await expect(graphTitle).toBeVisible();
    
    // Проверка наличия фильтров
    const filtersTitle = page.locator('text=Фильтры');
    await expect(filtersTitle).toBeVisible();
  });

  test('ReasoningTraceViewer - Interactive Elements', async ({ page }) => {
    await page.goto('http://localhost:3000/learning-analytics');
    
    // Проверка интерактивности фильтров
    const searchInput = page.locator('input[placeholder*="ключевые слова"]');
    await expect(searchInput).toBeVisible();
    
    // Ввод текста в поле поиска
    await searchInput.fill('анализ');
    await expect(searchInput).toHaveValue('анализ');
    
    // Проверка кнопок фильтрации
    const applyFiltersBtn = page.locator('text=Применить фильтры');
    const resetFiltersBtn = page.locator('text=Сбросить');
    await expect(applyFiltersBtn).toBeVisible();
    await expect(resetFiltersBtn).toBeVisible();
  });

  test('Component Data Flow - Loading and Display', async ({ page }) => {
    // Тестируем сценарий загрузки данных
    await page.route('**/learning/metrics', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          total_experiences_processed: 150,
          average_learning_effectiveness: 0.78,
          patterns_discovered: 42,
          skills_improved: 28,
          cognitive_maps_updated: 15,
          adaptation_success_rate: 0.85,
          time_period: '7d'
        })
      });
    });

    await page.goto('http://localhost:3000/learning-analytics');
    
    // Ожидаем, что данные загрузятся и отобразятся
    await page.waitForSelector('[data-testid="learning-metrics-dashboard"]');
    await expect(page.locator('[data-testid="learning-metrics-dashboard"]')).toBeVisible();
  });

  test('Performance - Large Dataset Handling', async ({ page }) => {
    // Создаем мок с большими объемами данных
    await page.route('**/learning/metrics', async (route) => {
      // Генерируем большие объемы данных для тестирования производительности
      const largeDataset = {
        performanceData: Array.from({ length: 1000 }, (_, i) => ({
          date: `2024-01-${String(i % 30 + 1).padStart(2, '0')}`,
          accuracy: Math.random() * 100,
          efficiency: Math.random() * 100,
          speed: Math.random() * 100
        })),
        patternStats: Array.from({ length: 50 }, (_, i) => ({
          patternType: `Pattern_${i}`,
          count: Math.floor(Math.random() * 1000),
          successRate: Math.random() * 100
        })),
        adaptationTrends: Array.from({ length: 1000 }, (_, i) => ({
          date: `2024-01-${String(i % 30 + 1).padStart(2, '0')}`,
          adaptationLevel: Math.random() * 100,
          confidence: Math.random() * 100
        }))
      };
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(largeDataset)
      });
    });

    await page.goto('http://localhost:3000/learning-analytics');
    
    // Проверяем, что страница загружается без ошибок даже с большими объемами данных
    await page.waitForSelector('[data-testid="learning-metrics-dashboard"]', { timeout: 10000 });
    await expect(page.locator('[data-testid="learning-metrics-dashboard"]')).toBeVisible();
    
    // Проверяем, что компоненты не падают при отображении больших объемов данных
    const chartContainers = page.locator('[data-testid="responsive-container"]');
    await expect(chartContainers).toHaveCount(3);
  });

  test('FSD Architecture - Component Isolation', async ({ page }) => {
    // Тестируем, что компоненты корректно изолированы в FSD архитектуре
    await page.goto('http://localhost:3000/learning-analytics');
    
    // Проверяем, что компоненты загружаются независимо
    const dashboardComponent = page.locator('[data-testid="learning-metrics-dashboard"]');
    const traceComponent = page.locator('[data-testid="reasoning-trace-viewer"]');
    
    await expect(dashboardComponent).toBeVisible();
    await expect(traceComponent).toBeVisible();
    
    // Проверяем, что оба компонента функционируют независимо
    await expect(page.locator('text=Аналитика обучения')).toBeVisible();
  });

  test('Responsive Design - Mobile View', async ({ page }) => {
    // Тестируем адаптивность компонентов
    await page.setViewportSize({ width: 375, height: 667 }); // Мобильный размер
    await page.goto('http://localhost:3000/learning-analytics');
    
    // Проверяем, что основные элементы видны на мобильном виде
    await expect(page.locator('text=Аналитика обучения')).toBeVisible();
    await expect(page.locator('text=Трассировка рассуждений')).toBeVisible();
    
    // Проверяем, что компоненты адаптируются к мобильному размеру
    const dashboardCards = page.locator('.grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-3');
    await expect(dashboardCards).toBeVisible();
  });

  test('Error Handling - Network Issues', async ({ page }) => {
    // Мокируем сетевую ошибку
    await page.route('**/learning/metrics', async (route) => {
      await route.abort('network_error');
    });

    await page.goto('http://localhost:3000/learning-analytics');
    
    // Проверяем, что отображается сообщение об ошибке
    await expect(page.locator('text=Ошибка загрузки метрик')).toBeVisible();
    await expect(page.locator('text=Повторить попытку')).toBeVisible();
  });

  test('Navigation - Between Visualization Components', async ({ page }) => {
    // Проверяем навигацию и взаимодействие между компонентами
    await page.goto('http://localhost:3000/learning-analytics');
    
    // Проверяем, что все основные компоненты отображаются
    await expect(page.locator('text=Аналитика обучения')).toBeVisible();
    await expect(page.locator('text=Производительность')).toBeVisible();
    await expect(page.locator('text=Трассировка рассуждений')).toBeVisible();
    
    // Проверяем, что пользователь может взаимодействовать с различными частями страницы
    const performanceSection = page.locator('text=Производительность').first();
    await expect(performanceSection).toBeVisible();
    
    const reasoningSection = page.locator('text=Трассировка рассуждений').first();
    await expect(reasoningSection).toBeVisible();
  });
});
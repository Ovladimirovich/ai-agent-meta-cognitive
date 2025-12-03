# MemoryVisualizer (3D)

Компонент 3D визуализации памяти для AI-агента. Представляет собой интерактивную 3D-сцену, отображающую структуру памяти агента в виде графа с узлами и связями.

## Описание

Компонент использует React Three Fiber и Three.js для создания интерактивной 3D-визуализации памяти агента. Позволяет визуализировать:

- Узлы памяти (контекст, знания, опыт)
- Связи между узлами
- Важность и тип узлов
- Взаимодействие с узлами (выбор, наведение)

## Особенности

- **Интерактивность**: вращение, масштабирование, навигация по сцене
- **Анимации**: плавающие узлы, пульсация при выделении
- **Адаптивность**: поддержка разных размеров экрана
- **Производительность**: оптимизирован для работы с большими графами
- **Информативность**: всплывающие подсказки, легенда, статистика

## Использование

```tsx
import { MemoryVisualizer } from './widgets/MemoryVisualizer';

function App() {
  return (
    <MemoryVisualizer className="h-96 w-full" />
  );
}
```

## Свойства (Props)

| Свойство | Тип | Описание |
|----------|-----|----------|
| `className` | `string` | Дополнительные CSS-классы для контейнера компонента |

## Структура данных

Компонент использует React Query для получения данных о памяти агента. Формат данных:

```ts
interface MemoryNode {
  id: string;
  label: string;
  group: string; // 'context', 'knowledge', 'experience'
  importance?: number;
  color?: string;
  nodeType?: string;
  x?: number;
  y?: number;
  z?: number;
}

interface MemoryLink {
  source: string;
  target: string;
 value?: number;
}

interface MemoryData {
  nodes: MemoryNode[];
  links: MemoryLink[];
}
```

## Архитектура

Компонент построен по архитектуре FSD (Feature-Sliced Design):

- `MemoryVisualizer.tsx` - основной компонент
- `index.ts` - экспорт компонента
- `README.md` - документация
- `MemoryVisualizer.test.tsx` - тесты

## Зависимости

- `@react-three/fiber` - интеграция Three.js с React
- `@react-three/drei` - вспомогательные компоненты и хуки
- `three` - 3D библиотека
- `@tanstack/react-query` - управление состоянием и запросами
- `react` - библиотека для построения пользовательских интерфейсов
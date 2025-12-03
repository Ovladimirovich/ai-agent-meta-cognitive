# CognitiveGraph3D

Компонент 3D визуализации когнитивных графов для AI-агента. Представляет собой интерактивную 3D-сцену, отображающую когнитивную структуру агента в виде графа с различными типами узлов и связей.

## Описание

Компонент использует React Three Fiber и Three.js для создания интерактивной 3D-визуализации когнитивных графов. Позволяет визуализировать:

- Узлы различных типов (убеждения, знания, опыт, контекст, эмоции, цели)
- Связи между узлами различных типов (причинные, ассоциативные, временные и др.)
- Уверенность и активацию узлов
- Взаимодействие с узлами (выбор, наведение)

## Особенности

- **Интерактивность**: вращение, масштабирование, навигация по сцене
- **Анимации**: плавающие узлы, пульсация при активации
- **Адаптивность**: поддержка разных размеров экрана
- **Производительность**: оптимизированная версия для работы с большими графами
- **Информативность**: всплывающие подсказки, легенда, статистика

## Использование

```tsx
import { CognitiveGraph3D } from './widgets/CognitiveGraph';

const cognitiveData = {
 nodes: [
    {
      id: 'node1',
      label: 'Тестовый узел',
      x: 0,
      y: 0,
      z: 0,
      type: 'belief',
      confidence: 0.8,
      activation: 0.7,
      connections: ['node2'],
    },
    // ... другие узлы
 ],
  links: [
    {
      source: 'node1',
      target: 'node2',
      type: 'causal',
      strength: 0.7,
    },
    // ... другие связи
  ],
};

function App() {
  return (
    <CognitiveGraph3D data={cognitiveData} className="h-96 w-full" />
  );
}
```

## Свойства (Props)

| Свойство | Тип | Описание |
|----------|-----|----------|
| `data` | `CognitiveGraphData` | Данные для визуализации когнитивного графа |
| `className` | `string` | Дополнительные CSS-классы для контейнера компонента |
| `onNodeClick` | `(node: CognitiveNode) => void` | Обработчик клика по узлу |
| `onNodeHover` | `(node: CognitiveNode \| null) => void` | Обработчик наведения на узел |
| `maxNodes` | `number` | Максимальное количество узлов для отображения (для оптимизированной версии) |

## Структура данных

```ts
interface CognitiveNode {
  id: string;
  label: string;
  x: number;
  y: number;
  z: number;
  type: 'belief' | 'knowledge' | 'experience' | 'context' | 'emotion' | 'goal';
  confidence: number;
  activation: number;
  connections: string[];
}

interface CognitiveLink {
  source: string;
  target: string;
 type: 'causal' | 'associative' | 'temporal' | 'inhibitory' | 'supportive';
  strength: number;
}

interface CognitiveGraphData {
  nodes: CognitiveNode[];
  links: CognitiveLink[];
}
```

## Типы узлов

- `belief` - убеждения (синий цвет)
- `knowledge` - знания (фиолетовый цвет)
- `experience` - опыт (розовый цвет)
- `context` - контекст (изумрудный цвет)
- `emotion` - эмоции (янтарный цвет)
- `goal` - цели (красный цвет)

## Типы связей

- `causal` - причинные связи (синий цвет)
- `associative` - ассоциативные связи (изумрудный цвет)
- `temporal` - временные связи (фиолетовый цвет)
- `inhibitory` - ингибирующие связи (красный цвет)
- `supportive` - поддерживающие связи (янтарный цвет)

## Архитектура

Компонент построен по архитектуре FSD (Feature-Sliced Design):

- `CognitiveGraph3D.tsx` - основной компонент
- `OptimizedCognitiveGraph3D.tsx` - оптимизированная версия
- `index.ts` - экспорт компонентов
- `README.md` - документация
- `CognitiveGraph3D.test.tsx` - тесты
- `OptimizedCognitiveGraph3D.test.tsx` - тесты оптимизированной версии

## Зависимости

- `@react-three/fiber` - интеграция Three.js с React
- `@react-three/drei` - вспомогательные компоненты и хуки
- `three` - 3D библиотека
- `react` - библиотека для построения пользовательских интерфейсов
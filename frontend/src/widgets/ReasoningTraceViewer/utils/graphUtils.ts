import { ReasoningStep, ReasoningTraceNode, ReasoningTraceLink } from '../types';

export const convertTraceToGraph = (steps: ReasoningStep[]): { nodes: ReasoningTraceNode[], links: ReasoningTraceLink[] } => {
  const nodes: ReasoningTraceNode[] = [];
  const links: ReasoningTraceLink[] = [];

  // Создание узлов
  steps.forEach((step, index) => {
    nodes.push({
      id: `step-${index}`,
      label: step.step_type,
      type: step.step_type,
      confidence: step.confidence,
      timestamp: step.timestamp,
      description: step.description,
      data: step.data
    });
  });

  // Создание связей между узлами (последовательные шаги)
  for (let i = 0; i < steps.length - 1; i++) {
    links.push({
      source: `step-${i}`,
      target: `step-${i + 1}`,
      type: 'sequential'
    });
  }

  // Добавление дополнительных связей на основе данных шагов
  steps.forEach((step, index) => {
    if (step.data && step.data.related_steps) {
      const relatedSteps = Array.isArray(step.data.related_steps) ? step.data.related_steps : [step.data.related_steps];
      
      relatedSteps.forEach((relatedIndex: number) => {
        if (relatedIndex >= 0 && relatedIndex < steps.length && relatedIndex !== index) {
          links.push({
            source: `step-${index}`,
            target: `step-${relatedIndex}`,
            type: 'dependency'
          });
        }
      });
    }
  });

  return { nodes, links };
};

// Функция для расчёта цвета узла на основе уверенности
export const getNodeColor = (confidence: number = 0.5): string => {
  if (confidence === undefined) {
    return '#94a3b8'; // neutral-400
  }

  // Интерполяция цвета от красного (низкая уверенность) к зелёному (высокая уверенность)
  const hue = 120 * confidence; // 0 - красный, 120 - зелёный
  return `hsl(${hue}, 70%, 50%)`;
};

// Функция для расчёта размера узла на основе типа шага
export const getNodeSize = (stepType: string): number => {
  const baseSize = 8;
  const typeMultipliers: Record<string, number> = {
    analysis: 1.2,
    strategy_selection: 1.5,
    execution: 1.0,
    completion: 1.3,
    decision: 1.4
  };

  return baseSize * (typeMultipliers[stepType] || 1.0);
};
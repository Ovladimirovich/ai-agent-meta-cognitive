import React from 'react';
import { ReasoningStep } from '../types';

interface TraceListProps {
  steps: ReasoningStep[];
  onStepClick?: (step: ReasoningStep, index: number) => void;
 className?: string;
}

const TraceList: React.FC<TraceListProps> = ({ 
  steps, 
  onStepClick,
  className = ''
}) => {
  const handleStepClick = (step: ReasoningStep, index: number) => {
    if (onStepClick) {
      onStepClick(step, index);
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow ${className}`}>
      <h3 className="text-lg font-semibold mb-4 p-4 text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700">
        Цепочка рассуждений
      </h3>
      <div className="max-h-96 overflow-y-auto">
        {steps.length === 0 ? (
          <div className="p-4 text-center text-gray-500 dark:text-gray-400">
            Нет данных для отображения
          </div>
        ) : (
          <ul className="divide-y divide-gray-200 dark:divide-gray-700">
            {steps.map((step, index) => (
              <li 
                key={index} 
                className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                onClick={() => handleStepClick(step, index)}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-cognitive-meta-100 text-cognitive-meta-800 dark:bg-cognitive-meta-900 dark:text-cognitive-meta-200">
                        {step.step_type}
                      </span>
                      {step.confidence !== undefined && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                                        bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                          Уверенность: {(step.confidence * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <p className="mt-2 text-sm text-gray-700 dark:text-gray-300 truncate">
                      {step.description}
                    </p>
                    <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                      <span>{new Date(step.timestamp).toLocaleString()}</span>
                      {step.execution_time !== undefined && (
                        <span className="ml-2">• {step.execution_time.toFixed(2)}с</span>
                      )}
                    </div>
                  </div>
                  {step.data && Object.keys(step.data).length > 0 && (
                    <div className="ml-4 text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded">
                      <details>
                        <summary className="cursor-pointer">Дополнительные данные</summary>
                        <pre className="mt-1 text-xs whitespace-pre-wrap break-words">
                          {JSON.stringify(step.data, null, 2)}
                        </pre>
                      </details>
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default TraceList;
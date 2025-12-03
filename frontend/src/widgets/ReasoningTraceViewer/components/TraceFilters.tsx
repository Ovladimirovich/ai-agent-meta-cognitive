import React, { useState } from 'react';
import { ReasoningTraceFilter } from '../types';

interface TraceFiltersProps {
  onFilterChange: (filter: ReasoningTraceFilter) => void;
  stepTypes: string[];
}

const TraceFilters: React.FC<TraceFiltersProps> = ({ 
  onFilterChange, 
  stepTypes 
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStepTypes, setSelectedStepTypes] = useState<string[]>([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number | null>(null);

  const handleApplyFilters = () => {
    const filter: ReasoningTraceFilter = {
      searchQuery: searchQuery || undefined,
      stepTypes: selectedStepTypes.length > 0 ? selectedStepTypes : undefined,
      confidenceThreshold: confidenceThreshold !== null ? confidenceThreshold : undefined
    };
    onFilterChange(filter);
  };

  const handleResetFilters = () => {
    setSearchQuery('');
    setSelectedStepTypes([]);
    setConfidenceThreshold(null);
    onFilterChange({});
  };

  const toggleStepType = (stepType: string) => {
    setSelectedStepTypes(prev => 
      prev.includes(stepType) 
        ? prev.filter(type => type !== stepType) 
        : [...prev, stepType]
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-4">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Фильтры</h3>
      
      <div className="space-y-4">
        {/* Поиск */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Поиск по описанию
          </label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-cognitive-meta-500 focus:border-cognitive-meta-500 dark:bg-gray-700 dark:text-white"
            placeholder="Введите ключевые слова..."
          />
        </div>

        {/* Фильтр по типам шагов */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Типы шагов
          </label>
          <div className="flex flex-wrap gap-2">
            {stepTypes.map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => toggleStepType(type)}
                className={`px-3 py-1 text-sm rounded-full ${
                  selectedStepTypes.includes(type)
                    ? 'bg-cognitive-meta-600 text-white'
                    : 'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Фильтр по уверенности */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Минимальная уверенность: {confidenceThreshold !== null ? `${(confidenceThreshold * 100).toFixed(0)}%` : 'Не установлена'}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={confidenceThreshold !== null ? confidenceThreshold : 0}
            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
            <span>0%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Кнопки управления */}
        <div className="flex space-x-2 pt-2">
          <button
            onClick={handleApplyFilters}
            className="flex-1 px-4 py-2 bg-cognitive-meta-600 text-white rounded-md hover:bg-cognitive-meta-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cognitive-meta-500"
          >
            Применить фильтры
          </button>
          <button
            onClick={handleResetFilters}
            className="px-4 py-2 bg-gray-300 text-gray-700 dark:bg-gray-600 dark:text-gray-200 rounded-md hover:bg-gray-400 dark:hover:bg-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
          >
            Сбросить
          </button>
        </div>
      </div>
    </div>
  );
};

export default TraceFilters;
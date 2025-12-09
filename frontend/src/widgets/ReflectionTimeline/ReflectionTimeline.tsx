import React, { useState, useEffect } from 'react';
import { ReflectionEvent, ReflectionFilter } from '../../shared/types/reflection';
import { apiClient } from '@/shared/lib/apiClient';

interface ReflectionTimelineProps {
  reflections?: ReflectionEvent[];
  className?: string;
}

const ReflectionTimeline: React.FC<ReflectionTimelineProps> = ({
  reflections: initialReflections,
  className = ''
}) => {
  const [reflections, setReflections] = useState<ReflectionEvent[]>([]);
  const [filters, setFilters] = useState<ReflectionFilter>({
    types: ['insight', 'analysis', 'adjustment', 'learning', 'error_pattern', 'strategy_improvement', 'performance_issue', 'feedback_insight'],
    dateRange: { start: null, end: null },
    minConfidence: 0
  });
  const [isLoading, setIsLoading] = useState(true);
  const [selectedReflection, setSelectedReflection] = useState<ReflectionEvent | null>(null);

  // Загружаем данные рефлексии из API
  useEffect(() => {
    const fetchReflectionData = async () => {
      try {
        if (initialReflections) {
          setReflections(initialReflections);
          setIsLoading(false);
        } else {
          // Запрашиваем данные рефлексии через API
          const response = await apiClient.query(`
            query GetReflectionTimeline($first: Int, $filters: ReflectionFiltersInput) {
              reflectionTimeline(first: $first, filters: $filters) {
                nodes {
                  id
                  interactionId
                  timestamp
                  type
                  title
                  description
                  confidence
                  relatedLearning
                  insights {
                    id
                    type
                    description
                    confidence
                    recommendation
                    timestamp
                  }
                  reflectionTime
                  reasoningAnalysis {
                    patterns {
                      patternType
                      frequency
                      description
                      effectiveness
                    }
                    efficiency {
                      stepsCount
                      averageStepTime
                      branchingFactor
                      depthScore
                      optimizationScore
                    }
                    qualityScore
                    issues
                    recommendations
                  }
                  performanceAnalysis {
                    metrics {
                      executionTime
                      confidenceScore
                      toolUsageCount
                      memoryUsage
                      apiCallsCount
                      errorCount
                      qualityScore
                    }
                    comparison {
                      expectedTime
                      expectedTools
                      expectedConfidence
                      deviationTime
                      deviationTools
                      deviationConfidence
                    }
                    inefficientStrategies {
                      name
                      reason
                      alternative
                      confidence
                    }
                    resourceUsage
                    forecast
                  }
                  errorAnalysis {
                    errors {
                      category
                      description
                      message
                      severity
                      context
                      timestamp
                    }
                    patterns {
                      patternId
                      description
                      confidence
                      examples {
                        category
                        description
                        message
                        severity
                        context
                        timestamp
                      }
                      recommendation
                    }
                    metrics
                    severityAssessment
                  }
                }
                totalCount
                pageInfo {
                  hasNextPage
                  hasPreviousPage
                  startCursor
                  endCursor
                }
              }
            }
          `, {
            first: 50,
            filters: {
              types: filters.types,
              dateRange: filters.dateRange,
              minConfidence: filters.minConfidence
            }
          });

          if (response.data?.reflectionTimeline?.nodes) {
            setReflections(response.data.reflectionTimeline.nodes);
          } else {
            // Резервный вариант с mock данными
            const mockReflections: ReflectionEvent[] = [
              {
                id: '1',
                interaction_id: 'interaction-1',
                timestamp: new Date(Date.now() - 360000).toISOString(),
                type: 'insight',
                title: 'Обнаружение паттерна',
                description: 'Агент обнаружил повторяющийся паттерн в запросах пользователя и сформировал гипотезу о его поведении',
                confidence: 0.85,
                relatedLearning: 'Поведенческий анализ',
                insights: [],
                reflectionTime: 0.5,
                reasoningAnalysis: {
                  patterns: [],
                  efficiency: {
                    stepsCount: 5,
                    averageStepTime: 0.1,
                    branchingFactor: 1.2,
                    depthScore: 0.8,
                    optimizationScore: 0.75
                  },
                  qualityScore: 0.85,
                  issues: [],
                  recommendations: []
                }
              },
              {
                id: '2',
                interaction_id: 'interaction-2',
                timestamp: new Date(Date.now() - 1800000).toISOString(),
                type: 'analysis',
                title: 'Анализ эффективности стратегии',
                description: 'Проведен анализ текущей стратегии решения задач. Выявлено снижение эффективности на 15% за последнюю неделю',
                confidence: 0.72,
                relatedLearning: 'Оптимизация стратегии',
                insights: [],
                reflectionTime: 0.8,
                performanceAnalysis: {
                  metrics: {
                    executionTime: 1.2,
                    confidenceScore: 0.72,
                    toolUsageCount: 3,
                    memoryUsage: 0.45,
                    apiCallsCount: 5,
                    errorCount: 0,
                    qualityScore: 0.78
                  },
                  comparison: {
                    expectedTime: 1.0,
                    expectedTools: 2,
                    expectedConfidence: 0.8,
                    deviationTime: 0.2,
                    deviationTools: 1,
                    deviationConfidence: -0.08
                  },
                  inefficientStrategies: [],
                  resourceUsage: {},
                  forecast: {}
                }
              },
              {
                id: '3',
                interaction_id: 'interaction-3',
                timestamp: new Date(Date.now() - 600000).toISOString(),
                type: 'adjustment',
                title: 'Корректировка подхода',
                description: 'На основе анализа выполнена корректировка стратегии решения задач для повышения эффективности',
                confidence: 0.91,
                relatedLearning: 'Адаптивное обучение',
                insights: [],
                reflectionTime: 0.6,
                errorAnalysis: {
                  errors: [],
                  patterns: [],
                  metrics: {},
                  severityAssessment: {}
                }
              },
              {
                id: '4',
                interaction_id: 'interaction-4',
                timestamp: new Date().toISOString(),
                type: 'learning',
                title: 'Интеграция нового знания',
                description: 'Интегрировано новое знание из последнего взаимодействия с пользователем в долгосрочную память',
                confidence: 0.96,
                relatedLearning: 'Пополнение базы знаний',
                insights: [],
                reflectionTime: 0.4
              }
            ];
            setReflections(mockReflections);
          }
        }
      } catch (error) {
        console.error('Error fetching reflection data:', error);
        // В случае ошибки используем mock данные
        const mockReflections: ReflectionEvent[] = [
          {
            id: '1',
            interaction_id: 'interaction-1',
            timestamp: new Date(Date.now() - 360000).toISOString(),
            type: 'insight',
            title: 'Обнаружение паттерна',
            description: 'Агент обнаружил повторяющийся паттерн в запросах пользователя и сформировал гипотезу о его поведении',
            confidence: 0.85,
            relatedLearning: 'Поведенческий анализ',
            insights: [],
            reflectionTime: 0.5,
            reasoningAnalysis: {
              patterns: [],
              efficiency: {
                stepsCount: 5,
                averageStepTime: 0.1,
                branchingFactor: 1.2,
                depthScore: 0.8,
                optimizationScore: 0.75
              },
              qualityScore: 0.85,
              issues: [],
              recommendations: []
            }
          },
          {
            id: '2',
            interaction_id: 'interaction-2',
            timestamp: new Date(Date.now() - 1800000).toISOString(),
            type: 'analysis',
            title: 'Анализ эффективности стратегии',
            description: 'Проведен анализ текущей стратегии решения задач. Выявлено снижение эффективности на 15% за последнюю неделю',
            confidence: 0.72,
            relatedLearning: 'Оптимизация стратегии',
            insights: [],
            reflectionTime: 0.8,
            performanceAnalysis: {
              metrics: {
                executionTime: 1.2,
                confidenceScore: 0.72,
                toolUsageCount: 3,
                memoryUsage: 0.45,
                apiCallsCount: 5,
                errorCount: 0,
                qualityScore: 0.78
              },
              comparison: {
                expectedTime: 1.0,
                expectedTools: 2,
                expectedConfidence: 0.8,
                deviationTime: 0.2,
                deviationTools: 1,
                deviationConfidence: -0.08
              },
              inefficientStrategies: [],
              resourceUsage: {},
              forecast: {}
            }
          },
          {
            id: '3',
            interaction_id: 'interaction-3',
            timestamp: new Date(Date.now() - 600000).toISOString(),
            type: 'adjustment',
            title: 'Корректировка подхода',
            description: 'На основе анализа выполнена корректировка стратегии решения задач для повышения эффективности',
            confidence: 0.91,
            relatedLearning: 'Адаптивное обучение',
            insights: [],
            reflectionTime: 0.6,
            errorAnalysis: {
              errors: [],
              patterns: [],
              metrics: {},
              severityAssessment: {}
            }
          },
          {
            id: '4',
            interaction_id: 'interaction-4',
            timestamp: new Date().toISOString(),
            type: 'learning',
            title: 'Интеграция нового знания',
            description: 'Интегрировано новое знание из последнего взаимодействия с пользователем в долгосрочную память',
            confidence: 0.96,
            relatedLearning: 'Пополнение базы знаний',
            insights: [],
            reflectionTime: 0.4
          }
        ];
        setReflections(mockReflections);
      } finally {
        setIsLoading(false);
      }
    };

    fetchReflectionData();
  }, [initialReflections, filters]);

  // Загружаем данные рефлексии из API
  useEffect(() => {
    const fetchReflectionData = async () => {
      try {
        if (initialReflections) {
          setReflections(initialReflections);
          setIsLoading(false);
        } else {
          // Запрашиваем данные рефлексии через API
          const response = await apiClient.query(`
            query GetReflectionTimeline($first: Int, $filters: ReflectionFiltersInput) {
              reflectionTimeline(first: $first, filters: $filters) {
                nodes {
                  id
                  interactionId
                  timestamp
                  type
                  title
                  description
                  confidence
                  relatedLearning
                  insights {
                    id
                    type
                    description
                    confidence
                    recommendation
                    timestamp
                  }
                  reflectionTime
                  reasoningAnalysis {
                    patterns {
                      patternType
                      frequency
                      description
                      effectiveness
                    }
                    efficiency {
                      stepsCount
                      averageStepTime
                      branchingFactor
                      depthScore
                      optimizationScore
                    }
                    qualityScore
                    issues
                    recommendations
                  }
                  performanceAnalysis {
                    metrics {
                      executionTime
                      confidenceScore
                      toolUsageCount
                      memoryUsage
                      apiCallsCount
                      errorCount
                      qualityScore
                    }
                    comparison {
                      expectedTime
                      expectedTools
                      expectedConfidence
                      deviationTime
                      deviationTools
                      deviationConfidence
                    }
                    inefficientStrategies {
                      name
                      reason
                      alternative
                      confidence
                    }
                    resourceUsage
                    forecast
                  }
                  errorAnalysis {
                    errors {
                      category
                      description
                      message
                      severity
                      context
                      timestamp
                    }
                    patterns {
                      patternId
                      description
                      confidence
                      examples {
                        category
                        description
                        message
                        severity
                        context
                        timestamp
                      }
                      recommendation
                    }
                    metrics
                    severityAssessment
                  }
                }
                totalCount
                pageInfo {
                  hasNextPage
                  hasPreviousPage
                  startCursor
                  endCursor
                }
              }
            }
          `, {
            first: 50,
            filters: {
              types: filters.types,
              dateRange: filters.dateRange,
              minConfidence: filters.minConfidence
            }
          });

          if (response.data?.reflectionTimeline?.nodes) {
            setReflections(response.data.reflectionTimeline.nodes);
          } else {
            // Резервный вариант с mock данными
            const mockReflections: ReflectionEvent[] = [
              {
                id: '1',
                interaction_id: 'interaction-1',
                timestamp: new Date(Date.now() - 360000).toISOString(),
                type: 'insight',
                title: 'Обнаружение паттерна',
                description: 'Агент обнаружил повторяющийся паттерн в запросах пользователя и сформировал гипотезу о его поведении',
                confidence: 0.85,
                relatedLearning: 'Поведенческий анализ',
                insights: [],
                reflectionTime: 0.5,
                reasoningAnalysis: {
                  patterns: [],
                  efficiency: {
                    stepsCount: 5,
                    averageStepTime: 0.1,
                    branchingFactor: 1.2,
                    depthScore: 0.8,
                    optimizationScore: 0.75
                  },
                  qualityScore: 0.85,
                  issues: [],
                  recommendations: []
                }
              },
              {
                id: '2',
                interaction_id: 'interaction-2',
                timestamp: new Date(Date.now() - 1800000).toISOString(),
                type: 'analysis',
                title: 'Анализ эффективности стратегии',
                description: 'Проведен анализ текущей стратегии решения задач. Выявлено снижение эффективности на 15% за последнюю неделю',
                confidence: 0.72,
                relatedLearning: 'Оптимизация стратегии',
                insights: [],
                reflectionTime: 0.8,
                performanceAnalysis: {
                  metrics: {
                    executionTime: 1.2,
                    confidenceScore: 0.72,
                    toolUsageCount: 3,
                    memoryUsage: 0.45,
                    apiCallsCount: 5,
                    errorCount: 0,
                    qualityScore: 0.78
                  },
                  comparison: {
                    expectedTime: 1.0,
                    expectedTools: 2,
                    expectedConfidence: 0.8,
                    deviationTime: 0.2,
                    deviationTools: 1,
                    deviationConfidence: -0.08
                  },
                  inefficientStrategies: [],
                  resourceUsage: {},
                  forecast: {}
                }
              },
              {
                id: '3',
                interaction_id: 'interaction-3',
                timestamp: new Date(Date.now() - 600000).toISOString(),
                type: 'adjustment',
                title: 'Корректировка подхода',
                description: 'На основе анализа выполнена корректировка стратегии решения задач для повышения эффективности',
                confidence: 0.91,
                relatedLearning: 'Адаптивное обучение',
                insights: [],
                reflectionTime: 0.6,
                errorAnalysis: {
                  errors: [],
                  patterns: [],
                  metrics: {},
                  severityAssessment: {}
                }
              },
              {
                id: '4',
                interaction_id: 'interaction-4',
                timestamp: new Date().toISOString(),
                type: 'learning',
                title: 'Интеграция нового знания',
                description: 'Интегрировано новое знание из последнего взаимодействия с пользователем в долгосрочную память',
                confidence: 0.96,
                relatedLearning: 'Пополнение базы знаний',
                insights: [],
                reflectionTime: 0.4
              }
            ];
            setReflections(mockReflections);
          }
        }
      } catch (error) {
        console.error('Error fetching reflection data:', error);
        // В случае ошибки используем mock данные
        const mockReflections: ReflectionEvent[] = [
          {
            id: '1',
            interaction_id: 'interaction-1',
            timestamp: new Date(Date.now() - 360000).toISOString(),
            type: 'insight',
            title: 'Обнаружение паттерна',
            description: 'Агент обнаружил повторяющийся паттерн в запросах пользователя и сформировал гипотезу о его поведении',
            confidence: 0.85,
            relatedLearning: 'Поведенческий анализ',
            insights: [],
            reflectionTime: 0.5,
            reasoningAnalysis: {
              patterns: [],
              efficiency: {
                stepsCount: 5,
                averageStepTime: 0.1,
                branchingFactor: 1.2,
                depthScore: 0.8,
                optimizationScore: 0.75
              },
              qualityScore: 0.85,
              issues: [],
              recommendations: []
            }
          },
          {
            id: '2',
            interaction_id: 'interaction-2',
            timestamp: new Date(Date.now() - 1800000).toISOString(),
            type: 'analysis',
            title: 'Анализ эффективности стратегии',
            description: 'Проведен анализ текущей стратегии решения задач. Выявлено снижение эффективности на 15% за последнюю неделю',
            confidence: 0.72,
            relatedLearning: 'Оптимизация стратегии',
            insights: [],
            reflectionTime: 0.8,
            performanceAnalysis: {
              metrics: {
                executionTime: 1.2,
                confidenceScore: 0.72,
                toolUsageCount: 3,
                memoryUsage: 0.45,
                apiCallsCount: 5,
                errorCount: 0,
                qualityScore: 0.78
              },
              comparison: {
                expectedTime: 1.0,
                expectedTools: 2,
                expectedConfidence: 0.8,
                deviationTime: 0.2,
                deviationTools: 1,
                deviationConfidence: -0.08
              },
              inefficientStrategies: [],
              resourceUsage: {},
              forecast: {}
            }
          },
          {
            id: '3',
            interaction_id: 'interaction-3',
            timestamp: new Date(Date.now() - 600000).toISOString(),
            type: 'adjustment',
            title: 'Корректировка подхода',
            description: 'На основе анализа выполнена корректировка стратегии решения задач для повышения эффективности',
            confidence: 0.91,
            relatedLearning: 'Адаптивное обучение',
            insights: [],
            reflectionTime: 0.6,
            errorAnalysis: {
              errors: [],
              patterns: [],
              metrics: {},
              severityAssessment: {}
            }
          },
          {
            id: '4',
            interaction_id: 'interaction-4',
            timestamp: new Date().toISOString(),
            type: 'learning',
            title: 'Интеграция нового знания',
            description: 'Интегрировано новое знание из последнего взаимодействия с пользователем в долгосрочную память',
            confidence: 0.96,
            relatedLearning: 'Пополнение базы знаний',
            insights: [],
            reflectionTime: 0.4
          }
        ];
        setReflections(mockReflections);
      } finally {
        setIsLoading(false);
      }
    };

    fetchReflectionData();
  }, [initialReflections, filters]);

  // Используем API для получения данных рефлексии
  useEffect(() => {
    const fetchReflectionData = async () => {
      try {
        if (initialReflections) {
          setReflections(initialReflections);
          setIsLoading(false);
        } else {
          // Запрашиваем данные рефлексии через API
          const queryParams = new URLSearchParams();
          if (filters.dateRange.start) queryParams.append('start_date', filters.dateRange.start);
          if (filters.dateRange.end) queryParams.append('end_date', filters.dateRange.end);
          queryParams.append('min_confidence', filters.minConfidence.toString());

          const response = await apiClient.query(`
            query GetReflectionTimeline($first: Int, $filters: ReflectionFilters) {
              reflectionTimeline(first: $first, filters: $filters) {
                nodes {
                  id
                  interaction_id
                  timestamp
                  type
                  title
                  description
                  confidence
                  relatedLearning
                  insights {
                    id
                    type
                    description
                    confidence
                    recommendation
                    timestamp
                  }
                  reflectionTime
                  reasoningAnalysis {
                    patterns {
                      patternType
                      frequency
                      description
                      effectiveness
                    }
                    efficiency {
                      stepsCount
                      averageStepTime
                      branchingFactor
                      depthScore
                      optimizationScore
                    }
                    qualityScore
                    issues
                    recommendations
                  }
                  performanceAnalysis {
                    metrics {
                      executionTime
                      confidenceScore
                      toolUsageCount
                      memoryUsage
                      apiCallsCount
                      errorCount
                      qualityScore
                    }
                    comparison {
                      expectedTime
                      expectedTools
                      expectedConfidence
                      deviationTime
                      deviationTools
                      deviationConfidence
                    }
                    inefficientStrategies {
                      name
                      reason
                      alternative
                      confidence
                    }
                    resourceUsage
                    forecast
                  }
                  errorAnalysis {
                    errors {
                      category
                      description
                      message
                      severity
                      context
                      timestamp
                    }
                    patterns {
                      patternId
                      description
                      confidence
                      examples {
                        category
                        description
                        message
                        severity
                        context
                        timestamp
                      }
                      recommendation
                    }
                    metrics
                    severityAssessment
                  }
                }
                totalCount
                pageInfo {
                  hasNextPage
                  hasPreviousPage
                  startCursor
                  endCursor
                }
              }
            }
          `, {
            first: 50,
            filters: {
              types: filters.types,
              startDate: filters.dateRange.start,
              endDate: filters.dateRange.end,
              minConfidence: filters.minConfidence
            }
          });

          if (response.data?.reflectionTimeline?.nodes) {
            setReflections(response.data.reflectionTimeline.nodes);
          } else {
            // Резервный вариант с mock данными
            const mockReflections: ReflectionEvent[] = [
              {
                id: '1',
                interaction_id: 'interaction-1',
                timestamp: new Date(Date.now() - 360000).toISOString(),
                type: 'insight',
                title: 'Обнаружение паттерна',
                description: 'Агент обнаружил повторяющийся паттерн в запросах пользователя и сформировал гипотезу о его поведении',
                confidence: 0.85,
                relatedLearning: 'Поведенческий анализ',
                insights: [],
                reflectionTime: 0.5,
                reasoningAnalysis: {
                  patterns: [],
                  efficiency: {
                    stepsCount: 5,
                    averageStepTime: 0.1,
                    branchingFactor: 1.2,
                    depthScore: 0.8,
                    optimizationScore: 0.75
                  },
                  qualityScore: 0.85,
                  issues: [],
                  recommendations: []
                }
              },
              {
                id: '2',
                interaction_id: 'interaction-2',
                timestamp: new Date(Date.now() - 1800000).toISOString(),
                type: 'analysis',
                title: 'Анализ эффективности стратегии',
                description: 'Проведен анализ текущей стратегии решения задач. Выявлено снижение эффективности на 15% за последнюю неделю',
                confidence: 0.72,
                relatedLearning: 'Оптимизация стратегии',
                insights: [],
                reflectionTime: 0.8,
                performanceAnalysis: {
                  metrics: {
                    executionTime: 1.2,
                    confidenceScore: 0.72,
                    toolUsageCount: 3,
                    memoryUsage: 0.45,
                    apiCallsCount: 5,
                    errorCount: 0,
                    qualityScore: 0.78
                  },
                  comparison: {
                    expectedTime: 1.0,
                    expectedTools: 2,
                    expectedConfidence: 0.8,
                    deviationTime: 0.2,
                    deviationTools: 1,
                    deviationConfidence: -0.08
                  },
                  inefficientStrategies: [],
                  resourceUsage: {},
                  forecast: {}
                }
              },
              {
                id: '3',
                interaction_id: 'interaction-3',
                timestamp: new Date(Date.now() - 600000).toISOString(),
                type: 'adjustment',
                title: 'Корректировка подхода',
                description: 'На основе анализа выполнена корректировка стратегии решения задач для повышения эффективности',
                confidence: 0.91,
                relatedLearning: 'Адаптивное обучение',
                insights: [],
                reflectionTime: 0.6,
                errorAnalysis: {
                  errors: [],
                  patterns: [],
                  metrics: {},
                  severityAssessment: {}
                }
              },
              {
                id: '4',
                interaction_id: 'interaction-4',
                timestamp: new Date().toISOString(),
                type: 'learning',
                title: 'Интеграция нового знания',
                description: 'Интегрировано новое знание из последнего взаимодействия с пользователем в долгосрочную память',
                confidence: 0.96,
                relatedLearning: 'Пополнение базы знаний',
                insights: [],
                reflectionTime: 0.4
              }
            ];
            setReflections(mockReflections);
          }
        }
      } catch (error) {
        console.error('Error fetching reflection data:', error);
        // В случае ошибки используем mock данные
        const mockReflections: ReflectionEvent[] = [
          {
            id: '1',
            interaction_id: 'interaction-1',
            timestamp: new Date(Date.now() - 360000).toISOString(),
            type: 'insight',
            title: 'Обнаружение паттерна',
            description: 'Агент обнаружил повторяющийся паттерн в запросах пользователя и сформировал гипотезу о его поведении',
            confidence: 0.85,
            relatedLearning: 'Поведенческий анализ',
            insights: [],
            reflectionTime: 0.5,
            reasoningAnalysis: {
              patterns: [],
              efficiency: {
                stepsCount: 5,
                averageStepTime: 0.1,
                branchingFactor: 1.2,
                depthScore: 0.8,
                optimizationScore: 0.75
              },
              qualityScore: 0.85,
              issues: [],
              recommendations: []
            }
          },
          {
            id: '2',
            interaction_id: 'interaction-2',
            timestamp: new Date(Date.now() - 1800000).toISOString(),
            type: 'analysis',
            title: 'Анализ эффективности стратегии',
            description: 'Проведен анализ текущей стратегии решения задач. Выявлено снижение эффективности на 15% за последнюю неделю',
            confidence: 0.72,
            relatedLearning: 'Оптимизация стратегии',
            insights: [],
            reflectionTime: 0.8,
            performanceAnalysis: {
              metrics: {
                executionTime: 1.2,
                confidenceScore: 0.72,
                toolUsageCount: 3,
                memoryUsage: 0.45,
                apiCallsCount: 5,
                errorCount: 0,
                qualityScore: 0.78
              },
              comparison: {
                expectedTime: 1.0,
                expectedTools: 2,
                expectedConfidence: 0.8,
                deviationTime: 0.2,
                deviationTools: 1,
                deviationConfidence: -0.08
              },
              inefficientStrategies: [],
              resourceUsage: {},
              forecast: {}
            }
          },
          {
            id: '3',
            interaction_id: 'interaction-3',
            timestamp: new Date(Date.now() - 600000).toISOString(),
            type: 'adjustment',
            title: 'Корректировка подхода',
            description: 'На основе анализа выполнена корректировка стратегии решения задач для повышения эффективности',
            confidence: 0.91,
            relatedLearning: 'Адаптивное обучение',
            insights: [],
            reflectionTime: 0.6,
            errorAnalysis: {
              errors: [],
              patterns: [],
              metrics: {},
              severityAssessment: {}
            }
          },
          {
            id: '4',
            interaction_id: 'interaction-4',
            timestamp: new Date().toISOString(),
            type: 'learning',
            title: 'Интеграция нового знания',
            description: 'Интегрировано новое знание из последнего взаимодействия с пользователем в долгосрочную память',
            confidence: 0.96,
            relatedLearning: 'Пополнение базы знаний',
            insights: [],
            reflectionTime: 0.4
          }
        ];
        setReflections(mockReflections);
      } finally {
        setIsLoading(false);
      }
    };

    fetchReflectionData();
  }, [initialReflections, filters]);

  const getEventTypeColor = (type: string) => {
    switch (type) {
      case 'insight':
        return 'bg-blue-500';
      case 'analysis':
        return 'bg-purple-500';
      case 'adjustment':
        return 'bg-yellow-500';
      case 'learning':
        return 'bg-green-500';
      case 'error_pattern':
        return 'bg-red-500';
      case 'strategy_improvement':
        return 'bg-indigo-500';
      case 'performance_issue':
        return 'bg-orange-500';
      case 'feedback_insight':
        return 'bg-pink-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getEventTypeLabel = (type: string) => {
    switch (type) {
      case 'insight':
        return 'Инсайт';
      case 'analysis':
        return 'Анализ';
      case 'adjustment':
        return 'Корректировка';
      case 'learning':
        return 'Обучение';
      case 'error_pattern':
        return 'Ошибка';
      case 'strategy_improvement':
        return 'Стратегия';
      case 'performance_issue':
        return 'Производительность';
      case 'feedback_insight':
        return 'Обратная связь';
      default:
        return type;
    }
  };

  const filteredReflections = reflections.filter(reflection => {
    // Фильтр по типу
    if (!filters.types.includes(reflection.type)) {
      return false;
    }

    // Фильтр по минимальной уверенности
    if (reflection.confidence < filters.minConfidence) {
      return false;
    }

    // Фильтр по дате
    if (filters.dateRange.start && new Date(reflection.timestamp) < new Date(filters.dateRange.start)) {
      return false;
    }
    if (filters.dateRange.end && new Date(reflection.timestamp) > new Date(filters.dateRange.end)) {
      return false;
    }

    return true;
  });

  const handleFilterChange = (filterType: keyof ReflectionFilter, value: any) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: value
    }));
  };

  const handleReflectionClick = (reflection: ReflectionEvent) => {
    setSelectedReflection(reflection);
  };

  const handleCloseDetail = () => {
    setSelectedReflection(null);
  };

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
          <p className="text-gray-600">Загрузка таймлайна рефлексии...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex flex-col h-full">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800">Таймлайн рефлексии</h2>
          <div className="flex space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-600">Мин. уверенность:</label>
              <select
                value={filters.minConfidence}
                onChange={(e) => handleFilterChange('minConfidence', parseFloat(e.target.value))}
                className="border border-gray-300 rounded px-2 py-1 text-sm"
              >
                <option value={0}>Все</option>
                <option value={0.5}>0.5+</option>
                <option value={0.7}>0.7+</option>
                <option value={0.9}>0.9+</option>
              </select>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-4 mb-6">
          {(['insight', 'analysis', 'adjustment', 'learning', 'error_pattern', 'strategy_improvement', 'performance_issue', 'feedback_insight'] as const).map(type => (
            <label key={type} className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={filters.types.includes(type)}
                onChange={(e) => {
                  if (e.target.checked) {
                    handleFilterChange('types', [...filters.types, type]);
                  } else {
                    handleFilterChange('types', filters.types.filter(t => t !== type));
                  }
                }}
                className="rounded text-blue-50"
              />
              <span className="text-sm">{getEventTypeLabel(type)}</span>
            </label>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto max-h-[600px] pr-2">
          {filteredReflections.length > 0 ? (
            <div className="relative">
              {/* Линия таймлайна */}
              <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200 transform -translate-x-1/2"></div>

              <div className="space-y-8">
                {filteredReflections.map((reflection, index) => (
                  <div key={reflection.id} className="relative pl-12">
                    {/* Точка на таймлайне */}
                    <div className={`absolute left-0 top-3 w-8 h-8 rounded-full flex items-center justify-center text-white ${getEventTypeColor(reflection.type)} shadow-md cursor-pointer`}
                      onClick={() => handleReflectionClick(reflection)}>
                      {getEventTypeLabel(reflection.type).charAt(0)}
                    </div>

                    <div className="bg-gray-50 rounded-lg p-4 border border-gray-20 hover:shadow-md transition-shadow cursor-pointer"
                      onClick={() => handleReflectionClick(reflection)}>
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-semibold text-lg text-gray-800">{reflection.title}</h3>
                          <div className="flex items-center space-x-4 mt-1">
                            <span className="text-sm text-gray-500">
                              {new Date(reflection.timestamp).toLocaleString('ru-RU')}
                            </span>
                            <span className={`px-2 py-1 rounded-full text-xs text-white ${getEventTypeColor(reflection.type)}`}>
                              {getEventTypeLabel(reflection.type)}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-medium text-gray-700">
                            Уверенность: {(reflection.confidence * 100).toFixed(0)}%
                          </span>
                          <div className="w-16 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${reflection.confidence * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>

                      <p className="mt-3 text-gray-600">{reflection.description}</p>

                      {reflection.relatedLearning && (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <span className="text-sm font-medium text-gray-700">Связанное обучение: </span>
                          <span className="text-sm text-gray-600">{reflection.relatedLearning}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-40 text-gray-500">
              <p>Нет рефлексий, соответствующих выбранным фильтрам</p>
            </div>
          )}
        </div>

        <div className="mt-6 text-sm text-gray-600">
          <p>История процессов саморефлексии агента. Фильтруйте по типам инсайтов и уровню уверенности.</p>
        </div>
      </div>

      {/* Модальное окно для детализации рефлексии */}
      {selectedReflection && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4" onClick={handleCloseDetail}>
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-xl font-bold text-gray-800">{selectedReflection.title}</h3>
                <button onClick={handleCloseDetail} className="text-gray-500 hover:text-gray-700 text-2xl">&times;</button>
              </div>

              <div className="space-y-4">
                <div>
                  <span className="font-medium text-gray-700">Тип: </span>
                  <span className={`px-2 py-1 rounded-full text-xs text-white ${getEventTypeColor(selectedReflection.type)}`}>
                    {getEventTypeLabel(selectedReflection.type)}
                  </span>
                </div>

                <div>
                  <span className="font-medium text-gray-700">Время: </span>
                  <span>{new Date(selectedReflection.timestamp).toLocaleString('ru-RU')}</span>
                </div>

                <div>
                  <span className="font-medium text-gray-700">Уверенность: </span>
                  <span>{(selectedReflection.confidence * 100).toFixed(0)}%</span>
                </div>

                <div>
                  <span className="font-medium text-gray-700">Описание: </span>
                  <p className="mt-1 text-gray-600">{selectedReflection.description}</p>
                </div>

                {selectedReflection.relatedLearning && (
                  <div>
                    <span className="font-medium text-gray-700">Связанное обучение: </span>
                    <p className="mt-1 text-gray-600">{selectedReflection.relatedLearning}</p>
                  </div>
                )}

                {selectedReflection.reasoningAnalysis && (
                  <div className="border-t pt-4 mt-4">
                    <h4 className="font-semibold text-gray-800 mb-2">Анализ рассуждений:</h4>
                    <p>Качество: {(selectedReflection.reasoningAnalysis!.qualityScore * 100).toFixed(0)}%</p>
                    <p>Эффективность: {selectedReflection.reasoningAnalysis!.efficiency.optimizationScore.toFixed(2)}</p>
                  </div>
                )}

                {selectedReflection.performanceAnalysis && (
                  <div className="border-t pt-4 mt-4">
                    <h4 className="font-semibold text-gray-800 mb-2">Анализ производительности:</h4>
                    <p>Время выполнения: {selectedReflection.performanceAnalysis!.metrics.executionTime.toFixed(2)}с</p>
                    <p>Качество: {(selectedReflection.performanceAnalysis!.metrics.qualityScore * 100).toFixed(0)}%</p>
                  </div>
                )}

                {selectedReflection.errorAnalysis && (
                  <div className="border-t pt-4 mt-4">
                    <h4 className="font-semibold text-gray-800 mb-2">Анализ ошибок:</h4>
                    <p>Количество ошибок: {selectedReflection.errorAnalysis!.metrics.total_errors || 0}</p>
                  </div>
                )}

                <div className="border-t pt-4 mt-4">
                  <h4 className="font-semibold text-gray-800 mb-2">Инсайты:</h4>
                  {selectedReflection.insights.length > 0 ? (
                    <ul className="list-disc pl-5 space-y-1">
                      {selectedReflection.insights.map((insight, idx) => (
                        <li key={insight.id} className="text-gray-600">{insight.description}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-500">Нет инсайтов</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReflectionTimeline;

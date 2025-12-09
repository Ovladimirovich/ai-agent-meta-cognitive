import React from 'react';
import { PageWrapper } from '@/shared/ui/PageWrapper';
import LearningMetricsDashboard from '@/widgets/LearningMetricsDashboard/LearningMetricsDashboard';
import ReasoningTraceViewer from '@/widgets/ReasoningTraceViewer/ReasoningTraceViewer';
import { useLearningMetrics } from '@/widgets/LearningMetricsDashboard/hooks/useLearningMetrics';

const LearningAnalyticsPage: React.FC = () => {
  const { loading, error, refresh } = useLearningMetrics();

  if (loading) {
    return (
      <PageWrapper title="Аналитика обучения">
        <div className="flex justify-center items-center h-64">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-cognitive-meta-500 mb-2"></div>
            <p className="text-gray-600 dark:text-gray-300">Загрузка метрик обучения...</p>
          </div>
        </div>
      </PageWrapper>
    );
  }

  if (error) {
    return (
      <PageWrapper title="Аналитика обучения">
        <div className="text-center py-8">
          <p className="text-red-600 dark:text-red-40 mb-2">Ошибка загрузки метрик</p>
          <p className="text-gray-600 dark:text-gray-300">{error}</p>
          <button
            onClick={refresh}
            className="mt-4 px-4 py-2 bg-cognitive-meta-500 text-white rounded hover:bg-cognitive-meta-600"
          >
            Повторить попытку
          </button>
        </div>
      </PageWrapper>
    );
  }

  return (
    <PageWrapper title="Аналитика обучения">
      <div className="space-y-8">
        <LearningMetricsDashboard />
        <ReasoningTraceViewer className="mt-8" />
      </div>
    </PageWrapper>
  );
};

export default LearningAnalyticsPage;

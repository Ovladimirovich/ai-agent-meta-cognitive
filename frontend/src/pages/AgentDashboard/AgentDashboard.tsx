/**
 * AgentDashboard - главная страница приложения
 */

import React, { useState } from 'react';
import { AgentChatInterface } from '../../features/agent-interaction/AgentChatInterface';
import { MemoryVisualizer } from '../../widgets';
import ReflectionTimeline from '../../widgets/ReflectionTimeline/ReflectionTimeline';
import CognitiveHealthMonitor from '../../widgets/CognitiveHealthMonitor/CognitiveHealthMonitor';
import LearningMetricsDashboard from '../../widgets/LearningMetricsDashboard/LearningMetricsDashboard';
import { WebSocketProvider } from '../../widgets/AdvancedAnalyticsDashboard/WebSocketProvider';
import { RealtimeDataProvider } from '../../widgets/RealtimeDataProvider/RealtimeDataProvider';
import { useTheme } from '../../app/providers/ThemeProvider';
import { ChatHistorySidebar } from '../../features/chat-history/ChatHistorySidebar';
import { useChatHistory } from '@/shared/hooks/useChatHistory';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/shared/lib/apiClient';

export const AgentDashboard: React.FC = () => {
  const { theme, toggleTheme } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const {
    sessions,
    currentSessionId,
    createSession,
    deleteSession,
    setCurrentSession
  } = useChatHistory();

  // Загрузка системной информации
  const { data: systemInfo, isLoading: isSystemInfoLoading } = useQuery<any>({
    queryKey: ['systemInfo'],
    queryFn: () => apiClient.getSystemInfo(),
    refetchInterval: 3000, // Обновление каждые 30 секунд
    staleTime: 10000, // Данные считаются свежими 10 секунд
  });

  const handleNewChat = () => {
    createSession();
    setSidebarOpen(false);
  };

  const handleSessionSelect = (sessionId: string) => {
    setCurrentSession(sessionId);
    setSidebarOpen(false);
  };

  const handleSessionDelete = (sessionId: string) => {
    deleteSession(sessionId);
  };

  return (
    <WebSocketProvider url="ws://localhost:8000/ws">
      <RealtimeDataProvider>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
          {/* Боковая панель истории чатов */}
          <ChatHistorySidebar
            sessions={sessions}
            currentSessionId={currentSessionId}
            onSessionSelect={handleSessionSelect}
            onSessionDelete={handleSessionDelete}
            onNewChat={handleNewChat}
            isOpen={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
          />
          {/* Заголовок приложения */}
          <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center py-4">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">AI</span>
                  </div>
                  <div>
                    <h1 className="text-xl font-bold text-gray-90 dark:text-white">
                      Meta-Cognitive AI Agent
                    </h1>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      Интеллектуальная система с самодиагностикой
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  {/* Кнопка истории чатов */}
                  <button
                    onClick={() => setSidebarOpen(true)}
                    className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                    title="История чатов"
                  >
                    <svg className="w-5 h-5 text-gray-800 dark:text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </button>

                  <button
                    onClick={toggleTheme}
                    className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                    title={`Переключить на ${theme === 'light' ? 'темную' : 'светлую'} тему`}
                  >
                    {theme === 'light' ? (
                      <svg className="w-5 h-5 text-gray-800" fill="none" stroke="currentColor" viewBox="0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.03 9.003 0 008.354-5.646z" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                      </svg>
                    )}
                  </button>

                  {/* Индикатор состояния бэкенда */}
                  <div className="text-sm text-gray-600 dark:text-gray-300 flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${isSystemInfoLoading ? 'bg-yellow-500' : systemInfo ? 'bg-green-600' : 'bg-red-60'}`}></div>
                    <span>Backend: {isSystemInfoLoading ? 'Проверка...' : systemInfo ? '● Online' : '○ Offline'}</span>
                  </div>

                  <div className="text-sm text-gray-600 dark:text-gray-300">
                    {systemInfo?.version || 'v1.0.0'}
                  </div>
                </div>
              </div>
            </div>
          </header>

          {/* Основной контент */}
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
              <div className="lg:col-span-2">
                <CognitiveHealthMonitor className="h-full" />
              </div>
              <div className="lg:col-span-1">
                <MemoryVisualizer className="h-full" />
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
              <div className="lg:col-span-1">
                <LearningMetricsDashboard />
              </div>
              <div className="lg:col-span-1">
                <ReflectionTimeline className="h-full" />
              </div>
              <div className="lg:col-span-1">
                <AgentChatInterface />
              </div>
            </div>
          </main>

          {/* Футер */}
          <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-8">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
              <div className="flex justify-between items-center text-sm text-gray-600 dark:text-gray-300">
                <div>
                  © 2025 Meta-Cognitive AI Agent. Построено с использованием React & FastAPI.
                </div>
                <div className="flex space-x-4">
                  <a href="/ai-agent-meta-cognitive/docs" className="hover:text-blue-600 dark:hover:text-blue-400">API Docs</a>
                  <a href="/ai-agent-meta-cognitive/graphql" className="hover:text-blue-600 dark:hover:text-blue-400">GraphQL</a>
                  <a href="https://github.com" className="hover:text-blue-60 dark:hover:text-blue-400">GitHub</a>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </RealtimeDataProvider>
    </WebSocketProvider>
  );
};

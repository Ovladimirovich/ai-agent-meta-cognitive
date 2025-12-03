import React from 'react';
import AgentChatInterface from '../../features/agent-interaction/ui/AgentChatInterface';
import SystemHealthMonitor from '../../widgets/SystemHealthMonitor/SystemHealthMonitor';

const AgentDashboard: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">AI Агент с Мета-когнитивными Способностями</h1>
        <p className="text-gray-600 dark:text-gray-300 mt-2">
          Интерактивный интерфейс для взаимодействия с мета-когнитивным агентом
        </p>
      </header>
      
      <main>
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
              <AgentChatInterface />
            </div>
          </div>
          
          <div className="lg:col-span-2">
            <div className="grid grid-rows-2 gap-6 h-full">
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Мета-когнитивные метрики</h2>
                <div className="space-y-4">
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <h3 className="font-medium text-gray-700 dark:text-gray-200">Уровень уверенности</h3>
                    <div className="mt-2 w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2.5">
                      <div className="bg-cognitive-meta-500 h-2.5 rounded-full" style={{ width: '75%' }}></div>
                    </div>
                    <span className="text-sm text-gray-500 dark:text-gray-400 mt-1">75%</span>
                  </div>
                  
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <h3 className="font-medium text-gray-700 dark:text-gray-200">Когнитивная нагрузка</h3>
                    <div className="mt-2 w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2.5">
                      <div className="bg-cognitive-learning-500 h-2.5 rounded-full" style={{ width: '45%' }}></div>
                    </div>
                    <span className="text-sm text-gray-500 dark:text-gray-400 mt-1">45%</span>
                  </div>
                  
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <h3 className="font-medium text-gray-70 dark:text-gray-200">Самооценка</h3>
                    <div className="mt-2 w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2.5">
                      <div className="bg-cognitive-reflection-500 h-2.5 rounded-full" style={{ width: '80%' }}></div>
                    </div>
                    <span className="text-sm text-gray-500 dark:text-gray-400 mt-1">80%</span>
                  </div>
                </div>
              </div>
              
              <SystemHealthMonitor pollingInterval={5000} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default AgentDashboard;
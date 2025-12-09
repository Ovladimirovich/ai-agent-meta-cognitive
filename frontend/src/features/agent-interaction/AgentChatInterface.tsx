/**
 * AgentChatInterface - основной компонент для взаимодействия с AI агентом
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from '@/shared/ui/atoms/Button';
import { apiClient } from '@/shared/lib/apiClient';
import { AgentRequest, AgentResponse } from '@/shared/types/api';
import { ChatMessage } from '@/shared/types/chat';
import { useQueryClient } from '@tanstack/react-query';
import DOMPurify from 'dompurify';

interface AgentChatInterfaceProps {
  className?: string;
}

export const AgentChatInterface: React.FC<AgentChatInterfaceProps> = ({
  className = ''
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  // Автопрокрутка к последнему сообщению
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: DOMPurify.sanitize(inputValue.trim()),
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      const request: AgentRequest = {
        query: userMessage.content,
        user_id: 'frontend_user',
        session_id: `session_${Date.now()}`,
      };

      const response: AgentResponse = await apiClient.processRequest(request);

      const agentMessage: ChatMessage = {
        id: response.id,
        type: 'agent',
        content: DOMPurify.sanitize(response.content),
        timestamp: response.timestamp,
        confidence: response.confidence,
        processingTime: response.processing_time,
      };

      setMessages(prev => [...prev, agentMessage]);
      // Инвалидация кэша для обновления данных
      queryClient.invalidateQueries({ queryKey: ['systemInfo'] });
    } catch (err) {
      console.error('Error sending message:', err);
      setError(err instanceof Error ? err.message : 'Произошла ошибка при отправке сообщения');

      // Добавляем сообщение об ошибке
      const errorMessage: ChatMessage = {
        id: `error_${Date.now()}`,
        type: 'agent',
        content: DOMPurify.sanitize('Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз.'),
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearHistory = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className={`flex flex-col h-full max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-lg ${className}`}>
      {/* Заголовок */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <div>
          <h2 className="text-xl font-semibold text-gray-90 dark:text-white">
            Мета-Когнитивный AI Агент
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Интеллектуальный помощник с самодиагностикой и обучением
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={clearHistory}
          disabled={messages.length === 0}
        >
          Очистить историю
        </Button>
      </div>

      {/* Сообщения */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 dark:text-gray-400 py-8">
            <div className="text-4xl mb-4">🤖</div>
            <p className="text-lg font-medium">Начните разговор с AI агентом</p>
            <p className="text-sm">Задайте любой вопрос или дайте задание</p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${message.type === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white'
                }`}
            >
              <p className="text-sm" dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(message.content) }} />

              {/* Метаданные для сообщений агента */}
              {message.type === 'agent' && message.confidence !== undefined && (
                <div className="mt-2 text-xs opacity-75">
                  <div className="flex items-center space-x-2">
                    <span>Уверенность: {Math.round(message.confidence * 100)}%</span>
                    {message.processingTime && (
                      <span>Ответ за {(message.processingTime).toFixed(1)} сек</span>
                    )}
                  </div>
                </div>
              )}

              <div className="text-xs opacity-50 mt-1">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}

        {/* Индикатор набора текста */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white px-4 py-2 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="text-sm">Агент печатает</div>
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-600 dark:bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-600 dark:bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-600 dark:bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Сообщение об ошибке */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-20 dark:border-red-800 rounded-lg p-3">
            <div className="flex items-center">
              <div className="text-red-60 dark:text-red-400 text-sm">
                ⚠️ {error}
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Ввод сообщения */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-4">
        <div className="flex space-x-2">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(DOMPurify.sanitize(e.target.value))}
            onKeyPress={handleKeyPress}
            placeholder="Введите ваше сообщение..."
            className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-40"
            rows={2}
            disabled={isLoading}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            loading={isLoading}
            className="self-end"
          >
            Отправить
          </Button>
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">
          Нажмите Enter для отправки, Shift+Enter для новой строки
        </div>
      </div>
    </div>
  );
};

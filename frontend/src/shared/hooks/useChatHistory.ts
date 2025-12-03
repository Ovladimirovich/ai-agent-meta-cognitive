/**
 * Хук для управления историей чатов
 */

import { useState, useEffect, useCallback } from 'react';
import { ChatSession, ChatMessage, ChatHistoryActions } from '../types/chat';

const STORAGE_KEY = 'ai_agent_chat_history';
const MAX_SESSIONS = 50; // Максимальное количество сохраненных сессий

export const useChatHistory = (): ChatHistoryActions & {
  sessions: ChatSession[];
  currentSessionId: string | null;
} => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  // Загрузка истории из localStorage при инициализации
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const history = JSON.parse(stored);
        setSessions(history.sessions || []);
        setCurrentSessionId(history.currentSessionId || null);
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  }, []);

  // Сохранение истории в localStorage
  const saveToStorage = useCallback((newSessions: ChatSession[], newCurrentId: string | null) => {
    try {
      const history = {
        sessions: newSessions,
        currentSessionId: newCurrentId
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (error) {
      console.error('Failed to save chat history:', error);
    }
  }, []);

  // Генерация заголовка для сессии на основе первого сообщения
  const generateTitle = (messages: ChatMessage[]): string => {
    if (messages.length === 0) return 'Новый чат';

    const firstUserMessage = messages.find(msg => msg.type === 'user');
    if (firstUserMessage) {
      // Берем первые 30 символов первого пользовательского сообщения
      const content = firstUserMessage.content.slice(0, 30);
      return content.length < firstUserMessage.content.length ? `${content}...` : content;
    }

    return `Чат от ${new Date().toLocaleDateString('ru-RU')}`;
  };

  // Создание новой сессии
  const createSession = useCallback((title?: string): string => {
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const now = new Date().toISOString();

    const newSession: ChatSession = {
      id: sessionId,
      title: title || 'Новый чат',
      messages: [],
      createdAt: now,
      updatedAt: now,
      messageCount: 0
    };

    setSessions(prev => {
      const updated = [newSession, ...prev].slice(0, MAX_SESSIONS);
      saveToStorage(updated, sessionId);
      return updated;
    });

    setCurrentSessionId(sessionId);
    saveToStorage(sessions, sessionId);

    return sessionId;
  }, [sessions, saveToStorage]);

  // Сохранение сессии
  const saveSession = useCallback((sessionId: string, messages: ChatMessage[]) => {
    if (messages.length === 0) return;

    setSessions(prev => {
      const updated = prev.map(session => {
        if (session.id === sessionId) {
          const title = session.title === 'Новый чат' ? generateTitle(messages) : session.title;
          return {
            ...session,
            messages: [...messages],
            title,
            updatedAt: new Date().toISOString(),
            messageCount: messages.length
          };
        }
        return session;
      });

      saveToStorage(updated, currentSessionId);
      return updated;
    });
  }, [currentSessionId, saveToStorage]);

  // Загрузка сессии
  const loadSession = useCallback((sessionId: string): ChatSession | null => {
    const session = sessions.find(s => s.id === sessionId);
    return session || null;
  }, [sessions]);

  // Удаление сессии
  const deleteSession = useCallback((sessionId: string) => {
    setSessions(prev => {
      const updated = prev.filter(session => session.id !== sessionId);

      // Если удаляем текущую сессию, сбрасываем currentSessionId
      let newCurrentId = currentSessionId;
      if (currentSessionId === sessionId) {
        newCurrentId = updated.length > 0 ? updated[0].id : null;
      }

      setCurrentSessionId(newCurrentId);
      saveToStorage(updated, newCurrentId);
      return updated;
    });
  }, [currentSessionId, saveToStorage]);

  // Получение всех сессий
  const getAllSessions = useCallback((): ChatSession[] => {
    return [...sessions];
  }, [sessions]);

  // Установка текущей сессии
  const setCurrentSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId);
    saveToStorage(sessions, sessionId);
  }, [sessions, saveToStorage]);

  // Очистка всей истории
  const clearHistory = useCallback(() => {
    setSessions([]);
    setCurrentSessionId(null);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  return {
    sessions,
    currentSessionId,
    createSession,
    saveSession,
    loadSession,
    deleteSession,
    getAllSessions,
    setCurrentSession,
    clearHistory
  };
};

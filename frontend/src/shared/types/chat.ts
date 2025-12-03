/**
 * Типы для системы истории чатов
 */

export interface ChatMessage {
  id: string;
  type: 'user' | 'agent';
  content: string;
  timestamp: string;
  confidence?: number;
  processingTime?: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export interface ChatHistory {
  sessions: ChatSession[];
  currentSessionId: string | null;
}

export interface ChatHistoryActions {
  createSession: (title?: string) => string;
  saveSession: (sessionId: string, messages: ChatMessage[]) => void;
  loadSession: (sessionId: string) => ChatSession | null;
  deleteSession: (sessionId: string) => void;
  getAllSessions: () => ChatSession[];
  setCurrentSession: (sessionId: string) => void;
  clearHistory: () => void;
}

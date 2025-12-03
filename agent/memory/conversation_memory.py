"""
Conversation Memory - Управление памятью разговоров
"""

import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ConversationMessage(BaseModel):
    """Структура сообщения в разговоре"""
    id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class ConversationSession(BaseModel):
    """Структура сессии разговора"""
    id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    messages: List[ConversationMessage] = []
    context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class ConversationMemory:
    """Управление памятью разговоров агента"""

    def __init__(self, memory_manager=None, max_sessions: int = 1000,
                 session_timeout_hours: int = 24):
        self.memory_manager = memory_manager
        self.max_sessions = max_sessions
        self.session_timeout = timedelta(hours=session_timeout_hours)

        # Активные сессии в памяти
        self.active_sessions: Dict[str, ConversationSession] = {}

        # Архивные сессии (для долгосрочного хранения)
        self.archived_sessions: Dict[str, ConversationSession] = {}

        # Статистика
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'archived_sessions': 0,
            'total_messages': 0
        }

        logger.info(f"ConversationMemory initialized with max {max_sessions} sessions")

    async def create_session(self, user_id: str, context: Dict[str, Any] = None) -> str:
        """Создание новой сессии разговора"""
        session_id = str(uuid.uuid4())
        now = datetime.now()

        session = ConversationSession(
            id=session_id,
            user_id=user_id,
            start_time=now,
            last_activity=now,
            messages=[],
            context=context or {},
            metadata={'created_at': now.isoformat()}
        )

        self.active_sessions[session_id] = session
        self.stats['total_sessions'] += 1
        self.stats['active_sessions'] = len(self.active_sessions)

        # Очистка старых сессий при необходимости
        await self._cleanup_expired_sessions()

        logger.info(f"Created conversation session {session_id} for user {user_id}")
        return session_id

    async def add_message(self, session_id: str, role: str, content: str,
                         metadata: Dict[str, Any] = None) -> bool:
        """Добавление сообщения в сессию"""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return False

        session = self.active_sessions[session_id]
        message_id = str(uuid.uuid4())

        message = ConversationMessage(
            id=message_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        session.messages.append(message)
        session.last_activity = datetime.now()
        self.stats['total_messages'] += 1

        # Сохранение в долгосрочную память
        await self._persist_message(session, message)

        logger.debug(f"Added message to session {session_id}: {role}")
        return True

    async def get_conversation_history(self, session_id: str, limit: int = 50,
                                     include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Получение истории разговора"""
        if session_id not in self.active_sessions:
            # Попытка найти в архиве
            if session_id in self.archived_sessions:
                session = self.archived_sessions[session_id]
            else:
                logger.warning(f"Session {session_id} not found")
                return []
        else:
            session = self.active_sessions[session_id]

        # Получение последних сообщений
        messages = session.messages[-limit:] if limit > 0 else session.messages

        # Форматирование для ответа
        history = []
        for msg in messages:
            message_dict = {
                'id': msg.id,
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat()
            }

            if include_metadata:
                message_dict['metadata'] = msg.metadata

            history.append(message_dict)

        return history

    async def get_recent_conversations(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Получение недавних разговоров пользователя"""
        user_sessions = [
            session for session in self.active_sessions.values()
            if session.user_id == user_id
        ]

        # Сортировка по времени последней активности
        user_sessions.sort(key=lambda s: s.last_activity, reverse=True)

        conversations = []
        for session in user_sessions[:limit]:
            conversations.append({
                'session_id': session.id,
                'start_time': session.start_time.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'message_count': len(session.messages),
                'context': session.context
            })

        return conversations

    async def search_conversations(self, user_id: str, query: str,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Поиск в разговорах пользователя"""
        results = []

        # Поиск во всех сессиях пользователя
        all_sessions = list(self.active_sessions.values()) + list(self.archived_sessions.values())
        user_sessions = [s for s in all_sessions if s.user_id == user_id]

        for session in user_sessions:
            for message in session.messages:
                if query.lower() in message.content.lower():
                    results.append({
                        'session_id': session.id,
                        'message_id': message.id,
                        'role': message.role,
                        'content': message.content,
                        'timestamp': message.timestamp.isoformat(),
                        'context': session.context
                    })

                    if len(results) >= limit:
                        break

            if len(results) >= limit:
                break

        return results

    async def get_conversation_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Получение сводки по разговору"""
        if session_id not in self.active_sessions and session_id not in self.archived_sessions:
            return None

        session = (self.active_sessions.get(session_id) or
                  self.archived_sessions.get(session_id))

        if not session or not session.messages:
            return None

        # Анализ разговора
        user_messages = [m for m in session.messages if m.role == 'user']
        assistant_messages = [m for m in session.messages if m.role == 'assistant']

        # Определение основных тем (простая эвристика)
        all_content = ' '.join([m.content for m in session.messages])
        topics = self._extract_topics(all_content)

        summary = {
            'session_id': session.id,
            'user_id': session.user_id,
            'duration': (session.last_activity - session.start_time).total_seconds(),
            'message_count': len(session.messages),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'topics': topics,
            'start_time': session.start_time.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'context': session.context
        }

        return summary

    async def archive_session(self, session_id: str) -> bool:
        """Архивирование сессии"""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        self.archived_sessions[session_id] = session
        del self.active_sessions[session_id]

        self.stats['active_sessions'] = len(self.active_sessions)
        self.stats['archived_sessions'] = len(self.archived_sessions)

        # Сохранение в долгосрочную память
        await self._persist_session(session)

        logger.info(f"Archived session {session_id}")
        return True

    async def delete_session(self, session_id: str) -> bool:
        """Удаление сессии"""
        deleted = False

        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.stats['active_sessions'] = len(self.active_sessions)
            deleted = True

        if session_id in self.archived_sessions:
            del self.archived_sessions[session_id]
            self.stats['archived_sessions'] = len(self.archived_sessions)
            deleted = True

        if deleted:
            # Удаление из долгосрочной памяти
            await self._delete_persisted_session(session_id)
            logger.info(f"Deleted session {session_id}")

        return deleted

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Получение статистики пользователя"""
        user_sessions = [
            s for s in list(self.active_sessions.values()) + list(self.archived_sessions.values())
            if s.user_id == user_id
        ]

        if not user_sessions:
            return {'user_id': user_id, 'total_sessions': 0}

        total_messages = sum(len(s.messages) for s in user_sessions)
        total_duration = sum(
            (s.last_activity - s.start_time).total_seconds()
            for s in user_sessions
        )

        return {
            'user_id': user_id,
            'total_sessions': len(user_sessions),
            'active_sessions': len([s for s in user_sessions if s.id in self.active_sessions]),
            'total_messages': total_messages,
            'avg_session_duration': total_duration / len(user_sessions),
            'avg_messages_per_session': total_messages / len(user_sessions) if user_sessions else 0
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Получение статистики памяти разговоров"""
        return {
            'active_sessions': len(self.active_sessions),
            'archived_sessions': len(self.archived_sessions),
            'total_sessions': self.stats['total_sessions'],
            'total_messages': self.stats['total_messages'],
            'avg_messages_per_session': (
                self.stats['total_messages'] / self.stats['total_sessions']
                if self.stats['total_sessions'] > 0 else 0
            )
        }

    async def _cleanup_expired_sessions(self):
        """Очистка истекших сессий"""
        now = datetime.now()
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            if now - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.archive_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def _persist_message(self, session: ConversationSession, message: ConversationMessage):
        """Сохранение сообщения в долгосрочную память"""
        if not self.memory_manager:
            return

        try:
            # Сохранение в эпизодическую память
            memory_data = {
                'session_id': session.id,
                'user_id': session.user_id,
                'message_id': message.id,
                'role': message.role,
                'content': message.content,
                'timestamp': message.timestamp,
                'context': session.context,
                'metadata': message.metadata
            }

            await self.memory_manager.store_episodic_memory(memory_data)

        except Exception as e:
            logger.warning(f"Failed to persist message: {e}")

    async def _persist_session(self, session: ConversationSession):
        """Сохранение сессии в долгосрочную память"""
        if not self.memory_manager:
            return

        try:
            # Сохранение сводки сессии
            session_summary = await self.get_conversation_summary(session.id)
            if session_summary:
                await self.memory_manager.store_episodic_memory({
                    'type': 'conversation_summary',
                    'session_data': session_summary,
                    'timestamp': datetime.now()
                })

        except Exception as e:
            logger.warning(f"Failed to persist session: {e}")

    async def _delete_persisted_session(self, session_id: str):
        """Удаление сессии из долгосрочной памяти"""
        # В будущем можно реализовать удаление из постоянного хранилища
        pass

    def _extract_topics(self, content: str, max_topics: int = 5) -> List[str]:
        """Извлечение основных тем из текста разговора"""
        # Простая эвристика - поиск часто встречающихся существительных
        words = content.lower().split()
        word_counts = {}

        # Фильтрация стоп-слов и коротких слов
        stop_words = {'и', 'в', 'на', 'с', 'по', 'из', 'к', 'о', 'а', 'the', 'a', 'an', 'to', 'of', 'in', 'for', 'on', 'at', 'by', 'with'}
        filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]

        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Топ слов как темы
        topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in topics[:max_topics]]

    async def export_conversations(self, user_id: str = None, format: str = "dict") -> Any:
        """Экспорт разговоров"""
        if user_id:
            sessions = [
                s for s in list(self.active_sessions.values()) + list(self.archived_sessions.values())
                if s.user_id == user_id
            ]
        else:
            sessions = list(self.active_sessions.values()) + list(self.archived_sessions.values())

        export_data = {
            'sessions': [s.dict() for s in sessions],
            'export_timestamp': datetime.now().isoformat(),
            'stats': self.get_memory_stats()
        }

        return export_data

    async def import_conversations(self, import_data: Dict[str, Any]):
        """Импорт разговоров"""
        try:
            for session_data in import_data.get('sessions', []):
                # Преобразование обратно в ConversationSession
                # (упрощенная версия)
                session = ConversationSession(**session_data)

                if session.id not in self.active_sessions and session.id not in self.archived_sessions:
                    # Определение, активная или архивная
                    if datetime.now() - session.last_activity < self.session_timeout:
                        self.active_sessions[session.id] = session
                    else:
                        self.archived_sessions[session.id] = session

            logger.info(f"Imported {len(import_data.get('sessions', []))} conversations")

        except Exception as e:
            logger.error(f"Failed to import conversations: {e}")

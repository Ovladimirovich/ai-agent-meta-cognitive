"""
PostgreSQL интеграция для AI Агента с Мета-Познанием
Фаза 5: Инфраструктура и интеграции
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import StaticPool

from agent.learning.models import AgentExperience, ProcessedExperience, Pattern, LearningResult

logger = logging.getLogger(__name__)

# SQLAlchemy модели
Base = declarative_base()


class ExperienceRecord(Base):
    """Запись опыта агента в БД"""
    __tablename__ = 'agent_experiences'

    id = Column(String, primary_key=True)
    query = Column(Text, nullable=False)
    result = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    execution_time = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    user_id = Column(String)
    session_id = Column(String)
    success_indicators = Column(JSON)
    error_indicators = Column(JSON)
    metadata_json = Column(JSON)
    processed = Column(Boolean, default=False)

    # Связи
    processed_experience = relationship("ProcessedExperienceRecord", back_populates="original_experience", uselist=False)


class ProcessedExperienceRecord(Base):
    """Обработанный опыт агента"""
    __tablename__ = 'processed_experiences'

    id = Column(String, primary_key=True)
    original_experience_id = Column(String, ForeignKey('agent_experiences.id'), nullable=False)
    significance_score = Column(Float, nullable=False)
    categories = Column(JSON)
    lessons = Column(JSON)
    key_elements = Column(JSON)
    emotional_context = Column(JSON)
    processing_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Связи
    original_experience = relationship("ExperienceRecord", back_populates="processed_experience")


class PatternRecord(Base):
    """Запись паттерна в БД"""
    __tablename__ = 'patterns'

    id = Column(String, primary_key=True)
    type = Column(String, nullable=False)
    trigger_conditions = Column(JSON)
    description = Column(Text)
    confidence = Column(Float, nullable=False)
    frequency = Column(Float, nullable=False, default=1.0)
    examples = Column(JSON)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)


class LearningResultRecord(Base):
    """Результаты обучения"""
    __tablename__ = 'learning_results'

    id = Column(String, primary_key=True)
    experience_processed_id = Column(String, ForeignKey('processed_experiences.id'))
    patterns_extracted = Column(Integer, nullable=False, default=0)
    cognitive_updates = Column(Integer, nullable=False, default=0)
    skills_developed = Column(Integer, nullable=False, default=0)
    learning_effectiveness = Column(Float, nullable=False)
    learning_time = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    adaptation_result = Column(JSON)


class UserSession(Base):
    """Сессия пользователя"""
    __tablename__ = 'user_sessions'

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime)
    total_requests = Column(Integer, default=0)
    total_processing_time = Column(Float, default=0.0)
    metadata_json = Column(JSON)


class SystemMetrics(Base):
    """Системные метрики"""
    __tablename__ = 'system_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    metadata_json = Column(JSON)


class PostgreSQLManager:
    """
    Менеджер PostgreSQL для постоянного хранения данных агента
    """

    def __init__(self, connection_string: str):
        """
        Инициализация PostgreSQL менеджера

        Args:
            connection_string: Строка подключения к PostgreSQL
        """
        self.connection_string = connection_string
        self.engine = None
        self.SessionLocal = None
        self.pool = None

        # Статистика
        self.stats = {
            'connections_created': 0,
            'queries_executed': 0,
            'errors_occurred': 0,
            'total_storage_size': 0
        }

        logger.info("PostgreSQLManager initialized")

    async def initialize(self):
        """Инициализация подключения к БД"""
        try:
            # Создание SQLAlchemy engine с оптимизациями для производительности
            self.engine = create_engine(
                self.connection_string,
                poolclass=StaticPool,
                echo=False,  # Отключить логирование SQL
                connect_args={
                    "check_same_thread": False,
                },
                # Оптимизации пула соединений
                pool_size=20,  # Увеличенный размер пула
                max_overflow=30,  # Максимальное количество дополнительных соединений
                pool_pre_ping=True,  # Проверка соединения перед использованием
                pool_recycle=300,  # Пересоздание соединений каждые 5 минут
                pool_timeout=30,  # Таймаут ожидания соединения
                echo_pool=False  # Отключить логирование пула
            )

            # Создание таблиц и индексов
            Base.metadata.create_all(bind=self.engine)
            
            # Создание дополнительных индексов для оптимизации запросов
            from sqlalchemy import Index
            
            # Индексы для таблицы agent_experiences
            Index('idx_agent_experiences_timestamp', ExperienceRecord.timestamp).create(self.engine)
            Index('idx_agent_experiences_user_id', ExperienceRecord.user_id).create(self.engine)
            Index('idx_agent_experiences_session_id', ExperienceRecord.session_id).create(self.engine)
            Index('idx_agent_experiences_confidence', ExperienceRecord.confidence).create(self.engine)
            Index('idx_agent_experiences_processed', ExperienceRecord.processed).create(self.engine)
            
            # Индексы для таблицы patterns
            Index('idx_patterns_type', PatternRecord.type).create(self.engine)
            Index('idx_patterns_confidence', PatternRecord.confidence).create(self.engine)
            Index('idx_patterns_frequency', PatternRecord.frequency).create(self.engine)
            
            # Индексы для таблицы learning_results
            Index('idx_learning_results_timestamp', LearningResultRecord.timestamp).create(self.engine)
            Index('idx_learning_results_effectiveness', LearningResultRecord.learning_effectiveness).create(self.engine)
            
            # Индексы для таблицы user_sessions
            Index('idx_user_sessions_user_id', UserSession.user_id).create(self.engine)
            Index('idx_user_sessions_start_time', UserSession.start_time).create(self.engine)
            
            # Индексы для таблицы system_metrics
            Index('idx_system_metrics_type', SystemMetrics.metric_type).create(self.engine)
            Index('idx_system_metrics_timestamp', SystemMetrics.timestamp).create(self.engine)
            Index('idx_system_metrics_value', SystemMetrics.value).create(self.engine)

            # Создание сессии с оптимизациями
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                expire_on_commit=False  # Не обновлять объекты после коммита для производительности
            )

            # Создание пула соединений asyncpg с оптимизациями
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=10,      # Увеличенный минимальный размер пула
                max_size=30,      # Увеличенный максимальный размер пула
                command_timeout=60,
                statement_cache_size=1000,  # Увеличенный кэш SQL-запросов
                max_cached_statement_lifetime=300,  # Время жизни кэшированного запроса
                max_queries=5000,  # Максимальное количество запросов до пересоздания соединения
                server_settings={
                    'application_name': 'AI_Agent_Meta_Cognitive',
                    'idle_in_transaction_session_timeout': '30000',  # 30 секунд
                }
            )

            logger.info("✅ PostgreSQL connection established with performance optimizations")

        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
            raise

    async def close(self):
        """Закрытие подключения к БД"""
        if self.pool:
            await self.pool.close()
        if self.engine:
            self.engine.dispose()
        logger.info("PostgreSQL connection closed")

    @asynccontextmanager
    async def get_session(self):
        """Получение сессии БД"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    # Методы для работы с опытом агента

    async def store_experience(self, experience: AgentExperience) -> bool:
        """
        Сохранение опыта агента

        Args:
            experience: Опыт агента

        Returns:
            bool: Успех операции
        """
        try:
            # Используем пул соединений asyncpg для более быстрой вставки
            async with self.pool.acquire() as conn:
                # Проверка существования
                existing = await conn.fetchval(
                    "SELECT id FROM agent_experiences WHERE id = $1",
                    experience.id
                )
                if existing:
                    logger.debug(f"Experience {experience.id} already exists")
                    return True

                # Массовая вставка с подготовленным запросом для лучшей производительности
                await conn.execute(
                    """
                    INSERT INTO agent_experiences (
                        id, query, result, confidence, execution_time, timestamp,
                        user_id, session_id, success_indicators, error_indicators, metadata_json
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    experience.id,
                    getattr(experience, 'query', ''),
                    getattr(experience, 'result', ''),
                    getattr(experience, 'confidence', 0.0),
                    getattr(experience, 'execution_time', 0.0),
                    getattr(experience, 'timestamp', datetime.utcnow()),
                    getattr(experience, 'user_id', None),
                    getattr(experience, 'session_id', None),
                    getattr(experience, 'success_indicators', []),
                    getattr(experience, 'error_indicators', []),
                    getattr(experience, 'metadata', {})
                )

                self.stats['queries_executed'] += 1
                logger.debug(f"Stored experience {experience.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store experience {experience.id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_experience(self, experience_id: str) -> Optional[AgentExperience]:
        """
        Получение опыта по ID

        Args:
            experience_id: ID опыта

        Returns:
            Optional[AgentExperience]: Опыт агента или None
        """
        try:
            async with self.get_session() as session:
                record = session.query(ExperienceRecord).filter_by(id=experience_id).first()

                if not record:
                    return None

                # Преобразование в объект AgentExperience
                experience = AgentExperience(
                    id=record.id,
                    query=record.query,
                    result=record.result,
                    confidence=record.confidence,
                    execution_time=record.execution_time,
                    timestamp=record.timestamp,
                    user_id=record.user_id,
                    session_id=record.session_id,
                    success_indicators=record.success_indicators or [],
                    error_indicators=record.error_indicators or [],
                    metadata=record.metadata_json or {}
                )

                self.stats['queries_executed'] += 1
                return experience

        except Exception as e:
            logger.error(f"Failed to get experience {experience_id}: {e}")
            self.stats['errors_occurred'] += 1
            return None

    async def get_recent_experiences(self, limit: int = 100, user_id: Optional[str] = None) -> List[AgentExperience]:
        """
        Получение недавних опытов

        Args:
            limit: Максимальное количество
            user_id: Фильтр по пользователю

        Returns:
            List[AgentExperience]: Список опытов
        """
        try:
            # Используем пул соединений asyncpg для более быстрого получения данных
            async with self.pool.acquire() as conn:
                # Построение SQL-запроса
                if user_id:
                    query = """
                        SELECT id, query, result, confidence, execution_time, timestamp,
                               user_id, session_id, success_indicators, error_indicators, metadata_json
                        FROM agent_experiences
                        WHERE user_id = $1
                        ORDER BY timestamp DESC
                        LIMIT $2
                    """
                    records = await conn.fetch(query, user_id, limit)
                else:
                    query = """
                        SELECT id, query, result, confidence, execution_time, timestamp,
                               user_id, session_id, success_indicators, error_indicators, metadata_json
                        FROM agent_experiences
                        ORDER BY timestamp DESC
                        LIMIT $1
                    """
                    records = await conn.fetch(query, limit)

                # Преобразование результатов в объекты AgentExperience
                experiences = []
                for record in records:
                    experience = AgentExperience(
                        id=record['id'],
                        query=record['query'],
                        result=record['result'],
                        confidence=record['confidence'],
                        execution_time=record['execution_time'],
                        timestamp=record['timestamp'],
                        user_id=record['user_id'],
                        session_id=record['session_id'],
                        success_indicators=record['success_indicators'] or [],
                        error_indicators=record['error_indicators'] or [],
                        metadata=record['metadata_json'] or {}
                    )
                    experiences.append(experience)

                self.stats['queries_executed'] += 1
                return experiences

        except Exception as e:
            logger.error(f"Failed to get recent experiences: {e}")
            self.stats['errors_occurred'] += 1
            return []

    # Методы для работы с паттернами

    async def store_pattern(self, pattern: Pattern) -> bool:
        """
        Сохранение паттерна

        Args:
            pattern: Паттерн

        Returns:
            bool: Успех операции
        """
        try:
            # Используем пул соединений asyncpg для более быстрой вставки/обновления
            async with self.pool.acquire() as conn:
                # Проверка существования
                existing = await conn.fetchval(
                    "SELECT id FROM patterns WHERE id = $1",
                    pattern.id
                )
                
                if existing:
                    # Обновление существующего паттерна
                    await conn.execute(
                        """
                        UPDATE patterns
                        SET trigger_conditions = $1, description = $2, confidence = $3,
                            frequency = $4, examples = $5, last_updated = $6
                        WHERE id = $7
                        """,
                        pattern.trigger_conditions,
                        getattr(pattern, 'description', ''),
                        pattern.confidence,
                        getattr(pattern, 'frequency', 1.0),
                        getattr(pattern, 'examples', []),
                        datetime.utcnow(),
                        pattern.id
                    )
                else:
                    # Создание нового паттерна
                    await conn.execute(
                        """
                        INSERT INTO patterns (
                            id, type, trigger_conditions, description, confidence,
                            frequency, examples, last_updated
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        pattern.id,
                        getattr(pattern, 'type', 'unknown'),
                        pattern.trigger_conditions,
                        getattr(pattern, 'description', ''),
                        pattern.confidence,
                        getattr(pattern, 'frequency', 1.0),
                        getattr(pattern, 'examples', []),
                        datetime.utcnow()
                    )

                self.stats['queries_executed'] += 1
                logger.debug(f"Stored pattern {pattern.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store pattern {pattern.id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """
        Получение паттерна по ID

        Args:
            pattern_id: ID паттерна

        Returns:
            Optional[Pattern]: Паттерн или None
        """
        try:
            async with self.get_session() as session:
                record = session.query(PatternRecord).filter_by(id=pattern_id).first()

                if not record:
                    return None

                # Преобразование в объект Pattern
                pattern = Pattern(
                    id=record.id,
                    type=record.type,
                    trigger_conditions=record.trigger_conditions,
                    description=record.description,
                    confidence=record.confidence,
                    frequency=record.frequency,
                    examples=record.examples,
                    last_updated=record.last_updated
                )

                self.stats['queries_executed'] += 1
                return pattern

        except Exception as e:
            logger.error(f"Failed to get pattern {pattern_id}: {e}")
            self.stats['errors_occurred'] += 1
            return None

    async def get_patterns_by_type(self, pattern_type: str, limit: int = 100) -> List[Pattern]:
        """
        Получение паттернов по типу

        Args:
            pattern_type: Тип паттерна
            limit: Максимальное количество

        Returns:
            List[Pattern]: Список паттернов
        """
        try:
            # Используем пул соединений asyncpg для более быстрого получения данных
            async with self.pool.acquire() as conn:
                query = """
                    SELECT id, type, trigger_conditions, description, confidence,
                           frequency, examples, last_updated
                    FROM patterns
                    WHERE type = $1
                    ORDER BY frequency DESC
                    LIMIT $2
                """
                records = await conn.fetch(query, pattern_type, limit)

                # Преобразование результатов в объекты Pattern
                patterns = []
                for record in records:
                    pattern = Pattern(
                        id=record['id'],
                        type=record['type'],
                        trigger_conditions=record['trigger_conditions'],
                        description=record['description'],
                        confidence=record['confidence'],
                        frequency=record['frequency'],
                        examples=record['examples'],
                        last_updated=record['last_updated']
                    )
                    patterns.append(pattern)

                self.stats['queries_executed'] += 1
                return patterns

        except Exception as e:
            logger.error(f"Failed to get patterns by type {pattern_type}: {e}")
            self.stats['errors_occurred'] += 1
            return []

    # Методы для работы с результатами обучения

    async def store_learning_result(self, result: LearningResult) -> bool:
        """
        Сохранение результата обучения

        Args:
            result: Результат обучения

        Returns:
            bool: Успех операции
        """
        try:
            async with self.get_session() as session:
                record = LearningResultRecord(
                    id=f"lr_{int(datetime.utcnow().timestamp())}_{hash(str(result)) % 1000}",
                    experience_processed_id=getattr(result, 'experience_processed', {}).get('id') if hasattr(result, 'experience_processed') and result.experience_processed else None,
                    patterns_extracted=getattr(result, 'patterns_extracted', 0),
                    cognitive_updates=getattr(result, 'cognitive_updates', 0),
                    skills_developed=getattr(result, 'skills_developed', 0),
                    learning_effectiveness=getattr(result, 'learning_effectiveness', 0.0),
                    learning_time=getattr(result, 'learning_time', 0.0),
                    timestamp=getattr(result, 'timestamp', datetime.utcnow()),
                    adaptation_result=getattr(result, 'adaptation_applied', {}).model_dump() if hasattr(result, 'adaptation_applied') and result.adaptation_applied else {}
                )

                session.add(record)
                session.commit()

                self.stats['queries_executed'] += 1
                logger.debug(f"Stored learning result")
                return True

        except Exception as e:
            logger.error(f"Failed to store learning result: {e}")
            self.stats['errors_occurred'] += 1
            return False

    # Методы для работы с сессиями пользователей

    async def create_user_session(self, user_id: str, session_id: str, metadata: Optional[Dict] = None) -> bool:
        """
        Создание сессии пользователя

        Args:
            user_id: ID пользователя
            session_id: ID сессии
            metadata: Метаданные сессии

        Returns:
            bool: Успех операции
        """
        try:
            async with self.get_session() as session:
                record = UserSession(
                    id=session_id,
                    user_id=user_id,
                    metadata_json=metadata or {}
                )

                session.add(record)
                session.commit()

                self.stats['queries_executed'] += 1
                logger.debug(f"Created user session {session_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to create user session {session_id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def update_user_session(self, session_id: str, total_requests: int, total_processing_time: float) -> bool:
        """
        Обновление статистики сессии

        Args:
            session_id: ID сессии
            total_requests: Общее количество запросов
            total_processing_time: Общее время обработки

        Returns:
            bool: Успех операции
        """
        try:
            # Используем пул соединений asyncpg для более быстрого обновления
            async with self.pool.acquire() as conn:
                rows_affected = await conn.execute(
                    """
                    UPDATE user_sessions
                    SET total_requests = $1, total_processing_time = $2
                    WHERE id = $3
                    """,
                    total_requests,
                    total_processing_time,
                    session_id
                )

                # Проверяем, была ли обновлена хотя бы одна строка
                if rows_affected:
                    self.stats['queries_executed'] += 1
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to update user session {session_id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    # Методы для системных метрик

    async def store_system_metric(self, metric_type: str, value: float, metadata: Optional[Dict] = None) -> bool:
        """
        Сохранение системной метрики

        Args:
            metric_type: Тип метрики
            value: Значение метрики
            metadata: Метаданные

        Returns:
            bool: Успех операции
        """
        try:
            # Используем пул соединений asyncpg для более быстрой вставки
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO system_metrics (metric_type, value, metadata_json)
                    VALUES ($1, $2, $3)
                    """,
                    metric_type,
                    value,
                    metadata or {}
                )

                self.stats['queries_executed'] += 1
                return True

        except Exception as e:
            logger.error(f"Failed to store system metric {metric_type}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_system_metrics(self, metric_type: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Получение системных метрик

        Args:
            metric_type: Тип метрики
            hours: Период в часах

        Returns:
            List[Dict[str, Any]]: Список метрик
        """
        try:
            cutoff_time = datetime.utcnow().replace(hour=datetime.utcnow().hour - hours)

            async with self.get_session() as session:
                records = session.query(SystemMetrics)\
                    .filter(SystemMetrics.metric_type == metric_type)\
                    .filter(SystemMetrics.timestamp >= cutoff_time)\
                    .order_by(SystemMetrics.timestamp.desc())\
                    .all()

                metrics = []
                for record in records:
                    metrics.append({
                        'id': record.id,
                        'metric_type': record.metric_type,
                        'value': record.value,
                        'timestamp': record.timestamp,
                        'metadata': record.metadata_json
                    })

                self.stats['queries_executed'] += 1
                return metrics

        except Exception as e:
            logger.error(f"Failed to get system metrics {metric_type}: {e}")
            self.stats['errors_occurred'] += 1
            return []

    # Служебные методы

    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Получение статистики базы данных

        Returns:
            Dict[str, Any]: Статистика БД
        """
        try:
            async with self.pool.acquire() as conn:
                # Получение размеров таблиц
                table_sizes = await conn.fetch("""
                    SELECT
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """)

                # Общая статистика
                total_stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_tables,
                        pg_size_pretty(pg_database_size(current_database())) as db_size
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)

                return {
                    'total_tables': total_stats['total_tables'],
                    'database_size': total_stats['db_size'],
                    'table_sizes': [dict(row) for row in table_sizes],
                    'connection_stats': self.stats.copy()
                }

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}

    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики подключений

        Returns:
            Dict[str, Any]: Статистика подключений
        """
        return self.stats.copy()

    async def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Очистка старых данных

        Args:
            days_to_keep: Количество дней для хранения данных
        """
        try:
            cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - days_to_keep)

            async with self.get_session() as session:
                # Удаление старых опытов (кроме обработанных)
                deleted_experiences = session.query(ExperienceRecord)\
                    .filter(ExperienceRecord.timestamp < cutoff_date)\
                    .filter(ExperienceRecord.processed == False)\
                    .delete()

                # Удаление старых системных метрик
                deleted_metrics = session.query(SystemMetrics)\
                    .filter(SystemMetrics.timestamp < cutoff_date)\
                    .delete()

                session.commit()

                logger.info(f"Cleaned up {deleted_experiences} old experiences and {deleted_metrics} old metrics")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    # Алиасы для совместимости с тестами (PostgreSQL не предназначен для кэширования)

    async def get_cache(self, key: str) -> Optional[Any]:
        """Алиас для совместимости - PostgreSQL не кэширует"""
        logger.warning("PostgreSQL manager does not support caching operations")
        return None

    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Алиас для совместимости - PostgreSQL не кэширует"""
        logger.warning("PostgreSQL manager does not support caching operations")
        return False

    async def exists_cache(self, key: str) -> bool:
        """Алиас для совместимости"""
        return False

    async def delete_cache(self, key: str) -> bool:
        """Алиас для совместимости"""
        return False

    async def get_cache_ttl(self, key: str) -> Optional[int]:
        """Алиас для совместимости"""
        return None

    async def store_conversation_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Сохранение сообщения разговора (заглушка)"""
        # PostgreSQL может хранить разговоры, но для совместимости возвращаем False
        return False

    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Получение истории разговора (заглушка)"""
        return []

    async def get_conversation_length(self, session_id: str) -> int:
        """Получение длины разговора (заглушка)"""
        return 0

    async def clear_conversation(self, session_id: str) -> bool:
        """Очистка разговора (заглушка)"""
        return False

    async def store_metrics(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> bool:
        """Сохранение метрики"""
        return await self.store_system_metric(name, value, tags)

    async def get_metrics(self, name: str, timeframe: str = "1h") -> List[Dict[str, Any]]:
        """Получение метрик"""
        # Преобразование timeframe в часы
        hours_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
        hours = hours_map.get(timeframe, 1)
        return await self.get_system_metrics(name, hours)

    async def get_metrics_stats(self) -> Dict[str, Any]:
        """Статистика метрик"""
        return {
            'total_metrics': 0,  # Заглушка
            'metrics_types': [],
            'storage_size': 0
        }

    async def cleanup_expired_metrics(self) -> int:
        """Очистка устаревших метрик"""
        return 0

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Использование памяти (заглушка)"""
        return {}

    async def get_connection_info(self) -> Dict[str, Any]:
        """Информация о подключении"""
        return {
            'connected': self.pool is not None,
            'engine_type': 'postgresql'
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Общая статистика"""
        return self.get_connection_stats()

    async def batch_operations(self, keys: List[str]) -> Dict[str, Any]:
        """Пакетные операции (заглушка)"""
        return {}

    async def pipeline_operations(self, operations: List[tuple]) -> bool:
        """Пайплайн операций (заглушка)"""
        return False

    async def subscribe_to_channel(self, channel: str, callback) -> None:
        """Подписка на канал (заглушка)"""
        pass

    async def publish_to_channel(self, channel: str, message: Dict[str, Any]) -> bool:
        """Публикация в канал (заглушка)"""
        return False

    async def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Получение ключей по паттерну (заглушка)"""
        return []

    async def atomic_incr(self, key: str, amount: int = 1) -> int:
        """Атомарный инкремент (заглушка)"""
        return 0

    async def atomic_decr(self, key: str) -> int:
        """Атомарный декремент (заглушка)"""
        return 0

    async def get_counter(self, key: str) -> int:
        """Получение счетчика (заглушка)"""
        return 0

    async def acquire_lock(self, lock_key: str, ttl: int = 30) -> bool:
        """Получение блокировки (заглушка)"""
        return False

    async def release_lock(self, lock_key: str) -> bool:
        """Освобождение блокировки (заглушка)"""
        return False

    async def health_check(self) -> bool:
        """Проверка здоровья"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                    return True
            return False
        except Exception:
            return False

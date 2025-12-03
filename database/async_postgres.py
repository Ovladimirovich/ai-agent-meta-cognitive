"""
Async PostgreSQL менеджер с поддержкой пула соединений и асинхронных операций.
Заменяет текущую реализацию на полностью асинхронную с поддержкой пула соединений.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager
import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)

# SQLAlchemy асинхронные модели
class Base(DeclarativeBase):
    pass


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


class AsyncPostgreSQLManager:
    """
    Асинхронный менеджер PostgreSQL с пулом соединений.
    Полностью заменяет текущую синхронную реализацию.
    """

    def __init__(self, connection_string: str):
        """
        Инициализация асинхронного PostgreSQL менеджера

        Args:
            connection_string: Строка подключения к PostgreSQL
        """
        self.connection_string = connection_string
        self.engine = None
        self.async_session = None
        self.pool = None

        # Статистика
        self.stats = {
            'connections_created': 0,
            'queries_executed': 0,
            'errors_occurred': 0,
            'total_storage_size': 0,
            'active_connections': 0,
            'max_connections': 0
        }

        logger.info("AsyncPostgreSQLManager initialized")

    async def initialize(self):
        """Инициализация асинхронного подключения к БД"""
        try:
            # Создание асинхронного SQLAlchemy engine
            self.engine = create_async_engine(
                self.connection_string,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False  # Отключить логирование SQL
            )

            # Создание асинхронной сессии
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Создание таблиц
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("✅ Async PostgreSQL connection established with connection pooling")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Async PostgreSQL: {e}")
            raise

    async def close(self):
        """Закрытие асинхронного подключения к БД"""
        if self.engine:
            await self.engine.dispose()
        logger.info("Async PostgreSQL connection closed")

    @asynccontextmanager
    async def get_session(self):
        """Получение асинхронной сессии БД"""
        async with self.async_session() as session:
            try:
                yield session
            finally:
                await session.close()

    # Методы для работы с опытом агента

    async def store_experience(self, experience: Any) -> bool:
        """
        Сохранение опыта агента

        Args:
            experience: Опыт агента

        Returns:
            bool: Успех операции
        """
        try:
            async with self.get_session() as session:
                # Проверка существования
                result = await session.execute(
                    text("SELECT id FROM agent_experiences WHERE id = :id"),
                    {"id": experience.id}
                )
                existing = result.fetchone()
                
                if existing:
                    logger.debug(f"Experience {experience.id} already exists")
                    return True

                # Создание записи
                record = ExperienceRecord(
                    id=experience.id,
                    query=getattr(experience, 'query', ''),
                    result=getattr(experience, 'result', ''),
                    confidence=getattr(experience, 'confidence', 0.0),
                    execution_time=getattr(experience, 'execution_time', 0.0),
                    timestamp=getattr(experience, 'timestamp', datetime.utcnow()),
                    user_id=getattr(experience, 'user_id', None),
                    session_id=getattr(experience, 'session_id', None),
                    success_indicators=getattr(experience, 'success_indicators', []),
                    error_indicators=getattr(experience, 'error_indicators', []),
                    metadata_json=getattr(experience, 'metadata', {})
                )

                session.add(record)
                await session.commit()

                self.stats['queries_executed'] += 1
                logger.debug(f"Stored experience {experience.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store experience {experience.id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_experience(self, experience_id: str) -> Optional[Any]:
        """
        Получение опыта по ID

        Args:
            experience_id: ID опыта

        Returns:
            Optional[Any]: Опыт агента или None
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text("SELECT * FROM agent_experiences WHERE id = :id"),
                    {"id": experience_id}
                )
                record = result.fetchone()

                if not record:
                    return None

                # Преобразование в объект
                from agent.learning.models import AgentExperience
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

    async def get_recent_experiences(self, limit: int = 100, user_id: Optional[str] = None) -> List[Any]:
        """
        Получение недавних опытов

        Args:
            limit: Максимальное количество
            user_id: Фильтр по пользователю

        Returns:
            List[Any]: Список опытов
        """
        try:
            async with self.get_session() as session:
                query = "SELECT * FROM agent_experiences ORDER BY timestamp DESC LIMIT :limit"
                params = {"limit": limit}
                
                if user_id:
                    query = "SELECT * FROM agent_experiences WHERE user_id = :user_id ORDER BY timestamp DESC LIMIT :limit"
                    params["user_id"] = user_id

                result = await session.execute(text(query), params)
                records = result.fetchall()

                experiences = []
                for record in records:
                    from agent.learning.models import AgentExperience
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
                    experiences.append(experience)

                self.stats['queries_executed'] += 1
                return experiences

        except Exception as e:
            logger.error(f"Failed to get recent experiences: {e}")
            self.stats['errors_occurred'] += 1
            return []

    # Методы для работы с паттернами

    async def store_pattern(self, pattern: Any) -> bool:
        """
        Сохранение паттерна

        Args:
            pattern: Паттерн

        Returns:
            bool: Успех операции
        """
        try:
            async with self.get_session() as session:
                # Проверка существования
                result = await session.execute(
                    text("SELECT id FROM patterns WHERE id = :id"),
                    {"id": pattern.id}
                )
                existing = result.fetchone()
                
                if existing:
                    # Обновление существующего
                    await session.execute(
                        text("""
                            UPDATE patterns 
                            SET trigger_conditions = :trigger_conditions,
                                description = :description,
                                confidence = :confidence,
                                frequency = :frequency,
                                examples = :examples,
                                last_updated = :last_updated
                            WHERE id = :id
                        """),
                        {
                            "id": pattern.id,
                            "trigger_conditions": pattern.trigger_conditions,
                            "description": getattr(pattern, 'description', ''),
                            "confidence": pattern.confidence,
                            "frequency": getattr(pattern, 'frequency', 1.0),
                            "examples": getattr(pattern, 'examples', []),
                            "last_updated": datetime.utcnow()
                        }
                    )
                else:
                    # Создание нового
                    record = PatternRecord(
                        id=pattern.id,
                        type=getattr(pattern, 'type', 'unknown'),
                        trigger_conditions=pattern.trigger_conditions,
                        description=getattr(pattern, 'description', ''),
                        confidence=pattern.confidence,
                        frequency=getattr(pattern, 'frequency', 1.0),
                        examples=getattr(pattern, 'examples', []),
                        last_updated=datetime.utcnow()
                    )
                    session.add(record)

                await session.commit()

                self.stats['queries_executed'] += 1
                logger.debug(f"Stored pattern {pattern.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to store pattern {pattern.id}: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_pattern(self, pattern_id: str) -> Optional[Any]:
        """
        Получение паттерна по ID

        Args:
            pattern_id: ID паттерна

        Returns:
            Optional[Any]: Паттерн или None
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text("SELECT * FROM patterns WHERE id = :id"),
                    {"id": pattern_id}
                )
                record = result.fetchone()

                if not record:
                    return None

                # Преобразование в объект
                from agent.learning.models import Pattern
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

    async def get_patterns_by_type(self, pattern_type: str, limit: int = 100) -> List[Any]:
        """
        Получение паттернов по типу

        Args:
            pattern_type: Тип паттерна
            limit: Максимальное количество

        Returns:
            List[Any]: Список паттернов
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT * FROM patterns 
                        WHERE type = :type 
                        ORDER BY frequency DESC 
                        LIMIT :limit
                    """),
                    {"type": pattern_type, "limit": limit}
                )
                records = result.fetchall()

                patterns = []
                for record in records:
                    from agent.learning.models import Pattern
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
                    patterns.append(pattern)

                self.stats['queries_executed'] += 1
                return patterns

        except Exception as e:
            logger.error(f"Failed to get patterns by type {pattern_type}: {e}")
            self.stats['errors_occurred'] += 1
            return []

    # Методы для работы с результатами обучения

    async def store_learning_result(self, result: Any) -> bool:
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
                await session.commit()

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
                await session.commit()

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
            async with self.get_session() as session:
                result = await session.execute(
                    text("SELECT id FROM user_sessions WHERE id = :session_id"),
                    {"session_id": session_id}
                )
                record = result.fetchone()

                if record:
                    await session.execute(
                        text("""
                            UPDATE user_sessions 
                            SET total_requests = :total_requests,
                                total_processing_time = :total_processing_time
                            WHERE id = :session_id
                        """),
                        {
                            "session_id": session_id,
                            "total_requests": total_requests,
                            "total_processing_time": total_processing_time
                        }
                    )
                    await session.commit()

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
            async with self.get_session() as session:
                record = SystemMetrics(
                    metric_type=metric_type,
                    value=value,
                    metadata_json=metadata or {}
                )

                session.add(record)
                await session.commit()

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
                result = await session.execute(
                    text("""
                        SELECT id, metric_type, value, timestamp, metadata_json 
                        FROM system_metrics
                        WHERE metric_type = :metric_type
                          AND timestamp >= :cutoff_time
                        ORDER BY timestamp DESC
                    """),
                    {
                        "metric_type": metric_type,
                        "cutoff_time": cutoff_time
                    }
                )
                records = result.fetchall()

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
            async with self.get_session() as session:
                # Получение размеров таблиц
                result = await session.execute(text("""
                    SELECT
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """))
                table_sizes = [dict(row) for row in result.fetchall()]

                # Общая статистика
                result = await session.execute(text("""
                    SELECT
                        COUNT(*) as total_tables,
                        pg_size_pretty(pg_database_size(current_database())) as db_size
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                """))
                total_stats = result.fetchone()

                return {
                    'total_tables': total_stats['total_tables'],
                    'database_size': total_stats['db_size'],
                    'table_sizes': table_sizes,
                    'connection_stats': self.stats.copy(),
                    'pool_stats': {
                        'size': self.engine.pool.size,
                        'checkedout': self.engine.pool.checkedout(),
                        'overflow': self.engine.pool.overflow(),
                        'timeout': self.engine.pool.timeout
                    }
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
                await session.execute(
                    text("""
                        DELETE FROM agent_experiences
                        WHERE timestamp < :cutoff_date
                          AND processed = FALSE
                    """),
                    {"cutoff_date": cutoff_date}
                )

                # Удаление старых системных метрик
                await session.execute(
                    text("""
                        DELETE FROM system_metrics
                        WHERE timestamp < :cutoff_date
                    """),
                    {"cutoff_date": cutoff_date}
                )

                await session.commit()

                logger.info(f"Cleaned up old data older than {cutoff_date}")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def batch_store_experiences(self, experiences: List[Any]) -> bool:
        """
        Пакетное сохранение опытов для улучшения производительности

        Args:
            experiences: Список опытов для сохранения

        Returns:
            bool: Успех операции
        """
        try:
            async with self.get_session() as session:
                for experience in experiences:
                    # Проверка существования
                    result = await session.execute(
                        text("SELECT id FROM agent_experiences WHERE id = :id"),
                        {"id": experience.id}
                    )
                    existing = result.fetchone()
                    
                    if existing:
                        continue  # Пропускаем если уже существует

                    # Создание записи
                    record = ExperienceRecord(
                        id=experience.id,
                        query=getattr(experience, 'query', ''),
                        result=getattr(experience, 'result', ''),
                        confidence=getattr(experience, 'confidence', 0.0),
                        execution_time=getattr(experience, 'execution_time', 0.0),
                        timestamp=getattr(experience, 'timestamp', datetime.utcnow()),
                        user_id=getattr(experience, 'user_id', None),
                        session_id=getattr(experience, 'session_id', None),
                        success_indicators=getattr(experience, 'success_indicators', []),
                        error_indicators=getattr(experience, 'error_indicators', []),
                        metadata_json=getattr(experience, 'metadata', {})
                    )
                    session.add(record)

                await session.commit()
                self.stats['queries_executed'] += len(experiences)
                logger.debug(f"Batch stored {len(experiences)} experiences")
                return True

        except Exception as e:
            logger.error(f"Failed to batch store experiences: {e}")
            self.stats['errors_occurred'] += 1
            return False

    async def get_connection_info(self) -> Dict[str, Any]:
        """Информация о подключении"""
        return {
            'connected': self.engine is not None,
            'engine_type': 'postgresql_async',
            'pool_size': self.engine.pool.size if self.engine else 0,
            'pool_max_overflow': self.engine.pool._max_overflow if self.engine else 0
        }

    async def health_check(self) -> bool:
        """Проверка здоровья"""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception:
            return False
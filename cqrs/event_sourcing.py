"""
Event Sourcing система для CQRS паттерна
Хранение состояния системы как последовательности событий
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Callable, Awaitable, Type, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class DomainEvent:
    """Базовое доменное событие"""
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    event_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainEvent':
        """Создание из словаря"""
        data_copy = data.copy()
        data_copy['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data_copy)

class EventStore(ABC):
    """Абстрактный интерфейс хранилища событий"""

    @abstractmethod
    async def save_events(self, events: List[DomainEvent]) -> None:
        """Сохранение событий"""
        pass

    @abstractmethod
    async def get_events(self, aggregate_id: str, aggregate_type: str = None,
                        from_version: int = 0) -> List[DomainEvent]:
        """Получение событий для агрегата"""
        pass

    @abstractmethod
    async def get_all_events(self, from_timestamp: datetime = None) -> List[DomainEvent]:
        """Получение всех событий"""
        pass

class InMemoryEventStore(EventStore):
    """In-memory реализация хранилища событий"""

    def __init__(self):
        self.events: List[DomainEvent] = []
        self.lock = asyncio.Lock()

    async def save_events(self, events: List[DomainEvent]) -> None:
        """Сохранение событий в памяти"""
        async with self.lock:
            self.events.extend(events)
            logger.debug(f"Saved {len(events)} events to in-memory store")

    async def get_events(self, aggregate_id: str, aggregate_type: str = None,
                        from_version: int = 0) -> List[DomainEvent]:
        """Получение событий для агрегата"""
        async with self.lock:
            events = [e for e in self.events
                     if e.aggregate_id == aggregate_id
                     and (aggregate_type is None or e.aggregate_type == aggregate_type)
                     and e.version >= from_version]
            return sorted(events, key=lambda e: e.version)

    async def get_all_events(self, from_timestamp: datetime = None) -> List[DomainEvent]:
        """Получение всех событий"""
        async with self.lock:
            if from_timestamp:
                events = [e for e in self.events if e.timestamp >= from_timestamp]
            else:
                events = self.events.copy()
            return sorted(events, key=lambda e: e.timestamp)

class FileEventStore(EventStore):
    """File-based реализация хранилища событий"""

    def __init__(self, file_path: str = "events.log"):
        self.file_path = file_path
        self.lock = asyncio.Lock()
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Убеждаемся, что файл существует"""
        try:
            with open(self.file_path, 'a'):
                pass
        except Exception as e:
            logger.error(f"Failed to create event store file: {e}")

    async def save_events(self, events: List[DomainEvent]) -> None:
        """Сохранение событий в файл"""
        async with self.lock:
            try:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    for event in events:
                        json_line = json.dumps(event.to_dict(), ensure_ascii=False)
                        f.write(json_line + '\n')
                logger.debug(f"Saved {len(events)} events to file store")
            except Exception as e:
                logger.error(f"Failed to save events to file: {e}")
                raise

    async def get_events(self, aggregate_id: str, aggregate_type: str = None,
                        from_version: int = 0) -> List[DomainEvent]:
        """Получение событий для агрегата из файла"""
        async with self.lock:
            events = []
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            event = DomainEvent.from_dict(data)

                            if (event.aggregate_id == aggregate_id and
                                (aggregate_type is None or event.aggregate_type == aggregate_type) and
                                event.version >= from_version):
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error(f"Failed to read events from file: {e}")

            return sorted(events, key=lambda e: e.version)

    async def get_all_events(self, from_timestamp: datetime = None) -> List[DomainEvent]:
        """Получение всех событий из файла"""
        async with self.lock:
            events = []
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            event = DomainEvent.from_dict(data)

                            if from_timestamp is None or event.timestamp >= from_timestamp:
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error(f"Failed to read all events from file: {e}")

            return sorted(events, key=lambda e: e.timestamp)

class AggregateRoot(ABC):
    """Базовый класс агрегата"""

    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self.uncommitted_events: List[DomainEvent] = []
        self.event_handlers: Dict[str, Callable[['AggregateRoot', DomainEvent], None]] = {}

    def register_event_handler(self, event_type: str, handler: Callable[['AggregateRoot', DomainEvent], None]):
        """Регистрация обработчика событий"""
        self.event_handlers[event_type] = handler

    def apply_event(self, event: DomainEvent):
        """Применение события к агрегату"""
        # Обновляем версию
        if event.version > self.version:
            self.version = event.version

        # Вызываем обработчик события
        if event.event_type in self.event_handlers:
            self.event_handlers[event.event_type](self, event)

        # Добавляем в список непримененных событий
        self.uncommitted_events.append(event)

    def raise_event(self, event_type: str, event_data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Генерация нового события"""
        if metadata is None:
            metadata = {}

        event = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            aggregate_id=self.aggregate_id,
            aggregate_type=self.__class__.__name__,
            event_data=event_data,
            metadata=metadata,
            timestamp=datetime.now(),
            version=self.version + 1
        )

        self.apply_event(event)
        logger.debug(f"Event raised: {event_type} for aggregate {self.aggregate_id}")

    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Получение непримененных событий"""
        return self.uncommitted_events.copy()

    def clear_uncommitted_events(self):
        """Очистка списка непримененных событий"""
        self.uncommitted_events.clear()

    @classmethod
    async def load_from_events(cls, aggregate_id: str, event_store: EventStore) -> Optional['AggregateRoot']:
        """Загрузка агрегата из событий"""
        events = await event_store.get_events(aggregate_id, cls.__name__)

        if not events:
            return None

        # Создаем агрегат
        aggregate = cls(aggregate_id)

        # Применяем все события
        for event in events:
            aggregate.apply_event(event)

        # Очищаем непримененные события (они уже сохранены)
        aggregate.clear_uncommitted_events()

        logger.debug(f"Loaded aggregate {aggregate_id} from {len(events)} events")
        return aggregate

class EventPublisher:
    """Публикатор событий"""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[DomainEvent], Awaitable[None]]]] = {}
        self.lock = asyncio.Lock()

    async def subscribe(self, event_type: str, handler: Callable[[DomainEvent], Awaitable[None]]):
        """Подписка на события"""
        async with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
            logger.info(f"Subscribed handler for event type: {event_type}")

    async def publish(self, event: DomainEvent):
        """Публикация события"""
        async with self.lock:
            handlers = self.subscribers.get(event.event_type, []).copy()

        # Вызываем всех подписчиков
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._safe_call_handler(handler, event))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"Published event {event.event_type} to {len(handlers)} subscribers")

    async def _safe_call_handler(self, handler: Callable[[DomainEvent], Awaitable[None]], event: DomainEvent):
        """Безопасный вызов обработчика"""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Event handler failed for event {event.event_type}: {e}")

class Repository(Generic[T]):
    """Репозиторий для работы с агрегатами"""

    def __init__(self, event_store: EventStore, event_publisher: EventPublisher):
        self.event_store = event_store
        self.event_publisher = event_publisher

    async def save(self, aggregate: AggregateRoot) -> None:
        """Сохранение агрегата"""
        events = aggregate.get_uncommitted_events()

        if events:
            # Сохраняем события
            await self.event_store.save_events(events)

            # Публикуем события
            for event in events:
                await self.event_publisher.publish(event)

            # Очищаем непримененные события
            aggregate.clear_uncommitted_events()

            logger.debug(f"Saved aggregate {aggregate.aggregate_id} with {len(events)} events")

    async def load(self, aggregate_id: str, aggregate_class: Type[T]) -> Optional[T]:
        """Загрузка агрегата"""
        return await aggregate_class.load_from_events(aggregate_id, self.event_store)

    async def get_events(self, aggregate_id: str, aggregate_type: str = None,
                        from_version: int = 0) -> List[DomainEvent]:
        """Получение событий для агрегата"""
        return await self.event_store.get_events(aggregate_id, aggregate_type, from_version)

# Глобальные экземпляры
event_store = InMemoryEventStore()
event_publisher = EventPublisher()
repository = Repository(event_store, event_publisher)

# Примеры доменных событий для AI агента
@dataclass
class TaskCreatedEvent:
    """Событие создания задачи"""
    task_id: str
    task_type: str
    priority: int
    created_by: str

@dataclass
class TaskProcessedEvent:
    """Событие обработки задачи"""
    task_id: str
    result: Dict[str, Any]
    processing_time: float
    success: bool

@dataclass
class AgentStateChangedEvent:
    """Событие изменения состояния агента"""
    agent_id: str
    old_state: str
    new_state: str
    reason: str

@dataclass
class ModelInferenceCompletedEvent:
    """Событие завершения инференса модели"""
    model_name: str
    input_tokens: int
    output_tokens: int
    inference_time: float
    cost: float

# Пример агрегата
class AgentAggregate(AggregateRoot):
    """Агрегат агента"""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = "IDLE"
        self.tasks_processed = 0
        self.total_inference_time = 0.0
        self.last_activity = datetime.now()

        # Регистрируем обработчики событий
        self.register_event_handler("TaskProcessed", lambda agg, evt: agg._on_task_processed(evt))
        self.register_event_handler("AgentStateChanged", lambda agg, evt: agg._on_state_changed(evt))
        self.register_event_handler("ModelInferenceCompleted", lambda agg, evt: agg._on_inference_completed(evt))

    def _on_task_processed(self, event: DomainEvent):
        """Обработчик события обработки задачи"""
        self.tasks_processed += 1
        self.last_activity = event.timestamp

    def _on_state_changed(self, event: DomainEvent):
        """Обработчик события изменения состояния"""
        self.state = event.event_data.get("new_state", self.state)
        self.last_activity = event.timestamp

    def _on_inference_completed(self, event: DomainEvent):
        """Обработчик события завершения инференса"""
        self.total_inference_time += event.event_data.get("inference_time", 0)

    def process_task(self, task_id: str, result: Dict[str, Any], processing_time: float):
        """Обработка задачи"""
        event_data = {
            "task_id": task_id,
            "result": result,
            "processing_time": processing_time,
            "success": True
        }
        self.raise_event("TaskProcessed", event_data)

    def change_state(self, new_state: str, reason: str):
        """Изменение состояния"""
        event_data = {
            "old_state": self.state,
            "new_state": new_state,
            "reason": reason
        }
        self.raise_event("AgentStateChanged", event_data)

    def record_inference(self, model_name: str, input_tokens: int, output_tokens: int,
                        inference_time: float, cost: float):
        """Запись инференса модели"""
        event_data = {
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "inference_time": inference_time,
            "cost": cost
        }
        self.raise_event("ModelInferenceCompleted", event_data)

# Сервис для работы с event sourcing
class EventSourcingService:
    """Сервис для работы с event sourcing"""

    def __init__(self, repository: Repository):
        self.repository = repository

    async def create_agent(self, agent_id: str) -> AgentAggregate:
        """Создание нового агента"""
        agent = AgentAggregate(agent_id)
        await self.repository.save(agent)
        return agent

    async def get_agent(self, agent_id: str) -> Optional[AgentAggregate]:
        """Получение агента"""
        return await self.repository.load(agent_id, AgentAggregate)

    async def process_task(self, agent_id: str, task_id: str, result: Dict[str, Any], processing_time: float):
        """Обработка задачи через event sourcing"""
        agent = await self.get_agent(agent_id)
        if not agent:
            agent = await self.create_agent(agent_id)

        agent.process_task(task_id, result, processing_time)
        await self.repository.save(agent)

    async def change_agent_state(self, agent_id: str, new_state: str, reason: str):
        """Изменение состояния агента"""
        agent = await self.get_agent(agent_id)
        if not agent:
            agent = await self.create_agent(agent_id)

        agent.change_state(new_state, reason)
        await self.repository.save(agent)

    async def record_model_inference(self, agent_id: str, model_name: str, input_tokens: int,
                                   output_tokens: int, inference_time: float, cost: float):
        """Запись инференса модели"""
        agent = await self.get_agent(agent_id)
        if not agent:
            agent = await self.create_agent(agent_id)

        agent.record_inference(model_name, input_tokens, output_tokens, inference_time, cost)
        await self.repository.save(agent)

    async def get_agent_history(self, agent_id: str) -> List[DomainEvent]:
        """Получение истории агента"""
        return await self.repository.get_events(agent_id, "AgentAggregate")

    async def replay_agent_state(self, agent_id: str, up_to_version: int = None) -> Optional[AgentAggregate]:
        """Воспроизведение состояния агента до определенной версии"""
        events = await self.repository.get_events(agent_id, "AgentAggregate")
        if up_to_version:
            events = [e for e in events if e.version <= up_to_version]

        if not events:
            return None

        agent = AgentAggregate(agent_id)
        for event in events:
            agent.apply_event(event)

        agent.clear_uncommitted_events()
        return agent

# Глобальный сервис
event_sourcing_service = EventSourcingService(repository)

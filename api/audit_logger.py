"""
Audit Logging система для AI Агента
Обеспечивает структурированное логирование всех операций для compliance и security monitoring
"""

import logging
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor


class AuditEventType(Enum):
    """Типы audit событий"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    API_ACCESS = "api_access"
    DATA_ACCESS = "data_access"
    MODEL_INFERENCE = "model_inference"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_OPERATION = "system_operation"
    SECURITY_EVENT = "security_event"
    ERROR_EVENT = "error_event"
    LEARNING_EVENT = "learning_event"

class AuditEventSeverity(Enum):
    """Уровни серьезности audit событий"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Audit событие"""
    event_id: str
    event_type: AuditEventType
    severity: AuditEventSeverity
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: str  # Что было затронуто (эндпоинт, модель, etc.)
    action: str    # Что было сделано (create, read, update, delete, etc.)
    status: str    # Результат (success, failure, blocked, etc.)
    details: Dict[str, Any]  # Дополнительные детали
    metadata: Dict[str, Any]  # Метаданные для анализа

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Создание из словаря"""
        data_copy = data.copy()
        data_copy['event_type'] = AuditEventType(data['event_type'])
        data_copy['severity'] = AuditEventSeverity(data['severity'])
        data_copy['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data_copy)

class AuditLogger:
    """
    Асинхронный audit logger с буферизацией и ротацией логов
    """

    def __init__(self,
                 log_file: str = "logs/audit.log",
                 buffer_size: int = 100,
                 flush_interval: float = 30.0,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 10):
        """
        Инициализация audit logger

        Args:
            log_file: Путь к файлу логов
            buffer_size: Размер буфера перед записью
            flush_interval: Интервал сброса буфера (сек)
            max_file_size: Максимальный размер файла (bytes)
            backup_count: Количество резервных копий
        """
        self.log_file = Path(log_file)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_file_size = max_file_size
        self.backup_count = backup_count

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.buffer: List[AuditEvent] = []
        self.buffer_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Статистика
        self.stats = {
            'events_logged': 0,
            'events_buffered': 0,
            'flushes': 0,
            'errors': 0
        }

        # Фоновые задачи
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False

        self.logger = logging.getLogger('audit')
        self.logger.info(f"AuditLogger initialized with file: {log_file}")

    async def start(self):
        """Запуск audit logger"""
        self.running = True
        self.flush_task = asyncio.create_task(self._periodic_flush())
        self.logger.info("AuditLogger started")

    async def stop(self):
        """Остановка audit logger"""
        self.running = False
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass

        # Финальный сброс буфера
        await self._flush_buffer()
        self.executor.shutdown(wait=True)
        self.logger.info("AuditLogger stopped")

    async def log_event(self, event: AuditEvent):
        """
        Логирование audit события

        Args:
            event: Audit событие
        """
        async with self.buffer_lock:
            self.buffer.append(event)
            self.stats['events_buffered'] += 1

            # Сброс буфера если достигнут лимит
            if len(self.buffer) >= self.buffer_size:
                await self._flush_buffer()

    async def log(self,
                  event_type: AuditEventType,
                  severity: AuditEventSeverity,
                  resource: str,
                  action: str,
                  status: str,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  request_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Упрощенный метод логирования

        Args:
            event_type: Тип события
            severity: Серьезность
            resource: Ресурс
            action: Действие
            status: Статус
            user_id: ID пользователя
            session_id: ID сессии
            request_id: ID запроса
            ip_address: IP адрес
            user_agent: User agent
            details: Детали
            metadata: Метаданные
        """
        event_id = self._generate_event_id()

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            status=status,
            details=details or {},
            metadata=metadata or {}
        )

        await self.log_event(event)

    async def _periodic_flush(self):
        """Периодический сброс буфера"""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic flush: {e}")
                self.stats['errors'] += 1

    async def _flush_buffer(self):
        """Сброс буфера на диск"""
        if not self.buffer:
            return

        async with self.buffer_lock:
            events_to_write = self.buffer.copy()
            self.buffer.clear()

        try:
            # Проверяем размер файла и ротируем если нужно
            await self._rotate_log_if_needed()

            # Пишем события в файл
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._write_events_sync,
                events_to_write
            )

            self.stats['events_logged'] += len(events_to_write)
            self.stats['flushes'] += 1

            self.logger.debug(f"Flushed {len(events_to_write)} audit events")

        except Exception as e:
            self.logger.error(f"Failed to flush audit buffer: {e}")
            self.stats['errors'] += 1

            # Возвращаем события обратно в буфер
            async with self.buffer_lock:
                self.buffer.extend(events_to_write)

    def _write_events_sync(self, events: List[AuditEvent]):
        """Синхронная запись событий (выполняется в executor)"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            for event in events:
                json_line = json.dumps(event.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')

    async def _rotate_log_if_needed(self):
        """Ротирует лог файл если он слишком большой"""
        if not self.log_file.exists():
            return

        file_size = self.log_file.stat().st_size
        if file_size >= self.max_file_size:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._rotate_files_sync)

    def _rotate_files_sync(self):
        """Синхронная ротация файлов"""
        # Удаляем самый старый файл
        oldest_file = self.log_file.parent / f"{self.log_file.name}.{self.backup_count}"
        if oldest_file.exists():
            oldest_file.unlink()

        # Сдвигаем существующие файлы
        for i in range(self.backup_count - 1, 0, -1):
            src = self.log_file.parent / f"{self.log_file.name}.{i}"
            dst = self.log_file.parent / f"{self.log_file.name}.{i + 1}"
            if src.exists():
                src.rename(dst)

        # Переименовываем текущий файл
        backup_file = self.log_file.parent / f"{self.log_file.name}.1"
        self.log_file.rename(backup_file)

    def _generate_event_id(self) -> str:
        """Генерация уникального ID события"""
        timestamp = datetime.now().isoformat()
        random_part = hashlib.md5(f"{timestamp}{id(self)}".encode()).hexdigest()[:8]
        return f"audit_{int(datetime.now().timestamp())}_{random_part}"

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики audit logger"""
        return {
            'buffer_size': len(self.buffer),
            'log_file': str(self.log_file),
            'log_file_size': self.log_file.stat().st_size if self.log_file.exists() else 0,
            **self.stats
        }

    async def search_events(self,
                           event_type: Optional[AuditEventType] = None,
                           user_id: Optional[str] = None,
                           resource: Optional[str] = None,
                           status: Optional[str] = None,
                           limit: int = 100) -> List[AuditEvent]:
        """
        Поиск audit событий

        Args:
            event_type: Тип события
            user_id: ID пользователя
            resource: Ресурс
            status: Статус
            limit: Максимальное количество результатов

        Returns:
            Список найденных событий
        """
        # Это упрощенная реализация - в продакшене нужна БД
        events = []
        try:
            loop = asyncio.get_event_loop()
            events = await loop.run_in_executor(
                self.executor,
                self._search_events_sync,
                event_type,
                user_id,
                resource,
                status,
                limit
            )
        except Exception as e:
            self.logger.error(f"Failed to search audit events: {e}")

        return events

    def _search_events_sync(self,
                           event_type: Optional[AuditEventType],
                           user_id: Optional[str],
                           resource: Optional[str],
                           status: Optional[str],
                           limit: int) -> List[AuditEvent]:
        """Синхронный поиск событий"""
        events = []

        # Ищем в текущем файле и backup файлах
        log_files = [self.log_file] + [
            self.log_file.parent / f"{self.log_file.name}.{i}"
            for i in range(1, self.backup_count + 1)
        ]

        for log_file in log_files:
            if not log_file.exists():
                continue

            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(events) >= limit:
                            break

                        try:
                            data = json.loads(line.strip())
                            event = AuditEvent.from_dict(data)

                            # Применяем фильтры
                            if event_type and event.event_type != event_type:
                                continue
                            if user_id and event.user_id != user_id:
                                continue
                            if resource and event.resource != resource:
                                continue
                            if status and event.status != status:
                                continue

                            events.append(event)

                        except json.JSONDecodeError:
                            continue

            except Exception:
                continue

        return events

# Глобальный audit logger
audit_logger = AuditLogger()

# Вспомогательные функции для логирования типичных событий
async def log_api_access(request_id: str,
                        user_id: Optional[str],
                        session_id: Optional[str],
                        ip_address: str,
                        user_agent: str,
                        method: str,
                        endpoint: str,
                        status_code: int,
                        response_time: float,
                        details: Optional[Dict[str, Any]] = None):
    """Логирование доступа к API"""
    severity = AuditEventSeverity.HIGH if status_code >= 400 else AuditEventSeverity.LOW

    await audit_logger.log(
        event_type=AuditEventType.API_ACCESS,
        severity=severity,
        resource=endpoint,
        action=method,
        status="success" if status_code < 400 else "error",
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        ip_address=ip_address,
        user_agent=user_agent,
        details={
            "status_code": status_code,
            "response_time": response_time,
            **(details or {})
        },
        metadata={"api_version": "v1"}
    )

async def log_authentication(user_id: str,
                           ip_address: str,
                           user_agent: str,
                           success: bool,
                           method: str = "jwt",
                           details: Optional[Dict[str, Any]] = None):
    """Логирование аутентификации"""
    severity = AuditEventSeverity.CRITICAL if not success else AuditEventSeverity.MEDIUM

    await audit_logger.log(
        event_type=AuditEventType.AUTHENTICATION,
        severity=severity,
        resource="auth",
        action="login",
        status="success" if success else "failure",
        user_id=user_id if success else None,
        ip_address=ip_address,
        user_agent=user_agent,
        details={
            "method": method,
            "success": success,
            **(details or {})
        }
    )

async def log_model_inference(user_id: Optional[str],
                            session_id: Optional[str],
                            request_id: str,
                            model: str,
                            tokens_used: int,
                            cost: float,
                            success: bool,
                            details: Optional[Dict[str, Any]] = None):
    """Логирование использования модели"""
    await audit_logger.log(
        event_type=AuditEventType.MODEL_INFERENCE,
        severity=AuditEventSeverity.MEDIUM,
        resource=f"model:{model}",
        action="inference",
        status="success" if success else "error",
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        details={
            "model": model,
            "tokens_used": tokens_used,
            "cost": cost,
            "success": success,
            **(details or {})
        },
        metadata={"cost_usd": cost}
    )

async def log_security_event(event_type: str,
                           severity: AuditEventSeverity,
                           user_id: Optional[str],
                           ip_address: str,
                           details: Dict[str, Any]):
    """Логирование security события"""
    await audit_logger.log(
        event_type=AuditEventType.SECURITY_EVENT,
        severity=severity,
        resource="security",
        action=event_type,
        status="detected",
        user_id=user_id,
        ip_address=ip_address,
        details=details
    )

async def log_error_event(error_type: str,
                         user_id: Optional[str],
                         request_id: Optional[str],
                         ip_address: Optional[str],
                         details: Dict[str, Any]):
    """Логирование ошибки"""
    await audit_logger.log(
        event_type=AuditEventType.ERROR_EVENT,
        severity=AuditEventSeverity.HIGH,
        resource="system",
        action="error",
        status="error",
        user_id=user_id,
        request_id=request_id,
        ip_address=ip_address,
        details={
            "error_type": error_type,
            **details
        }
    )

# Middleware для автоматического логирования API доступа
class AuditMiddleware:
    """Middleware для audit логирования API запросов"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        import time
        from urllib.parse import unquote

        start_time = time.time()

        # Извлекаем информацию о запросе
        headers = dict(scope.get("headers", []))
        headers = {k.decode(): v.decode() for k, v in headers.items()}

        method = scope.get("method", "GET")
        path = unquote(scope.get("path", "/"))
        query_string = unquote(scope.get("query_string", b"").decode())

        ip_address = self._get_client_ip(headers)
        user_agent = headers.get("User-Agent", "Unknown")
        request_id = headers.get("X-Request-ID", f"req_{int(time.time())}")

        # Сохраняем в scope для использования в обработчиках
        scope["audit_info"] = {
            "ip_address": ip_address,
            "user_agent": user_agent,
            "request_id": request_id,
            "start_time": start_time
        }

        # Обертываем send для логирования ответа
        original_send = send

        async def audit_send(message):
            if message["type"] == "http.response.start":
                scope["audit_info"]["status_code"] = message["status"]
            elif message["type"] == "http.response.body":
                if "status_code" in scope["audit_info"]:
                    # Логируем завершенный запрос
                    response_time = time.time() - start_time
                    status_code = scope["audit_info"]["status_code"]

                    # Получаем user_id из scope если есть
                    user_id = getattr(scope.get("user"), "id", None) if "user" in scope else None
                    session_id = scope.get("session_id")

                    await log_api_access(
                        request_id=request_id,
                        user_id=user_id,
                        session_id=session_id,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        method=method,
                        endpoint=f"{path}?{query_string}" if query_string else path,
                        status_code=status_code,
                        response_time=response_time
                    )

            await original_send(message)

        await self.app(scope, receive, audit_send)

    def _get_client_ip(self, headers: Dict[str, str]) -> str:
        """Получение IP адреса клиента"""
        # Проверка X-Forwarded-For для прокси
        forwarded = headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Проверка X-Real-IP
        real_ip = headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback
        return "unknown"

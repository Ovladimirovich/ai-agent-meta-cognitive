import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from ..core.models import AgentState

logger = logging.getLogger("StateManager")

@dataclass
class StateTransitionResult:
    """Результат перехода состояния"""
    success: bool
    from_state: AgentState
    to_state: AgentState
    reason: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
            "error_message": self.error_message
        }


class InvalidStateTransitionError(Exception):
    """Ошибка недопустимого перехода состояния"""
    pass


class StateTransition:
    """Представляет переход между состояниями"""

    def __init__(self, from_state: AgentState, to_state: AgentState,
                 timestamp: datetime, reason: Optional[str] = None):
        self.from_state = from_state
        self.to_state = to_state
        self.timestamp = timestamp
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason
        }


class StateManager:
    """
    Менеджер состояний агента.

    Управляет переходами между состояниями агента с валидацией
    и ведением истории состояний для мета-познания.
    """

    def __init__(self):
        self.current_state = AgentState.IDLE
        self.state_history: List[StateTransition] = []
        self.state_entry_times: Dict[AgentState, datetime] = {}

        # Определение допустимых переходов
        self.state_transitions = {
            AgentState.IDLE: [AgentState.ANALYZING, AgentState.ERROR],
            AgentState.ANALYZING: [AgentState.EXECUTING, AgentState.ERROR, AgentState.IDLE],
            AgentState.EXECUTING: [AgentState.COMPLETED, AgentState.ERROR, AgentState.IDLE],
            AgentState.COMPLETED: [AgentState.IDLE, AgentState.ANALYZING],
            AgentState.ERROR: [AgentState.IDLE, AgentState.RECOVERY],
            AgentState.RECOVERY: [AgentState.IDLE, AgentState.ERROR]
        }

        # Инициализация времени входа в начальное состояние
        self.state_entry_times[self.current_state] = datetime.now()

        logger.info(f"🚀 StateManager initialized with state: {self.current_state.value}")

    def transition_to_safe(self, new_state: AgentState, reason: Optional[str] = None) -> StateTransitionResult:
        """
        Безопасный переход в новое состояние без исключений.

        Args:
            new_state: Новое состояние
            reason: Причина перехода (для логирования и анализа)

        Returns:
            StateTransitionResult: Результат перехода
        """
        from_state = self.current_state

        if new_state not in self.state_transitions.get(self.current_state, []):
            error_msg = f"Cannot transition from {self.current_state.value} to {new_state.value}"
            logger.error(f"❌ {error_msg}")

            # Автоматический переход в состояние ошибки
            self._force_error_state_safe(f"Invalid transition attempt: {error_msg}")

            return StateTransitionResult(
                success=False,
                from_state=from_state,
                to_state=new_state,
                reason=reason,
                error_message=error_msg
            )

        # Создаем запись о переходе
        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            timestamp=datetime.now(),
            reason=reason
        )

        # Обновляем время выхода из старого состояния
        if self.current_state in self.state_entry_times:
            exit_time = datetime.now()
            entry_time = self.state_entry_times[self.current_state]
            duration = (exit_time - entry_time).total_seconds()
            logger.debug(f"State {self.current_state.value} duration: {duration:.2f}s")

        # Выполняем переход
        old_state = self.current_state
        self.current_state = new_state
        self.state_history.append(transition)

        # Записываем время входа в новое состояние
        self.state_entry_times[new_state] = datetime.now()

        logger.info(f"✅ State transition: {old_state.value} → {new_state.value} ({reason or 'no reason'})")

        return StateTransitionResult(
            success=True,
            from_state=from_state,
            to_state=new_state,
            reason=reason
        )

    def transition_to(self, new_state: AgentState, reason: Optional[str] = None) -> bool:
        """
        Переход в новое состояние с валидацией (устаревший метод, используйте transition_to_safe).

        Args:
            new_state: Новое состояние
            reason: Причина перехода (для логирования и анализа)

        Returns:
            True если переход успешен, False если недопустим

        Raises:
            InvalidStateTransitionError: Если переход недопустим
        """
        result = self.transition_to_safe(new_state, reason)
        if not result.success:
            raise InvalidStateTransitionError(result.error_message)
        return result.success

    def _force_error_state_safe(self, reason: str):
        """Принудительный переход в состояние ошибки (без исключений)"""
        if self.current_state != AgentState.ERROR:
            # Пытаемся перейти в ERROR состояние безопасно
            result = self.transition_to_safe(AgentState.ERROR, reason)
            if not result.success:
                # Если даже в ERROR нельзя перейти, принудительно устанавливаем
                logger.error(f"❌ Cannot transition to ERROR state: {reason}")
                self.current_state = AgentState.ERROR
                self.state_entry_times[AgentState.ERROR] = datetime.now()

    def _force_error_state(self, reason: str):
        """Принудительный переход в состояние ошибки (для обработки исключений, устаревший)"""
        try:
            # Пытаемся перейти в ERROR состояние
            if self.current_state != AgentState.ERROR:
                self.transition_to(AgentState.ERROR, reason)
        except InvalidStateTransitionError:
            # Если даже в ERROR нельзя перейти, принудительно устанавливаем
            logger.error(f"❌ Cannot transition to ERROR state: {reason}")
            self.current_state = AgentState.ERROR
            self.state_entry_times[AgentState.ERROR] = datetime.now()

    def can_transition_to(self, target_state: AgentState) -> bool:
        """Проверка возможности перехода в указанное состояние"""
        return target_state in self.state_transitions.get(self.current_state, [])

    def get_available_transitions(self) -> List[AgentState]:
        """Получение списка доступных переходов из текущего состояния"""
        return self.state_transitions.get(self.current_state, [])

    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Получение истории переходов состояний.

        Args:
            limit: Максимальное количество записей (None для всех)

        Returns:
            Список переходов в виде словарей
        """
        history = [transition.to_dict() for transition in self.state_history]

        if limit:
            history = history[-limit:]

        return history

    def get_state_duration(self, state: AgentState) -> Optional[float]:
        """
        Получение времени нахождения в указанном состоянии.

        Returns:
            Время в секундах или None если состояние не найдено
        """
        if state not in self.state_entry_times:
            return None

        entry_time = self.state_entry_times[state]

        if state == self.current_state:
            # Текущее состояние - время от входа до сейчас
            return (datetime.now() - entry_time).total_seconds()
        else:
            # Прошлое состояние - нужно найти время выхода
            for transition in reversed(self.state_history):
                if transition.from_state == state:
                    return (transition.timestamp - entry_time).total_seconds()

        return None

    def get_state_statistics(self) -> Dict[str, Any]:
        """Получение статистики по состояниям"""
        stats = {
            "current_state": self.current_state.value,
            "total_transitions": len(self.state_history),
            "state_durations": {},
            "transition_counts": {}
        }

        # Подсчет продолжительности состояний
        for state in AgentState:
            duration = self.get_state_duration(state)
            if duration is not None:
                stats["state_durations"][state.value] = duration

        # Подсчет переходов
        for transition in self.state_history:
            key = f"{transition.from_state.value}→{transition.to_state.value}"
            stats["transition_counts"][key] = stats["transition_counts"].get(key, 0) + 1

        return stats

    def reset(self):
        """Сброс менеджера состояний"""
        self.current_state = AgentState.IDLE
        self.state_history.clear()
        self.state_entry_times.clear()
        self.state_entry_times[self.current_state] = datetime.now()

        logger.info("🔄 StateManager reset to IDLE state")

    def is_in_error_state(self) -> bool:
        """Проверка нахождения в состоянии ошибки"""
        return self.current_state == AgentState.ERROR

    def is_idle(self) -> bool:
        """Проверка нахождения в состоянии ожидания"""
        return self.current_state == AgentState.IDLE

    def is_processing(self) -> bool:
        """Проверка активной обработки"""
        return self.current_state in [AgentState.ANALYZING, AgentState.EXECUTING, AgentState.RECOVERY]

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
import time


class ToolResult(BaseModel):
    success: bool
    data: Any
    metadata: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


class Task(BaseModel):
    query: str
    context: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    async def execute(self, task: Task) -> ToolResult:
        pass

    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        pass

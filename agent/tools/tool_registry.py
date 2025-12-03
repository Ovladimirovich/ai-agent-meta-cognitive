from typing import Dict, Optional, List
from .base_tool import BaseTool, Task

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_available_tools(self) -> Dict[str, BaseTool]:
        return self._tools.copy()

    def get_tools_for_task(self, task: Task) -> List[BaseTool]:
        return [tool for tool in self._tools.values() if tool.can_handle(task)]

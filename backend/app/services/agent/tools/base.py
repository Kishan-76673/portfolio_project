from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    """Abstract base class for all agent tools."""
    name: str
    description: str

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Returns standard function declaration schema."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Safely executes the tool action with error handling."""
        pass

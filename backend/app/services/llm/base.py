from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', 'system', or 'tool'")
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class StreamChunk(BaseModel):
    delta: str = ""
    tool_call: Optional[Dict[str, Any]] = None
    is_done: bool = False

class BaseLLMProvider(ABC):
    """Abstract interface decoupling AI components from specific model vendors."""

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream assistant response tokens and structured tool calls."""
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate dense vector embedding for RAG indexing and query retrieval."""
        pass

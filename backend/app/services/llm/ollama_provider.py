import json
from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx
from app.services.llm.base import BaseLLMProvider, ChatMessage, StreamChunk

class OllamaProvider(BaseLLMProvider):
    """Local Ollama Provider optimized for low-latency streaming and zero cold starts."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

    async def warmup(self):
        """Pre-warms the Ollama model into VRAM/RAM so subsequent recruiter queries have zero cold-start delay."""
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": "24h",
                "options": {"num_predict": 1}
            }
            await self._client.post(url, json=payload, timeout=60.0)
            print(f"[Ollama Warmup] Model '{self.model}' is primed in memory with 24h keep-alive.")
        except Exception as e:
            print(f"[Ollama Warmup] Non-critical warning during warmup: {e}")

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        url = f"{self.base_url}/api/chat"

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for m in messages:
            formatted_messages.append({"role": m.role, "content": m.content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
            "keep_alive": "24h",  # Prevent Ollama from unloading model after 5m of idle time
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 450,
            }
        }
        if tools:
            payload["tools"] = tools

        try:
            async with self._client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    yield StreamChunk(delta=f"Ollama error ({response.status_code}): {err_text.decode('utf-8')}", is_done=True)
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield StreamChunk(delta=content)
                        if "tool_calls" in msg and msg["tool_calls"]:
                            for tc in msg["tool_calls"]:
                                yield StreamChunk(tool_call=tc.get("function", {}))
                        if data.get("done", False):
                            break
                    except Exception:
                        continue
        except httpx.ConnectError:
            yield StreamChunk(
                delta="[Ollama Offline] Ollama is not responding on http://localhost:11434. Please start it using 'ollama serve'.",
                is_done=True,
            )
            return

        yield StreamChunk(is_done=True)

    async def generate_embedding(self, text: str) -> List[float]:
        # Fast non-blocking embedding check with short 1.0s timeout
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            res = await self._client.post(url, json=payload, timeout=1.0)
            if res.status_code == 200:
                emb = res.json().get("embedding", [])
                if emb:
                    return emb
        except Exception:
            pass

        # Instant fallback vector
        return [0.0] * 768

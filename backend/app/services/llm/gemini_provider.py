import json
from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx
from app.services.llm.base import BaseLLMProvider, ChatMessage, StreamChunk

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API Provider for production deployment (₹0 free tier)."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        if not self.api_key:
            yield StreamChunk(
                delta="[Gemini Error] GEMINI_API_KEY is not configured in .env. Switch LLM_PROVIDER=ollama for local dev.",
                is_done=True,
            )
            return

        url = f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"

        contents = []
        for m in messages:
            role = "user" if m.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        payload: Dict[str, Any] = {"contents": contents}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]

        try:
            async with self._client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield StreamChunk(delta=f"Gemini API Error ({response.status_code}): {error_text.decode('utf-8')}", is_done=True)
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                            candidates = data.get("candidates", [])
                            if not candidates:
                                continue
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    yield StreamChunk(delta=part["text"])
                                if "functionCall" in part:
                                    yield StreamChunk(tool_call=part["functionCall"])
                        except Exception:
                            continue
        except Exception as e:
            yield StreamChunk(delta=f"Gemini Connection Error: {e}", is_done=True)
            return

        yield StreamChunk(is_done=True)

    async def generate_embedding(self, text: str) -> List[float]:
        if not self.api_key:
            return [0.0] * 768

        url = f"{self.base_url}/models/text-embedding-004:embedContent?key={self.api_key}"
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]},
        }
        try:
            res = await self._client.post(url, json=payload)
            if res.status_code == 200:
                return res.json().get("embedding", {}).get("values", [])
        except Exception:
            pass

        return [0.0] * 768

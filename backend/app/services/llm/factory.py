from app.core.config import Settings
from app.services.llm.base import BaseLLMProvider
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.ollama_provider import OllamaProvider

def get_llm_provider(settings: Settings) -> BaseLLMProvider:
    """Factory to cleanly resolve the configured LLM provider."""
    provider_name = settings.LLM_PROVIDER.lower().strip()

    if provider_name == "ollama":
        return OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
        )

    return GeminiProvider(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )

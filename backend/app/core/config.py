from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    PORT: int = 8001
    DEBUG: bool = True

    # LLM Settings
    LLM_PROVIDER: str = "ollama"  # "ollama" or "gemini"
    
    # Ollama Local Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:1b"

    # Gemini Production Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Paths
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    DATA_RAW_DIR: str = "./data/raw"
    SQLITE_DB_PATH: str = "./data/portfolio.db"

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

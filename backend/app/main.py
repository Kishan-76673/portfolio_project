from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.v1.chat import router as chat_router
from app.services.memory.session_store import SessionMemoryManager
from app.services.rag.vector_store import ChromaVectorStore
from app.services.llm.factory import get_llm_provider

from app.services.agent.orchestrator import AgentOrchestrator
import asyncio

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize SQLite session database
    memory = SessionMemoryManager(settings.SQLITE_DB_PATH)
    await memory.init_db()
    app.state.memory = memory
    print("[Startup] SQLite session memory initialized.")

    # Initialize / verify vector store index
    llm = get_llm_provider(settings)
    app.state.llm = llm

    store = ChromaVectorStore(settings.CHROMA_PERSIST_DIR, settings.DATA_RAW_DIR)
    app.state.vector_store = store

    try:
        await store.build_index(llm)
    except Exception as e:
        print(f"[Startup Warning] Vector indexing skipped during startup: {e}")

    # Pre-create orchestrator singleton
    app.state.orchestrator = AgentOrchestrator(llm, store)

    # Pre-warm local LLM in background so first user prompt has zero cold-start delay
    if hasattr(llm, "warmup"):
        asyncio.create_task(llm.warmup())

    yield
    print("[Shutdown] Releasing application resources.")

app = FastAPI(
    title="Robin's AI Portfolio Assistant API",
    version="1.0.0",
    description="Production-grade AI chatbot API for Robin's portfolio recruiters.",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1 routes
app.include_router(chat_router, prefix="/api/v1")

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "provider": settings.LLM_PROVIDER,
        "model": settings.OLLAMA_MODEL if settings.LLM_PROVIDER == "ollama" else settings.GEMINI_MODEL,
        "port": settings.PORT,
        "environment": settings.ENVIRONMENT,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)

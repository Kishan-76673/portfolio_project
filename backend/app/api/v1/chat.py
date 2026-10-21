import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.core.config import get_settings, Settings
from app.services.llm.factory import get_llm_provider
from app.services.rag.vector_store import ChromaVectorStore
from app.services.memory.session_store import SessionMemoryManager
from app.services.agent.orchestrator import AgentOrchestrator

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client UUID session identifier")
    message: str = Field(..., min_length=1, max_length=1000, description="Recruiter message")

@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Server-Sent Events (SSE) streaming chat endpoint (zero-allocation fast path)."""
    # Reuse singletons from app.state to avoid filesystem/SQLite locks on each request
    memory: SessionMemoryManager = getattr(request.app.state, "memory", None)
    if memory is None:
        memory = SessionMemoryManager(settings.SQLITE_DB_PATH)

    orchestrator: AgentOrchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        llm = get_llm_provider(settings)
        vector_store = ChromaVectorStore(settings.CHROMA_PERSIST_DIR, settings.DATA_RAW_DIR)
        orchestrator = AgentOrchestrator(llm, vector_store)

    # 1. Fetch conversational history (sliding window)
    history = await memory.get_history(req.session_id)

    async def event_generator():
        full_assistant_reply = []
        async for event in orchestrator.run_stream(req.message, history):
            event_type = event.get("type")
            if event_type == "token":
                delta = event.get("delta", "")
                full_assistant_reply.append(delta)
                yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"
            elif event_type in ("tool_start", "tool_end"):
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
            elif event_type == "done":
                yield f"event: done\ndata: {json.dumps({'session_id': req.session_id})}\n\n"

        # Save turn to SQLite session memory asynchronously
        await memory.add_message(req.session_id, "user", req.message)
        await memory.add_message(req.session_id, "assistant", "".join(full_assistant_reply))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Retrieve recent chat history for the session."""
    memory: SessionMemoryManager = getattr(request.app.state, "memory", None)
    if memory is None:
        memory = SessionMemoryManager(settings.SQLITE_DB_PATH)

    messages = await memory.get_history(session_id)
    return {
        "session_id": session_id,
        "messages": [{"role": m.role, "content": m.content} for m in messages]
    }

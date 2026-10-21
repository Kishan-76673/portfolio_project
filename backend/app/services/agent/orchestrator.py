import json
from typing import AsyncGenerator, Dict, Any, List
from app.services.llm.base import BaseLLMProvider, ChatMessage, StreamChunk
from app.services.rag.vector_store import ChromaVectorStore
from app.services.agent.tools.live_demo import LiveDemoVerifierTool

SYSTEM_GROUNDING_PROMPT = """
You are the AI Career Assistant for Robin's professional portfolio website.
Your role is to assist recruiters, engineering managers, and technical interviewers by providing accurate, grounded information about Robin.

CRITICAL GROUNDING RULES:
1. Answer ONLY using the facts from the provided <verified_knowledge> section or tool outputs.
2. If the user asks about a technology, company, credential, or experience NOT mentioned in the context:
   Do NOT invent or extrapolate. State clearly:
   "I don't have enough information in Robin's portfolio knowledge base to answer that accurately."
3. DO NOT hallucinate, invent previous employers, degrees, or years of experience.
4. Keep answers professional, concise, and structured with bullet points where appropriate.
5. If the user asks whether a project demo is live or working, call the `verify_live_demo` tool.
"""

class AgentOrchestrator:
    """Orchestrates RAG retrieval, tool calling, and grounded response streaming."""

    def __init__(self, llm_provider: BaseLLMProvider, vector_store: ChromaVectorStore):
        self.llm = llm_provider
        self.vector_store = vector_store
        self.live_tool = LiveDemoVerifierTool()
        self.tools = [self.live_tool]

    async def run_stream(
        self,
        user_message: str,
        history: List[ChatMessage],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # 1. Retrieve RAG context from verified knowledge base
        retrieved_chunks = await self.vector_store.search(user_message, self.llm, top_k=3)
        context_str = "\n\n---\n\n".join([c["content"] for c in retrieved_chunks])
        if not context_str:
            context_str = "No specific context retrieved. Rely only on general professional summary."

        system_instruction = f"{SYSTEM_GROUNDING_PROMPT}\n\n<verified_knowledge>\n{context_str}\n</verified_knowledge>"

        # Smart tool gating: only pass tool schemas if user asks about live demo / deployment status
        demo_keywords = {"live", "demo", "deploy", "deployed", "online", "up", "working", "status", "verify", "health"}
        user_lower = user_message.lower()
        should_use_tools = any(kw in user_lower for kw in demo_keywords)
        tool_schemas = [t.get_schema() for t in self.tools] if should_use_tools else None

        messages = history + [ChatMessage(role="user", content=user_message)]

        # 2. First pass: Call LLM with tools only when relevant
        pending_tool_call = None
        async for chunk in self.llm.stream_chat(messages, tools=tool_schemas, system_prompt=system_instruction):
            if chunk.tool_call:
                pending_tool_call = chunk.tool_call
            elif chunk.delta:
                yield {"type": "token", "delta": chunk.delta}

        # 3. If model requested a tool execution
        if pending_tool_call:
            fn_name = pending_tool_call.get("name")
            fn_args = pending_tool_call.get("args") or pending_tool_call.get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    fn_args = {}

            yield {"type": "tool_start", "tool": fn_name, "args": fn_args}

            tool_result = {}
            if fn_name == self.live_tool.name:
                tool_result = await self.live_tool.execute(**fn_args)

            yield {"type": "tool_end", "tool": fn_name, "result": tool_result}

            # Second pass: Feed tool result back to LLM for final grounded synthesis
            tool_msg = ChatMessage(
                role="user",
                content=f"[Tool Output for {fn_name}]: {json.dumps(tool_result)}. Summarize this verification cleanly for the recruiter."
            )
            async for chunk in self.llm.stream_chat(messages + [tool_msg], system_prompt=system_instruction):
                if chunk.delta:
                    yield {"type": "token", "delta": chunk.delta}

        yield {"type": "done"}

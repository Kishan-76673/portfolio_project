import json
from typing import AsyncGenerator, Dict, Any, List
from app.services.llm.base import BaseLLMProvider, ChatMessage, StreamChunk
from app.services.rag.vector_store import ChromaVectorStore
from app.services.agent.tools.live_demo import LiveDemoVerifierTool
from app.core.config import get_settings

SYSTEM_GROUNDING_PROMPT_TEMPLATE = """
You are an AI Career Assistant for {name}'s professional portfolio website.
Your ONLY job is to answer questions about {name} using the resume context provided to you.

ABSOLUTE RULES — follow these without exception:
1. SOURCE OF TRUTH: The [PORTFOLIO CONTEXT] in each user message is your ONLY source of information.
   Never use your own training knowledge about what {name} might know. If it's not in the context, you don't know it.
2. NAME: Always refer to the portfolio owner as "{name}". Never use any other name (not Robin, not Raza, not "the candidate").
3. NO HALLUCINATION: Do NOT invent technologies, job titles, companies, dates, projects, skills, or achievements.
   If something is not explicitly mentioned in the context, say: "{name}'s resume does not mention that."
4. NATURAL TONE: Answer in first-person-about-third-person style. Be concise, polished, and professional.
   Do NOT start with phrases like "According to the context" or "Based on the provided information".
5. FORMATTING: No markdown bold (**text**). Use plain text and simple bullet points (-) only.
6. LENGTH: Keep responses under 150 words. Be direct. Recruiters are busy.
7. OUTDATED INFO: If asked about something that may have changed (e.g. current role), note that the resume
   reflects the most recently uploaded version and may not capture the very latest updates.
"""

class AgentOrchestrator:
    """Orchestrates RAG retrieval, tool calling, and grounded response streaming."""

    def __init__(self, llm_provider: BaseLLMProvider, vector_store: ChromaVectorStore):
        self.llm = llm_provider
        self.vector_store = vector_store
        self.live_tool = LiveDemoVerifierTool()
        self.tools = [self.live_tool]
        # Resolve owner name at instance creation time (safe from hot-reload cache issues)
        try:
            self._owner_name = get_settings().PORTFOLIO_OWNER_NAME
        except Exception:
            self._owner_name = "Kishan"

    @property
    def owner_name(self) -> str:
        return self._owner_name

    async def run_stream(
        self,
        user_message: str,
        history: List[ChatMessage],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # 1. Retrieve RAG context from verified knowledge base
        retrieved_chunks = await self.vector_store.search(user_message, self.llm, top_k=3)
        context_str = "\n\n---\n\n".join([c["content"] for c in retrieved_chunks])
        name = self.owner_name
        if not context_str:
            context_str = f"No resume content found. The resume PDF may not have been indexed yet."

        system_instruction = SYSTEM_GROUNDING_PROMPT_TEMPLATE.format(name=name)

        # Smart tool gating: only pass tool schemas if user asks about live demo / deployment status
        demo_keywords = {"live", "demo", "deploy", "deployed", "online", "up", "working", "status", "verify", "health"}
        user_lower = user_message.lower()
        should_use_tools = any(kw in user_lower for kw in demo_keywords)
        tool_schemas = [t.get_schema() for t in self.tools] if should_use_tools else None

        # Anchor context directly in the active user turn so local 1B models pay full attention to it
        prompt_with_context = (
            f"[PORTFOLIO CONTEXT — {name.upper()}'s RESUME]\n"
            f"{context_str}\n\n"
            f"---\n"
            f"Recruiter Question: {user_message}\n\n"
            f"INSTRUCTION: Answer using ONLY the resume context above. "
            f"Refer to the owner as '{name}'. "
            f"If the answer is not in the context, say '{name}\'s resume does not mention that.' "
            f"Do not use any outside knowledge. Keep it under 150 words."
        )

        messages = history + [ChatMessage(role="user", content=prompt_with_context)]

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

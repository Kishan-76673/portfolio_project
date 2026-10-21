import asyncio
import time
import sys
import os

# Add backend to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.core.config import get_settings
from app.services.llm.factory import get_llm_provider
from app.services.rag.vector_store import ChromaVectorStore
from app.services.agent.orchestrator import AgentOrchestrator
from app.services.agent.tools.live_demo import LiveDemoVerifierTool

async def benchmark():
    print("=" * 60)
    print("ROBIN AI PORTFOLIO ASSISTANT PERFORMANCE BENCHMARK")
    print("=" * 60)

    settings = get_settings()
    llm = get_llm_provider(settings)
    store = ChromaVectorStore(settings.CHROMA_PERSIST_DIR, settings.DATA_RAW_DIR)

    # 1. Benchmark RAG Retrieval
    print("\n[1/3] Benchmarking RAG Retrieval...")
    test_queries = [
        "Show me recent work and portfolio projects",
        "Briefly tell me about your work experience",
        "What are your core technical skills in AI and Python?",
        "Explain the ATS project"
    ]
    rag_times = []
    for q in test_queries:
        t0 = time.perf_counter()
        results = await store.search(q, llm, top_k=3)
        dt = (time.perf_counter() - t0) * 1000
        rag_times.append(dt)
        print(f"  - Query: '{q[:35]}...' -> {len(results)} chunks in {dt:.3f} ms")

    avg_rag = sum(rag_times) / len(rag_times)
    print(f"  ==> Average RAG latency: {avg_rag:.3f} ms")

    # 2. Benchmark Tool Execution
    print("\n[2/3] Benchmarking Tool Execution (Live Demo Probe)...")
    tool = LiveDemoVerifierTool()
    t0 = time.perf_counter()
    tool_res = await tool.execute(project_slug="ats-screening")
    tool_dt = (time.perf_counter() - t0) * 1000
    print(f"  - Tool Result: {tool_res.get('status')}, code={tool_res.get('http_code')} in {tool_dt:.2f} ms")

    # 3. Benchmark End-to-End Streaming
    print("\n[3/3] Benchmarking End-to-End Orchestrator Stream...")
    orchestrator = AgentOrchestrator(llm, store)
    t_start = time.perf_counter()
    first_token_time = None
    token_count = 0

    print("  Streaming response: ", end="", flush=True)
    async for event in orchestrator.run_stream("What are Robin's skills?", []):
        if event.get("type") == "token":
            if first_token_time is None:
                first_token_time = (time.perf_counter() - t_start) * 1000
            token_count += 1
            if token_count <= 15:
                print(event.get("delta", ""), end="", flush=True)
    print("...")

    total_time = (time.perf_counter() - t_start) * 1000
    print(f"  - Time to First Token (TTFT): {first_token_time:.1f} ms" if first_token_time else "  - No tokens received")
    print(f"  - Total Stream Time: {total_time:.1f} ms ({token_count} tokens)")
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(benchmark())

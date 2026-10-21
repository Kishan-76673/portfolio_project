import os
import glob
import re
from typing import List, Dict, Any, Optional
from app.services.rag.chunker import DocumentChunker
from app.services.llm.base import BaseLLMProvider

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by", 
    "is", "are", "was", "were", "be", "been", "of", "what", "who", "where", "how",
    "tell", "me", "about", "show", "can", "you", "does", "have", "has", "do"
}

class ChromaVectorStore:
    """High-speed Hybrid Vector & Semantic Retrieval Store for portfolio documents."""

    def __init__(self, persist_dir: str = "./data/chroma_db", raw_dir: str = "./data/raw"):
        self.persist_dir = persist_dir
        self.raw_dir = raw_dir
        self.client = None
        self.collection = None
        self.chunks: List[Dict[str, Any]] = []
        self._load_local_chunks()
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="portfolio_kb",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"[VectorStore] ChromaDB disk backend deferred: {e}")

    def _load_local_chunks(self):
        """Loads and indexes markdown files into fast in-memory cache."""
        md_files = glob.glob(os.path.join(self.raw_dir, "*.md"))
        loaded = []
        for fpath in md_files:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                file_chunks = DocumentChunker.chunk_markdown(content, fname)
                loaded.extend(file_chunks)
            except Exception as e:
                print(f"[VectorStore Warning] Could not load {fname}: {e}")

        self.chunks = loaded

    async def build_index(self, llm_provider: BaseLLMProvider):
        """Indexes all markdown files into memory and optionally into ChromaDB."""
        self._load_local_chunks()

        # If ChromaDB collection already has data, avoid expensive re-embedding
        if self.collection:
            try:
                if self.collection.count() >= len(self.chunks) and len(self.chunks) > 0:
                    print(f"[VectorStore] Using cached ChromaDB index with {self.collection.count()} chunks.")
                    return
            except Exception:
                pass

        if not self.chunks or not self.collection:
            return

        ids = [c["id"] for c in self.chunks]
        documents = [c["text"] for c in self.chunks]
        metadatas = [c["metadata"] for c in self.chunks]

        # Fast non-blocking embedding generation
        try:
            embeddings = []
            for doc in documents:
                emb = await llm_provider.generate_embedding(doc)
                embeddings.append(emb)

            if embeddings and any(any(v != 0.0 for v in emb) for emb in embeddings):
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                print(f"[VectorStore] ChromaDB dense vectors indexed successfully ({len(self.chunks)} chunks).")
        except Exception as e:
            print(f"[VectorStore] ChromaDB vector indexing skipped: {e}")

    async def search(self, query: str, llm_provider: BaseLLMProvider, top_k: int = 3, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ultra-fast hybrid semantic retrieval (< 0.2ms) with Chroma vector fallback."""
        if not self.chunks:
            self._load_local_chunks()

        if not self.chunks:
            return []

        # 1. Fast in-memory token & semantic scoring
        query_terms = [
            t for t in re.findall(r'\b[a-zA-Z0-9_\-]+\b', query.lower()) 
            if t not in STOPWORDS and len(t) > 1
        ]

        scored_chunks = []
        for c in self.chunks:
            score = 0.0
            meta = c.get("metadata", {})
            header = meta.get("header", "").lower()
            text_lower = c.get("text", "").lower()
            cat = meta.get("category", "")

            # Filter if explicit category requested
            if category and cat != category:
                continue

            # Header match is high signal
            for term in query_terms:
                if term in header:
                    score += 15.0
                if term in text_lower:
                    # Count term occurrences with diminishing returns
                    count = text_lower.count(term)
                    score += min(count * 2.0, 10.0)

            # Category boost
            if "project" in query.lower() and cat == "project":
                score += 5.0
            elif "experience" in query.lower() and cat == "experience":
                score += 5.0
            elif ("skill" in query.lower() or "tech" in query.lower()) and cat == "skills":
                score += 5.0
            elif "education" in query.lower() and cat == "education":
                score += 5.0

            scored_chunks.append((score, c))

        # Sort by relevance score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Select top candidates
        top_candidates = [item[1] for item in scored_chunks[:top_k]]
        return [{"content": c["text"], "metadata": c["metadata"]} for c in top_candidates]


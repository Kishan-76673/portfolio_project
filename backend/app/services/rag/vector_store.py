import os
import glob
import re
from typing import List, Dict, Any, Optional
from app.services.rag.chunker import DocumentChunker
from app.services.llm.base import BaseLLMProvider

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "of", "what", "who", "where", "how",
    "tell", "me", "about", "show", "can", "you", "does", "have", "has", "do",
    "briefly", "give", "list", "describe", "explain", "your", "his", "her", "its"
}

# Maps query intent keywords → preferred chunk category → score boost
INTENT_ROUTING = [
    # (query_keywords, target_category, boost)
    (["experience", "work", "job", "role", "position", "employ", "company", "companies",
      "responsibility", "responsibilities", "past", "previous", "career"], "experience", 25.0),
    (["skill", "skills", "technology", "technologies", "tech", "stack", "tool", "tools",
      "know", "language", "framework", "proficient", "expertise", "competenc"], "skills", 25.0),
    (["project", "projects", "built", "build", "developed", "portfolio", "github"], "project", 25.0),
    (["education", "degree", "university", "college", "study", "studied", "graduate",
      "graduation", "gpa", "cgpa", "academic", "course", "qualification"], "education", 25.0),
    (["summary", "profile", "about", "overview", "who", "introduce", "background",
      "tell me about"], "general", 15.0),
]

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

    def _check_for_updates(self):
        """Auto-detects if any .md or .pdf files were added or modified in raw_dir."""
        current_files = glob.glob(os.path.join(self.raw_dir, "*.md")) + glob.glob(os.path.join(self.raw_dir, "*.pdf"))
        mod_times = tuple((f, os.path.getmtime(f)) for f in sorted(current_files) if os.path.exists(f))
        if mod_times != getattr(self, "_cached_mod_times", None):
            self._cached_mod_times = mod_times
            self._load_local_chunks()

    def _load_local_chunks(self):
        """Loads and indexes both markdown and PDF files into fast in-memory cache."""
        loaded = []

        # 1. Load Markdown files
        md_files = glob.glob(os.path.join(self.raw_dir, "*.md"))
        for fpath in md_files:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                file_chunks = DocumentChunker.chunk_markdown(content, fname)
                loaded.extend(file_chunks)
            except Exception as e:
                print(f"[VectorStore Warning] Could not load markdown {fname}: {e}")

        # 2. Load PDF files
        pdf_files = glob.glob(os.path.join(self.raw_dir, "*.pdf"))
        for fpath in pdf_files:
            fname = os.path.basename(fpath)
            try:
                pdf_chunks = DocumentChunker.chunk_pdf(fpath, fname)
                loaded.extend(pdf_chunks)
                if pdf_chunks:
                    print(f"[VectorStore] Indexed {len(pdf_chunks)} chunks from PDF '{fname}'")
            except Exception as e:
                print(f"[VectorStore Warning] Could not load PDF '{fname}': {e}")

        self.chunks = loaded

    def get_loaded_documents(self) -> List[Dict[str, Any]]:
        """Returns summary of all currently indexed markdown and PDF documents."""
        self._check_for_updates()
        docs_summary: Dict[str, int] = {}
        for c in self.chunks:
            src = c.get("metadata", {}).get("source", "unknown")
            docs_summary[src] = docs_summary.get(src, 0) + 1
        return [
            {"filename": fname, "chunks": count, "type": "pdf" if fname.lower().endswith(".pdf") else "markdown"}
            for fname, count in docs_summary.items()
        ]

    async def build_index(self, llm_provider: BaseLLMProvider):
        """Indexes all markdown and PDF files into memory and optionally into ChromaDB."""
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
        """Intent-aware hybrid retrieval: routes query to correct resume section first."""
        self._check_for_updates()

        if not self.chunks:
            return []

        q_lower = query.lower()

        # 1. Detect query intent — which resume section is being asked about?
        intent_category: Optional[str] = None
        intent_boost: float = 0.0
        for keywords, cat, boost in INTENT_ROUTING:
            if any(kw in q_lower for kw in keywords):
                if intent_category is None or boost > intent_boost:
                    intent_category = cat
                    intent_boost = boost

        # 2. Build query terms (minus stopwords)
        query_terms = [
            t for t in re.findall(r'\b[a-zA-Z0-9_\-]+\b', q_lower)
            if t not in STOPWORDS and len(t) > 1
        ]

        scored_chunks = []
        for c in self.chunks:
            score = 0.0
            meta = c.get("metadata", {})
            header = meta.get("header", "").lower()
            text_lower = c.get("text", "").lower()
            cat = meta.get("category", "")

            # Hard filter if explicit category override provided
            if category and cat != category:
                continue

            # Intent-based category boost — primary routing signal
            if intent_category and cat == intent_category:
                score += intent_boost

            # Header keyword match (very high signal — section headers are precise)
            for term in query_terms:
                if term in header:
                    score += 18.0
                if term in text_lower:
                    count = text_lower.count(term)
                    score += min(count * 2.0, 12.0)

            scored_chunks.append((score, c))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Return top_k — always return at least something even if score is 0
        top_candidates = [item[1] for item in scored_chunks[:top_k]]
        return [{"content": c["text"], "metadata": c["metadata"]} for c in top_candidates]


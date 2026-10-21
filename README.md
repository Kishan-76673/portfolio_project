# AI Resume & Portfolio Chatbot

A production-style, zero-cost (₹0) conversational AI assistant for personal portfolios and recruiters.

## Architecture Highlights
- **Backend**: FastAPI with Server-Sent Events (SSE) streaming
- **Frontend**: React (Vite) with responsive dark/light UI
- **AI Models**: Google Gemini 2.0 Flash (Production Free Tier) & Ollama (Local Dev)
- **RAG & Storage**: Header-aware semantic chunking, ChromaDB vector store, SQLite session memory
- **Agent Tools**: Playwright browser verification sandbox (SSRF protected)

---

## Quickstart Guide

### 1. Setup Backend (Python)
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
cp ../.env.example .env
```

Set your `GEMINI_API_KEY` in `.env` (or use `LLM_PROVIDER=ollama` for local testing).

Run the backend:
```bash
python -m app.main
```
The API starts at `http://localhost:8000`.

---

### 2. Setup Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

### 3. Knowledge Base Customization
Add or edit your own career data in:
- `data/raw/resume.md`
- `data/raw/projects.md`

On server restart, the vector store automatically chunks and indexes your updated documents!

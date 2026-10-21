# Featured Portfolio Projects

## Project: ATS / Resume Screening & Candidate Ranking Engine
- **Slug**: `ats-screening`
- **Category**: AI / Machine Learning & Backend
- **Live Demo URL**: https://example.com/demo/ats-screening
- **GitHub Repository**: https://github.com/alexsharma-dev/ats-screening-ai
- **Tech Stack**: Python, FastAPI, ChromaDB, Sentence Transformers, React, Docker
- **Overview**: An enterprise-grade AI Resume Screening and Job Matching platform that automates candidate profile evaluation against technical job descriptions.
- **Robin's Role & Architecture**:
  - Architected the end-to-end RAG and document ingestion pipeline using Python and FastAPI.
  - Implemented semantic chunking and dense vector embeddings in ChromaDB to compute multi-factor cosine similarity scores.
  - Developed strict grounding prompts to generate structured candidate evaluation summaries without hallucinating skills or qualifications.
  - Integrated automated scoring metrics (keyword alignment, technical depth, and seniority match).

## Project: AI Portfolio & Career Agent
- **Slug**: `ai-portfolio`
- **Category**: Agentic AI / Full-Stack
- **Live Demo URL**: https://example.com/demo/portfolio-bot
- **GitHub Repository**: https://github.com/alexsharma-dev/ai-portfolio-agent
- **Tech Stack**: Python, FastAPI, React, ChromaDB, Ollama, Google Gemini, Playwright
- **Overview**: A production-grade recruiter assistant built into a personal portfolio. It answers technical queries using verified resume chunks and uses Playwright to verify live deployment health.
- **Key Achievements**:
  - Created a pluggable LLM provider layer supporting local Ollama and production Gemini free tier.
  - Implemented headless Playwright browser verification with strict SSRF security controls.
  - Zero-budget ₹0 architecture with real-time SSE token streaming.

## Project: CloudMetrics Dashboard
- **Slug**: `cloud-metrics`
- **Category**: Cloud Systems / Web
- **Live Demo URL**: https://example.com/demo/cloud-metrics
- **GitHub Repository**: https://github.com/alexsharma-dev/cloud-metrics
- **Tech Stack**: TypeScript, React, Vite, FastAPI
- **Overview**: High-frequency telemetry dashboard tracking microservice health, memory usage, and latency metrics.

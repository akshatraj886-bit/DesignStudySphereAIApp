# StudySphere AI 🎓🤖

> **Intelligent Academic Study Partner & Factual PDF Q&A Platform**  
> High-precision, zero-hallucination Retrieval-Augmented Generation (RAG) platform tailored for college students and academic research.

[![React](https://img.shields.io/badge/React-19.0-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.0-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4.0-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🌟 Key Features

### 1. 📄 Genuine Document Ingestion & Dense Semantic Retrieval
- **Page-by-Page Extraction**: Powered by PyMuPDF (`pymupdf`) with page tracking and layout preservation.
- **Sliding-Window Semantic Chunking**: Overlapping character windows (500 chars with 100 char overlap) ensure technical terms and context boundaries are preserved.
- **Dense Vector Embeddings**: 384-dimensional dense vectors generated via `sentence-transformers/all-MiniLM-L6-v2`, with automatic CUDA GPU hardware acceleration and graceful CPU fallback.
- **Dual-Gate Relevance Verification**: Filters out irrelevant and out-of-scope student questions using combined cosine similarity thresholds and topical entity coverage, preventing hallucinated answers on unmentioned topics.

### 2. 🌐 Strict Question-Driven Language Matching
- **Automatic Multi-Language Intelligence**: Dynamically detects the language of the student's **current question**:
  - **English**: Structured, polished academic English.
  - **Hinglish (Roman Script Hindi)**: Natural conversational Hinglish with key technical terminology bolded in English.
  - **Devanagari Hindi (हिन्दी)**: Pure Devanagari Hindi with technical terms preserved.
- **Cross-Language Isolation**: Guarantees that document text language, previous chat turns, or template prompts **never** pollute or contaminate the response language.

### 3. 🛡️ Grounded Citations & Multi-Provider Generation
- **Exact Source Citations**: Every factual claim, formula, or concept is backed by explicit page citations (e.g., `[Page 2]`).
- **Multi-Provider Dispatch**:
  1. **Google Gemini Flash Lite**: Ultra-fast, cost-effective LLM generation (`gemini-flash-lite-latest` or `gemini-2.0-flash`).
  2. **Extractive Grounded Synthesizer**: ₹0 / zero-API offline fallback ensuring the system always works even without external keys or internet connection.
  3. **Local Ollama**: Supports offline local LLMs (e.g. `llama3`, `mistral`) running on `localhost:11434`.
- **Streaming Responses**: Real-time Server-Sent Events (SSE) token streaming with wall-clock deadline bounding and stall-protection.

### 4. 📚 4-Mode AI PDF Summarizer
- **Short**: Crisp, executive high-level summary.
- **Detailed**: Comprehensive multi-section chapter breakdown with definitions and mechanics.
- **Key Points**: High-yield bullet takeaways optimized for rapid exam revision.
- **Topic-Wise**: Structured topic sections with titles, narrative content, key points, and cited pages.

### 5. 🎯 Interactive Practice Quiz & Self-Testing
- Generates targeted multiple-choice, short answer, and conceptual questions categorized by difficulty (`easy`, `medium`, `hard`).
- Evaluates student submissions with real-time feedback and explanation citations.

### 6. 🔒 Multi-Tenant Privacy & Session Isolation
- Ephemeral client sessions isolated via `X-Session-Token`.
- Strict multi-tenant boundaries prevent documents, chats, or quiz histories from leaking across users.
- Dual-mode persistence: zero-setup local SQLite or durable Cloud PostgreSQL (Neon).

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│               Frontend (React 19 + Vite)               │
│      Tailwind CSS v4 • Lucide Icons • Vercel Host      │
└───────────────────────────┬────────────────────────────┘
                            │ REST API / SSE Streams
                            ▼
┌────────────────────────────────────────────────────────┐
│             FastAPI Backend (Render Host)              │
│ ┌──────────────────────┐      ┌──────────────────────┐ │
│ │  PDF Upload Service  │      │   RAG Orchestrator   │ │
│ │  (PyMuPDF Extraction)│      │(Dual-Gate Relevance) │ │
│ └──────────┬───────────┘      └──────────┬───────────┘ │
│            │                             │             │
│            ▼                             ▼             │
│   ┌─────────────────┐           ┌─────────────────┐    │
│   │ MiniLM-L6-v2    │           │ Gemini / Ollama │    │
│   │ 384d Embeddings │           │ / Extractive    │    │
│   └────────┬────────┘           └─────────────────┘    │
└────────────┼───────────────────────────────────────────┘
             │ Vector & Metadata Storage
             ▼
┌────────────────────────────────────────────────────────┐
│                   Persistence Layer                    │
│   • Local Dev: SQLite (`backend/data/studysphere.db`)  │
│   • Production: Neon Cloud PostgreSQL (SSL Required)   │
└────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Directory Structure

```text
DesignStudySphereAIApp/
├── .figma/                     # Figma Make integration metadata
├── .gitignore                  # Git ignore rules for node, python, env, sqlite
├── .mise.toml                  # Toolchain configuration (Node, pnpm)
├── AGENTS.md                   # Agent and IDE execution conventions
├── ARCHITECTURE.md             # Detailed engineering and architecture specs
├── DEMO_CHECKLIST.md           # Presentation, viva, and demo walkthrough checklist
├── PROJECT_REPORT.md           # Comprehensive technical project documentation
├── VIVA_QUESTIONS_AND_ANSWERS.md # Academic viva defense questions and model answers
├── combine_code.py             # Script to snapshot codebase for review
├── start-local.bat             # 1-Click launcher for Windows local development
├── index.html                  # Vite frontend HTML entrypoint
├── package.json                # Frontend dependencies and npm scripts
├── tsconfig.json               # TypeScript compiler options
├── vercel.json                 # Vercel deployment configuration
├── vite.config.ts              # Vite + Tailwind v4 + React configuration
│
├── src/                        # Frontend React 19 source code
│   ├── App.tsx                 # Main application UI layout & state
│   ├── main.tsx                # React DOM entrypoint
│   ├── index.css               # Global styles & Tailwind v4 theme
│   ├── components/             # Reusable UI components
│   │   ├── ChatPanel.tsx       # Real-time Q&A interface with source drawer
│   │   ├── DocumentViewer.tsx  # Interactive PDF preview with page navigation
│   │   ├── SummarizerModal.tsx # Multi-mode summarizer modal
│   │   ├── QuizGenerator.tsx   # Practice question generator and grading
│   │   └── ...
│   └── lib/                    # Client libraries
│       └── api.ts              # Typed API client for FastAPI endpoints
│
└── backend/                    # FastAPI Backend Service
    ├── Dockerfile              # Container deployment for Render/Cloud Run
    ├── requirements.txt        # Python dependency manifest
    ├── .env.example            # Backend environment template
    ├── app/
    │   ├── main.py             # FastAPI app, CORS, rate limits, routes
    │   ├── schemas.py          # Pydantic request/response models
    │   ├── pdf_service.py      # PyMuPDF document extraction & validation
    │   ├── rate_limiter.py     # Sliding-window rate limiter
    │   └── rag/                # Retrieval-Augmented Generation engine
    │       ├── chunker.py      # Sliding-window text chunker
    │       ├── embeddings.py   # all-MiniLM-L6-v2 vector generator
    │       ├── store.py        # Vector store & similarity search
    │       ├── retrieval.py    # Dual-gate retrieval and intent detector
    │       ├── generator.py    # Grounded answer synthesis & language engine
    │       ├── summarizer.py   # Grounded 4-mode document summarizer
    │       ├── question_generator.py # Quiz generator and evaluator
    │       └── db_adapter.py   # Unified SQLite / Neon Postgres adapter
    └── test_general_rag_suite.py # 21-point comprehensive RAG test suite
```

---

## 🚀 Getting Started (Local Development)

### Prerequisites
- **Node.js**: `v18.0+` (or `v20.0+`)
- **Python**: `3.10+` (Python `3.11` recommended)
- **Git**

---

### Method 1: 1-Click Launch (Windows)

Double click or execute the bundled script from the root directory:
```cmd
start-local.bat
```
This automatically launches both the FastAPI backend (`http://127.0.0.1:8000`) and the Vite frontend dev server (`http://localhost:8443`).

---

### Method 2: Manual Setup

#### Step 1: Clone the Repository
```bash
git clone https://github.com/akshatraj886-bit/DesignStudySphereAIApp.git
cd DesignStudySphereAIApp
```

#### Step 2: Configure & Start Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   *(Optional: Add your `GEMINI_API_KEY` in `.env` to enable Gemini generation; otherwise, the built-in offline Extractive Synthesizer will be used automatically).*
5. Start the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
   - **Backend API**: `http://127.0.0.1:8000`
   - **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
   - **Health Check**: `http://127.0.0.1:8000/api/health`

#### Step 3: Configure & Start Frontend

1. Open a new terminal at the project root:
   ```bash
   cd DesignStudySphereAIApp
   ```
2. Install frontend dependencies:
   ```bash
   npm install
   ```
3. Create frontend environment configuration:
   ```bash
   cp .env.example .env
   ```
   Verify that `VITE_API_BASE_URL=http://127.0.0.1:8000`.
4. Start the Vite dev server:
   ```bash
   npm run dev
   ```
5. Open your browser and navigate to `http://localhost:8443` (or `http://localhost:5173`).

---

## ⚙️ Environment Variables Reference

### Frontend Configuration (`.env`)
| Variable | Description | Default |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | Base URL of the FastAPI backend service | `http://127.0.0.1:8000` |

### Backend Configuration (`backend/.env`)
| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `ENVIRONMENT` | Environment mode (`development` or `production`) | `development` |
| `HOST` | Backend bind host | `127.0.0.1` |
| `PORT` | Backend bind port | `8000` |
| `ALLOWED_ORIGINS` | Comma-separated allowed CORS origins | `http://localhost:8443,http://localhost:5173,...` |
| `GEMINI_API_KEY` | Google AI Studio API Key | *(Optional; empty for offline fallback)* |
| `GEMINI_MODEL` | Gemini model name | `gemini-flash-lite-latest` |
| `GEMINI_TIMEOUT_SECONDS` | Request timeout for external LLM calls | `12.0` |
| `DATABASE_URL` | Cloud PostgreSQL URI with SSL (Neon) | *(Optional; empty for local SQLite)* |
| `RATE_LIMIT_ENABLED` | Enable in-memory sliding window rate limits | `true` |
| `MAX_UPLOAD_SIZE_MB`| Max allowed PDF upload size in MB | `25` |

---

## 🧪 Testing & Quality Assurance

The codebase includes an extensive automated evaluation benchmark covering retrieval recall, grounding faithfulness, cross-language matching, and streaming timeouts:

```bash
# Run the 21-dimension comprehensive RAG evaluation suite
cd backend
.\venv\Scripts\python.exe test_general_rag_suite.py
```

### Verified Benchmark Capabilities:
- ✅ **Test 1**: Whole-Document Summarization across multi-page notes.
- ✅ **Test 2**: Strict Multi-Document Session Isolation.
- ✅ **Test 3**: Factual Grounding & Explicit Page Citations `[Page X]`.
- ✅ **Test 4**: Concept Explanations & Technical Precision.
- ✅ **Test 5**: Multi-Part Questions & Subheading Organization.
- ✅ **Test 6**: Hindi & Hinglish Queries on English Notes.
- ✅ **Test 7**: Unrelated Question Rejection (`insufficient_evidence`).
- ✅ **Test 8**: Page-Specific Targeted Question Answering.
- ✅ **Test 9**: Complex Analytical & Comparative Queries.
- ✅ **Test 10**: Conversational Pronoun & Follow-Up Coreference Resolution.
- ✅ **Test 11**: Prompt Injection & Adversarial Attack Resistance.
- ✅ **Test 12**: Empty / Scanned PDF Error Handling.
- ✅ **Test 13**: High-Concurrency Load Simulation.
- ✅ **Test 14**: Citation Integrity & Non-Hallucination.
- ✅ **Test 15**: Vector Score Transparency & Calibration.
- ✅ **Test 16**: Multi-Mode Summarizer Engine (Short, Detailed, Key Points, Topic-Wise).
- ✅ **Test 17**: Question Generator & Quiz Evaluator.
- ✅ **Test 18**: Server-Sent Events (SSE) Token Streaming.
- ✅ **Test 19**: Gemini Request Timeouts & Graceful Fallbacks.
- ✅ **Test 20**: Streaming Stalled Connection & Deadline Safeguards.
- ✅ **Test 21**: Cross-Language Robustness & Strict Question-Driven Language Matching.

---

## 🚢 Deployment Guide

- **Frontend**: Deployed on [Vercel](https://vercel.com/) with SPA rewrite rules configured in `vercel.json`. Set `VITE_API_BASE_URL` to your Render backend URL.
- **Backend**: Deployed on [Render](https://render.com/) using the provided [Dockerfile](file:///c:/Users/aksha/Desktop/DesignStudySphereAIApp/backend/Dockerfile). Set environment variables in Render Dashboard (`GEMINI_API_KEY`, `GEMINI_MODEL=gemini-flash-lite-latest`, `DATABASE_URL`).
- **Database**: Hosted on [Neon](https://neon.tech/) PostgreSQL free tier with automated SSL encryption and persistent storage.

---

## 👥 Collaboration Guidelines

1. **Branching Strategy**: Create feature branches from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Never Commit Secrets**:
   - Keep `.env` files untracked. All secrets must stay in `.env` and never be committed to Git.
3. **Run Regression Tests Before Pushing**:
   - Ensure the evaluation suite passes cleanly:
     ```bash
     python backend/test_general_rag_suite.py
     ```
4. **Pull Requests**:
   - Write clear, concise commit messages explaining the rationale for modifications.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).

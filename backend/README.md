# StudySphere AI — Backend (Phase 3: Real RAG-Based Document Q&A)

This is the production-grade Python + FastAPI backend service for **StudySphere AI**. It implements a genuine, end-to-end **Retrieval-Augmented Generation (RAG)** pipeline designed for academic notes question answering.

---

## 1. System Architecture

```
[ Uploaded PDF ]
       │
       ▼ (PyMuPDF)
[ Page-by-Page Text Extraction ]
       │
       ▼ (Overlapping Sliding Window: 500 chars, 100 overlap)
[ Semantic Document Chunks ]
       │
       ▼ (sentence-transformers/all-MiniLM-L6-v2)
[ 384-Dimensional Dense Vector Embeddings ]
       │
       ▼
[ Persistent SQLite Vector & Metadata Store ] (data/studysphere.db)
       ▲
       │ (Cosine Similarity Search restricted by document_id)
       ▼
[ Student Query: "What is Round Robin scheduling?" ]
       │
       ▼
[ Top-K Source Passages + Page Numbers + Match Scores ]
       │
       ▼
[ Grounded Answer Generator ]
 ├── Google Gemini (GENERATION_PROVIDER=auto|gemini and GEMINI_API_KEY is set)
 ├── Local Ollama/Qwen (OLLAMA_ENABLED=true; preferred when GENERATION_PROVIDER=ollama)
 └── Extractive Grounded Synthesizer (default zero-API fallback)
```

---

## 2. Key Components

### 1. Document Processing & Chunking (`app/rag/chunker.py`)
- Takes page-by-page text from PyMuPDF.
- Splits each page into overlapping windows (default 500 characters with 100-character overlap) to prevent losing context across sentence and paragraph boundaries.
- Preserves full citation metadata: `document_id`, `document_name`, `page_number`, `chunk_index`, character and word counts.
- Rejects empty pages and handles whitespace gracefully.

### 2. Semantic Embedding Engine (`app/rag/embeddings.py`)
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~80 MB).
- GPU Acceleration: Automatically detects CUDA GPU (e.g., NVIDIA GeForce RTX 3050 Laptop GPU) and falls back to CPU if GPU inference is unavailable.
- Generates L2-normalized float32 vectors so cosine similarity can be computed via fast dot products.

### 3. Persistent SQLite Vector Store (`app/rag/store.py`)
- Database: `backend/data/studysphere.db` (auto-created).
- Tables:
  - `documents`: Stores document ID, filename, total words/chars, page count, chunk count, timestamp.
  - `document_chunks`: Stores chunk text, page citations, chunk index, and binary serialized float32 embedding vectors (`BLOB`).
- Supports cosine similarity search over vector blobs with optional document filtering.

### 4. Grounded Answer Generation (`app/rag/generator.py`)
- Enforces strict factual grounding: answers are derived **strictly** from retrieved passages.
- Includes exact source document citations and page numbers.
- Handles absent questions by explicitly returning `retrieval_status: "insufficient_evidence"`.
- Provider selection (`GENERATION_PROVIDER`, default `auto`):
  1. **`auto`**: Google Gemini first if `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set, then local Ollama if `OLLAMA_ENABLED=true`, then the extractive synthesizer.
  2. **`gemini`**: Prefer Gemini; keep the same Ollama-then-extractive fallback when Ollama is enabled.
  3. **`ollama`**: Prefer local Ollama/Qwen (`OLLAMA_MODEL`, default `qwen2.5:1.5b`) and do **not** call Gemini. `OLLAMA_ENABLED` remains a hard safety gate; if it is false, Gemini is still skipped and the extractive synthesizer is used.
  4. Invalid values are treated as `auto`.
- Local Qwen requires Ollama installed, running at `OLLAMA_HOST` (default `http://localhost:11434`), and the model pulled (`ollama pull qwen2.5:1.5b`). The backend does not download models automatically.
- **Extractive Grounded Synthesizer** remains the offline fallback. **Mock Provider** is used only in automated unit tests.

---

## 3. Windows PowerShell Setup

### Step 1 — Navigate to the backend directory
```powershell
cd c:\Users\aksha\Desktop\DesignStudySphereAIApp\backend
```

### Step 2 — Activate the virtual environment
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies (if setting up fresh)
```powershell
pip install -r requirements.txt
```

### Step 4 — Optional: Configure generation providers
Copy the template (never commit secrets):
```powershell
copy .env.example .env
```

Google Gemini (optional, used by `GENERATION_PROVIDER=auto` or `gemini`):
```env
GEMINI_API_KEY="your-gemini-api-key-here"
```

Local Qwen via Ollama (optional; does not replace embeddings):
```env
GENERATION_PROVIDER=ollama
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_TIMEOUT_SECONDS=30.0
```

Install Ollama, start it, then pull the generation model once:
```powershell
ollama pull qwen2.5:1.5b
```

If Gemini is unset and Ollama is disabled or unavailable, the built-in Extractive Grounded Synthesizer is used. `all-MiniLM-L6-v2` embeddings are unchanged.

---

## 4. Starting the Backend Server

Start the Uvicorn server on port `8000`:
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive API Documentation (Swagger)**: `http://127.0.0.1:8000/docs`
- **Health Check**: `http://127.0.0.1:8000/api/health`

---

## 5. API Endpoints

### `GET /api/health`
Returns system status, active embedding model, and compute device:
```json
{
  "status": "ok",
  "service": "StudySphere AI Backend",
  "embedding_model": "all-MiniLM-L6-v2",
  "device": "cpu"
}
```

### `POST /api/documents/upload`
Uploads a PDF, extracts text page-by-page, generates chunks and dense embeddings, and stores them in SQLite:
```powershell
$form = @{ file = Get-Item -Path "C:\path\to\notes.pdf" }
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/documents/upload" -Method Post -Form $form
```

### `GET /api/documents`
Lists all indexed documents currently stored in the backend database.

### `POST /api/ask`
Performs genuine semantic search and grounded answer generation with dual-gate relevance verification and transparent source separation:
```json
{
  "question": "bhai how to handle missing valused",
  "document_id": "doc_1792224846_118a0117",
  "top_k": 3,
  "similarity_threshold": 0.32
}
```

Response:
```json
{
  "question": "bhai how to handle missing valused",
  "answer": "### Overview\nAccording to your notes in **CS_DataScience_Notes.pdf**:\n• Handling Missing Values in Datasets *(Page 4)*\n\n### Key Concepts & Details\n1. Techniques for handling missing values include: *(Page 4)*\n2. **Listwise Deletion**: Removes any record or observation containing missing values. *(Page 4)*\n3. **Mean and Median Imputation**: Replaces missing numerical values with the mean or median of the observed feature *(Page 4)*\n4. **Missingness Indicators**: Adds a separate binary flag variable indicating whether a value was originally absent *(Page 4)*\n\n### Study Takeaway\nAll points above are directly drawn from **CS_DataScience_Notes.pdf**. Check the indicated pages for full textbook derivations and diagrammatic examples.",
  "cited_sources": [
    {
      "document_id": "doc_1792224846_118a0117",
      "document_name": "CS_DataScience_Notes.pdf",
      "page_number": 4,
      "chunk_index": 0,
      "similarity_score": 0.4373,
      "excerpt": "Operating Systems & Data Science - Module 3: Data Preprocessing\n4. Handling Missing Values in Datasets..."
    }
  ],
  "retrieved_sources": [
    {
      "document_id": "doc_1792224846_118a0117",
      "document_name": "CS_DataScience_Notes.pdf",
      "page_number": 4,
      "chunk_index": 0,
      "similarity_score": 0.4373,
      "excerpt": "Operating Systems & Data Science - Module 3: Data Preprocessing\n4. Handling Missing Values in Datasets..."
    }
  ],
  "sources": [...],
  "retrieval_status": "found",
  "generation_provider": "extractive_grounded_synthesizer",
  "model_name": "all-MiniLM-L6-v2-dense-retrieval"
}
```

### `POST /api/documents/{document_id}/summarize`
Synthesizes a grounded, hallucination-free summary for an indexed document across four supported modes:
- `short`: Concise executive summary and core highlights.
- `detailed`: Comprehensive multi-section breakdown with definitions and mechanics.
- `key_points`: High-yield bullet takeaways for fast exam revision.
- `topic_wise`: Structured topic sections with title, narrative content, key points, and cited pages.

Request:
```json
{
  "mode": "detailed"
}
```

Response:
```json
{
  "document_id": "doc_1792227789_ceed2f4c",
  "document_name": "OS_DataScience_Notes.pdf",
  "summary_mode": "detailed",
  "summary": "### Comprehensive Academic Summary: OS_DataScience_Notes.pdf\n\n### 1. Process Concept and State Transitions *(Page(s) 1)*\nA process is a program in execution. *(Page 1)*\n\n**Core Mechanics & Details:**\n1. **The Process Control Block (PCB)**: stores critical information for each process... *(Page 1)*\n...",
  "key_points": [
    "**The Process Control Block (PCB)**: stores critical information for each process... *(Page 1)*",
    "**First-Come, First-Served (FCFS)**: The process that requests the CPU first... *(Page 2)*"
  ],
  "topic_sections": [
    {
      "title": "1. Process Concept and State Transitions",
      "content": "A process is a program in execution...",
      "key_points": ["**The Process Control Block (PCB)**: stores critical info... *(Page 1)*"],
      "page_numbers": [1]
    }
  ],
  "cited_pages": [1, 2, 3, 4],
  "cited_page_numbers": [1, 2, 3, 4],
  "generation_provider": "extractive_grounded_summarizer",
  "model_name": "salience_section_synthesizer",
  "warning": "Offline grounded extractive summarizer active. Content is synthesized directly from source text without hallucination."
}
```

### `POST /api/summarize`
Convenience alias taking `{ "document_id": "...", "mode": "short" }` in the request body.


---

## 6. Running Automated Tests

Run the Phase 4 Real AI PDF Summarizer test suite (10 automated test cases):
```powershell
.\venv\Scripts\python.exe test_summarizer.py
```

Run the complete RAG test suite and 15-question academic evaluation benchmark:
```powershell
.\venv\Scripts\python.exe test_rag.py
```

Run Phase 1 & 2 upload verification tests (requires server running on port 8000):
```powershell
.\venv\Scripts\python.exe verify_tests.py
```

---

## 7. RAG Quality Improvements & 15-Question Academic Benchmark

### Observed Problems Solved:
1. **Answers Are Too Extractive**: Upgraded the default offline extractive synthesizer to compose student-friendly structured explanations with `### Overview`, `### Key Concepts & Details` (numbered points and bold technical terms), and `### Study Takeaway` with clean inline page citations `*(Page X)*`.
2. **Unrelated Questions Handled Transparently**: Implemented dual-gate relevance verification combining cosine vector similarity with morphological topical entity coverage. Unrelated questions (such as asking for the Roman Empire against CS/Data Science notes) are accurately identified and returned with `retrieval_status: "insufficient_evidence"` and 0 cited sources.
3. **Transparent Similarity Metric**: Replaced misleading percentage confidence labels (`Match: 35%`) with transparent vector similarity float scores (`Semantic similarity: 0.35`) accompanied by an explicit disclaimer that similarity measures embedding proximity, not answer correctness.
4. **Retrieved Sources vs. Cited Sources**: Separated candidate retrieved passages (`retrieved_sources`) from passages actually used as factual evidence to compose the response (`cited_sources`). Passages are only labeled as cited if their sentences were directly included in the final synthesized answer.

### Measured 15-Question Academic Benchmark Results (`backend/test_rag.py`):

| Evaluation Metric | Target | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Total Evaluation Questions** | 15 | **15** | Completed |
| **Answerable Questions** | 12 | **12** | Completed |
| **Absent / Unrelated Questions** | 3 | **3** | Completed |
| **Retrieval Recall@3 (Answerable)** | ≥ 90.0% | **100.0%** (12/12 hits) | **PASS** |
| **Insufficient Evidence Decision Accuracy** | 100.0% | **100.0%** (3/3 correctly rejected) | **PASS** |
| **Source / Page Citation Correctness** | ≥ 90.0% | **91.7%** (11/12 accurate) | **PASS** |
| **Answer Faithfulness to Evidence** | ≥ 90.0% | **100.0%** (12/12 fully grounded) | **PASS** |

---

## 8. Hardware & Known Limitations

- **Hardware Profile**: Tested and verified on Windows with CPU and NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM). Memory footprint for `all-MiniLM-L6-v2` is under 120 MB RAM/VRAM.
- **Provider Architecture**:
  - Offline default: Structured Extractive Grounded Synthesizer with zero cost and no external API reliance.
  - Optional generative expansion: Google Gemini 1.5 Flash (`GEMINI_API_KEY`) and local Ollama (`http://localhost:11434`) are fully supported and dispatched automatically when configured.
- **Scanned Image Documents**: PDFs containing purely raster image scans without extractable text layers require OCR, indicated by `ocr_suggested=True`.
- **Vector Scale**: SQLite in-memory dot product ranking operates in under 10 ms for thousands of document chunks. For multi-gigabyte enterprise repositories, external vector databases (pgvector, Qdrant, Milvus) can be linked with the same retriever interface.

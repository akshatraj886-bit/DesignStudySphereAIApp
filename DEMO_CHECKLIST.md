# StudySphere AI — College Capstone Demonstration Guide & Checklist

This document provides exact instructions, commands, demo scripts, algorithmic disclosures, and recovery procedures for presenting **StudySphere AI** at the live college capstone defense.

---

## 1. Quick Startup Commands

### Backend (FastAPI + SQLite + Sentence-Transformers)
Open a terminal in the project directory:
```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Alternative (without activating virtual environment in shell)*:
```powershell
.\backend\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

### Frontend (React 19 + TypeScript + Vite)
In a second terminal in the project root:
```powershell
npm run dev
```

---

## 2. Key URLs for Demonstration

| Service | URL | Purpose |
| :--- | :--- | :--- |
| **Frontend Application** | `http://localhost:8443` | Main interactive student interface |
| **Alternative Frontend URL** | `http://127.0.0.1:8443` | Direct loopback access |
| **Backend Health Check** | `http://127.0.0.1:8000/api/health` | Verifies service status, embedding model, and device |
| **Interactive Swagger Docs** | `http://127.0.0.1:8000/docs` | OpenAPI documentation and interactive endpoint testing |
| **Alternative API ReDoc** | `http://127.0.0.1:8000/redoc` | Clean documentation view for technical questions |

---

## 3. PDF Preparation Requirements

- **Supported Format:** Genuine PDF documents containing selectable text.
- **Recommended File Size:** 100 KB – 10 MB (up to 20 MB supported).
- **Page Count:** 2 to 20 pages recommended for responsive live demonstration.
- **Scanned / Image PDFs:** Documents without extractable text (e.g. pure raster scans) are detected automatically; the system returns an informative notice stating OCR is required.
- **Demo Document Provided:** A verified 4-page academic demo PDF is located at:
  `test_notes_capstone.pdf` (covering Process Management, CPU Scheduling, Deadlocks, and Missing Data Imputation).
- **Offline Model Requirements:**
  - Embedding Model: `sentence-transformers/all-MiniLM-L6-v2` (~80 MB).
  - Storage: Cached locally in `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/`.
  - Internet Connection: **NOT REQUIRED** during the live demonstration.
  - Memory Footprint: ~150 MB RAM for embedding inference on CPU.

---

## 4. Expected Behavior for Each Major Feature

| Feature | User Action | Expected System Behavior |
| :--- | :--- | :--- |
| **PDF Upload** | Upload `test_notes_capstone.pdf` | Extracts 4 pages, chunks text into 9 segments, computes embeddings, saves to SQLite, and updates active document dropdown. |
| **Grounded Q&A** | Ask *"What are the principal states of a process?"* | Returns structured answer identifying New, Running, Waiting, Ready, Terminated with verified **Page 1** citation. |
| **Hallucination Prevention** | Ask *"Explain the history of the Roman Empire using my uploaded notes"* | Dual-gate filter detects zero topical evidence, sets `retrieval_status: "insufficient_evidence"`, and displays helpful notice citing **0 sources**. |
| **Short Summary** | Click **Generate summary** in Short mode | Produces 3–4 high-level bullet takeaways citing source pages 1–4. |
| **Key Points Summary** | Select Key Points mode & generate | Returns structured, high-yield examination bullet points with page citations. |
| **Detailed Summary** | Select Detailed mode & generate | Synthesizes an organized section-by-section academic summary covering the entire document. |
| **Topic-wise Summary** | Select Topic-wise mode & generate | Segments notes into distinct modules corresponding to course headers. |
| **Quiz Generation** | Select 5 MCQs & Medium difficulty | Returns 5 questions with options and page references. Server stores authoritative answer keys in SQLite and **redacts** them from the response. |
| **Quiz Evaluation** | Submit answers | Server compares student answers against stored SQLite session key, returns score, percentage, detailed feedback, and reveals correct answers and explanations. |
| **Document Cleanup** | Delete document in My Documents | Removes document from list; cascades in SQLite to delete all chunks and associated quiz sessions. |

---

## 5. Suggested 7-Minute Demo Sequence

### Step 1: System Overview & Architecture (1 Minute)
1. Open `http://localhost:8443` in the browser. Show the Dashboard and persistent document library.
2. Open `http://127.0.0.1:8000/api/health` in a new tab to show that the offline embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is running locally on CPU with zero paid API keys or subscription dependencies.

### Step 2: PDF Upload & Text Extraction (1.5 Minutes)
1. Click **Add Notes** in the top bar.
2. Select or drag-and-drop `test_notes_capstone.pdf`.
3. Click **Upload & Extract Text**.
4. Show the notification confirming page-by-page extraction, word count, and semantic chunk indexing in SQLite.
5. Point out that the document now appears in the active study material dropdown across all tabs.

### Step 3: Grounded RAG Q&A ("Ask Your Notes") (1.5 Minutes)
1. Navigate to **Ask Your Notes** in the sidebar.
2. Select `test_notes_capstone.pdf` from the dropdown.
3. Ask a grounded question:
   > *"What are the principal states of a process?"*
4. Show the structured answer citing **Page 1**, listing: *New, Running, Waiting, Ready, Terminated*. Click to expand and view the exact retrieved chunk excerpt.
5. Ask an unrelated question to demonstrate grounding and hallucination resistance:
   > *"Explain the history of the Roman Empire using my uploaded notes"*
6. Show that StudySphere AI recognizes the lack of evidence, sets status to `insufficient_evidence`, and cites 0 sources rather than fabricating an answer.

### Step 4: Multi-Mode AI Summarizer (1.5 Minutes)
1. Navigate to **AI Summarizer**.
2. Select **Short Summary** and click **Generate summary**. Show the concise bullet takeaways and page citations.
3. Switch to **Key Points** mode and regenerate to show structured, high-yield examination bullet points.
4. Switch to **Detailed** mode to show structured section-by-section academic coverage.
5. Switch to **Topic-wise** mode to highlight automated topic segmentation.

### Step 5: Secure Revision Questions & Interactive Quiz (1.5 Minutes)
1. Navigate to **Revision Questions**.
2. Select **Multiple Choice (MCQ)**, Count: **5**, Difficulty: **Medium**. Click **Generate revision questions**.
3. **Highlight Security & Integrity:** Open browser DevTools Network tab if desired—show that the response contains only question text, options, and page citations. The `correct_answer` and `explanation` are **redacted** on the server and stored in SQLite under a secure `session_id`.
4. Select answers interactively in the UI.
5. Click **Submit Quiz & Check Answers**.
6. Show that the server evaluates the submission against its stored session key, displaying:
   - Total score and percentage.
   - Transparent disclosure of the grading methodology (`deterministic_and_keyword_overlap`).
   - Detailed feedback, correct answers, explanations, and source page citations.
7. Click **New Quiz**, switch to **Short Answer** or **True/False**, and demonstrate varied question types with natural phrasing.

---

## 5. Capstone Presentation Accuracy & Algorithmic Disclosure

To ensure complete academic honesty during faculty questioning, present the actual algorithms implemented:

| Feature | Actual Implementation | What It Is NOT |
| :--- | :--- | :--- |
| **Embeddings** | Local `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors run locally on CPU). | NOT an OpenAI or paid cloud embedding API. |
| **Retrieval** | Dense vector cosine similarity search with sliding window text chunks (300 chars, 60 overlap). | NOT a full-text SQL LIKE query or BM25 keyword search. |
| **Relevance Gating** | Dual-gate filtering: vector similarity threshold (≥0.32) combined with query substantive term coverage. | NOT unvalidated top-K dumping. |
| **Summarization** | Grounded extractive summarizer: ranks and organizes source sentences by topical density, lead terms, and page structure. | NOT a generative LLM (unless optional OpenAI key is explicitly supplied). |
| **Question Generator** | Deterministic concept extraction: parses definitional sentences, term-colon pairs, and multi-part rules into valid exam questions. | NOT an uncontrolled hallucination engine. |
| **Quiz Evaluation** | Deterministic exact-match for MCQ/True-False; normalized keyword-overlap for short/long answers. Authoritative state in SQLite. | NOT subjective AI grading. |

> **Presentation Note:** Explicitly state: *"StudySphere AI is designed for free, offline, fully local deployment in educational environments. It uses sentence-transformer embeddings and rule-based NLP extraction to guarantee 100% citation faithfulness without recurring API costs or hallucinations."*

---

## 6. Known Limitations to Acknowledge Honestly

1. **Scanned Documents (OCR):** The current build extracts text natively via PyMuPDF. Pure raster scans (photographed book pages without an OCR text layer) will be flagged as containing zero extractable text.
2. **Generative Synthesis vs. Extractive:** The default offline synthesizer compiles exact sentences and structured excerpts from notes. It does not rewrite prose like a 70B parameter LLM unless an optional OpenAI API key is configured.
3. **Short/Long Answer Grading:** Uses keyword-overlap against key rubric points rather than deep semantic reasoning. Students must include primary technical terms to achieve full credit.

---

## 7. Demo Recovery Procedures

### If Backend Stops or Times Out:
1. Open PowerShell and check if port 8000 is occupied:
   ```powershell
   netstat -ano | findstr "8000"
   ```
2. Restart the server cleanly:
   ```powershell
   cd backend
   .\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
3. Refresh `http://localhost:8443` in the browser. Previously indexed documents and quiz sessions will automatically load from `backend/data/studysphere.db`.

### If Frontend Shows "Cannot connect to backend server":
1. Verify `http://127.0.0.1:8000/api/health` in a browser tab.
2. Ensure CORS has not blocked the request. Default origins include `http://localhost:8443` and `http://127.0.0.1:8443`.
3. Check browser DevTools console for network error details.

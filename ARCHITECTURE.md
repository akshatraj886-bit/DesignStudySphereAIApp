# StudySphere AI — System Architecture & Workflow Specifications

This document outlines the detailed system architecture, pipeline flowcharts, and component interaction models for **StudySphere AI**.

---

## 1. High-Level System Architecture Diagram

```mermaid
graph TD
    subgraph Client ["Frontend Presentation Layer (React 19 + TypeScript)"]
        UI["Web UI (Dashboard, Notes Q&A, Summarizer, Quizzes)"]
        API_CLIENT["Typed API Client (src/lib/api.ts)"]
        UI <--> API_CLIENT
    end

    subgraph Server ["API & Gateway Layer (FastAPI)"]
        ROUTER["FastAPI Router (app/main.py)"]
        P_SCHEMA["Pydantic Schemas & Validators (app/schemas.py)"]
        API_CLIENT <-->|HTTP / REST JSON| ROUTER
        ROUTER --- P_SCHEMA
    end

    subgraph Ingestion ["Ingestion & Preprocessing Pipeline"]
        PDF_PARSER["PDF Text Extractor (PyMuPDF)"]
        CHUNKER["Overlapping Sliding Window Chunker (app/rag/chunker.py)"]
        EMBED["SentenceTransformer: all-MiniLM-L6-v2 (app/rag/embeddings.py)"]
        
        ROUTER -->|Upload PDF| PDF_PARSER
        PDF_PARSER -->|Pages & Metadata| CHUNKER
        CHUNKER -->|Overlapping Text Chunks| EMBED
    end

    subgraph Storage ["Persistence Layer (SQLite - backend/data/studysphere.db)"]
        DB_DOCS[("documents Table")]
        DB_CHUNKS[("document_chunks Table: Text & Vector Blobs")]
        DB_QUIZ[("quiz_sessions Table: Authoritative Answer Keys")]
        
        PDF_PARSER -->|Document Metadata| DB_DOCS
        EMBED -->|Serialized Embeddings & Chunks| DB_CHUNKS
    end

    subgraph RAG_Engine ["RAG Q&A Pipeline (app/rag/generator.py)"]
        Q_EMBED["Query Embedding (all-MiniLM-L6-v2)"]
        COS_SEARCH["Cosine Similarity Vector Dot Product"]
        DUAL_GATE["Dual-Gate Relevance Filter (Threshold >= 0.32 & Term Coverage)"]
        SYNTH["Extractive Grounded Synthesizer"]

        ROUTER -->|Ask Question| Q_EMBED
        Q_EMBED --> COS_SEARCH
        DB_CHUNKS -.->|Candidate Vectors| COS_SEARCH
        COS_SEARCH --> DUAL_GATE
        DUAL_GATE -->|Pass: Relevant Chunks| SYNTH
        DUAL_GATE -->|Fail: Insufficient Evidence| ROUTER
        SYNTH -->|Grounded Answer + Page Citations| ROUTER
    end

    subgraph Summarizer_Branch ["Summarizer Pipeline (app/rag/summarizer.py)"]
        SUM_ENGINE["Extractive Sentence Saliency Ranker"]
        SUM_MODES["Mode Formatter: Short | Key Points | Detailed | Topic-wise"]

        ROUTER -->|Summarize Request| SUM_ENGINE
        DB_CHUNKS -.->|Document Text| SUM_ENGINE
        SUM_ENGINE --> SUM_MODES
        SUM_MODES -->|Structured Summary + Page Citations| ROUTER
    end

    subgraph Quiz_Gen_Branch ["Quiz Generation Pipeline (app/rag/question_generator.py)"]
        CONCEPT_EXT["Concept Extractor (Regex & Pattern NLP)"]
        DISTRACTOR_SAMPLER["Distractor Sampler (In-Document Term Pool)"]
        SESSION_GEN["Session Token Generator (qses_...)"]
        KEY_REDACTOR["Answer-Key Redactor (Public Payload)"]

        ROUTER -->|Generate Questions Request| CONCEPT_EXT
        DB_CHUNKS -.->|Document Concepts| CONCEPT_EXT
        CONCEPT_EXT --> DISTRACTOR_SAMPLER
        DISTRACTOR_SAMPLER --> SESSION_GEN
        SESSION_GEN -->|Save Authoritative Answer Keys| DB_QUIZ
        SESSION_GEN --> KEY_REDACTOR
        KEY_REDACTOR -->|Public Questions (Keys Redacted)| ROUTER
    end

    subgraph Quiz_Eval_Branch ["Authoritative Quiz Evaluation Pipeline"]
        EVAL_VALIDATOR["Request & Session Validator"]
        GRADING_ENGINE["Deterministic & Keyword-Overlap Grader"]

        ROUTER -->|Submit Student Answers + Session ID| EVAL_VALIDATOR
        DB_QUIZ -.->|Fetch Authoritative Answer Keys| EVAL_VALIDATOR
        EVAL_VALIDATOR -->|Authoritative Keys + User Answers| GRADING_ENGINE
        GRADING_ENGINE -->|Score + Feedback + Revealed Keys| ROUTER
    end
```

---

## 2. Ingestion & Indexing Pipeline Flow

The ingestion pipeline converts raw uploaded PDF documents into searchable semantic vector embeddings:

```mermaid
sequenceDiagram
    autonumber
    actor User as Student
    participant UI as React UI (Dashboard)
    participant API as FastAPI Router
    participant PDF as PyMuPDF Extractor
    participant Chunker as Sliding Window Chunker
    participant Embed as MiniLM-L6-v2 Embedder
    participant DB as SQLite Database

    User->>UI: Selects PDF & clicks "Upload & Extract Text"
    UI->>API: POST /api/documents/upload (multipart/form-data)
    API->>PDF: Validate %PDF- header & parse pages
    alt Invalid / Corrupt PDF
        PDF-->>API: Magic byte mismatch
        API-->>UI: HTTP 400 (Invalid PDF format)
    else Scanned PDF (0 characters)
        PDF-->>API: Extracted 0 text characters
        API->>DB: Record document with has_extractable_text = False
        API-->>UI: HTTP 200 (OCR Suggested notification)
    else Valid Text PDF
        PDF->>PDF: Extract text, word counts, page numbers
        PDF->>Chunker: Raw page texts
        Chunker->>Chunker: Split into 300-char windows with 60-char overlap
        Chunker->>Embed: Overlapping text chunks
        Embed->>Embed: Compute 384-dim dense vectors & L2-normalize
        Embed->>DB: INSERT into documents & document_chunks
        DB-->>API: Commit transaction (doc_id, chunk_count)
        API-->>UI: HTTP 200 (Indexed metadata, page counts, word counts)
        UI-->>User: Document loaded into active study dropdown
    end
```

---

## 3. RAG Q&A Pipeline Flow

The Retrieval-Augmented Generation pipeline answers student queries grounded in source notes:

```mermaid
sequenceDiagram
    autonumber
    actor Student
    participant UI as React UI ("Ask Your Notes")
    participant API as FastAPI Router
    participant Gen as Generator & Relevance Filter
    participant Embed as MiniLM-L6-v2 Embedder
    participant DB as SQLite Vector Store

    Student->>UI: Types question: "What is Round Robin scheduling?"
    UI->>API: POST /api/ask {question, document_id}
    API->>Embed: Generate normalized query vector (384-dim)
    API->>DB: Fetch chunks for document_id
    DB-->>Gen: Chunks and binary vector blobs
    Gen->>Gen: Compute Cosine Dot Product Sim(q, c)
    Gen->>Gen: Select Top-K chunks sorted by similarity
    Gen->>Gen: Apply Dual-Gate Relevance Verification:
    Note over Gen: Gate 1: Max similarity score >= 0.32<br/>Gate 2: Substantive query terms overlap in passages
    alt Gate Fails (Unrelated Topic, e.g. "Roman Empire" in CS Notes)
        Gen-->>API: retrieval_status = "insufficient_evidence", cited_sources = []
        API-->>UI: "I could not find sufficient evidence in your notes..."
    else Gate Passes (Relevant Evidence Found)
        Gen->>Gen: Synthesize structured answer citing exact [Page X]
        Gen-->>API: retrieval_status = "found", cited_sources = [Passages with scores & pages]
        API-->>UI: Structured markdown answer + interactive page citations
        UI-->>Student: Renders answer and source evidence cards
    end
```

---

## 4. Multi-Mode Summarizer Pipeline Flow

```mermaid
flowchart TD
    A[Student Selects Document & Mode] -->|POST /api/documents/id/summarize| B(FastAPI Endpoint)
    B --> C{Document Has Text?}
    C -->|No| D[Return HTTP 400 OCR Notice]
    C -->|Yes| E[Fetch All Document Chunks from SQLite]
    E --> F[Extract Sentences & Filter Boilerplate]
    F --> G[Score Sentences by Topical Salience & Concept Density]
    G --> H{Active Mode}
    H -->|Short| I[Synthesize 3-4 High-Level Key Takeaways]
    H -->|Key Points| J[Synthesize Bulleted Examination Highlights]
    H -->|Detailed| K[Synthesize Section-by-Section Hierarchical Narrative]
    H -->|Topic-wise| L[Segment Content by Lecture Headings & Modules]
    I --> M[Attach Source Page Numbers & Return JSON]
    J --> M
    K --> M
    L --> M
    M --> N[Render in React UI with Expandable Citations]
```

---

## 5. Secure Revision Quiz Generation & Evaluation Flow

This diagram illustrates how answer keys are isolated from the frontend and evaluated server-side:

```mermaid
sequenceDiagram
    autonumber
    actor Student
    participant Frontend as React Client
    participant API as FastAPI Backend
    participant QGen as Question Generator
    participant DB as SQLite (quiz_sessions)

    Note over Frontend, DB: 1. SECURE GENERATION PHASE (NO KEYS LEAKED)
    Student->>Frontend: Selects Question Types (MCQ, T/F, Short, Long), Count: 5, Diff: Medium
    Frontend->>API: POST /api/documents/{id}/questions
    API->>QGen: Extract concepts, sample distractors, build questions
    QGen->>QGen: Generate unique session token: qses_<timestamp>_<hex>
    QGen->>DB: INSERT into quiz_sessions (id, doc_id, questions_json with answer keys)
    QGen->>QGen: Redact correct_answer & explanation from questions
    QGen-->>API: GenerateQuestionsResponse (session_id, redacted questions)
    API-->>Frontend: HTTP 200 (session_id, questions WITHOUT correct answers)
    Frontend-->>Student: Displays questions and options (answer keys cannot be found in DOM)

    Note over Frontend, DB: 2. INTERACTIVE ATTEMPT & SECURE EVALUATION PHASE
    Student->>Frontend: Selects options / types answers
    Student->>Frontend: Clicks "Submit Quiz & Check Answers"
    Frontend->>API: POST /api/questions/evaluate {session_id, answers: [{qid, user_answer}]}
    API->>DB: SELECT questions_json FROM quiz_sessions WHERE id = session_id
    alt Session ID Not Found
        DB-->>API: None
        API-->>Frontend: HTTP 404 (Unknown quiz session)
    else Session Exists
        DB-->>API: Authoritative questions JSON (with correct_answer & rubrics)
        API->>API: Validate question IDs belong to session (reject foreign/duplicates)
        API->>API: Grade MCQ / True-False (Exact string match)
        API->>API: Grade Short / Long Answer (Rubric keyword-overlap)
        API-->>Frontend: HTTP 200 {total_score, max_score, evaluations with correct answers & explanations}
        Frontend-->>Student: Displays score breakdown, correct answers, explanations, and citations
    end
```

---

## 6. Entity Relationship Diagram (ERD)

```mermaid
erDiagram
    documents ||--o{ document_chunks : "has chunks (CASCADE DELETE)"
    documents ||--o{ quiz_sessions : "has sessions (CASCADE DELETE)"

    documents {
        TEXT id PK "Unique identifier (doc_...)"
        TEXT filename "Original file name"
        INTEGER page_count "Total pages"
        INTEGER total_characters "Total characters"
        INTEGER total_words "Total words"
        INTEGER chunk_count "Total chunks"
        TIMESTAMP created_at "Upload timestamp"
    }

    document_chunks {
        TEXT id PK "Chunk identifier (chk_...)"
        TEXT document_id FK "References documents(id)"
        TEXT document_name "Denormalized file name"
        INTEGER page_number "Source page number"
        INTEGER chunk_index "Index in document"
        TEXT text "Clean chunk text"
        BLOB embedding "384-dim float32 binary vector"
        TIMESTAMP created_at "Created timestamp"
    }

    quiz_sessions {
        TEXT id PK "Session identifier (qses_...)"
        TEXT document_id FK "References documents(id)"
        TEXT document_name "Denormalized file name"
        TEXT difficulty "Difficulty (easy, medium, hard)"
        TEXT questions_json "Full authoritative questions with keys"
        TIMESTAMP created_at "Created timestamp"
    }
```

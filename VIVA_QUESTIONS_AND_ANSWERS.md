# StudySphere AI — Capstone Viva Examination Preparation Guide

This guide contains **25 essential technical questions and model answers** for the final capstone defense of **StudySphere AI**. The answers are concise, technically accurate, and structured for easy verbal delivery.

---

## Category 1: RAG Architecture, Embeddings & Vector Search

### Q1: What is Retrieval-Augmented Generation (RAG), and why did you choose it over fine-tuning a model?
**Answer:**  
RAG is an information retrieval pattern that combines an external knowledge base with a language or synthesis model. Instead of relying solely on parametric memory (weights), RAG first retrieves relevant document chunks matching a user query and passes them as grounding context. We chose RAG over fine-tuning because:
1. Academic notes change frequently; RAG indexes new PDFs dynamically in seconds without expensive retraining.
2. RAG provides exact, verifiable page citations (`[Page X]`), whereas fine-tuned models cannot reliably cite sources.
3. RAG drastically minimizes hallucinations by constraining answers to retrieved passages.

---

### Q2: Which embedding model does StudySphere AI use, and what are its vector dimensions?
**Answer:**  
We use `sentence-transformers/all-MiniLM-L6-v2`, a distilled transformer model based on MiniLM architecture:
- **Output Dimensions:** 384-dimensional dense float32 vector.
- **Model Size:** ~80 MB disk footprint, requiring ~150 MB RAM.
- **Execution:** Runs locally on consumer CPU in under 15 ms per query, or automatically accelerates via CUDA if an NVIDIA GPU is detected.
- **Training Objective:** Contrastive learning on over 1 billion sentence pairs to capture semantic similarity.

---

### Q3: How do you compute cosine similarity between the student query and stored document chunks?
**Answer:**  
Cosine similarity measures the angle between two multi-dimensional vectors:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
Because our embedding service pre-normalizes every vector to unit Euclidean length ($\|\mathbf{u}\|_2 = 1.0$), the denominator equals 1.0. Therefore, cosine similarity reduces to a fast dot product:
$$\text{Sim}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{384} u_i \cdot v_i$$
We compute this directly in Python using NumPy over serialized float32 arrays retrieved from SQLite.

---

### Q4: Why is vector normalization (L2 norm) important in your retrieval pipeline?
**Answer:**  
L2 normalization rescales vectors to length 1.0. This ensures that:
1. Long and short text chunks are compared solely based on their conceptual orientation rather than vector magnitude (which correlates with length).
2. Dot product calculations directly equal cosine similarity, avoiding costly square root and division operations during runtime retrieval.

---

### Q5: What is the dual-gate relevance filter in your RAG pipeline, and what problem does it solve?
**Answer:**  
Standard RAG systems always return the top-K chunks, even if the user asks a completely unrelated question (e.g. asking about the Roman Empire in Operating Systems notes). The dual-gate filter prevents this by requiring two conditions:
1. **Gate 1 (Vector Threshold):** The maximum cosine similarity score must be $\ge 0.32$.
2. **Gate 2 (Substantive Entity Coverage):** Key query nouns and technical terms must appear in the retrieved text passages.  
If either gate fails, the system immediately rejects the query with status `insufficient_evidence` and cites 0 sources.

---

### Q6: How does StudySphere AI handle out-of-domain or ungrounded questions?
**Answer:**  
When the dual-gate filter detects insufficient evidence, the system bypasses synthesis and returns an explicit, transparent message:
> *"I could not find sufficient evidence in the uploaded document(s) to answer this question. The retrieved passages do not discuss this topic."*  
It cites 0 sources and sets `retrieval_status: "insufficient_evidence"`, ensuring zero hallucinations.

---

## Category 2: Text Processing, Chunking & Citations

### Q7: Why do you use sliding-window chunking with overlap instead of single sentences or entire pages?
**Answer:**  
- **Entire pages** are too large (~3,000 characters), diluting vector embeddings with multiple distinct topics and reducing retrieval precision.
- **Single sentences** are too short (~80 characters), losing essential contextual antecedents (e.g., pronouns like *"It is optimal because..."* lose the referent *"SJF"*).
- **Overlapping sliding windows** (300 characters with 60-character overlap) provide the ideal semantic granularity while ensuring sentences split across chunk boundaries maintain continuous context.

---

### Q8: What chunk size and overlap parameters did you choose and why?
**Answer:**  
We chose a target chunk size of 300 characters and an overlap of 60 characters (20% overlap). This accommodates typical academic definitions (usually 150–250 characters) within a single chunk while allowing the 60-character overlap to preserve transition words and connecting clauses between consecutive chunks.

---

### Q9: How are exact source page numbers tracked and preserved throughout chunking and retrieval?
**Answer:**  
PyMuPDF extracts text page-by-page. When chunks are created in `app/rag/chunker.py`, each chunk is tagged with its parent `document_id`, `document_name`, 1-indexed `page_number`, and `chunk_index`. This metadata is stored alongside the embedding in the SQLite `document_chunks` table and is returned in the API response to render interactive page badges in the UI.

---

### Q10: Why did you choose PyMuPDF (fitz) over alternatives like pypdf or pdfminer?
**Answer:**  
PyMuPDF is a Python binding for the MuPDF C library. It is between 10x and 30x faster than pure-Python libraries like `pypdf` or `pdfminer`. It reliably handles multi-column layouts, cleans whitespace, preserves font metadata, and extracts page-level text streams in a few milliseconds per page on standard CPUs.

---

## Category 3: Extractive vs. Generative AI

### Q11: What is the fundamental difference between extractive and generative (abstractive) NLP?
**Answer:**  
- **Extractive NLP:** Selects, ranks, and compiles exact sentences or verbatim phrases directly from the source material without generating new vocabulary. It guarantees 100% factual fidelity and zero hallucination risk.
- **Generative (Abstractive) NLP:** Uses a decoder transformer (like GPT-4 or Gemini) to rewrite and paraphrase content. While more fluent, it has a non-zero probability of hallucinating ungrounded facts or misattributing citations.

---

### Q12: Why did you implement an extractive synthesizer as the default offline generator?
**Answer:**  
1. **Academic Grounding:** Students need verifiable facts from their specific syllabus, not rephrased approximations.
2. **Offline Hardware Feasibility:** High-parameter generative LLMs require 8–16 GB of VRAM. An extractive synthesizer runs in <10 ms on standard laptops using <150 MB RAM.
3. **Zero Financial Cost:** Runs completely offline without cloud API keys, recurring token fees, or rate limits.

---

### Q13: How does the system generate four distinct summary modes from the same document?
**Answer:**  
In `app/rag/summarizer.py`, sentences are scored based on term frequency, positional prominence, and academic concept density. The mode formatter formats these sentences differently:
- **Short Summary:** Selects top 3–4 highest-ranked sentences across the entire document.
- **Key Points:** Formats high-yield concept sentences into concise bullet points with page citations.
- **Detailed Summary:** Groups top sentences into an organized multi-section narrative.
- **Topic-wise Summary:** Groups sentences under their corresponding lecture/module headings.

---

### Q14: How does the revision question generator create questions without an LLM?
**Answer:**  
In `app/rag/question_generator.py`, we implement deterministic pattern-matching NLP:
1. Regex patterns extract definitional sentences (e.g. `X is a Y`) and colon-delimited terms (e.g. `Term: Definition`).
2. MCQs use the extracted definition as the prompt and sample 3 unique terms from other concepts in the same document as plausible distractors.
3. True/False questions alternate between verbatim facts and domain-specific negated statements.
4. Short and Long Answer questions generate prompts paired with structured marking rubrics.

---

## Category 4: Database Design & Persistence

### Q15: How does StudySphere AI store high-dimensional vector embeddings in SQLite?
**Answer:**  
SQLite does not have a native vector type, so we serialize 384-dimensional float32 NumPy arrays into raw binary bytes using `.tobytes()` and store them in a `BLOB` column. When querying, we read the blob and deserialize it in-memory via `np.frombuffer(blob, dtype=np.float32)`. This provides high-speed vector retrieval without external dependencies.

---

### Q16: Why did you choose SQLite over dedicated vector databases like Pinecone, Milvus, or ChromaDB?
**Answer:**  
1. **Zero Configuration & Single Binary:** SQLite is built into Python's standard library; it requires no separate Docker container, cloud account, or background daemon.
2. **ACID Relational Integrity:** We can link documents, chunks, and quiz sessions using standard foreign keys with `ON DELETE CASCADE`.
3. **Student Scale:** For individual student course notes (dozens of documents, a few thousand chunks), SQLite vector dot products take <20 ms, making heavy vector databases unnecessary overhead.

---

### Q17: How does the database enforce relational integrity when a document is deleted?
**Answer:**  
We enable SQLite foreign keys (`PRAGMA foreign_keys = ON;`). In the schema:
- `document_chunks` references `documents(id) ON DELETE CASCADE`.
- `quiz_sessions` references `documents(id) ON DELETE CASCADE`.  
When a user deletes a document via `DELETE /api/documents/{id}`, SQLite automatically and atomically deletes all corresponding text chunks, vector blobs, and stored quiz sessions.

---

### Q18: What is stored in the `quiz_sessions` table, and why is it separated from the document table?
**Answer:**  
The `quiz_sessions` table stores:
- `id`: Unique session identifier (`qses_<timestamp>_<hex>`).
- `document_id`: Foreign key to parent document.
- `difficulty`: Selected difficulty level.
- `questions_json`: JSON string containing complete authoritative questions, option sets, source pages, and **private answer keys and marking rubrics**.  
It is separated from `documents` because multiple unique quizzes can be generated from the same document over time.

---

## Category 5: Quiz Security & Integrity

### Q19: What was the critical security vulnerability identified in Phase 5 regarding quizzes?
**Answer:**  
The integration audit revealed two security flaws:
1. **Answer-Key Exposure:** The generation endpoint returned `correct_answer` and `explanation` directly to the frontend before the quiz was attempted.
2. **Client-Side Score Manipulation:** The evaluation endpoint trusted question data and answer keys supplied in the client request body, allowing a student to modify `correct_answer` to match their guess and achieve 100%.

---

### Q20: How did you solve answer-key leakage in the public generation response?
**Answer:**  
In Phase 6, we separated public and private schemas:
- `QuestionItem` (private) contains `correct_answer` and `explanation` and is saved strictly in SQLite `quiz_sessions`.
- `PublicQuestionItem` inherits from the base schema but completely omits `correct_answer` and `explanation`.
- The generation endpoint returns only `PublicQuestionItem` objects and the `session_id`. The browser network payload contains zero answer keys.

---

### Q21: How does the server prevent a client from manipulating their quiz score?
**Answer:**  
When evaluating a quiz (`POST /api/questions/evaluate`), the client submits only `session_id` and student answers `[{question_id, user_answer}]`. The backend:
1. Retrieves the authoritative questions and answer keys directly from SQLite using `session_id`.
2. Validates that all question IDs belong to that session and rejects foreign or duplicate question IDs.
3. Completely ignores any client-supplied question content, correct answers, or marks.
4. Grades strictly server-side using stored answer keys before revealing feedback.

---

## Category 6: Evaluation, Performance & Future Scope

### Q22: What evaluation metrics did you use to benchmark your RAG system, and what were the results?
**Answer:**  
We evaluated the RAG system using a 15-question academic benchmark (`backend/test_rag.py`):
- **Retrieval Recall@3 (Answerable):** **100.0% (12/12)** — The correct source page was retrieved in the top 3 candidates for all answerable queries.
- **Insufficient Evidence Decision Accuracy:** **100.0% (3/3)** — 100% of out-of-domain queries were correctly rejected.
- **Source Page Citation Correctness:** **91.7% (11/12)** — Cited the exact correct page for answers.
- **Answer Faithfulness to Evidence:** **100.0% (12/12)** — 100% of answer statements were verified to originate from cited passages.

---

### Q23: How are short-answer and long-answer student submissions evaluated and scored?
**Answer:**  
In `app/rag/question_generator.py`, short and long answers are evaluated using normalized keyword overlap against key rubric points stored in the session:
1. Stop words and punctuation are removed from the student answer and rubric.
2. The server measures the proportion of rubric terms present in the student submission.
3. Points are awarded proportionally (0.0 to 1.0) and accompanied by transparent feedback explaining which concepts were covered or missed.

---

### Q24: What are the main limitations of your current implementation?
**Answer:**  
1. **Scanned Images (OCR):** The system relies on native PDF text streams via PyMuPDF; scanned physical book pages require an external OCR step.
2. **Keyword vs. Semantic Short Answer Grading:** Scoring uses keyword overlap rather than deep semantic inference, requiring students to mention specific technical terms.
3. **Extractive Summaries:** Summaries compile exact sentences from the text rather than rewriting fluid prose like a 70B parameter LLM.

---

### Q25: If you had another semester to expand StudySphere AI, what features would you implement next?
**Answer:**  
1. **Local OCR Integration:** Embed Tesseract or PaddleOCR to automatically extract text from photographed lecture slides and scanned textbooks.
2. **Quantized Local Small Language Model (SLM):** Integrate a quantized local model (like Phi-3 Mini 3.8B or Qwen-2.5 1.5B via llama.cpp) for offline abstractive synthesis without cloud API dependencies.
3. **Semantic Embedding Dot-Product Grading:** Upgrade short-answer grading to compute vector similarity between student responses and rubric points to reward accurate conceptual paraphrasing.
4. **Cross-Document Search:** Enable cross-document querying across an entire semester folder.

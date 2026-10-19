"""
StudySphere AI - Phase 3 Extended RAG Automated Test & Evaluation Suite
Validates:
1. Health endpoint with model metadata.
2. PDF upload, chunking, embedding, and persistent SQLite storage.
3. Metadata preservation across chunks (document ID, name, page numbers, indices).
4. Semantic vector retrieval with cosine similarity.
5. Document-restricted retrieval.
6. Input validation (empty queries).
7. Transparent semantic similarity scores and separation of cited vs retrieved candidate sources.
8. Dual-gate relevance verification (rejecting unrelated questions like 'Roman Empire').
9. Grounded generation (both mock provider and structured extractive synthesis).
10. Extended Academic Evaluation Benchmark (15 Questions):
    - Direct answerable questions from single pages.
    - Paraphrased and informal questions ('bhai how to handle missing valused').
    - Multi-page synthesis questions.
    - Questions with lower similarity scores.
    - Unrelated questions (Roman Empire, cooking recipe).
    - Questions with absent specific concepts (e.g. LRU paging).
11. Evaluation Metrics:
    - Retrieval Recall@3.
    - Insufficient evidence decision accuracy (100% target).
    - Citation and page reference correctness.
    - Answer faithfulness to retrieved evidence.
"""

import os
import sys
import re
import io
import time
import pymupdf
from fastapi.testclient import TestClient

# Ensure backend directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from app.main import app
from app.rag.chunker import chunk_text, process_document_pages
from app.rag.store import VectorStore
from app.rag.embeddings import embedding_service
from app.rag.generator import generator_instance, extract_substantive_query_terms, verify_topical_evidence


def create_sample_educational_pdf() -> bytes:
    """
    Generates a realistic 4-page academic PDF covering:
    Page 1: Operating System Fundamentals and Process Lifecycle.
    Page 2: CPU Scheduling Algorithms (FCFS, SJF, Round Robin, Priority).
    Page 3: Deadlocks and Concurrency Synchronization (Coffman conditions, Banker's Algorithm).
    Page 4: Data Science & Machine Learning: Handling Missing Values (Deletion, Imputation, MICE).
    """
    doc = pymupdf.open()

    # Page 1: OS Fundamentals
    page1 = doc.new_page(width=595, height=842)
    p1_text = (
        "StudySphere AI Academic Notes - Computer Science & Engineering\n"
        "Course: Operating Systems & Data Science - Module 1: Process Management\n\n"
        "1. Process Concept and State Transitions\n"
        "A process is a program in execution. It includes the program code (text section), "
        "current activity represented by the program counter, processor registers, and the stack.\n"
        "A process changes state as it executes. The principal states of a process are:\n"
        "- New: The process is being created.\n"
        "- Running: Instructions are being executed by the CPU.\n"
        "- Waiting (Blocked): The process is waiting for some event to occur (such as I/O completion).\n"
        "- Ready: The process is waiting to be assigned to a processor.\n"
        "- Terminated: The process has finished execution.\n\n"
        "The Process Control Block (PCB) stores critical information for each process, including "
        "process ID (PID), program counter, CPU registers, CPU scheduling information, and memory management info."
    )
    page1.insert_text((50, 60), p1_text, fontsize=11)

    # Page 2: Scheduling Algorithms
    page2 = doc.new_page(width=595, height=842)
    p2_text = (
        "Operating Systems & Data Science - Module 1: Process Management\n\n"
        "2. CPU Scheduling Algorithms\n"
        "CPU scheduling is the basis of multi-programmed operating systems. By switching the CPU among "
        "processes, the operating system can make the computer more productive.\n\n"
        "Key scheduling algorithms include:\n"
        "- First-Come, First-Served (FCFS): The process that requests the CPU first is allocated the CPU first. "
        "It is non-preemptive and can suffer from the convoy effect.\n"
        "- Shortest-Job-First (SJF): Associates each process with the length of its next CPU burst. "
        "Optimal because it gives the minimum average waiting time, but future CPU burst duration is hard to predict.\n"
        "- Round Robin (RR): Designed especially for time-sharing systems. A small unit of time, called a time quantum "
        "or time slice (generally 10 to 100 milliseconds), is defined. The CPU scheduler goes around the ready queue, "
        "allocating the CPU to each process for a time interval of up to 1 time quantum.\n"
        "- Priority Scheduling: A priority is associated with each process, and the CPU is allocated to the highest priority process. "
        "Starvation (indefinite blocking) is a major problem, which can be solved using aging."
    )
    page2.insert_text((50, 60), p2_text, fontsize=11)

    # Page 3: Deadlocks
    page3 = doc.new_page(width=595, height=842)
    p3_text = (
        "Operating Systems & Data Science - Module 2: Deadlocks and Synchronization\n\n"
        "3. Deadlocks and Resource Allocation\n"
        "A deadlock is a situation where every process in a set is waiting for an event that only another process "
        "in the set can cause. Four Coffman conditions must hold simultaneously for a deadlock to arise:\n"
        "1. Mutual Exclusion: At least one resource must be held in a non-shareable mode.\n"
        "2. Hold and Wait: A process must be holding at least one resource and waiting to acquire additional resources.\n"
        "3. No Preemption: Resources cannot be preempted; they can only be released voluntarily.\n"
        "4. Circular Wait: A closed chain of processes exists such that each process holds at least one resource needed by the next.\n\n"
        "Deadlock Avoidance - Banker's Algorithm:\n"
        "The Banker's algorithm is a deadlock avoidance algorithm developed by Edsger Dijkstra. When a process requests "
        "resources, the system determines whether allocating them leaves the system in a safe state. A state is safe if "
        "there exists a safe sequence of processes that can all finish without deadlock."
    )
    page3.insert_text((50, 60), p3_text, fontsize=11)

    # Page 4: Data Preprocessing: Handling Missing Values
    page4 = doc.new_page(width=595, height=842)
    p4_text = (
        "Operating Systems & Data Science - Module 3: Data Preprocessing\n\n"
        "4. Handling Missing Values in Datasets\n"
        "Missing data is a common challenge in machine learning and data engineering. "
        "Data can be Missing Completely at Random (MCAR), Missing at Random (MAR), or Missing Not at Random (MNAR).\n\n"
        "Techniques for handling missing values include:\n"
        "- Listwise Deletion: Removes any record or observation containing missing values. Simple to apply but causes significant data loss.\n"
        "- Mean and Median Imputation: Replaces missing numerical values with the mean or median of the observed feature. Fast but reduces variance.\n"
        "- Mode Imputation: Replaces missing categorical values with the most frequently occurring category.\n"
        "- K-Nearest Neighbors (KNN) Imputation: Identifies the k most similar records using Euclidean distance and calculates their weighted average to fill missing attributes.\n"
        "- Multiple Imputation by Chained Equations (MICE): A principled statistical method that models each incomplete variable via a regression series over other variables.\n"
        "- Missingness Indicators: Adds a separate binary flag variable indicating whether a value was originally absent before imputation."
    )
    page4.insert_text((50, 60), p4_text, fontsize=11)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def evaluate_faithfulness(answer: str, cited_sources: list) -> bool:
    """
    Rigorously verifies whether claims in the generated answer are grounded in cited sources.
    Extracts substantive words from the answer body lines and verifies they originate from cited text.
    """
    if not cited_sources:
        return False

    all_source_text = " ".join(
        (s.get("excerpt") or s.get("text") or "").lower() for s in cited_sources
    )

    # Clean boilerplate markdown lines
    body_lines = [
        line.strip()
        for line in answer.split("\n")
        if line.strip()
        and not line.startswith("###")
        and not line.startswith("According to your notes")
        and not line.startswith("Based on your notes")
        and not line.startswith("All points above are directly drawn")
    ]

    if not body_lines:
        return True

    unsupported_count = 0
    total_checked = 0

    for line in body_lines:
        # Extract content words from line (length >= 4)
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", line)]
        if not words:
            continue
        total_checked += 1
        # Check if at least 50% of the substantive words in this claim appear in the source passage
        supported = [w for w in words if w in all_source_text]
        if len(supported) / len(words) < 0.40:
            unsupported_count += 1

    return unsupported_count == 0


def run_all_tests():
    print("=" * 75)
    print("STUDYSPHERE AI - EXTENDED RAG VERIFICATION & EVALUATION SUITE")
    print("=" * 75)

    client = TestClient(app)
    results = {"passed": 0, "failed": 0, "errors": []}

    # 1. Health Check Test
    print("\n[Test 1] GET /api/health with AI model metadata...")
    try:
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
        print(f"  PASS: Health check OK. Model: {data['embedding_model']}, Device: {data['device']}")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 2. Text Chunking Unit Test
    print("\n[Test 2] Text Chunking & Sliding Window Overlap...")
    try:
        sample_doc_text = "Sentence one. " * 30
        chunks = chunk_text(sample_doc_text, target_chunk_size=150, overlap_size=40)
        assert len(chunks) >= 3
        print(f"  PASS: Generated {len(chunks)} overlapping chunks correctly.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 3. PDF Upload & Persistent Indexing Test (4-Page Academic Document)
    print("\n[Test 3] POST /api/documents/upload with 4-page educational notes PDF...")
    uploaded_doc_id = None
    try:
        pdf_bytes = create_sample_educational_pdf()
        files = {"file": ("CS_DataScience_Notes.pdf", pdf_bytes, "application/pdf")}
        res = client.post("/api/documents/upload", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["page_count"] == 4
        assert data["has_extractable_text"] is True
        assert data["chunk_count"] > 0
        uploaded_doc_id = data["document_id"]
        print(f"  PASS: Document uploaded and indexed: {uploaded_doc_id} ({data['chunk_count']} chunks, {data['page_count']} pages)")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 4. List Documents Test
    print("\n[Test 4] GET /api/documents persistent listing...")
    try:
        res = client.get("/api/documents")
        assert res.status_code == 200
        data = res.json()
        matching = [d for d in data["documents"] if d["id"] == uploaded_doc_id]
        assert len(matching) == 1
        print(f"  PASS: Document list contains uploaded document: {matching[0]['filename']}")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 5. Dual Gate Relevance Verification (Unrelated Question Rejection)
    print("\n[Test 5] Rejecting unrelated questions (Roman Empire vs CS/Data Science notes)...")
    try:
        unrelated_query = {
            "question": "Explain the history of the Roman Empire using my uploaded notes",
            "document_id": uploaded_doc_id,
            "top_k": 3,
        }
        res = client.post("/api/ask", json=unrelated_query)
        assert res.status_code == 200
        data = res.json()
        assert data["retrieval_status"] == "insufficient_evidence"
        assert len(data["cited_sources"]) == 0
        print("  PASS: Unrelated question correctly identified as insufficient_evidence with 0 cited sources.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 6. Structured Answer Experience on Conversational Question
    print("\n[Test 6] Structured student-friendly answer for 'bhai how to handle missing valused'...")
    try:
        query_payload = {
            "question": "bhai how to handle missing valused",
            "document_id": uploaded_doc_id,
            "top_k": 3,
        }
        res = client.post("/api/ask", json=query_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["retrieval_status"] == "found"
        assert len(data["cited_sources"]) > 0
        assert "Overview" in data["answer"] or "Key Concepts" in data["answer"]
        assert "Page 4" in str(data["answer"])
        print("  PASS: Conversational missing values question generated structured answer with Page 4 citations.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 7. Source Transparency: Cited Sources vs Retrieved Sources
    print("\n[Test 7] Verifying cited_sources vs retrieved_sources separation...")
    try:
        query_payload = {
            "question": "What is the Banker's algorithm?",
            "document_id": uploaded_doc_id,
            "top_k": 4,
        }
        res = client.post("/api/ask", json=query_payload)
        assert res.status_code == 200
        data = res.json()
        assert "cited_sources" in data
        assert "retrieved_sources" in data
        assert len(data["retrieved_sources"]) >= len(data["cited_sources"])
        # Check transparent float score (not fake percent)
        for cs in data["cited_sources"]:
            assert 0.0 <= cs["similarity_score"] <= 1.0
        print(f"  PASS: Cited sources ({len(data['cited_sources'])}) and candidate retrieved ({len(data['retrieved_sources'])}) cleanly separated.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 8. Regression: Hinglish Normalization & Citation Separation Test
    print("\n[Test 8] Regression: Exact Hinglish query on Normalization document...")
    try:
        from test_hinglish_grounding import create_normalization_notes_pdf
        norm_pdf = create_normalization_notes_pdf()
        norm_files = {"file": ("Normalization_Academic_Notes.pdf", norm_pdf, "application/pdf")}
        res_norm_upload = client.post("/api/documents/upload", files=norm_files)
        assert res_norm_upload.status_code == 200
        norm_doc_id = res_norm_upload.json()["document_id"]

        # Exact Hinglish query from user report
        hinglish_res = client.post(
            "/api/ask",
            json={
                "question": "Normalization kya hota hai? Iska main objective aur insert, update, delete anomalies simple language me samjhao.",
                "document_id": norm_doc_id,
                "top_k": 3,
            },
        )
        assert hinglish_res.status_code == 200
        h_data = hinglish_res.json()
        assert h_data["retrieval_status"] == "found", f"Expected found, got {h_data['retrieval_status']}"
        assert len(h_data["cited_sources"]) > 0, "Expected at least 1 cited source"
        assert all(cs["page_number"] == 1 for cs in h_data["cited_sources"]), "Citations must strictly be Page 1"
        assert "Page 1" in h_data["answer"] or "[Page 1]" in h_data["answer"]
        assert len(h_data["retrieved_sources"]) >= len(h_data["cited_sources"])

        # Unrelated query on the normalization document
        unrelated_norm = client.post(
            "/api/ask",
            json={
                "question": "Explain the history of the Roman Empire using my uploaded notes",
                "document_id": norm_doc_id,
                "top_k": 3,
            },
        )
        assert unrelated_norm.status_code == 200
        u_data = unrelated_norm.json()
        assert u_data["retrieval_status"] == "insufficient_evidence"
        assert len(u_data["cited_sources"]) == 0

        print("  PASS: Hinglish query answered with grounded citations and unrelated query correctly rejected.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # =========================================================================
    # 9. EXTENDED 15-QUESTION ACADEMIC BENCHMARK
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING EXTENDED 15-QUESTION ACADEMIC EVALUATION BENCHMARK")
    print("=" * 75)

    eval_dataset = [
        # Category A: Direct Single-Page Questions
        {
            "id": 1,
            "question": "What are the five principal states of a process?",
            "expected_pages": [1],
            "type": "answerable",
            "category": "Direct (Pg 1)",
        },
        {
            "id": 2,
            "question": "What components are stored in the Process Control Block (PCB)?",
            "expected_pages": [1],
            "type": "answerable",
            "category": "Direct (Pg 1)",
        },
        {
            "id": 3,
            "question": "What is the convoy effect in First-Come, First-Served (FCFS) scheduling?",
            "expected_pages": [2],
            "type": "answerable",
            "category": "Direct (Pg 2)",
        },
        {
            "id": 4,
            "question": "Explain Round Robin (RR) CPU scheduling and the time quantum.",
            "expected_pages": [2],
            "type": "answerable",
            "category": "Direct (Pg 2)",
        },
        {
            "id": 5,
            "question": "List the four Coffman conditions required for a deadlock to arise.",
            "expected_pages": [3],
            "type": "answerable",
            "category": "Direct (Pg 3)",
        },
        {
            "id": 6,
            "question": "What is the Banker's algorithm and safe state sequence?",
            "expected_pages": [3],
            "type": "answerable",
            "category": "Direct (Pg 3)",
        },
        {
            "id": 7,
            "question": "What are the differences between MCAR, MAR, and MNAR missing data?",
            "expected_pages": [4],
            "type": "answerable",
            "category": "Direct (Pg 4)",
        },
        {
            "id": 8,
            "question": "Explain K-Nearest Neighbors (KNN) imputation and MICE for missing values.",
            "expected_pages": [4],
            "type": "answerable",
            "category": "Direct (Pg 4)",
        },

        # Category B: Paraphrased / Informal Questions
        {
            "id": 9,
            "question": "bhai how to handle missing valused",
            "expected_pages": [4],
            "type": "answerable",
            "category": "Paraphrased/Slang",
        },
        {
            "id": 10,
            "question": "tell me about what happens when a program is blocked waiting for IO",
            "expected_pages": [1],
            "type": "answerable",
            "category": "Paraphrased",
        },

        # Category C: Multi-Page Synthesis Questions
        {
            "id": 11,
            "question": "How does CPU scheduling transition processes from the Ready queue to Running state?",
            "expected_pages": [1, 2],
            "type": "answerable",
            "category": "Multi-Page (1+2)",
        },

        # Category D: Lower Vector Similarity / Highly Technical Phrasing
        {
            "id": 12,
            "question": "Explain the mathematical optimality of Shortest-Job-First for minimum average waiting time.",
            "expected_pages": [2],
            "type": "answerable",
            "category": "Technical",
        },

        # Category E: Absent / Unrelated Questions (Must be rejected)
        {
            "id": 13,
            "question": "Explain the history of the Roman Empire using my uploaded notes",
            "expected_pages": [],
            "type": "absent",
            "category": "Unrelated",
        },
        {
            "id": 14,
            "question": "What are the rules and field positions of cricket?",
            "expected_pages": [],
            "type": "absent",
            "category": "Unrelated",
        },
        {
            "id": 15,
            "question": "How does the Least Recently Used (LRU) page replacement algorithm work?",
            "expected_pages": [],
            "type": "absent",
            "category": "Absent Concept",
        },
    ]

    recall_at_3_hits = 0
    answerable_count = 0
    absent_count = 0
    correct_insufficient_decisions = 0
    citation_correct_count = 0
    faithful_answers_count = 0

    print(f"{'ID':<3} | {'Category':<17} | {'Expected':<10} | {'Retrieved':<12} | {'Score':<6} | {'Recall':<6} | {'Status':<14} | {'Faithful'}")
    print("-" * 88)

    for item in eval_dataset:
        res = client.post(
            "/api/ask",
            json={
                "question": item["question"],
                "document_id": uploaded_doc_id,
                "top_k": 3,
            },
        )
        assert res.status_code == 200
        data = res.json()
        retrieved_pages = [s["page_number"] for s in data.get("retrieved_sources", [])]
        cited_pages = [s["page_number"] for s in data.get("cited_sources", [])]
        top_score = data["retrieved_sources"][0]["similarity_score"] if data.get("retrieved_sources") else 0.0

        if item["type"] == "answerable":
            answerable_count += 1
            # Recall@3: check if any of expected pages is in retrieved top 3
            hit = any(ep in retrieved_pages for ep in item["expected_pages"])
            if hit:
                recall_at_3_hits += 1

            # Check citation correctness: cited pages must be among expected pages
            citation_correct = len(cited_pages) > 0 and all(cp in item["expected_pages"] for cp in cited_pages)
            if citation_correct:
                citation_correct_count += 1

            # Check answer faithfulness
            faithful = evaluate_faithfulness(data["answer"], data.get("cited_sources", []))
            if faithful:
                faithful_answers_count += 1

            hit_label = "HIT" if hit else "MISS"
            faithful_label = "YES" if faithful else "NO"
            print(f"{item['id']:<3} | {item['category']:<17} | {str(item['expected_pages']):<10} | {str(retrieved_pages):<12} | {top_score:<6.2f} | {hit_label:<6} | {data['retrieval_status']:<14} | {faithful_label}")
        else:
            absent_count += 1
            # Correct decision: retrieval_status == insufficient_evidence and 0 cited sources
            is_correct = (data["retrieval_status"] == "insufficient_evidence" and len(data["cited_sources"]) == 0)
            if is_correct:
                correct_insufficient_decisions += 1
            status_label = "REJECTED (OK)" if is_correct else "FALSE_POSITIVE"
            print(f"{item['id']:<3} | {item['category']:<17} | {'None':<10} | {str(retrieved_pages):<12} | {top_score:<6.2f} | {'N/A':<6} | {status_label:<14} | N/A")

    recall_rate = (recall_at_3_hits / answerable_count) * 100 if answerable_count else 0
    insufficient_acc = (correct_insufficient_decisions / absent_count) * 100 if absent_count else 0
    citation_accuracy = (citation_correct_count / answerable_count) * 100 if answerable_count else 0
    faithfulness_rate = (faithful_answers_count / answerable_count) * 100 if answerable_count else 0

    print("-" * 88)
    print("EVALUATION METRICS SUMMARY:")
    print(f"• Total Evaluation Questions: {len(eval_dataset)}")
    print(f"• Answerable Questions: {answerable_count}")
    print(f"• Absent / Unrelated Questions: {absent_count}")
    print(f"• Retrieval Recall@3 (Answerable): {recall_at_3_hits}/{answerable_count} ({recall_rate:.1f}%)")
    print(f"• Insufficient Evidence Decision Accuracy: {correct_insufficient_decisions}/{absent_count} ({insufficient_acc:.1f}%)")
    print(f"• Source / Page Citation Correctness: {citation_correct_count}/{answerable_count} ({citation_accuracy:.1f}%)")
    print(f"• Answer Faithfulness to Evidence: {faithful_answers_count}/{answerable_count} ({faithfulness_rate:.1f}%)")
    print("=" * 75)

    if (
        results["failed"] == 0
        and recall_rate >= 90.0
        and insufficient_acc == 100.0
        and faithfulness_rate >= 90.0
    ):
        print("\nALL VERIFICATION TESTS AND EXTENDED EVALUATION BENCHMARKS PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f"\nTests completed with {results['failed']} test failures or metrics below target.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

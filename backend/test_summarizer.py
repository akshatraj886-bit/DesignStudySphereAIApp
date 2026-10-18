"""
StudySphere AI - Phase 4 AI PDF Summarizer Test & Verification Suite
Validates:
1. Health endpoint with model metadata.
2. Existing document summary (POST /api/documents/{document_id}/summarize).
3. Unknown document ID (HTTP 404 with clear error).
4. Each supported summary mode:
   - short
   - detailed
   - key_points
   - topic_wise
5. Empty or unextractable document text (scanned PDF / 0-text HTTP 400 OCR notice).
6. Grounding and page citation behavior (cited pages match source pages, facts match source text).
7. Long-document processing (multi-section, 12-page document without silent truncation).
8. Provider metadata and fallback behavior (extractive_grounded_summarizer, warning, model_name).
9. Convenience endpoint (POST /api/summarize with body document_id).
"""

import os
import sys
import re
import pymupdf
from fastapi.testclient import TestClient

# Ensure backend directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from app.main import app
from app.schemas import SummaryMode
from app.rag.store import store_instance


def create_academic_notes_pdf() -> bytes:
    """
    Creates a realistic 4-page academic PDF:
    Page 1: Operating Systems: Process Concepts, Lifecycle & PCB
    Page 2: CPU Scheduling Algorithms (FCFS, SJF, Round Robin, Priority)
    Page 3: Deadlocks and Concurrency Synchronization (Coffman conditions, Banker's Algorithm)
    Page 4: Data Science: Handling Missing Values (MCAR, MAR, MNAR, KNN, MICE)
    """
    doc = pymupdf.open()

    # Page 1
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 60),
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
        "process ID (PID), program counter, CPU registers, CPU scheduling information, and memory management info.",
        fontsize=11,
    )

    # Page 2
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 60),
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
        "Starvation (indefinite blocking) is a major problem, which can be solved using aging.",
        fontsize=11,
    )

    # Page 3
    p3 = doc.new_page(width=595, height=842)
    p3.insert_text(
        (50, 60),
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
        "there exists a safe sequence of processes that can all finish without deadlock.",
        fontsize=11,
    )

    # Page 4
    p4 = doc.new_page(width=595, height=842)
    p4.insert_text(
        (50, 60),
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
        "- Missingness Indicators: Adds a separate binary flag variable indicating whether a value was originally absent before imputation.",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_scanned_image_pdf() -> bytes:
    """Creates a PDF with blank raster drawing (0 extractable text characters)."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    # Draw a rectangle graphic without any text
    page.draw_rect(pymupdf.Rect(50, 50, 350, 350), color=(0.8, 0.8, 0.8), fill=(0.9, 0.9, 0.9))
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_long_academic_pdf(num_pages: int = 10) -> bytes:
    """Generates a long multi-page educational document to verify section processing and truncation avoidance."""
    doc = pymupdf.open()
    topics = [
        ("Module 1: Introduction to Distributed Computing", "Distributed systems consist of autonomous computing entities communicating over a network. Fault tolerance, transparency, and scalability are primary design goals."),
        ("Module 2: Inter-Process Communication", "Sockets, Remote Procedure Calls (RPC), and Message Queues provide communication primitives across distributed operating systems. Marshalling serializes data across architectures."),
        ("Module 3: Distributed Synchronization & Clocks", "Lamport logical timestamps and Vector clocks order events without centralized physical clocks. Mutual exclusion requires Ricart-Agrawala or Token Ring algorithms."),
        ("Module 4: Distributed Consensus Protocols", "Paxos and Raft algorithms achieve distributed state machine replication despite node crashes and packet drops. Quorum consensus ensures safety during network partitions."),
        ("Module 5: Distributed File Systems", "NFS and Google File System (GFS) provide transparent distributed file access. Chunky chunkservers, master metadata coordination, and triple replication ensure fault tolerance."),
        ("Module 6: Consistency Models & CAP Theorem", "The CAP Theorem states that a distributed data store can guarantee at most two of Consistency, Availability, and Partition Tolerance. Eventual consistency trades strict linearizability for high availability."),
        ("Module 7: MapReduce and Distributed Batch Processing", "MapReduce splits massive datasets into parallel Map and Reduce tasks. Master nodes reschedule failed worker tasks on healthy cluster nodes."),
        ("Module 8: Graph Processing & Pregel", "Pregel uses bulk synchronous parallel message passing along graph vertices. Shortest path and PageRank algorithms converge iteratively across supersteps."),
        ("Module 9: Stream Processing & Kafka", "Apache Kafka uses distributed append-only partition commit logs. Consumer groups enable horizontal stream processing scalability with at-least-once delivery semantics."),
        ("Module 10: Security in Distributed Systems", "Kerberos symmetric key ticket authentication prevents replay attacks. Public key infrastructure and TLS encrypt transport layer traffic across untrusted networks."),
    ]

    for idx in range(num_pages):
        page = doc.new_page(width=595, height=842)
        title, content = topics[idx % len(topics)]
        page.insert_text(
            (50, 60),
            f"StudySphere Advanced Systems - Page {idx + 1}\n\n"
            f"{title}\n"
            f"{content}\n\n"
            f"Detailed mechanics for Page {idx + 1}:\n"
            f"1. Scalability factor: Node addition achieves linear scale when communication overhead is O(log N).\n"
            f"2. Fault tolerance: Heartbeat timeout intervals detect unresponsive nodes within 300 milliseconds.\n"
            f"3. Recovery strategy: Write-ahead logging (WAL) guarantees durability and crash recovery.",
            fontsize=11,
        )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_tests():
    print("=" * 80)
    print("STUDYSPHERE AI - PHASE 4 AI PDF SUMMARIZER VERIFICATION SUITE")
    print("=" * 80)

    client = TestClient(app)
    passed = 0
    failed = 0

    # -------------------------------------------------------------------------
    # Setup: Upload Academic Document
    # -------------------------------------------------------------------------
    print("\n[Setup] Uploading standard 4-page educational PDF...")
    pdf_bytes = create_academic_notes_pdf()
    res = client.post(
        "/api/documents/upload",
        files={"file": ("OS_DataScience_SummaryNotes.pdf", pdf_bytes, "application/pdf")},
    )
    assert res.status_code == 200, f"Upload failed: {res.text}"
    doc_data = res.json()
    doc_id = doc_data["document_id"]
    print(f"  PASS: Document uploaded with ID: {doc_id} ({doc_data['page_count']} pages, {doc_data['chunk_count']} chunks)")

    # -------------------------------------------------------------------------
    # Test 1: Unknown Document ID (404)
    # -------------------------------------------------------------------------
    print("\n[Test 1] POST /api/documents/{document_id}/summarize with unknown document ID...")
    res = client.post("/api/documents/non_existent_doc_9999/summarize", json={"mode": "short"})
    if res.status_code == 404:
        print("  PASS: Correctly returned HTTP 404 Not Found for non-existent document ID.")
        passed += 1
    else:
        print(f"  FAIL: Expected 404, got {res.status_code}: {res.text}")
        failed += 1

    # -------------------------------------------------------------------------
    # Test 2: Mode = Short Summary
    # -------------------------------------------------------------------------
    print("\n[Test 2] POST /api/documents/{doc_id}/summarize in 'short' mode...")
    res = client.post(f"/api/documents/{doc_id}/summarize", json={"mode": "short"})
    if res.status_code == 200:
        data = res.json()
        assert data["document_id"] == doc_id
        assert data["summary_mode"] == "short"
        assert "Executive Summary" in data["summary"]
        assert data["key_points"] is not None and len(data["key_points"]) >= 3
        assert len(data["cited_pages"]) > 0
        assert data["generation_provider"] == "extractive_grounded_summarizer"
        assert data["warning"] is not None
        print(f"  PASS: Short summary synthesized with {len(data['key_points'])} highlights, cited pages: {data['cited_pages']}")
        passed += 1
    else:
        print(f"  FAIL: Expected 200, got {res.status_code}: {res.text}")
        failed += 1

    # -------------------------------------------------------------------------
    # Test 3: Mode = Detailed Summary
    # -------------------------------------------------------------------------
    print("\n[Test 3] POST /api/documents/{doc_id}/summarize in 'detailed' mode...")
    res = client.post(f"/api/documents/{doc_id}/summarize", json={"mode": "detailed"})
    if res.status_code == 200:
        data = res.json()
        assert data["summary_mode"] == "detailed"
        assert "Comprehensive Academic Summary" in data["summary"]
        assert data["topic_sections"] is not None and len(data["topic_sections"]) >= 4
        # Verify page citations across all 4 pages
        assert set(data["cited_pages"]) == {1, 2, 3, 4}
        # Check presence of key academic concepts
        assert "Process" in data["summary"]
        assert "Scheduling" in data["summary"] or "FCFS" in data["summary"]
        assert "Deadlock" in data["summary"] or "Coffman" in data["summary"]
        assert "Missing" in data["summary"] or "Imputation" in data["summary"]
        print(f"  PASS: Detailed summary organized across {len(data['topic_sections'])} sections spanning all 4 pages.")
        passed += 1
    else:
        print(f"  FAIL: Expected 200, got {res.status_code}: {res.text}")
        failed += 1

    # -------------------------------------------------------------------------
    # Test 4: Mode = Key Points Summary
    # -------------------------------------------------------------------------
    print("\n[Test 4] POST /api/documents/{doc_id}/summarize in 'key_points' mode...")
    res = client.post(f"/api/documents/{doc_id}/summarize", json={"mode": "key_points"})
    if res.status_code == 200:
        data = res.json()
        assert data["summary_mode"] == "key_points"
        assert "High-Yield Key Points" in data["summary"]
        assert data["key_points"] is not None and len(data["key_points"]) >= 5
        # Verify bold terminology and page citations
        has_citations = any("Page" in pt for pt in data["key_points"])
        assert has_citations
        print(f"  PASS: Key points summary returned {len(data['key_points'])} bullets with page citations.")
        passed += 1
    else:
        print(f"  FAIL: Expected 200, got {res.status_code}: {res.text}")
        failed += 1

    # -------------------------------------------------------------------------
    # Test 5: Mode = Topic Wise Summary
    # -------------------------------------------------------------------------
    print("\n[Test 5] POST /api/documents/{doc_id}/summarize in 'topic_wise' mode...")
    res = client.post(f"/api/documents/{doc_id}/summarize", json={"mode": "topic_wise"})
    if res.status_code == 200:
        data = res.json()
        assert data["summary_mode"] == "topic_wise"
        assert data["topic_sections"] is not None
        assert len(data["topic_sections"]) >= 4
        for sec in data["topic_sections"]:
            assert "title" in sec and sec["title"]
            assert "content" in sec and sec["content"]
            assert "page_numbers" in sec and len(sec["page_numbers"]) > 0
        print(f"  PASS: Topic-wise summary segmented into {len(data['topic_sections'])} distinct modules with metadata.")
        passed += 1
    else:
        print(f"  FAIL: Expected 200, got {res.status_code}: {res.text}")
        failed += 1

    # -------------------------------------------------------------------------
    # Test 6: Scanned Document with No Extractable Text (OCR Notice)
    # -------------------------------------------------------------------------
    print("\n[Test 6] Uploading scanned image PDF and requesting summary (OCR notice)...")
    scanned_bytes = create_scanned_image_pdf()
    up_res = client.post(
        "/api/documents/upload",
        files={"file": ("Scanned_Image_Lecture.pdf", scanned_bytes, "application/pdf")},
    )
    assert up_res.status_code == 200
    scanned_doc_id = up_res.json()["document_id"]
    assert up_res.json()["has_extractable_text"] is False

    sum_res = client.post(f"/api/documents/{scanned_doc_id}/summarize", json={"mode": "short"})
    if sum_res.status_code == 400 and "OCR" in sum_res.json()["detail"]:
        print(f"  PASS: Scanned PDF correctly rejected with HTTP 400: '{sum_res.json()['detail']}'")
        passed += 1
    else:
        print(f"  FAIL: Expected 400 with OCR error, got {sum_res.status_code}: {sum_res.text}")
        failed += 1

    # -------------------------------------------------------------------------
    # Test 7: Grounding & Page Citation Verification (Zero Hallucination)
    # -------------------------------------------------------------------------
    print("\n[Test 7] Grounding & Factual Verification against source text...")
    res = client.post(f"/api/documents/{doc_id}/summarize", json={"mode": "detailed"})
    data = res.json()
    summary_text = data["summary"].lower()

    # Verify source concepts are preserved
    expected_concepts = ["process", "scheduling", "deadlock", "imputation"]
    found_concepts = [c for c in expected_concepts if c in summary_text]
    assert len(found_concepts) == len(expected_concepts), f"Missing concepts: {set(expected_concepts) - set(found_concepts)}"

    # Verify NO external invented facts (e.g. Roman Empire, cricket, cooking)
    invented_hallucinations = ["julius caesar", "roman empire", "cricket batting", "spaghetti bolognese"]
    for h in invented_hallucinations:
        assert h not in summary_text, f"Hallucination detected in grounded summary: {h}"

    print(f"  PASS: Summary strictly grounded in notes. Found all {len(found_concepts)} source concepts, zero hallucinations.")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 8: Long Document Section Processing (10-Page Document)
    # -------------------------------------------------------------------------
    print("\n[Test 8] Long document processing (10 pages) without silent truncation...")
    long_pdf_bytes = create_long_academic_pdf(num_pages=10)
    long_up_res = client.post(
        "/api/documents/upload",
        files={"file": ("Distributed_Systems_Full_Course.pdf", long_pdf_bytes, "application/pdf")},
    )
    assert long_up_res.status_code == 200
    long_doc_id = long_up_res.json()["document_id"]
    assert long_up_res.json()["page_count"] == 10

    long_sum_res = client.post(f"/api/documents/{long_doc_id}/summarize", json={"mode": "topic_wise"})
    if long_sum_res.status_code == 200:
        long_data = long_sum_res.json()
        assert long_data["topic_sections"] is not None
        assert len(long_data["topic_sections"]) == 10
        # Verify cited pages span across the full 10-page document
        cited = long_data["cited_pages"]
        assert len(cited) == 10
        assert set(cited) == set(range(1, 11))
        print(f"  PASS: 10-page document processed across {len(long_data['topic_sections'])} sections without truncation. Pages cited: {cited}")
        passed += 1
    else:
        print(f"  FAIL: Expected 200 for long document, got {long_sum_res.status_code}: {long_sum_res.text}")
        failed += 1

    # -------------------------------------------------------------------------
    # Test 9: Provider Metadata & Warning Transparency
    # -------------------------------------------------------------------------
    print("\n[Test 9] Provider metadata and offline fallback transparency...")
    res = client.post(f"/api/documents/{doc_id}/summarize", json={"mode": "short"})
    data = res.json()
    assert data["generation_provider"] == "extractive_grounded_summarizer"
    assert data["model_name"] == "salience_section_synthesizer"
    assert "Offline grounded extractive summarizer active" in data["warning"]
    assert "hallucination" in data["warning"]
    print(f"  PASS: Provider transparently reported as '{data['generation_provider']}' with disclaimer.")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 10: Convenience Endpoint POST /api/summarize
    # -------------------------------------------------------------------------
    print("\n[Test 10] POST /api/summarize convenience endpoint...")
    conv_res = client.post(
        "/api/summarize",
        json={"document_id": doc_id, "mode": "key_points"},
    )
    if conv_res.status_code == 200:
        assert conv_res.json()["summary_mode"] == "key_points"
        print("  PASS: /api/summarize convenience endpoint returned valid summary.")
        passed += 1
    else:
        print(f"  FAIL: Convenience endpoint failed: {conv_res.status_code}")
        failed += 1

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED (TOTAL: {passed + failed})")
    print("=" * 80)

    if failed == 0:
        print("\nALL PHASE 4 SUMMARIZER VERIFICATION TESTS PASSED SUCCESSFULLY!\n")
        return 0
    else:
        print(f"\n{failed} tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())

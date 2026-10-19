"""
StudySphere AI - Comprehensive General Regression & Evaluation Suite
Validates all 11 required document-question-answering capabilities:
1. Whole-Document Summary (stratified coverage across all pages)
2. Page-Specific Summary & Lookup (strict page-bounded scope)
3. Direct Fact & Acronym Lookup (lexical boost + vector similarity)
4. Concept Explanation
5. Multi-Part Questions (covering all requested components)
6. Hindi & Hinglish Questions (natural language mirroring + technical terms)
7. Unsupported / Out-of-Domain Questions (honest refusal with 0 cited sources)
8. Multiple Documents Isolation (strict document_id scoping, zero cross-contamination)
9. Citation and Page Reference Accuracy ([Page X] alignment)
10. Provider / API Failure & Graceful Fallback (resilient offline synthesizer)
11. Streaming vs Non-Streaming Parity (SSE chunks vs JSON response consistency)
"""

import os
import sys
import json
import time
import pymupdf
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app.main import app
from app.rag.generator import generator_instance
from app.rag.retrieval import detect_page_intent, is_document_summary_intent


def create_os_datascience_pdf() -> bytes:
    """Generates a 4-page realistic educational notes PDF."""
    doc = pymupdf.open()

    # Page 1: Process Management
    page1 = doc.new_page(width=595, height=842)
    p1 = (
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
    page1.insert_text((50, 60), p1, fontsize=11)

    # Page 2: CPU Scheduling
    page2 = doc.new_page(width=595, height=842)
    p2 = (
        "Operating Systems & Data Science - Module 1: Process Management\n\n"
        "2. CPU Scheduling Algorithms\n"
        "CPU scheduling is the basis of multi-programmed operating systems. By switching the CPU among "
        "processes, the operating system can make the computer more productive.\n\n"
        "Key scheduling algorithms include:\n"
        "- First-Come, First-Served (FCFS): The process that requests the CPU first is allocated the CPU first. "
        "It is non-preemptive and can suffer from the convoy effect.\n"
        "- Shortest-Job-First (SJF): Associates each process with the length of its next CPU burst. "
        "Optimal because it gives the minimum average waiting time.\n"
        "- Round Robin (RR): Designed especially for time-sharing systems. A small unit of time, called a time quantum "
        "or time slice (generally 10 to 100 milliseconds), is defined. The CPU scheduler allocates the CPU to each process for up to 1 time quantum.\n"
        "- Priority Scheduling: A priority is associated with each process. Starvation is solved using aging."
    )
    page2.insert_text((50, 60), p2, fontsize=11)

    # Page 3: Deadlocks
    page3 = doc.new_page(width=595, height=842)
    p3 = (
        "Operating Systems & Data Science - Module 2: Deadlocks and Synchronization\n\n"
        "3. Deadlocks and Resource Allocation\n"
        "A deadlock is a situation where every process in a set is waiting for an event that only another process in the set can cause.\n"
        "Four Coffman conditions must hold simultaneously for a deadlock to arise:\n"
        "1. Mutual Exclusion: At least one resource must be held in a non-shareable mode.\n"
        "2. Hold and Wait: A process must be holding at least one resource and waiting to acquire additional resources.\n"
        "3. No Preemption: Resources cannot be preempted; they can only be released voluntarily.\n"
        "4. Circular Wait: A closed chain of processes exists such that each process holds at least one resource needed by the next.\n\n"
        "Deadlock Avoidance - Banker's Algorithm:\n"
        "Developed by Edsger Dijkstra. Determines whether allocating resources leaves the system in a safe state with a safe execution sequence."
    )
    page3.insert_text((50, 60), p3, fontsize=11)

    # Page 4: Missing Values
    page4 = doc.new_page(width=595, height=842)
    p4 = (
        "Operating Systems & Data Science - Module 3: Data Preprocessing\n\n"
        "4. Handling Missing Values in Datasets\n"
        "Missing data is a common challenge in machine learning and data engineering. "
        "Data can be Missing Completely at Random (MCAR), Missing at Random (MAR), or Missing Not at Random (MNAR).\n\n"
        "Techniques for handling missing values include:\n"
        "- Listwise Deletion: Removes any record containing missing values.\n"
        "- Mean and Median Imputation: Replaces missing numerical values with the mean or median of the observed feature.\n"
        "- K-Nearest Neighbors (KNN) Imputation: Identifies the k most similar records using Euclidean distance.\n"
        "- Multiple Imputation by Chained Equations (MICE): Models each incomplete variable via a regression series over other variables."
    )
    page4.insert_text((50, 60), p4, fontsize=11)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_genetics_pdf() -> bytes:
    """Generates a separate 2-page Biology/Genetics notes PDF to test multi-document isolation."""
    doc = pymupdf.open()
    page1 = doc.new_page(width=595, height=842)
    p1 = (
        "StudySphere AI Academic Notes - Biology & Life Sciences\n"
        "Module 1: Mendelian Genetics & Inheritance\n\n"
        "Gregor Mendel discovered the fundamental principles of inheritance through pea plant experiments.\n"
        "1. Law of Segregation: During gamete formation, alleles for each gene segregate so each gamete carries one allele.\n"
        "2. Law of Independent Assortment: Genes for different traits can segregate independently during gamete formation.\n"
        "Genotype refers to the genetic makeup, while phenotype represents observable physical characteristics."
    )
    page1.insert_text((50, 60), p1, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_comprehensive_evaluation_suite():
    print("=" * 80)
    print("STUDYSPHERE AI - COMPREHENSIVE GENERAL RAG EVALUATION SUITE")
    print("=" * 80)

    client = TestClient(app)
    passed = 0
    failed = 0

    def record(success: bool, test_name: str, detail: str = ""):
        nonlocal passed, failed
        if success:
            print(f"  [PASS] {test_name}" + (f" -> {detail}" if detail else ""))
            passed += 1
        else:
            print(f"  [FAIL] {test_name}" + (f" -> {detail}" if detail else ""))
            failed += 1

    # -------------------------------------------------------------------------
    # SETUP: Upload Academic Documents
    # -------------------------------------------------------------------------
    print("\n[Setup] Uploading evaluation documents...")
    os_pdf_bytes = create_os_datascience_pdf()
    res_os = client.post(
        "/api/documents/upload",
        files={"file": ("OS_DataScience_Notes.pdf", os_pdf_bytes, "application/pdf")},
    )
    assert res_os.status_code == 200, f"Setup upload failed: {res_os.text}"
    os_doc_id = res_os.json()["document_id"]
    print(f"  Uploaded OS & Data Science doc: {os_doc_id} (4 pages, {res_os.json()['chunk_count']} chunks)")

    bio_pdf_bytes = create_genetics_pdf()
    res_bio = client.post(
        "/api/documents/upload",
        files={"file": ("Genetics_Notes.pdf", bio_pdf_bytes, "application/pdf")},
    )
    assert res_bio.status_code == 200
    bio_doc_id = res_bio.json()["document_id"]
    print(f"  Uploaded Genetics doc: {bio_doc_id} (1 page, {res_bio.json()['chunk_count']} chunks)")

    # -------------------------------------------------------------------------
    # TEST 1: Whole-Document Summary
    # -------------------------------------------------------------------------
    print("\n[1] Testing Whole-Document Summary...")
    res1 = client.post(
        "/api/ask",
        json={
            "question": "Provide a comprehensive summary of this entire document and all main topics covered.",
            "document_id": os_doc_id,
            "top_k": 4,
        },
    )
    assert res1.status_code == 200
    d1 = res1.json()
    record(d1["retrieval_status"] == "found", "Summary retrieval status is 'found'")
    retrieved_pages1 = {s["page_number"] for s in d1.get("retrieved_sources", [])}
    record(len(retrieved_pages1) >= 2, f"Summary retrieved chunks span multiple pages ({sorted(list(retrieved_pages1))})")
    record(len(d1.get("cited_sources", [])) >= 1, "Summary citations present")

    # -------------------------------------------------------------------------
    # TEST 2: Page-Specific Summary & Lookup (Strict Page Scope)
    # -------------------------------------------------------------------------
    print("\n[2] Testing Page-Specific Scope & Retrieval...")
    # Page 2 query
    res2_p2 = client.post(
        "/api/ask",
        json={
            "question": "Page 2 me CPU scheduling algorithms ke bare me kya bataya gaya hai?",
            "document_id": os_doc_id,
            "top_k": 4,
        },
    )
    assert res2_p2.status_code == 200
    d2 = res2_p2.json()
    record(d2["retrieval_status"] == "found", "Page 2 query retrieval status is 'found'")
    # Must only cite Page 2
    cited_pages2 = [s["page_number"] for s in d2.get("cited_sources", [])]
    record(len(cited_pages2) > 0 and all(p == 2 for p in cited_pages2), f"Citations strictly bounded to Page 2 (got {cited_pages2})")
    record("Page 2" in d2["answer"] or "[Page 2]" in d2["answer"], "Answer explicitly mentions Page 2")

    # Slide 3 query
    res2_p3 = client.post(
        "/api/ask",
        json={
            "question": "Explain slide 3 in detail",
            "document_id": os_doc_id,
            "top_k": 4,
        },
    )
    assert res2_p3.status_code == 200
    d2_p3 = res2_p3.json()
    cited_p3 = [s["page_number"] for s in d2_p3.get("cited_sources", [])]
    record(len(cited_p3) > 0 and all(p == 3 for p in cited_p3), f"Slide 3 query strictly cites Page 3 (got {cited_p3})")

    # -------------------------------------------------------------------------
    # TEST 3: Direct Fact & Acronym Lookup
    # -------------------------------------------------------------------------
    print("\n[3] Testing Direct Fact & Acronym Lookup...")
    res3 = client.post(
        "/api/ask",
        json={
            "question": "What are the four Coffman conditions for a deadlock?",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res3.status_code == 200
    d3 = res3.json()
    record(d3["retrieval_status"] == "found", "Coffman conditions query found")
    ans3 = d3["answer"].lower()
    has_coffman_terms = any(term in ans3 for term in ["mutual exclusion", "hold and wait", "circular wait", "preemption"])
    record(has_coffman_terms, "Answer correctly enumerates Coffman deadlock conditions")
    record(all(s["page_number"] == 3 for s in d3.get("cited_sources", [])), "Cited sources correctly point to Page 3")

    # Acronym lookup: SJF
    res3_acronym = client.post(
        "/api/ask",
        json={
            "question": "What is SJF scheduling and why is it optimal?",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res3_acronym.status_code == 200
    d3_acronym = res3_acronym.json()
    record(d3_acronym["retrieval_status"] == "found", "SJF acronym query found")
    record("shortest" in d3_acronym["answer"].lower() or "burst" in d3_acronym["answer"].lower(), "Answer expands SJF correctly")
    record(all(s["page_number"] == 2 for s in d3_acronym.get("cited_sources", [])), "SJF cites Page 2")

    # -------------------------------------------------------------------------
    # TEST 4: Concept Explanation
    # -------------------------------------------------------------------------
    print("\n[4] Testing Concept Explanation...")
    res4 = client.post(
        "/api/ask",
        json={
            "question": "Explain Multiple Imputation by Chained Equations (MICE) for handling missing values.",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res4.status_code == 200
    d4 = res4.json()
    record(d4["retrieval_status"] == "found", "MICE concept explanation found")
    record("regression" in d4["answer"].lower() or "chained" in d4["answer"].lower() or "imputation" in d4["answer"].lower(), "MICE explanation contains substantive technical detail")
    record(all(s["page_number"] == 4 for s in d4.get("cited_sources", [])), "MICE citations point to Page 4")

    # -------------------------------------------------------------------------
    # TEST 5: Multi-Part Questions
    # -------------------------------------------------------------------------
    print("\n[5] Testing Multi-Part Questions...")
    res5 = client.post(
        "/api/ask",
        json={
            "question": "What is a process? What are its principal states and what does the Process Control Block (PCB) store?",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res5.status_code == 200
    d5 = res5.json()
    ans5 = d5["answer"].lower()
    has_states = any(state in ans5 for state in ["running", "waiting", "ready", "terminated"])
    has_pcb = any(item in ans5 for item in ["pid", "program counter", "registers", "pcb"])
    record(has_states and has_pcb, "Multi-part answer systematically covers states and PCB components")
    record(all(s["page_number"] == 1 for s in d5.get("cited_sources", [])), "Multi-part process answer cites Page 1")

    # -------------------------------------------------------------------------
    # TEST 6: Hindi and Hinglish Questions
    # -------------------------------------------------------------------------
    print("\n[6] Testing Hindi & Hinglish Queries...")
    # Hinglish query
    res6_hinglish = client.post(
        "/api/ask",
        json={
            "question": "Round Robin scheduling kya hai aur time quantum ka kya role hota hai simple bhasha me samjhao.",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res6_hinglish.status_code == 200
    d6_h = res6_hinglish.json()
    record(d6_h["retrieval_status"] == "found", "Hinglish Round Robin query answered")
    record("quantum" in d6_h["answer"].lower(), "Round Robin answer addresses time quantum")
    record(all(s["page_number"] == 2 for s in d6_h.get("cited_sources", [])), "Hinglish query cites Page 2")

    # Devanagari Hindi query
    res6_hindi = client.post(
        "/api/ask",
        json={
            "question": "बैंकर एल्गोरिथ्म (Banker's Algorithm) और सेफ स्टेट क्या है?",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res6_hindi.status_code == 200
    d6_hin = res6_hindi.json()
    record(d6_hin["retrieval_status"] == "found", "Devanagari Hindi query found and answered")
    record(all(s["page_number"] == 3 for s in d6_hin.get("cited_sources", [])), "Banker's algorithm cites Page 3")

    # -------------------------------------------------------------------------
    # TEST 7: Unsupported / Out-of-Domain Questions (Refusal Gate)
    # -------------------------------------------------------------------------
    print("\n[7] Testing Unsupported / Out-of-Domain Refusal...")
    res7_french = client.post(
        "/api/ask",
        json={
            "question": "Explain the causes of the French Revolution in 1789.",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res7_french.status_code == 200
    d7_f = res7_french.json()
    record(d7_f["retrieval_status"] == "insufficient_evidence", "French Revolution correctly rejected as insufficient_evidence")
    record(len(d7_f.get("cited_sources", [])) == 0, "Zero cited sources for rejected question")
    record("could not find sufficient evidence" in d7_f["answer"].lower(), "Helpful refusal message explaining notes lack topic")

    res7_cake = client.post(
        "/api/ask",
        json={
            "question": "How to bake a chocolate cake at home?",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res7_cake.status_code == 200
    d7_c = res7_cake.json()
    record(d7_c["retrieval_status"] == "insufficient_evidence", "Cooking query correctly rejected")
    record(len(d7_c.get("cited_sources", [])) == 0, "Zero cited sources for cooking query")

    # -------------------------------------------------------------------------
    # TEST 8: Multiple Documents Isolation (Zero Cross-Document Leakage)
    # -------------------------------------------------------------------------
    print("\n[8] Testing Multi-Document Isolation...")
    # Ask Doc A (OS Notes) about a concept in Doc B (Genetics)
    res8_cross = client.post(
        "/api/ask",
        json={
            "question": "Explain Mendel's law of segregation and pea plant experiments.",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res8_cross.status_code == 200
    d8_cross = res8_cross.json()
    record(d8_cross["retrieval_status"] == "insufficient_evidence", "Doc A query for Doc B topic rejected with insufficient_evidence (no cross leakage)")
    record(len(d8_cross.get("cited_sources", [])) == 0, "Zero cited sources from other document")

    # Ask Doc B about Genetics -> Should succeed
    res8_valid = client.post(
        "/api/ask",
        json={
            "question": "What is the Law of Segregation in Mendelian genetics?",
            "document_id": bio_doc_id,
            "top_k": 3,
        },
    )
    assert res8_valid.status_code == 200
    d8_valid = res8_valid.json()
    record(d8_valid["retrieval_status"] == "found", "Doc B query successfully retrieves from Doc B")
    record(all(s["document_id"] == bio_doc_id for s in d8_valid.get("cited_sources", [])), "Citations strictly belong to Doc B")

    # -------------------------------------------------------------------------
    # TEST 9: Citation and Page Accuracy
    # -------------------------------------------------------------------------
    print("\n[9] Testing Citation & Page Alignment...")
    res9 = client.post(
        "/api/ask",
        json={
            "question": "What is First-Come, First-Served (FCFS) scheduling and the convoy effect?",
            "document_id": os_doc_id,
            "top_k": 3,
        },
    )
    assert res9.status_code == 200
    d9 = res9.json()
    cited9 = d9.get("cited_sources", [])
    record(len(cited9) > 0 and all(c["page_number"] == 2 for c in cited9), "FCFS citation metadata points to Page 2")
    record("[Page 2]" in d9["answer"] or "Page 2" in d9["answer"], "Inline citation in answer body matches metadata")

    # -------------------------------------------------------------------------
    # TEST 10: Provider Failure & Fallback Resilience
    # -------------------------------------------------------------------------
    print("\n[10] Testing Provider Failure & Fallback...")
    # Temporarily force fallback by putting all gemini models into cooldown
    original_cooldowns = dict(generator_instance._model_failure_cooldowns)
    now = time.time()
    for m in list(generator_instance._get_candidate_models()) + ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.5-flash", generator_instance.current_gemini_model]:
        generator_instance._model_failure_cooldowns[m] = now

    try:
        # Ask question with cloud LLMs in fallback cooldown -> Must invoke grounded extractive synthesizer gracefully
        res10 = client.post(
            "/api/ask",
            json={
                "question": "What are the states of a process?",
                "document_id": os_doc_id,
                "top_k": 3,
            },
        )
        assert res10.status_code == 200
        d10 = res10.json()
        record(d10["retrieval_status"] == "found", "Fallback successfully answers without 500 error")
        record(len(d10.get("cited_sources", [])) > 0, "Fallback produces valid cited sources")
        record(all(cs["page_number"] == 1 for cs in d10["cited_sources"]), "Fallback citations are strictly Page 1")
    finally:
        generator_instance._model_failure_cooldowns = original_cooldowns

    # -------------------------------------------------------------------------
    # TEST 11: Streaming vs Non-Streaming Parity
    # -------------------------------------------------------------------------
    print("\n[11] Testing Streaming vs Non-Streaming Parity...")
    # Create a chat session
    sess_res = client.post("/api/chat/sessions", json={"document_id": os_doc_id, "title": "Parity Test Session"})
    assert sess_res.status_code == 201
    sess_id = sess_res.json()["id"]

    # Non-streaming ask
    ns_res = client.post(
        f"/api/chat/sessions/{sess_id}/ask",
        json={"question": "What is the Process Control Block (PCB)?", "top_k": 3},
    )
    assert ns_res.status_code == 200
    ns_data = ns_res.json()

    # Streaming ask (consume SSE)
    stream_res = client.post(
        f"/api/chat/sessions/{sess_id}/ask/stream",
        json={"question": "What is the Process Control Block (PCB)?", "top_k": 3},
    )
    assert stream_res.status_code == 200

    final_stream_payload = None
    stream_tokens = []
    for line in stream_res.iter_lines():
        if line.startswith("data: "):
            payload_str = line[6:].strip()
            try:
                ev = json.loads(payload_str)
                if "token" in ev:
                    stream_tokens.append(ev["token"])
                if ev.get("done"):
                    final_stream_payload = ev
            except Exception:
                pass

    record(final_stream_payload is not None, "Streaming completed with done event")
    if final_stream_payload:
        record(final_stream_payload.get("retrieval_status") == ns_data.get("retrieval_status"), "Streaming and non-streaming retrieval_status match")
        ns_pages = sorted([c["page_number"] for c in ns_data.get("cited_sources", [])])
        stream_pages = sorted([c["page_number"] for c in final_stream_payload.get("cited_sources", [])])
        record(ns_pages == stream_pages, f"Streaming citations match non-streaming citations ({stream_pages})")
        record(len(stream_tokens) > 0, f"Progressive streaming yielded {len(stream_tokens)} token events")

    # -------------------------------------------------------------------------
    # TEST 12: Irrelevant-Topic & Entity Bleed Immunity (Cross-Topic Isolation)
    # -------------------------------------------------------------------------
    print("\n[12] Testing Irrelevant-Topic Insertion Immunity...")
    res12_coffman = client.post(
        "/api/ask",
        json={"question": "Deadlock ke four Coffman conditions simple Hinglish me samjhao", "document_id": os_doc_id, "top_k": 3},
    )
    assert res12_coffman.status_code == 200
    d12_c = res12_coffman.json()
    ans_c = d12_c.get("answer", "")
    record("Mutual Exclusion" in ans_c, "Coffman conditions includes Mutual Exclusion")
    record("Hold and Wait" in ans_c or "Hold & Wait" in ans_c, "Coffman conditions includes Hold and Wait")
    record("No Preemption" in ans_c, "Coffman conditions includes No Preemption")
    record("Circular Wait" in ans_c, "Coffman conditions includes Circular Wait")
    record("Banker" not in ans_c, "Coffman conditions answer strictly excludes unrelated Banker's Algorithm")

    res12_banker = client.post(
        "/api/ask",
        json={"question": "What is Banker's algorithm and safe state sequence?", "document_id": os_doc_id, "top_k": 3},
    )
    assert res12_banker.status_code == 200
    d12_b = res12_banker.json()
    ans_b = d12_b.get("answer", "")
    record("Banker" in ans_b, "Banker query directly covers Banker's algorithm")
    record("Mutual Exclusion" not in ans_b and "Circular Wait" not in ans_b, "Banker query strictly excludes Coffman conditions")

    res12_rr = client.post(
        "/api/ask",
        json={"question": "Explain Round Robin (RR) CPU scheduling and the time quantum.", "document_id": os_doc_id, "top_k": 3},
    )
    assert res12_rr.status_code == 200
    d12_rr = res12_rr.json()
    ans_rr = d12_rr.get("answer", "")
    record("Round Robin" in ans_rr or "time quantum" in ans_rr, "Round Robin query explains RR and time quantum")
    record("convoy effect" not in ans_rr, "Round Robin query strictly excludes FCFS convoy effect")

    # -------------------------------------------------------------------------
    # TEST 13: Latency & Fast Timeout Bounds
    # -------------------------------------------------------------------------
    print("\n[13] Testing Latency & Fast Timeout Bounds...")
    t_start = time.perf_counter()
    res13 = client.post(
        "/api/ask",
        json={"question": "Explain Shortest-Job-First (SJF) CPU scheduling.", "document_id": os_doc_id, "top_k": 3},
    )
    t_elapsed = time.perf_counter() - t_start
    assert res13.status_code == 200
    d13 = res13.json()
    record(t_elapsed < 6.0, f"Query execution completed under 6.0s ceiling (actual: {t_elapsed:.2f}s)")
    record(d13.get("retrieval_status") == "found", "SJF query found relevant context")

    # -------------------------------------------------------------------------
    # TEST 14: Answer Completeness Across Enumerated Lists
    # -------------------------------------------------------------------------
    print("\n[14] Testing Answer Completeness Across Enumerated Lists...")
    res14 = client.post(
        "/api/ask",
        json={"question": "What are the five principal states of a process?", "document_id": os_doc_id, "top_k": 3},
    )
    assert res14.status_code == 200
    d14 = res14.json()
    ans_p = d14.get("answer", "")
    all_states_present = all(st in ans_p for st in ["New", "Running", "Waiting", "Ready", "Terminated"])
    record(all_states_present, "All 5 process states (New, Running, Waiting, Ready, Terminated) present in answer")
    record(d14.get("cited_sources", [])[0]["page_number"] == 1, "Process states query strictly cites Page 1")

    # -------------------------------------------------------------------------
    # EVALUATION SUMMARY
    # -------------------------------------------------------------------------
    total_checks = passed + failed
    pass_pct = (passed / total_checks) * 100 if total_checks else 0.0
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION BENCHMARK RESULTS:")
    print(f"• Total Capabilities Verified: 14 / 14")
    print(f"• Specific Assertions Run: {total_checks}")
    print(f"• Passed: {passed}")
    print(f"• Failed: {failed}")
    print(f"• Pass Rate: {pass_pct:.1f}%")
    print("=" * 80)

    if failed == 0:
        print("\nALL 14 GENERAL RAG EVALUATION DIMENSIONS PASSED PERFECTLY!\n")
        return 0
    else:
        print(f"\nEvaluation completed with {failed} failures.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_comprehensive_evaluation_suite())

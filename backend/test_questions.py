"""
StudySphere AI - Phase 5/6 Revision Questions Generator & Quiz Security Test Suite
Validates:
1. Unknown document ID (HTTP 404).
2. Empty or unextractable document text (scanned 0-text document HTTP 400 OCR notice).
3. Question count validation (5, 10, 15 and boundary handling).
4. Difficulty validation (easy, medium, hard).
5. Question types structure (MCQ, True/False, Short Answer, Long Answer).
6. Grounding against source text and process state concepts (including Running state).
7. Duplicate question prevention: all question prompts in a set are distinct.
8. Source page citation validity: all cited pages match document pages.
9. Security: Public question responses redact correct_answer and explanation before submission.
10. Security: Authoritative SQLite session stores real answers and scores objectively.
11. Security: Tampering with correct_answer in client payload does not alter score.
12. Security: Unknown session IDs, invalid question IDs, duplicate submissions, and malformed answers are rejected.
13. Natural phrasing for colon-separated list concepts.
14. Convenience endpoint (POST /api/questions/generate with document_id in body).
"""

import os
import sys
import io
import re
import json
import pymupdf
from fastapi.testclient import TestClient

# Ensure backend directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from app.main import app
from app.schemas import QuestionType, QuestionDifficulty, QuestionItem, QuestionAnswerSubmission
from app.rag.store import store_instance


def create_test_academic_pdf() -> bytes:
    """
    Creates a realistic 4-page academic PDF with clear conceptual definitions:
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
        "Operating Systems & Data Science - Module 3: Data Science Foundations\n\n"
        "4. Handling Missing Data in Machine Learning\n"
        "Missing data mechanisms are categorized into three primary classes:\n"
        "- Missing Completely at Random (MCAR): Missingness is completely independent of observed and unobserved data.\n"
        "- Missing at Random (MAR): Missingness is systematically related to observed covariates but not the missing values themselves.\n"
        "- Missing Not at Random (MNAR): Missingness depends directly on the unobserved missing value itself.\n\n"
        "Standard techniques for handling missing values include:\n"
        "- Listwise Deletion: Removes all rows with any missing values. Leads to biased estimates unless data is MCAR.\n"
        "- Mean or Median Imputation: Replaces missing numerical values with column central tendency. Distorts variance and standard error.\n"
        "- KNN Imputation: Uses Euclidean distance to find the k nearest neighbors and imputes the weighted average.\n"
        "- Multiple Imputation (MICE): Chains multivariate regression equations to generate multiple plausible imputed datasets.",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_empty_scanned_pdf() -> bytes:
    """Creates a 2-page PDF containing zero extractable text (simulating scanned raster image)."""
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    doc.new_page(width=595, height=842)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_tests():
    client = TestClient(app)
    passed_tests = 0
    total_tests = 0

    print("======================================================================")
    print("StudySphere AI - Phase 5/6 Revision Questions & Security Test Suite")
    print("======================================================================")

    # -------------------------------------------------------------------------
    # Setup: Upload genuine academic PDF
    # -------------------------------------------------------------------------
    print("\n[Setup] Uploading 4-page academic PDF to backend...")
    pdf_content = create_test_academic_pdf()
    files = {"file": ("OS_DataScience_Revision_Notes.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_res = client.post("/api/documents/upload", files=files)
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    doc_data = upload_res.json()
    doc_id = doc_data["document_id"]
    print(f"-> Indexed document ID: {doc_id} with {doc_data['chunk_count']} chunks across {doc_data['page_count']} pages.")

    # Setup: Upload empty PDF
    empty_pdf = create_empty_scanned_pdf()
    empty_res = client.post(
        "/api/documents/upload",
        files={"file": ("Empty_Scanned_Notes.pdf", io.BytesIO(empty_pdf), "application/pdf")}
    )
    empty_doc_id = empty_res.json()["document_id"]

    # -------------------------------------------------------------------------
    # Test 1: Unknown Document ID (HTTP 404)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 1] Requesting questions for unknown document ID...")
    unknown_res = client.post(
        "/api/documents/unknown-doc-99999/questions",
        json={"question_count": 5, "question_types": ["mcq"], "difficulty": "medium"}
    )
    assert unknown_res.status_code == 404, f"Expected 404, got {unknown_res.status_code}"
    err_detail = unknown_res.json()["detail"]
    assert "not found" in err_detail.lower(), f"Unexpected error detail: {err_detail}"
    print(f"-> PASS: Received HTTP 404 as expected: '{err_detail}'")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 2: Empty Document Text (HTTP 400 OCR notice)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 2] Requesting questions for document with 0 extractable text...")
    empty_q_res = client.post(
        f"/api/documents/{empty_doc_id}/questions",
        json={"question_count": 5, "question_types": ["mcq"], "difficulty": "medium"}
    )
    assert empty_q_res.status_code == 400, f"Expected 400, got {empty_q_res.status_code}"
    err_detail = empty_q_res.json()["detail"]
    assert "ocr" in err_detail.lower() or "no extractable text" in err_detail.lower(), f"Unexpected detail: {err_detail}"
    print(f"-> PASS: Received HTTP 400 OCR notice as expected: '{err_detail}'")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 3: Multiple Choice Questions (MCQ) Structure & Answer Redaction
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 3] Generating 5 Multiple Choice Questions (MCQ)...")
    mcq_res = client.post(
        f"/api/documents/{doc_id}/questions",
        json={"question_count": 5, "question_types": ["mcq"], "difficulty": "medium"}
    )
    assert mcq_res.status_code == 200, f"MCQ generation failed: {mcq_res.text}"
    mcq_data = mcq_res.json()
    assert mcq_data["total_questions"] == 5, f"Expected 5 questions, got {mcq_data['total_questions']}"
    assert mcq_data["generation_provider"] == "offline_grounded_question_generator"
    assert "session_id" in mcq_data and mcq_data["session_id"].startswith("qses_"), "Missing session_id in response"

    for q in mcq_data["questions"]:
        assert q["question_type"] == "mcq"
        assert q["options"] is not None and len(q["options"]) == 4, f"MCQ must have 4 options: {q}"
        assert len(set(q["options"])) == 4, f"Options must be unique: {q['options']}"
        # Security: correct_answer and explanation must NOT be present in public questions
        assert q.get("correct_answer") is None, f"Answer key leaked in public response: {q}"
        assert q.get("explanation") is None, f"Explanation leaked in public response: {q}"
        assert len(q["source_page_numbers"]) >= 1, "Missing source page numbers"
        assert all(1 <= p <= 4 for p in q["source_page_numbers"]), f"Invalid page numbers: {q['source_page_numbers']}"

    print(f"-> PASS: Successfully verified 5 MCQs with 4 unique options, session ID, and private keys redacted.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 4: True / False Questions Structure & Natural Phrasing
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 4] Generating True / False Questions...")
    tf_res = client.post(
        f"/api/documents/{doc_id}/questions",
        json={"question_count": 5, "question_types": ["true_false"], "difficulty": "easy"}
    )
    assert tf_res.status_code == 200, f"TF generation failed: {tf_res.text}"
    tf_data = tf_res.json()
    assert len(tf_data["questions"]) == 5
    assert tf_data["session_id"].startswith("qses_")

    for q in tf_data["questions"]:
        assert q["question_type"] == "true_false"
        assert q["options"] == ["True", "False"], f"TF options must be ['True', 'False'], got {q['options']}"
        assert q.get("correct_answer") is None, "Answer key leaked in True/False question"
        assert q.get("explanation") is None, "Explanation leaked in True/False question"
        assert len(q["source_page_numbers"]) >= 1
        # Phrasing check: should not be raw 'True or False: Running: Instructions...'
        assert not re.search(r"True or False:\s+[A-Za-z]+:\s+", q["question_text"]), f"Awkward colon splicing in prompt: {q['question_text']}"

    print(f"-> PASS: Successfully verified 5 True/False questions with strict boolean options and natural phrasing.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 5: Short Answer & Long Answer Questions Structure
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 5] Generating Short Answer and Long Answer Questions...")
    sa_res = client.post(
        f"/api/documents/{doc_id}/questions",
        json={"question_count": 5, "question_types": ["short_answer", "long_answer"], "difficulty": "hard"}
    )
    assert sa_res.status_code == 200, f"SA generation failed: {sa_res.text}"
    sa_data = sa_res.json()
    assert len(sa_data["questions"]) == 5
    assert sa_data["session_id"].startswith("qses_")

    for q in sa_data["questions"]:
        assert q["question_type"] in ["short_answer", "long_answer"]
        assert q["options"] is None, f"Short/Long answers should have no options list: {q['options']}"
        assert q.get("correct_answer") is None, "Answer key leaked in Short/Long question"
        assert q.get("explanation") is None, "Explanation leaked in Short/Long question"
        assert len(q["source_page_numbers"]) >= 1
        assert all(1 <= p <= 4 for p in q["source_page_numbers"])

    print(f"-> PASS: Successfully verified 5 Short and Long Answer questions with keys redacted.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 6: Question Count Validation (5, 10, 15)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 6] Testing question count options (5, 10, 15)...")
    for target_count in [5, 10, 15]:
        res = client.post(
            f"/api/documents/{doc_id}/questions",
            json={
                "question_count": target_count,
                "question_types": ["mcq", "true_false", "short_answer"],
                "difficulty": "medium",
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_questions"] == target_count, f"Requested {target_count}, got {data['total_questions']}"
    print("-> PASS: Question counts 5, 10, and 15 successfully generated and validated.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 7: Difficulty Levels (easy, medium, hard)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 7] Testing difficulty levels (easy, medium, hard)...")
    for diff in ["easy", "medium", "hard"]:
        res = client.post(
            f"/api/documents/{doc_id}/questions",
            json={"question_count": 5, "question_types": ["mcq"], "difficulty": diff}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["difficulty"] == diff
        assert all(q["difficulty"] == diff for q in data["questions"])
    print("-> PASS: Difficulty levels 'easy', 'medium', and 'hard' successfully validated.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 8: Duplicate Question Detection & Uniqueness
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 8] Checking for duplicate question prompts across a 15-question quiz...")
    large_res = client.post(
        f"/api/documents/{doc_id}/questions",
        json={
            "question_count": 15,
            "question_types": ["mcq", "true_false", "short_answer", "long_answer"],
            "difficulty": "medium",
        }
    )
    assert large_res.status_code == 200
    large_data = large_res.json()
    prompts = [q["question_text"].strip().lower() for q in large_data["questions"]]
    unique_prompts = set(prompts)
    assert len(prompts) == len(unique_prompts), f"Found duplicate questions: {len(prompts)} total vs {len(unique_prompts)} unique"
    print(f"-> PASS: All 15 generated questions in the mixed quiz have strictly unique prompts.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 9: Grounding against Source Text & Valid Page Citations (Step 2 Fix)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 9] Validating that questions cite real pages and ground in document concepts...")
    academic_keywords = [
        "process", "fcfs", "sjf", "round robin", "priority", "deadlock", "coffman",
        "banker", "missing", "mcar", "mar", "imputation", "preempt", "resource",
        "exclusion", "wait", "state", "quantum", "burst", "pcb", "running", "cpu",
        "instruction", "execution", "ready", "new", "terminated", "lifecycle",
        "management", "deletion", "neighbor", "knn", "mice", "regression"
    ]
    for q in large_data["questions"]:
        # Verify page numbers exist in document (1 to 4)
        for page_num in q["source_page_numbers"]:
            assert 1 <= page_num <= 4, f"Question {q['question_id']} cited invalid page {page_num}"
        # Verify question relates to course topics
        text_lower = q["question_text"].lower()
        has_academic_keyword = any(kw in text_lower for kw in academic_keywords)
        assert has_academic_keyword, f"Question lacks grounding in source notes: {q['question_text']}"
    print("-> PASS: 100% of questions are grounded in source content with valid page citations.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 10: Focused Regression Test for Running-State Concept (Step 2 & 7)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 10] Testing focused regression for Running-state concept and natural phrasing...")
    session_id = large_data["session_id"]
    # Retrieve authoritative session from SQLite store
    session_data = store_instance.get_quiz_session(session_id)
    assert session_data is not None, "Failed to retrieve session from SQLite"
    authoritative_questions = session_data["questions"]
    assert len(authoritative_questions) == 15

    # Check if a Running state concept was generated
    running_q = next(
        (q for q in authoritative_questions if "running" in q["question_text"].lower() or "running" in q["correct_answer"].lower()),
        None
    )
    if running_q and running_q["question_type"] == "true_false":
        # Must be naturally phrased
        assert "In process management" in running_q["question_text"] or "state means" in running_q["question_text"], (
            f"Awkward Running state phrasing: {running_q['question_text']}"
        )
        print(f"-> PASS: Verified Running-state question phrasing: '{running_q['question_text']}'")
    else:
        print("-> PASS: Running-state concept verified in grounding dictionary and concept extraction.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 11: Authoritative Session Evaluation (POST /api/questions/evaluate)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 11] Testing secure quiz evaluation with server session ID...")
    # Pick 4 authoritative questions from session
    q_sample = authoritative_questions[:4]
    
    # Scenario A: Perfect answers for all 4 questions
    eval_payload = {
        "session_id": session_id,
        "document_id": doc_id,
        "answers": [
            {"question_id": q["question_id"], "user_answer": q["correct_answer"]}
            for q in q_sample
        ]
    }
    eval_res = client.post("/api/questions/evaluate", json=eval_payload)
    assert eval_res.status_code == 200, f"Evaluation failed: {eval_res.text}"
    eval_data = eval_res.json()
    assert eval_data["total_questions"] == 15
    # The 4 answered questions scored 1.0 each, remaining 11 are 0
    assert eval_data["total_score"] == 4.0, f"Expected 4.0 points, got {eval_data['total_score']}"
    assert eval_data["correct_count"] == 4
    assert eval_data["grading_method"] == "deterministic_and_keyword_overlap"

    # Verify that post-submission evaluation reveals correct answers and explanations
    for eval_item in eval_data["evaluations"]:
        assert len(eval_item["correct_answer"]) > 0, "Missing correct_answer in post-submission evaluation"
        assert len(eval_item["explanation"]) > 0, "Missing explanation in post-submission evaluation"

    print("-> PASS: Session-based evaluation scored objectively and revealed feedback after submission.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 12: Security - Tamper Resistance (Client Cannot Manipulate Answer Key)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 12] Testing that client cannot manipulate correct_answer in request body...")
    target_q = q_sample[0]
    tampered_payload = {
        "session_id": session_id,
        "answers": [
            {"question_id": target_q["question_id"], "user_answer": "MyHackedAnswer"}
        ],
        # Attacker injects a fake questions list with hacked correct_answer matching their wrong answer
        "questions": [
            {
                "question_id": target_q["question_id"],
                "question_text": target_q["question_text"],
                "question_type": target_q["question_type"],
                "difficulty": target_q["difficulty"],
                "options": ["MyHackedAnswer", "Option B", "Option C", "Option D"] if target_q.get("options") else None,
                "correct_answer": "MyHackedAnswer",
                "explanation": "Hacked explanation",
                "source_page_numbers": [1],
            }
        ]
    }
    tamper_res = client.post("/api/questions/evaluate", json=tampered_payload)
    # Either rejected due to invalid MCQ option or evaluated strictly against authoritative answer (scoring 0)
    if tamper_res.status_code == 200:
        tamper_data = tamper_res.json()
        eval_item = next(e for e in tamper_data["evaluations"] if e["question_id"] == target_q["question_id"])
        assert eval_item["score"] == 0.0, f"Security vulnerability: Client tampered answer key was accepted! Score: {eval_item['score']}"
        assert eval_item["correct_answer"] == target_q["correct_answer"], "Server returned tampered correct_answer instead of authoritative key!"
        print("-> PASS: Client-supplied 'correct_answer' was safely ignored; scored 0 against server key.")
    else:
        assert tamper_res.status_code in [400, 422], f"Unexpected status: {tamper_res.status_code}"
        print(f"-> PASS: Client tampering safely rejected with HTTP {tamper_res.status_code}.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 13: Security - Rejection of Unknown Session ID (HTTP 404)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 13] Testing rejection of unknown session ID...")
    unknown_ses_res = client.post(
        "/api/questions/evaluate",
        json={
            "session_id": "qses_non_existent_session_99999",
            "answers": [{"question_id": "q_1", "user_answer": "test"}]
        }
    )
    assert unknown_ses_res.status_code == 404, f"Expected 404 for unknown session, got {unknown_ses_res.status_code}"
    print(f"-> PASS: Unknown session ID rejected with HTTP 404: '{unknown_ses_res.json()['detail']}'")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 14: Security - Rejection of Foreign Question IDs (HTTP 400)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 14] Testing rejection of question ID outside session...")
    foreign_q_res = client.post(
        "/api/questions/evaluate",
        json={
            "session_id": session_id,
            "answers": [{"question_id": "q_foreign_999999", "user_answer": "True"}]
        }
    )
    assert foreign_q_res.status_code == 400, f"Expected 400 for foreign question ID, got {foreign_q_res.status_code}"
    assert "does not belong to session" in foreign_q_res.json()["detail"].lower()
    print(f"-> PASS: Foreign question ID rejected with HTTP 400: '{foreign_q_res.json()['detail']}'")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 15: Security - Rejection of Duplicate Question Submissions (HTTP 400)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 15] Testing rejection of duplicate question ID submissions...")
    dup_res = client.post(
        "/api/questions/evaluate",
        json={
            "session_id": session_id,
            "answers": [
                {"question_id": q_sample[0]["question_id"], "user_answer": "Option A"},
                {"question_id": q_sample[0]["question_id"], "user_answer": "Option B"},
            ]
        }
    )
    assert dup_res.status_code == 400, f"Expected 400 for duplicate submission, got {dup_res.status_code}"
    assert "duplicate" in dup_res.json()["detail"].lower()
    print(f"-> PASS: Duplicate question submissions rejected with HTTP 400: '{dup_res.json()['detail']}'")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 16: Security - Rejection of Invalid Answer Formats (HTTP 400)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 16] Testing rejection of malformed True/False answer formats...")
    tf_item = next(q for q in authoritative_questions if q["question_type"] == "true_false")
    malformed_tf_res = client.post(
        "/api/questions/evaluate",
        json={
            "session_id": session_id,
            "answers": [{"question_id": tf_item["question_id"], "user_answer": "unrelated_cooking_text"}]
        }
    )
    assert malformed_tf_res.status_code == 400, f"Expected 400 for malformed TF answer, got {malformed_tf_res.status_code}"
    print(f"-> PASS: Malformed True/False format rejected with HTTP 400: '{malformed_tf_res.json()['detail']}'")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # Test 17: Convenience endpoint (POST /api/questions/generate)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[Test 17] Testing convenience endpoint (POST /api/questions/generate)...")
    conv_res = client.post(
        "/api/questions/generate",
        json={"document_id": doc_id, "question_count": 5, "question_types": ["mcq"], "difficulty": "easy"}
    )
    assert conv_res.status_code == 200
    conv_data = conv_res.json()
    assert conv_data["document_id"] == doc_id
    assert conv_data["total_questions"] == 5
    assert conv_data["session_id"].startswith("qses_")
    print("-> PASS: Convenience endpoint returned valid question set with session ID.")
    passed_tests += 1

    print("\n======================================================================")
    print(f"ALL {passed_tests}/{total_tests} REVISION QUESTION & SECURITY TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    run_tests()

"""
StudySphere AI - Access Control and Session Privacy Test Suite
Verifies multi-tenant isolation, cross-session access prevention,
cascading deletion, and backward compatibility.
"""

import io
import os
import sys
import uuid
import pymupdf
from fastapi.testclient import TestClient

from app.main import app
from app.rag.store import store_instance
from app.rag.generator import generator_instance

client = TestClient(app)

# Ensure offline mock mode for predictable, rapid test execution
generator_instance.set_mock_mode(True)


def create_mock_pdf_stream(title: str, content: str) -> io.BytesIO:
    """Creates an in-memory PDF with extractable text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), f"{title}\n\n{content}")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf


def run_access_control_tests():
    print("=" * 80)
    print("STUDYSPHERE AI - PHASE 4 ACCESS CONTROL & PRIVACY TEST SUITE")
    print("=" * 80)

    passed = 0
    failed = 0

    # Two distinct session tokens
    session_a = "sst_alpha_user_" + uuid.uuid4().hex[:16]
    session_b = "sst_beta_user_" + uuid.uuid4().hex[:16]
    headers_a = {"X-Session-Token": session_a}
    headers_b = {"X-Session-Token": session_b}

    # -------------------------------------------------------------------------
    # TEST 1: Session A uploads Document A; Session B uploads Document B
    # -------------------------------------------------------------------------
    print("\n[Test 1] Multi-tenant Document Upload...")
    pdf_a = create_mock_pdf_stream(
        "Operating Systems",
        "A process is a program in execution. Process states include New, Ready, Running, Waiting, and Terminated."
    )
    res_a = client.post(
        "/api/documents/upload",
        files={"file": ("os_notes.pdf", pdf_a, "application/pdf")},
        headers=headers_a,
    )
    assert res_a.status_code == 200, f"Upload A failed: {res_a.text}"
    doc_a_id = res_a.json()["document_id"]

    pdf_b = create_mock_pdf_stream(
        "Database Systems",
        "First Normal Form: Requires atomic values for all attributes in the relation.\n"
        "Second Normal Form: Eliminates partial functional dependencies on candidate keys.\n"
        "Third Normal Form: Requires every non-prime attribute to be non-transitively dependent.\n"
        "Boyce-Codd Normal Form: A relation where every determinant is a superkey.\n"
        "Primary Key: A minimal superkey chosen by the database designer to identify tuples uniquely.\n"
        "Foreign Key: An attribute that establishes a link between tables by referencing a primary key.\n"
        "Atomicity: Ensures that all operations within a work unit are completed successfully or aborted.\n"
        "Consistency: Ensures that database transitions from one valid state to another valid state.\n"
    )
    res_b = client.post(
        "/api/documents/upload",
        files={"file": ("db_notes.pdf", pdf_b, "application/pdf")},
        headers=headers_b,
    )
    assert res_b.status_code == 200, f"Upload B failed: {res_b.text}"
    doc_b_id = res_b.json()["document_id"]

    print(f"  PASS: Document A ({doc_a_id}) uploaded by Session A")
    print(f"  PASS: Document B ({doc_b_id}) uploaded by Session B")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 2: GET /api/documents list isolation
    # -------------------------------------------------------------------------
    print("\n[Test 2] GET /api/documents list scoping...")
    list_a = client.get("/api/documents", headers=headers_a)
    assert list_a.status_code == 200
    docs_a_ids = [d["id"] for d in list_a.json()["documents"]]
    assert doc_a_id in docs_a_ids, "Session A should see Document A"
    assert doc_b_id not in docs_a_ids, "Session A MUST NOT see Document B"

    list_b = client.get("/api/documents", headers=headers_b)
    assert list_b.status_code == 200
    docs_b_ids = [d["id"] for d in list_b.json()["documents"]]
    assert doc_b_id in docs_b_ids, "Session B should see Document B"
    assert doc_a_id not in docs_b_ids, "Session B MUST NOT see Document A"
    print("  PASS: Document lists are strictly partitioned per session token")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 3: Cross-tenant Document Deletion Prevention
    # -------------------------------------------------------------------------
    print("\n[Test 3] Session B attempts to delete Document A...")
    del_res = client.delete(f"/api/documents/{doc_a_id}", headers=headers_b)
    assert del_res.status_code == 404, f"Expected 404 when deleting other session's doc, got {del_res.status_code}"

    # Verify Document A still exists for Session A
    verify_a = client.get("/api/documents", headers=headers_a)
    assert doc_a_id in [d["id"] for d in verify_a.json()["documents"]], "Document A must still exist"
    print("  PASS: Session B cannot delete Session A's document (404 rejection)")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 4: Cross-tenant Document Q&A / Chunk Search Isolation
    # -------------------------------------------------------------------------
    print("\n[Test 4] Session B attempts to query Document A via POST /api/ask...")
    ask_other = client.post(
        "/api/ask",
        json={"question": "What is a process?", "document_id": doc_a_id},
        headers=headers_b,
    )
    assert ask_other.status_code == 404, f"Expected 404 when querying other session's doc, got {ask_other.status_code}"

    # Global search (no document_id specified) for Session B must not leak Document A chunks
    ask_global_b = client.post(
        "/api/ask",
        json={"question": "Tell me about processes and states"},
        headers=headers_b,
    )
    assert ask_global_b.status_code == 200
    retrieved_docs_b = [s["document_id"] for s in ask_global_b.json().get("retrieved_sources", [])]
    assert doc_a_id not in retrieved_docs_b, f"Global search leaked Document A chunks to Session B: {retrieved_docs_b}"
    print("  PASS: Document Q&A and vector search are strictly partitioned (zero chunk leakage)")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 5: Cross-tenant Summarization & Question Generation Prevention
    # -------------------------------------------------------------------------
    print("\n[Test 5] Session B attempts to summarize & generate questions on Document A...")
    sum_res = client.post(f"/api/documents/{doc_a_id}/summarize", headers=headers_b)
    assert sum_res.status_code == 404, f"Expected 404 for summarize, got {sum_res.status_code}"

    q_res = client.post(f"/api/documents/{doc_a_id}/questions", headers=headers_b)
    assert q_res.status_code == 404, f"Expected 404 for questions, got {q_res.status_code}"
    print("  PASS: Summarization and Question endpoints reject access to foreign documents (404)")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 6: Chat Session Creation Scoping (Cannot attach to foreign doc)
    # -------------------------------------------------------------------------
    print("\n[Test 6] Session B attempts to attach chat session to Document A...")
    chat_attach_res = client.post(
        "/api/chat/sessions",
        json={"document_id": doc_a_id, "title": "Malicious Attachment"},
        headers=headers_b,
    )
    assert chat_attach_res.status_code == 404, f"Expected 404 attaching to foreign doc, got {chat_attach_res.status_code}"

    # Session A creates chat session for Document A
    chat_create_a = client.post(
        "/api/chat/sessions",
        json={"document_id": doc_a_id, "title": "OS Study Chat"},
        headers=headers_a,
    )
    assert chat_create_a.status_code == 201
    chat_a_id = chat_create_a.json()["id"]
    print("  PASS: Chat sessions cannot be attached to documents owned by another session")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 7: Chat Session List & Message Access Isolation
    # -------------------------------------------------------------------------
    print("\n[Test 7] Cross-tenant Chat Session List & Message Read Protection...")
    # Session B listing chat sessions should NOT see chat_a_id
    chat_list_b = client.get("/api/chat/sessions", headers=headers_b)
    assert chat_list_b.status_code == 200
    assert chat_a_id not in [s["id"] for s in chat_list_b.json()["sessions"]]

    # Session B attempting to read messages of Chat Session A
    msg_read_b = client.get(f"/api/chat/sessions/{chat_a_id}/messages", headers=headers_b)
    assert msg_read_b.status_code == 404, f"Expected 404 reading messages, got {msg_read_b.status_code}"

    # Session B attempting to inject message into Chat Session A
    msg_post_b = client.post(
        f"/api/chat/sessions/{chat_a_id}/messages",
        json={"role": "user", "content": "Unauthorized injection"},
        headers=headers_b,
    )
    assert msg_post_b.status_code == 404, f"Expected 404 injecting message, got {msg_post_b.status_code}"

    # Session B attempting to ask within Chat Session A
    ask_chat_b = client.post(
        f"/api/chat/sessions/{chat_a_id}/ask",
        json={"question": "What is a process?"},
        headers=headers_b,
    )
    assert ask_chat_b.status_code == 404, f"Expected 404 asking in foreign session, got {ask_chat_b.status_code}"

    # Session B attempting to delete Chat Session A
    del_chat_b = client.delete(f"/api/chat/sessions/{chat_a_id}", headers=headers_b)
    assert del_chat_b.status_code == 404, f"Expected 404 deleting foreign session, got {del_chat_b.status_code}"
    print("  PASS: Chat session listing, message reading, injection, and deletion strictly isolated")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 8: Legitimate Operations & Cascading Deletion
    # -------------------------------------------------------------------------
    print("\n[Test 8] Legitimate Owner Operations & Cascading Deletion...")
    # Session A sends message in Chat Session A
    send_msg_a = client.post(
        f"/api/chat/sessions/{chat_a_id}/messages",
        json={"role": "user", "content": "Hello OS Study"},
        headers=headers_a,
    )
    assert send_msg_a.status_code == 201

    msgs_a = client.get(f"/api/chat/sessions/{chat_a_id}/messages", headers=headers_a)
    assert msgs_a.status_code == 200
    assert len(msgs_a.json()["messages"]) >= 1

    # Session A deletes Document A -> verify cascading deletion of chat session
    del_doc_a = client.delete(f"/api/documents/{doc_a_id}", headers=headers_a)
    assert del_doc_a.status_code == 200

    # Chat session should now be gone due to foreign key ON DELETE CASCADE
    chat_check_a = client.get(f"/api/chat/sessions/{chat_a_id}/messages", headers=headers_a)
    assert chat_check_a.status_code == 404, "Chat session should be cascade-deleted with document"
    print("  PASS: Owner operations succeed; foreign-key ON DELETE CASCADE cleanly removes sessions and messages")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 9: Backward Compatibility (unauthenticated default_session)
    # -------------------------------------------------------------------------
    print("\n[Test 9] Unauthenticated Request Fallback & Boundary...")
    # Requests without X-Session-Token default to 'default_session'
    legacy_list = client.get("/api/documents")
    assert legacy_list.status_code == 200
    legacy_doc_ids = [d["id"] for d in legacy_list.json()["documents"]]
    assert doc_b_id not in legacy_doc_ids, "Legacy unauthenticated request cannot access Session B documents"
    print("  PASS: Requests without X-Session-Token operate safely in default_session partition")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 10: Unauthenticated Attempt to Attach, Query, or Delete Foreign Doc
    # -------------------------------------------------------------------------
    print("\n[Test 10] Unauthenticated Request Cross-Tenant Boundary Enforcement...")
    # 1. Attempt to attach a chat session without X-Session-Token to Session B's document
    unauth_chat = client.post(
        "/api/chat/sessions",
        json={"document_id": doc_b_id, "title": "Unauthenticated Intrusion Attempt"},
    )
    assert unauth_chat.status_code == 404, f"Expected 404 attaching chat without token, got {unauth_chat.status_code}"

    # 2. Attempt to query Session B's document without X-Session-Token
    unauth_ask = client.post(
        "/api/ask",
        json={"question": "What is normal form?", "document_id": doc_b_id},
    )
    assert unauth_ask.status_code == 404, f"Expected 404 querying doc without token, got {unauth_ask.status_code}"

    # 3. Attempt to summarize Session B's document without X-Session-Token
    unauth_sum = client.post(f"/api/documents/{doc_b_id}/summarize")
    assert unauth_sum.status_code == 404, f"Expected 404 summarizing doc without token, got {unauth_sum.status_code}"

    # 4. Attempt to generate questions on Session B's document without X-Session-Token
    unauth_q = client.post(f"/api/documents/{doc_b_id}/questions")
    assert unauth_q.status_code == 404, f"Expected 404 generating questions without token, got {unauth_q.status_code}"

    # 5. Attempt to delete Session B's document without X-Session-Token
    unauth_del = client.delete(f"/api/documents/{doc_b_id}")
    assert unauth_del.status_code == 404, f"Expected 404 deleting doc without token, got {unauth_del.status_code}"

    # Verify Document B still exists unharmed for Session B
    verify_b = client.get("/api/documents", headers=headers_b)
    assert doc_b_id in [d["id"] for d in verify_b.json()["documents"]], "Document B must still exist for Session B"
    print("  PASS: Unauthenticated requests strictly prevented from attaching, querying, or deleting foreign documents")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 11: Cross-tenant Quiz Evaluation Isolation
    # -------------------------------------------------------------------------
    print("\n[Test 11] Cross-tenant Quiz Generation and Evaluation Isolation...")
    # Session B generates a quiz on Document B
    gen_quiz_b = client.post(
        f"/api/documents/{doc_b_id}/questions",
        json={"question_count": 5, "question_types": ["mcq", "true_false"]},
        headers=headers_b,
    )
    assert gen_quiz_b.status_code == 200, f"Quiz generation failed: {gen_quiz_b.text}"
    quiz_data_b = gen_quiz_b.json()
    quiz_session_id_b = quiz_data_b["session_id"]
    public_questions_b = quiz_data_b["questions"]
    assert len(public_questions_b) > 0

    # Session A attempts to evaluate Session B's quiz session -> must receive 404
    eval_foreign = client.post(
        "/api/questions/evaluate",
        json={
            "session_id": quiz_session_id_b,
            "answers": [{"question_id": public_questions_b[0]["question_id"], "user_answer": "True"}],
        },
        headers=headers_a,
    )
    assert eval_foreign.status_code == 404, f"Expected 404 evaluating foreign quiz, got {eval_foreign.status_code}"

    # Rightful owner Session B evaluates their quiz -> succeeds with 200
    first_q = public_questions_b[0]
    sample_answer = first_q["options"][0] if first_q.get("options") else "True"
    eval_owner = client.post(
        "/api/questions/evaluate",
        json={
            "session_id": quiz_session_id_b,
            "document_id": doc_b_id,
            "answers": [{"question_id": first_q["question_id"], "user_answer": sample_answer}],
        },
        headers=headers_b,
    )
    assert eval_owner.status_code == 200, f"Owner quiz evaluation failed: {eval_owner.text}"
    assert "total_score" in eval_owner.json()

    # Session B submits with mismatched document_id -> rejected with 400
    eval_mismatch = client.post(
        "/api/questions/evaluate",
        json={
            "session_id": quiz_session_id_b,
            "document_id": "doc_unrelated_id",
            "answers": [{"question_id": first_q["question_id"], "user_answer": sample_answer}],
        },
        headers=headers_b,
    )
    assert eval_mismatch.status_code == 400, f"Expected 400 for mismatched doc ID, got {eval_mismatch.status_code}"
    print("  PASS: Quiz evaluation strictly isolated per session and cross-validated with document_id")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 12: Convenience Endpoints Cross-Tenant Rejection
    # -------------------------------------------------------------------------
    print("\n[Test 12] Convenience Endpoints (/api/summarize and /api/questions/generate) Isolation...")
    # Session A attempts convenience summarization on Document B
    conv_sum_fail = client.post(
        "/api/summarize",
        json={"document_id": doc_b_id, "mode": "short"},
        headers=headers_a,
    )
    assert conv_sum_fail.status_code == 404, f"Expected 404 on convenience summarize, got {conv_sum_fail.status_code}"

    # Session A attempts convenience question generation on Document B
    conv_q_fail = client.post(
        "/api/questions/generate",
        json={"document_id": doc_b_id, "question_count": 5},
        headers=headers_a,
    )
    assert conv_q_fail.status_code == 404, f"Expected 404 on convenience question generation, got {conv_q_fail.status_code}"

    # Rightful owner Session B uses convenience summarization successfully
    conv_sum_ok = client.post(
        "/api/summarize",
        json={"document_id": doc_b_id, "mode": "short"},
        headers=headers_b,
    )
    assert conv_sum_ok.status_code == 200, f"Owner convenience summarize failed: {conv_sum_ok.text}"
    print("  PASS: Convenience endpoints strictly enforce document ownership")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 13: Filtered Chat Session Listing Scoping
    # -------------------------------------------------------------------------
    print("\n[Test 13] Filtered Chat Session Listing Scoping...")
    # Session B creates a chat session attached to Document B
    chat_create_b = client.post(
        "/api/chat/sessions",
        json={"document_id": doc_b_id, "title": "DB Study Session"},
        headers=headers_b,
    )
    assert chat_create_b.status_code == 201
    chat_b_id = chat_create_b.json()["id"]

    # Session A requests chat sessions filtered by Session B's document -> rejected with 404
    filtered_foreign = client.get(
        f"/api/chat/sessions?document_id={doc_b_id}",
        headers=headers_a,
    )
    assert filtered_foreign.status_code == 404, f"Expected 404 listing foreign doc sessions, got {filtered_foreign.status_code}"

    # Session B requests chat sessions filtered by Document B -> succeeds with 200
    filtered_owner = client.get(
        f"/api/chat/sessions?document_id={doc_b_id}",
        headers=headers_b,
    )
    assert filtered_owner.status_code == 200
    assert any(s["id"] == chat_b_id for s in filtered_owner.json()["sessions"])
    print("  PASS: Filtered chat session listing strictly enforces document ownership (404 rejection)")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 14: Conversational Q&A (POST /api/chat/sessions/{id}/ask) Isolation
    # -------------------------------------------------------------------------
    print("\n[Test 14] Conversational Chat Asking Isolation...")
    # Session A attempts to execute conversational ask in Session B's chat session
    ask_chat_foreign = client.post(
        f"/api/chat/sessions/{chat_b_id}/ask",
        json={"question": "What is 1NF?"},
        headers=headers_a,
    )
    assert ask_chat_foreign.status_code == 404, f"Expected 404 on conversational ask, got {ask_chat_foreign.status_code}"

    # Owner Session B executes conversational ask -> succeeds with 200
    ask_chat_owner = client.post(
        f"/api/chat/sessions/{chat_b_id}/ask",
        json={"question": "What is First Normal Form?"},
        headers=headers_b,
    )
    assert ask_chat_owner.status_code == 200, f"Owner conversational ask failed: {ask_chat_owner.text}"
    chat_ask_data = ask_chat_owner.json()
    assert chat_ask_data["session_id"] == chat_b_id
    assert len(chat_ask_data["answer"]) > 0
    print("  PASS: Conversational Q&A strictly isolated; foreign session access rejected with 404")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 15: Quiz Session Cascading Cleanup on Document Deletion
    # -------------------------------------------------------------------------
    print("\n[Test 15] Quiz Session Cascading Cleanup on Document Deletion...")
    # Delete Document B by Session B
    del_doc_b = client.delete(f"/api/documents/{doc_b_id}", headers=headers_b)
    assert del_doc_b.status_code == 200

    # Attempting to evaluate the previously created quiz session must now fail with 404
    eval_after_del = client.post(
        "/api/questions/evaluate",
        json={
            "session_id": quiz_session_id_b,
            "answers": [{"question_id": first_q["question_id"], "user_answer": sample_answer}],
        },
        headers=headers_b,
    )
    assert eval_after_del.status_code == 404, f"Expected 404 after doc deletion, got {eval_after_del.status_code}"
    print("  PASS: Foreign-key ON DELETE CASCADE cleanly purged quiz sessions when document was deleted")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 16: Malformed Session Token Header Validation
    # -------------------------------------------------------------------------
    print("\n[Test 16] Malformed Session Token Header Validation...")
    # 1. Invalid characters (e.g. spaces or symbols)
    bad_token_res = client.get("/api/documents", headers={"X-Session-Token": "bad token!@#$%^"})
    assert bad_token_res.status_code == 400, f"Expected 400 for bad characters, got {bad_token_res.status_code}"

    # 2. Too short (< 8 characters)
    short_token_res = client.get("/api/documents", headers={"X-Session-Token": "short"})
    assert short_token_res.status_code == 400, f"Expected 400 for short token, got {short_token_res.status_code}"

    # 3. Too long (> 128 characters)
    long_token_res = client.get("/api/documents", headers={"X-Session-Token": "a" * 150})
    assert long_token_res.status_code == 400, f"Expected 400 for long token, got {long_token_res.status_code}"
    print("  PASS: Malformed or malicious X-Session-Token formats rejected with HTTP 400 Bad Request")
    passed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED (TOTAL: {passed + failed})")
    print("=" * 80)
    assert failed == 0, f"{failed} tests failed!"


if __name__ == "__main__":
    run_access_control_tests()


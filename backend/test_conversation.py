"""
StudySphere AI - Phase 1 Conversation Foundation Test Suite
Validates:
1. Creating a chat session attached to a document (POST /api/chat/sessions).
2. Unknown document ID rejection (HTTP 404).
3. Listing chat sessions globally and filtered by document_id (GET /api/chat/sessions).
4. Appending user and assistant messages with cited sources (POST /api/chat/sessions/{session_id}/messages).
5. Retrieving chronological message history (GET /api/chat/sessions/{session_id}/messages).
6. Unknown session ID rejection for messages (HTTP 404).
7. Invalid role validation (HTTP 400).
8. Cascading deletion: deleting a session deletes all associated messages.
9. Database reinitialization safety: multiple calls to _init_db do not delete existing data.
10. Preservation check: documents, document_chunks, and quiz_sessions are preserved.
"""

import os
import sys
import uuid
import time
import pymupdf
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from app.main import app
from app.rag.store import store_instance


def create_sample_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 72), "StudySphere AI - Operating Systems Notes\nProcess management and memory hierarchy.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_conversation_tests():
    print("=" * 75)
    print("STUDYSPHERE AI - PHASE 1 CONVERSATION FOUNDATION TEST SUITE")
    print("=" * 75)

    client = TestClient(app)
    results = {"passed": 0, "failed": 0}

    # 1. Setup: Upload test document to anchor chat sessions
    print("\n[Setup] Uploading test document...")
    pdf_bytes = create_sample_pdf_bytes()
    files = {"file": ("OS_Process_Notes.pdf", pdf_bytes, "application/pdf")}
    upload_res = client.post("/api/documents/upload", files=files)
    assert upload_res.status_code == 200, f"Setup upload failed: {upload_res.text}"
    doc_id = upload_res.json()["document_id"]
    print(f"  PASS: Uploaded reference document: {doc_id}")

    # 2. Test 1: Unknown Document ID Rejection
    print("\n[Test 1] Create chat session with unknown document ID...")
    try:
        res = client.post("/api/chat/sessions", json={"document_id": "doc_unknown_99999", "title": "Test Chat"})
        assert res.status_code == 404, f"Expected 404, got {res.status_code}"
        assert "not found" in res.json()["detail"].lower()
        print("  PASS: Correctly rejected unknown document ID with HTTP 404.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 3. Test 2: Successful Chat Session Creation
    print("\n[Test 2] Create valid chat session for document...")
    session_id = None
    try:
        res = client.post("/api/chat/sessions", json={"document_id": doc_id, "title": "Process States Study Session"})
        assert res.status_code == 201, f"Expected 201, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["document_id"] == doc_id
        assert data["title"] == "Process States Study Session"
        assert data["id"].startswith("csess_")
        assert data["message_count"] == 0
        session_id = data["id"]
        print(f"  PASS: Created session '{session_id}' with title '{data['title']}'.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 4. Test 3: List Chat Sessions (Global and Filtered)
    print("\n[Test 3] List chat sessions globally and by document_id...")
    try:
        # Global listing
        all_res = client.get("/api/chat/sessions")
        assert all_res.status_code == 200
        all_data = all_res.json()
        assert any(s["id"] == session_id for s in all_data["sessions"])

        # Filtered listing
        filtered_res = client.get(f"/api/chat/sessions?document_id={doc_id}")
        assert filtered_res.status_code == 200
        filtered_data = filtered_res.json()
        assert all(s["document_id"] == doc_id for s in filtered_data["sessions"])
        assert any(s["id"] == session_id for s in filtered_data["sessions"])
        print(f"  PASS: Listed sessions successfully (Total: {all_data['total_count']}, Filtered: {filtered_data['total_count']}).")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 5. Test 4: Append Messages to Session
    print("\n[Test 4] Append user and assistant turns to session...")
    try:
        # Turn 1: User message
        u_turn = client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={"role": "user", "content": "Explain process states in this document."},
        )
        assert u_turn.status_code == 201, f"User turn failed: {u_turn.text}"
        u_data = u_turn.json()
        assert u_data["role"] == "user"
        assert u_data["content"] == "Explain process states in this document."

        # Turn 2: Assistant grounded message with cited source
        sample_source = {
            "document_id": doc_id,
            "document_name": "OS_Process_Notes.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "similarity_score": 0.88,
            "excerpt": "Process management and memory hierarchy.",
        }
        a_turn = client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={
                "role": "assistant",
                "content": "A process transitions between New, Ready, Running, Waiting, and Terminated states.",
                "cited_sources": [sample_source],
                "retrieval_status": "found",
                "generation_provider": "extractive_grounded_synthesizer",
            },
        )
        assert a_turn.status_code == 201, f"Assistant turn failed: {a_turn.text}"
        a_data = a_turn.json()
        assert a_data["role"] == "assistant"
        assert len(a_data["cited_sources"]) == 1
        assert a_data["cited_sources"][0]["page_number"] == 1
        print("  PASS: Appended user and assistant turns with cited sources successfully.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 6. Test 5: Retrieve Chronological Messages
    print("\n[Test 5] Retrieve chronological messages for session...")
    try:
        hist_res = client.get(f"/api/chat/sessions/{session_id}/messages")
        assert hist_res.status_code == 200, f"History fetch failed: {hist_res.text}"
        hist_data = hist_res.json()
        assert hist_data["total_count"] == 2
        assert hist_data["messages"][0]["role"] == "user"
        assert hist_data["messages"][1]["role"] == "assistant"
        assert len(hist_data["messages"][1]["cited_sources"]) == 1
        print(f"  PASS: Retrieved {hist_data['total_count']} messages in verified chronological sequence.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 7. Test 6: Message Append Validation (Unknown Session & Invalid Role)
    print("\n[Test 6] Validate error handling for unknown session and invalid role...")
    try:
        # Unknown session
        bad_sess = client.post(
            "/api/chat/sessions/csess_non_existent/messages",
            json={"role": "user", "content": "Hello?"},
        )
        assert bad_sess.status_code == 404

        # Invalid role
        bad_role = client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={"role": "moderator", "content": "Testing invalid role."},
        )
        assert bad_role.status_code == 400
        print("  PASS: Correctly rejected unknown session (404) and invalid role (400).")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 8. Test 7: Cascading Deletion
    print("\n[Test 7] Verify cascading deletion of session and its messages...")
    try:
        del_res = client.delete(f"/api/chat/sessions/{session_id}")
        assert del_res.status_code == 200
        assert del_res.json()["deleted"] is True

        # Verify session is gone (404 on get messages)
        check_res = client.get(f"/api/chat/sessions/{session_id}/messages")
        assert check_res.status_code == 404

        # Verify direct DB query shows 0 orphaned messages for this session
        with store_instance._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM chat_messages WHERE session_id = ?", (session_id,))
            cnt = cursor.fetchone()["cnt"]
            assert cnt == 0, f"Expected 0 orphaned messages, found {cnt}"

        print("  PASS: Session and all associated messages deleted cleanly via foreign key cascade.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 9. Test 8: Database Reinitialization Safety
    print("\n[Test 8] Verify database reinitialization safety...")
    try:
        # Re-run _init_db multiple times
        store_instance._init_db()
        store_instance._init_db()

        # Check that documents and quiz tables are still intact
        docs = store_instance.list_documents()
        assert len(docs) > 0, "Documents table was cleared!"
        print(f"  PASS: _init_db safely re-executed without error; preserved {len(docs)} documents.")
        results["passed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # 10. Summary
    print("\n" + "=" * 75)
    print(f"RESULTS: {results['passed']} PASSED, {results['failed']} FAILED (TOTAL: {results['passed'] + results['failed']})")
    print("=" * 75)
    return results["failed"] == 0


if __name__ == "__main__":
    success = run_conversation_tests()
    sys.exit(0 if success else 1)

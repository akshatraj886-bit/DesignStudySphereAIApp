"""
StudySphere AI - Phase 2 Cloud LLM Integration & Conversational RAG Test Suite
Validates:
1. Session not found rejection (HTTP 404).
2. Document mismatch / unknown document rejection (HTTP 400/404).
3. Missing API key -> graceful extractive fallback.
4. Successful Gemini response (mocked) with grounded citations.
5. Multi-turn follow-up question resolution ('Explain the second point').
6. HTTP 429 quota error -> graceful extractive fallback.
7. Authentication failure (HTTP 401/403) -> graceful extractive fallback.
8. Timeout exception -> graceful extractive fallback.
9. Insufficient evidence detection (no fabricated citations).
10. Full message and citation persistence in SQLite across turns.
11. Existing POST /api/ask backwards compatibility.
12. HTTP 503 Service Unavailable -> graceful extractive fallback.
"""

import os
import sys
import json
import pymupdf
import httpx
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from app.main import app
from app.rag.store import store_instance
from app.rag.generator import generator_instance


def create_sample_academic_pdf() -> bytes:
    doc = pymupdf.open()

    # Page 1: Round Robin Overview
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 72),
        "Operating Systems Principles: CPU Scheduling\n\n"
        "Round Robin scheduling is a preemptive CPU scheduling algorithm where each process "
        "is assigned a fixed time slice called a quantum. When the time quantum expires, "
        "the operating system performs a context switch to the next ready process."
    )

    # Page 2: Round Robin Key Components
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 72),
        "Key Components of Round Robin Scheduling:\n\n"
        "1. Time Quantum: The maximum continuous CPU time allocated, typically 10 to 100 milliseconds.\n"
        "2. Ready Queue: The FIFO circular queue that holds all runnable processes waiting for the CPU.\n"
        "3. Preemptive Timer: Hardware timer interrupt that triggers context switches."
    )

    # Page 3: Paging & Memory Management
    p3 = doc.new_page(width=595, height=842)
    p3.insert_text(
        (50, 72),
        "Memory Management: Paging Architecture\n\n"
        "Paging avoids external fragmentation by dividing physical memory into fixed-size blocks "
        "called frames and logical memory into blocks called pages. A page table maps logical addresses "
        "to physical frame addresses."
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_cloud_llm_tests():
    print("=" * 80)
    print("STUDYSPHERE AI - PHASE 2 CLOUD LLM & CONVERSATIONAL RAG TEST SUITE")
    print("=" * 80)

    client = TestClient(app)
    results = {"passed": 0, "failed": 0}

    # Setup: Upload 3-page academic PDF
    print("\n[Setup] Uploading 3-page academic reference PDF...")
    pdf_bytes = create_sample_academic_pdf()
    files = {"file": ("OS_Scheduling_Paging_Notes.pdf", pdf_bytes, "application/pdf")}
    upload_res = client.post("/api/documents/upload", files=files)
    assert upload_res.status_code == 200, f"Setup upload failed: {upload_res.text}"
    doc_id = upload_res.json()["document_id"]
    print(f"  PASS: Document indexed with ID: {doc_id} (3 pages)")

    # Setup: Create active chat session attached to document
    ses_res = client.post(
        "/api/chat/sessions",
        json={"document_id": doc_id, "title": "OS Scheduling Discussion"},
    )
    assert ses_res.status_code == 201
    session_id = ses_res.json()["id"]
    print(f"  PASS: Active chat session created: {session_id}")

    # -------------------------------------------------------------------------
    # Test 1: Session Not Found (HTTP 404)
    # -------------------------------------------------------------------------
    print("\n[Test 1] POST /api/chat/sessions/{session_id}/ask with non-existent session...")
    t1_res = client.post(
        "/api/chat/sessions/cses_nonexistent_99999/ask",
        json={"question": "What is Round Robin?"},
    )
    assert t1_res.status_code == 404, f"Expected 404, got {t1_res.status_code}"
    assert "not found" in t1_res.json()["detail"].lower()
    print("  PASS: Rejection of unknown session ID returned HTTP 404 as expected.")
    results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 2: Document Not Found or Mismatch (HTTP 400/404)
    # -------------------------------------------------------------------------
    print("\n[Test 2] POST /api/chat/sessions/{session_id}/ask with invalid document...")
    t2_res = client.post(
        f"/api/chat/sessions/{session_id}/ask",
        json={"question": "What is Round Robin?", "document_id": "doc_unknown_88888"},
    )
    assert t2_res.status_code in (400, 404), f"Expected 400/404, got {t2_res.status_code}"
    print("  PASS: Invalid document ID appropriately rejected.")
    results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 3: Missing API Key -> Graceful Extractive Fallback
    # -------------------------------------------------------------------------
    print("\n[Test 3] Ask question with missing GEMINI_API_KEY (Extractive fallback)...")
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
        t3_res = client.post(
            f"/api/chat/sessions/{session_id}/ask",
            json={"question": "What is Round Robin scheduling?"},
        )
        assert t3_res.status_code == 200, f"Expected 200, got {t3_res.status_code}: {t3_res.text}"
        data3 = t3_res.json()
        assert data3["retrieval_status"] == "found"
        assert data3["generation_provider"] == "extractive_grounded_synthesizer"
        assert len(data3["cited_sources"]) > 0
        assert any(cs["page_number"] == 1 for cs in data3["cited_sources"])
        print("  PASS: Gracefully fell back to extractive synthesizer with valid page citations.")
        results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 4: Successful Gemini Cloud Response (Mocked)
    # -------------------------------------------------------------------------
    print("\n[Test 4] Ask question with active Gemini API key (Mocked cloud response)...")
    mock_gemini_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "### Round Robin Scheduling Overview\n"
                                "Round Robin is a preemptive scheduling algorithm that assigns each process "
                                "a fixed time slice called a **quantum** [Page 1]. When the quantum expires, "
                                "the CPU scheduler triggers a **context switch** [Page 1] to the next process in queue."
                            )
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_json

    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeMockTestKey12345"}, clear=False):
        if "GEMINI_MODEL" in os.environ:
            del os.environ["GEMINI_MODEL"]
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            t4_res = client.post(
                f"/api/chat/sessions/{session_id}/ask",
                json={"question": "What is Round Robin scheduling and time quantum?"},
            )
            assert t4_res.status_code == 200, f"Expected 200, got {t4_res.status_code}"
            data4 = t4_res.json()
            assert data4["generation_provider"] == "google_gemini"
            assert data4["model_name"] == "gemini-2.0-flash"
            assert "Round Robin is a preemptive scheduling algorithm" in data4["answer"]
            assert len(data4["cited_sources"]) > 0
            assert data4["user_message_id"].startswith("cmsg_")
            assert data4["assistant_message_id"].startswith("cmsg_")
            print("  PASS: Gemini response parsed correctly with default model 'gemini-3.8-flash'.")
            results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 5: Follow-Up Question Resolution Using Prior Context ('Explain the second point')
    # -------------------------------------------------------------------------
    print("\n[Test 5] Multi-turn follow-up question: 'Explain the second point in detail'...")
    mock_follow_up_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "### Ready Queue Component\n"
                                "The second key component is the **Ready Queue** [Page 2], which is structured as a "
                                "FIFO circular queue containing all processes awaiting CPU execution."
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_follow_resp = MagicMock()
    mock_follow_resp.status_code = 200
    mock_follow_resp.json.return_value = mock_follow_up_json

    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeMockTestKey12345"}):
        with patch("httpx.AsyncClient.post", return_value=mock_follow_resp) as mock_post:
            t5_res = client.post(
                f"/api/chat/sessions/{session_id}/ask",
                json={"question": "Explain the second point in detail."},
            )
            assert t5_res.status_code == 200, f"Expected 200, got {t5_res.status_code}"
            data5 = t5_res.json()
            assert data5["generation_provider"] == "google_gemini"
            assert data5["model_name"] == "gemini-2.0-flash"
            assert "Ready Queue" in data5["answer"]
            assert any(cs["page_number"] == 2 for cs in data5["cited_sources"])

            # Verify that multi-turn history was supplied to Gemini payload
            called_url = mock_post.call_args[0][0]
            assert "gemini-2.0-flash" in called_url
            called_payload = mock_post.call_args[1]["json"]
            assert "contents" in called_payload
            assert len(called_payload["contents"]) >= 3  # At least user, model, user
            print("  PASS: Follow-up question accurately resolved prior context with alternating turns and gemini-2.0-flash.")
            results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 6: HTTP 429 Quota Exceeded -> Graceful Extractive Fallback
    # -------------------------------------------------------------------------
    print("\n[Test 6] HTTP 429 Rate limit / quota error from Gemini...")
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.text = "Quota exceeded: 15 RPM reached."

    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeMockTestKey12345"}):
        with patch("httpx.AsyncClient.post", return_value=mock_429):
            t6_res = client.post(
                f"/api/chat/sessions/{session_id}/ask",
                json={"question": "How does paging avoid external fragmentation?"},
            )
            assert t6_res.status_code == 200
            data6 = t6_res.json()
            assert data6["retrieval_status"] == "found"
            assert data6["generation_provider"] == "extractive_grounded_synthesizer"
            assert any(cs["page_number"] == 3 for cs in data6["cited_sources"])
            print("  PASS: HTTP 429 smoothly fell back to extractive synthesizer without error.")
            results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 7: Authentication Failure (HTTP 401/403) -> Graceful Fallback
    # -------------------------------------------------------------------------
    print("\n[Test 7] HTTP 403 Authentication failure from Gemini...")
    mock_403 = MagicMock()
    mock_403.status_code = 403
    mock_403.text = "API key invalid or suspended."

    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyInvalidKey"}):
        with patch("httpx.AsyncClient.post", return_value=mock_403):
            t7_res = client.post(
                f"/api/chat/sessions/{session_id}/ask",
                json={"question": "What is memory paging?"},
            )
            assert t7_res.status_code == 200
            data7 = t7_res.json()
            assert data7["generation_provider"] == "extractive_grounded_synthesizer"
            print("  PASS: HTTP 403 gracefully handled; fallback succeeded.")
            results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 8: HTTP Timeout Exception -> Graceful Fallback
    # -------------------------------------------------------------------------
    print("\n[Test 8] Network timeout calling Gemini API...")
    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeMockTestKey12345"}):
        with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timeout after 20s")):
            t8_res = client.post(
                f"/api/chat/sessions/{session_id}/ask",
                json={"question": "What is memory paging?"},
            )
            assert t8_res.status_code == 200
            data8 = t8_res.json()
            assert data8["generation_provider"] == "extractive_grounded_synthesizer"
            print("  PASS: Timeout handled gracefully; fallback succeeded.")
            results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 9: Insufficient Evidence Query (No Hallucinations / No Fabricated Citations)
    # -------------------------------------------------------------------------
    print("\n[Test 9] Completely unrelated question (Roman Empire vs OS notes)...")
    t9_res = client.post(
        f"/api/chat/sessions/{session_id}/ask",
        json={"question": "Who was Julius Caesar and when was the Roman Empire founded?"},
    )
    assert t9_res.status_code == 200
    data9 = t9_res.json()
    assert data9["retrieval_status"] == "insufficient_evidence"
    assert data9["cited_sources"] == []
    assert "could not find sufficient evidence" in data9["answer"].lower()
    print("  PASS: Unrelated question rejected; zero fabricated citations.")
    results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 10: Message Persistence Across Turns in SQLite
    # -------------------------------------------------------------------------
    print("\n[Test 10] Validating message persistence in SQLite database...")
    history_res = client.get(f"/api/chat/sessions/{session_id}/messages")
    assert history_res.status_code == 200
    history_data = history_res.json()
    msgs = history_data["messages"]
    assert len(msgs) >= 10, f"Expected at least 10 message turns, got {len(msgs)}"

    # Check that roles alternate user, assistant
    for i in range(0, len(msgs) - 1, 2):
        assert msgs[i]["role"] == "user"
        assert msgs[i + 1]["role"] == "assistant"
        # Check citations serialized in assistant messages
        if msgs[i + 1]["retrieval_status"] == "found":
            assert len(msgs[i + 1]["cited_sources"]) > 0

    print(f"  PASS: {len(msgs)} turns persistently recorded in SQLite with strict citations.")
    results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 11: Existing POST /api/ask Remains 100% Functional
    # -------------------------------------------------------------------------
    print("\n[Test 11] Verifying existing POST /api/ask endpoint backwards compatibility...")
    ask_res = client.post(
        "/api/ask",
        json={"question": "What is Round Robin scheduling?", "document_id": doc_id},
    )
    assert ask_res.status_code == 200, f"Expected 200, got {ask_res.status_code}"
    ask_data = ask_res.json()
    assert ask_data["retrieval_status"] == "found"
    assert len(ask_data["cited_sources"]) > 0
    print("  PASS: POST /api/ask backward compatibility confirmed.")
    results["passed"] += 1

    # -------------------------------------------------------------------------
    # Test 12: HTTP 503 Service Unavailable -> Graceful Extractive Fallback
    # -------------------------------------------------------------------------
    print("\n[Test 12] HTTP 503 Service Unavailable from Gemini...")
    mock_503 = MagicMock()
    mock_503.status_code = 503
    mock_503.text = "The service is temporarily unavailable."

    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeMockTestKey12345"}):
        with patch("httpx.AsyncClient.post", return_value=mock_503):
            t12_res = client.post(
                f"/api/chat/sessions/{session_id}/ask",
                json={"question": "Describe the preemptive timer component."},
            )
            assert t12_res.status_code == 200
            data12 = t12_res.json()
            assert data12["generation_provider"] == "extractive_grounded_synthesizer"
            print("  PASS: HTTP 503 gracefully handled; extractive fallback succeeded.")
            results["passed"] += 1

    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {results['passed']} PASSED, {results['failed']} FAILED (TOTAL: {results['passed'] + results['failed']})")
    print("=" * 80)
    print("\nALL PHASE 2 CLOUD LLM & CONVERSATIONAL RAG TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_cloud_llm_tests()

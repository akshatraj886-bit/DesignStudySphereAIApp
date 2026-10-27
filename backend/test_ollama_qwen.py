"""
StudySphere AI - Ollama/Qwen Integration Test Suite
====================================================
Validates:
1. Provider-selection logic: Gemini → Ollama → Extractive fallback.
2. Qwen/Ollama invocation via mocked /api/chat endpoint.
3. Grounded prompt format: system + history + user_context_prompt.
4. Provider attribution: generation_provider = "local_ollama", model_name present.
5. Failure handling: Ollama unreachable, model missing (404), timeout, refusal.
6. Citation validation on Ollama-generated responses.
7. Streaming Ollama with CitationStreamFilter.
8. Configuration properties: GENERATION_PROVIDER, OLLAMA_ENABLED, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS.
9. GENERATION_PROVIDER=auto|gemini|ollama dispatch, including ollama-with-Gemini-key and disabled-Ollama safety gate.

ALL tests use mocked HTTP responses — no live Ollama instance is required.
Each test is clearly labeled as MOCKED.
"""

import os
import sys
import json
import asyncio
import pymupdf
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from app.main import app
from app.rag.generator import generator_instance, AnswerGenerator


def create_sample_pdf() -> bytes:
    """Creates a minimal 2-page PDF for retrieval tests."""
    doc = pymupdf.open()

    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 72),
        "Operating Systems: CPU Scheduling Algorithms\n\n"
        "Round Robin scheduling is a preemptive CPU scheduling algorithm where each process "
        "is assigned a fixed time slice called a quantum. When the time quantum expires, "
        "the operating system performs a context switch to the next ready process."
    )

    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 72),
        "Memory Management: Paging Architecture\n\n"
        "Paging avoids external fragmentation by dividing physical memory into fixed-size blocks "
        "called frames and logical memory into blocks called pages. A page table maps logical "
        "addresses to physical frame addresses."
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_ollama_qwen_tests():
    print("=" * 80)
    print("STUDYSPHERE AI - OLLAMA/QWEN INTEGRATION TEST SUITE (ALL MOCKED)")
    print("=" * 80)

    client = TestClient(app)
    results = {"passed": 0, "failed": 0}

    # Setup: Upload document
    print("\n[Setup] Uploading 2-page reference PDF...")
    pdf_bytes = create_sample_pdf()
    files = {"file": ("OS_Notes.pdf", pdf_bytes, "application/pdf")}
    upload_res = client.post("/api/documents/upload", files=files)
    assert upload_res.status_code == 200, f"Setup upload failed: {upload_res.text}"
    doc_id = upload_res.json()["document_id"]
    print(f"  Document indexed: {doc_id}")

    # Setup: Create chat session
    ses_res = client.post(
        "/api/chat/sessions",
        json={"document_id": doc_id, "title": "Qwen Test Session"},
    )
    assert ses_res.status_code == 201
    session_id = ses_res.json()["id"]
    print(f"  Chat session: {session_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Configuration Properties (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 1] MOCKED: Ollama configuration properties...")
    try:
        gen = AnswerGenerator()
        with patch.dict(os.environ, {
            "OLLAMA_ENABLED": "true",
            "OLLAMA_HOST": "http://my-ollama:11434",
            "OLLAMA_MODEL": "qwen2.5:1.5b",
            "OLLAMA_TIMEOUT_SECONDS": "45.0",
            "GENERATION_PROVIDER": "ollama",
        }):
            assert gen.ollama_enabled is True
            assert gen.ollama_host == "http://my-ollama:11434"
            assert gen.ollama_model == "qwen2.5:1.5b"
            assert gen.ollama_timeout == 45.0
            assert gen.generation_provider_mode == "ollama"

        with patch.dict(os.environ, {
            "OLLAMA_ENABLED": "false",
            "OLLAMA_MODEL": "",
            "OLLAMA_TIMEOUT_SECONDS": "invalid",
            "GENERATION_PROVIDER": "not-a-provider",
        }):
            assert gen.ollama_enabled is False
            assert gen.ollama_model == "qwen2.5:1.5b"  # default
            assert gen.ollama_timeout == 30.0  # fallback on invalid
            assert gen.generation_provider_mode == "auto"  # invalid values fall back to auto

        with patch.dict(os.environ, {"GENERATION_PROVIDER": ""}, clear=False):
            assert gen.generation_provider_mode == "auto"

        print("  PASS: All configuration properties work correctly.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Provider Selection - Gemini disabled, Ollama enabled (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 2] MOCKED: Ollama/Qwen invoked when Gemini key is absent...")

    mock_ollama_resp = MagicMock()
    mock_ollama_resp.status_code = 200
    mock_ollama_resp.json.return_value = {
        "model": "qwen2.5:1.5b",
        "message": {
            "role": "assistant",
            "content": (
                "### Round Robin Scheduling\n"
                "Round Robin is a preemptive scheduling algorithm that assigns each process "
                "a fixed time slice called a **quantum** [Page 1]. When the time quantum expires, "
                "a context switch occurs to the next process."
            ),
        },
        "done": True,
    }

    try:
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "true",
            "OLLAMA_MODEL": "qwen2.5:1.5b",
        }):
            with patch("httpx.AsyncClient.post", return_value=mock_ollama_resp):
                with patch("httpx.AsyncClient.get") as mock_get:
                    # Mock /api/tags to return qwen model
                    mock_tags = MagicMock()
                    mock_tags.status_code = 200
                    mock_tags.json.return_value = {"models": [{"name": "qwen2.5:1.5b"}]}
                    mock_get.return_value = mock_tags

                    t2_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask",
                        json={"question": "What is Round Robin scheduling?"},
                    )
                    assert t2_res.status_code == 200, f"Expected 200, got {t2_res.status_code}"
                    data2 = t2_res.json()
                    assert data2["generation_provider"] == "local_ollama", \
                        f"Expected 'local_ollama', got '{data2['generation_provider']}'"
                    assert data2["model_name"] == "qwen2.5:1.5b"
                    assert "Round Robin" in data2["answer"]
                    assert data2["retrieval_status"] == "found"
                    assert len(data2["cited_sources"]) > 0
                    print(f"  PASS: Ollama/Qwen selected when Gemini absent. Provider: {data2['generation_provider']}, Model: {data2['model_name']}")
                    results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Grounded Prompt Structure (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 3] MOCKED: Ollama receives grounded prompt with document passages...")
    try:
        gen = AnswerGenerator()
        candidates = [
            {
                "text": "Round Robin is preemptive with time quantum.",
                "page_number": 1,
                "document_name": "OS_Notes.pdf",
                "similarity_score": 0.85,
                "chunk_id": "c1",
            }
        ]
        messages = gen._format_ollama_messages(
            question="What is Round Robin?",
            valid_candidates=candidates,
            conversation_history=[
                {"role": "user", "content": "Tell me about scheduling."},
                {"role": "assistant", "content": "Scheduling assigns CPU time to processes."},
            ],
            scope="targeted_concept",
        )

        # Check structure
        assert messages[0]["role"] == "system"
        assert "document passages" in messages[0]["content"].lower() or "evidence" in messages[0]["content"].lower() or "grounded" in messages[0]["content"].lower()
        # Conversation history injected
        assert any(m["role"] == "user" and "scheduling" in m["content"].lower() for m in messages[1:-1])
        assert any(m["role"] == "assistant" and "assigns" in m["content"].lower() for m in messages[1:-1])
        # Last message is the current user question with context
        assert messages[-1]["role"] == "user"
        assert "Round Robin" in messages[-1]["content"]
        # Document passages included in the user message
        assert "Page 1" in messages[-1]["content"] or "time quantum" in messages[-1]["content"]

        print("  PASS: Grounded prompt correctly structured with system + history + user_context.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Ollama Unreachable - Graceful Fallback (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 4] MOCKED: Ollama unreachable falls back to extractive synthesizer...")
    try:
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "true",
        }):
            with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
                with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
                    t4_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask",
                        json={"question": "What is paging?"},
                    )
                    assert t4_res.status_code == 200
                    data4 = t4_res.json()
                    assert data4["generation_provider"] == "extractive_grounded_synthesizer"
                    print(f"  PASS: Graceful fallback to extractive synthesizer. Provider: {data4['generation_provider']}")
                    results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: Ollama Model Missing (404) - Graceful Fallback (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 5] MOCKED: Ollama model not found (404) falls back to extractive...")
    try:
        mock_404 = MagicMock()
        mock_404.status_code = 404
        mock_404.json.return_value = {"error": "model 'qwen2.5:1.5b' not found"}

        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "true",
            "OLLAMA_MODEL": "qwen2.5:1.5b",
        }):
            with patch("httpx.AsyncClient.post", return_value=mock_404):
                with patch("httpx.AsyncClient.get") as mock_get:
                    mock_tags = MagicMock()
                    mock_tags.status_code = 200
                    mock_tags.json.return_value = {"models": []}
                    mock_get.return_value = mock_tags

                    t5_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask",
                        json={"question": "What is Round Robin scheduling?"},
                    )
                    assert t5_res.status_code == 200
                    data5 = t5_res.json()
                    assert data5["generation_provider"] == "extractive_grounded_synthesizer"
                    print(f"  PASS: Model 404 handled gracefully. Provider: {data5['generation_provider']}")
                    results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Ollama Timeout - Graceful Fallback (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 6] MOCKED: Ollama timeout falls back to extractive synthesizer...")
    try:
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "true",
            "OLLAMA_TIMEOUT_SECONDS": "1.0",
        }):
            with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timeout")):
                with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Connect timeout")):
                    t6_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask",
                        json={"question": "What is paging?"},
                    )
                    assert t6_res.status_code == 200
                    data6 = t6_res.json()
                    assert data6["generation_provider"] == "extractive_grounded_synthesizer"
                    print(f"  PASS: Timeout handled gracefully. Provider: {data6['generation_provider']}")
                    results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 7: Ollama Refusal - Falls through to extractive (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 7] MOCKED: Ollama refusal response falls back to extractive...")
    try:
        mock_refusal = MagicMock()
        mock_refusal.status_code = 200
        mock_refusal.json.return_value = {
            "model": "qwen2.5:1.5b",
            "message": {
                "role": "assistant",
                "content": (
                    "I could not find sufficient evidence in the uploaded document(s) "
                    "to answer this question. The retrieved passages do not discuss this topic."
                ),
            },
            "done": True,
        }

        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "true",
        }):
            with patch("httpx.AsyncClient.post", return_value=mock_refusal):
                with patch("httpx.AsyncClient.get") as mock_get:
                    mock_tags = MagicMock()
                    mock_tags.status_code = 200
                    mock_tags.json.return_value = {"models": [{"name": "qwen2.5:1.5b"}]}
                    mock_get.return_value = mock_tags

                    t7_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask",
                        json={"question": "What is Round Robin scheduling?"},
                    )
                    assert t7_res.status_code == 200
                    data7 = t7_res.json()
                    assert data7["generation_provider"] == "extractive_grounded_synthesizer", \
                        f"Expected extractive after refusal, got '{data7['generation_provider']}'"
                    print(f"  PASS: Refusal fell through to extractive. Provider: {data7['generation_provider']}")
                    results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 8: Provider Attribution Accuracy (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 8] MOCKED: Provider attribution is accurate across all providers...")
    try:
        # 8a: Gemini with key
        mock_gemini = MagicMock()
        mock_gemini.status_code = 200
        mock_gemini.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Round Robin uses time quantum [Page 1]."}]}}]
        }
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeKey", "OLLAMA_ENABLED": "true"}):
            with patch("httpx.AsyncClient.post", return_value=mock_gemini):
                t8a_res = client.post(
                    f"/api/chat/sessions/{session_id}/ask",
                    json={"question": "What is Round Robin scheduling?"},
                )
                assert t8a_res.status_code == 200
                assert t8a_res.json()["generation_provider"] == "google_gemini"

        # 8b: No Gemini, Ollama enabled and working
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "OLLAMA_ENABLED": "true"}):
            with patch("httpx.AsyncClient.post", return_value=mock_ollama_resp):
                with patch("httpx.AsyncClient.get") as mock_get:
                    mock_tags = MagicMock()
                    mock_tags.status_code = 200
                    mock_tags.json.return_value = {"models": [{"name": "qwen2.5:1.5b"}]}
                    mock_get.return_value = mock_tags
                    t8b_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask",
                        json={"question": "What is Round Robin scheduling?"},
                    )
                    assert t8b_res.status_code == 200
                    assert t8b_res.json()["generation_provider"] == "local_ollama"

        # 8c: No Gemini, Ollama disabled
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "OLLAMA_ENABLED": "false"}):
            t8c_res = client.post(
                f"/api/chat/sessions/{session_id}/ask",
                json={"question": "What is Round Robin scheduling?"},
            )
            assert t8c_res.status_code == 200
            assert t8c_res.json()["generation_provider"] == "extractive_grounded_synthesizer"

        print("  PASS: Provider attribution accurate for Gemini, Ollama, and extractive.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 9: Citation Validation on Ollama Response (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 9] MOCKED: Hallucinated page citations are stripped from Ollama response...")
    try:
        # Response claims Page 5 which doesn't exist (only Pages 1-2)
        mock_halluc = MagicMock()
        mock_halluc.status_code = 200
        mock_halluc.json.return_value = {
            "model": "qwen2.5:1.5b",
            "message": {
                "role": "assistant",
                "content": (
                    "Round Robin scheduling uses a quantum [Page 1]. "
                    "The algorithm was invented in 1965 [Page 5]. "
                    "Context switches happen automatically [Page 1]."
                ),
            },
            "done": True,
        }

        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "true",
        }):
            with patch("httpx.AsyncClient.post", return_value=mock_halluc):
                with patch("httpx.AsyncClient.get") as mock_get:
                    mock_tags = MagicMock()
                    mock_tags.status_code = 200
                    mock_tags.json.return_value = {"models": [{"name": "qwen2.5:1.5b"}]}
                    mock_get.return_value = mock_tags

                    t9_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask",
                        json={"question": "What is Round Robin scheduling?"},
                    )
                    assert t9_res.status_code == 200
                    data9 = t9_res.json()
                    assert data9["generation_provider"] == "local_ollama"
                    # Page 5 should be stripped since the document only has 2 pages
                    assert "[Page 5]" not in data9["answer"]
                    # Page 1 should remain
                    assert "[Page 1]" in data9["answer"] or "Page 1" in data9["answer"]
                    print(f"  PASS: Hallucinated [Page 5] stripped. Valid [Page 1] preserved.")
                    results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 10: Ollama Disabled - Skipped (No Network Call) (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 10] MOCKED: Ollama disabled means no Ollama call is made...")
    try:
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "false",
        }):
            t10_res = client.post(
                f"/api/chat/sessions/{session_id}/ask",
                json={"question": "What is paging?"},
            )
            assert t10_res.status_code == 200
            data10 = t10_res.json()
            # Should go directly to extractive since both Gemini and Ollama are off
            assert data10["generation_provider"] == "extractive_grounded_synthesizer"
            print(f"  PASS: Ollama disabled, extractive used directly. Provider: {data10['generation_provider']}")
            results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    def _mock_tags(model_name="qwen2.5:1.5b"):
        mock_tags = MagicMock()
        mock_tags.status_code = 200
        mock_tags.json.return_value = {"models": [{"name": model_name}] if model_name else []}
        return mock_tags

    # ─────────────────────────────────────────────────────────────────────────
    # Test 11: GENERATION_PROVIDER=auto preserves Gemini-first behavior (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 11] MOCKED: GENERATION_PROVIDER=auto keeps Gemini-first dispatch...")
    try:
        mock_gemini = MagicMock()
        mock_gemini.status_code = 200
        mock_gemini.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Round Robin uses a time quantum [Page 1]."}]}}]
        }
        with patch.dict(os.environ, {
            "GENERATION_PROVIDER": "auto",
            "GEMINI_API_KEY": "AIzaFakeKey",
            "OLLAMA_ENABLED": "true",
            "OLLAMA_MODEL": "qwen2.5:1.5b",
        }):
            with patch("httpx.AsyncClient.post", return_value=mock_gemini):
                t11_res = client.post(
                    f"/api/chat/sessions/{session_id}/ask",
                    json={"question": "What is Round Robin scheduling?"},
                )
                assert t11_res.status_code == 200
                assert t11_res.json()["generation_provider"] == "google_gemini"
        print("  PASS: auto mode selected google_gemini before Ollama.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 12: GENERATION_PROVIDER=gemini selects Gemini when configured (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 12] MOCKED: GENERATION_PROVIDER=gemini selects Gemini when a key is set...")
    try:
        mock_gemini = MagicMock()
        mock_gemini.status_code = 200
        mock_gemini.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Paging maps pages to frames [Page 2]."}]}}]
        }
        with patch.dict(os.environ, {
            "GENERATION_PROVIDER": "gemini",
            "GEMINI_API_KEY": "AIzaFakeKey",
            "OLLAMA_ENABLED": "true",
        }):
            with patch("httpx.AsyncClient.post", return_value=mock_gemini):
                t12_res = client.post(
                    f"/api/chat/sessions/{session_id}/ask",
                    json={"question": "What is paging?"},
                )
                assert t12_res.status_code == 200
                assert t12_res.json()["generation_provider"] == "google_gemini"
        print("  PASS: gemini mode selected google_gemini.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 13: GENERATION_PROVIDER=ollama prefers Qwen even with a Gemini key (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 13] MOCKED: ollama mode uses Qwen even when GEMINI_API_KEY is set...")
    try:
        gemini_called = {"value": False}

        def post_side_effect(*args, **kwargs):
            url = ""
            if len(args) >= 2:
                url = str(args[1])
            elif len(args) == 1:
                url = str(args[0])
            url = str(kwargs.get("url", url))
            if "generativelanguage.googleapis.com" in url:
                gemini_called["value"] = True
                raise AssertionError("Gemini must not be invoked when GENERATION_PROVIDER=ollama")
            return mock_ollama_resp

        with patch.dict(os.environ, {
            "GENERATION_PROVIDER": "ollama",
            "GEMINI_API_KEY": "AIzaFakeKeyStillPresent",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "true",
            "OLLAMA_MODEL": "qwen2.5:1.5b",
        }):
            with patch("httpx.AsyncClient.post", side_effect=post_side_effect):
                with patch("httpx.AsyncClient.get", return_value=_mock_tags()):
                    t13_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask",
                        json={"question": "What is Round Robin scheduling?"},
                    )
                    assert t13_res.status_code == 200
                    data13 = t13_res.json()
                    assert data13["generation_provider"] == "local_ollama"
                    assert data13["model_name"] == "qwen2.5:1.5b"
                    assert gemini_called["value"] is False
        print("  PASS: ollama mode selected local_ollama and skipped Gemini.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 14: GENERATION_PROVIDER=ollama with OLLAMA_ENABLED=false (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 14] MOCKED: ollama mode with OLLAMA_ENABLED=false uses extractive, not Gemini...")
    try:
        with patch.dict(os.environ, {
            "GENERATION_PROVIDER": "ollama",
            "GEMINI_API_KEY": "AIzaFakeKeyStillPresent",
            "OLLAMA_ENABLED": "false",
        }):
            with patch("httpx.AsyncClient.post") as mock_post:
                t14_res = client.post(
                    f"/api/chat/sessions/{session_id}/ask",
                    json={"question": "What is paging?"},
                )
                assert t14_res.status_code == 200
                assert t14_res.json()["generation_provider"] == "extractive_grounded_synthesizer"
                mock_post.assert_not_called()
        print("  PASS: disabled Ollama safety gate skipped Gemini and used extractive fallback.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 15: Missing model in /api/tags does not call /api/chat (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 15] MOCKED: missing Ollama model in /api/tags skips chat and falls back...")
    try:
        with patch.dict(os.environ, {
            "GENERATION_PROVIDER": "ollama",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "true",
            "OLLAMA_MODEL": "qwen2.5:1.5b",
        }):
            with patch("httpx.AsyncClient.post") as mock_post:
                with patch("httpx.AsyncClient.get", return_value=_mock_tags("")):
                    t15_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask",
                        json={"question": "What is Round Robin scheduling?"},
                    )
                    assert t15_res.status_code == 200
                    assert t15_res.json()["generation_provider"] == "extractive_grounded_synthesizer"
                    mock_post.assert_not_called()
        print("  PASS: preflight missing-model diagnostic skipped /api/chat and used extractive.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 16: SSE token + done events remain compatible for Ollama (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 16] MOCKED: Ollama streaming emits token events and a compatible done payload...")
    try:
        class MockOllamaStreamContext:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                class MockResp:
                    status_code = 200

                    async def aiter_lines(self):
                        yield json.dumps({
                            "message": {"content": "Paging divides memory into **pages** "},
                            "done": False,
                        })
                        yield json.dumps({
                            "message": {"content": "and frames [Page 2]."},
                            "done": False,
                        })
                        yield json.dumps({"message": {"content": ""}, "done": True})

                return MockResp()

            async def __aexit__(self, *args):
                pass

        with patch.dict(os.environ, {
            "GENERATION_PROVIDER": "ollama",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "true",
            "OLLAMA_MODEL": "qwen2.5:1.5b",
        }):
            with patch.object(httpx.AsyncClient, "stream", side_effect=MockOllamaStreamContext):
                with patch("httpx.AsyncClient.get", return_value=_mock_tags()):
                    stream_res = client.post(
                        f"/api/chat/sessions/{session_id}/ask/stream",
                        json={"question": "What is paging?"},
                    )
                    assert stream_res.status_code == 200
                    events = []
                    done_payload = None
                    for line in stream_res.iter_lines():
                        if line.startswith("data: "):
                            ev = json.loads(line[6:])
                            events.append(ev)
                            if ev.get("done"):
                                done_payload = ev
                    token_events = [e for e in events if "token" in e]
                    assert len(token_events) >= 1, f"Expected token events, got {events}"
                    assert done_payload is not None and done_payload.get("done") is True
                    assert done_payload.get("generation_provider") == "local_ollama"
                    assert done_payload.get("model_name") == "qwen2.5:1.5b"
                    assert done_payload.get("retrieval_status") == "found"
                    assert "cited_sources" in done_payload
                    assert "[Page 5]" not in (done_payload.get("answer") or "")
        print("  PASS: SSE token/done contract preserved for local_ollama streaming.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Test 17: Invalid GENERATION_PROVIDER falls back to auto (MOCKED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Test 17] MOCKED: invalid GENERATION_PROVIDER is treated as auto (Gemini-first)...")
    try:
        mock_gemini = MagicMock()
        mock_gemini.status_code = 200
        mock_gemini.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Round Robin uses a quantum [Page 1]."}]}}]
        }
        with patch.dict(os.environ, {
            "GENERATION_PROVIDER": "banana",
            "GEMINI_API_KEY": "AIzaFakeKey",
            "OLLAMA_ENABLED": "true",
        }):
            with patch("httpx.AsyncClient.post", return_value=mock_gemini):
                t17_res = client.post(
                    f"/api/chat/sessions/{session_id}/ask",
                    json={"question": "What is Round Robin scheduling?"},
                )
                assert t17_res.status_code == 200
                assert t17_res.json()["generation_provider"] == "google_gemini"
        print("  PASS: invalid provider value fell back to auto/Gemini-first.")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  FAIL: {e}")
        results["failed"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    total = results["passed"] + results["failed"]
    print("\n" + "=" * 80)
    print(f"OLLAMA/QWEN TEST RESULTS: {results['passed']} PASSED, {results['failed']} FAILED (TOTAL: {total})")
    print("=" * 80)
    if results["failed"] == 0:
        print("ALL OLLAMA/QWEN INTEGRATION TESTS PASSED (ALL MOCKED).\n")
    else:
        print(f"WARNING: {results['failed']} test(s) failed.\n")
    return results["failed"] == 0


if __name__ == "__main__":
    success = run_ollama_qwen_tests()
    sys.exit(0 if success else 1)

"""
StudySphere AI - Phase 4 Controlled Verification Suite
Validates Task 3 requirements:
1. Operates on a completely isolated temporary SQLite database (zero access or mutation to production database).
2. Uses synthetic multi-page educational test document.
3. Tests questions present in document (grounded answer, Page 1 citation preserved).
4. Tests questions absent / out-of-scope (clean refusal, zero false positive citations).
5. Verifies safe citation validation (hallucinated page numbers cleanly stripped, never invented).
6. Records and verifies exact generation_provider and model_name across:
   - Google Gemini (mocked for deterministic verification)
   - Extractive Grounded Synthesizer (offline fallback)
   - Grounding Gate Filter (rejection of out-of-scope questions)
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from app.rag.chunker import DocumentChunk
from app.rag.store import VectorStore
from app.rag.embeddings import embedding_service
from app.rag.generator import AnswerGenerator, validate_answer_citations, extract_cited_sources, verify_topical_evidence


class TestPhase4ControlledVerification(unittest.IsolatedAsyncioTestCase):
    """Controlled, isolated test suite that never touches production database."""

    async def asyncSetUp(self):
        # 1. Create isolated temporary SQLite database
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name

        # Ensure DATABASE_URL is not set so VectorStore operates strictly on isolated SQLite
        self.env_patcher = patch.dict(os.environ, {"DATABASE_URL": ""}, clear=False)
        self.env_patcher.start()

        self.store = VectorStore(db_path=self.db_path)
        self.generator = AnswerGenerator()

        # 2. Index synthetic 2-page educational document
        self.doc_id = "doc_test_synthetic_p4"
        self.filename = "Systems_Concurrency_Notes.pdf"
        self.session_token = "test_session_p4_isolated"

        # Page 1: Cache Replacement
        p1_text = (
            "Cache Replacement Policies in Operating Systems and Architecture. "
            "When a cache miss occurs and the cache is full, a line must be evicted. "
            "Least Recently Used (LRU) discards the item that has not been accessed for the longest time. "
            "First In First Out (FIFO) replaces the oldest loaded block regardless of access frequency. "
            "Least Frequently Used (LFU) evicts blocks with the smallest access counter."
        )

        # Page 2: Database Concurrency Control
        p2_text = (
            "Database Concurrency Control and Serializability. "
            "Two-Phase Locking (2PL) guarantees conflict serializability. "
            "In the growing phase, a transaction may obtain locks but release none. "
            "In the shrinking phase, a transaction may release locks but acquire no new ones. "
            "Strict 2PL holds all exclusive locks until the transaction commits or aborts."
        )

        chunks = [
            DocumentChunk(
                chunk_id=f"{self.doc_id}_p1_c0",
                document_id=self.doc_id,
                document_name=self.filename,
                page_number=1,
                chunk_index=0,
                text=p1_text,
                character_count=len(p1_text),
                word_count=len(p1_text.split()),
            ),
            DocumentChunk(
                chunk_id=f"{self.doc_id}_p2_c0",
                document_id=self.doc_id,
                document_name=self.filename,
                page_number=2,
                chunk_index=1,
                text=p2_text,
                character_count=len(p2_text),
                word_count=len(p2_text.split()),
            ),
        ]

        # Generate genuine 384-dim embeddings locally using all-MiniLM-L6-v2
        chunk_embeddings = embedding_service.embed_texts([c.text for c in chunks])

        self.store.save_document(
            document_id=self.doc_id,
            filename=self.filename,
            page_count=2,
            total_characters=sum(c.character_count for c in chunks),
            total_words=sum(c.word_count for c in chunks),
            chunks=chunks,
            embeddings=chunk_embeddings,
            session_token=self.session_token,
        )

    async def asyncTearDown(self):
        self.env_patcher.stop()
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
        except Exception:
            pass

    async def test_1_isolated_db_has_zero_foreign_records(self):
        """Verify the test runs on an isolated DB with exactly 1 document and 2 chunks."""
        docs = self.store.list_documents(session_token=self.session_token)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["id"], self.doc_id)

        # Verify default_session or other sessions see 0 records in this DB
        other_docs = self.store.list_documents(session_token="foreign_session")
        self.assertEqual(len(other_docs), 0)

    async def test_2_local_embedding_and_retrieval_verification(self):
        """Verify query is embedded locally and retrieves correct page with cosine similarity."""
        query = "How does Least Recently Used LRU cache eviction work?"
        q_emb = embedding_service.embed_query(query)
        self.assertEqual(q_emb.shape, (384,))
        self.assertEqual(q_emb.dtype, np.float32)

        # Retrieve from isolated store
        results = self.store.search_chunks(
            query_embedding=q_emb,
            document_id=self.doc_id,
            session_token=self.session_token,
            top_k=2,
        )
        self.assertGreaterEqual(len(results), 1)
        top_match = results[0]
        self.assertEqual(top_match["page_number"], 1)
        self.assertIn("Least Recently Used", top_match["text"])
        self.assertGreater(top_match["similarity_score"], 0.45)

    async def test_3_grounded_generation_mock_gemini(self):
        """Verify that when Gemini is called, answer is grounded and provider is 'google_gemini'."""
        query = "Explain the difference between LRU and FIFO cache replacement."
        q_emb = embedding_service.embed_query(query)
        sources = self.store.search_chunks(
            query_embedding=q_emb,
            document_id=self.doc_id,
            session_token=self.session_token,
            top_k=2,
        )

        # Simulate Gemini returning grounded answer
        mock_gemini_resp = MagicMock()
        mock_gemini_resp.status_code = 200
        mock_gemini_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "LRU discards the item that has not been accessed for the longest time [Page 1]. "
                                    "In contrast, FIFO replaces the oldest loaded block regardless of access frequency [Page 1]."
                                )
                            }
                        ]
                    }
                }
            ]
        }

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForControlledMock123"}):
            with patch("httpx.AsyncClient.post", return_value=mock_gemini_resp):
                ans = await self.generator.generate_answer(
                    question=query,
                    sources=sources,
                )

        self.assertEqual(ans["retrieval_status"], "found")
        self.assertEqual(ans["generation_provider"], "google_gemini")
        self.assertIn("gemini", ans["model_name"])
        self.assertIn("[Page 1]", ans["answer"])
        self.assertEqual(len(ans["cited_sources"]), 1)
        self.assertEqual(ans["cited_sources"][0]["page_number"], 1)

    async def test_4_citation_validation_strips_hallucinated_pages(self):
        """Verify that hallucinated page numbers injected by model are stripped and never invented."""
        query = "Explain 2PL concurrency control."
        q_emb = embedding_service.embed_query(query)
        sources = self.store.search_chunks(
            query_embedding=q_emb,
            document_id=self.doc_id,
            session_token=self.session_token,
            top_k=2,
        )
        # Note: Document only has Page 1 and Page 2. Mock model hallucinates [Page 9].
        mock_gemini_resp = MagicMock()
        mock_gemini_resp.status_code = 200
        mock_gemini_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "Two-Phase Locking guarantees conflict serializability [Page 2]. "
                                    "Distributed timestamps are also discussed [Page 9]."
                                )
                            }
                        ]
                    }
                }
            ]
        }

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForControlledMock123"}):
            with patch("httpx.AsyncClient.post", return_value=mock_gemini_resp):
                ans = await self.generator.generate_answer(
                    question=query,
                    sources=sources,
                )

        self.assertEqual(ans["retrieval_status"], "found")
        self.assertIn("[Page 2]", ans["answer"])
        self.assertNotIn("[Page 9]", ans["answer"])  # Hallucinated page must be stripped
        # Assert cited sources does NOT include hallucinated page 9
        cited_pages = [cs["page_number"] for cs in ans["cited_sources"]]
        self.assertIn(2, cited_pages)
        self.assertNotIn(9, cited_pages)

    async def test_5_out_of_scope_question_rejected_by_gate(self):
        """Verify that out-of-scope question is rejected by rule_based_grounding_filter."""
        unrelated_query = "What were the primary economic causes of the French Revolution in 1789?"
        q_emb = embedding_service.embed_query(unrelated_query)
        sources = self.store.search_chunks(
            query_embedding=q_emb,
            document_id=self.doc_id,
            session_token=self.session_token,
            top_k=2,
        )

        ans = await self.generator.generate_answer(
            question=unrelated_query,
            sources=sources,
        )

        self.assertEqual(ans["retrieval_status"], "insufficient_evidence")
        self.assertEqual(ans["generation_provider"], "rule_based_grounding_filter")
        self.assertIsNone(ans["model_name"])
        self.assertEqual(len(ans["cited_sources"]), 0)
        self.assertIn("could not find sufficient evidence", ans["answer"].lower())

    async def test_6_extractive_grounded_fallback_when_gemini_fails(self):
        """Verify graceful fallback to extractive_grounded_synthesizer when cloud API fails."""
        query = "What is Strict 2PL and when are locks released?"
        q_emb = embedding_service.embed_query(query)
        sources = self.store.search_chunks(
            query_embedding=q_emb,
            document_id=self.doc_id,
            session_token=self.session_token,
            top_k=2,
        )

        # Mock 429 quota exhaustion from Gemini
        mock_429_resp = MagicMock()
        mock_429_resp.status_code = 429

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFakeTestKeyForControlledMock123", "OLLAMA_ENABLED": "false"}):
            with patch("httpx.AsyncClient.post", return_value=mock_429_resp):
                ans = await self.generator.generate_answer(
                    question=query,
                    sources=sources,
                )

        self.assertEqual(ans["retrieval_status"], "found")
        self.assertEqual(ans["generation_provider"], "extractive_grounded_synthesizer")
        self.assertEqual(ans["model_name"], "all-MiniLM-L6-v2-dense-retrieval")
        self.assertIn("[Page 2]", ans["answer"])
        self.assertEqual(len(ans["cited_sources"]), 1)
        self.assertEqual(ans["cited_sources"][0]["page_number"], 2)


if __name__ == "__main__":
    unittest.main()

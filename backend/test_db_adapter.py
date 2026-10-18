"""
StudySphere AI - Database Adapter and Dual-Engine Test Suite
Validates the storage abstraction layer:
- SQLite connection management & context manager rollback/commit
- Query translation (? to %s)
- AdaptedRow dict compatibility & memoryview handling
- Temporary database lifecycle (clean init, write, close, reopen, verify)
- Foreign key cascading deletes
- Multi-tenant session isolation
- Engine detection from environment
- Postgres DDL validation
- NEVER touches the production database.
"""

import os
import sys
import tempfile
import numpy as np

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.rag.db_adapter import (
    get_db_connection,
    get_engine_type,
    is_postgres,
    is_sqlite,
    adapt_query_for_engine,
    AdaptedRow,
    AdaptedConnection,
    POSTGRES_SCHEMA_STATEMENTS,
    ENGINE_SQLITE,
    ENGINE_POSTGRES,
)
from app.rag.store import VectorStore
from app.rag.chunker import DocumentChunk


def test_db_adapter_suite():
    print("=" * 70)
    print("STUDYSPHERE AI - DATABASE ADAPTER & STORAGE SUITE")
    print("=" * 70)

    passed = 0
    failed = 0

    def assert_test(condition: bool, name: str):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    # -------------------------------------------------------------------------
    # TEST 1: Engine Detection
    # -------------------------------------------------------------------------
    print("\n[1] Testing Engine Detection...")
    assert_test(is_sqlite() is True, "Default engine is SQLite")
    assert_test(is_postgres() is False, "Default engine is not Postgres")

    # Verify DATABASE_URL engages PostgreSQL engine
    os.environ["DATABASE_URL"] = "postgresql://test_user:test_pass@ep-xyz.aws.neon.tech/neondb?sslmode=require"
    assert_test(is_postgres() is True, "DATABASE_URL dynamically engages PostgreSQL engine")
    assert_test(is_sqlite() is False, "is_sqlite returns False when DATABASE_URL is set")
    del os.environ["DATABASE_URL"]
    assert_test(is_sqlite() is True, "Unsetting DATABASE_URL reverts safely to SQLite")

    # -------------------------------------------------------------------------
    # TEST 2: Query Translation
    # -------------------------------------------------------------------------
    print("\n[2] Testing Query Parameter Translation...")
    q1 = "SELECT * FROM documents WHERE id = ? AND session_token = ?"
    adapted_sqlite = adapt_query_for_engine(q1, ENGINE_SQLITE)
    adapted_pg = adapt_query_for_engine(q1, ENGINE_POSTGRES)
    assert_test(adapted_sqlite == q1, "SQLite query leaves '?' placeholders intact")
    assert_test(adapted_pg == "SELECT * FROM documents WHERE id = %s AND session_token = %s", "PostgreSQL translates '?' to '%s'")

    # -------------------------------------------------------------------------
    # TEST 3: AdaptedRow Behavior
    # -------------------------------------------------------------------------
    print("\n[3] Testing AdaptedRow Dictionary Compatibility...")
    test_dict = {"id": "doc_123", "filename": "test.pdf", "embedding": memoryview(b"\x00\x01\x02\x03")}
    row = AdaptedRow(test_dict)
    assert_test(row["id"] == "doc_123", "Key-based dictionary lookup")
    assert_test(row.get("filename") == "test.pdf", "Safe .get() lookup")
    assert_test(row.get("missing", "default") == "default", "Default parameter on .get()")
    assert_test(isinstance(row["embedding"], bytes), "Converts memoryview to bytes")
    assert_test(row[0] == "doc_123", "Integer index access")
    converted_dict = dict(row)
    assert_test(isinstance(converted_dict, dict) and converted_dict["id"] == "doc_123", "dict(row) produces valid standard Python dict")

    # -------------------------------------------------------------------------
    # TEST 4: Clean Temporary Database Creation & Reopening
    # -------------------------------------------------------------------------
    print("\n[4] Testing Temporary Database Lifecycle & Persistence Across Reopening...")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db_path = os.path.join(tmpdir, "test_lifecycle.db")
        store = VectorStore(db_path=temp_db_path)

        # Create dummy chunk & embedding
        chunk = DocumentChunk(
            chunk_id="chunk_test_1",
            document_id="doc_test_1",
            document_name="test.pdf",
            page_number=1,
            chunk_index=0,
            text="Antigravity database abstraction test content.",
            character_count=45,
            word_count=5,
        )
        emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        store.save_document(
            document_id="doc_test_1",
            filename="test.pdf",
            page_count=1,
            total_characters=45,
            total_words=5,
            chunks=[chunk],
            embeddings=[emb],
            session_token="session_test_a",
        )

        doc = store.get_document("doc_test_1", session_token="session_test_a")
        assert_test(doc is not None and doc["id"] == "doc_test_1", "Document saved and retrieved in temp database")

        chunks = store.get_document_chunks("doc_test_1", session_token="session_test_a")
        assert_test(len(chunks) == 1 and chunks[0]["id"] == "chunk_test_1", "Chunks saved and retrieved in temp database")

        # Now simulate application restart by creating a brand-new VectorStore instance pointing to the same file
        reopened_store = VectorStore(db_path=temp_db_path)
        doc_reopened = reopened_store.get_document("doc_test_1", session_token="session_test_a")
        assert_test(doc_reopened is not None and doc_reopened["id"] == "doc_test_1", "Data persists across database closure and reopen")

        # Test vector search in reopened database
        results = reopened_store.search_chunks(
            query_embedding=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            top_k=1,
            session_token="session_test_a",
        )
        assert_test(len(results) == 1 and results[0]["chunk_id"] == "chunk_test_1", "Vector search functions correctly on reopened database")

    # -------------------------------------------------------------------------
    # TEST 5: Foreign Key Cascading Deletions
    # -------------------------------------------------------------------------
    print("\n[5] Testing Foreign Key Cascading Deletions...")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db_path = os.path.join(tmpdir, "test_fk.db")
        store = VectorStore(db_path=temp_db_path)

        chunk = DocumentChunk(
            chunk_id="chunk_fk_1",
            document_id="doc_fk_1",
            document_name="fk.pdf",
            page_number=1,
            chunk_index=0,
            text="Cascade test.",
            character_count=13,
            word_count=2,
        )
        emb = np.array([0.5, 0.5], dtype=np.float32)

        store.save_document(
            document_id="doc_fk_1",
            filename="fk.pdf",
            page_count=1,
            total_characters=13,
            total_words=2,
            chunks=[chunk],
            embeddings=[emb],
            session_token="session_fk",
        )

        # Create attached chat session and quiz session
        store.create_chat_session("chat_fk_1", "doc_fk_1", title="FK Chat", session_token="session_fk")
        store.save_chat_message("msg_fk_1", "chat_fk_1", "user", "Hello FK", session_token="session_fk")
        store.create_quiz_session("quiz_fk_1", "doc_fk_1", "fk.pdf", "easy", [{"q": 1}], session_token="session_fk")

        # Delete parent document
        deleted = store.delete_document("doc_fk_1", session_token="session_fk")
        assert_test(deleted is True, "Parent document deleted")

        # Verify child records were cascaded
        chunks_after = store.get_document_chunks("doc_fk_1", session_token="session_fk")
        assert_test(len(chunks_after) == 0, "Document chunks cascaded on document deletion")

        chat_after = store.get_chat_session("chat_fk_1", session_token="session_fk")
        assert_test(chat_after is None, "Chat session cascaded on document deletion")

        quiz_after = store.get_quiz_session("quiz_fk_1", session_token="session_fk")
        assert_test(quiz_after is None, "Quiz session cascaded on document deletion")

    # -------------------------------------------------------------------------
    # TEST 6: Multi-Tenant Session Isolation
    # -------------------------------------------------------------------------
    print("\n[6] Testing Session Isolation on Temporary Database...")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db_path = os.path.join(tmpdir, "test_iso.db")
        store = VectorStore(db_path=temp_db_path)

        chunk_a = DocumentChunk(
            chunk_id="chk_a",
            document_id="doc_a",
            document_name="a.pdf",
            page_number=1,
            chunk_index=0,
            text="Tenant A private notes.",
            character_count=23,
            word_count=4,
        )
        store.save_document(
            document_id="doc_a",
            filename="a.pdf",
            page_count=1,
            total_characters=23,
            total_words=4,
            chunks=[chunk_a],
            embeddings=[np.array([1.0, 0.0], dtype=np.float32)],
            session_token="session_alpha",
        )

        # Tenant B queries Tenant A document
        doc_b_view = store.get_document("doc_a", session_token="session_beta")
        assert_test(doc_b_view is None, "Session Beta cannot get Session Alpha's document")

        chunks_b_view = store.get_document_chunks("doc_a", session_token="session_beta")
        assert_test(len(chunks_b_view) == 0, "Session Beta cannot get Session Alpha's chunks")

        # Session Beta search chunks
        search_b = store.search_chunks(
            query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            top_k=5,
            session_token="session_beta",
        )
        assert_test(len(search_b) == 0, "Session Beta vector search returns 0 results for Session Alpha chunks")

        # Session Beta cannot delete Session Alpha document
        del_attempt = store.delete_document("doc_a", session_token="session_beta")
        assert_test(del_attempt is False, "Session Beta cannot delete Session Alpha's document")

        # Document still exists for Session Alpha
        assert_test(store.get_document("doc_a", session_token="session_alpha") is not None, "Document remains intact after rejected delete attempt")

    # -------------------------------------------------------------------------
    # TEST 7: Connection Context Manager & Rollback
    # -------------------------------------------------------------------------
    print("\n[7] Testing Connection Context Manager Rollback on Exception...")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db_path = os.path.join(tmpdir, "test_rollback.db")
        conn = get_db_connection(temp_db_path)
        with conn as c:
            c.execute("CREATE TABLE test_table (id TEXT PRIMARY KEY, val TEXT);")

        # Attempt transaction that errors
        try:
            with get_db_connection(temp_db_path) as c:
                c.execute("INSERT INTO test_table VALUES (?, ?);", ("key_1", "val_1"))
                # Deliberate syntax/constraint error to trigger rollback
                raise RuntimeError("Simulated transaction failure")
        except RuntimeError:
            pass

        # Verify key_1 was rolled back
        with get_db_connection(temp_db_path) as c:
            cur = c.cursor()
            cur.execute("SELECT COUNT(*) as cnt FROM test_table WHERE id = ?", ("key_1",))
            row = cur.fetchone()
            assert_test(row["cnt"] == 0, "Transaction was cleanly rolled back on exception")

    # -------------------------------------------------------------------------
    # TEST 8: PostgreSQL Schema Statements Validation
    # -------------------------------------------------------------------------
    print("\n[8] Testing PostgreSQL Schema Statements...")
    assert_test(len(POSTGRES_SCHEMA_STATEMENTS) >= 10, "PostgreSQL DDL defines all tables and indexes")
    schema_text = " ".join(POSTGRES_SCHEMA_STATEMENTS)
    assert_test("BYTEA" in schema_text, "PostgreSQL schema uses BYTEA for embedding storage")
    assert_test("ON DELETE CASCADE" in schema_text, "PostgreSQL schema includes ON DELETE CASCADE")
    assert_test("session_token" in schema_text, "PostgreSQL schema includes session_token columns")

    print("\n" + "=" * 70)
    print(f"DATABASE ADAPTER SUITE: {passed} PASSED, {failed} FAILED (TOTAL: {passed + failed})")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = test_db_adapter_suite()
    sys.exit(0 if success else 1)

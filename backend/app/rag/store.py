"""
StudySphere AI - Document and Vector SQLite Store
Provides reliable, persistent storage for documents, chunk metadata,
and normalized embedding vectors using standard SQLite.
Enforces multi-tenant session isolation using unguessable session tokens.
"""

import os
import json
import sqlite3
import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from app.rag.chunker import DocumentChunk


from app.rag.db_adapter import (
    get_db_connection,
    is_postgres,
    init_postgres_schema,
    AdaptedConnection,
)

DEFAULT_DB_PATH = os.getenv(
    "DATABASE_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "studysphere.db",
    ),
)


class VectorStore:
    """
    Persistent vector store for StudySphere AI.
    Supports local SQLite (zero-setup, default) and durable cloud PostgreSQL (DATABASE_URL).
    Stores documents and chunks with raw float32 embedding bytes.
    Performs fast cosine similarity ranking in memory.
    Enforces tenant/session isolation via session_token.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if not is_postgres():
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> AdaptedConnection:
        return get_db_connection(self.db_path)

    def _migrate_schema(self, conn: AdaptedConnection):
        """Idempotently migrates existing tables to support session_token."""
        cursor = conn.cursor()
        for table, idx_name in [
            ("documents", "idx_documents_session_token"),
            ("document_chunks", "idx_chunks_session_token"),
            ("quiz_sessions", "idx_quiz_sessions_session_token"),
            ("chat_sessions", "idx_chat_sessions_session_token"),
        ]:
            if is_postgres():
                cursor.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = ? AND column_name = 'session_token'",
                    (table,),
                )
                cols = {row["column_name"] for row in cursor.fetchall()}
                if "session_token" not in cols:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN session_token TEXT DEFAULT 'default_session';")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}(session_token);")
            else:
                cursor.execute(f"PRAGMA table_info({table});")
                cols = {row["name"] for row in cursor.fetchall()}
                if "session_token" not in cols:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN session_token TEXT DEFAULT 'default_session';")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}(session_token);")
        conn.commit()

    def _init_db(self):
        """Initializes tables if they do not already exist and runs migrations."""
        if is_postgres():
            init_postgres_schema()
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    total_characters INTEGER NOT NULL,
                    total_words INTEGER NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    session_token TEXT DEFAULT 'default_session'
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    character_count INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    session_token TEXT DEFAULT 'default_session',
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS quiz_sessions (
                    session_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    questions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    session_token TEXT DEFAULT 'default_session',
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_quiz_sessions_doc_id ON quiz_sessions(document_id);"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    session_token TEXT DEFAULT 'default_session',
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_sessions_doc_id ON chat_sessions(document_id);"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    cited_sources_json TEXT,
                    retrieval_status TEXT,
                    generation_provider TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);"
            )
            conn.commit()
            self._migrate_schema(conn)

    def save_document(
        self,
        document_id: str,
        filename: str,
        page_count: int,
        total_characters: int,
        total_words: int,
        chunks: List[DocumentChunk],
        embeddings: List[np.ndarray],
        session_token: str = "default_session",
    ) -> None:
        """
        Saves document metadata and its corresponding chunk embeddings.
        Replaces existing document if same ID is uploaded.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})."
            )

        clean_token = session_token.strip() if session_token and session_token.strip() else "default_session"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Delete existing document if present
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

            # Insert document record
            cursor.execute(
                """
                INSERT INTO documents (id, filename, page_count, total_characters, total_words, chunk_count, created_at, session_token)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    page_count,
                    total_characters,
                    total_words,
                    len(chunks),
                    now_iso,
                    clean_token,
                ),
            )

            # Insert chunk records
            chunk_records = []
            for chunk, emb in zip(chunks, embeddings):
                # Ensure float32 contiguous bytes
                emb_float32 = np.asarray(emb, dtype=np.float32)
                chunk_records.append(
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.document_name,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.character_count,
                        chunk.word_count,
                        emb_float32.tobytes(),
                        clean_token,
                    )
                )

            cursor.executemany(
                """
                INSERT INTO document_chunks (
                    id, document_id, document_name, page_number,
                    chunk_index, text, character_count, word_count, embedding, session_token
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_records,
            )
            conn.commit()

    def list_documents(self, session_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns metadata for stored documents, optionally scoped to session_token."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_token:
                cursor.execute(
                    """
                    SELECT id, filename, page_count, total_characters, total_words, chunk_count, created_at
                    FROM documents
                    WHERE session_token = ?
                    ORDER BY created_at DESC
                    """,
                    (session_token.strip(),),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, filename, page_count, total_characters, total_words, chunk_count, created_at
                    FROM documents
                    ORDER BY created_at DESC
                    """
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_document(self, document_id: str, session_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a single document by its ID, enforcing session_token ownership if provided."""
        clean_doc_id = document_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_token:
                cursor.execute(
                    """
                    SELECT id, filename, page_count, total_characters, total_words, chunk_count, created_at
                    FROM documents
                    WHERE id = ? AND session_token = ?
                    """,
                    (clean_doc_id, session_token.strip()),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, filename, page_count, total_characters, total_words, chunk_count, created_at
                    FROM documents
                    WHERE id = ?
                    """,
                    (clean_doc_id,),
                )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_document(self, document_id: str, session_token: Optional[str] = None) -> bool:
        """Deletes a document and its associated chunks from persistent storage, scoped to session_token."""
        clean_doc_id = document_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_token:
                cursor.execute("DELETE FROM documents WHERE id = ? AND session_token = ?", (clean_doc_id, session_token.strip()))
            else:
                cursor.execute("DELETE FROM documents WHERE id = ?", (clean_doc_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def get_document_chunks(self, document_id: str, session_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all chunks belonging to a document ordered sequentially by chunk_index.
        Verifies session ownership if session_token is provided.
        """
        clean_doc_id = document_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_token:
                cursor.execute(
                    "SELECT id FROM documents WHERE id = ? AND session_token = ?",
                    (clean_doc_id, session_token.strip()),
                )
                if not cursor.fetchone():
                    return []

            cursor.execute(
                """
                SELECT id, document_id, document_name, page_number, chunk_index, text, character_count, word_count
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (clean_doc_id,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        document_id: Optional[str] = None,
        session_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks matching query_embedding by cosine similarity.
        Restricts candidate chunks strictly to session_token (and document_id if provided).
        """
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        clean_doc_id = document_id.strip() if document_id and document_id.strip() else None
        clean_token = session_token.strip() if session_token and session_token.strip() else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if clean_token and clean_doc_id:
                cursor.execute(
                    """
                    SELECT id, document_id, document_name, page_number, chunk_index, text, embedding
                    FROM document_chunks
                    WHERE session_token = ? AND document_id = ?
                    """,
                    (clean_token, clean_doc_id),
                )
            elif clean_token:
                cursor.execute(
                    """
                    SELECT id, document_id, document_name, page_number, chunk_index, text, embedding
                    FROM document_chunks
                    WHERE session_token = ?
                    """,
                    (clean_token,),
                )
            elif clean_doc_id:
                cursor.execute(
                    """
                    SELECT id, document_id, document_name, page_number, chunk_index, text, embedding
                    FROM document_chunks
                    WHERE document_id = ?
                    """,
                    (clean_doc_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, document_id, document_name, page_number, chunk_index, text, embedding
                    FROM document_chunks
                    """
                )
            rows = cursor.fetchall()

        if not rows:
            return []

        scored_results = []
        for row in rows:
            raw_blob = row["embedding"]
            chunk_vec = np.frombuffer(raw_blob, dtype=np.float32)
            c_norm = np.linalg.norm(chunk_vec)
            if c_norm > 0:
                chunk_vec = chunk_vec / c_norm

            sim_score = float(np.dot(query_vec, chunk_vec))
            clamped_score = max(0.0, min(1.0, (sim_score + 1.0) / 2.0 if sim_score < 0 else sim_score))

            scored_results.append(
                {
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "document_name": row["document_name"],
                    "page_number": row["page_number"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "similarity_score": round(clamped_score, 4),
                }
            )

        scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_results[:top_k]

    # =========================================================================
    # AUTHORITATIVE REVISION QUIZ SESSION PERSISTENCE
    # =========================================================================

    def create_quiz_session(
        self,
        session_id: str,
        document_id: str,
        document_name: str,
        difficulty: str,
        questions: List[Dict[str, Any]],
        session_token: str = "default_session",
    ) -> None:
        """
        Stores an authoritative quiz session with full question details and answer keys.
        Enforces that document_id belongs to session_token.
        """
        clean_token = session_token.strip() if session_token and session_token.strip() else "default_session"
        clean_doc_id = document_id.strip()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        questions_serialized = json.dumps(questions)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM documents WHERE id = ? AND session_token = ?", (clean_doc_id, clean_token))
            if not cursor.fetchone():
                raise KeyError(f"Document with ID '{clean_doc_id}' not found or access denied.")

            cursor.execute(
                """
                INSERT INTO quiz_sessions (session_id, document_id, document_name, difficulty, questions_json, created_at, session_token)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, clean_doc_id, document_name, difficulty, questions_serialized, now_iso, clean_token),
            )
            conn.commit()

    save_quiz_session = create_quiz_session

    def get_quiz_session(self, session_id: str, session_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves an authoritative quiz session by its unique session ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_token:
                cursor.execute(
                    """
                    SELECT session_id, document_id, document_name, difficulty, questions_json, created_at
                    FROM quiz_sessions
                    WHERE session_id = ? AND session_token = ?
                    """,
                    (session_id.strip(), session_token.strip()),
                )
            else:
                cursor.execute(
                    """
                    SELECT session_id, document_id, document_name, difficulty, questions_json, created_at
                    FROM quiz_sessions
                    WHERE session_id = ?
                    """,
                    (session_id.strip(),),
                )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["questions"] = json.loads(data["questions_json"])
            return data

    def delete_quiz_session(self, session_id: str, session_token: Optional[str] = None) -> bool:
        """Deletes a quiz session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_token:
                cursor.execute("DELETE FROM quiz_sessions WHERE session_id = ? AND session_token = ?", (session_id.strip(), session_token.strip()))
            else:
                cursor.execute("DELETE FROM quiz_sessions WHERE session_id = ?", (session_id.strip(),))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    # =========================================================================
    # CHAT SESSION & MESSAGE PERSISTENCE METHODS
    # =========================================================================

    def create_chat_session(
        self,
        session_id: str,
        document_id: str,
        title: Optional[str] = None,
        session_token: str = "default_session",
    ) -> Dict[str, Any]:
        """
        Creates a new persistent conversation session attached to a document.
        Enforces that document_id belongs to session_token.
        Raises KeyError if document_id does not exist or access is denied.
        """
        clean_doc_id = document_id.strip()
        clean_session_id = session_id.strip()
        clean_token = session_token.strip() if session_token and session_token.strip() else "default_session"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Verify document exists and belongs to this session_token
            cursor.execute(
                "SELECT filename FROM documents WHERE id = ? AND session_token = ?",
                (clean_doc_id, clean_token),
            )
            doc_row = cursor.fetchone()
            if not doc_row:
                raise KeyError(f"Document with ID '{clean_doc_id}' not found or access denied.")

            filename = doc_row["filename"]
            final_title = title.strip() if title and title.strip() else f"Chat about {filename}"
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            cursor.execute(
                """
                INSERT INTO chat_sessions (id, document_id, title, created_at, updated_at, session_token)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (clean_session_id, clean_doc_id, final_title, now_iso, now_iso, clean_token),
            )
            conn.commit()

            return {
                "id": clean_session_id,
                "document_id": clean_doc_id,
                "title": final_title,
                "created_at": now_iso,
                "updated_at": now_iso,
                "message_count": 0,
            }

    def get_chat_session(self, session_id: str, session_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves a chat session by its unique identifier, including message count.
        Enforces session_token isolation if provided.
        """
        clean_session_id = session_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_token:
                cursor.execute(
                    """
                    SELECT s.id, s.document_id, s.title, s.created_at, s.updated_at,
                           COUNT(m.id) as message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON s.id = m.session_id
                    WHERE s.id = ? AND s.session_token = ?
                    GROUP BY s.id, s.document_id, s.title, s.created_at, s.updated_at
                    """,
                    (clean_session_id, session_token.strip()),
                )
            else:
                cursor.execute(
                    """
                    SELECT s.id, s.document_id, s.title, s.created_at, s.updated_at,
                           COUNT(m.id) as message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON s.id = m.session_id
                    WHERE s.id = ?
                    GROUP BY s.id, s.document_id, s.title, s.created_at, s.updated_at
                    """,
                    (clean_session_id,),
                )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_chat_sessions(
        self, document_id: Optional[str] = None, session_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Lists chat sessions ordered by most recently updated, optionally filtered by document_id and session_token.
        """
        clean_doc_id = document_id.strip() if document_id and document_id.strip() else None
        clean_token = session_token.strip() if session_token and session_token.strip() else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if clean_token and clean_doc_id:
                cursor.execute(
                    """
                    SELECT s.id, s.document_id, s.title, s.created_at, s.updated_at,
                           COUNT(m.id) as message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON s.id = m.session_id
                    WHERE s.session_token = ? AND s.document_id = ?
                    GROUP BY s.id, s.document_id, s.title, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    """,
                    (clean_token, clean_doc_id),
                )
            elif clean_token:
                cursor.execute(
                    """
                    SELECT s.id, s.document_id, s.title, s.created_at, s.updated_at,
                           COUNT(m.id) as message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON s.id = m.session_id
                    WHERE s.session_token = ?
                    GROUP BY s.id, s.document_id, s.title, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    """,
                    (clean_token,),
                )
            elif clean_doc_id:
                cursor.execute(
                    """
                    SELECT s.id, s.document_id, s.title, s.created_at, s.updated_at,
                           COUNT(m.id) as message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON s.id = m.session_id
                    WHERE s.document_id = ?
                    GROUP BY s.id, s.document_id, s.title, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    """,
                    (clean_doc_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT s.id, s.document_id, s.title, s.created_at, s.updated_at,
                           COUNT(m.id) as message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON s.id = m.session_id
                    GROUP BY s.id, s.document_id, s.title, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    """
                )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def delete_chat_session(self, session_id: str, session_token: Optional[str] = None) -> bool:
        """
        Deletes a chat session and cascades deletion to all contained messages.
        Enforces session_token isolation if provided.
        """
        clean_session_id = session_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_token:
                cursor.execute(
                    "DELETE FROM chat_sessions WHERE id = ? AND session_token = ?",
                    (clean_session_id, session_token.strip()),
                )
            else:
                cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (clean_session_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def save_chat_message(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        cited_sources: Optional[List[Dict[str, Any]]] = None,
        retrieval_status: Optional[str] = None,
        generation_provider: Optional[str] = None,
        session_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Saves a user or assistant message to an existing session and updates the session's updated_at timestamp.
        Enforces session_token ownership if provided.
        Raises KeyError if session_id does not exist or access is denied.
        """
        clean_msg_id = message_id.strip()
        clean_session_id = session_id.strip()
        clean_role = role.strip().lower()
        if clean_role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid message role '{role}'. Must be 'user', 'assistant', or 'system'.")

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        sources_json = json.dumps(cited_sources) if cited_sources else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Verify session exists and belongs to session_token if provided
            if session_token:
                cursor.execute(
                    "SELECT id FROM chat_sessions WHERE id = ? AND session_token = ?",
                    (clean_session_id, session_token.strip()),
                )
            else:
                cursor.execute("SELECT id FROM chat_sessions WHERE id = ?", (clean_session_id,))
            if not cursor.fetchone():
                raise KeyError(f"Chat session with ID '{clean_session_id}' not found or access denied.")

            # 2. Insert message
            cursor.execute(
                """
                INSERT INTO chat_messages (id, session_id, role, content, cited_sources_json, retrieval_status, generation_provider, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (clean_msg_id, clean_session_id, clean_role, content, sources_json, retrieval_status, generation_provider, now_iso),
            )

            # 3. Touch session updated_at
            cursor.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now_iso, clean_session_id))
            conn.commit()

            return {
                "id": clean_msg_id,
                "session_id": clean_session_id,
                "role": clean_role,
                "content": content,
                "cited_sources": cited_sources or [],
                "retrieval_status": retrieval_status,
                "generation_provider": generation_provider,
                "created_at": now_iso,
            }

    def get_session_messages(
        self, session_id: str, session_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all messages for a session in chronological order.
        Enforces session_token ownership if provided.
        Raises KeyError if session_id does not exist or access is denied.
        """
        clean_session_id = session_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Verify session exists and belongs to session_token if provided
            if session_token:
                cursor.execute(
                    "SELECT id FROM chat_sessions WHERE id = ? AND session_token = ?",
                    (clean_session_id, session_token.strip()),
                )
            else:
                cursor.execute("SELECT id FROM chat_sessions WHERE id = ?", (clean_session_id,))
            if not cursor.fetchone():
                raise KeyError(f"Chat session with ID '{clean_session_id}' not found or access denied.")

            cursor.execute(
                """
                SELECT id, session_id, role, content, cited_sources_json, retrieval_status, generation_provider, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (clean_session_id,),
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                raw_sources = item.pop("cited_sources_json", None)
                item["cited_sources"] = json.loads(raw_sources) if raw_sources else []
                results.append(item)
            return results


# Singleton instance
store_instance = VectorStore()

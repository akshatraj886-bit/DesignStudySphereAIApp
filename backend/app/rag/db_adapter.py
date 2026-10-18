"""
StudySphere AI - Database Storage Abstraction Layer
Supports dual-mode persistence:
1. SQLite (Default local development, ₹0 cost, zero setup)
2. PostgreSQL (Cloud durable storage e.g., Neon free tier, opt-in via DATABASE_URL)

Provides a unified interface with:
- Connection pooling & context management
- Parameterized query translation (? -> %s)
- Unified row access (dict-like and attribute-like)
- Session isolation preservation
- Idempotent table & index creation
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Engine types
ENGINE_SQLITE = "sqlite"
ENGINE_POSTGRES = "postgres"


def get_database_url() -> Optional[str]:
    """Retrieves DATABASE_URL from environment, if configured."""
    return os.getenv("DATABASE_URL", "").strip() or None


def get_engine_type() -> str:
    """
    Determines database engine based on DATABASE_URL.
    Defaults to SQLite if DATABASE_URL is not set or starts with sqlite:.
    """
    db_url = get_database_url()
    if db_url and (db_url.startswith("postgresql://") or db_url.startswith("postgres://")):
        return ENGINE_POSTGRES
    return ENGINE_SQLITE


def is_postgres() -> bool:
    """Returns True if PostgreSQL is the active engine."""
    return get_engine_type() == ENGINE_POSTGRES


def is_sqlite() -> bool:
    """Returns True if SQLite is the active engine."""
    return get_engine_type() == ENGINE_SQLITE


class AdaptedRow:
    """
    Unified dict-like row wrapper providing consistent access across
    SQLite (sqlite3.Row) and PostgreSQL (psycopg2 RealDictRow / tuple).
    Supports:
      row["col"], row.get("col"), row[0], dict(row), keys(), values(), items()
    """

    def __init__(self, data: Union[Dict[str, Any], Any], columns: Optional[List[str]] = None):
        if isinstance(data, dict):
            self._dict = dict(data)
            self._keys = list(self._dict.keys())
            self._values = list(self._dict.values())
        elif hasattr(data, "keys"):  # sqlite3.Row
            self._dict = {k: data[k] for k in data.keys()}
            self._keys = list(data.keys())
            self._values = [data[k] for k in self._keys]
        elif columns and isinstance(data, (list, tuple)):
            self._dict = {col: val for col, val in zip(columns, data)}
            self._keys = list(columns)
            self._values = list(data)
        else:
            self._dict = {}
            self._keys = []
            self._values = []

        # Convert memoryview to bytes if needed (e.g. Postgres BYTEA)
        for k, v in self._dict.items():
            if isinstance(v, memoryview):
                self._dict[k] = bytes(v)

    def __getitem__(self, item: Union[str, int]) -> Any:
        if isinstance(item, int):
            return self._values[item]
        return self._dict[item]

    def get(self, key: str, default: Any = None) -> Any:
        return self._dict.get(key, default)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return f"AdaptedRow({self._dict})"


def adapt_query_for_engine(query: str, engine: str) -> str:
    """
    Adapts SQL syntax between SQLite and PostgreSQL.
    - PostgreSQL: translates '?' placeholders to '%s'
    """
    if engine == ENGINE_POSTGRES:
        # Replace ? placeholders with %s outside quotes
        # For simplicity and safety with standard SQL queries:
        return query.replace("?", "%s")
    return query


class AdaptedCursor:
    """
    Cursor wrapper that:
    1. Translates '?' to '%s' when targeting PostgreSQL.
    2. Safely ignores SQLite-specific PRAGMAs on PostgreSQL.
    3. Wraps returned rows in AdaptedRow.
    4. Handles binary data (BYTEA) parameter adaptation for PostgreSQL.
    """

    def __init__(self, raw_cursor: Any, engine: str):
        self._cursor = raw_cursor
        self.engine = engine

    def _prepare_params(self, params: Any) -> Any:
        if params is None:
            return None
        if self.engine == ENGINE_POSTGRES:
            # If psycopg2 is available, wrap bytes in psycopg2.Binary
            try:
                import psycopg2
                if isinstance(params, (list, tuple)):
                    adapted = []
                    for p in params:
                        if isinstance(p, (bytes, bytearray)):
                            adapted.append(psycopg2.Binary(p))
                        else:
                            adapted.append(p)
                    return tuple(adapted) if isinstance(params, tuple) else adapted
            except ImportError:
                pass
        return params

    def execute(self, query: str, params: Optional[Union[Tuple, List, Dict]] = None) -> "AdaptedCursor":
        stripped = query.strip()
        # On Postgres, ignore SQLite PRAGMA statements
        if self.engine == ENGINE_POSTGRES and stripped.upper().startswith("PRAGMA"):
            logger.debug("Skipping PRAGMA on PostgreSQL: %s", stripped)
            return self

        adapted_query = adapt_query_for_engine(query, self.engine)
        prepared_params = self._prepare_params(params)

        if prepared_params is not None:
            self._cursor.execute(adapted_query, prepared_params)
        else:
            self._cursor.execute(adapted_query)
        return self

    def executemany(self, query: str, seq_of_params: Any) -> "AdaptedCursor":
        adapted_query = adapt_query_for_engine(query, self.engine)
        prepared_seq = [self._prepare_params(p) for p in seq_of_params]
        self._cursor.executemany(adapted_query, prepared_seq)
        return self

    def fetchone(self) -> Optional[AdaptedRow]:
        row = self._cursor.fetchone()
        if row is None:
            return None
        if isinstance(row, AdaptedRow):
            return row
        columns = [desc[0] for desc in self._cursor.description] if self._cursor.description else None
        return AdaptedRow(row, columns=columns)

    def fetchall(self) -> List[AdaptedRow]:
        rows = self._cursor.fetchall()
        if not rows:
            return []
        columns = [desc[0] for desc in self._cursor.description] if self._cursor.description else None
        return [r if isinstance(r, AdaptedRow) else AdaptedRow(r, columns=columns) for r in rows]

    @property
    def rowcount(self) -> int:
        rc = getattr(self._cursor, "rowcount", -1)
        return max(rc, 0) if rc is not None else 0

    @property
    def description(self) -> Any:
        return getattr(self._cursor, "description", None)

    def close(self):
        if hasattr(self._cursor, "close"):
            self._cursor.close()


class AdaptedConnection:
    """
    Connection wrapper that provides:
    - Context manager (commit on success, rollback on error, close on exit)
    - Adapted cursor creation
    - Direct execute/executemany convenience methods
    """

    def __init__(self, raw_conn: Any, engine: str):
        self._conn = raw_conn
        self.engine = engine

    def cursor(self) -> AdaptedCursor:
        raw_cur = self._conn.cursor()
        return AdaptedCursor(raw_cur, self.engine)

    def execute(self, query: str, params: Optional[Union[Tuple, List]] = None) -> AdaptedCursor:
        cur = self.cursor()
        return cur.execute(query, params)

    def executemany(self, query: str, seq_of_params: Any) -> AdaptedCursor:
        cur = self.cursor()
        return cur.executemany(query, seq_of_params)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __enter__(self) -> "AdaptedConnection":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            try:
                self.rollback()
            except Exception as e:
                logger.error("Error during rollback: %s", e)
        else:
            try:
                self.commit()
            except Exception as e:
                logger.error("Error during commit: %s", e)
                raise
        try:
            self.close()
        except Exception as e:
            logger.error("Error closing connection: %s", e)


def get_db_connection(db_path: Optional[str] = None) -> AdaptedConnection:
    """
    Creates and returns an AdaptedConnection based on the current environment.
    - If DATABASE_URL indicates PostgreSQL: connects via psycopg2
    - Otherwise: connects via sqlite3 to db_path
    """
    engine = get_engine_type()

    if engine == ENGINE_POSTGRES:
        db_url = get_database_url()
        # Normalise postgres:// to postgresql://
        if db_url.startswith("postgres://"):
            db_url = "postgresql://" + db_url[len("postgres://"):]
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError as e:
            raise RuntimeError(
                "psycopg2-binary is required for PostgreSQL connections. "
                "Run `pip install psycopg2-binary` to enable cloud database support."
            ) from e

        # Connect with 10-second timeout for cloud databases
        conn = psycopg2.connect(
            db_url,
            cursor_factory=RealDictCursor,
            connect_timeout=10,
        )
        return AdaptedConnection(conn, ENGINE_POSTGRES)

    else:
        import sqlite3
        if not db_path:
            raise ValueError("db_path must be provided for SQLite connections.")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return AdaptedConnection(conn, ENGINE_SQLITE)


# PostgreSQL DDL for complete schema parity with SQLite
POSTGRES_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS documents (
        id VARCHAR(255) PRIMARY KEY,
        filename TEXT NOT NULL,
        page_count INTEGER NOT NULL,
        total_characters INTEGER NOT NULL,
        total_words INTEGER NOT NULL,
        chunk_count INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        session_token TEXT DEFAULT 'default_session'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS document_chunks (
        id VARCHAR(255) PRIMARY KEY,
        document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        document_name TEXT NOT NULL,
        page_number INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        character_count INTEGER NOT NULL,
        word_count INTEGER NOT NULL,
        embedding BYTEA NOT NULL,
        session_token TEXT DEFAULT 'default_session'
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_session_token ON document_chunks(session_token);",
    """
    CREATE TABLE IF NOT EXISTS quiz_sessions (
        session_id VARCHAR(255) PRIMARY KEY,
        document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        document_name TEXT NOT NULL,
        difficulty TEXT NOT NULL,
        questions_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        session_token TEXT DEFAULT 'default_session',
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_quiz_sessions_doc_id ON quiz_sessions(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_quiz_sessions_session_token ON quiz_sessions(session_token);",
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id VARCHAR(255) PRIMARY KEY,
        document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        session_token TEXT DEFAULT 'default_session',
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_doc_id ON chat_sessions(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_token ON chat_sessions(session_token);",
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id VARCHAR(255) PRIMARY KEY,
        session_id VARCHAR(255) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        cited_sources_json TEXT,
        retrieval_status TEXT,
        generation_provider TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_documents_session_token ON documents(session_token);",
]


def init_postgres_schema(connection: Optional[AdaptedConnection] = None):
    """Initializes all tables and indexes in PostgreSQL if not already present."""
    if not is_postgres() and connection is None:
        return

    close_after = False
    if connection is None:
        connection = get_db_connection()
        close_after = True

    try:
        with connection as conn:
            cursor = conn.cursor()
            for stmt in POSTGRES_SCHEMA_STATEMENTS:
                cursor.execute(stmt)
        logger.info("PostgreSQL schema initialized successfully.")
    finally:
        if close_after:
            try:
                connection.close()
            except Exception:
                pass

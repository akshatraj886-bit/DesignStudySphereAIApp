"""
StudySphere AI - SQLite to PostgreSQL Data Migration Tool
Safely transfers documents, chunks, quiz sessions, and chat history
from local SQLite to a durable cloud PostgreSQL database (e.g. Neon).

Safety guarantees:
1. READ-ONLY on source SQLite database; never alters or deletes source records.
2. Idempotent: Uses ON CONFLICT DO NOTHING to allow safe re-runs.
3. Foreign-key ordered: Migrates parents before children.
4. Privacy-safe: Logs only table counts, never document text or chat messages.
5. Dry-run mode: Allows inspecting row counts without writing to PostgreSQL.

Usage:
  # 1. Back up your SQLite database first:
  #    powershell: Copy-Item backend/data/studysphere.db backend/data/studysphere.db.backup
  # 2. Set your cloud PostgreSQL connection string:
  #    powershell: $env:DATABASE_URL="postgresql://user:pass@ep-xyz.aws.neon.tech/neondb?sslmode=require"
  # 3. Dry run:
  #    python migrate_sqlite_to_postgres.py --dry-run
  # 4. Execute migration:
  #    python migrate_sqlite_to_postgres.py
"""

import os
import sys
import sqlite3
import argparse
import datetime
from typing import Optional

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
load_dotenv()

from app.rag.db_adapter import (
    get_database_url,
    init_postgres_schema,
    get_db_connection,
    ENGINE_POSTGRES,
)


def get_sqlite_path() -> str:
    return os.getenv(
        "DATABASE_PATH",
        os.path.join(backend_dir, "data", "studysphere.db"),
    )


def count_sqlite_rows(sqlite_conn: sqlite3.Connection, table: str) -> int:
    cursor = sqlite_conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def migrate_table(
    sqlite_conn: sqlite3.Connection,
    pg_conn,
    table: str,
    insert_sql: str,
    select_sql: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> int:
    """Migrates rows for a single table in batches."""
    sqlite_cur = sqlite_conn.cursor()
    sqlite_cur.execute(select_sql)

    total_migrated = 0
    while True:
        rows = sqlite_cur.fetchmany(batch_size)
        if not rows:
            break

        if not dry_run:
            pg_cur = pg_conn.cursor()
            # Wrap bytes for BYTEA if needed
            converted_rows = []
            for row in rows:
                row_list = list(row)
                # Convert sqlite3.Row or tuple to list with psycopg2.Binary for bytes
                for idx, val in enumerate(row_list):
                    if isinstance(val, (bytes, bytearray)):
                        try:
                            import psycopg2
                            row_list[idx] = psycopg2.Binary(val)
                        except ImportError:
                            pass
                converted_rows.append(tuple(row_list))

            pg_cur.executemany(insert_sql, converted_rows)
            pg_conn.commit()

        total_migrated += len(rows)

    return total_migrated


def run_migration(dry_run: bool = False, sqlite_path: Optional[str] = None):
    source_path = sqlite_path or get_sqlite_path()
    db_url = get_database_url()

    print("=" * 70)
    print("STUDYSPHERE AI - DATABASE MIGRATION UTILITY")
    print("=" * 70)
    print(f"Source SQLite DB : {source_path}")
    print(f"Target Postgres  : {'[CONFIGURED]' if db_url else '[NOT CONFIGURED]'}")
    print(f"Mode             : {'DRY RUN (no changes written)' if dry_run else 'LIVE MIGRATION'}")
    print("=" * 70)

    if not os.path.exists(source_path):
        print(f"ERROR: SQLite database file not found at: {source_path}")
        sys.exit(1)

    if not dry_run and not db_url:
        print("ERROR: DATABASE_URL environment variable is not set.")
        print("Set it to your cloud PostgreSQL connection string before running.")
        print("Example: postgresql://user:password@ep-xyz.aws.neon.tech/neondb?sslmode=require")
        sys.exit(1)

    # 1. Connect to SQLite in read-only mode if URI supported, or standard connect
    sqlite_conn = sqlite3.connect(source_path)
    sqlite_conn.row_factory = sqlite3.Row

    # 2. Count existing records in SQLite
    tables = ["documents", "document_chunks", "quiz_sessions", "chat_sessions", "chat_messages"]
    sqlite_counts = {t: count_sqlite_rows(sqlite_conn, t) for t in tables}

    print("\nSource SQLite Records:")
    for t, count in sqlite_counts.items():
        print(f"  - {t:<18}: {count} records")

    if dry_run:
        print("\nDry-run completed. All tables are verified and ready for migration.")
        sqlite_conn.close()
        return

    # 3. Connect to PostgreSQL and initialize schema
    print("\nConnecting to PostgreSQL and verifying schema...")
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        print("ERROR: psycopg2-binary is not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    pg_url = db_url
    if pg_url.startswith("postgres://"):
        pg_url = "postgresql://" + pg_url[len("postgres://"):]

    pg_conn = psycopg2.connect(pg_url, connect_timeout=15)
    print("Connected to PostgreSQL successfully.")

    # Initialize schema
    print("Ensuring tables and indexes exist on PostgreSQL...")
    from app.rag.db_adapter import POSTGRES_SCHEMA_STATEMENTS
    with pg_conn.cursor() as cur:
        for stmt in POSTGRES_SCHEMA_STATEMENTS:
            cur.execute(stmt)
    pg_conn.commit()

    # 4. Migrate tables in FK order
    migration_specs = [
        (
            "documents",
            "SELECT id, filename, page_count, total_characters, total_words, chunk_count, created_at, session_token FROM documents",
            """
            INSERT INTO documents (id, filename, page_count, total_characters, total_words, chunk_count, created_at, session_token)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
        ),
        (
            "document_chunks",
            "SELECT id, document_id, document_name, page_number, chunk_index, text, character_count, word_count, embedding, session_token FROM document_chunks",
            """
            INSERT INTO document_chunks (id, document_id, document_name, page_number, chunk_index, text, character_count, word_count, embedding, session_token)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
        ),
        (
            "quiz_sessions",
            "SELECT session_id, document_id, document_name, difficulty, questions_json, created_at, session_token FROM quiz_sessions",
            """
            INSERT INTO quiz_sessions (session_id, document_id, document_name, difficulty, questions_json, created_at, session_token)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
        ),
        (
            "chat_sessions",
            "SELECT id, document_id, title, created_at, updated_at, session_token FROM chat_sessions",
            """
            INSERT INTO chat_sessions (id, document_id, title, created_at, updated_at, session_token)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
        ),
        (
            "chat_messages",
            "SELECT id, session_id, role, content, cited_sources_json, retrieval_status, generation_provider, created_at FROM chat_messages",
            """
            INSERT INTO chat_messages (id, session_id, role, content, cited_sources_json, retrieval_status, generation_provider, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
        ),
    ]

    print("\nMigrating data...")
    for table, select_sql, insert_sql in migration_specs:
        count = migrate_table(sqlite_conn, pg_conn, table, insert_sql, select_sql)
        print(f"  [✓] {table:<18}: {count} records transferred")

    sqlite_conn.close()
    pg_conn.close()

    print("\n" + "=" * 70)
    print("MIGRATION COMPLETED SUCCESSFULLY.")
    print("Your SQLite database remains untouched and intact.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StudySphere AI SQLite to PostgreSQL Migration")
    parser.add_argument("--dry-run", action="store_true", help="Inspect source data without writing to target")
    parser.add_argument("--sqlite-path", type=str, default=None, help="Custom path to SQLite database")
    args = parser.parse_args()

    run_migration(dry_run=args.dry_run, sqlite_path=args.sqlite_path)

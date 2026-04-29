"""Tests for the FlexVec MCP adapter."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from flexvec.mcp_core import execute_query
from flexvec import mcp_server


def _keyword_db(path: Path) -> None:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE _raw_chunks (id TEXT PRIMARY KEY, content TEXT)")
    db.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, content='_raw_chunks', content_rowid='rowid'
        )
        """
    )
    db.execute("INSERT INTO _raw_chunks VALUES ('a', 'authentication security')")
    db.execute("INSERT INTO _raw_chunks VALUES ('b', 'database migration')")
    db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    db.commit()
    db.close()


def test_mcp_core_executes_readonly_sql_and_blocks_writes():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
    db.execute("INSERT INTO items (name) VALUES ('alpha')")

    assert '"alpha"' in execute_query(db, "SELECT name FROM items")
    assert "Write operations not allowed" in execute_query(db, "DELETE FROM items")


def test_mcp_server_opens_readonly_and_materializes_keyword(tmp_path: Path):
    db_path = tmp_path / "sample.db"
    _keyword_db(db_path)

    db = mcp_server.open_database(db_path)
    try:
        result = json.loads(
            mcp_server.execute_query(
                db,
                "SELECT k.id, k.rank FROM keyword('authentication') k ORDER BY k.rank DESC",
            )
        )
        assert result[0]["id"] == "a"
        assert "Write operations not allowed" in mcp_server.execute_query(
            db, "INSERT INTO _raw_chunks VALUES ('c', 'write')"
        )
    finally:
        db.close()


def test_query_database_supports_sql_keyword_only_mode(tmp_path: Path):
    db_path = tmp_path / "sample.db"
    _keyword_db(db_path)

    mcp_server.configure(db_path, no_embed=True)
    result = json.loads(mcp_server.query_database("SELECT COUNT(*) AS n FROM _raw_chunks"))

    assert result == [{"n": 2}]


def test_mcp_files_do_not_import_flex_core():
    root = Path(__file__).resolve().parents[1] / "flexvec"
    for rel in ("mcp_core.py", "mcp_server.py"):
        source = (root / rel).read_text(encoding="utf-8")
        assert "from flex." not in source
        assert "import flex." not in source

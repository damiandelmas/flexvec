"""Convenience wrapper — chains both materializers, then executes."""

import json
import sqlite3

from .vec_ops import materialize_vec_ops
from .keyword import materialize_keyword


def execute(db: sqlite3.Connection, sql: str) -> list[dict] | dict:
    """Chain vec_ops and keyword materializers, then execute.

    Returns list of row dicts on success, or error dict on failure.

    Usage:
        from flexvec import execute
        rows = execute(db, "SELECT v.id FROM vec_ops('similar:auth') v LIMIT 10")
    """
    sql = sql.strip()

    # Materialize vec_ops -> temp table
    sql = materialize_vec_ops(db, sql)
    if sql.startswith('{"error"'):
        return json.loads(sql)

    # Materialize keyword -> temp table
    sql = materialize_keyword(db, sql)
    if sql.startswith('{"error"'):
        return json.loads(sql)

    # Execute
    try:
        rows = db.execute(sql).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.DatabaseError as e:
        return {"error": str(e)}

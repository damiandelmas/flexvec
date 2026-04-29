"""FlexVec MCP adapter.

This module exposes a direct SQLite query endpoint for agent clients. It does
not know about Flex registries, cells, modules, services, or release state.
Callers point it at one database file and FlexVec handles read-only SQL plus
the local vec_ops()/keyword() materializers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path

from flexvec.keyword import materialize_keyword
from flexvec.mcp_core import execute_query as _execute_mcp_query
from flexvec.vec_ops import VectorCache, materialize_vec_ops, register_vec_ops


_db_path: Path | None = None
_table = "_raw_chunks"
_embedding_col = "embedding"
_id_col = "id"
_no_embed = False


def open_database(path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database for read-only MCP queries."""
    db_path = Path(path).expanduser().resolve()
    uri = f"file:{db_path}?mode=ro"
    db = sqlite3.connect(uri, uri=True, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def configure(
    db_path: str | Path,
    *,
    table: str = "_raw_chunks",
    embedding_col: str = "embedding",
    id_col: str = "id",
    no_embed: bool = False,
) -> None:
    """Configure the process-global MCP database target."""
    global _db_path, _table, _embedding_col, _id_col, _no_embed
    _db_path = Path(db_path).expanduser().resolve()
    _table = table
    _embedding_col = embedding_col
    _id_col = id_col
    _no_embed = no_embed


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row["name"] for row in db.execute(f"PRAGMA table_info([{table}])").fetchall()}
    except sqlite3.DatabaseError:
        return set()


def _cell_config(db: sqlite3.Connection) -> dict:
    if not _table_columns(db, "_meta"):
        return {}
    try:
        return {
            row["key"]: row["value"]
            for row in db.execute("SELECT key, value FROM _meta WHERE key LIKE 'vec:%'").fetchall()
        }
    except sqlite3.DatabaseError:
        return {}


def register_default_vec_ops(db: sqlite3.Connection) -> bool:
    """Register vec_ops() against the configured embedding table if possible."""
    if _no_embed:
        return False

    columns = _table_columns(db, _table)
    if _embedding_col not in columns or _id_col not in columns:
        return False

    from flexvec.embed import get_embed_fn

    cache = VectorCache()
    cache.load_from_db(db, _table, _embedding_col, _id_col)
    cache.load_columns(db, _table, _id_col)

    # New-form vec_ops() defaults to _raw_chunks; keep custom table support by
    # registering both names against the same cache when needed.
    caches = {_table: cache}
    if _table != "_raw_chunks":
        caches["_raw_chunks"] = cache

    register_vec_ops(db, caches, get_embed_fn(), cell_config=_cell_config(db))
    return cache.size > 0


def materialize(db: sqlite3.Connection, sql: str) -> str:
    """Run FlexVec table-function materializers in the canonical order."""
    sql = materialize_vec_ops(db, sql)
    if sql.startswith('{"error"'):
        return sql
    return materialize_keyword(db, sql)


def execute_query(db: sqlite3.Connection, query: str) -> str:
    """Execute a FlexVec MCP query against an open database connection."""
    return _execute_mcp_query(db, query, materializer=materialize)


def query_database(query: str) -> str:
    """Execute a query against the configured MCP database path."""
    if _db_path is None:
        return json.dumps({"error": "FlexVec MCP database path is not configured."})

    with open_database(_db_path) as db:
        try:
            register_default_vec_ops(db)
        except ImportError as exc:
            if "flexvec[embed]" in str(exc):
                return json.dumps(
                    {
                        "error": str(exc),
                        "hint": "Install with: pip install 'flexvec[mcp]' or run with --no-embed for SQL/keyword-only queries.",
                    }
                )
            raise
        return execute_query(db, query)


def build_tool_description() -> str:
    return (
        "Query one SQLite database through FlexVec. Use read-only SQL. "
        "keyword('term') works when chunks_fts/_raw_chunks exist; "
        "vec_ops('similar:text') works when embeddings and flexvec[mcp] are installed."
    )


def build_tool_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Read-only SQL query. Table functions: vec_ops('similar:text'), keyword('term').",
            }
        },
        "required": ["query"],
    }


def get_server():
    """Build the MCP server. Imports mcp lazily so flexvec core stays minimal."""
    from mcp.server.lowlevel import Server
    import mcp.types as types

    server = Server("flexvec")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="flexvec_search",
                description=build_tool_description(),
                inputSchema=build_tool_schema(),
            )
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        if name != "flexvec_search":
            return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
        if not arguments or "query" not in arguments:
            return [types.TextContent(type="text", text=json.dumps({"error": "Missing required argument: query"}))]
        result = await asyncio.to_thread(query_database, arguments["query"])
        return [types.TextContent(type="text", text=result)]

    return server


async def _run_stdio() -> None:
    from mcp.server.stdio import stdio_server

    server = get_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the FlexVec MCP server over stdio.")
    parser.add_argument("db", help="SQLite database path")
    parser.add_argument("--table", default="_raw_chunks", help="Embedding table name")
    parser.add_argument("--embedding-col", default="embedding", help="Embedding BLOB column")
    parser.add_argument("--id-col", default="id", help="Embedding row id column")
    parser.add_argument("--no-embed", action="store_true", help="Skip vec_ops registration")
    args = parser.parse_args(argv)

    configure(
        args.db,
        table=args.table,
        embedding_col=args.embedding_col,
        id_col=args.id_col,
        no_embed=args.no_embed,
    )
    asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()

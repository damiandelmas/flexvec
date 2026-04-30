"""Agent-native SQLite retrieval contracts for FlexVec."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


META_TABLE = "_flexvec_meta"
DEFAULT_CHUNK_TABLE = "_raw_chunks"
DEFAULT_FTS_TABLE = "chunks_fts"
DEFAULT_EMBEDDING_COL = "embedding"


@dataclass
class RetrievalSpec:
    """Machine-editable contract for turning one SQLite table into retrieval rows."""

    table: str
    id_col: str
    text_col: str = ""
    text_cols: list[str] = field(default_factory=list)
    metadata_cols: list[str] = field(default_factory=list)
    chunk_table: str = DEFAULT_CHUNK_TABLE
    fts_table: str = DEFAULT_FTS_TABLE
    embedding_col: str = DEFAULT_EMBEDDING_COL

    def __post_init__(self) -> None:
        if self.text_cols:
            self.text_cols = [str(col) for col in self.text_cols]
            if not self.text_col:
                self.text_col = self.text_cols[0]
        elif self.text_col:
            self.text_cols = [str(self.text_col)]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RetrievalSpec":
        id_col = data.get("id_col") or data.get("id_column")
        raw_text_cols = data.get("text_cols") or data.get("text_columns")
        text_cols: list[str]
        if raw_text_cols:
            if isinstance(raw_text_cols, str):
                text_cols = [raw_text_cols]
            else:
                text_cols = [str(col) for col in raw_text_cols if str(col)]
        else:
            text_col = data.get("text_col") or data.get("text_column")
            text_cols = [str(text_col)] if text_col else []

        missing = []
        if not data.get("table"):
            missing.append("table")
        if not id_col:
            missing.append("id_col")
        if not text_cols:
            missing.append("text_col")
        if missing:
            raise ValueError(f"Missing required spec fields: {', '.join(missing)}")
        return cls(
            table=str(data["table"]),
            id_col=str(id_col),
            text_col=str(data.get("text_col") or data.get("text_column") or text_cols[0]),
            text_cols=text_cols,
            metadata_cols=[str(col) for col in data.get("metadata_cols", [])],
            chunk_table=str(data.get("chunk_table") or DEFAULT_CHUNK_TABLE),
            fts_table=str(data.get("fts_table") or DEFAULT_FTS_TABLE),
            embedding_col=str(data.get("embedding_col") or DEFAULT_EMBEDDING_COL),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def quote_ident(identifier: str) -> str:
    if not identifier:
        raise ValueError("SQLite identifier cannot be empty")
    return '"' + identifier.replace('"', '""') + '"'


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def connect(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    db_path = Path(path).expanduser()
    if readonly:
        uri = f"file:{db_path.resolve()}?mode=ro"
        db = sqlite3.connect(uri, uri=True)
    else:
        db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    return db


def table_columns(db: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = db.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [
        {
            "name": row["name"],
            "type": row["type"],
            "notnull": bool(row["notnull"]),
            "pk": bool(row["pk"]),
        }
        for row in rows
    ]


def table_exists(db: sqlite3.Connection, table: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? AND type IN ('table', 'view')",
        (table,),
    ).fetchone()
    return row is not None


def inspect_database(path: str | Path) -> dict[str, Any]:
    """Return table/column facts and likely id/text candidates for an agent."""
    with connect(path, readonly=True) as db:
        table_rows = db.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        tables = []
        for row in table_rows:
            name = row["name"]
            columns = table_columns(db, name)
            try:
                count = db.execute(f"SELECT COUNT(*) AS n FROM {quote_ident(name)}").fetchone()["n"]
            except sqlite3.DatabaseError:
                count = None

            id_candidates = [
                col["name"]
                for col in columns
                if col["pk"] or col["name"].lower() in {"id", "uuid", "key", "rowid"}
            ]
            text_candidates = [
                col["name"]
                for col in columns
                if "TEXT" in (col["type"] or "").upper()
                or col["name"].lower() in {"text", "body", "content", "description", "title"}
            ]
            tables.append(
                {
                    "name": name,
                    "type": row["type"],
                    "row_count": count,
                    "columns": columns,
                    "id_candidates": id_candidates,
                    "text_candidates": text_candidates,
                }
            )
        return {"database": str(Path(path).expanduser()), "tables": tables}


def load_spec(path: str | Path | None, db: sqlite3.Connection | None = None) -> RetrievalSpec:
    if path:
        return RetrievalSpec.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
    if db is None:
        raise ValueError("A spec path or database connection is required")
    if not table_exists(db, META_TABLE):
        raise ValueError("No FlexVec spec found in _flexvec_meta; pass --spec")
    row = db.execute(f"SELECT value FROM {quote_ident(META_TABLE)} WHERE key = 'spec'").fetchone()
    if not row:
        raise ValueError("No FlexVec spec found in _flexvec_meta; pass --spec")
    return RetrievalSpec.from_mapping(json.loads(row["value"]))


def write_spec(db: sqlite3.Connection, spec: RetrievalSpec) -> None:
    db.execute(
        f"CREATE TABLE IF NOT EXISTS {quote_ident(META_TABLE)} "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    pairs = {
        "spec": spec.to_json(),
        "version": "1",
        "prepared_at": str(int(time.time())),
    }
    db.executemany(
        f"INSERT OR REPLACE INTO {quote_ident(META_TABLE)} (key, value) VALUES (?, ?)",
        pairs.items(),
    )


def _ensure_column(db: sqlite3.Connection, table: str, name: str, decl: str) -> None:
    existing = {col["name"] for col in table_columns(db, table)}
    if name not in existing:
        db.execute(f"ALTER TABLE {quote_ident(table)} ADD COLUMN {quote_ident(name)} {decl}")


def prepare_database(path: str | Path, spec: RetrievalSpec) -> dict[str, Any]:
    """Create FlexVec-owned retrieval tables inside an existing SQLite DB."""
    with connect(path) as db:
        if not table_exists(db, spec.table):
            raise ValueError(f"Source table does not exist: {spec.table}")
        source_cols = {col["name"] for col in table_columns(db, spec.table)}
        required = {spec.id_col, *spec.text_cols, *spec.metadata_cols}
        missing = sorted(required - source_cols)
        if missing:
            raise ValueError(f"Source columns missing from {spec.table}: {', '.join(missing)}")

        db.execute(
            f"CREATE TABLE IF NOT EXISTS {quote_ident(spec.chunk_table)} ("
            "id TEXT PRIMARY KEY, "
            "content TEXT NOT NULL, "
            f"{quote_ident(spec.embedding_col)} BLOB, "
            "source_table TEXT NOT NULL, "
            "source_id TEXT NOT NULL, "
            "metadata TEXT)"
        )
        # Older experimental tables may exist; make them compatible in place.
        _ensure_column(db, spec.chunk_table, "content", "TEXT")
        _ensure_column(db, spec.chunk_table, spec.embedding_col, "BLOB")
        _ensure_column(db, spec.chunk_table, "source_table", "TEXT")
        _ensure_column(db, spec.chunk_table, "source_id", "TEXT")
        _ensure_column(db, spec.chunk_table, "metadata", "TEXT")

        db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {quote_ident(spec.fts_table)} "
            f"USING fts5(content, content={json.dumps(spec.chunk_table)}, content_rowid='rowid')"
        )
        write_spec(db, spec)
        db.commit()
        return {
            "prepared": True,
            "source_table": spec.table,
            "chunk_table": spec.chunk_table,
            "fts_table": spec.fts_table,
            "spec": asdict(spec),
        }


def _select_source_rows(db: sqlite3.Connection, spec: RetrievalSpec, limit: int | None) -> list[sqlite3.Row]:
    cols = _ordered_unique([spec.id_col, *spec.text_cols, *spec.metadata_cols])
    text_predicate = " OR ".join(
        f"{quote_ident(col)} IS NOT NULL" for col in spec.text_cols
    )
    sql = (
        "SELECT "
        + ", ".join(quote_ident(col) for col in cols)
        + f" FROM {quote_ident(spec.table)} "
        + f"WHERE ({text_predicate})"
    )
    if limit is not None:
        sql += " LIMIT ?"
        return db.execute(sql, (limit,)).fetchall()
    return db.execute(sql).fetchall()


def _embedding_blob(embed_fn: Callable[[str], Any], text: str) -> bytes:
    vec = np.asarray(embed_fn(text), dtype=np.float32)
    if vec.ndim > 1:
        vec = vec.reshape(-1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32).tobytes()


def index_database(
    path: str | Path,
    spec: RetrievalSpec,
    *,
    embed_fn: Callable[[str], Any] | None = None,
    skip_embeddings: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Populate FlexVec retrieval rows and FTS, optionally embedding text."""
    if embed_fn is None and not skip_embeddings:
        from flexvec.embed import get_embed_fn

        embed_fn = get_embed_fn(prefix="search_document: ")

    prepare_database(path, spec)
    with connect(path) as db:
        rows = _select_source_rows(db, spec, limit)
        indexed = 0
        embedded = 0
        emb_col = quote_ident(spec.embedding_col)
        for row in rows:
            source_id = str(row[spec.id_col])
            parts = [str(row[col]) for col in spec.text_cols if row[col] is not None]
            content = "\n".join(part for part in parts if part.strip())
            if not content.strip():
                continue
            metadata = {col: row[col] for col in spec.metadata_cols}
            embedding = None
            if embed_fn is not None and not skip_embeddings:
                embedding = _embedding_blob(embed_fn, content)
                embedded += 1
            db.execute(
                f"INSERT INTO {quote_ident(spec.chunk_table)} "
                f"(id, content, {emb_col}, source_table, source_id, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "content = excluded.content, "
                f"{emb_col} = COALESCE(excluded.{emb_col}, {emb_col}), "
                "source_table = excluded.source_table, "
                "source_id = excluded.source_id, "
                "metadata = excluded.metadata",
                (
                    source_id,
                    content,
                    embedding,
                    spec.table,
                    source_id,
                    json.dumps(metadata, sort_keys=True, default=str),
                ),
            )
            indexed += 1
        db.execute(f"INSERT INTO {quote_ident(spec.fts_table)}({quote_ident(spec.fts_table)}) VALUES('rebuild')")
        db.execute(
            f"INSERT OR REPLACE INTO {quote_ident(META_TABLE)} (key, value) VALUES (?, ?)",
            ("indexed_at", str(int(time.time()))),
        )
        db.commit()
        return {
            "indexed": indexed,
            "embedded": embedded,
            "skip_embeddings": skip_embeddings,
            "chunk_table": spec.chunk_table,
            "fts_table": spec.fts_table,
        }


def doctor_database(path: str | Path, spec_path: str | Path | None = None) -> dict[str, Any]:
    """Report retrieval readiness and coverage for an agent."""
    with connect(path, readonly=True) as db:
        try:
            spec = load_spec(spec_path, db)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        source_count = db.execute(f"SELECT COUNT(*) AS n FROM {quote_ident(spec.table)}").fetchone()["n"]
        chunk_count = db.execute(f"SELECT COUNT(*) AS n FROM {quote_ident(spec.chunk_table)}").fetchone()["n"]
        embedded_count = db.execute(
            f"SELECT COUNT(*) AS n FROM {quote_ident(spec.chunk_table)} "
            f"WHERE {quote_ident(spec.embedding_col)} IS NOT NULL"
        ).fetchone()["n"]
        fts_count = db.execute(f"SELECT COUNT(*) AS n FROM {quote_ident(spec.fts_table)}").fetchone()["n"]
        return {
            "ok": chunk_count > 0 and fts_count == chunk_count,
            "source_table": spec.table,
            "source_rows": source_count,
            "chunk_rows": chunk_count,
            "embedded_rows": embedded_count,
            "fts_rows": fts_count,
            "embedding_coverage": (embedded_count / chunk_count) if chunk_count else 0.0,
            "spec": asdict(spec),
        }

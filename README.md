# flexvec

[![PyPI](https://img.shields.io/pypi/v/flexvec)](https://pypi.org/project/flexvec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Agent-native vector retrieval for SQLite.

FlexVec turns ordinary SQLite databases into structured retrieval databases for
AI agents. It can inspect an existing DB, create a retrieval contract, build a
local retrieval surface, query it with SQL, and expose it over MCP. It also
includes the lower-level vector scoring primitives: suppress a topic, weight by
recency, spread across subtopics, or project through embedding space in one SQL
statement.

No hosted service. No separate vector database. No Flex runtime required.

```bash
pip install "flexvec[mcp]"
```

## Quickstart

Start with any SQLite database:

```bash
flexvec inspect app.db --json
```

Create a retrieval contract:

```json
{
  "table": "docs",
  "id_column": "id",
  "text_columns": ["title", "body"],
  "metadata_cols": ["created_at"]
}
```

Prepare and index the database:

```bash
flexvec prepare app.db --spec spec.json --json
flexvec index app.db --spec spec.json --json
flexvec doctor app.db --json
```

Query with SQL or hand the DB to an agent over MCP:

```bash
flexvec sql app.db "SELECT v.id, v.score, c.content FROM vec_ops('similar:refund policy') v JOIN _raw_chunks c ON c.id = v.id LIMIT 10" --json
flexvec sql app.db "SELECT k.id, k.rank, c.content FROM keyword('refund policy') k JOIN _raw_chunks c ON c.id = k.id LIMIT 10" --no-embed --json
flexvec mcp app.db
```

`prepare` creates FlexVec-owned `_flexvec_meta`, `_raw_chunks`, and `chunks_fts`
surfaces inside the target database. FlexVec does not require a Flex registry,
Flex cells, services, modules, or Labs packages.
If those tables already exist, `prepare`/`index` return warnings before reusing
or rebuilding them; copy the DB first or choose custom table names in the spec
when reuse is not intended.

## What You Get

| surface | purpose |
|---|---|
| `flexvec inspect` | return table/column facts so an agent can choose a source table |
| `flexvec prepare` | create `_flexvec_meta`, `_raw_chunks`, and `chunks_fts` |
| `flexvec index` | copy source rows, rebuild FTS, and optionally embed content |
| `flexvec doctor` | report retrieval readiness and embedding coverage |
| `flexvec sql` | run read-only SQL with `keyword()` and `vec_ops()` materializers |
| `flexvec mcp` / `flexvec-mcp` | expose one SQLite DB as an MCP tool |
| `flexvec skill-path` | print the packaged agent skill path |

All command output is JSON-first for agent workflows.

## Existing Embeddings

If your database already has float32 embedding BLOBs, use FlexVec directly as a
Python library.

Example table:

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding BLOB  -- float32, L2-normalized
);
```

Load embeddings into memory once. Every query after that is a matmul.

```python
import sqlite3
from flexvec import VectorCache, register_vec_ops, execute, get_embed_fn

db = sqlite3.connect("my.db")
cache = VectorCache()
cache.load_from_db(db, "chunks", "embedding", "id")
register_vec_ops(db, {"chunks": cache}, get_embed_fn())
```

Write SQL. flexvec handles the vector math behind the scenes.

```python
rows = execute(db, """
    SELECT v.id, v.score, c.content
    FROM vec_ops('similar:authentication patterns') v
    JOIN chunks c ON v.id = c.id
    ORDER BY v.score DESC LIMIT 5
""")
```

## Examples

### Suppress and diversify

Find authentication patterns without drowning in deployment and testing discussions.

```sql
SELECT v.id, v.score, c.content
FROM vec_ops(
    'similar:authentication patterns
     diverse suppress:deployment suppress:testing',
    'SELECT id FROM chunks WHERE length(content) > 200') v
JOIN chunks c ON v.id = c.id
ORDER BY v.score DESC LIMIT 10
```

`suppress:` pushes deployment and testing content out of the results. `diverse` spreads across subtopics instead of returning ten variations of the same match. The pre-filter scopes to chunks over 200 characters — cutting out noise before anything gets scored.

### Hybrid retrieval

Find the session where you actually fixed that OOM error — not just the logs.

```sql
SELECT k.id, k.rank, v.score, c.content
FROM keyword('OOM') k
JOIN vec_ops('similar:memory limit debugging worker crash fix') v ON k.id = v.id
JOIN chunks c ON k.id = c.id
ORDER BY v.score DESC LIMIT 10
```

`keyword('OOM')` finds every chunk containing the term. `vec_ops()` scores by relevance to debugging and fixing. The JOIN keeps only chunks that match both — exact term plus semantic relevance.

## Tokens

Tokens reshape scores. They compose freely in a single string.

| token | what it does |
|---|---|
| `similar:TEXT` | search for this concept |
| `suppress:TEXT` | push this topic out of results (stackable) |
| `diverse` | spread across subtopics instead of ten versions of the same answer |
| `decay:N` | favor recent content — N-day half-life |
| `centroid:id1,id2` | "more like these" — search from the average of examples |
| `from:A to:B` | find content along a conceptual arc |
| `pool:N` | how many candidates to score (default 500) |

`'similar:auth diverse suppress:oauth decay:7'` — four operations, one query.

## How it works

Every query runs three phases in one SQL statement.

```
SQL pre-filter  →  numpy modulation  →  SQL compose
```

1. **SQL pre-filter** narrows what enters scoring — by date, type, length, or any SQL expression.
2. **numpy modulation** scores candidates and reshapes the score array with tokens before selection.
3. **SQL compose** joins results back to your tables for grouping, filtering, or reranking.

The query path is read-only aside from temporary result tables. The CLI
`prepare` and `index` commands are explicit write operations that add or update
FlexVec retrieval surfaces inside the target database.

## Performance

No ANN index. Brute-force matmul on a numpy matrix.

| corpus | matmul | full pipeline |
|---|---|---|
| 250K | 5ms | 19ms |
| 500K | 7ms | 37ms |
| 1M | 17ms | 82ms |

128 dimensions, Nomic Embed v1.5 (Matryoshka). Pre-filtering narrows candidates before the matmul — scoped queries run in single-digit ms.

## Install

```bash
pip install flexvec              # core (numpy only)
pip install flexvec[embed]       # + ONNX embedder
pip install flexvec[mcp]         # + MCP server and embedder
```

## See also

- **[arXiv paper](https://arxiv.org/abs/2603.22587)** — architecture and evaluation
- **[flex](https://github.com/damiandelmas/flex)** — search and retrieval for AI agents (uses flexvec)
- **[getflex.dev](https://getflex.dev)**

MIT · Python 3.10+ · SQLite · numpy

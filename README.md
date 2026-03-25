<p align="center">
  <img src="assets/banner.png" alt="flexvec" width="100%">
</p>

# flexvec

[![PyPI](https://img.shields.io/pypi/v/flexvec)](https://pypi.org/project/flexvec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Composable vector retrieval with SQL.

flexvec is a Python library that reshapes vector search scores before selection. Suppress a topic, weight by recency, spread across subtopics, project a direction through embedding space — all in one SQL statement. Runs in-process on any SQLite database. No server, no index.

```bash
pip install flexvec
```

## Getting started

### Your table

Any SQLite database with an embedding column works.

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding BLOB  -- float32, L2-normalized
);
```

### Connect

Load embeddings into memory once. Every query after that is a matmul.

```python
import sqlite3
from flexvec import VectorCache, register_vec_ops, execute, get_embed_fn

db = sqlite3.connect("my.db")
cache = VectorCache()
cache.load_from_db(db, "chunks", "embedding", "id")
register_vec_ops(db, {"chunks": cache}, get_embed_fn())
```

### Search

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

The database is never modified. Results materialize as a temp table that SQL composes over.

## Performance

No index. Brute-force matmul on a numpy matrix.

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
pip install flexvec[embed,graph] # everything
```

## See also

- **[arXiv paper](https://arxiv.org/abs/2603.22587)** — architecture and evaluation
- **[flex](https://github.com/damiandelmas/flex)** — search and retrieval for AI agents (uses flexvec)
- **[getflex.dev](https://getflex.dev)**

MIT · Python 3.10+ · SQLite · numpy

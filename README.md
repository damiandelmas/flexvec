<p align="center">
  <img src="assets/banner.png" alt="flexvec" width="100%">
</p>

# flexvec

[![PyPI](https://img.shields.io/pypi/v/flexvec)](https://pypi.org/project/flexvec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Composable vector search with SQL.

Standard retrieval gives you top-K by similarity. flexvec lets you reshape the scores before selection — suppress a topic, weight by recency, spread across subtopics, project a direction through embedding space. All in one SQL statement.

```bash
pip install flexvec
```

## Getting started

All you need is a SQLite table with an embedding column:

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding BLOB  -- float32, L2-normalized
);
```

```python
import sqlite3
from flexvec import VectorCache, register_vec_ops, execute, get_embed_fn

db = sqlite3.connect("my.db")

# Load vectors into memory
cache = VectorCache()
cache.load_from_db(db, "chunks", "embedding", "id")

# Register vec_ops + keyword as SQL functions
embed_fn = get_embed_fn()  # bundled Nomic ONNX model
register_vec_ops(db, {"chunks": cache}, embed_fn)

# Query — SQL in, rows out
rows = execute(db, """
    SELECT v.id, v.score, c.content
    FROM vec_ops('similar:debugging crash fix diverse') v
    JOIN chunks c ON v.id = c.id
    ORDER BY v.score DESC LIMIT 5
""")
```

Any SQLite database with an embedding column works. Load it, register it, query it.

## Sample usage

### Semantic search with modulation

Want to find authentication patterns without drowning in deployment and testing discussions?

```sql
SELECT v.id, v.score, c.content
FROM vec_ops(
    'similar:authentication patterns
     diverse suppress:deployment suppress:testing',
    'SELECT id FROM chunks WHERE length(content) > 200') v
JOIN chunks c ON v.id = c.id
ORDER BY v.score DESC LIMIT 10
```

The SQL pre-filter scopes to chunks longer than 200 characters — cutting out tool calls and one-liners before anything gets scored. `suppress:deployment` and `suppress:testing` push those clusters out of the results so the actual auth architecture surfaces. `diverse` makes sure you get breadth across subtopics — token handling, session management, middleware — instead of ten variations of the same login flow.

### Hybrid retrieval

Remember hitting an OOM error last month but can't find the session where you actually fixed it?

```sql
SELECT k.id, k.rank, v.score, c.content
FROM keyword('OOM') k
JOIN vec_ops('similar:memory limit debugging worker crash fix') v ON k.id = v.id
JOIN chunks c ON k.id = c.id
ORDER BY v.score DESC LIMIT 10
```

`keyword('OOM')` finds every chunk that literally contains "OOM" — could be dozens, most of them just error logs or passing mentions. The semantic side scores by relevance to actually debugging and fixing memory issues. The intersection keeps only the chunks where OOM appears AND the content is about the fix, not just the crash. You skip the noise and land on the session where you solved it.

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

```
SQL pre-filter  →  numpy modulation  →  SQL compose
```

SQL narrows what enters scoring. numpy runs cosine similarity + token modulations on the score array. Results materialize as a temp table — SQL composes the rest. The database is never modified.

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

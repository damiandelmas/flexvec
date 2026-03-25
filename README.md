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

## Sample usage

```sql
-- find auth patterns, suppress deployment noise, spread across subtopics
SELECT v.id, v.score, c.content
FROM vec_ops(
    'similar:authentication patterns
     diverse suppress:deployment suppress:testing',
    'SELECT id FROM chunks WHERE length(content) > 200') v
JOIN chunks c ON v.id = c.id
ORDER BY v.score DESC LIMIT 10
```

```sql
-- hybrid: keyword AND semantic
SELECT k.id, k.rank, v.score, c.content
FROM keyword('JWT rotation') k
JOIN vec_ops('similar:authentication token refresh') v ON k.id = v.id
JOIN chunks c ON k.id = c.id
ORDER BY v.score DESC LIMIT 10
```

## Tokens

Tokens reshape scores. They compose freely in a single string.

| token | what it does |
|---|---|
| `similar:TEXT` | embed text, cosine search |
| `suppress:TEXT` | push a topic out of results (stackable) |
| `diverse` | spread across subtopics instead of clustering |
| `decay:N` | weight recent content — N-day half-life |
| `centroid:id1,id2` | "more like these" — search from the average of examples |
| `from:A to:B` | find content along a conceptual direction |
| `pool:N` | candidate pool size (default 500) |

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

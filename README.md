# flexvec

Vector search returns the most similar results. It doesn't return what's similar *but different from X*, or similar but spread across subtopics, or similar but weighted toward last week.

flexvec keeps embeddings as numpy arrays you can operate on. Retrieve a concept and subtract another from it. Decay by recency. Spread across subtopics. Project a direction through embedding space. Compose them in a query.

```bash
pip install flexvec
```

---

## Three-Phase Pipeline

Every query runs three steps. SQL narrows the candidates. NumPy scores them. SQL composes the results.

```
SQL pre-filter  →  NumPy vector operations  →  SQL compose
```

flexvec filters first — a SQL WHERE clause can reduce 240K chunks to 3K before any vector math runs. At that scale, the matmul is sub-millisecond.

```python
import sqlite3
from flexvec import VectorCache, register_vec_ops, execute, get_embed_fn

db = sqlite3.connect("my.db")

# Load vectors into memory (once)
cache = VectorCache()
cache.load_from_db(db, "chunks", "embedding", "id")

# Default: bundled Nomic ONNX model (pip install flexvec[embed])
embed_fn = get_embed_fn()

# Register vec_ops as a SQL function
register_vec_ops(db, {"chunks": cache}, embed_fn)

# Query — three phases in one statement
rows = execute(db, """
    SELECT v.id, v.score, c.content
    FROM vec_ops('chunks', 'authentication patterns',
      'diverse suppress:oauth decay:7',
      'SELECT id FROM chunks WHERE created_at > 1709000000') v
    JOIN chunks c ON v.id = c.id
    ORDER BY v.score DESC
    LIMIT 10
""")
```

---

## NumPy Operations

Your embeddings are numpy arrays. Each operation runs on the score array before candidate selection.

```python
# cosine similarity — the baseline
scores = candidates @ query_vec

# contrastive — subtract similarity to a concept
neg_scores = candidates @ embed(unlike_text)
scores -= 0.5 * neg_scores

# MMR — iterative selection penalizing redundancy
for each selection:
    score = λ * relevance - (1-λ) * max_similarity_to_selected

# temporal decay — exponential half-life
scores *= 1 / (1 + days_ago / half_life)

# centroid — search from the mean of examples
query = mean(embeddings[id1], embeddings[id2])

# trajectory — direction vector as scoring lens
direction = embed(to_text) - embed(from_text)
scores = 0.5 * query_scores + 0.5 * (candidates @ direction)
```

These compose because they're sequential operations on the same array. You can use them directly or through modulation tokens.

---

## Modulation Tokens

Tokens are a convenience layer — shorthand for the numpy operations above, composable inside a SQL query by an agent or a fixed pipeline.

They compose freely. `diverse suppress:oauth decay:7` is three operations in one pass.

| token | operation |
|---|---|
| `diverse` | MMR — spread results across subtopics |
| `decay:N` | temporal decay with N-day half-life |
| `suppress:TEXT` | contrastive — suppress similarity to a concept |
| `centroid:id1,id2` | centroid — search from the mean of examples |
| `from:T to:T` | trajectory — direction vector through embedding space |
| `communities` | per-query Louvain clustering, adds `_community` column |
| `pool:N` | candidate pool size (default 500) |

---

## Benchmarks

The operations are simple linear algebra — the question is whether they produce useful effects on real retrieval. Tested across four BEIR benchmarks, every token produces consistent measurable deltas.

| Dataset | Domain | Docs | nDCG@10 | diverse ILS | suppress | decay shift | centroid |
|---|---|---|---|---|---|---|---|
| SciFact | Science/claims | 5,183 | 0.60 | 0.46→0.40 | 0.54→0.52 | +45.9 days | 0.64→0.71 |
| NFCorpus | Biomedical | 3,633 | 0.13 | 0.33→0.20 | 0.31→0.24 | +49.6 days | 0.53→0.65 |
| FiQA | Financial QA | 57,638 | 0.13 | 0.51→0.42 | 0.60→0.54 | +42.7 days | 0.68→0.74 |
| SCIDOCS | Citation pred | 25,657 | 0.18 | 0.52→0.41 | 0.58→0.52 | +48.3 days | 0.68→0.75 |

52/52 tests passing. `diverse` reduces intra-list similarity (ILS↓ = more topical spread). `suppress` suppresses the targeted concept. `decay` shifts mean result age forward. `centroid` improves similarity to the example centroid.

nDCG@10 measured at 128d (Matryoshka truncation from 768d) — the operating point chosen for brute-force viability. Absolute scores reflect the dimensionality tradeoff; the token deltas are the finding.

Reproduce: `python -m pytest tests/test_tokens_beir.py`

---

## Materializers

`vec_ops()` and `keyword()` are **materializers** — functions that appear in SQL but execute outside SQLite. The pattern:

1. **Detect** the function call in your SQL string (`vec_ops(...)` or `keyword(...)`)
2. **Execute** the non-SQL engine (numpy matmul or FTS5 BM25)
3. **Materialize** results into a temp table with `(id, score)` columns
4. **Rewrite** the SQL to reference the temp table
5. **Return** the rewritten SQL for SQLite to execute normally

This is why `vec_ops()` and `keyword()` compose — each rewrites its own pattern and leaves the other untouched. `execute()` chains them: vec_ops first, keyword second, then SQLite runs the final query with both temp tables available for JOINs.

---

## FTS5 Keyword Search

`keyword()` is the FTS5 materializer. Same convention as `vec_ops`, returns `(id, rank, snippet)` instead of `(id, score)`.

```python
# Keyword search
rows = execute(db, """
    SELECT k.id, k.rank, k.snippet
    FROM keyword('authentication') k
    ORDER BY k.rank DESC
""")

# Hybrid: only chunks matching BOTH keyword AND semantic
rows = execute(db, """
    SELECT k.id, k.rank, v.score
    FROM keyword('auth') k
    JOIN vec_ops('chunks', 'authentication patterns') v ON k.id = v.id
    ORDER BY k.rank + v.score DESC
    LIMIT 10
""")
```

---

## Performance

Vectors loaded once into a numpy matrix. Queries are a single matmul — no index to build, no server to run.

| corpus size | matmul | full pipeline |
|---|---|---|
| 250K vectors | 5ms | 19ms |
| 500K vectors | 7ms | 37ms |
| 1M vectors | 17ms | 82ms |

Full pipeline includes SQL pre-filter, three modulations + MMR, temp table write, and outer SQL compose. Measured at 128 dimensions (Nomic Embed v1.5, Matryoshka truncation) on Intel i9-13900KF. Pre-filtering narrows the candidate set before the matmul, so larger corpora scoped by a WHERE clause often run in single-digit milliseconds.

Memory: ~5MB per 10K vectors at 128d, ~30MB per 10K at 768d. Dimension-agnostic — works with any float32 BLOB.

---

## Schema

One table. An ID column and an embedding BLOB column.

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding BLOB,       -- float32 array, any dimension
    timestamp INTEGER     -- optional: epoch seconds, enables decay:N
);
```

---

## Install

```bash
pip install flexvec              # core: vec_ops + keyword (numpy only)
pip install flexvec[embed]       # + ONNX embedder (onnxruntime, tokenizers)
pip install flexvec[graph]       # + local_communities token (networkx)
pip install flexvec[embed,graph] # everything
```

---

flexvec powers retrieval in [flex](https://github.com/damiandelmas/flex), a local search and retrieval system for AI agents. Extracted as a standalone library — same code, zero dependencies on flex.

MIT · Python 3.10+ · SQLite · numpy
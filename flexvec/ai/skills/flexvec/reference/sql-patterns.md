# SQL Patterns

FlexVec queries are SQL-first. `vec_ops()` and `keyword()` materialize temporary result tables that can be joined to `_raw_chunks` or to source tables.

Semantic retrieval:

```sql
SELECT v.id, v.score, c.content
FROM vec_ops('similar:refund policy') v
JOIN _raw_chunks c ON c.id = v.id
ORDER BY v.score DESC
LIMIT 10;
```

Keyword retrieval:

```sql
SELECT k.id, k.rank, k.snippet, c.content
FROM keyword('refund') k
JOIN _raw_chunks c ON c.id = k.id
ORDER BY k.rank DESC
LIMIT 10;
```

Keyword-only CLI query without embedding/model setup:

```bash
flexvec sql app.db "SELECT k.id, k.rank, c.content FROM keyword('refund policy') k JOIN _raw_chunks c ON c.id = k.id ORDER BY k.rank DESC LIMIT 10" --no-embed --json
```

Hybrid retrieval:

```sql
SELECT k.id, k.rank, v.score, c.content
FROM keyword('refund') k
JOIN vec_ops('similar:policy exception') v ON v.id = k.id
JOIN _raw_chunks c ON c.id = k.id
ORDER BY v.score DESC
LIMIT 10;
```

Scoped retrieval:

```sql
SELECT v.id, v.score, c.content
FROM vec_ops(
  'similar:customer escalation',
  'SELECT id FROM _raw_chunks WHERE metadata LIKE ''%support%'''
) v
JOIN _raw_chunks c ON c.id = v.id
ORDER BY v.score DESC
LIMIT 10;
```

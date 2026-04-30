# Retrieval Contract

The retrieval contract tells FlexVec how to create a retrieval surface from one source table.

```json
{
  "table": "docs",
  "id_column": "id",
  "text_columns": ["title", "body"],
  "metadata_cols": ["created_at"],
  "chunk_table": "_raw_chunks",
  "fts_table": "chunks_fts",
  "embedding_col": "embedding"
}
```

Required fields:

- `table`: source table to index.
- `id_column`: stable row identifier. `id_col` is also accepted.
- `text_columns`: text columns to join, embed, and search. `text_col` or `text_cols` are also accepted.

Optional fields:

- `metadata_cols`: columns serialized into `_raw_chunks.metadata`.
- `chunk_table`: FlexVec-owned retrieval table, default `_raw_chunks`.
- `fts_table`: FlexVec-owned FTS5 table, default `chunks_fts`.
- `embedding_col`: BLOB column in `chunk_table`, default `embedding`.

`prepare` stores this contract in `_flexvec_meta`. Later `index`, `doctor`, and MCP invocations can use the stored contract.

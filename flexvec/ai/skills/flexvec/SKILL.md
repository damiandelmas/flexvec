# FlexVec

Use this skill when an agent needs to turn an existing SQLite database into a structured vector-retrieval database, query it with SQL, or expose it over MCP.

FlexVec is agent-native. Prefer deterministic JSON commands and explicit database paths. Do not assume a Flex registry, Flex cells, Flex modules, daemon state, or Labs packages are available.

## Workflow

1. Inspect the database:

```bash
flexvec inspect /path/to/app.db --json
```

2. Create a retrieval contract JSON. Pick one source table, one stable id column, one or more text/content columns, and any metadata columns the agent should preserve.

```json
{
  "table": "docs",
  "id_column": "id",
  "text_columns": ["title", "body"],
  "metadata_cols": ["created_at"]
}
```

3. Prepare the database. This creates FlexVec-owned `_flexvec_meta`, `_raw_chunks`, and `chunks_fts` surfaces inside the target DB.

```bash
flexvec prepare /path/to/app.db --spec spec.json --json
```

4. Index the database. This copies source rows into `_raw_chunks`, rebuilds FTS, and embeds rows when `flexvec[embed]` is installed.

```bash
flexvec index /path/to/app.db --spec spec.json --json
```

5. Verify readiness:

```bash
flexvec doctor /path/to/app.db --json
```

6. Query with SQL:

```bash
flexvec sql /path/to/app.db "SELECT v.id, v.score, c.content FROM vec_ops('similar:refund policy') v JOIN _raw_chunks c ON c.id = v.id ORDER BY v.score DESC LIMIT 10" --json
```

7. Serve over MCP:

```bash
flexvec mcp /path/to/app.db
```

## Rules

- Treat the SQLite DB path as explicit state. Do not rely on hidden project globals.
- Use `inspect` before creating a spec.
- Use `prepare` before `index`.
- Use `doctor` before handing the DB to another agent.
- Use SQL as the canonical query language; `vec_ops()` and `keyword()` are table sources.
- Do not import or depend on `flex.*` or Labs packages.

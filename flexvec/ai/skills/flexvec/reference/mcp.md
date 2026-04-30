# MCP

FlexVec MCP serves one SQLite database over stdio.

```bash
flexvec mcp /path/to/app.db
```

The MCP tool is `flexvec_search`. It accepts read-only SQL:

```json
{
  "query": "SELECT v.id, v.score, c.content FROM vec_ops('similar:refund policy') v JOIN _raw_chunks c ON c.id = v.id LIMIT 10"
}
```

For SQL/keyword-only operation without loading embeddings:

```bash
flexvec mcp /path/to/app.db --no-embed
```

FlexVec MCP does not use a Flex registry, Flex cells, Flex modules, services, or Labs packages. The database itself carries the retrieval contract in `_flexvec_meta`.

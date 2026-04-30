"""JSON-first FlexVec command line for AI agents."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources

from flexvec import mcp_server
from flexvec.spec import (
    RetrievalSpec,
    doctor_database,
    index_database,
    inspect_database,
    load_spec,
    prepare_database,
)


def _print(data) -> None:
    if isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data, indent=2, default=str))


def _load_spec_for_db(db_path: str, spec_path: str | None) -> RetrievalSpec:
    if spec_path:
        return load_spec(spec_path)
    with mcp_server.open_database(db_path) as db:
        return load_spec(None, db)


def _cmd_inspect(args) -> int:
    _print(inspect_database(args.db))
    return 0


def _cmd_prepare(args) -> int:
    spec = load_spec(args.spec)
    _print(prepare_database(args.db, spec))
    return 0


def _cmd_index(args) -> int:
    spec = _load_spec_for_db(args.db, args.spec)
    _print(
        index_database(
            args.db,
            spec,
            skip_embeddings=args.skip_embeddings,
            limit=args.limit,
        )
    )
    return 0


def _cmd_doctor(args) -> int:
    result = doctor_database(args.db, args.spec)
    _print(result)
    return 0 if result.get("ok") else 1


def _cmd_sql(args) -> int:
    with mcp_server.open_database(args.db) as db:
        try:
            if not args.no_embed:
                mcp_server.register_default_vec_ops(db)
        except ImportError as exc:
            _print(
                {
                    "error": str(exc),
                    "hint": "Install with: pip install 'flexvec[embed]' or use --no-embed for SQL/keyword-only queries.",
                }
            )
            return 1
        _print(mcp_server.execute_query(db, args.query))
    return 0


def _cmd_mcp(args) -> int:
    spec = None
    if args.spec:
        # Validate early so misconfigured MCP invocations fail before stdio starts.
        spec = _load_spec_for_db(args.db, args.spec)
    else:
        try:
            spec = _load_spec_for_db(args.db, None)
        except Exception:
            spec = None
    table = args.table or (spec.chunk_table if spec else "_raw_chunks")
    embedding_col = args.embedding_col or (spec.embedding_col if spec else "embedding")
    id_col = args.id_col or "id"
    mcp_server.main(
        [
            args.db,
            "--table",
            table,
            "--embedding-col",
            embedding_col,
            "--id-col",
            id_col,
            *(["--no-embed"] if args.no_embed else []),
        ]
    )
    return 0


def _cmd_skill_path(args) -> int:  # noqa: ARG001
    path = resources.files("flexvec.ai.skills.flexvec")
    _print({"skill": "flexvec", "path": str(path)})
    return 0


def _cmd_download_model(args) -> int:  # noqa: ARG001
    from flexvec.onnx.fetch import download_model

    path = download_model()
    _print({"model_dir": str(path)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flexvec",
        description="Agent-native SQLite vectorization and retrieval commands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect a SQLite DB and return table/column facts.")
    inspect_p.add_argument("db")
    inspect_p.add_argument("--json", action="store_true", help="Accepted for agent symmetry; output is always JSON.")
    inspect_p.set_defaults(func=_cmd_inspect)

    prepare_p = sub.add_parser("prepare", help="Create FlexVec retrieval tables from a JSON spec.")
    prepare_p.add_argument("db")
    prepare_p.add_argument("--spec", required=True, help="Path to retrieval contract JSON.")
    prepare_p.add_argument("--json", action="store_true", help="Accepted for agent symmetry; output is always JSON.")
    prepare_p.set_defaults(func=_cmd_prepare)

    index_p = sub.add_parser("index", help="Populate retrieval rows, FTS, and embeddings.")
    index_p.add_argument("db")
    index_p.add_argument("--spec", help="Path to retrieval contract JSON. Defaults to _flexvec_meta spec.")
    index_p.add_argument("--limit", type=int, help="Limit source rows during indexing.")
    index_p.add_argument("--skip-embeddings", action="store_true", help="Populate rows/FTS without embeddings.")
    index_p.add_argument("--json", action="store_true", help="Accepted for agent symmetry; output is always JSON.")
    index_p.set_defaults(func=_cmd_index)

    doctor_p = sub.add_parser("doctor", help="Report retrieval readiness and coverage.")
    doctor_p.add_argument("db")
    doctor_p.add_argument("--spec", help="Optional retrieval contract JSON.")
    doctor_p.add_argument("--json", action="store_true", help="Accepted for agent symmetry; output is always JSON.")
    doctor_p.set_defaults(func=_cmd_doctor)

    sql_p = sub.add_parser("sql", help="Run read-only SQL with keyword()/vec_ops() materializers.")
    sql_p.add_argument("db")
    sql_p.add_argument("query")
    sql_p.add_argument("--no-embed", action="store_true", help="Skip vec_ops registration for SQL/keyword-only queries.")
    sql_p.add_argument("--json", action="store_true", help="Accepted for agent symmetry; output is JSON.")
    sql_p.set_defaults(func=_cmd_sql)

    mcp_p = sub.add_parser("mcp", help="Serve one SQLite DB over MCP stdio.")
    mcp_p.add_argument("db")
    mcp_p.add_argument("--spec", help="Validate a retrieval contract before starting.")
    mcp_p.add_argument("--table")
    mcp_p.add_argument("--embedding-col")
    mcp_p.add_argument("--id-col")
    mcp_p.add_argument("--no-embed", action="store_true")
    mcp_p.set_defaults(func=_cmd_mcp)

    skill_p = sub.add_parser("skill-path", help="Print the packaged FlexVec skill path.")
    skill_p.add_argument("--json", action="store_true", help="Accepted for agent symmetry; output is always JSON.")
    skill_p.set_defaults(func=_cmd_skill_path)

    model_p = sub.add_parser("download-model", help="Download the optional ONNX embedding model.")
    model_p.add_argument("--json", action="store_true", help="Accepted for agent symmetry; output is JSON after download.")
    model_p.set_defaults(func=_cmd_download_model)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        _print({"error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

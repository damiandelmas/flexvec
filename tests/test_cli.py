"""Tests for the agent-native FlexVec CLI and retrieval contract."""

from __future__ import annotations

import json
import sqlite3
from importlib import resources
from pathlib import Path

from flexvec import cli
from flexvec.spec import RetrievalSpec, doctor_database, index_database, inspect_database, prepare_database


def _db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE docs (id TEXT PRIMARY KEY, title TEXT, body TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO docs VALUES ('a', 'Refunds', 'refund policy exception for annual plans', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO docs VALUES ('b', 'Security', 'authentication and audit log controls', '2026-01-02')"
    )
    conn.commit()
    conn.close()


def _spec() -> RetrievalSpec:
    return RetrievalSpec(
        table="docs",
        id_col="id",
        text_col="body",
        metadata_cols=["title", "created_at"],
    )


def _write_spec(path: Path) -> Path:
    spec_path = path / "spec.json"
    spec_path.write_text(_spec().to_json(), encoding="utf-8")
    return spec_path


def _write_alias_spec(path: Path) -> Path:
    spec_path = path / "alias-spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "table": "docs",
                "id_column": "id",
                "text_columns": ["title", "body"],
                "metadata_cols": ["created_at"],
            }
        ),
        encoding="utf-8",
    )
    return spec_path


def test_inspect_prepare_index_and_doctor(tmp_path: Path):
    db_path = tmp_path / "app.db"
    _db(db_path)

    inspected = inspect_database(db_path)
    docs = [t for t in inspected["tables"] if t["name"] == "docs"][0]
    assert "id" in docs["id_candidates"]
    assert "body" in docs["text_candidates"]

    prepared = prepare_database(db_path, _spec())
    assert prepared["prepared"] is True

    indexed = index_database(
        db_path,
        _spec(),
        embed_fn=lambda text: [float(len(text)), 1.0, 0.5],
    )
    assert indexed["indexed"] == 2
    assert indexed["embedded"] == 2

    doctor = doctor_database(db_path)
    assert doctor["ok"] is True
    assert doctor["chunk_rows"] == 2
    assert doctor["embedded_rows"] == 2
    assert doctor["fts_rows"] == 2


def test_cli_json_commands_and_keyword_sql(tmp_path: Path, capsys):
    db_path = tmp_path / "app.db"
    _db(db_path)
    spec_path = _write_spec(tmp_path)

    assert cli.main(["prepare", str(db_path), "--spec", str(spec_path), "--json"]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["chunk_table"] == "_raw_chunks"

    assert cli.main(
        ["index", str(db_path), "--spec", str(spec_path), "--skip-embeddings", "--json"]
    ) == 0
    indexed = json.loads(capsys.readouterr().out)
    assert indexed["indexed"] == 2
    assert indexed["skip_embeddings"] is True

    assert cli.main(["doctor", str(db_path), "--json"]) == 0
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["ok"] is True

    query = "SELECT k.id FROM keyword('refund') k ORDER BY k.rank DESC"
    assert cli.main(["sql", str(db_path), query, "--no-embed", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["id"] == "a"


def test_cli_accepts_agent_alias_spec_with_multiple_text_columns(tmp_path: Path, capsys):
    db_path = tmp_path / "app.db"
    _db(db_path)
    spec_path = _write_alias_spec(tmp_path)

    assert cli.main(["prepare", str(db_path), "--spec", str(spec_path), "--json"]) == 0
    capsys.readouterr()
    assert cli.main(
        ["index", str(db_path), "--spec", str(spec_path), "--skip-embeddings", "--json"]
    ) == 0
    capsys.readouterr()

    query = (
        "SELECT k.id, c.content FROM keyword('annual') k "
        "JOIN _raw_chunks c ON c.id = k.id"
    )
    assert cli.main(["sql", str(db_path), query, "--no-embed", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["id"] == "a"
    assert "Refunds" in rows[0]["content"]
    assert "annual plans" in rows[0]["content"]


def test_skill_path_points_to_packaged_skill(capsys):
    assert cli.main(["skill-path", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    skill_path = Path(result["path"])
    assert skill_path.name == "flexvec"
    assert (skill_path / "SKILL.md").exists()
    assert resources.files("flexvec.ai.skills.flexvec").joinpath("SKILL.md").is_file()


def test_mcp_command_uses_stored_spec_defaults(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "app.db"
    _db(db_path)
    spec = RetrievalSpec(
        table="docs",
        id_col="id",
        text_col="body",
        chunk_table="_agent_chunks",
        embedding_col="vec",
    )
    prepare_database(db_path, spec)
    calls = []
    monkeypatch.setattr(cli.mcp_server, "main", lambda argv: calls.append(argv))

    assert cli.main(["mcp", str(db_path), "--no-embed"]) == 0

    assert calls == [
        [
            str(db_path),
            "--table",
            "_agent_chunks",
            "--embedding-col",
            "vec",
            "--id-col",
            "id",
            "--no-embed",
        ]
    ]


def test_agent_native_files_do_not_import_flex_core():
    root = Path(__file__).resolve().parents[1] / "flexvec"
    for rel in ("cli.py", "spec.py", "mcp_core.py", "mcp_server.py"):
        source = (root / rel).read_text(encoding="utf-8")
        assert "from flex." not in source
        assert "import flex." not in source

"""Regression tests for the rebuildable Relay telemetry projection."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from hermes_cli.telemetry import TelemetryProjection, summarize_connection


def _event(
    *,
    uuid: str,
    parent: str | None,
    name: str,
    category: str,
    phase: str,
    data=None,
    metadata=None,
    profile=None,
):
    return {
        "atof_version": "0.1",
        "kind": "scope",
        "uuid": uuid,
        "parent_uuid": parent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "category": category,
        "scope_category": phase,
        "data": data or {},
        "metadata": metadata or {},
        "category_profile": profile or {},
    }


def test_real_relay_children_correlate_by_parent_uuid_without_run_id(tmp_path):
    db_path = tmp_path / "state.db"
    projection = TelemetryProjection(db_path)
    run_uuid = "00000000-0000-7000-8000-000000000001"
    llm_uuid = "00000000-0000-7000-8000-000000000002"
    tool_uuid = "00000000-0000-7000-8000-000000000003"

    # Deliver children before their parent to exercise backfill. Real Relay
    # children carry parent_uuid and turn metadata, not Hermes run_id.
    projection.consume(
        _event(
            uuid=llm_uuid,
            parent=run_uuid,
            name="openai",
            category="llm",
            phase="start",
            metadata={"turn_id": "turn-1"},
            profile={"model_name": "demo"},
        )
    )
    projection.consume(
        _event(
            uuid=llm_uuid,
            parent=run_uuid,
            name="openai",
            category="llm",
            phase="end",
            data={
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_tokens": 2,
                }
            },
        )
    )
    projection.consume(
        _event(
            uuid=tool_uuid,
            parent=run_uuid,
            name="read_file",
            category="tool",
            phase="start",
            metadata={"turn_id": "turn-1"},
        )
    )
    projection.consume(
        _event(
            uuid=tool_uuid,
            parent=run_uuid,
            name="read_file",
            category="tool",
            phase="end",
            data={
                "status": "error",
                "error_type": "tool_error",
                "result_class": "object",
            },
        )
    )
    projection.consume(
        _event(
            uuid=run_uuid,
            parent=None,
            name="hermes.run:run-1",
            category="agent",
            phase="start",
            data={"session_id": "s1", "entrypoint": "cli"},
            metadata={"provider": "openai", "model": "demo"},
        )
    )
    projection.consume(
        _event(
            uuid=run_uuid,
            parent=None,
            name="hermes.run:run-1",
            category="agent",
            phase="end",
            data={"completed": True, "outcome": "completed"},
        )
    )
    projection.close()

    with sqlite3.connect(db_path) as conn:
        summary = summarize_connection(conn)
    assert summary["run_count"] == 1
    assert summary["model_calls"] == 1
    assert summary["tool_calls"] == 1
    assert summary["tool_failures"] == 1
    assert summary["cache_hit_rate"] == 1
    assert summary["entrypoints"] == {"cli": 1}


def test_projection_schema_is_replaced_when_version_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE tel_projection_state(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO tel_projection_state VALUES('schema_version','1')")
        conn.execute("CREATE TABLE tel_runs(id TEXT PRIMARY KEY, legacy TEXT)")

    TelemetryProjection(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(tel_runs)")}
        version = conn.execute(
            "SELECT value FROM tel_projection_state WHERE key='schema_version'"
        ).fetchone()[0]
    assert "legacy" not in columns
    assert "entrypoint" in columns
    assert version == "3"


def test_late_intermediate_scope_backfills_existing_descendants(tmp_path):
    db_path = tmp_path / "state.db"
    projection = TelemetryProjection(db_path)
    run_uuid = "00000000-0000-7000-8000-000000000011"
    intermediate_uuid = "00000000-0000-7000-8000-000000000012"
    llm_uuid = "00000000-0000-7000-8000-000000000013"

    projection.consume(
        _event(
            uuid=llm_uuid,
            parent=intermediate_uuid,
            name="openai",
            category="llm",
            phase="end",
            data={"usage": {"input_tokens": 4}},
        )
    )
    projection.consume(
        _event(
            uuid=run_uuid,
            parent=None,
            name="hermes.run:nested",
            category="agent",
            phase="start",
            data={"session_id": "s-nested"},
        )
    )
    projection.consume(
        _event(
            uuid=intermediate_uuid,
            parent=run_uuid,
            name="delegated-agent",
            category="agent",
            phase="start",
        )
    )
    projection.close()

    with sqlite3.connect(db_path) as conn:
        run_id = conn.execute(
            "SELECT run_id FROM tel_model_calls WHERE id=?", (llm_uuid,)
        ).fetchone()[0]
    assert run_id == "nested"

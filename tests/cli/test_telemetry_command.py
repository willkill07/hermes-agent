"""Tests for the NeMo Relay telemetry projection and CLI helpers."""

from __future__ import annotations

import json
import sqlite3
from argparse import Namespace

from hermes_cli.telemetry import (
    TelemetryProjection,
    summarize_connection,
    telemetry_command,
    telemetry_status,
)


def _events():
    base = {"kind": "scope", "metadata": {"run_id": "run-1", "platform": "cli"}}
    return [
        {
            **base,
            "uuid": "run-uuid",
            "name": "hermes.run:run-1",
            "category": "agent",
            "scope_category": "start",
            "timestamp": "2026-01-01T00:00:00Z",
            "data": {"session_id": "session-1"},
        },
        {
            "kind": "scope",
            "uuid": "llm-uuid",
            "parent_uuid": "run-uuid",
            "name": "chat",
            "category": "llm",
            "scope_category": "start",
            "timestamp": "2026-01-01T00:00:01Z",
            "metadata": {"turn_id": "run-1"},
            "category_profile": {"model_name": "test-model"},
            "data": {},
        },
        {
            "kind": "scope",
            "uuid": "llm-uuid",
            "parent_uuid": "run-uuid",
            "name": "chat",
            "category": "llm",
            "scope_category": "end",
            "timestamp": "2026-01-01T00:00:03Z",
            "metadata": {"turn_id": "run-1"},
            "category_profile": {"model_name": "test-model"},
            "data": {
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "finish_reason": "stop",
            },
        },
        {
            "kind": "scope",
            "uuid": "tool-uuid",
            "parent_uuid": "run-uuid",
            "name": "search",
            "category": "tool",
            "scope_category": "start",
            "timestamp": "2026-01-01T00:00:04Z",
            "metadata": {"turn_id": "run-1"},
            "data": {},
        },
        {
            "kind": "scope",
            "uuid": "tool-uuid",
            "parent_uuid": "run-uuid",
            "name": "search",
            "category": "tool",
            "scope_category": "end",
            "timestamp": "2026-01-01T00:00:05Z",
            "metadata": {"turn_id": "run-1"},
            "data": {"result_class": "object"},
        },
        {
            **base,
            "uuid": "run-uuid",
            "name": "hermes.run:run-1",
            "category": "agent",
            "scope_category": "end",
            "timestamp": "2026-01-01T00:00:06Z",
            "data": {
                "outcome": "completed",
                "completed": True,
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_tokens": 2,
                "estimated_cost_usd": 0.01,
            },
        },
    ]


def test_projection_is_idempotent_and_summarizes(tmp_path):
    projection = TelemetryProjection(tmp_path / "state.db")
    for event in _events() + _events():
        projection.consume(event)

    assert projection.counts() == {
        "runs": 1,
        "model_calls": 1,
        "tool_calls": 1,
        "errors": 0,
    }
    with sqlite3.connect(projection.db_path) as conn:
        summary = summarize_connection(conn, limit=1)
    assert summary["run_count"] == 1
    assert summary["completion_rate"] == 1
    assert summary["run_latency_p50"] == 6
    assert summary["model_latency_p50"] == 2
    assert summary["input_tokens"] == 10
    assert summary["recent_runs"][0]["run_id"] == "run-1"
    with sqlite3.connect(projection.db_path) as conn:
        assert (
            conn.execute("SELECT run_id FROM tel_model_calls").fetchone()[0] == "run-1"
        )
        assert (
            conn.execute("SELECT run_id FROM tel_tool_calls").fetchone()[0] == "run-1"
        )


def test_rebuild_reads_atof_and_ignores_bad_lines(tmp_path):
    atof = tmp_path / "atof"
    atof.mkdir()
    path = atof / "events.jsonl"
    path.write_text(
        "\n".join([json.dumps(event) for event in _events()] + ["not-json"]),
        encoding="utf-8",
    )
    projection = TelemetryProjection(tmp_path / "state.db")
    result = projection.rebuild(atof)
    assert result["events"] == 6
    assert result["runs"] == 1


def test_export_outputs_canonical_atof(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home"
    atof = home / "telemetry" / "atof"
    atof.mkdir(parents=True)
    (atof / "events.jsonl").write_text(
        json.dumps(_events()[0]) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    args = Namespace(
        telemetry_action="export",
        out="-",
        since=None,
        format="ndjson",
        include_content=False,
        otlp=False,
        openinference=False,
    )
    assert telemetry_command(args) == 0
    assert json.loads(capsys.readouterr().out)["uuid"] == "run-uuid"


def test_status_does_not_create_state_database(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    result = telemetry_status()
    assert result["projection"]["runs"] == 0
    assert not (home / "state.db").exists()


def test_export_rejects_invalid_since(capsys):
    args = Namespace(
        telemetry_action="export",
        out="-",
        since="not-a-timestamp",
        format="ndjson",
        include_content=False,
        otlp=False,
        openinference=False,
    )
    assert telemetry_command(args) == 2
    assert "ISO-8601" in capsys.readouterr().err

"""Governance and privacy tests for local Relay telemetry."""

from __future__ import annotations

import json
import stat
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli.config import load_config, save_config, validate_config_structure
from hermes_cli.telemetry import (
    _structural_event,
    aggregate_bundle,
    compact_legacy_atof,
    enforce_retention,
    install_id,
)


def _configure(monkeypatch, tmp_path, **telemetry):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    config["telemetry"].update(telemetry)
    save_config(config)


def test_structural_export_reconstructs_allowlisted_payload():
    event = {
        "atof_version": "0.1",
        "kind": "mark",
        "uuid": "u",
        "timestamp": "now",
        "name": "hermes.subagent.start",
        "data": {
            "run_id": "r",
            "child_goal": "email alice@example.com",
            "prompt": "secret",
        },
        "metadata": {
            "platform": "cli",
            "authorization": "Bearer secret",
            "path": "/private",
        },
        "category_profile": {
            "model_name": "safe-model",
            "annotated_request": {"messages": ["alice@example.com"]},
            "annotated_response": {"content": "Bearer secret"},
        },
    }
    safe = _structural_event(event)
    encoded = json.dumps(safe)
    assert safe["data"] == {"run_id": "r"}
    assert safe["metadata"] == {"platform": "cli"}
    assert safe["category_profile"] == {"model_name": "safe-model"}
    assert "alice@example.com" not in encoded
    assert "Bearer secret" not in encoded


def test_aggregate_export_requires_both_consent_gates(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, consent_state="unknown", allow_aggregate=False)
    with pytest.raises(PermissionError):
        aggregate_bundle()
    _configure(monkeypatch, tmp_path, consent_state="granted", allow_aggregate=False)
    with pytest.raises(PermissionError):
        aggregate_bundle()
    _configure(monkeypatch, tmp_path, consent_state="granted", allow_aggregate=True)
    bundle = aggregate_bundle()
    assert bundle["schema"] == "hermes.telemetry.aggregate/v1"
    assert "recent_runs" not in bundle["metrics"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("content_redaction", "none"),
        ("consent_state", "maybe"),
        ("retention_days", 0),
    ],
)
def test_governance_config_values_are_validated(field, value):
    issues = validate_config_structure({"telemetry": {field: value}})
    assert any(f"telemetry.{field}" in issue.message for issue in issues)


def test_generated_install_id_is_stable_and_private(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, install_id="")
    first = install_id()
    second = install_id()
    path = tmp_path / "telemetry" / "install_id"
    assert first == second
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_retention_deletes_only_expired_owned_files(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, retention_days=2)
    directory = tmp_path / "telemetry" / "atof"
    directory.mkdir(parents=True)
    expired = (datetime.now(timezone.utc) - timedelta(days=5)).date()
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    expired_path = directory / f"hermes-atof-{expired}.jsonl"
    recent_path = directory / f"hermes-atof-{recent}.jsonl"
    unrelated = directory / "other.jsonl"
    for path in (expired_path, recent_path, unrelated):
        path.write_text("{}\n", encoding="utf-8")

    result = enforce_retention()

    assert result["atof"] == 1
    assert not expired_path.exists()
    assert recent_path.exists()
    assert unrelated.exists()


def test_legacy_atof_compaction_resumes_from_in_progress_file(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    directory = tmp_path / "telemetry" / "atof"
    directory.mkdir(parents=True)
    timestamp = datetime(2026, 6, 30, 12, tzinfo=timezone.utc)
    in_progress = directory / "hermes-atof.jsonl.compacting"
    in_progress.write_text(
        json.dumps({
            "atof_version": "0.1",
            "uuid": "u",
            "timestamp": timestamp.isoformat(),
        })
        + "\n",
        encoding="utf-8",
    )

    assert compact_legacy_atof() == 1
    assert not in_progress.exists()
    rotated = directory / "hermes-atof-2026-06-30.jsonl"
    assert json.loads(rotated.read_text(encoding="utf-8"))["uuid"] == "u"

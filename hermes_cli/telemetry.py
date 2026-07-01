"""User-facing telemetry projection and export helpers.

NeMo Relay ATOF files are the source of truth.  The SQLite tables in this
module are a disposable read model used by the CLI and Insights UX.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tel_runs (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    parent_id TEXT,
    session_id TEXT,
    turn_id TEXT,
    platform TEXT,
    started_at REAL,
    ended_at REAL,
    outcome TEXT,
    completed INTEGER,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    error_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_tel_runs_started_at ON tel_runs(started_at);
CREATE TABLE IF NOT EXISTS tel_model_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    parent_id TEXT,
    model TEXT,
    started_at REAL,
    ended_at REAL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    finish_reason TEXT,
    error_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_tel_model_run ON tel_model_calls(run_id);
CREATE TABLE IF NOT EXISTS tel_tool_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    parent_id TEXT,
    tool_name TEXT,
    started_at REAL,
    ended_at REAL,
    result_class TEXT,
    error_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_tel_tool_run ON tel_tool_calls(run_id);
CREATE TABLE IF NOT EXISTS tel_error_events (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    parent_id TEXT,
    event_name TEXT,
    error_type TEXT,
    occurred_at REAL
);
CREATE TABLE IF NOT EXISTS tel_projection_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any, default: float = 0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _event_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value if isinstance(value, dict) else {}
    return {}


def _error_type(
    event: dict[str, Any], data: dict[str, Any], metadata: dict[str, Any]
) -> str | None:
    value = data.get("error_type") or metadata.get("error_type")
    if value:
        return str(value)
    error = event.get("error")
    if error:
        return type(error).__name__ if not isinstance(error, str) else error
    return None


def telemetry_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        return _dict(load_config().get("telemetry"))
    except Exception:
        return {}


def atof_directory(config: dict[str, Any] | None = None) -> Path:
    config = config if config is not None else telemetry_config()
    configured = _dict(config.get("atof")).get("output_directory")
    return (
        Path(configured).expanduser()
        if configured
        else get_hermes_home() / "telemetry" / "atof"
    )


def atif_directory(config: dict[str, Any] | None = None) -> Path:
    config = config if config is not None else telemetry_config()
    configured = _dict(config.get("atif")).get("output_directory")
    return (
        Path(configured).expanduser()
        if configured
        else get_hermes_home() / "telemetry" / "atif"
    )


def _json_files(directory: Path) -> list[Path]:
    return (
        sorted(path for path in directory.glob("*.json*") if path.is_file())
        if directory.exists()
        else []
    )


def iter_atof_events(directory: Path | None = None) -> Iterable[dict[str, Any]]:
    for path in _json_files(directory or atof_directory()):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(value, dict):
                    yield value


class TelemetryProjection:
    """Idempotent SQLite projection of structural Relay events."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else get_hermes_home() / "state.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def consume(self, event: Any) -> None:
        value = _event_dict(event)
        if not value.get("uuid"):
            return
        with self._connect() as conn:
            self._apply(conn, value)

    def rebuild(self, directory: Path | None = None) -> dict[str, int]:
        event_count = 0
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for table in (
                    "tel_runs",
                    "tel_model_calls",
                    "tel_tool_calls",
                    "tel_error_events",
                    "tel_projection_state",
                ):
                    conn.execute(f"DELETE FROM {table}")
                for event in iter_atof_events(directory):
                    self._apply(conn, event)
                    event_count += 1
                conn.execute(
                    "INSERT INTO tel_projection_state(key, value) VALUES('last_rebuild_at', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (datetime.now(timezone.utc).isoformat(),),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        counts = self.counts()
        counts["events"] = event_count
        return counts

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                "runs": conn.execute("SELECT COUNT(*) FROM tel_runs").fetchone()[0],
                "model_calls": conn.execute(
                    "SELECT COUNT(*) FROM tel_model_calls"
                ).fetchone()[0],
                "tool_calls": conn.execute(
                    "SELECT COUNT(*) FROM tel_tool_calls"
                ).fetchone()[0],
                "errors": conn.execute(
                    "SELECT COUNT(*) FROM tel_error_events"
                ).fetchone()[0],
            }

    @staticmethod
    def _apply(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
        event_id = str(event["uuid"])
        parent_id = str(event.get("parent_uuid") or "") or None
        category = str(event.get("category") or "").lower()
        scope_category = str(event.get("scope_category") or "").lower()
        name = str(event.get("name") or "")
        timestamp = _epoch(event.get("timestamp"))
        metadata = _dict(event.get("metadata"))
        data = _dict(event.get("data"))
        profile = _dict(event.get("category_profile"))
        run_id = str(metadata.get("run_id") or data.get("run_id") or "") or None
        error_type = _error_type(event, data, metadata)

        if category == "agent" and (run_id or name.startswith("hermes.run:")):
            run_id = run_id or name.removeprefix("hermes.run:")
            if scope_category == "start":
                conn.execute(
                    """INSERT INTO tel_runs(id,run_id,parent_id,session_id,turn_id,platform,started_at)
                       VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                       run_id=COALESCE(excluded.run_id,run_id),
                       parent_id=COALESCE(excluded.parent_id,parent_id),
                       session_id=COALESCE(excluded.session_id,session_id),
                       turn_id=COALESCE(excluded.turn_id,turn_id),
                       platform=COALESCE(excluded.platform,platform),
                       started_at=COALESCE(excluded.started_at,started_at)""",
                    (
                        event_id,
                        run_id,
                        parent_id,
                        data.get("session_id") or metadata.get("session_id"),
                        metadata.get("turn_id"),
                        metadata.get("platform")
                        or metadata.get("source")
                        or data.get("entrypoint"),
                        timestamp,
                    ),
                )
            elif scope_category == "end":
                outcome = data.get("outcome") or (
                    "completed" if data.get("completed") else None
                )
                conn.execute(
                    """INSERT INTO tel_runs(id,run_id,parent_id,ended_at,outcome,completed,input_tokens,output_tokens,
                       cache_read_tokens,cache_write_tokens,estimated_cost_usd,error_type)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                       run_id=COALESCE(excluded.run_id,run_id),
                       ended_at=COALESCE(excluded.ended_at,ended_at), outcome=COALESCE(excluded.outcome,outcome),
                       completed=COALESCE(excluded.completed,completed), input_tokens=excluded.input_tokens,
                       output_tokens=excluded.output_tokens, cache_read_tokens=excluded.cache_read_tokens,
                       cache_write_tokens=excluded.cache_write_tokens, estimated_cost_usd=excluded.estimated_cost_usd,
                       error_type=COALESCE(excluded.error_type,error_type)""",
                    (
                        event_id,
                        run_id,
                        parent_id,
                        timestamp,
                        outcome,
                        int(bool(data.get("completed"))),
                        int(_number(data.get("input_tokens"))),
                        int(_number(data.get("output_tokens"))),
                        int(_number(data.get("cache_read_tokens"))),
                        int(_number(data.get("cache_write_tokens"))),
                        _number(data.get("estimated_cost_usd")),
                        error_type,
                    ),
                )
            return

        if category in {"llm", "tool"}:
            if scope_category == "start":
                if category == "llm":
                    conn.execute(
                        "INSERT INTO tel_model_calls(id,run_id,parent_id,model,started_at) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET run_id=COALESCE(excluded.run_id,run_id), parent_id=COALESCE(excluded.parent_id,parent_id), started_at=COALESCE(excluded.started_at,started_at), model=COALESCE(excluded.model,model)",
                        (
                            event_id,
                            run_id,
                            parent_id,
                            profile.get("model_name") or data.get("model"),
                            timestamp,
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO tel_tool_calls(id,run_id,parent_id,tool_name,started_at) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET run_id=COALESCE(excluded.run_id,run_id), parent_id=COALESCE(excluded.parent_id,parent_id), started_at=COALESCE(excluded.started_at,started_at), tool_name=COALESCE(excluded.tool_name,tool_name)",
                        (
                            event_id,
                            run_id,
                            parent_id,
                            profile.get("tool_name") or data.get("tool_name") or name,
                            timestamp,
                        ),
                    )
            elif scope_category == "end":
                if category == "llm":
                    usage = _dict(data.get("usage"))
                    conn.execute(
                        "INSERT INTO tel_model_calls(id,run_id,parent_id,model,ended_at,input_tokens,output_tokens,cache_read_tokens,finish_reason,error_type) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET run_id=COALESCE(excluded.run_id,run_id), parent_id=COALESCE(excluded.parent_id,parent_id), ended_at=COALESCE(excluded.ended_at,ended_at), input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens, cache_read_tokens=excluded.cache_read_tokens, finish_reason=COALESCE(excluded.finish_reason,finish_reason), error_type=COALESCE(excluded.error_type,error_type)",
                        (
                            event_id,
                            run_id,
                            parent_id,
                            profile.get("model_name") or data.get("model"),
                            timestamp,
                            int(
                                _number(
                                    usage.get("input_tokens")
                                    or usage.get("prompt_tokens")
                                )
                            ),
                            int(
                                _number(
                                    usage.get("output_tokens")
                                    or usage.get("completion_tokens")
                                )
                            ),
                            int(_number(usage.get("cache_read_tokens"))),
                            data.get("finish_reason"),
                            error_type,
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO tel_tool_calls(id,run_id,parent_id,tool_name,ended_at,result_class,error_type) VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET run_id=COALESCE(excluded.run_id,run_id), parent_id=COALESCE(excluded.parent_id,parent_id), ended_at=COALESCE(excluded.ended_at,ended_at), result_class=COALESCE(excluded.result_class,result_class), error_type=COALESCE(excluded.error_type,error_type)",
                        (
                            event_id,
                            run_id,
                            parent_id,
                            profile.get("tool_name") or data.get("tool_name") or name,
                            timestamp,
                            data.get("result_class"),
                            error_type,
                        ),
                    )
            return

        if "error" in name.lower() or error_type:
            conn.execute(
                "INSERT OR REPLACE INTO tel_error_events(id,run_id,parent_id,event_name,error_type,occurred_at) VALUES(?,?,?,?,?,?)",
                (
                    event_id,
                    run_id,
                    parent_id,
                    name,
                    error_type or "error",
                    timestamp,
                ),
            )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[
        min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    ]


def summarize_connection(
    conn: sqlite3.Connection,
    *,
    cutoff: float = 0,
    source: str | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    try:
        clauses = ["started_at >= ?"]
        params: list[Any] = [cutoff]
        if source:
            clauses.append("platform = ?")
            params.append(source)
        runs = conn.execute(
            f"SELECT started_at,ended_at,outcome,completed,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,estimated_cost_usd,error_type FROM tel_runs WHERE {' AND '.join(clauses)}",
            params,
        ).fetchall()
        child_clause = "r.started_at >= ?"
        child_params: list[Any] = [cutoff]
        if source:
            child_clause += " AND r.platform = ?"
            child_params.append(source)
        model = conn.execute(
            "SELECT c.started_at,c.ended_at FROM tel_model_calls c "
            "JOIN tel_runs r ON r.run_id=c.run_id WHERE " + child_clause,
            child_params,
        ).fetchall()
        tools = conn.execute(
            "SELECT c.started_at,c.ended_at,c.error_type FROM tel_tool_calls c "
            "JOIN tel_runs r ON r.run_id=c.run_id WHERE " + child_clause,
            child_params,
        ).fetchall()
        recent_runs = []
        if limit > 0:
            recent_runs = conn.execute(
                f"SELECT run_id,session_id,platform,started_at,ended_at,outcome,error_type "
                f"FROM tel_runs WHERE {' AND '.join(clauses)} ORDER BY started_at DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return {"run_count": 0}
        raise
    run_durations = [
        end - start for start, end, *_ in runs if start is not None and end is not None
    ]
    model_durations = [
        end - start for start, end in model if start is not None and end is not None
    ]
    tool_durations = [
        end - start for start, end, _ in tools if start is not None and end is not None
    ]
    completed = sum(bool(row[3]) or row[2] in {"completed", "success"} for row in runs)
    failed = sum(bool(row[9]) or row[2] in {"failed", "error"} for row in runs)
    interrupted = sum(row[2] in {"interrupted", "cancelled", "timeout"} for row in runs)
    return {
        "run_count": len(runs),
        "completed": completed,
        "failed": failed,
        "interrupted": interrupted,
        "completion_rate": completed / len(runs) if runs else 0,
        "failure_rate": failed / len(runs) if runs else 0,
        "run_latency_p50": _percentile(run_durations, 0.5),
        "run_latency_p95": _percentile(run_durations, 0.95),
        "model_latency_p50": _percentile(model_durations, 0.5),
        "model_latency_p95": _percentile(model_durations, 0.95),
        "tool_latency_p50": _percentile(tool_durations, 0.5),
        "tool_latency_p95": _percentile(tool_durations, 0.95),
        "model_calls": len(model),
        "tool_calls": len(tools),
        "tool_failures": sum(bool(row[2]) for row in tools),
        "input_tokens": sum(row[4] or 0 for row in runs),
        "output_tokens": sum(row[5] or 0 for row in runs),
        "cache_read_tokens": sum(row[6] or 0 for row in runs),
        "cache_write_tokens": sum(row[7] or 0 for row in runs),
        "estimated_cost_usd": sum(row[8] or 0 for row in runs),
        "recent_runs": [
            {
                "run_id": row[0],
                "session_id": row[1],
                "platform": row[2],
                "started_at": row[3],
                "duration_seconds": row[4] - row[3]
                if row[3] is not None and row[4] is not None
                else None,
                "outcome": row[5],
                "error_type": row[6],
            }
            for row in recent_runs
        ],
    }


def telemetry_status() -> dict[str, Any]:
    config = telemetry_config()
    enabled_plugins: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        root = load_config()
        enabled_plugins = _dict(root.get("plugins"))
    except Exception:
        pass
    atof_files = _json_files(atof_directory(config))
    atif_files = _json_files(atif_directory(config))
    projection = {"runs": 0, "model_calls": 0, "tool_calls": 0, "errors": 0}
    projection_error = None
    db_path = get_hermes_home() / "state.db"
    if db_path.exists():
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                projection = {
                    "runs": conn.execute("SELECT COUNT(*) FROM tel_runs").fetchone()[0],
                    "model_calls": conn.execute(
                        "SELECT COUNT(*) FROM tel_model_calls"
                    ).fetchone()[0],
                    "tool_calls": conn.execute(
                        "SELECT COUNT(*) FROM tel_tool_calls"
                    ).fetchone()[0],
                    "errors": conn.execute(
                        "SELECT COUNT(*) FROM tel_error_events"
                    ).fetchone()[0],
                }
        except sqlite3.OperationalError as exc:
            # An existing pre-telemetry state DB simply has no projection yet.
            if "no such table" not in str(exc).lower():
                projection_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            projection_error = f"{type(exc).__name__}: {exc}"
    names = set(enabled_plugins.get("enabled") or [])
    plugin_enabled = bool({"nemo_relay", "observability/nemo_relay"} & names)
    try:
        import nemo_relay

        relay_dependency = {
            "available": True,
            "version": getattr(nemo_relay, "__version__", "unknown"),
        }
    except Exception as exc:
        relay_dependency = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    trajectories = _dict(config.get("trajectories"))
    return {
        "local_enabled": bool(config.get("local", True)),
        "plugin_enabled": plugin_enabled,
        "relay_dependency": relay_dependency,
        "consent_state": config.get("consent_state", "unknown"),
        "content_capture": bool(
            trajectories.get("enabled", False) or config.get("capture_content", False)
        ),
        "redaction": config.get("content_redaction", "secrets"),
        "atof_directory": str(atof_directory(config)),
        "atof_files": len(atof_files),
        "atof_bytes": sum(path.stat().st_size for path in atof_files),
        "atif_files": len(atif_files),
        "projection": projection,
        "projection_error": projection_error,
        "otlp": _dict(_dict(config.get("export")).get("otlp")),
        "openinference": _dict(_dict(config.get("export")).get("openinference")),
    }


def telemetry_command(args: Any) -> int:
    action = getattr(args, "telemetry_action", None) or "status"
    if action == "status":
        result = telemetry_status()
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            state = (
                "enabled"
                if result["plugin_enabled"] and result["local_enabled"]
                else "not collecting"
            )
            print(f"Telemetry: {state}")
            print(
                f"Relay plugin: {'enabled' if result['plugin_enabled'] else 'disabled'}"
            )
            print(f"Relay dependency: {result['relay_dependency']}")
            print(
                f"ATOF: {result['atof_files']} file(s), {result['atof_bytes']} bytes at {result['atof_directory']}"
            )
            print(f"Projection: {result['projection'] or result['projection_error']}")
            print(
                f"Content: {'enabled' if result['content_capture'] else 'structural only'}; redaction={result['redaction']}"
            )
            if not result["plugin_enabled"]:
                print(
                    "Enable collection with: hermes plugins enable observability/nemo_relay"
                )
        return 0
    if action == "rebuild":
        result = TelemetryProjection().rebuild()
        print(
            json.dumps(result, indent=2, sort_keys=True)
            if getattr(args, "json", False)
            else f"Rebuilt telemetry projection: {result}"
        )
        return 0
    if action == "preview":
        projection = TelemetryProjection()
        with projection._connect() as conn:
            result = summarize_connection(
                conn,
                cutoff=datetime.now().timestamp() - args.days * 86400,
                limit=max(0, args.limit),
            )
        print(
            json.dumps(result, indent=2, sort_keys=True)
            if args.json
            else "\n".join(f"{key}: {value}" for key, value in result.items())
        )
        return 0
    if action == "export":
        if args.otlp or args.openinference:
            print(
                "Historical exporter replay is not available; enable the corresponding Relay live exporter in config.yaml.",
                file=sys.stderr,
            )
            return 2
        since = _epoch(args.since) if args.since else None
        if args.since and since is None:
            print("--since must be an ISO-8601 timestamp", file=sys.stderr)
            return 2
        records = [
            event
            for event in iter_atof_events()
            if since is None or (_epoch(event.get("timestamp")) or 0) >= since
        ]
        if args.include_content:
            config = telemetry_config()
            if not (
                _dict(config.get("trajectories")).get("enabled", False)
                or config.get("capture_content", False)
            ):
                print(
                    "--include-content requires telemetry.trajectories.enabled: true",
                    file=sys.stderr,
                )
                return 2
            for path in _json_files(atif_directory(config)):
                try:
                    records.append({
                        "_kind": "atif",
                        "trajectory": json.loads(path.read_text(encoding="utf-8")),
                    })
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        payload = (
            json.dumps(records, indent=2)
            if args.format == "json"
            else "\n".join(
                json.dumps(record, separators=(",", ":")) for record in records
            )
            + ("\n" if records else "")
        )
        if args.out == "-":
            print(payload, end="" if payload.endswith("\n") else "\n")
        else:
            Path(args.out).expanduser().write_text(payload, encoding="utf-8")
            print(f"Exported {len(records)} record(s) to {args.out}")
        return 0
    return 2

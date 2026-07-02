"""Privacy-safe local telemetry projection and export helpers.

NeMo Relay ATOF JSONL is canonical. The ``tel_*`` SQLite tables are a
disposable read model used by Hermes UX and can always be rebuilt.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import tempfile
import uuid
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

_SCHEMA_VERSION = "3"
_DATE_FILE_RE = re.compile(r"^hermes-atof-(\d{4}-\d{2}-\d{2})\.jsonl$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:authorization|api[_-]?key|token|secret|password|credential)", re.I
)
_SECRET_RE = re.compile(r"(?i)(?:bearer\s+|sk-|gh[pousr]_|AKIA)[A-Za-z0-9_./+=-]{8,}")
_PII_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
)
_BUILTIN_TOOLS = {
    "terminal",
    "web_search",
    "web_extract",
    "browser",
    "delegate_task",
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "send_message",
}
_KNOWN_ENTRYPOINTS = {
    "cli",
    "gateway",
    "tui",
    "browser",
    "electron",
    "api",
    "cron",
    "eval",
}
_KNOWN_PLATFORMS = {
    "cli",
    "telegram",
    "discord",
    "slack",
    "teams",
    "web",
    "desktop",
    "api",
}
_KNOWN_PROVIDERS = {
    "openai",
    "anthropic",
    "openrouter",
    "nous",
    "google",
    "mistral",
    "ollama",
    "custom",
}
_KNOWN_ERRORS = {
    "tool_error",
    "provider_error",
    "timeout",
    "cancelled",
    "authentication_error",
    "rate_limit",
    "invalid_request",
    "connection_error",
    "internal_error",
    "error",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tel_projection_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tel_scopes (
    id TEXT PRIMARY KEY, parent_id TEXT, run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_tel_scopes_parent ON tel_scopes(parent_id);
CREATE TABLE IF NOT EXISTS tel_runs (
    id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, session_id TEXT, turn_id TEXT,
    entrypoint TEXT, platform TEXT, provider TEXT, model TEXT,
    started_at REAL, ended_at REAL, outcome TEXT, completed INTEGER,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0, cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0, error_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_tel_runs_started ON tel_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_tel_runs_session ON tel_runs(session_id);
CREATE TABLE IF NOT EXISTS tel_model_calls (
    id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, provider TEXT, model TEXT,
    started_at REAL, ended_at REAL, input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0, cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0, estimated_cost_usd REAL NOT NULL DEFAULT 0,
    finish_reason TEXT, error_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_tel_model_run ON tel_model_calls(run_id);
CREATE TABLE IF NOT EXISTS tel_tool_calls (
    id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, tool_name TEXT,
    started_at REAL, ended_at REAL, status TEXT, result_class TEXT, error_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_tel_tool_run ON tel_tool_calls(run_id);
CREATE TABLE IF NOT EXISTS tel_error_events (
    id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, category TEXT,
    event_name TEXT, error_type TEXT, occurred_at REAL
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


def telemetry_db_path() -> Path:
    return get_hermes_home() / "telemetry" / "telemetry.db"


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


def _error_type(
    event: dict[str, Any], data: dict[str, Any], metadata: dict[str, Any]
) -> str | None:
    value = data.get("error_type") or metadata.get("error_type")
    if value:
        return str(value)
    error = event.get("error")
    if isinstance(error, dict):
        return str(error.get("type") or error.get("code") or "error")
    return "error" if error else None


class TelemetryProjection:
    """Thread-safe, idempotent SQLite projection of structural Relay events."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else telemetry_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._projector_connection: sqlite3.Connection | None = None
        self._pending_events: list[dict[str, Any]] = []
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            self._initialize(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _connection(self) -> sqlite3.Connection:
        conn = self._projector_connection
        if conn is None:
            conn = self._connect()
            self._projector_connection = conn
        return conn

    def _initialize(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tel_projection_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = conn.execute(
            "SELECT value FROM tel_projection_state WHERE key='schema_version'"
        ).fetchone()
        rebuild_required = bool(row and row[0] != _SCHEMA_VERSION)
        if rebuild_required:
            for table in (
                "tel_scopes",
                "tel_runs",
                "tel_model_calls",
                "tel_tool_calls",
                "tel_error_events",
            ):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO tel_projection_state(key,value) VALUES('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_SCHEMA_VERSION,),
        )
        conn.commit()
        if rebuild_required:
            conn.execute("BEGIN IMMEDIATE")
            for event in iter_atof_events():
                self._apply(conn, event)
            conn.execute(
                "INSERT INTO tel_projection_state(key,value) VALUES('last_rebuild_at',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (datetime.now(timezone.utc).isoformat(),),
            )
            conn.commit()

    def consume(self, event: Any) -> None:
        value = _event_dict(event)
        if not value.get("uuid"):
            return
        with self._lock:
            self._connection()
            self._pending_events.append(value)
            is_completion = value.get("scope_category") == "end" and (
                str(value.get("name", "")).startswith("hermes.run:")
                or str(value.get("category", "")).lower() in {"llm", "tool"}
            )
            if len(self._pending_events) >= 50 or is_completion:
                self._flush_pending()

    def _flush_pending(self) -> None:
        events = self._pending_events
        if not events:
            return
        conn = self._connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            for event in events:
                self._apply(conn, event)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        events.clear()

    def close(self) -> None:
        with self._lock:
            conn = self._projector_connection
            if conn is not None:
                self._flush_pending()
                conn.close()
                self._projector_connection = None

    def rebuild(self, directory: Path | None = None) -> dict[str, int]:
        self.close()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for table in (
                "tel_scopes",
                "tel_runs",
                "tel_model_calls",
                "tel_tool_calls",
                "tel_error_events",
            ):
                conn.execute(f"DELETE FROM {table}")
            events = 0
            for event in iter_atof_events(directory):
                self._apply(conn, event)
                events += 1
            conn.execute(
                "INSERT INTO tel_projection_state(key,value) VALUES('last_rebuild_at',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (datetime.now(timezone.utc).isoformat(),),
            )
            conn.commit()
        result = self.counts()
        result["events"] = events
        return result

    def counts(self) -> dict[str, int]:
        with closing(self._connect()) as conn:
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

    def _resolve_run(
        self, conn: sqlite3.Connection, parent_id: str | None
    ) -> str | None:
        if not parent_id:
            return None
        row = conn.execute(
            "WITH RECURSIVE ancestors(id,parent_id,run_id) AS ("
            " SELECT id,parent_id,run_id FROM tel_scopes WHERE id=? UNION ALL "
            " SELECT s.id,s.parent_id,s.run_id FROM tel_scopes s JOIN ancestors a ON s.id=a.parent_id"
            ") SELECT run_id FROM ancestors WHERE run_id IS NOT NULL LIMIT 1",
            (parent_id,),
        ).fetchone()
        return str(row[0]) if row and row[0] else None

    def _propagate_run(
        self, conn: sqlite3.Connection, scope_id: str, run_id: str
    ) -> None:
        conn.execute(
            "WITH RECURSIVE descendants(id) AS (SELECT ? UNION ALL "
            "SELECT s.id FROM tel_scopes s JOIN descendants d ON s.parent_id=d.id) "
            "UPDATE tel_scopes SET run_id=? WHERE id IN (SELECT id FROM descendants)",
            (scope_id, run_id),
        )
        for table in ("tel_model_calls", "tel_tool_calls", "tel_error_events"):
            conn.execute(
                f"UPDATE {table} SET run_id=? WHERE id IN ("
                "WITH RECURSIVE descendants(id) AS (SELECT ? UNION ALL "
                "SELECT s.id FROM tel_scopes s JOIN descendants d ON s.parent_id=d.id) SELECT id FROM descendants)",
                (run_id, scope_id),
            )

    def _apply(self, conn: sqlite3.Connection, event: dict[str, Any]) -> None:
        event_id = str(event["uuid"])
        parent_id = str(event.get("parent_uuid") or "") or None
        category = str(event.get("category") or "").lower()
        phase = str(event.get("scope_category") or "").lower()
        name = str(event.get("name") or "")
        timestamp = _epoch(event.get("timestamp"))
        metadata = _dict(event.get("metadata"))
        data = _dict(event.get("data"))
        profile = _dict(event.get("category_profile"))
        explicit_run = metadata.get("run_id") or data.get("run_id")
        run_id = (
            str(explicit_run) if explicit_run else self._resolve_run(conn, parent_id)
        )
        if phase:
            conn.execute(
                "INSERT INTO tel_scopes(id,parent_id,run_id) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "parent_id=COALESCE(excluded.parent_id,parent_id),run_id=COALESCE(excluded.run_id,run_id)",
                (event_id, parent_id, run_id),
            )
            if run_id:
                self._propagate_run(conn, event_id, run_id)
        error_type = _error_type(event, data, metadata)

        if category == "agent" and name.startswith("hermes.run:"):
            run_id = run_id or name.removeprefix("hermes.run:")
            conn.execute(
                "UPDATE tel_scopes SET run_id=? WHERE id=?", (run_id, event_id)
            )
            if phase == "start":
                conn.execute(
                    "INSERT INTO tel_runs(id,run_id,parent_id,session_id,turn_id,entrypoint,platform,provider,model,started_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                    "run_id=excluded.run_id,parent_id=COALESCE(excluded.parent_id,parent_id),"
                    "session_id=COALESCE(excluded.session_id,session_id),turn_id=COALESCE(excluded.turn_id,turn_id),"
                    "entrypoint=COALESCE(excluded.entrypoint,entrypoint),platform=COALESCE(excluded.platform,platform),"
                    "provider=COALESCE(excluded.provider,provider),model=COALESCE(excluded.model,model),"
                    "started_at=COALESCE(excluded.started_at,started_at)",
                    (
                        event_id,
                        run_id,
                        parent_id,
                        data.get("session_id") or metadata.get("session_id"),
                        metadata.get("turn_id"),
                        data.get("entrypoint") or metadata.get("entrypoint"),
                        metadata.get("platform"),
                        metadata.get("provider"),
                        metadata.get("model"),
                        timestamp,
                    ),
                )
            elif phase == "end":
                usage = _dict(data.get("usage"))
                outcome = str(
                    data.get("outcome")
                    or ("completed" if data.get("completed") else "failed")
                )
                conn.execute(
                    "INSERT INTO tel_runs(id,run_id,parent_id,ended_at,outcome,completed,input_tokens,output_tokens,"
                    "cache_read_tokens,cache_write_tokens,estimated_cost_usd,error_type) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET ended_at=excluded.ended_at,outcome=excluded.outcome,"
                    "completed=excluded.completed,input_tokens=excluded.input_tokens,output_tokens=excluded.output_tokens,"
                    "cache_read_tokens=excluded.cache_read_tokens,cache_write_tokens=excluded.cache_write_tokens,"
                    "estimated_cost_usd=excluded.estimated_cost_usd,error_type=COALESCE(excluded.error_type,error_type)",
                    (
                        event_id,
                        run_id,
                        parent_id,
                        timestamp,
                        outcome,
                        int(bool(data.get("completed"))),
                        int(
                            _number(
                                usage.get("input_tokens") or usage.get("prompt_tokens")
                            )
                        ),
                        int(
                            _number(
                                usage.get("output_tokens")
                                or usage.get("completion_tokens")
                            )
                        ),
                        int(_number(usage.get("cache_read_tokens"))),
                        int(_number(usage.get("cache_write_tokens"))),
                        _number(data.get("estimated_cost_usd") or usage.get("cost")),
                        error_type,
                    ),
                )
            self._propagate_run(conn, event_id, run_id)
            return

        if category == "llm":
            if phase == "start":
                conn.execute(
                    "INSERT INTO tel_model_calls(id,run_id,parent_id,provider,model,started_at) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET run_id=COALESCE(excluded.run_id,run_id),parent_id=COALESCE(excluded.parent_id,parent_id),"
                    "provider=COALESCE(excluded.provider,provider),model=COALESCE(excluded.model,model),started_at=COALESCE(excluded.started_at,started_at)",
                    (
                        event_id,
                        run_id,
                        parent_id,
                        metadata.get("provider") or name,
                        profile.get("model_name")
                        or metadata.get("model")
                        or data.get("model"),
                        timestamp,
                    ),
                )
            elif phase == "end":
                usage = _dict(data.get("usage"))
                conn.execute(
                    "INSERT INTO tel_model_calls(id,run_id,parent_id,provider,model,ended_at,input_tokens,output_tokens,"
                    "cache_read_tokens,cache_write_tokens,estimated_cost_usd,finish_reason,error_type) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET run_id=COALESCE(excluded.run_id,run_id),ended_at=excluded.ended_at,"
                    "input_tokens=excluded.input_tokens,output_tokens=excluded.output_tokens,cache_read_tokens=excluded.cache_read_tokens,"
                    "cache_write_tokens=excluded.cache_write_tokens,estimated_cost_usd=excluded.estimated_cost_usd,"
                    "finish_reason=COALESCE(excluded.finish_reason,finish_reason),"
                    "error_type=COALESCE(excluded.error_type,error_type)",
                    (
                        event_id,
                        run_id,
                        parent_id,
                        metadata.get("provider") or name,
                        profile.get("model_name")
                        or metadata.get("model")
                        or data.get("model"),
                        timestamp,
                        int(
                            _number(
                                usage.get("input_tokens") or usage.get("prompt_tokens")
                            )
                        ),
                        int(
                            _number(
                                usage.get("output_tokens")
                                or usage.get("completion_tokens")
                            )
                        ),
                        int(_number(usage.get("cache_read_tokens"))),
                        int(_number(usage.get("cache_write_tokens"))),
                        _number(usage.get("cost") or data.get("estimated_cost_usd")),
                        data.get("finish_reason"),
                        error_type,
                    ),
                )
        elif category == "tool":
            tool_name = profile.get("tool_name") or data.get("tool_name") or name
            if phase == "start":
                conn.execute(
                    "INSERT INTO tel_tool_calls(id,run_id,parent_id,tool_name,started_at) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET run_id=COALESCE(excluded.run_id,run_id),parent_id=COALESCE(excluded.parent_id,parent_id),"
                    "tool_name=COALESCE(excluded.tool_name,tool_name),started_at=COALESCE(excluded.started_at,started_at)",
                    (event_id, run_id, parent_id, tool_name, timestamp),
                )
            elif phase == "end":
                conn.execute(
                    "INSERT INTO tel_tool_calls(id,run_id,parent_id,tool_name,ended_at,status,result_class,error_type) "
                    "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET run_id=COALESCE(excluded.run_id,run_id),"
                    "ended_at=excluded.ended_at,status=COALESCE(excluded.status,status),result_class=COALESCE(excluded.result_class,result_class),"
                    "error_type=COALESCE(excluded.error_type,error_type)",
                    (
                        event_id,
                        run_id,
                        parent_id,
                        tool_name,
                        timestamp,
                        data.get("status"),
                        data.get("result_class"),
                        error_type,
                    ),
                )
        if error_type or "error" in name.lower():
            conn.execute(
                "INSERT OR REPLACE INTO tel_error_events(id,run_id,parent_id,category,event_name,error_type,occurred_at) VALUES(?,?,?,?,?,?,?)",
                (
                    event_id,
                    run_id,
                    parent_id,
                    category or "custom",
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
    limit: int = 20,
) -> dict[str, Any]:
    clauses = ["started_at >= ?"]
    params: list[Any] = [cutoff]
    if source:
        clauses.append("platform = ?")
        params.append(source)
    try:
        runs = conn.execute(
            f"SELECT run_id,session_id,entrypoint,platform,provider,model,started_at,ended_at,outcome,completed,"
            f"input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,estimated_cost_usd,error_type FROM tel_runs "
            f"WHERE {' AND '.join(clauses)} ORDER BY started_at DESC",
            params,
        ).fetchall()
        child_where = "r.started_at >= ?" + (" AND r.platform=?" if source else "")
        child_params = [cutoff] + ([source] if source else [])
        models = conn.execute(
            "SELECT c.provider,c.model,c.started_at,c.ended_at,c.input_tokens,c.output_tokens,c.cache_read_tokens,"
            "c.cache_write_tokens,c.estimated_cost_usd,c.error_type FROM tel_model_calls c "
            "JOIN tel_runs r ON r.run_id=c.run_id WHERE " + child_where,
            child_params,
        ).fetchall()
        tools = conn.execute(
            "SELECT c.tool_name,c.started_at,c.ended_at,c.status,c.error_type FROM tel_tool_calls c "
            "JOIN tel_runs r ON r.run_id=c.run_id WHERE " + child_where,
            child_params,
        ).fetchall()
        errors = conn.execute(
            "SELECT e.category,e.error_type,COUNT(*) FROM tel_error_events e JOIN tel_runs r ON r.run_id=e.run_id "
            "WHERE " + child_where + " GROUP BY e.category,e.error_type",
            child_params,
        ).fetchall()
    except sqlite3.OperationalError:
        return {"run_count": 0, "available": False}
    run_durations = [
        row[7] - row[6] for row in runs if row[6] is not None and row[7] is not None
    ]
    model_durations = [
        row[3] - row[2] for row in models if row[2] is not None and row[3] is not None
    ]
    tool_durations = [
        row[2] - row[1] for row in tools if row[1] is not None and row[2] is not None
    ]
    completed = sum(bool(row[9]) or row[8] in {"completed", "success"} for row in runs)
    failed = sum(bool(row[15]) or row[8] in {"failed", "error"} for row in runs)
    entrypoints = Counter(row[2] or "unknown" for row in runs)
    platforms = Counter(row[3] or "unknown" for row in runs)
    providers = Counter(row[0] or "unknown" for row in models)
    model_mix = Counter(row[1] or "unknown" for row in models)
    tool_groups: dict[str, dict[str, Any]] = {}
    for tool_name, start, end, status, error_type in tools:
        item = tool_groups.setdefault(
            tool_name or "unknown", {"calls": 0, "failures": 0, "durations": []}
        )
        item["calls"] += 1
        item["failures"] += int(bool(error_type) or status == "error")
        if start is not None and end is not None:
            item["durations"].append(end - start)
    per_tool = {
        name: {
            "calls": item["calls"],
            "failures": item["failures"],
            "failure_rate": item["failures"] / item["calls"] if item["calls"] else 0,
            "latency_p50": _percentile(item["durations"], 0.5),
            "latency_p95": _percentile(item["durations"], 0.95),
        }
        for name, item in tool_groups.items()
    }

    cache_hits = sum((row[6] or 0) > 0 for row in models)
    daily: dict[str, dict[str, Any]] = {}
    for row in runs:
        if row[6] is None:
            continue
        day = datetime.fromtimestamp(row[6], timezone.utc).strftime("%Y-%m-%d")
        item = daily.setdefault(
            day,
            {
                "day": day,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "reasoning_tokens": 0,
                "estimated_cost": 0.0,
                "actual_cost": 0.0,
                "sessions": 0,
                "api_calls": 0,
            },
        )
        item["sessions"] += 1
    by_model: dict[str, dict[str, Any]] = {}
    for row in models:
        name = row[1] or "unknown"
        item = by_model.setdefault(
            name,
            {
                "model": name,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0,
                "sessions": 0,
                "api_calls": 0,
            },
        )
        item["input_tokens"] += row[4] or 0
        item["output_tokens"] += row[5] or 0
        item["estimated_cost"] += row[8] or 0
        item["sessions"] += 1
        item["api_calls"] += 1
        if row[2] is not None:
            day = datetime.fromtimestamp(row[2], timezone.utc).strftime("%Y-%m-%d")
            day_item = daily.setdefault(
                day,
                {
                    "day": day,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "reasoning_tokens": 0,
                    "estimated_cost": 0.0,
                    "actual_cost": 0.0,
                    "sessions": 0,
                    "api_calls": 0,
                },
            )
            day_item["input_tokens"] += row[4] or 0
            day_item["output_tokens"] += row[5] or 0
            day_item["cache_read_tokens"] += row[6] or 0
            day_item["estimated_cost"] += row[8] or 0
            day_item["api_calls"] += 1
    totals = {
        "total_input": sum(row[4] or 0 for row in models),
        "total_output": sum(row[5] or 0 for row in models),
        "total_cache_read": sum(row[6] or 0 for row in models),
        "total_reasoning": 0,
        "total_estimated_cost": sum(row[8] or 0 for row in models),
        "total_actual_cost": 0,
        "total_sessions": len(runs),
        "total_api_calls": len(models),
    }
    return {
        "available": True,
        "run_count": len(runs),
        "completed": completed,
        "failed": failed,
        "interrupted": sum(
            row[8] in {"interrupted", "cancelled", "timeout"} for row in runs
        ),
        "completion_rate": completed / len(runs) if runs else 0,
        "failure_rate": failed / len(runs) if runs else 0,
        "run_latency_p50": _percentile(run_durations, 0.5),
        "run_latency_p95": _percentile(run_durations, 0.95),
        "model_latency_p50": _percentile(model_durations, 0.5),
        "model_latency_p95": _percentile(model_durations, 0.95),
        "tool_latency_p50": _percentile(tool_durations, 0.5),
        "tool_latency_p95": _percentile(tool_durations, 0.95),
        "model_calls": len(models),
        "tool_calls": len(tools),
        "tool_failures": sum(item["failures"] for item in tool_groups.values()),
        "input_tokens": sum(row[4] or 0 for row in models),
        "output_tokens": sum(row[5] or 0 for row in models),
        "cache_read_tokens": sum(row[6] or 0 for row in models),
        "cache_write_tokens": sum(row[7] or 0 for row in models),
        "cache_hit_rate": cache_hits / len(models) if models else 0,
        "estimated_cost_usd": sum(row[8] or 0 for row in models),
        "totals": totals,
        "daily": [daily[key] for key in sorted(daily)],
        "by_model": sorted(
            by_model.values(),
            key=lambda item: item["input_tokens"] + item["output_tokens"],
            reverse=True,
        ),
        "entrypoints": dict(entrypoints),
        "platforms": dict(platforms),
        "providers": dict(providers),
        "models": dict(model_mix),
        "tools": per_tool,
        "error_classes": [
            {"category": row[0], "error_type": row[1], "count": row[2]}
            for row in errors
        ],
        "recent_runs": [
            {
                "run_id": row[0],
                "session_id": row[1],
                "entrypoint": row[2],
                "platform": row[3],
                "started_at": row[6],
                "duration_seconds": row[7] - row[6]
                if row[6] is not None and row[7] is not None
                else None,
                "outcome": row[8],
                "error_type": row[15],
            }
            for row in runs[: max(0, limit)]
        ],
    }


def summarize_telemetry(
    *,
    cutoff: float = 0,
    source: str | None = None,
    limit: int = 20,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(db_path) if db_path else telemetry_db_path()
    if not path.exists():
        return {"run_count": 0, "available": False}
    with closing(sqlite3.connect(path)) as conn:
        return summarize_connection(conn, cutoff=cutoff, source=source, limit=limit)


def session_usage_lines(session_id: str | None, *, markdown: bool = False) -> list[str]:
    if not session_id:
        return []
    db_path = telemetry_db_path()
    if not db_path.exists():
        return []
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            run = conn.execute(
                "SELECT COUNT(*),"
                "SUM(CASE WHEN error_type IS NOT NULL OR outcome IN ('failed','error') THEN 1 ELSE 0 END) "
                "FROM tel_runs WHERE session_id=?",
                (session_id,),
            ).fetchone()
            calls = conn.execute(
                "SELECT (SELECT COUNT(*) FROM tel_model_calls WHERE run_id IN (SELECT run_id FROM tel_runs WHERE session_id=?)),"
                "(SELECT COALESCE(SUM(input_tokens),0) FROM tel_model_calls WHERE run_id IN (SELECT run_id FROM tel_runs WHERE session_id=?)),"
                "(SELECT COALESCE(SUM(output_tokens),0) FROM tel_model_calls WHERE run_id IN (SELECT run_id FROM tel_runs WHERE session_id=?)),"
                "(SELECT COALESCE(SUM(cache_read_tokens),0) FROM tel_model_calls WHERE run_id IN (SELECT run_id FROM tel_runs WHERE session_id=?)),"
                "(SELECT COALESCE(SUM(estimated_cost_usd),0) FROM tel_model_calls WHERE run_id IN (SELECT run_id FROM tel_runs WHERE session_id=?)),"
                "(SELECT COUNT(*) FROM tel_tool_calls WHERE run_id IN (SELECT run_id FROM tel_runs WHERE session_id=?)),"
                "(SELECT COUNT(*) FROM tel_tool_calls WHERE run_id IN (SELECT run_id FROM tel_runs WHERE session_id=?) "
                "AND (error_type IS NOT NULL OR status='error'))",
                (session_id,) * 7,
            ).fetchone()
    except sqlite3.OperationalError:
        return []
    if not run or not run[0]:
        return []
    header = "**🔭 Relay runtime:**" if markdown else "🔭 Relay runtime"
    return [
        header,
        f"Runs: {run[0]} | Model calls: {calls[0]} | Tool calls: {calls[5]} | Failures: {run[1] + calls[6]}",
        f"Tokens: {calls[1] + calls[2]:,} (in: {calls[1]:,} / out: {calls[2]:,}) | Cache read: {calls[3]:,}",
        f"Estimated cost: ${calls[4]:.4f}",
    ]


def _structural_event(event: dict[str, Any]) -> dict[str, Any]:
    allowed_data = {
        "run_id",
        "session_id",
        "turn_id",
        "entrypoint",
        "provider",
        "model",
        "tool_name",
        "tool_call_id",
        "status",
        "outcome",
        "completed",
        "duration_ms",
        "result_class",
        "error_type",
        "finish_reason",
        "usage",
        "parent_session_id",
        "child_session_id",
        "child_subagent_id",
        "child_role",
        "child_status",
    }
    allowed_metadata = allowed_data | {
        "telemetry_schema_version",
        "platform",
        "task_id",
        "api_request_id",
        "api_mode",
    }
    safe = {
        key: event[key]
        for key in (
            "atof_version",
            "kind",
            "parent_uuid",
            "uuid",
            "timestamp",
            "name",
            "scope_category",
            "category",
        )
        if key in event
    }
    profile = _dict(event.get("category_profile"))
    safe["category_profile"] = {
        key: profile[key]
        for key in ("model_name", "tool_call_id", "tool_name")
        if key in profile
    }
    safe_data = {
        key: value
        for key, value in _dict(event.get("data")).items()
        if key in allowed_data
    }
    if isinstance(safe_data.get("usage"), dict):
        safe_data["usage"] = {
            key: value
            for key, value in safe_data["usage"].items()
            if key
            in {
                "input_tokens",
                "prompt_tokens",
                "output_tokens",
                "completion_tokens",
                "total_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "cost",
            }
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    safe["data"] = safe_data
    safe["metadata"] = {
        key: value
        for key, value in _dict(event.get("metadata")).items()
        if key in allowed_metadata
    }
    return safe


def _event_date(event: dict[str, Any], fallback: datetime) -> str:
    timestamp = _epoch(event.get("timestamp"))
    value = (
        datetime.fromtimestamp(timestamp, timezone.utc)
        if timestamp is not None
        else fallback
    )
    return value.strftime("%Y-%m-%d")


def compact_legacy_atof(config: dict[str, Any] | None = None) -> int:
    """Atomically split the one-time legacy JSONL file into daily owned files."""
    config = config if config is not None else telemetry_config()
    directory = atof_directory(config)
    legacy = directory / str(
        _dict(config.get("atof")).get("filename") or "hermes-atof.jsonl"
    )
    in_progress = legacy.with_suffix(legacy.suffix + ".compacting")
    if not in_progress.exists():
        if not legacy.exists():
            return 0
        directory.mkdir(parents=True, exist_ok=True)
        legacy.replace(in_progress)

    staged: dict[str, tuple[Path, Any]] = {}
    count = 0
    fallback = datetime.fromtimestamp(in_progress.stat().st_mtime, timezone.utc)
    try:
        with in_progress.open(encoding="utf-8") as source:
            for line in source:
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                day = _event_date(event, fallback)
                if day not in staged:
                    handle = tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=directory,
                        prefix=f".{day}-",
                        suffix=".stage",
                        delete=False,
                    )
                    staged[day] = (Path(handle.name), handle)
                staged[day][1].write(json.dumps(event, separators=(",", ":")) + "\n")
                count += 1
        for _, handle in staged.values():
            handle.flush()
            handle.close()
        for day, (stage_path, _) in staged.items():
            destination = directory / f"hermes-atof-{day}.jsonl"
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=directory,
                prefix=f".{day}-",
                suffix=".merge",
                delete=False,
            ) as merged:
                merged_path = Path(merged.name)
                if destination.exists():
                    with destination.open("rb") as existing:
                        while chunk := existing.read(1024 * 1024):
                            merged.write(chunk)
                with stage_path.open("rb") as addition:
                    while chunk := addition.read(1024 * 1024):
                        merged.write(chunk)
                merged.flush()
            merged_path.replace(destination)
            stage_path.unlink(missing_ok=True)
        in_progress.unlink(missing_ok=True)
    finally:
        for stage_path, handle in staged.values():
            if not handle.closed:
                handle.close()
            stage_path.unlink(missing_ok=True)
    return count


def install_id(config: dict[str, Any] | None = None) -> str:
    config = config if config is not None else telemetry_config()
    configured = str(config.get("install_id") or "").strip()
    if configured:
        return configured
    path = get_hermes_home() / "telemetry" / "install_id"
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    value = str(uuid.uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text(encoding="utf-8").strip()
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value + "\n")
    return value


def enforce_retention(config: dict[str, Any] | None = None) -> dict[str, int]:
    config = config if config is not None else telemetry_config()
    compact_legacy_atof(config)
    days = max(1, int(config.get("retention_days", 90)))
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
    deleted = {"atof": 0, "atif": 0, "rows": 0}
    for path in _json_files(atof_directory(config)):
        match = _DATE_FILE_RE.match(path.name)
        if (
            match
            and datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
            < cutoff_dt
        ):
            path.unlink(missing_ok=True)
            deleted["atof"] += 1
    for path in _json_files(atif_directory(config)):
        if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) < cutoff_dt:
            path.unlink(missing_ok=True)
            deleted["atif"] += 1
    db_path = telemetry_db_path()
    if db_path.exists():
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                cutoff = cutoff_dt.timestamp()
                deleted["rows"] = conn.execute(
                    "DELETE FROM tel_runs WHERE started_at < ?", (cutoff,)
                ).rowcount
                for table in ("tel_model_calls", "tel_tool_calls", "tel_error_events"):
                    conn.execute(
                        f"DELETE FROM {table} WHERE run_id NOT IN (SELECT run_id FROM tel_runs)"
                    )
                conn.execute(
                    "DELETE FROM tel_scopes WHERE run_id NOT IN (SELECT run_id FROM tel_runs) AND run_id IS NOT NULL"
                )
                conn.commit()
        except sqlite3.OperationalError:
            pass
    return deleted


def telemetry_status() -> dict[str, Any]:
    config = telemetry_config()
    try:
        import nemo_relay

        dependency: dict[str, Any] = {
            "available": True,
            "version": getattr(nemo_relay, "__version__", "unknown"),
        }
    except Exception as exc:
        dependency = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        projection = TelemetryProjection().counts()
        projection_error = None
    except Exception as exc:
        projection, projection_error = {}, f"{type(exc).__name__}: {exc}"
    files = _json_files(atof_directory(config))
    available = dependency.get("available", False) and projection_error is None
    return {
        "health": "healthy" if available else "degraded",
        "local_enabled": bool(config.get("local", True)),
        "plugin_default_enabled": True,
        "relay_dependency": dependency,
        "consent_state": config.get("consent_state", "unknown"),
        "allow_aggregate": bool(config.get("allow_aggregate", False)),
        "retention_days": int(config.get("retention_days", 90)),
        "content_capture": bool(
            config.get("capture_content", False)
            or _dict(config.get("trajectories")).get("enabled", False)
        ),
        "redaction": config.get("content_redaction", "pii"),
        "atof_directory": str(atof_directory(config)),
        "atof_files": len(files),
        "atof_bytes": sum(path.stat().st_size for path in files),
        "projection": projection,
        "projection_error": projection_error,
        "network_export": {
            "otlp_enabled": bool(
                _dict(_dict(config.get("export")).get("otlp")).get("enabled", False)
            ),
            "openinference_enabled": bool(
                _dict(_dict(config.get("export")).get("openinference")).get(
                    "enabled", False
                )
            ),
        },
    }


def aggregate_bundle(days: int = 30) -> dict[str, Any]:
    config = telemetry_config()
    if config.get("consent_state") != "granted" or not config.get(
        "allow_aggregate", False
    ):
        raise PermissionError(
            "aggregate export requires consent_state=granted and allow_aggregate=true"
        )
    projection = TelemetryProjection()
    with closing(projection._connect()) as conn:
        summary = summarize_connection(
            conn, cutoff=(datetime.now().timestamp() - days * 86400), limit=0
        )
    tools = summary.pop("tools", {})
    safe_tools: dict[str, dict[str, int]] = {}
    for name, values in tools.items():
        bucket = name if name in _BUILTIN_TOOLS else "custom"
        item = safe_tools.setdefault(bucket, {"calls": 0, "failures": 0})
        item["calls"] += int(values["calls"])
        item["failures"] += int(values["failures"])
    summary["tools"] = safe_tools
    summary.pop("recent_runs", None)
    summary.pop("daily", None)
    summary.pop("by_model", None)
    summary.pop("totals", None)
    summary["entrypoints"] = _bounded_buckets(
        summary.get("entrypoints", {}), _KNOWN_ENTRYPOINTS, "other"
    )
    summary["platforms"] = _bounded_buckets(
        summary.get("platforms", {}), _KNOWN_PLATFORMS, "other"
    )
    summary["providers"] = _bounded_buckets(
        summary.get("providers", {}), _KNOWN_PROVIDERS, "unknown"
    )
    summary["models"] = {"known": sum(summary.pop("models", {}).values())}
    for item in summary.get("error_classes", []):
        if item.get("error_type") not in _KNOWN_ERRORS:
            item["error_type"] = "other_error"
    try:
        from hermes_cli import __version__ as hermes_version
        import nemo_relay

        relay_version = getattr(nemo_relay, "__version__", "0.4.0")
    except Exception:
        hermes_version, relay_version = "unknown", "unavailable"
    return {
        "schema": "hermes.telemetry.aggregate/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "install_id": install_id(config),
        "hermes_version": hermes_version,
        "relay_version": relay_version,
        "metrics": summary,
    }


def _bounded_buckets(
    values: dict[str, int], allowed: set[str], fallback: str
) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, count in values.items():
        bucket = name if name in allowed else fallback
        result[bucket] = result.get(bucket, 0) + int(count)
    return result


def purge(*, reset_install_id: bool = False) -> dict[str, int]:
    removed = {"atof": 0, "atif": 0, "rows": 0}
    for key, directory in (("atof", atof_directory()), ("atif", atif_directory())):
        for path in _json_files(directory):
            path.unlink(missing_ok=True)
            removed[key] += 1
    db_path = telemetry_db_path()
    if db_path.exists():
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                for table in (
                    "tel_scopes",
                    "tel_runs",
                    "tel_model_calls",
                    "tel_tool_calls",
                    "tel_error_events",
                ):
                    removed["rows"] += conn.execute(f"DELETE FROM {table}").rowcount
                conn.commit()
        except sqlite3.OperationalError:
            pass
    if reset_install_id:
        (get_hermes_home() / "telemetry" / "install_id").unlink(missing_ok=True)
    return removed


def _set_consent(value: str) -> None:
    from hermes_cli.config import load_config, save_config

    config = load_config()
    config.setdefault("telemetry", {})["consent_state"] = value
    save_config(config)


def _records(*, include_content: bool, since: float = 0) -> Iterable[dict[str, Any]]:
    redaction = str(telemetry_config().get("content_redaction") or "pii")
    for event in iter_atof_events():
        timestamp = _epoch(event.get("timestamp")) or 0
        if timestamp >= since:
            yield (
                _redact_historical(event, redaction)
                if include_content
                else _structural_event(event)
            )


def _redact_historical(value: Any, policy: str) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if _SENSITIVE_KEY_RE.search(str(key))
            else _redact_historical(item, policy)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_historical(item, policy) for item in value]
    if not isinstance(value, str):
        return value
    redacted = _SECRET_RE.sub("[REDACTED]", value)
    if policy != "secrets":
        for pattern in _PII_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _file_records(
    *, include_content: bool, since: float = 0
) -> Iterable[dict[str, Any]]:
    yield from _records(include_content=include_content, since=since)
    if not include_content:
        return
    redaction = str(telemetry_config().get("content_redaction") or "pii")
    for path in _json_files(atif_directory()):
        try:
            trajectory = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        yield {"_kind": "atif", "trajectory": _redact_historical(trajectory, redaction)}


def _write_records(
    records: Iterable[dict[str, Any]], destination: str, output_format: str
) -> int:
    stream = (
        sys.stdout
        if destination == "-"
        else Path(destination).expanduser().open("w", encoding="utf-8")
    )
    count = 0
    try:
        if output_format == "json":
            stream.write("[\n")
        for record in records:
            if output_format == "json":
                if count:
                    stream.write(",\n")
                stream.write(json.dumps(record, indent=2))
            else:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
        if output_format == "json":
            stream.write("\n]\n")
    finally:
        if stream is not sys.stdout:
            stream.close()
    return count


def telemetry_command(args: Any) -> int:
    action = getattr(args, "telemetry_action", None) or "status"
    if action == "status":
        result = telemetry_status()
    elif action == "rebuild":
        enforce_retention()
        result = TelemetryProjection().rebuild()
    elif action == "preview":
        projection = TelemetryProjection()
        with closing(projection._connect()) as conn:
            result = summarize_connection(
                conn,
                cutoff=datetime.now().timestamp() - args.days * 86400,
                limit=args.limit,
            )
    elif action == "consent":
        consent_action = getattr(args, "consent_action", None) or "status"
        if consent_action in {"grant", "deny"}:
            _set_consent("granted" if consent_action == "grant" else "denied")
        result = {"consent_state": telemetry_config().get("consent_state", "unknown")}
    elif action == "aggregate":
        result = aggregate_bundle(args.days)
        Path(args.out).expanduser().write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    elif action == "purge":
        if not args.confirm:
            print("purge requires --confirm", file=sys.stderr)
            return 2
        result = purge(reset_install_id=args.reset_install_id)
    elif action == "export":
        enforce_retention()
        since = _epoch(getattr(args, "since", None)) or 0
        count = _write_records(
            _file_records(include_content=args.include_content, since=since),
            args.out,
            args.format,
        )
        result = {"exported": count, "out": args.out}
    else:
        return 2
    if getattr(args, "json", False):
        print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    elif action not in {"export", "aggregate"}:
        print("\n".join(f"{key}: {value}" for key, value in result.items()))
    return 0

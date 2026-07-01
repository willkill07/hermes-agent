"""``hermes telemetry`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_telemetry_parser(subparsers, *, cmd_telemetry: Callable) -> None:
    parser = subparsers.add_parser(
        "telemetry",
        help="Inspect and export local NeMo Relay telemetry",
        description="Inspect the local ATOF source and its rebuildable SQLite projection.",
    )
    actions = parser.add_subparsers(dest="telemetry_action")
    status = actions.add_parser(
        "status", help="Show collection, privacy, and projection health"
    )
    status.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    preview = actions.add_parser(
        "preview", help="Preview aggregate run metrics without uploading data"
    )
    preview.add_argument(
        "--days", type=int, default=30, help="Number of days to include (default: 30)"
    )
    preview.add_argument(
        "--limit", type=int, default=20, help="Reserved for detailed previews"
    )
    preview.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    rebuild = actions.add_parser(
        "rebuild", help="Rebuild state.db telemetry tables from canonical ATOF"
    )
    rebuild.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    export = actions.add_parser("export", help="Export canonical local telemetry")
    export.add_argument("--out", default="-", help="Output file, or - for stdout")
    export.add_argument(
        "--since", help="Only include events at or after this ISO-8601 timestamp"
    )
    export.add_argument("--format", choices=("ndjson", "json"), default="ndjson")
    export.add_argument(
        "--include-content",
        action="store_true",
        help="Include opted-in ATIF trajectories",
    )
    export.add_argument(
        "--otlp", action="store_true", help="Replay to the configured OTLP exporter"
    )
    export.add_argument(
        "--openinference",
        action="store_true",
        help="Replay to the configured OpenInference exporter",
    )
    parser.set_defaults(func=cmd_telemetry)

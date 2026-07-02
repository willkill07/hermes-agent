"""``hermes telemetry`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_telemetry_parser(subparsers, *, cmd_telemetry: Callable) -> None:
    parser = subparsers.add_parser(
        "telemetry", help="Inspect and govern local NeMo Relay telemetry"
    )
    actions = parser.add_subparsers(dest="telemetry_action")
    status = actions.add_parser(
        "status", help="Show collection, privacy, and projection health"
    )
    status.add_argument("--json", action="store_true")
    preview = actions.add_parser("preview", help="Preview local aggregate metrics")
    preview.add_argument("--days", type=int, default=30)
    preview.add_argument("--limit", type=int, default=20)
    preview.add_argument("--json", action="store_true")
    rebuild = actions.add_parser("rebuild", help="Rebuild SQLite from canonical ATOF")
    rebuild.add_argument("--json", action="store_true")
    consent = actions.add_parser(
        "consent", help="Inspect or change aggregate-sharing consent"
    )
    consent_actions = consent.add_subparsers(dest="consent_action")
    consent_actions.add_parser("status")
    consent_actions.add_parser("grant")
    consent_actions.add_parser("deny")
    consent.add_argument("--json", action="store_true")
    aggregate = actions.add_parser(
        "aggregate", help="Export a consent-gated anonymous aggregate bundle"
    )
    aggregate.add_argument("aggregate_action", choices=("export",))
    aggregate.add_argument("--out", required=True)
    aggregate.add_argument("--days", type=int, default=30)
    aggregate.add_argument("--json", action="store_true")
    purge = actions.add_parser("purge", help="Delete local telemetry")
    purge.add_argument("--confirm", action="store_true")
    purge.add_argument("--reset-install-id", action="store_true")
    purge.add_argument("--json", action="store_true")
    export = actions.add_parser("export", help="Export canonical local telemetry")
    export.add_argument("--out", default="-")
    export.add_argument("--format", choices=("ndjson", "json"), default="ndjson")
    export.add_argument(
        "--since", help="Only export events at or after this ISO-8601 timestamp"
    )
    export.add_argument("--include-content", action="store_true")
    export.add_argument("--json", action="store_true")
    parser.set_defaults(func=cmd_telemetry)

---
name: telemetry
description: Inspect, govern, rebuild, purge, and export Hermes local NeMo Relay telemetry safely.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [telemetry, observability, privacy, relay, atof, atif]
    related_skills: [telemetry-analysis]
---

# Hermes Telemetry Operations

Use `hermes telemetry status` first. Local structural collection is enabled by
default; content and network exporters are not.

- Rebuild the disposable SQLite projection with `hermes telemetry rebuild`.
- Preview local metrics with `hermes telemetry preview --days 30`.
- Export structural ATOF with `hermes telemetry export --out events.ndjson`.
- Use `--include-content` only when the user explicitly requests content-bearing data.
- Aggregate bundles require `hermes telemetry consent grant` and
  `telemetry.allow_aggregate: true`; they are written locally and never uploaded automatically.
- Delete telemetry only after explicit confirmation with `hermes telemetry purge --confirm`.

Never print raw event content, exporter headers, credentials, install IDs, or
arbitrary tool arguments/results into chat unless the user explicitly requests
the sensitive content and understands the risk.

---
name: telemetry
description: "Inspect and operate local runtime telemetry."
version: 1.0.0
author: "Will Killian (@willkill07), Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [telemetry, observability, nemo-relay, privacy]
    category: devops
    related_skills: [telemetry-analysis]
---

# Telemetry Skill

This skill operates Hermes runtime telemetry through its supported CLI. It
does not edit canonical event files or bypass content-consent controls.

## When to Use

- A user asks whether telemetry collection is healthy.
- A local projection appears stale or incomplete.
- A user needs a structural export for offline inspection.
- A user wants to understand content capture or exporter configuration.

## Prerequisites

- The `hermes` CLI is available.
- Collection requires the bundled `observability/nemo_relay` plugin.
- OTLP and OpenInference destinations must already be approved by the user.

## How to Run

Use the `terminal` tool for every command. Start with:

```bash
hermes telemetry status
```

Use `--json` when another program will consume the result.

## Quick Reference

| Goal | Command |
|---|---|
| Check health and privacy | `hermes telemetry status` |
| Preview local aggregates | `hermes telemetry preview --days 30` |
| Rebuild the query index | `hermes telemetry rebuild` |
| Export structural events | `hermes telemetry export --out telemetry.ndjson` |
| Limit an export | `hermes telemetry export --since TIMESTAMP` |

## Procedure

1. Run `hermes telemetry status`.
2. If collection is disabled, explain the current state before changing it.
   Enable it only when the user asks, with
   `hermes plugins enable observability/nemo_relay`.
3. Treat `$HERMES_HOME/telemetry/atof` as the source of truth. Treat the
   `tel_*` tables in `state.db` as a disposable query index.
4. When ATOF files exist but projection counts are stale, run
   `hermes telemetry rebuild`, then check status again.
5. Use `hermes telemetry export` for local structural exports. Add
   `--include-content` only after confirming trajectory consent is enabled.
6. Configure OTLP and OpenInference as Relay live exporters. Store secret
   header values in environment variables referenced by `headers_env`.

## Pitfalls

- Do not delete or rewrite ATOF to repair the SQLite projection.
- Do not claim historical OTLP replay succeeded; this POC supports live
  exporters and reports replay as unavailable.
- Do not print exporter credential values.
- Do not enable content-bearing trajectories implicitly.
- Do not confuse a disabled plugin with a missing Relay dependency.

## Verification

- `hermes telemetry status` reports Relay, ATOF, privacy, and projection state.
- After rebuild, projection counts match the available structural events.
- A structural export parses as NDJSON or JSON in the requested format.
- Content export fails closed unless trajectory capture is enabled.

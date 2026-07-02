---
name: telemetry-analysis
description: Analyze Hermes Relay-backed run, model, tool, cache, latency, and failure metrics.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [analytics, insights, reliability, latency, relay]
    related_skills: [telemetry]
---

# Relay Telemetry Analysis

Prefer `hermes insights --days N` for user-facing analysis and
`hermes telemetry preview --days N --json` for machine-readable metrics.

Interpret the Relay section as runtime truth:

- Runs are correlated by ATOF parent ancestry, not synthetic child `run_id` fields.
- Cache hit rate is the share of model calls reporting non-zero cache-read tokens.
- Tool failure rate includes exceptions and returned error objects/JSON strings.
- Latencies are wall-clock scope durations; p50 describes the median and p95 the slow tail.
- SQLite is disposable. If counts look inconsistent with ATOF files, rebuild before drawing conclusions.

Call out the selected time range, filters, sample size, incomplete runs, and any
Relay health warning. Do not infer causality from a small number of failures.

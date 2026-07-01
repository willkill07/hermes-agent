---
name: telemetry-analysis
description: "Analyze runtime performance and reliability."
version: 1.0.0
author: "Will Killian (@willkill07), Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [telemetry, performance, reliability, cost]
    category: devops
    related_skills: [telemetry]
---

# Telemetry Analysis Skill

This skill interprets Hermes run, model, and tool metrics without uploading
them. It does not infer causes that the available structural data cannot show.

## When to Use

- A user asks about completion or failure rates.
- A user wants p50 or p95 runtime latency.
- A user asks about tokens, cache use, or estimated cost.
- A user wants to identify unreliable tool execution.

## Prerequisites

- Local collection has produced NeMo Relay ATOF events.
- The SQLite projection is healthy; use the `telemetry` skill if it is stale.
- Choose a time window that matches the user's question.

## How to Run

Use the `terminal` tool. For a private machine-readable preview:

```bash
hermes telemetry preview --days 30 --json
```

For telemetry alongside session and skill activity, run:

```bash
hermes insights --days 30
```

## Quick Reference

| Signal | Interpretation |
|---|---|
| Completion rate | Completed runs divided by observed runs |
| Failure rate | Failed runs divided by observed runs |
| Run p50/p95 | Typical and tail end-to-end latency |
| LLM p50/p95 | Typical and tail model-call latency |
| Tool failures | Tool calls carrying structural errors |
| Cache reads | Input tokens served from provider cache |
| Estimated cost | Non-authoritative cost estimate in USD |

## Procedure

1. Confirm the requested time window and optional platform filter.
2. Run `hermes telemetry preview --days N --json`.
3. Report `run_count` with every rate or percentile so the sample size is
   visible.
4. Separate run, LLM, and tool latency; they answer different questions.
5. Separate input, output, cache-read, and cache-write tokens.
6. Label cost as estimated unless a known source proves otherwise.
7. Use `hermes insights --days N` when session context helps interpretation.
8. For deeper analysis, export structural NDJSON and keep content disabled
   unless the question genuinely requires approved trajectories.

## Pitfalls

- A small sample can make p95 and failure rates unstable.
- Missing end events represent incomplete work, not zero latency.
- Correlation between a tool and failed runs does not establish causation.
- Aggregate token totals can hide one unusually expensive run.
- Structural telemetry cannot explain prompt semantics or response quality.

## Verification

- The stated time window and run count accompany conclusions.
- Completion, failure, and interruption are not combined silently.
- Latency units and percentile names are explicit.
- Cost is labeled estimated and cache tokens are reported separately.
- No event content or exporter secret appears in the analysis.

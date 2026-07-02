# Hermes–NeMo Relay Observability

Status: Implementation in progress on `wkk_relay-observability-parity`

## Decision

Hermes uses NeMo Relay as its required runtime observability dependency. The
trusted bundled plugin is enabled by default, while `plugins.disabled` remains
the explicit kill switch. Local structural collection is on by default;
content, trajectories, and network exporters remain opt-in.

Canonical ATOF JSONL is the durable source of truth. The `tel_*` SQLite schema
in `~/.hermes/telemetry/telemetry.db` is a versioned, disposable read model and
is transactionally rebuilt from retained ATOF when incompatible.

Historical event replay is intentionally out of scope. It is orthogonal to
Hermes analytics parity and does not justify expanding Relay's public API or
every language binding. Hermes supports safe structural or explicitly
content-bearing local file export instead.

This integration requires no NeMo Relay product changes. Hermes uses the
released Relay APIs for scopes, managed LLM/tool calls, sanitization,
subscribers, live exporters, and the single-file ATOF exporter. Hermes expands
its own UTC-dated filename before constructing that exporter and performs
retention while the exporter is closed. OTLP and OpenInference receive live
events only; Hermes does not retransmit stored events.

## Runtime and privacy contract

- Run, LLM, tool, approval, session, and delegated-subagent events preserve
  Relay UUID and `parent_uuid` lineage.
- Child calls resolve their nearest run ancestor from persisted scope ancestry;
  they never require a synthetic child `run_id`.
- Lifecycle events never include goals, prompts, messages, paths, credentials,
  or arbitrary hook arguments.
- Returned error objects and JSON strings are classified into stable statuses
  and error types without persisting error messages by default.
- Opted-in content always receives secret redaction. The default `pii` policy
  additionally covers common identifiers, payment data, authorization tokens,
  and cloud credentials through Relay sanitization boundaries.
- Default export reconstructs an allowlisted structural event. Explicit
  `--include-content` export still reapplies the configured redaction policy.

## Governance contract

- Consent is `unknown`, `granted`, or `denied`.
- Aggregate export requires both granted consent and
  `telemetry.allow_aggregate: true`.
- The pseudonymous installation UUID is generated in a mode-0600 state file.
- Aggregate schema `hermes.telemetry.aggregate/v1` contains only enumerated or
  bucketed dimensions and numeric distributions. It contains no prompts,
  responses, paths, endpoints, headers, usernames, or run/session identifiers.
- Purge is explicit and can independently reset the installation UUID.
- UTC daily ATOF rotation and retention apply at startup, every 24 hours, and
  before rebuild or export. Legacy monolithic ATOF is compacted once.

## User surfaces

The same Relay projection powers CLI and gateway `/usage` and `/insights`, the
browser Analytics page, and Electron command-center Analytics. The HTTP API is
backward-compatible: legacy fields remain, with Relay health, totals,
breakdowns, and recent runs added under `telemetry`. Clients use Relay metrics
when healthy and label the legacy fallback otherwise.

## Acceptance checklist

- [x] Trusted bundled default enablement with explicit disable precedence.
- [x] Exact required prebuilt Relay wheel dependency; initialization is
  fail-open and visible in status/logs.
- [x] Structural defaults and content/network opt-in behavior.
- [x] PII and secret redaction at capture and export boundaries.
- [x] Parent-ancestry correlation, late-parent backfill, and duplicate-safe
  projection.
- [x] Tool error classification for dictionaries, JSON strings, exceptions,
  timeouts, cancellations, and successes.
- [x] Consent, stable installation identity, aggregate export, retention,
  rebuild, and purge commands.
- [x] Relay-backed insights, usage, browser, and Electron UX with provenance
  and fallback states.
- [x] Bundled telemetry and telemetry-analysis skills.
- [x] No Relay API, binding, dispatcher, exporter, or release changes.
- [ ] Complete the existing `0.4.0` multi-platform wheel install smoke matrix.

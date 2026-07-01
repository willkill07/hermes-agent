# NeMo Relay Runtime Integration

The bundled `observability/nemo_relay` plugin makes NeMo Relay the execution
and observability boundary around Hermes Agent runs. Hermes continues to own
planning, model routing, tools, memory, approvals, and delegation. NeMo Relay
owns the reusable runtime concerns around those calls:

- One Relay agent scope for every Hermes run, closed on success, failure,
  interruption, cancellation, or timeout.
- Managed Relay execution for every LLM and tool call while the plugin is
  enabled. No separate adaptive component is required.
- Provider codecs for OpenAI Chat Completions, OpenAI/Codex Responses, and
  Anthropic Messages payloads.
- Relay-native ATOF, ATIF, OpenTelemetry, and OpenInference export paths.
- Explicit parent handles for delegated subagent runs.
- Structural telemetry by default, with opt-in content capture and mandatory
  secret redaction.

## Enable the integration

NeMo Relay is a required Hermes dependency. The plugin remains opt-in so an
operator chooses when Hermes calls should pass through the Relay runtime:

```bash
hermes plugins enable observability/nemo_relay
```

Once enabled, the safe default writes canonical ATOF events to:

```text
$HERMES_HOME/telemetry/atof/hermes-atof.jsonl
```

No prompts, model responses, raw tool arguments, or raw tool results are
recorded under the default configuration. Counts, sizes, identifiers, usage,
timings, outcomes, and parent-child relationships remain available.

## Configure exports

Configure the integration in `$HERMES_HOME/config.yaml`:

```yaml
plugins:
  enabled:
    - observability/nemo_relay

telemetry:
  capture_content: false
  content_redaction: secrets

  atof:
    enabled: true
    output_directory: null  # $HERMES_HOME/telemetry/atof
    filename: hermes-atof.jsonl
    mode: append

  atif:
    enabled: false
    output_directory: null  # $HERMES_HOME/telemetry/atif
    filename_template: hermes-atif-{session_id}.json

  export:
    otlp:
      enabled: false
      endpoint: http://localhost:4318/v1/traces
      headers_env:
        Authorization: OTEL_AUTH_HEADER

    openinference:
      enabled: false
      endpoint: http://localhost:4318/v1/traces
      headers_env: {}
```

`headers_env` maps an outbound header name to the name of an environment
variable. Secret header values never belong in `config.yaml`.

### Content capture

Set `telemetry.capture_content: true` only when the destination is approved to
receive prompts, responses, tool arguments, and tool results. Hermes still
force-redacts recognizable credentials and sensitive fields before Relay
emits an event. Structural mode is recommended for shared developer machines,
CI, and production systems without an explicit content-retention policy.

### ATIF and delegated runs

ATIF is disabled by default because trajectories commonly need content to be
useful for replay and evaluation. Relay writes one trajectory per top-level
agent scope and embeds delegated child runs in their parent trajectory.

## Advanced Relay configuration

Set `telemetry.plugins_toml` to a NeMo Relay component configuration when Relay
should own additional process-wide plugins:

```yaml
telemetry:
  plugins_toml: /etc/hermes/relay-plugins.toml
```

That configuration is initialized once per Hermes process. Exporters declared
there take precedence over the matching direct `telemetry` exporter, avoiding
duplicate output.

Use `config.yaml` for all non-secret behavior. Exporter credentials remain in
environment variables referenced by `headers_env`.

For the generic execution middleware contract, see
[`docs/middleware/README.md`](../../../docs/middleware/README.md).

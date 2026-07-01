"""NeMo Relay runtime integration for Hermes execution boundaries."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_INIT_FAILED = object()
_LOCK = threading.RLock()
_RUNTIME: "_Runtime | object | None" = None


@dataclass
class _RunState:
    session_id: str
    run_id: str
    turn_id: str
    handle: Any


@dataclass
class _SubagentParent:
    parent_session_id: str
    parent_handle: Any
    metadata: dict[str, Any]


@dataclass
class _Settings:
    plugins_config: dict[str, Any]
    capture_content: bool = False
    content_redaction: str = "secrets"


class _Runtime:
    def __init__(self, nemo_relay: Any, settings: _Settings) -> None:
        self.nemo_relay = nemo_relay
        self.settings = settings
        self.runs: dict[str, _RunState] = {}
        self.runs_by_turn: dict[str, _RunState] = {}
        self.runs_by_session: dict[str, _RunState] = {}
        self.subagent_parents: dict[str, _SubagentParent] = {}
        self.diagnostics: list[str] = []
        self._state_lock = threading.RLock()
        self._configure_plugins()
        self._configure_guardrails()

    def _configure_plugins(self) -> None:
        plugin_mod = getattr(self.nemo_relay, "plugin", None)
        initialize = getattr(plugin_mod, "initialize", None)
        if not callable(initialize):
            self.diagnostics.append("plugin.initialize unavailable")
            return
        try:
            _ensure_plugin_output_dirs(self.settings.plugins_config)
            result = _resolve_awaitable(initialize(self.settings.plugins_config))
            for diagnostic in _as_dict(result).get("diagnostics", []):
                self.diagnostics.append(str(diagnostic))
        except Exception as exc:
            self.diagnostics.append(f"plugin.initialize: {type(exc).__name__}: {exc}")
            logger.debug("NeMo Relay plugin initialization failed", exc_info=True)

    def _configure_guardrails(self) -> None:
        guardrails = getattr(self.nemo_relay, "guardrails", None)
        if guardrails is None:
            self.diagnostics.append("guardrails unavailable")
            return
        registrations = (
            (
                "register_tool_sanitize_request",
                "hermes.tool.request",
                self._sanitize_tool_request,
            ),
            (
                "register_tool_sanitize_response",
                "hermes.tool.response",
                self._sanitize_tool_response,
            ),
            (
                "register_llm_sanitize_request",
                "hermes.llm.request",
                self._sanitize_llm_request,
            ),
            (
                "register_llm_sanitize_response",
                "hermes.llm.response",
                self._sanitize_llm_response,
            ),
        )
        for method_name, name, callback in registrations:
            method = getattr(guardrails, method_name, None)
            if not callable(method):
                self.diagnostics.append(f"{method_name} unavailable")
                continue
            try:
                method(name, -10_000, callback)
            except Exception as exc:
                self.diagnostics.append(f"{name}: {type(exc).__name__}: {exc}")

    def _sanitize_llm_request(self, request: Any) -> Any:
        content = _jsonable(getattr(request, "content", {}))
        content = (
            _redact_payload(content, self.settings.content_redaction)
            if self.settings.capture_content
            else _structural_llm_request(content)
        )
        return self.nemo_relay.LLMRequest({}, content)

    def _sanitize_llm_response(self, response: Any) -> dict[str, Any]:
        payload = _jsonable(response)
        if self.settings.capture_content:
            redacted = _redact_payload(payload, self.settings.content_redaction)
            return redacted if isinstance(redacted, dict) else {"value": redacted}
        return _structural_llm_response(payload)

    def _sanitize_tool_request(self, tool_name: str, args: Any) -> Any:
        if self.settings.capture_content:
            return _redact_payload(args, self.settings.content_redaction)
        payload = _as_dict(args)
        return {
            "argument_keys": sorted(str(key) for key in payload),
            "argument_count": len(payload),
            "argument_chars": _json_size(args),
        }

    def _sanitize_tool_response(self, tool_name: str, result: Any) -> Any:
        if self.settings.capture_content:
            return _redact_payload(result, self.settings.content_redaction)
        return {
            "result_class": _tool_result_class(result),
            "result_chars": _json_size(result),
        }

    def start_run(self, kwargs: dict[str, Any]) -> _RunState:
        """Create one Relay agent scope for one Hermes run."""
        run_id = _run_id(kwargs)
        with self._state_lock:
            existing = self.runs.get(run_id)
            if existing is not None:
                return existing

            session_id = _session_id(kwargs)
            turn_id = str(kwargs.get("turn_id") or run_id)
            parent = self.subagent_parents.pop(session_id, None)
            metadata = _metadata(kwargs)
            parent_handle = None
            if parent is not None:
                parent_handle = parent.parent_handle
                metadata = {**metadata, **parent.metadata}

            handle = self.nemo_relay.scope.push(
                f"hermes.run:{run_id}",
                self.nemo_relay.ScopeType.Agent,
                handle=parent_handle,
                data={
                    "run_id": run_id,
                    "session_id": session_id,
                    "entrypoint": kwargs.get("entrypoint"),
                },
                metadata=metadata,
                input={
                    "user_message_chars": _safe_int(kwargs.get("user_message_chars"))
                },
                timestamp=_datetime_from_timestamp(kwargs.get("started_at")),
            )
            state = _RunState(
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_id,
                handle=handle,
            )
            self.runs[run_id] = state
            self.runs_by_turn[turn_id] = state
            if session_id:
                self.runs_by_session[session_id] = state
            return state

    def active_run(self, kwargs: dict[str, Any]) -> _RunState | None:
        with self._state_lock:
            for value in (
                kwargs.get("run_id"),
                kwargs.get("turn_id"),
                kwargs.get("parent_turn_id"),
            ):
                key = str(value or "")
                if key in self.runs:
                    return self.runs[key]
                if key in self.runs_by_turn:
                    return self.runs_by_turn[key]
            return self.runs_by_session.get(_session_id(kwargs))

    def end_run(self, kwargs: dict[str, Any]) -> None:
        state = self.active_run(kwargs)
        if state is None:
            return
        with self._state_lock:
            self.runs.pop(state.run_id, None)
            self.runs_by_turn.pop(state.turn_id, None)
            if self.runs_by_session.get(state.session_id) is state:
                self.runs_by_session.pop(state.session_id, None)
        self.nemo_relay.scope.pop(
            state.handle,
            output=_run_output(kwargs),
            metadata=_metadata(kwargs),
            timestamp=_datetime_from_timestamp(kwargs.get("ended_at")),
        )

    def mark(self, name: str, kwargs: dict[str, Any]) -> None:
        state = self.active_run(kwargs)
        if state is None:
            return
        self.nemo_relay.scope.event(
            name,
            handle=state.handle,
            data=_event_payload(kwargs, self.settings),
            metadata=_metadata(kwargs),
        )

    def mark_subagent_start(self, kwargs: dict[str, Any]) -> None:
        parent_state = self.active_run(kwargs)
        if parent_state is None:
            return
        metadata = _metadata(kwargs)
        child_session_id = _child_session_id(kwargs)
        if child_session_id:
            with self._state_lock:
                self.subagent_parents[child_session_id] = _SubagentParent(
                    parent_session_id=parent_state.session_id,
                    parent_handle=parent_state.handle,
                    metadata=_subagent_child_metadata(kwargs, metadata),
                )
        self.nemo_relay.scope.event(
            "hermes.subagent.start",
            handle=parent_state.handle,
            data=_event_payload(kwargs, self.settings),
            metadata=metadata,
        )

    def mark_subagent_stop(self, kwargs: dict[str, Any]) -> None:
        child_session_id = _child_session_id(kwargs)
        if child_session_id:
            with self._state_lock:
                self.subagent_parents.pop(child_session_id, None)
        self.mark("hermes.subagent.stop", kwargs)

    def managed_llm_enabled(self) -> bool:
        return callable(
            getattr(getattr(self.nemo_relay, "llm", None), "execute", None)
        ) and callable(getattr(self.nemo_relay, "LLMRequest", None))

    def managed_tool_enabled(self) -> bool:
        return callable(
            getattr(getattr(self.nemo_relay, "tools", None), "execute", None)
        )

    def _run_managed_with_downstream_preservation(
        self,
        next_call: Callable[[Any], Any],
        normalize_payload: Callable[[Any], Any],
        shape_response: Callable[[Any], Any],
        make_managed_execute: Callable[[Callable[[Any], Any]], Any],
    ) -> Any:
        raw_response: dict[str, Any] = {"set": False, "value": None}
        callback_error: Exception | None = None
        downstream_error: BaseException | None = None

        def _impl(next_payload: Any) -> Any:
            nonlocal callback_error, downstream_error
            try:
                raw = next_call(normalize_payload(next_payload))
            except Exception as exc:
                callback_error = exc
                downstream_error = _original_downstream_error(exc)
                raise
            raw_response["set"] = True
            raw_response["value"] = raw
            return shape_response(raw)

        try:
            managed_result = _resolve_awaitable(make_managed_execute(_impl))
        except Exception as exc:
            if downstream_error is not None and _is_relay_wrapped_callback_error(
                exc, callback_error
            ):
                raise downstream_error
            raise
        return raw_response["value"] if raw_response["set"] else managed_result

    def execute_llm(self, kwargs: dict[str, Any]) -> Any:
        state = self.active_run(kwargs)
        next_call = kwargs.get("next_call")
        request_body = _jsonable(kwargs.get("request") or {})
        if state is None or not callable(next_call):
            return next_call(request_body) if callable(next_call) else request_body

        request = self.nemo_relay.LLMRequest({}, request_body)
        codec = _llm_codec(
            self.nemo_relay,
            kwargs.get("api_mode"),
            kwargs.get("provider"),
        )

        def _normalize(next_request: Any) -> Any:
            next_body = getattr(next_request, "content", next_request)
            return next_body if isinstance(next_body, dict) else request_body

        def _make_managed(impl: Callable[[Any], Any]) -> Any:
            async def _managed_execute() -> Any:
                result = self.nemo_relay.llm.execute(
                    str(kwargs.get("provider") or "llm"),
                    request,
                    impl,
                    handle=state.handle,
                    data={
                        "turn_id": kwargs.get("turn_id"),
                        "api_request_id": kwargs.get("api_request_id"),
                        "api_call_count": kwargs.get("api_call_count"),
                    },
                    metadata=_metadata(kwargs),
                    model_name=str(kwargs.get("model") or ""),
                    codec=codec,
                    response_codec=codec if self.settings.capture_content else None,
                )
                return await result if inspect.isawaitable(result) else result

            return _managed_execute()

        return self._run_managed_with_downstream_preservation(
            next_call,
            _normalize,
            _llm_response_payload,
            _make_managed,
        )

    def execute_tool(self, kwargs: dict[str, Any]) -> Any:
        state = self.active_run(kwargs)
        next_call = kwargs.get("next_call")
        args = _jsonable(kwargs.get("args") or {})
        if state is None or not callable(next_call):
            return next_call(args) if callable(next_call) else args

        tool_name = str(kwargs.get("tool_name") or "tool")

        def _normalize(next_args: Any) -> Any:
            return next_args if isinstance(next_args, dict) else args

        def _make_managed(impl: Callable[[Any], Any]) -> Any:
            async def _managed_execute() -> Any:
                result = self.nemo_relay.tools.execute(
                    tool_name,
                    args,
                    impl,
                    handle=state.handle,
                    data={
                        "turn_id": kwargs.get("turn_id"),
                        "api_request_id": kwargs.get("api_request_id"),
                        "tool_call_id": kwargs.get("tool_call_id"),
                    },
                    metadata=_metadata(kwargs),
                )
                return await result if inspect.isawaitable(result) else result

            return _managed_execute()

        return self._run_managed_with_downstream_preservation(
            next_call,
            _normalize,
            _jsonable,
            _make_managed,
        )


def register(ctx) -> None:
    ctx.register_hook("on_run_start", on_run_start)
    ctx.register_hook("on_run_end", on_run_end)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("pre_approval_request", on_pre_approval_request)
    ctx.register_hook("post_approval_response", on_post_approval_response)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
    ctx.register_middleware("llm_execution", on_llm_execution_middleware)
    ctx.register_middleware("tool_execution", on_tool_execution_middleware)


def on_run_start(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.start_run(kwargs))


def on_run_end(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.end_run(kwargs))


def on_session_end(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.session.end", kwargs))


def on_pre_llm_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.turn.start", kwargs))


def on_post_llm_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.turn.end", kwargs))


def on_api_request_error(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.api.error", kwargs))


def on_pre_approval_request(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.approval.request", kwargs))


def on_post_approval_response(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.approval.response", kwargs))


def on_subagent_start(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark_subagent_start(kwargs))


def on_subagent_stop(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark_subagent_stop(kwargs))


def on_llm_execution_middleware(**kwargs: Any) -> Any:
    runtime = _get_runtime()
    next_call = kwargs.get("next_call")
    request = kwargs.get("request") or {}
    if runtime is not None and runtime.managed_llm_enabled():
        return runtime.execute_llm(kwargs)
    return next_call(request) if callable(next_call) else request


def on_tool_execution_middleware(**kwargs: Any) -> Any:
    runtime = _get_runtime()
    next_call = kwargs.get("next_call")
    args = kwargs.get("args") or {}
    if runtime is not None and runtime.managed_tool_enabled():
        return runtime.execute_tool(kwargs)
    return next_call(args) if callable(next_call) else args


def _get_runtime() -> Optional[_Runtime]:
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is _INIT_FAILED:
            return None
        if isinstance(_RUNTIME, _Runtime):
            return _RUNTIME
        try:
            import nemo_relay as nemo_runtime
        except Exception as exc:
            logger.debug("NeMo Relay plugin disabled: import failed: %s", exc)
            _RUNTIME = _INIT_FAILED
            return None
        try:
            _RUNTIME = _Runtime(nemo_runtime, _load_settings())
        except Exception as exc:
            logger.debug(
                "NeMo Relay plugin disabled: init failed: %s", exc, exc_info=True
            )
            _RUNTIME = _INIT_FAILED
            return None
        return _RUNTIME


def _load_settings() -> _Settings:
    try:
        from hermes_cli.config import load_config
        from hermes_constants import get_hermes_home

        telemetry = _as_dict(load_config().get("telemetry"))
        telemetry_root = get_hermes_home() / "telemetry"
    except Exception:
        logger.debug("Hermes telemetry config load failed", exc_info=True)
        telemetry = {}
        telemetry_root = Path.home() / ".hermes" / "telemetry"

    plugins_toml = str(telemetry.get("plugins_toml") or "").strip()
    generated_config = _generated_plugins_config(telemetry, telemetry_root)
    plugins_config = _merge_plugins_config(
        generated_config,
        _load_plugins_config(plugins_toml),
    )
    return _Settings(
        plugins_config=plugins_config,
        capture_content=bool(telemetry.get("capture_content", False)),
        content_redaction=str(telemetry.get("content_redaction") or "secrets"),
    )


def _load_plugins_config(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("NeMo Relay plugins.toml load failed: %s", exc, exc_info=True)
        return None


def _merge_plugins_config(
    generated: dict[str, Any], configured: dict[str, Any] | None
) -> dict[str, Any]:
    """Keep Hermes observability unless plugins.toml explicitly replaces it."""
    if not configured:
        return generated
    components = configured.get("components")
    if not isinstance(components, list):
        components = []
    if any(
        isinstance(component, dict) and component.get("kind") == "observability"
        for component in components
    ):
        return configured
    return {
        **configured,
        "version": configured.get("version", 1),
        "components": [*generated["components"], *components],
    }


def _resolved_export_config(raw: dict[str, Any]) -> dict[str, Any]:
    config = {
        "enabled": bool(raw.get("enabled", False)),
        "endpoint": raw.get("endpoint"),
        "service_name": "hermes-agent",
    }
    headers = {
        str(header): value
        for header, env_name in _as_dict(raw.get("headers_env")).items()
        if (value := os.environ.get(str(env_name)))
    }
    if headers:
        config["headers"] = headers
    return config


def _generated_plugins_config(
    telemetry: dict[str, Any], telemetry_root: Path
) -> dict[str, Any]:
    """Translate Hermes telemetry settings into Relay's native plugin config."""
    atof = _as_dict(telemetry.get("atof"))
    atif = _as_dict(telemetry.get("atif"))
    exporters = _as_dict(telemetry.get("export"))
    observability = {
        "version": 1,
        "atof": {
            "enabled": bool(atof.get("enabled", True)),
            "output_directory": str(
                atof.get("output_directory") or telemetry_root / "atof"
            ),
            "filename": str(atof.get("filename") or "hermes-atof.jsonl"),
            "mode": str(atof.get("mode") or "append"),
        },
        "atif": {
            "enabled": bool(atif.get("enabled", False)),
            "output_directory": str(
                atif.get("output_directory") or telemetry_root / "atif"
            ),
            "filename_template": str(
                atif.get("filename_template") or "hermes-atif-{session_id}.json"
            ),
            "agent_name": "Hermes Agent",
            "model_name": "unknown",
            "extra": {"source": "hermes-agent"},
        },
        "opentelemetry": _resolved_export_config(_as_dict(exporters.get("otlp"))),
        "openinference": _resolved_export_config(
            _as_dict(exporters.get("openinference"))
        ),
    }
    return {
        "version": 1,
        "components": [
            {"kind": "observability", "enabled": True, "config": observability}
        ],
    }


def _ensure_plugin_output_dirs(config: dict[str, Any]) -> None:
    for component in config.get("components", []):
        if not isinstance(component, dict) or component.get("kind") != "observability":
            continue
        component_config = _as_dict(component.get("config"))
        for name in ("atof", "atif"):
            exporter = _as_dict(component_config.get(name))
            if exporter.get("enabled") is False:
                continue
            output_directory = exporter.get("output_directory")
            if isinstance(output_directory, str) and output_directory.strip():
                Path(output_directory).mkdir(parents=True, exist_ok=True)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _datetime_from_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _run_id(kwargs: dict[str, Any]) -> str:
    return str(
        kwargs.get("run_id") or kwargs.get("turn_id") or f"{_session_id(kwargs)}-run"
    )


def _run_output(kwargs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "outcome",
        "completed",
        "failed",
        "interrupted",
        "turn_exit_reason",
        "api_calls",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "estimated_cost_usd",
        "cost_status",
        "cost_source",
        "error_type",
    )
    return {key: _jsonable(kwargs[key]) for key in keys if kwargs.get(key) is not None}


def _llm_codec(nemo_relay: Any, api_mode: Any, provider: Any) -> Any:
    mode = str(api_mode or "").lower()
    provider_name = str(provider or "").lower()
    if "anthropic" in mode or (not mode and provider_name == "anthropic"):
        class_name = "AnthropicMessagesCodec"
    elif "response" in mode or "codex" in mode:
        class_name = "OpenAIResponsesCodec"
    elif "chat" in mode or mode in {"", "openai"}:
        class_name = "OpenAIChatCodec"
    else:
        return None
    codecs = getattr(nemo_relay, "codecs", nemo_relay)
    codec_type = getattr(codecs, class_name, None)
    return codec_type() if callable(codec_type) else None


def _json_size(value: Any) -> int:
    try:
        return len(
            json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
        )
    except Exception:
        return len(str(value))


def _message_content_chars(message: Any) -> int:
    content = _as_dict(message).get("content")
    if isinstance(content, str):
        return len(content)
    return _json_size(content) if content is not None else 0


def _structural_llm_request(payload: Any) -> dict[str, Any]:
    body = _as_dict(payload)
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    roles: dict[str, int] = {}
    for message in messages:
        role = str(_as_dict(message).get("role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
    structural = {
        "model": body.get("model"),
        "message_count": len(messages),
        "message_roles": roles,
        "message_chars": sum(_message_content_chars(message) for message in messages),
        "tool_definition_count": len(body.get("tools") or []),
    }
    for key in ("max_tokens", "max_output_tokens", "temperature", "stream"):
        if body.get(key) is not None:
            structural[key] = body[key]
    return structural


def _structural_llm_response(payload: Any) -> dict[str, Any]:
    body = _as_dict(payload)
    normalized = _as_dict(_llm_response_payload(body))
    assistant = _as_dict(normalized.get("assistant_message"))
    content = assistant.get("content")
    return {
        "model": normalized.get("model") or body.get("model"),
        "finish_reason": normalized.get("finish_reason"),
        "usage": normalized.get("usage"),
        "content_chars": len(content)
        if isinstance(content, str)
        else _json_size(content),
        "tool_call_count": len(assistant.get("tool_calls") or []),
        "has_reasoning": bool(assistant.get("reasoning_content")),
    }


def _tool_result_class(result: Any) -> str:
    if result is None:
        return "null"
    if isinstance(result, dict):
        return "object"
    if isinstance(result, (list, tuple)):
        return "array"
    if isinstance(result, bool):
        return "boolean"
    if isinstance(result, (int, float)):
        return "number"
    return "string"


_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "credential",
    "id_token",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}


def _redact_payload(value: Any, mode: str = "secrets") -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "«redacted-secret»"
                if str(key).lower() in _SENSITIVE_KEYS
                else _redact_payload(item, mode)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact_payload(item, mode) for item in value]
    if not isinstance(value, str):
        return _jsonable(value)
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)
    except Exception:
        return "[redaction-unavailable]"


def _event_payload(kwargs: dict[str, Any], settings: _Settings) -> dict[str, Any]:
    if settings.capture_content:
        redacted = _redact_payload(_jsonable(kwargs), settings.content_redaction)
        return redacted if isinstance(redacted, dict) else {"value": redacted}
    allowed = (
        "run_id",
        "turn_id",
        "task_id",
        "session_id",
        "platform",
        "entrypoint",
        "provider",
        "model",
        "api_mode",
        "api_request_id",
        "api_call_count",
        "tool_name",
        "tool_call_id",
        "approval_id",
        "approved",
        "status",
        "reason",
        "duration_ms",
        "retryable",
        "parent_session_id",
        "parent_turn_id",
        "child_session_id",
        "child_subagent_id",
        "child_role",
        "child_status",
    )
    return {
        key: _jsonable(kwargs[key]) for key in allowed if kwargs.get(key) is not None
    }


def _session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("session_id") or kwargs.get("parent_session_id") or "")


def _child_session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("child_session_id") or "")


def _subagent_child_metadata(
    kwargs: dict[str, Any], parent_metadata: dict[str, Any]
) -> dict[str, Any]:
    child_session_id = _child_session_id(kwargs)
    metadata = {
        "session_id": child_session_id,
        "trajectory_id": child_session_id,
        "nemo_relay_scope_role": "subagent",
    }
    for target, source in (
        ("subagent_id", "child_subagent_id"),
        ("child_session_id", "child_session_id"),
        ("child_subagent_id", "child_subagent_id"),
        ("child_role", "child_role"),
        ("parent_session_id", "parent_session_id"),
        ("parent_turn_id", "parent_turn_id"),
        ("parent_subagent_id", "parent_subagent_id"),
        ("parent_trajectory_id", "parent_trajectory_id"),
    ):
        value = parent_metadata.get(source)
        if value is not None:
            metadata[target] = value
    return metadata


def _metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "session_id",
        "platform",
        "entrypoint",
        "task_id",
        "turn_id",
        "api_request_id",
        "tool_call_id",
        "parent_session_id",
        "parent_turn_id",
        "parent_subagent_id",
        "child_session_id",
        "child_subagent_id",
        "child_role",
        "child_status",
        "provider",
        "model",
        "api_mode",
        "is_delegated",
        "outcome",
        "error_type",
        "status",
        "reason",
    )
    metadata = {
        key: _jsonable(kwargs[key])
        for key in keys
        if key in kwargs and kwargs[key] is not None
    }
    if "session_id" in metadata:
        metadata.setdefault("trajectory_id", metadata["session_id"])
    if "parent_session_id" in metadata:
        metadata.setdefault("parent_trajectory_id", metadata["parent_session_id"])
    if "child_session_id" in metadata:
        metadata.setdefault("child_trajectory_id", metadata["child_session_id"])
    return metadata


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    try:
        if hasattr(value, "model_dump"):
            return _jsonable(value.model_dump(mode="json"))
    except Exception:
        pass
    try:
        if hasattr(value, "__dict__"):
            return _jsonable(vars(value))
    except Exception:
        pass
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    return (
        obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    )


def _original_downstream_error(exc: Exception) -> BaseException:
    original = getattr(exc, "original", None)
    if exc.__class__.__name__ == "_DownstreamExecutionError" and isinstance(
        original, BaseException
    ):
        return original
    return exc


def _is_relay_wrapped_callback_error(
    exc: Exception, callback_error: Exception | None
) -> bool:
    if callback_error is None or not isinstance(exc, RuntimeError):
        return False
    expected = f"internal error: {callback_error.__class__.__name__}: {callback_error}"
    return str(exc).startswith(expected)


def _llm_response_payload(response: Any) -> Any:
    """Return the LLM response shape Relay's codecs and ATIF expect."""
    payload = _jsonable(response)
    if isinstance(payload, dict) and "assistant_message" in payload:
        return payload

    choices = _value(response, "choices")
    if choices is None and isinstance(payload, dict):
        choices = payload.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    message = _value(first_choice, "message")
    assistant_message: dict[str, Any] = {"role": "assistant", "content": ""}
    if message is not None:
        assistant_message["role"] = _value(message, "role", "assistant") or "assistant"
        content = _value(message, "content")
        if content is not None:
            assistant_message["content"] = _jsonable(content)
        tool_calls = _tool_calls_payload(_value(message, "tool_calls"))
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        reasoning = _value(message, "reasoning_content")
        if reasoning is not None:
            assistant_message["reasoning_content"] = _jsonable(reasoning)
    elif isinstance(payload, dict):
        assistant_message["content"] = (
            payload.get("content") or payload.get("output_text") or ""
        )

    return {
        "model": _value(
            response,
            "model",
            payload.get("model") if isinstance(payload, dict) else None,
        ),
        "assistant_message": assistant_message,
        "finish_reason": _value(first_choice, "finish_reason"),
        "usage": _jsonable(
            _value(
                response,
                "usage",
                payload.get("usage") if isinstance(payload, dict) else None,
            )
        ),
    }


def _tool_calls_payload(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        function = _value(call, "function")
        normalized.append({
            "id": _value(call, "id"),
            "type": _value(call, "type", "function") or "function",
            "function": {
                "name": _value(function, "name"),
                "arguments": _value(function, "arguments"),
            },
        })
    return normalized


def _safe(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception as exc:
        logger.debug("NeMo Relay hook handling failed: %s", exc, exc_info=True)


def _resolve_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(value)
        except BaseException as exc:  # pragma: no cover - re-raised below
            error["exc"] = exc

    thread = threading.Thread(
        target=_runner,
        name="hermes-nemo-relay-awaitable",
        daemon=True,
    )
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def reset_for_tests() -> None:
    global _RUNTIME
    with _LOCK:
        _RUNTIME = None

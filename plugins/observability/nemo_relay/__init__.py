"""nemo_relay — optional Hermes plugin for NeMo Relay observability."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import threading
import time
import tomllib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_INIT_FAILED = object()
_LOCK = threading.RLock()
_RUNTIME: "_Runtime | object | None" = None

_RELAY_PII_PATTERN = (
    r"(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|\+?[0-9][0-9()\-\s]{6,}[0-9]"
    r"|\b(?:\d{1,3}\.){3}\d{1,3}\b"
    r"|(?i:\bBearer\s+[A-Za-z0-9._~+/\-]{12,}={0,2}\b)"
    r"|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
    r"|\b(?:\d[ -]?){13,19}\b"
    r"|\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AIPA|ANPA|ANVA|APKA|AROA|AUSA)[A-Z0-9]{16}\b"
    r"|\bAIza[0-9A-Za-z\-_]{35}\b"
    r"|\b[A-Za-z0-9+/]{86}==)"
)


@dataclass
class _SessionState:
    session_id: str
    handle: Any = None
    atif_exporter: Any = None
    atif_subscriber_name: str = ""
    is_embedded_subagent: bool = False
    parent_session_id: str = ""
    llm_spans: dict[str, Any] = field(default_factory=dict)
    tool_spans: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SubagentParent:
    parent_session_id: str
    parent_handle: Any
    metadata: dict[str, Any]


@dataclass
class _RunState:
    run_id: str
    session_id: str
    handle: Any


@dataclass
class _Settings:
    plugins_toml_path: str = ""
    plugins_config: dict[str, Any] | None = None
    adaptive_enabled: bool = False
    adaptive_mode: str = "observe_only"
    atof_enabled: bool = False
    atof_output_directory: str = ""
    atof_mode: str = "append"
    atif_enabled: bool = False
    atif_output_directory: str = ""
    atif_filename_template: str = "hermes-atif-{session_id}.json"
    atif_subagent_export_mode: str = "embedded"
    atif_agent_name: str = "Hermes Agent"
    atif_agent_version: str = "unknown"
    atif_model_name: str = "unknown"
    local_enabled: bool = True
    capture_content: bool = False
    content_redaction: str = "pii"
    atof_filename_template: str = "hermes-atof-{date}.jsonl"


class _Runtime:
    def __init__(self, nemo_relay: Any, settings: _Settings) -> None:
        self.nemo_relay = nemo_relay
        self.settings = settings
        self.sessions: dict[str, _SessionState] = {}
        self.runs: dict[str, _RunState] = {}
        self.active_runs_by_session: dict[str, str] = {}
        self.subagent_parents: dict[str, _SubagentParent] = {}
        self.atof_exporter: Any = None
        self._last_maintenance = 0.0
        self._atof_subscriber_name = "hermes.nemo_relay.atof"
        self._maintain_local_storage(force=True)
        self._plugin_config_initialized = self._configure_plugins_toml()
        self._plugin_config_needs_reinit = False
        if not self._plugins_toml_owns_exporter("atof"):
            self._activate_direct_fallbacks()
        self._configure_guardrails()
        self._configure_projection()

    def _configure_projection(self) -> None:
        if not self.settings.local_enabled:
            return
        register = getattr(
            getattr(self.nemo_relay, "subscribers", None), "register", None
        )
        if not callable(register):
            return
        from hermes_cli.telemetry import TelemetryProjection

        self.projection = TelemetryProjection()
        register("hermes.telemetry.projector", self.projection.consume)

    def _maintain_local_storage(self, *, force: bool = False) -> None:
        if not self.settings.local_enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_maintenance < 24 * 60 * 60:
            return
        direct_atof_active = self.atof_exporter is not None
        try:
            if direct_atof_active:
                force_flush = getattr(self.atof_exporter, "force_flush", None)
                if callable(force_flush):
                    force_flush()
                self._clear_atof()
            from hermes_cli.telemetry import enforce_retention

            enforce_retention()
            self._last_maintenance = now
        except Exception:
            logger.warning("NeMo Relay telemetry retention failed", exc_info=True)
        finally:
            if direct_atof_active:
                self._configure_atof()

    def _configure_guardrails(self) -> None:
        guardrails = getattr(self.nemo_relay, "guardrails", None)
        if guardrails is None:
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
            if callable(method):
                method(name, -10_000, callback)

    def _sanitize_llm_request(self, request: Any) -> Any:
        content = _jsonable(getattr(request, "content", {}))
        if isinstance(content, dict) and {
            "message_count",
            "tool_count",
            "request_chars",
        }.issubset(content):
            return self.nemo_relay.LLMRequest({}, content)
        content = (
            _redact_payload(content, self.settings.content_redaction)
            if self.settings.capture_content
            else _structural_llm_request(content)
        )
        return self.nemo_relay.LLMRequest({}, content)

    def _sanitize_llm_response(self, response: Any) -> dict[str, Any]:
        payload = _jsonable(response)
        if isinstance(payload, dict) and {"usage", "response_chars"}.issubset(payload):
            return payload
        if self.settings.capture_content:
            redacted = _redact_payload(payload, self.settings.content_redaction)
            return redacted if isinstance(redacted, dict) else {"value": redacted}
        return _structural_llm_response(payload)

    def _sanitize_tool_request(self, tool_name: str, args: Any) -> Any:
        if isinstance(args, dict) and {
            "argument_keys",
            "argument_count",
            "argument_chars",
        }.issubset(args):
            return args
        if self.settings.capture_content:
            return _redact_payload(args, self.settings.content_redaction)
        payload = args if isinstance(args, dict) else {}
        return {
            "argument_keys": sorted(str(key) for key in payload),
            "argument_count": len(payload),
            "argument_chars": _json_size(args),
        }

    def _sanitize_tool_response(self, tool_name: str, result: Any) -> Any:
        if (
            isinstance(result, dict)
            and "status" in result
            and ("error_type" in result or "result_class" in result)
        ):
            return _redact_payload(result, self.settings.content_redaction)
        status, error_type, _ = _tool_result_fields(result)
        if self.settings.capture_content:
            payload = _redact_payload(result, self.settings.content_redaction)
            return {"result": payload, "status": status, "error_type": error_type}
        return {
            "result_class": _tool_result_class(result),
            "result_chars": _json_size(result),
            "status": status,
            "error_type": error_type,
        }

    def _configure_plugins_toml(self) -> bool:
        if not self.settings.plugins_config:
            return False
        plugin_mod = getattr(self.nemo_relay, "plugin", None)
        initialize = getattr(plugin_mod, "initialize", None)
        if not callable(initialize):
            return False
        try:
            self._ensure_plugin_config_output_dirs(self.settings.plugins_config)
            _resolve_awaitable(initialize(self.settings.plugins_config))
            return True
        except Exception as exc:
            logger.debug("NeMo Relay plugins.toml init failed: %s", exc, exc_info=True)
            return False

    def _clear_plugins_toml(self) -> None:
        if not self._plugin_config_initialized:
            return
        plugin_mod = getattr(self.nemo_relay, "plugin", None)
        clear = getattr(plugin_mod, "clear", None)
        if not callable(clear):
            return
        try:
            _resolve_awaitable(clear())
        finally:
            self._plugin_config_initialized = False
            self._plugin_config_needs_reinit = bool(self.settings.plugins_config)

    def _activate_direct_fallbacks(self) -> None:
        self._plugin_config_needs_reinit = False
        self._configure_atof()

    def _maybe_reinitialize_plugins_toml(self) -> None:
        if not self._plugin_config_needs_reinit or self._plugin_config_initialized:
            return
        self._plugin_config_initialized = self._configure_plugins_toml()
        if not self._plugin_config_initialized:
            self._activate_direct_fallbacks()
            return
        if self._plugins_toml_owns_exporter("atof"):
            self._clear_atof()
        else:
            self._configure_atof()
        self._plugin_config_needs_reinit = False

    def _plugins_toml_owns_exporter(self, exporter_name: str) -> bool:
        return self._plugin_config_initialized and _observability_exporter_enabled(
            self.settings.plugins_config,
            exporter_name,
        )

    def _ensure_plugin_config_output_dirs(self, config: dict[str, Any]) -> None:
        for component in config.get("components", []):
            if not isinstance(component, dict):
                continue
            if component.get("kind") != "observability":
                continue
            if component.get("enabled") is False:
                continue
            component_config = component.get("config")
            if not isinstance(component_config, dict):
                continue
            for exporter_name in ("atof", "atif"):
                exporter_config = component_config.get(exporter_name)
                if not isinstance(exporter_config, dict):
                    continue
                output_directory = exporter_config.get("output_directory")
                if isinstance(output_directory, str) and output_directory.strip():
                    Path(output_directory).mkdir(parents=True, exist_ok=True)

    def _configure_atof(self) -> None:
        if (
            not self.settings.local_enabled
            or not self.settings.atof_enabled
            or self.atof_exporter is not None
        ):
            return
        config = self.nemo_relay.AtofExporterConfig()
        if self.settings.atof_output_directory:
            Path(self.settings.atof_output_directory).mkdir(parents=True, exist_ok=True)
            config.output_directory = self.settings.atof_output_directory
        config.filename = self.settings.atof_filename_template.format(
            date=time.strftime("%Y-%m-%d", time.gmtime())
        )
        if self.settings.atof_mode.lower() == "overwrite":
            config.mode = self.nemo_relay.AtofExporterMode.Overwrite
        else:
            config.mode = self.nemo_relay.AtofExporterMode.Append
        self.atof_exporter = self.nemo_relay.AtofExporter(config)
        self.atof_exporter.register(self._atof_subscriber_name)

    def _clear_atof(self) -> None:
        if self.atof_exporter is None:
            return
        deregister = getattr(self.atof_exporter, "deregister", None)
        if callable(deregister):
            try:
                deregister(self._atof_subscriber_name)
            except Exception:
                logger.debug("NeMo Relay ATOF deregister failed", exc_info=True)
        self.atof_exporter = None

    def ensure_session(self, kwargs: dict[str, Any]) -> _SessionState:
        self._maybe_reinitialize_plugins_toml()
        session_id = _session_id(kwargs)
        state = self.sessions.get(session_id)
        if state is not None:
            return state

        state = _SessionState(session_id=session_id)
        if self.settings.atif_enabled and not self._plugins_toml_owns_exporter("atif"):
            state.atif_exporter = self.nemo_relay.AtifExporter(
                session_id,
                self.settings.atif_agent_name,
                self.settings.atif_agent_version,
                model_name=str(kwargs.get("model") or self.settings.atif_model_name),
                extra={"source": "hermes-agent", "plugin": "observability/nemo_relay"},
            )
            state.atif_subscriber_name = f"hermes.nemo_relay.atif.{session_id}"
            state.atif_exporter.register(state.atif_subscriber_name)

        subagent_parent = self.subagent_parents.get(session_id)
        metadata = _metadata(kwargs)
        parent_handle = None
        if subagent_parent is not None:
            parent_handle = subagent_parent.parent_handle
            metadata = {**metadata, **subagent_parent.metadata}
            state.is_embedded_subagent = True
            state.parent_session_id = subagent_parent.parent_session_id

        state.handle = self.nemo_relay.scope.push(
            f"hermes-session-{session_id}",
            self.nemo_relay.ScopeType.Agent,
            handle=parent_handle,
            data={"session_id": session_id},
            metadata=metadata,
        )
        self.sessions[session_id] = state
        return state

    def start_run(self, kwargs: dict[str, Any]) -> _RunState:
        self._maintain_local_storage()
        run_id = str(kwargs.get("run_id") or uuid.uuid4().hex)
        existing = self.runs.get(run_id)
        if existing is not None:
            return existing
        session = self.ensure_session(kwargs)
        handle = self.nemo_relay.scope.push(
            f"hermes.run:{run_id}",
            self.nemo_relay.ScopeType.Agent,
            handle=session.handle,
            data={
                "run_id": run_id,
                "session_id": session.session_id,
                "entrypoint": kwargs.get("entrypoint"),
            },
            metadata=_metadata(kwargs),
        )
        state = _RunState(run_id=run_id, session_id=session.session_id, handle=handle)
        self.runs[run_id] = state
        self.active_runs_by_session[session.session_id] = run_id
        return state

    def end_run(self, kwargs: dict[str, Any]) -> None:
        run_id = str(kwargs.get("run_id") or "")
        state = self.runs.pop(run_id, None)
        if state is None:
            return
        if self.active_runs_by_session.get(state.session_id) == run_id:
            self.active_runs_by_session.pop(state.session_id, None)
        self.nemo_relay.scope.pop(
            state.handle, output=_event_payload(kwargs), metadata=_metadata(kwargs)
        )

    def active_handle(self, kwargs: dict[str, Any], session: _SessionState) -> Any:
        run_id = str(kwargs.get("run_id") or "") or self.active_runs_by_session.get(
            session.session_id, ""
        )
        run = self.runs.get(run_id)
        return run.handle if run is not None else session.handle

    def export_atif(self, state: _SessionState) -> None:
        if not self.settings.atif_enabled or state.atif_exporter is None:
            return
        if (
            state.is_embedded_subagent
            and self.settings.atif_subagent_export_mode != "all"
        ):
            return
        output_dir = self.settings.atif_output_directory
        if not output_dir:
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = self.settings.atif_filename_template.format(
            session_id=state.session_id
        )
        Path(output_dir, filename).write_text(
            state.atif_exporter.export_json(), encoding="utf-8"
        )

    def close_session(self, kwargs: dict[str, Any]) -> None:
        session_id = _session_id(kwargs)
        self.subagent_parents.pop(session_id, None)
        state = self.sessions.pop(session_id, None)
        if state is None:
            return
        if state.handle is not None:
            try:
                self.nemo_relay.scope.pop(state.handle, output=_event_payload(kwargs))
            except Exception:
                logger.debug("NeMo Relay session pop failed", exc_info=True)
        flush = getattr(getattr(self.nemo_relay, "subscribers", None), "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                logger.warning("NeMo Relay subscriber flush failed", exc_info=True)
        self.export_atif(state)
        if state.atif_exporter is not None and state.atif_subscriber_name:
            try:
                state.atif_exporter.deregister(state.atif_subscriber_name)
            except Exception:
                logger.debug("NeMo Relay ATIF deregister failed", exc_info=True)
        if self._plugin_config_initialized and not self.sessions:
            try:
                self._clear_plugins_toml()
            except Exception:
                logger.debug("NeMo Relay plugins.toml clear failed", exc_info=True)
        elif self.settings.plugins_config and not self.sessions:
            self._plugin_config_needs_reinit = True
        if not self.sessions and getattr(self, "projection", None) is not None:
            self.projection.close()

    def mark(self, name: str, kwargs: dict[str, Any]) -> None:
        state = self.ensure_session(kwargs)
        self.nemo_relay.scope.event(
            name,
            handle=self.active_handle(kwargs, state),
            data=_event_payload(kwargs),
            metadata=_metadata(kwargs),
        )

    def mark_subagent_start(self, kwargs: dict[str, Any]) -> None:
        parent_state = self.ensure_session(kwargs)
        metadata = _metadata(kwargs)
        child_session_id = _child_session_id(kwargs)
        if child_session_id:
            self.subagent_parents[child_session_id] = _SubagentParent(
                parent_session_id=parent_state.session_id,
                parent_handle=parent_state.handle,
                metadata=_subagent_child_metadata(kwargs, metadata),
            )
        self.nemo_relay.scope.event(
            "hermes.subagent.start",
            handle=self.active_handle(kwargs, parent_state),
            data=_event_payload(kwargs),
            metadata=metadata,
        )

    def mark_subagent_stop(self, kwargs: dict[str, Any]) -> None:
        child_session_id = _child_session_id(kwargs)
        if child_session_id:
            self.subagent_parents.pop(child_session_id, None)
        self.mark("hermes.subagent.stop", kwargs)

    def managed_llm_enabled(self) -> bool:
        return (
            self.settings.adaptive_enabled
            and callable(
                getattr(getattr(self.nemo_relay, "llm", None), "execute", None)
            )
            and callable(getattr(self.nemo_relay, "LLMRequest", None))
        )

    def managed_tool_enabled(self) -> bool:
        return self.settings.adaptive_enabled and callable(
            getattr(getattr(self.nemo_relay, "tools", None), "execute", None)
        )

    def _run_managed_with_downstream_preservation(
        self,
        next_call: Callable[[Any], Any],
        normalize_payload: Callable[[Any], Any],
        shape_response: Callable[[Any], Any],
        make_managed_execute: Callable[[Callable[[Any], Any]], Any],
    ) -> Any:
        # NeMo Relay's native managed execution may wrap a failing callback as an
        # internal runtime error, hiding the real downstream provider/tool
        # exception. Capture the original here and re-raise it after managed
        # execution so Hermes retry classification still sees it. The LLM and tool
        # paths share this scaffolding; they differ only in payload normalization,
        # response shaping, and the Relay call itself.
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
        state = self.ensure_session(kwargs)
        request_body = _jsonable(kwargs.get("request") or {})
        request = self.nemo_relay.LLMRequest({}, request_body)
        next_call = kwargs.get("next_call")
        if not callable(next_call):
            return request_body

        def _normalize(next_request: Any) -> Any:
            next_body = getattr(next_request, "content", next_request)
            return next_body if isinstance(next_body, dict) else request_body

        def _make_managed(impl: Callable[[Any], Any]) -> Any:
            async def _managed_execute() -> Any:
                result = self.nemo_relay.llm.execute(
                    str(kwargs.get("provider") or "llm"),
                    request,
                    impl,
                    handle=self.active_handle(kwargs, state),
                    data=_jsonable({
                        "turn_id": kwargs.get("turn_id"),
                        "api_request_id": kwargs.get("api_request_id"),
                        "api_call_count": kwargs.get("api_call_count"),
                        "mode": self.settings.adaptive_mode,
                    }),
                    metadata=_metadata(kwargs),
                    model_name=str(kwargs.get("model") or ""),
                )
                if inspect.isawaitable(result):
                    return await result
                return result

            return _managed_execute()

        return self._run_managed_with_downstream_preservation(
            next_call, _normalize, _llm_response_payload, _make_managed
        )

    def execute_tool(self, kwargs: dict[str, Any]) -> Any:
        state = self.ensure_session(kwargs)
        tool_name = str(kwargs.get("tool_name") or "tool")
        args = _jsonable(kwargs.get("args") or {})
        next_call = kwargs.get("next_call")
        if not callable(next_call):
            return args

        def _normalize(next_args: Any) -> Any:
            return next_args if isinstance(next_args, dict) else args

        def _make_managed(impl: Callable[[Any], Any]) -> Any:
            async def _managed_execute() -> Any:
                result = self.nemo_relay.tools.execute(
                    tool_name,
                    args,
                    impl,
                    handle=self.active_handle(kwargs, state),
                    data=_jsonable({
                        "turn_id": kwargs.get("turn_id"),
                        "api_request_id": kwargs.get("api_request_id"),
                        "tool_call_id": kwargs.get("tool_call_id"),
                        "mode": self.settings.adaptive_mode,
                    }),
                    metadata=_metadata(kwargs),
                )
                if inspect.isawaitable(result):
                    return await result
                return result

            return _managed_execute()

        return self._run_managed_with_downstream_preservation(
            next_call, _normalize, _jsonable, _make_managed
        )


def register(ctx) -> None:
    ctx.register_hook("on_run_start", on_run_start)
    ctx.register_hook("on_run_end", on_run_end)
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
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


def on_session_start(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.ensure_session(kwargs))


def on_session_end(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(
            lambda: (
                runtime.mark("hermes.session.end", kwargs),
                runtime.export_atif(runtime.ensure_session(kwargs)),
            )
        )


def on_session_finalize(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.close_session(kwargs))


def on_session_reset(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.close_session(kwargs))


def on_pre_llm_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.turn.start", kwargs))


def on_post_llm_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.turn.end", kwargs))


def on_pre_api_request(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is None:
        return
    if runtime.managed_llm_enabled():
        return

    def _record() -> None:
        state = runtime.ensure_session(kwargs)
        request_payload = kwargs.get("request")
        request_body = (
            request_payload.get("body") if isinstance(request_payload, dict) else {}
        )
        request = runtime._sanitize_llm_request(
            runtime.nemo_relay.LLMRequest({}, _jsonable(request_body))
        )
        span = runtime.nemo_relay.llm.call(
            str(kwargs.get("provider") or "llm"),
            request,
            handle=runtime.active_handle(kwargs, state),
            data=_jsonable({
                "turn_id": kwargs.get("turn_id"),
                "api_request_id": kwargs.get("api_request_id"),
            }),
            metadata=_metadata(kwargs),
            model_name=str(kwargs.get("model") or ""),
        )
        state.llm_spans[_api_key(kwargs)] = span

    _safe(_record)


def on_post_api_request(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is None:
        return
    if runtime.managed_llm_enabled():
        return

    def _record() -> None:
        state = runtime.ensure_session(kwargs)
        span = state.llm_spans.pop(_api_key(kwargs), None)
        if span is None:
            runtime.mark("hermes.api.response.unmatched", kwargs)
            return
        runtime.nemo_relay.llm.call_end(
            span,
            runtime._sanitize_llm_response(kwargs.get("response") or {}),
            data=_jsonable({
                "usage": kwargs.get("usage"),
                "finish_reason": kwargs.get("finish_reason"),
            }),
            metadata=_metadata(kwargs),
        )

    _safe(_record)


def on_api_request_error(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is None:
        return
    if runtime.managed_llm_enabled():
        return

    def _record() -> None:
        state = runtime.ensure_session(kwargs)
        span = state.llm_spans.pop(_api_key(kwargs), None)
        if span is None:
            runtime.mark("hermes.api.error", kwargs)
            return
        runtime.nemo_relay.llm.call_end(
            span,
            {
                "error_type": str(
                    kwargs.get("error_type") or type(kwargs.get("error")).__name__
                )
            },
            data=_event_payload(kwargs),
            metadata=_metadata(kwargs),
        )

    _safe(_record)


def on_pre_tool_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is None:
        return
    if runtime.managed_tool_enabled():
        return

    def _record() -> None:
        state = runtime.ensure_session(kwargs)
        span = runtime.nemo_relay.tools.call(
            str(kwargs.get("tool_name") or "tool"),
            runtime._sanitize_tool_request(
                str(kwargs.get("tool_name") or "tool"), kwargs.get("args") or {}
            ),
            handle=runtime.active_handle(kwargs, state),
            data=_jsonable({
                "turn_id": kwargs.get("turn_id"),
                "api_request_id": kwargs.get("api_request_id"),
            }),
            metadata=_metadata(kwargs),
            tool_call_id=str(kwargs.get("tool_call_id") or ""),
        )
        state.tool_spans[_tool_key(kwargs)] = span

    _safe(_record)


def on_post_tool_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is None:
        return
    if runtime.managed_tool_enabled():
        return

    def _record() -> None:
        state = runtime.ensure_session(kwargs)
        span = state.tool_spans.pop(_tool_key(kwargs), None)
        if span is None:
            runtime.mark("hermes.tool.response.unmatched", kwargs)
            return
        result = kwargs.get("result")
        status, error_type, _ = _tool_result_fields(result)
        data = {
            **_event_payload(kwargs),
            "status": status,
            "result_class": _tool_result_class(result),
        }
        if error_type:
            data["error_type"] = error_type
        runtime.nemo_relay.tools.call_end(
            span,
            runtime._sanitize_tool_response(
                str(kwargs.get("tool_name") or "tool"), result
            ),
            data=data,
            metadata=_metadata(kwargs),
        )

    _safe(_record)


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
    if callable(next_call):
        return next_call(request)
    return request


def on_tool_execution_middleware(**kwargs: Any) -> Any:
    runtime = _get_runtime()
    next_call = kwargs.get("next_call")
    args = kwargs.get("args") or {}
    if runtime is not None and runtime.managed_tool_enabled():
        return runtime.execute_tool(kwargs)
    if callable(next_call):
        return next_call(args)
    return args


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
            logger.warning(
                "NeMo Relay observability unavailable: dependency import failed: %s",
                exc,
            )
            _RUNTIME = _INIT_FAILED
            return None
        try:
            _RUNTIME = _Runtime(nemo_relay=nemo_runtime, settings=_load_settings())
        except Exception as exc:
            logger.warning(
                "NeMo Relay observability initialization failed: %s", exc, exc_info=True
            )
            _RUNTIME = _INIT_FAILED
            return None
        return _RUNTIME


def _load_settings() -> _Settings:
    from hermes_cli.config import load_config
    from hermes_constants import get_hermes_home

    telemetry = _jsonable(load_config().get("telemetry") or {})
    atof = _jsonable(telemetry.get("atof") or {})
    atif = _jsonable(telemetry.get("atif") or {})
    trajectories = _jsonable(telemetry.get("trajectories") or {})
    local_enabled = bool(telemetry.get("local", True))
    capture_content = bool(
        telemetry.get("capture_content", False) or trajectories.get("enabled", False)
    )
    redaction = str(telemetry.get("content_redaction") or "pii").lower()
    if redaction not in {"pii", "secrets"}:
        redaction = "pii"
    plugins_toml_path = _env("HERMES_NEMO_RELAY_PLUGINS_TOML")
    if not plugins_toml_path and telemetry.get("plugins_toml"):
        plugins_toml_path = str(telemetry["plugins_toml"])
    plugins_config = _load_plugins_config(plugins_toml_path)
    if plugins_config is None and capture_content and redaction == "pii":
        plugins_config = {
            "version": 1,
            "components": [
                {
                    "kind": "pii_redaction",
                    "enabled": True,
                    "config": {
                        "version": 1,
                        "mode": "builtin",
                        "input": False,
                        "output": False,
                        "tool_input": True,
                        "tool_output": True,
                        "builtin": {
                            "action": "redact",
                            "pattern": _RELAY_PII_PATTERN,
                            "replacement": "[REDACTED]",
                            "target_paths": [],
                        },
                    },
                }
            ],
        }
    adaptive_config = _enabled_component_config(plugins_config, "adaptive")
    default_atof_dir = get_hermes_home() / "telemetry" / "atof"
    default_atif_dir = get_hermes_home() / "telemetry" / "atif"
    return _Settings(
        plugins_toml_path=plugins_toml_path,
        plugins_config=plugins_config,
        adaptive_enabled=adaptive_config is not None,
        adaptive_mode=_adaptive_mode(adaptive_config),
        atof_enabled=_env_bool_default(
            "HERMES_NEMO_RELAY_ATOF_ENABLED", bool(atof.get("enabled", True))
        ),
        atof_output_directory=_env("HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY")
        or str(atof.get("output_directory") or default_atof_dir),
        atof_filename_template=_env("HERMES_NEMO_RELAY_ATOF_FILENAME")
        or str(
            atof.get("filename_template")
            or atof.get("filename")
            or "hermes-atof-{date}.jsonl"
        ),
        atof_mode=_env("HERMES_NEMO_RELAY_ATOF_MODE")
        or str(atof.get("mode") or "append"),
        atif_enabled=_env_bool_default(
            "HERMES_NEMO_RELAY_ATIF_ENABLED",
            bool(atif.get("enabled", False) or trajectories.get("enabled", False)),
        ),
        atif_output_directory=_env("HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY")
        or str(atif.get("output_directory") or default_atif_dir),
        atif_filename_template=_env("HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE")
        or str(atif.get("filename_template") or "hermes-atif-{session_id}.json"),
        atif_subagent_export_mode=_atif_subagent_export_mode(),
        atif_agent_name=_env("HERMES_NEMO_RELAY_ATIF_AGENT_NAME") or "Hermes Agent",
        atif_agent_version=_env("HERMES_NEMO_RELAY_ATIF_AGENT_VERSION") or "unknown",
        atif_model_name=_env("HERMES_NEMO_RELAY_ATIF_MODEL_NAME") or "unknown",
        local_enabled=local_enabled,
        capture_content=capture_content,
        content_redaction=redaction,
    )


def _load_plugins_config(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("NeMo Relay plugins.toml load failed: %s", exc, exc_info=True)
        return None


def _enabled_component_config(
    plugins_config: dict[str, Any] | None,
    kind: str,
) -> dict[str, Any] | None:
    if not isinstance(plugins_config, dict):
        return None
    components = plugins_config.get("components")
    if not isinstance(components, list):
        return None
    for component in components:
        if not isinstance(component, dict):
            continue
        if component.get("kind") != kind or not component.get("enabled", True):
            continue
        config = component.get("config")
        return config if isinstance(config, dict) else {}
    return None


def _adaptive_mode(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return "observe_only"
    tool_parallelism = config.get("tool_parallelism")
    if isinstance(tool_parallelism, dict):
        mode = tool_parallelism.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
    mode = config.get("mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    return "observe_only"


def _observability_exporter_enabled(
    plugins_config: dict[str, Any] | None,
    exporter_name: str,
) -> bool:
    observability_config = _enabled_component_config(plugins_config, "observability")
    if not isinstance(observability_config, dict):
        return False
    exporter_config = observability_config.get(exporter_name)
    if not isinstance(exporter_config, dict):
        return False
    return exporter_config.get("enabled", True) is not False


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _atif_subagent_export_mode() -> str:
    mode = _env("HERMES_NEMO_RELAY_ATIF_SUBAGENT_EXPORT_MODE").lower()
    return "all" if mode == "all" else "embedded"


def _env_bool(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}


def _env_bool_default(name: str, default: bool) -> bool:
    return _env_bool(name) if name in os.environ else default


def _session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("session_id") or kwargs.get("parent_session_id") or "default")


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
        ("telemetry_schema_version", "telemetry_schema_version"),
    ):
        value = parent_metadata.get(source)
        if value is not None:
            metadata[target] = value
    return metadata


def _api_key(kwargs: dict[str, Any]) -> str:
    return str(
        kwargs.get("api_request_id")
        or f"{_session_id(kwargs)}:{kwargs.get('api_call_count') or 'api'}"
    )


def _tool_key(kwargs: dict[str, Any]) -> str:
    return str(
        kwargs.get("tool_call_id")
        or f"{_session_id(kwargs)}:{kwargs.get('turn_id') or ''}:{kwargs.get('tool_name') or 'tool'}"
    )


def _metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "telemetry_schema_version",
        "run_id",
        "entrypoint",
        "session_id",
        "platform",
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
        "status",
        "reason",
        "error_type",
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
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
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


_EVENT_FIELDS = (
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
    "outcome",
    "completed",
    "reason",
    "duration_ms",
    "retryable",
    "error_type",
    "finish_reason",
    "usage",
    "parent_session_id",
    "parent_turn_id",
    "child_session_id",
    "child_subagent_id",
    "child_role",
    "child_status",
)


def _event_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return structural lifecycle data; arbitrary hook payloads never enter ATOF."""
    return {
        key: _jsonable(kwargs[key])
        for key in _EVENT_FIELDS
        if kwargs.get(key) is not None
    }


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(_jsonable(value), ensure_ascii=False))
    except Exception:
        return len(str(value))


def _tool_result_class(result: Any) -> str:
    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except Exception:
            return "string"
    if isinstance(parsed, dict):
        return "object"
    if isinstance(parsed, list):
        return "array"
    if parsed is None:
        return "null"
    if isinstance(parsed, bool):
        return "boolean"
    if isinstance(parsed, (int, float)):
        return "number"
    return "string"


def _tool_result_fields(result: Any) -> tuple[str, str | None, str | None]:
    try:
        from model_tools import _tool_result_observer_fields

        return _tool_result_observer_fields(result)
    except Exception:
        parsed = result
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
        except Exception:
            pass
        if isinstance(parsed, dict) and parsed.get("error"):
            return "error", "tool_error", str(parsed["error"])
        return "ok", None, None


def _structural_llm_request(content: Any) -> dict[str, Any]:
    payload = content if isinstance(content, dict) else {}
    messages = (
        payload.get("messages") if isinstance(payload.get("messages"), list) else []
    )
    return {
        "model": payload.get("model"),
        "message_count": len(messages),
        "tool_count": len(payload.get("tools") or [])
        if isinstance(payload.get("tools"), list)
        else 0,
        "request_chars": _json_size(content),
    }


def _structural_llm_response(payload: Any) -> dict[str, Any]:
    value = payload if isinstance(payload, dict) else {}
    return {
        "model": value.get("model"),
        "finish_reason": value.get("finish_reason"),
        "usage": _jsonable(value.get("usage") or {}),
        "response_chars": _json_size(payload),
    }


def _redact_payload(value: Any, mode: str = "pii") -> Any:
    if isinstance(value, dict):
        sensitive = {
            "access_token",
            "api_key",
            "apikey",
            "authorization",
            "client_secret",
            "password",
            "private_key",
            "refresh_token",
            "secret",
            "token",
        }
        return {
            str(key): (
                "«redacted-secret»"
                if str(key).lower() in sensitive
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

        value = redact_sensitive_text(value, force=True)
    except Exception:
        value = "[redaction-unavailable]"
    if mode == "pii":
        value = re.sub(_RELAY_PII_PATTERN, "«redacted-pii»", value)
    return value


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _original_downstream_error(exc: Exception) -> BaseException:
    # Hermes wraps downstream execution failures in a local/private exception
    # class, so detect the wrapper by shape instead of importing it here.
    original = getattr(exc, "original", None)
    if exc.__class__.__name__ == "_DownstreamExecutionError" and isinstance(
        original, BaseException
    ):
        return original
    return exc


def _is_relay_wrapped_callback_error(
    exc: Exception, callback_error: Exception | None
) -> bool:
    # NeMo Relay re-wraps a failing callback as ``RuntimeError("internal error:
    # <ClassName>: <message>")``. Match by prefix rather than exact equality so a
    # trailing traceback/suffix in a future Relay version doesn't silently defeat
    # the unwrap; the class-name + message prefix still discriminates the real
    # downstream failure from unrelated Relay-internal errors. If Relay drops the
    # leading ``internal error:`` shape entirely, this returns False and Hermes
    # falls back to surfacing Relay's error (the pre-fix behavior) rather than
    # masking it.
    if callback_error is None or not isinstance(exc, RuntimeError):
        return False
    expected = f"internal error: {callback_error.__class__.__name__}: {callback_error}"
    return str(exc).startswith(expected)


def _llm_response_payload(response: Any) -> Any:
    """Return the LLM response shape NeMo Relay's ATIF conversion expects."""
    payload = _jsonable(response)
    if isinstance(payload, dict) and "assistant_message" in payload:
        return payload

    choices = _value(response, "choices")
    if choices is None and isinstance(payload, dict):
        choices = payload.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    message = _value(first_choice, "message")
    finish_reason = _value(first_choice, "finish_reason")

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
        "finish_reason": finish_reason,
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


def _safe(fn) -> None:
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

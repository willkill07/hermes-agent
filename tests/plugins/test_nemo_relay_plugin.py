"""Tests for the bundled observability/nemo_relay plugin."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.plugins import PluginManager


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "nemo_relay"


class _FakeLLMRequest:
    def __init__(self, headers, content):
        self.headers = headers
        self.content = content


class _FakeCodec:
    pass


class _FakeChatCodec(_FakeCodec):
    pass


class _FakeResponsesCodec(_FakeCodec):
    pass


class _FakeAnthropicCodec(_FakeCodec):
    pass


class _FakeAtofExporterConfig:
    def __init__(self):
        self.output_directory = ""
        self.filename = "events.jsonl"
        self.mode = "append"


class _FakeAtofExporter:
    def __init__(self, events, config):
        self.events = events
        self.config = config

    def register(self, name):
        self.events.append(("atof.register", name, self.config.output_directory))

    def deregister(self, name):
        self.events.append(("atof.deregister", name))
        return True


class _FakeAtifExporter:
    def __init__(self, events, run_id, agent_name, agent_version, kwargs):
        self.events = events
        self.run_id = run_id
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.kwargs = kwargs

    def register(self, name):
        self.events.append(("atif.register", name, self.run_id))

    def deregister(self, name):
        self.events.append(("atif.deregister", name, self.run_id))
        return True

    def export_json(self):
        return json.dumps({"run_id": self.run_id, "agent_name": self.agent_name})


class _FakeNemoRelay:
    def __init__(self):
        self.events = []
        self.ScopeType = SimpleNamespace(Agent="agent")
        self.scope = SimpleNamespace(
            push=self._scope_push,
            pop=self._scope_pop,
            event=self._scope_event,
        )
        self.llm = SimpleNamespace(
            call=self._llm_call,
            call_end=self._llm_call_end,
            execute=self._llm_execute,
        )
        self.tools = SimpleNamespace(
            call=self._tool_call,
            call_end=self._tool_call_end,
            execute=self._tool_execute,
        )
        self.guardrails = SimpleNamespace(
            register_tool_sanitize_request=self._register_guardrail,
            register_tool_sanitize_response=self._register_guardrail,
            register_llm_sanitize_request=self._register_guardrail,
            register_llm_sanitize_response=self._register_guardrail,
        )
        self.plugin = SimpleNamespace(
            initialize=self._plugin_initialize,
            clear=self._plugin_clear,
        )
        self.subscribers = SimpleNamespace(register=self._register_subscriber)
        self.codecs = SimpleNamespace(
            OpenAIChatCodec=_FakeChatCodec,
            OpenAIResponsesCodec=_FakeResponsesCodec,
            AnthropicMessagesCodec=_FakeAnthropicCodec,
        )
        self.LLMRequest = _FakeLLMRequest
        self.AtofExporterConfig = _FakeAtofExporterConfig
        self.AtofExporterMode = SimpleNamespace(Append="append", Overwrite="overwrite")
        self.AtofExporter = lambda config: _FakeAtofExporter(self.events, config)
        self.AtifExporter = lambda run_id, name, version, **kwargs: _FakeAtifExporter(
            self.events, run_id, name, version, kwargs
        )

    def _scope_push(self, name, scope_type, **kwargs):
        handle = ("scope", name)
        self.events.append(("scope.push", name, scope_type, kwargs))
        return handle

    def _scope_pop(self, handle, **kwargs):
        self.events.append(("scope.pop", handle, kwargs))

    def _scope_event(self, name, **kwargs):
        self.events.append(("scope.event", name, kwargs))

    def _llm_call(self, name, request, **kwargs):
        handle = ("llm", name)
        self.events.append(("llm.call", name, request.content, kwargs))
        return handle

    def _llm_call_end(self, handle, response, **kwargs):
        self.events.append(("llm.call_end", handle, response, kwargs))

    def _llm_execute(self, name, request, func, **kwargs):
        self.events.append(("llm.execute.start", name, request.content, kwargs))
        result = func(
            _FakeLLMRequest(request.headers, {"intercepted": True, **request.content})
        )
        self.events.append(("llm.execute.end", name, result, kwargs))
        return result

    def _tool_call(self, name, args, **kwargs):
        handle = ("tool", name)
        self.events.append(("tool.call", name, args, kwargs))
        return handle

    def _tool_call_end(self, handle, result, **kwargs):
        self.events.append(("tool.call_end", handle, result, kwargs))

    def _tool_execute(self, name, args, func, **kwargs):
        self.events.append(("tool.execute.start", name, args, kwargs))
        result = func({"intercepted": True, **args})
        self.events.append(("tool.execute.end", name, result, kwargs))
        return result

    def _register_guardrail(self, name, priority, callback):
        self.events.append(("guardrail.register", name, priority, callback))

    def _register_subscriber(self, name, callback):
        self.events.append(("subscriber.register", name, callback))

    async def _plugin_initialize(self, config):
        self.events.append(("plugin.initialize", config))
        return {"diagnostics": []}

    async def _plugin_clear(self):
        self.events.append(("plugin.clear",))


def _fresh_plugin(monkeypatch, fake, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setitem(sys.modules, "nemo_relay", fake)
    sys.modules.pop("plugins.observability.nemo_relay", None)
    plugin = importlib.import_module("plugins.observability.nemo_relay")
    plugin.reset_for_tests()
    return plugin


def _run_payload(run_id="run-1", session_id="session-1", **overrides):
    payload = {
        "run_id": run_id,
        "turn_id": run_id,
        "task_id": "task-1",
        "session_id": session_id,
        "entrypoint": "cli",
        "platform": "cli",
        "provider": "openai",
        "model": "demo-model",
        "api_mode": "chat_completions",
        "started_at": 1_800_000_000.0,
        "user_message_chars": 12,
    }
    payload.update(overrides)
    return payload


def _wrapped_downstream_error(original):
    class _DownstreamExecutionError(Exception):
        def __init__(self, error):
            super().__init__(str(error))
            self.original = error

    return _DownstreamExecutionError(original)


def test_manifest_and_discovery(tmp_path, monkeypatch):
    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    assert {"on_run_start", "on_run_end"}.issubset(manifest["hooks"])

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["observability/nemo_relay"]
    assert loaded.manifest.source == "bundled"
    assert not loaded.enabled


def test_one_relay_scope_per_run_and_process_observability_plugin(
    tmp_path, monkeypatch
):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)

    first = _run_payload("run-1")
    second = _run_payload("run-2")
    plugin.on_run_start(**first)
    plugin.on_run_end(**first, ended_at=1_800_000_001.0, outcome="completed")
    plugin.on_run_start(**second)
    plugin.on_run_end(**second, ended_at=1_800_000_002.0, outcome="failed")

    pushes = [event for event in fake.events if event[0] == "scope.push"]
    pops = [event for event in fake.events if event[0] == "scope.pop"]
    assert [event[1] for event in pushes] == ["hermes.run:run-1", "hermes.run:run-2"]
    assert len(pops) == 2
    assert pops[0][2]["output"]["outcome"] == "completed"
    assert pops[1][2]["output"]["outcome"] == "failed"
    assert [event[0] for event in fake.events].count("plugin.initialize") == 1
    assert [event[1] for event in fake.events if event[0] == "subscriber.register"] == [
        "hermes.telemetry.projector"
    ]
    assert not any(event[0] == "atof.register" for event in fake.events)
    initialize = next(event for event in fake.events if event[0] == "plugin.initialize")
    observability = initialize[1]["components"][0]["config"]
    assert observability["atof"]["enabled"] is True
    assert observability["atof"]["filename"] == "hermes-atof.jsonl"
    assert observability["atif"]["enabled"] is False
    assert not plugin._get_runtime().runs


def test_managed_llm_and_tool_execution_are_default(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)
    plugin.on_run_start(**_run_payload())

    seen_request = {}

    def llm_call(request):
        seen_request.update(request)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    raw_response = plugin.on_llm_execution_middleware(
        **_run_payload(),
        api_request_id="api-1",
        request={"messages": [{"role": "user", "content": "hello"}]},
        next_call=llm_call,
    )
    tool_response = plugin.on_tool_execution_middleware(
        **_run_payload(),
        tool_name="terminal",
        tool_call_id="tool-1",
        args={"command": "pwd"},
        next_call=lambda args: {"raw": args},
    )

    assert raw_response["choices"][0]["message"]["content"] == "ok"
    assert seen_request["intercepted"] is True
    assert tool_response["raw"]["intercepted"] is True
    llm_start = next(event for event in fake.events if event[0] == "llm.execute.start")
    tool_start = next(
        event for event in fake.events if event[0] == "tool.execute.start"
    )
    assert isinstance(llm_start[3]["codec"], _FakeCodec)
    assert llm_start[3]["response_codec"] is None
    assert llm_start[3]["handle"] == ("scope", "hermes.run:run-1")
    assert tool_start[3]["handle"] == ("scope", "hermes.run:run-1")


def test_execution_bypass_preserves_original_payload_objects(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)
    request = {}
    args = {}

    llm_result = plugin.on_llm_execution_middleware(
        request=request,
        next_call=lambda payload: payload,
    )
    tool_result = plugin.on_tool_execution_middleware(
        tool_name="terminal",
        args=args,
        next_call=lambda payload: payload,
    )

    assert llm_result is request
    assert tool_result is args


@pytest.mark.parametrize(
    ("api_mode", "provider", "codec_type"),
    [
        ("chat_completions", "openai", _FakeChatCodec),
        ("codex_responses", "openai", _FakeResponsesCodec),
        ("anthropic_messages", "anthropic", _FakeAnthropicCodec),
    ],
)
def test_provider_api_modes_select_native_codecs(
    tmp_path, monkeypatch, api_mode, provider, codec_type
):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)

    assert isinstance(plugin._llm_codec(fake, api_mode, provider), codec_type)


def test_managed_execution_preserves_original_downstream_error(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()

    def native_like_execute(name, request, func, **kwargs):
        try:
            return func(request)
        except Exception as exc:
            raise RuntimeError(f"internal error: {type(exc).__name__}: {exc}") from None

    fake.llm.execute = native_like_execute
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)
    plugin.on_run_start(**_run_payload())

    class ProviderAuthError(Exception):
        status_code = 403

    provider_error = ProviderAuthError("provider auth failed")
    with pytest.raises(ProviderAuthError) as caught:
        plugin.on_llm_execution_middleware(
            **_run_payload(),
            request={"messages": []},
            next_call=lambda request: (_ for _ in ()).throw(
                _wrapped_downstream_error(provider_error)
            ),
        )

    assert caught.value is provider_error


def test_subagent_run_has_explicit_parent_handle(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)
    plugin.on_run_start(**_run_payload("parent-run", "parent-session"))
    plugin.on_subagent_start(
        parent_session_id="parent-session",
        parent_turn_id="parent-run",
        child_session_id="child-session",
        child_subagent_id="child-1",
        child_role="researcher",
    )
    plugin.on_run_start(
        **_run_payload(
            "child-run",
            "child-session",
            is_delegated=True,
            parent_session_id="parent-session",
            parent_turn_id="parent-run",
        )
    )

    child = next(
        event
        for event in fake.events
        if event[0] == "scope.push" and event[1] == "hermes.run:child-run"
    )
    assert child[3]["handle"] == ("scope", "hermes.run:parent-run")
    assert child[3]["metadata"]["nemo_relay_scope_role"] == "subagent"
    assert child[3]["metadata"]["subagent_id"] == "child-1"


def test_structural_sanitizers_keep_content_out_of_events(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)
    runtime = plugin._get_runtime()

    llm_request = runtime._sanitize_llm_request(
        _FakeLLMRequest(
            {},
            {
                "messages": [{"role": "user", "content": "secret prompt"}],
                "tools": [{"name": "terminal"}],
            },
        )
    ).content
    llm_response = runtime._sanitize_llm_response({
        "choices": [{"message": {"role": "assistant", "content": "secret answer"}}]
    })
    tool_request = runtime._sanitize_tool_request(
        "terminal", {"command": "echo secret", "token": "sk-abcdefghijklmnop"}
    )
    tool_response = runtime._sanitize_tool_response("terminal", "secret result")

    encoded = json.dumps([llm_request, llm_response, tool_request, tool_response])
    assert "secret prompt" not in encoded
    assert "secret answer" not in encoded
    assert "echo secret" not in encoded
    assert "secret result" not in encoded
    assert llm_request["message_count"] == 1
    assert llm_response["content_chars"] == len("secret answer")
    assert tool_request["argument_keys"] == ["command", "token"]


def test_content_capture_force_redacts_secrets(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "telemetry:\n  capture_content: true\n  atof:\n    enabled: false\n",
        encoding="utf-8",
    )
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)
    runtime = plugin._get_runtime()

    sanitized = runtime._sanitize_tool_request(
        "demo", {"prompt": "use sk-abcdefghijklmnop", "api_key": "plain-secret"}
    )
    encoded = json.dumps(sanitized)
    assert "sk-abcdefghijklmnop" not in encoded
    assert "plain-secret" not in encoded


def test_plugins_toml_is_process_scoped_not_cleared_per_run(tmp_path, monkeypatch):
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text("version = 1\n", encoding="utf-8")
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"telemetry:\n  plugins_toml: {plugins_toml}\n",
        encoding="utf-8",
    )
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)

    plugin.on_run_start(**_run_payload("run-1"))
    plugin.on_run_end(**_run_payload("run-1"), ended_at=1_800_000_001.0)
    plugin.on_run_start(**_run_payload("run-2"))
    plugin.on_run_end(**_run_payload("run-2"), ended_at=1_800_000_002.0)

    names = [event[0] for event in fake.events]
    assert names.count("plugin.initialize") == 1
    assert "plugin.clear" not in names


def test_exporter_headers_are_resolved_from_environment(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
telemetry:
  export:
    otlp:
      enabled: true
      endpoint: https://collector.example/v1/traces
      headers_env:
        Authorization: RELAY_TEST_AUTH
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("RELAY_TEST_AUTH", "Bearer in-memory-secret")
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)

    plugin.on_run_start(**_run_payload())

    initialize = next(event for event in fake.events if event[0] == "plugin.initialize")
    config = initialize[1]["components"][0]["config"]["opentelemetry"]
    assert config["endpoint"] == "https://collector.example/v1/traces"
    assert config["headers"] == {"Authorization": "Bearer in-memory-secret"}
    assert "in-memory-secret" not in (home / "config.yaml").read_text(encoding="utf-8")


def test_config_failure_keeps_profile_safe_telemetry_root(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: (_ for _ in ()).throw(RuntimeError("broken config")),
    )

    settings = plugin._load_settings()
    observability = settings.plugins_config["components"][0]["config"]

    expected_root = tmp_path / "hermes-home" / "telemetry"
    assert observability["atof"]["output_directory"] == str(expected_root / "atof")
    assert observability["atif"]["output_directory"] == str(expected_root / "atif")


def test_disabled_observability_component_does_not_create_directories(
    tmp_path, monkeypatch
):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake, tmp_path)
    output = tmp_path / "must-not-exist"

    plugin._ensure_plugin_output_dirs({
        "components": [
            {
                "kind": "observability",
                "enabled": False,
                "config": {
                    "atof": {
                        "enabled": True,
                        "output_directory": str(output),
                    }
                },
            }
        ]
    })

    assert not output.exists()


def test_real_nemo_relay_writes_structural_atof(tmp_path, monkeypatch):
    pytest.importorskip("nemo_relay")
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "telemetry:\n  atif:\n    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    sys.modules.pop("plugins.observability.nemo_relay", None)
    plugin = importlib.import_module("plugins.observability.nemo_relay")
    plugin.reset_for_tests()

    payload = _run_payload("real-run", "real-session")
    plugin.on_run_start(**payload)
    raw_llm = plugin.on_llm_execution_middleware(
        **payload,
        api_request_id="api-real",
        request={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "raw-prompt-sentinel"}],
        },
        next_call=lambda request: {
            "model": "demo-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "raw-response-sentinel",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    )
    raw_tool = plugin.on_tool_execution_middleware(
        **payload,
        tool_name="terminal",
        tool_call_id="tool-real",
        args={"command": "raw-tool-argument-sentinel"},
        next_call=lambda args: "raw-tool-result-sentinel",
    )
    plugin.on_run_end(
        **payload,
        ended_at=1_800_000_001.0,
        outcome="completed",
        completed=True,
    )

    runtime = plugin._get_runtime()
    runtime.nemo_relay.subscribers.flush()
    output = home / "telemetry" / "atof" / "hermes-atof.jsonl"
    atif_files = list((home / "telemetry" / "atif").glob("hermes-atif-*.json"))
    text = output.read_text(encoding="utf-8")
    events = [json.loads(line) for line in text.splitlines()]
    run_start = next(
        event
        for event in events
        if event["name"] == "hermes.run:real-run" and event["scope_category"] == "start"
    )
    descendants = [event for event in events if event["category"] in {"llm", "tool"}]
    assert raw_llm["choices"][0]["message"]["content"] == "raw-response-sentinel"
    assert raw_tool == "raw-tool-result-sentinel"
    assert "raw-prompt-sentinel" not in text
    assert "raw-response-sentinel" not in text
    assert "raw-tool-argument-sentinel" not in text
    assert "raw-tool-result-sentinel" not in text
    assert "real-run" in text
    assert "api-real" in text
    assert "tool-real" in text
    assert descendants
    assert all(event["parent_uuid"] == run_start["uuid"] for event in descendants)
    assert len(atif_files) == 1
    atif = json.loads(atif_files[0].read_text(encoding="utf-8"))
    assert atif["steps"]

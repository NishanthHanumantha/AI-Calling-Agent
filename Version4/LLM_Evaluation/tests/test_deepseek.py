from __future__ import annotations

import json
from unittest.mock import patch

from evaluator.runner import load_yaml_config, resolve_model_runtime, run_benchmark
from interactive.renderer import render_turn
from interactive.schemas import DISPLAY_ORDER, TurnResponse
from interactive.session import InteractiveSession, format_check_report
from models.deepseek import DeepSeekProvider, thinking_body
from models import build_provider


SECRET = "sk-test-deepseek-secret-value"


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def _provider(**kwargs) -> DeepSeekProvider:
    params = {
        "api_key": SECRET,
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "timeout": 5,
        "max_retries": 0,
        "thinking_mode": "disabled",
        "model_alias": "deepseek_flagship",
        "model_role": "flagship",
    }
    params.update(kwargs)
    return DeepSeekProvider(**params)


def test_deepseek_configuration_loading(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_FLAGSHIP_MODEL", "")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "")
    monkeypatch.setenv("DEEPSEEK_THINKING_MODE", "")
    config = load_yaml_config()
    assert config["interactive_model_aliases"] == [
        "sarvam_conversational",
        "deepseek_flagship",
        "claude_sonnet",
        "claude_flagship",
    ]
    assert "sarvam_flagship" not in config["interactive_model_aliases"]
    assert "deepseek_flagship" not in config["active_model_aliases"]
    assert "sarvam_flagship" in config["active_model_aliases"]
    spec = config["models"]["deepseek_flagship"]
    assert spec["provider"] == "deepseek"
    assert spec["default_model"] == "deepseek-flash"
    assert spec["api_key_env"] == "DEEPSEEK_API_KEY"
    assert spec["default_base_url"] == "https://api.deepseek.com"
    assert spec["thinking_mode"] == "disabled"
    runtime = resolve_model_runtime("deepseek_flagship", spec, config)
    assert runtime["provider"] == "deepseek"
    assert runtime["model"] == "deepseek-flash"
    assert runtime["thinking_mode"] == "disabled"
    assert runtime["available"] is False
    assert runtime["in_scope"] is True


def test_deepseek_adapter_initialization():
    provider = build_provider(
        "deepseek",
        api_key="present",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        thinking_mode="disabled",
    )
    assert isinstance(provider, DeepSeekProvider)
    assert provider.name == "deepseek"
    assert provider.model == "deepseek-flash"
    assert provider.base_url == "https://api.deepseek.com/chat/completions"
    assert provider.thinking_mode == "disabled"
    assert thinking_body("disabled") == {"type": "disabled"}
    assert thinking_body("non-thinking") == {"type": "disabled"}
    assert thinking_body("enabled") == {"type": "enabled"}
    assert thinking_body("max") == {"type": "disabled"}


def test_missing_deepseek_api_key_is_unavailable(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_FLAGSHIP_MODEL", "")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "")
    session = InteractiveSession(dry_run=False, providers={})
    slot = next(item for item in session.slots if item.alias == "deepseek_flagship")
    assert slot.model_id == "deepseek-flash"
    assert slot.provider == "deepseek"
    assert slot.available is False
    assert slot.status == "UNAVAILABLE — missing API key"
    assert slot.error == "DEEPSEEK_API_KEY: NOT CONFIGURED"
    assert "deepseek_flagship" not in session._providers
    with patch("models.http_util.requests.request") as request:
        row = session._probe_slot(slot)
        request.assert_not_called()
    assert row["status"] == "UNAVAILABLE"
    assert row["auth"] == "NOT CONFIGURED"
    assert row["model_check"] == "UNAVAILABLE"
    assert row["error_type"] == "MISSING_API_KEY"
    report = format_check_report({"results": [row]})
    assert "DEEPSEEK_API_KEY: NOT CONFIGURED" in report
    assert "deepseek-flash" in report
    assert SECRET not in report


def test_deepseek_response_normalization():
    payload = {
        "model": "deepseek-flash",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"intent":"LOCATION_QUERY","language":"en","answer":"Near Electronic City.","action":"ANSWER"}',
                    "reasoning_content": "do not surface this trace",
                }
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    captured = {}

    def fake_request(method, url, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        captured["headers"] = kwargs["headers"]
        return _Resp(200, payload)

    provider = _provider()
    with patch("models.http_util.requests.request", side_effect=fake_request):
        result = provider.generate(
            system_prompt="You are the calling agent.",
            conversation_history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello from DeepSeek only."},
            ],
            customer_utterance="Where is it located?",
            retrieved_context="SOBHA Townpark is near Electronic City.",
        )
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["json"]["model"] == "deepseek-flash"
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in captured["json"]
    messages = captured["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are the calling agent."
    assert messages[1]["role"] == "user"
    assert "Where is it located?" in messages[1]["content"]
    assert "Hello from DeepSeek only." in messages[1]["content"]
    assert "SOBHA Townpark is near Electronic City." in messages[1]["content"]
    assert result["provider"] == "deepseek"
    assert result["model"] == "deepseek-flash"
    assert result["text"]
    assert "do not surface this trace" not in result["text"]
    assert result["answer"] == "Near Electronic City."
    assert result["intent"] == "LOCATION_QUERY"
    assert result["action"] == "ANSWER"
    assert result["input_tokens"] == 11
    assert result["output_tokens"] == 7
    assert result["total_tokens"] == 18
    assert result["usage"]["input_tokens"] == 11
    assert result["error_type"] is None
    assert SECRET not in json.dumps(result)


def test_deepseek_error_is_not_a_quality_result():
    def fake_request(method, url, timeout=None, **kwargs):
        return _Resp(429, {"error": {"message": "rate limit"}}, text="HTTP rate limit")

    provider = _provider()
    with patch("models.http_util.requests.request", side_effect=fake_request):
        result = provider.generate(
            system_prompt="role",
            conversation_history=[],
            customer_utterance="Hello",
            retrieved_context="kb",
        )
    assert result["error_type"] == "RATE_LIMIT"
    assert result["error_message"] == "HTTP 429"
    assert result["answer"] is None
    assert result["text"] is None
    assert result["intent"] is None
    assert "hallucination" not in result or result.get("hallucination") in {None, False}
    assert SECRET not in json.dumps(result)


def test_deepseek_latency_capture():
    def fake_request(method, url, timeout=None, **kwargs):
        return _Resp(200, {"model": "deepseek-flash", "choices": [{"message": {"content": "Hello."}}]})

    provider = _provider()
    with patch("models.http_util.requests.request", side_effect=fake_request):
        with patch("models.deepseek.time.perf_counter", side_effect=[10.0, 10.25]):
            result = provider.complete(
                [
                    {"role": "system", "content": "Reply briefly."},
                    {"role": "user", "content": "Hello. Please respond with one short sentence."},
                ]
            )
    assert result["latency_ms"] == 250
    assert result["text"] == "Hello."
    assert result["model"] == "deepseek-flash"


def test_deepseek_token_usage_is_not_invented():
    def fake_request(method, url, timeout=None, **kwargs):
        return _Resp(200, {"choices": [{"message": {"content": "Hi."}}]})

    provider = _provider()
    with patch("models.http_util.requests.request", side_effect=fake_request):
        result = provider.complete([{"role": "user", "content": "Hi"}])
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None
    assert result["total_tokens"] is None
    assert result["usage"]["total_tokens"] is None
    assert result["model"] == "deepseek-flash"


def test_deepseek_partial_usage_does_not_invent_total():
    def fake_request(method, url, timeout=None, **kwargs):
        return _Resp(
            200,
            {"choices": [{"message": {"content": "Hi."}}], "usage": {"prompt_tokens": 4}},
        )

    provider = _provider()
    with patch("models.http_util.requests.request", side_effect=fake_request):
        result = provider.complete([{"role": "user", "content": "Hi"}])
    assert result["input_tokens"] == 4
    assert result["output_tokens"] is None
    assert result["total_tokens"] is None


def test_thinking_mode_is_configurable_and_not_max_by_default():
    captured = {}

    def fake_request(method, url, timeout=None, **kwargs):
        captured["json"] = kwargs["json"]
        return _Resp(200, {"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(thinking_mode="enabled")
    with patch("models.http_util.requests.request", side_effect=fake_request):
        provider.complete([{"role": "user", "content": "Hi"}])
    assert captured["json"]["thinking"] == {"type": "enabled"}
    assert captured["json"].get("reasoning_effort") != "max"
    assert "reasoning_effort" not in captured["json"]


def test_four_model_interactive_session(monkeypatch):
    for name in (
        "DEEPSEEK_FLAGSHIP_MODEL",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_THINKING_MODE",
        "SARVAM_CONVERSATIONAL_MODEL",
        "SARVAM_BASE_URL",
        "CLAUDE_SONNET_MODEL",
        "CLAUDE_FLAGSHIP_MODEL",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.setenv(name, "")
    session = InteractiveSession(dry_run=True)
    aliases = [slot.alias for slot in session.slots]
    assert aliases == list(DISPLAY_ORDER)
    assert "sarvam_flagship" not in aliases
    by_alias = {slot.alias: slot for slot in session.slots}
    assert by_alias["sarvam_conversational"].model_id == "sarvam-105b-conversations"
    assert by_alias["deepseek_flagship"].model_id == "deepseek-flash"
    assert by_alias["deepseek_flagship"].provider == "deepseek"
    assert by_alias["claude_sonnet"].provider == "claude"
    assert by_alias["claude_flagship"].provider == "claude"
    assert "sarvam-105b" not in {slot.model_id for slot in session.slots}
    report = session.preview_payloads("Where is it located?")
    assert set(report) == set(DISPLAY_ORDER)
    assert len(set(report.values())) == 1


def test_batch_benchmark_does_not_add_deepseek():
    result = run_benchmark(dry_run=True)
    assert result["api_calls"] == 0
    assert "sarvam_flagship" in result["models_configured_list"]
    assert "deepseek_flagship" not in result["models_configured_list"]
    assert result["sarvam_models"] == 2
    assert result["claude_models"] == 2


def test_check_report_shape_for_available_deepseek():
    row = {
        "provider": "deepseek",
        "alias": "deepseek_flagship",
        "model_id": "deepseek-flash",
        "status": "AVAILABLE",
        "auth": "PASS",
        "model_check": "FOUND",
    }
    text = format_check_report({"results": [row]})
    assert "Provider: deepseek" in text
    assert "Logical Model Name: deepseek_flagship" in text
    assert "Configured Model ID: deepseek-flash" in text
    assert "API availability: AVAILABLE" in text
    assert "Authentication: PASS" in text
    assert "Model check: FOUND" in text


def test_turn_response_tokens_stay_null():
    response = TurnResponse(
        alias="deepseek_flagship",
        provider="deepseek",
        model_id="deepseek-flash",
        answer="Hi.",
        intent=None,
        action=None,
        language=None,
        stage=None,
        raw_response="Hi.",
        latency_ms=12,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        error_type=None,
        error_message=None,
    )
    assert response.input_tokens is None
    assert response.latency_ms == 12
    rendered = render_turn(1, "Hi", [response])
    assert "latency=0.01s" in rendered

"""Mock sequential controller tests. Zero real Twilio calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from live_voice.cli import main, parse_args
from live_voice.constants import CANDIDATE_ORDER, CONFIRMATION_PHRASE
from live_voice.controller import run_campaign
from live_voice.session_store import SessionStore


class FakeProvider:
    def __init__(self, alias: str, model_id: str, answer: str = "Townpark is near Electronic City."):
        self.alias = alias
        self.model_id = model_id
        self.answer = answer
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "provider": "fake",
            "model_id": self.model_id,
            "answer": self.answer,
            "intent": "PROJECT_FAQ",
            "action": "ANSWER",
            "schema_valid": True,
            "latency_ms": 40,
            "error": None,
            "error_type": None,
            "error_message": None,
        }


class FakeCall:
    def __init__(self, sid: str, status: str = "completed"):
        self.sid = sid
        self.status = status
        self.duration = "18"

    def fetch(self):
        return self


class FakeTwilio:
    def __init__(self, status: str = "completed"):
        self.creates: list[dict] = []
        self.calls_by_sid: dict[str, FakeCall] = {}
        self.status = status
        self.calls = self

    def create(self, **kwargs):
        self.creates.append(kwargs)
        sid = f"CA{len(self.creates):04d}"
        call = FakeCall(sid, status=self.status)
        self.calls_by_sid[sid] = call
        return call

    def __call__(self, sid: str):
        return self.calls_by_sid[sid]


def _providers() -> dict[str, FakeProvider]:
    return {
        "sarvam_conversational": FakeProvider("sarvam_conversational", "sarvam-105b-conversations"),
        "deepseek_flagship": FakeProvider("deepseek_flagship", "deepseek-flash"),
        "claude_sonnet": FakeProvider("claude_sonnet", "claude-sonnet-4-6"),
        "claude_flagship": FakeProvider("claude_flagship", "claude-opus-4-8"),
    }


def _env(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+10000000000")
    monkeypatch.setenv("TWILIO_TO_NUMBER", "+10000000001")
    monkeypatch.setenv("PUBLIC_URL", "https://eval.example.test")


def _prompts(*answers):
    queue = list(answers)

    def prompt(_message: str) -> str:
        if not queue:
            return "Abort"
        return queue.pop(0)

    return prompt


def test_parse_args_default_is_not_live():
    args = parse_args([])
    assert args.live is False


def test_dry_run_creates_zero_calls(tmp_path, monkeypatch):
    _env(monkeypatch)
    client = FakeTwilio()

    def factory(_sid, _token):
        raise AssertionError("Twilio client must not be built in dry-run")

    result = run_campaign(
        live=False,
        twilio_factory=factory,
        store=SessionStore(base_dir=tmp_path),
        providers=_providers(),
        stdout_write=lambda _t: None,
    )
    assert result["mode"] == "dry-run"
    assert result["calls_created"] == 0
    assert client.creates == []
    assert [row["alias"] for row in result["candidates"]] == list(CANDIDATE_ORDER)


def test_live_without_confirmation_creates_zero_calls(tmp_path, monkeypatch):
    _env(monkeypatch)
    client = FakeTwilio()
    result = run_campaign(
        live=True,
        prompt_fn=_prompts("no"),
        twilio_factory=lambda *_a: client,
        store=SessionStore(base_dir=tmp_path),
        providers=_providers(),
        stdout_write=lambda _t: None,
    )
    assert result["calls_created"] == 0
    assert client.creates == []
    assert result["reason"] == "no_confirmation"


def test_sequential_continue_then_abort(tmp_path, monkeypatch):
    _env(monkeypatch)
    client = FakeTwilio()
    result = run_campaign(
        live=True,
        prompt_fn=_prompts(CONFIRMATION_PHRASE, "Continue", "Abort"),
        twilio_factory=lambda *_a: client,
        fetch_status=lambda sid: {"sid": sid, "status": "completed", "duration": 20},
        store=SessionStore(base_dir=tmp_path),
        providers=_providers(),
        poll_seconds=0,
        call_timeout_seconds=1,
        sleep_fn=lambda _s: None,
        stdout_write=lambda _t: None,
    )
    assert result["calls_created"] == 2
    assert [row["alias"] for row in result["results"][:2]] == [
        "sarvam_conversational",
        "deepseek_flagship",
    ]
    assert result["results"][2]["status"] == "ABORTED_BY_OPERATOR"
    assert result["results"][3]["status"] == "ABORTED_BY_OPERATOR"
    assert all("/eval/voice" in item["url"] for item in client.creates)
    assert all(item["record"] is True for item in client.creates)
    assert all("/eval/recording" in item["recording_status_callback"] for item in client.creates)


def test_skip_does_not_create_next_call(tmp_path, monkeypatch):
    _env(monkeypatch)
    client = FakeTwilio()
    result = run_campaign(
        live=True,
        prompt_fn=_prompts(CONFIRMATION_PHRASE, "Skip", "Abort"),
        twilio_factory=lambda *_a: client,
        fetch_status=lambda sid: {"sid": sid, "status": "completed", "duration": 9},
        store=SessionStore(base_dir=tmp_path),
        providers=_providers(),
        poll_seconds=0,
        call_timeout_seconds=1,
        sleep_fn=lambda _s: None,
        stdout_write=lambda _t: None,
    )
    assert result["calls_created"] == 1
    assert result["results"][1]["status"] == "SKIPPED_BY_OPERATOR"
    assert result["results"][2]["status"] == "ABORTED_BY_OPERATOR"


def test_failed_call_is_incomplete_without_retry(tmp_path, monkeypatch):
    _env(monkeypatch)
    client = FakeTwilio(status="failed")
    result = run_campaign(
        live=True,
        prompt_fn=_prompts(CONFIRMATION_PHRASE, "Abort"),
        twilio_factory=lambda *_a: client,
        fetch_status=lambda sid: {"sid": sid, "status": "failed", "duration": 0},
        store=SessionStore(base_dir=tmp_path),
        providers=_providers(),
        poll_seconds=0,
        call_timeout_seconds=1,
        sleep_fn=lambda _s: None,
        stdout_write=lambda _t: None,
    )
    assert result["calls_created"] == 1
    assert result["results"][0]["status"] == "failed"
    assert result["results"][0]["incomplete"] is True
    assert len(client.creates) == 1


def test_timeout_marks_incomplete_without_retry(tmp_path, monkeypatch):
    _env(monkeypatch)
    client = FakeTwilio()
    result = run_campaign(
        live=True,
        prompt_fn=_prompts(CONFIRMATION_PHRASE, "Abort"),
        twilio_factory=lambda *_a: client,
        fetch_status=lambda sid: {"sid": sid, "status": "in-progress", "duration": None},
        store=SessionStore(base_dir=tmp_path),
        providers=_providers(),
        poll_seconds=0,
        call_timeout_seconds=0.01,
        sleep_fn=lambda _s: None,
        stdout_write=lambda _t: None,
    )
    assert result["calls_created"] == 1
    assert result["results"][0]["status"] == "timeout"
    assert result["results"][0]["incomplete"] is True
    assert len(client.creates) == 1


def test_cli_dry_run_exit_zero():
    assert main(["--dry-run"], stdout_write=lambda _t: None) == 0

"""Twilio mock tests for eval webhooks, recording association, and CallSid isolation."""

from __future__ import annotations

import asyncio

from live_voice.handlers import handle_recording, handle_speech, start_call
from live_voice.runtime import EvalRuntime, set_runtime
from live_voice.session_store import SessionStore


class FakeProvider:
    def __init__(self, alias: str, model_id: str, answer: str):
        self.alias = alias
        self.model_id = model_id
        self.answer = answer
        self.utterances: list[str] = []
        self.histories: list[list[dict]] = []

    def generate(self, **kwargs):
        self.utterances.append(kwargs["customer_utterance"])
        self.histories.append(list(kwargs["conversation_history"]))
        return {
            "provider": "fake",
            "model_id": self.model_id,
            "answer": self.answer,
            "intent": "LOCATION_QUERY",
            "action": "ANSWER",
            "schema_valid": True,
            "latency_ms": 55,
            "error": None,
            "error_type": None,
            "error_message": None,
        }


def _bind(store: SessionStore, *, run_id: str, alias: str, model_id: str, token: str = "tok"):
    store.register_pending(
        {
            "campaign_id": "camp1",
            "evaluation_run_id": run_id,
            "alias": alias,
            "provider": alias.split("_")[0],
            "model_id": model_id,
            "token": token,
            "consumed": False,
        }
    )


def test_recording_associates_to_matching_callsid_only(tmp_path):
    sarvam = FakeProvider("sarvam_conversational", "sarvam-105b-conversations", "Near Electronic City.")
    deepseek = FakeProvider("deepseek_flagship", "deepseek-flash", "Starting at 1.8 Crore.")
    store = SessionStore(base_dir=tmp_path)
    _bind(store, run_id="run_a", alias="sarvam_conversational", model_id="sarvam-105b-conversations", token="a")
    _bind(store, run_id="run_b", alias="deepseek_flagship", model_id="deepseek-flash", token="b")
    runtime = EvalRuntime(
        store=store,
        providers={"sarvam_conversational": sarvam, "deepseek_flagship": deepseek},
        system_prompt="test",
        retrieved_context="kb",
    )

    async def turn_runner(job):
        generated = runtime.providers[job["alias"]].generate(
            system_prompt=job["system_prompt"],
            conversation_history=job["history"],
            customer_utterance=job["utterance"],
            retrieved_context=job["retrieved_context"],
        )
        return {
            "spoken": generated["answer"],
            "fallback_used": False,
            "latency_ms": generated["latency_ms"],
            "error_type": None,
            "error_class": None,
            "schema_valid": True,
            "candidate_declared_intent": generated["intent"],
            "candidate_declared_action": generated["action"],
            "spawn_overrun": False,
            "deadline_exceeded": False,
            "error_message": None,
        }

    runtime.turn_runner = turn_runner
    set_runtime(runtime)
    start_call({"run_id": "run_a", "token": "a"}, {"CallSid": "CAaaa"})
    start_call({"run_id": "run_b", "token": "b"}, {"CallSid": "CAbbb"})
    asyncio.run(handle_speech({"CallSid": "CAaaa", "SpeechResult": "Where is it located?"}))
    asyncio.run(handle_speech({"CallSid": "CAbbb", "SpeechResult": "How much does it cost?"}))

    ok = handle_recording(
        {"CallSid": "CAaaa", "RecordingSid": "REaaa", "RecordingUrl": "https://api.twilio.com/re/aaa", "RecordingStatus": "completed"}
    )
    rejected = handle_recording(
        {"CallSid": "CAunknown", "RecordingSid": "RExxx", "RecordingUrl": "https://api.twilio.com/re/xxx", "RecordingStatus": "completed"}
    )
    missing = handle_recording({"CallSid": "CAbbb", "RecordingStatus": "absent"})

    assert ok["ok"] is True
    assert ok["ref"]["model_id"] == "sarvam-105b-conversations"
    assert ok["ref"]["evaluation_run_id"] == "run_a"
    assert rejected["ok"] is False
    assert missing["recording_status"] == "UNAVAILABLE"

    rec_a = (tmp_path / "camp1" / "run_a" / "recording.ref.json").read_text(encoding="utf-8")
    assert "REaaa" in rec_a
    assert "sarvam-105b-conversations" in rec_a
    rec_b = (tmp_path / "camp1" / "run_b" / "recording.ref.json").read_text(encoding="utf-8")
    assert "UNAVAILABLE" in rec_b
    assert "deepseek-flash" in rec_b


def test_wrong_token_does_not_bind_session(tmp_path):
    store = SessionStore(base_dir=tmp_path)
    _bind(store, run_id="run_a", alias="sarvam_conversational", model_id="sarvam-105b-conversations", token="secret")
    set_runtime(EvalRuntime(store=store, providers={}))
    xml = start_call({"run_id": "run_a", "token": "wrong"}, {"CallSid": "CAzzz"})
    assert "not active" in xml
    assert store.get_by_call_sid("CAzzz") is None

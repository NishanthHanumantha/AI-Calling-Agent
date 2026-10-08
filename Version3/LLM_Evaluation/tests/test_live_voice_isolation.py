"""Per-CallSid and per-model isolation. Production files must remain unchanged."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from live_voice.handlers import handle_speech, start_call
from live_voice.runtime import EvalRuntime, set_runtime
from live_voice.session_store import SessionStore

REPO = Path(__file__).resolve().parents[2]


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
            "intent": "PROJECT_FAQ",
            "action": "ANSWER",
            "schema_valid": True,
            "latency_ms": 11,
            "error": None,
            "error_type": None,
            "error_message": None,
        }


def test_histories_and_transcripts_do_not_leak_across_calls(tmp_path):
    left = FakeProvider("sarvam_conversational", "sarvam-105b-conversations", "Answer from Sarvam.")
    right = FakeProvider("claude_flagship", "claude-opus-4-8", "Answer from Opus.")
    store = SessionStore(base_dir=tmp_path)
    store.register_pending(
        {
            "campaign_id": "camp-iso",
            "evaluation_run_id": "run_sarvam",
            "alias": "sarvam_conversational",
            "provider": "sarvam",
            "model_id": "sarvam-105b-conversations",
            "token": "t1",
            "consumed": False,
        }
    )
    store.register_pending(
        {
            "campaign_id": "camp-iso",
            "evaluation_run_id": "run_opus",
            "alias": "claude_flagship",
            "provider": "claude",
            "model_id": "claude-opus-4-8",
            "token": "t2",
            "consumed": False,
        }
    )
    runtime = EvalRuntime(
        store=store,
        providers={"sarvam_conversational": left, "claude_flagship": right},
        system_prompt="sys",
        retrieved_context="ctx",
    )

    async def turn_runner(job):
        provider = runtime.providers[job["alias"]]
        generated = provider.generate(
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
    start_call({"run_id": "run_sarvam", "token": "t1"}, {"CallSid": "CA1"})
    start_call({"run_id": "run_opus", "token": "t2"}, {"CallSid": "CA2"})
    asyncio.run(handle_speech({"CallSid": "CA1", "SpeechResult": "Where is it located?"}))
    asyncio.run(handle_speech({"CallSid": "CA2", "SpeechResult": "How much does it cost?"}))
    asyncio.run(handle_speech({"CallSid": "CA1", "SpeechResult": "Can I visit this Saturday?"}))

    assert left.utterances == ["Where is it located?", "Can I visit this Saturday?"]
    assert right.utterances == ["How much does it cost?"]
    second_history = left.histories[1]
    texts = " ".join(item["content"] for item in second_history)
    assert "Where is it located?" in texts
    assert "How much does it cost?" not in texts
    assert "Answer from Opus." not in texts

    t1 = (tmp_path / "camp-iso" / "run_sarvam" / "transcript.jsonl").read_text(encoding="utf-8")
    t2 = (tmp_path / "camp-iso" / "run_opus" / "transcript.jsonl").read_text(encoding="utf-8")
    assert "sarvam-105b-conversations" in t1
    assert "claude-opus-4-8" not in t1
    assert "claude-opus-4-8" in t2
    assert "sarvam-105b-conversations" not in t2
    assert '"call_sid": "CA1"' in t1
    assert '"call_sid": "CA2"' in t2
    assert '"campaign_id": "camp-iso"' in t1


def test_production_app_and_make_call_unchanged():
    app = (REPO / "app_v3.py").read_text(encoding="utf-8")
    make_call = (REPO / "make_call.py").read_text(encoding="utf-8")
    assert 'MODEL = "sarvam-105b-conversations"' in app
    assert "mount_eval_routes" not in app
    assert "LIVE_EVAL_ENABLED" not in app
    assert "/eval/" not in app
    assert "/eval/" not in make_call
    assert "record=" not in make_call
    assert "voice_url = PUBLIC_URL.rstrip(\"/\") + \"/voice\"" in make_call
    live_voice = REPO / "LLM_Evaluation" / "live_voice"
    for path in live_voice.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import app_v3" not in text
        assert "from app_v3" not in text
        assert "import make_call" not in text


def test_live_voice_artifacts_are_gitignored():
    probe = "LLM_Evaluation/output/live_voice_runs/campaign/run/transcript.jsonl"
    completed = subprocess.run(
        ["git", "check-ignore", "-q", probe],
        cwd=REPO,
        check=False,
    )
    assert completed.returncode == 0
    env_probe = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=REPO, check=False)
    assert env_probe.returncode == 0

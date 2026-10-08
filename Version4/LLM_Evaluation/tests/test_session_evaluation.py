from __future__ import annotations

import json
from io import StringIO

from interactive.cli import handle_command
from interactive.conversation import STAGE_LABELS
from interactive.evaluator import (
    EVAL_STATUS_MISSING_GROUND_TRUTH,
    EVAL_STATUS_MISSING_RESPONSE,
    evaluate_turn,
    flatten_evaluation_records,
    session_summary,
)
from interactive.outbound import OUTBOUND_BASELINE, outbound_expected_stage
from interactive.schemas import DISPLAY_ORDER, TurnResponse
from interactive.session import InteractiveSession


class FakeProvider:
    def __init__(self, alias: str, answer: str, latency_ms=40, latencies=None, intent="PROJECT_FAQ", action="ANSWER"):
        self.alias = alias
        self.answer = answer
        self.intent = intent
        self.action = action
        self.latency_ms = latency_ms
        self.latencies = list(latencies) if latencies is not None else None
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        latency = self.latencies.pop(0) if self.latencies is not None else self.latency_ms
        return {
            "provider": "fake",
            "model_id": f"fake-{self.alias}",
            "answer": self.answer,
            "intent": self.intent,
            "action": self.action,
            "language": "en",
            "raw_response": self.answer,
            "schema_valid": True,
            "usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
            "latency_ms": latency,
            "error": None,
            "error_type": None,
            "error_message": None,
        }


def _providers(latencies=None) -> dict[str, FakeProvider]:
    answers = {
        "sarvam_conversational": "Townpark is near Electronic City.",
        "deepseek_flagship": "Located on Hosur Road, Bengaluru.",
        "claude_sonnet": "SOBHA Townpark is near Electronic City.",
        "claude_flagship": "It is near Electronic City on Hosur Road.",
    }
    out = {}
    for alias, text in answers.items():
        kwargs = {"alias": alias, "answer": text}
        if latencies and alias in latencies:
            kwargs["latencies"] = latencies[alias]
        out[alias] = FakeProvider(**kwargs)
    return out


def _session(providers=None, mode="customer", opening="fixed") -> InteractiveSession:
    return InteractiveSession(
        providers=providers if providers is not None else _providers(),
        conversation_mode=mode,
        opening_mode=opening,
    )


def _response(alias: str, answer: str, latency_ms=10) -> TurnResponse:
    return TurnResponse(
        alias=alias,
        provider=alias.split("_")[0],
        model_id=f"fake-{alias}",
        answer=answer,
        intent="PROJECT_FAQ",
        action="ANSWER",
        language="en",
        stage=None,
        raw_response=answer,
        latency_ms=latency_ms,
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
        error_type=None,
        error_message=None,
        schema_valid=True,
    )


def _records(session: InteractiveSession) -> list[dict]:
    return flatten_evaluation_records(session.evaluations, session_id=session.run_id)


def test_full_session_evaluates_every_turn_and_model():
    session = _session()
    session.generate_turn("Hi, I am interested in Sobha Townpark.")
    session.generate_turn("Where is it located?")
    session.generate_turn("How much does a 3 BHK cost?")
    coverage = session.evaluate_session()
    records = _records(session)
    n_turns = len(session.turns)
    n_models = len(DISPLAY_ORDER)
    assert n_turns == 3
    assert n_models == 4
    assert len(records) == n_turns * n_models
    assert coverage["expected"] == n_turns * n_models
    assert coverage["record_count"] == n_turns * n_models
    assert coverage["skipped"] == 0
    for record in records:
        assert record["session_id"] == session.run_id
        assert record["turn_id"] in {1, 2, 3}
        assert record["model_id"]
        assert record["provider"]
        assert record["customer_message"]
        assert record["model_response"]


def test_outbound_baseline_shape_is_turns_times_models(tmp_path):
    session = _session(mode="outbound", opening="fixed")
    session.begin_outbound()
    for utterance in OUTBOUND_BASELINE:
        session.generate_turn(utterance)
    n_turns = len(session.turns)
    n_models = len(DISPLAY_ORDER)
    assert n_turns == 1 + len(OUTBOUND_BASELINE)
    coverage = session.evaluate_session()
    records = _records(session)
    assert len(records) == n_turns * n_models
    assert coverage["expected"] == n_turns * n_models
    exported = session.export(tmp_path)
    eval_lines = (exported / "evaluations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    response_lines = (exported / "model_responses.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(eval_lines) == len(response_lines) == n_turns * n_models


def test_single_turn_evaluate_still_works():
    session = _session()
    session.generate_turn("Hi, I am interested in Sobha Townpark.")
    session.generate_turn("Where is the project located?")
    ev = session.evaluate_latest()
    assert ev is not None
    assert ev["turn_index"] == 2
    records = _records(session)
    assert len(records) == len(DISPLAY_ORDER)
    assert {row["turn_id"] for row in records} == {2}
    assert ev["ground_truth"] == "L003"
    labeled = [row for row in ev["models"] if row.get("ground_truth") != "NOT AVAILABLE"]
    assert labeled
    buf = StringIO()
    handle_command(session, "/evaluate", "", buf.write)
    assert "TURN 02 — EVALUATION" in buf.getvalue()


def test_no_duplicate_turn_model_evaluations():
    session = _session()
    session.generate_turn("Hi, I am interested in Sobha Townpark.")
    session.generate_turn("Where is it located?")
    session.generate_turn("Can I visit on Saturday?")
    session.evaluate_session()
    session.evaluate_session()
    records = _records(session)
    keys = [(row["turn_id"], row["alias"]) for row in records]
    assert len(keys) == len(set(keys))
    assert len(keys) == len(session.turns) * len(DISPLAY_ORDER)


def test_session_metrics_use_all_turn_latencies():
    latencies = {
        "sarvam_conversational": [10, 20, 30],
        "deepseek_flagship": [11, 21, 31],
        "claude_sonnet": [12, 22, 32],
        "claude_flagship": [13, 23, 33],
    }
    session = _session(providers=_providers(latencies=latencies))
    session.generate_turn("Hi, I am interested in Sobha Townpark.")
    session.generate_turn("Where is it located?")
    session.generate_turn("How much does a 3 BHK cost?")
    session.evaluate_session()
    summary = session_summary(session.evaluations)
    sarvam = summary["sarvam_conversational"]
    assert sarvam["n_turns"] == 3
    assert sarvam["avg_latency"] == 20.0
    assert sarvam["p50_latency"] == 20
    assert sarvam["p95_latency"] == 30
    session.evaluate_latest()
    latest_only = session_summary([session.evaluations[-1]])
    assert latest_only["sarvam_conversational"]["avg_latency"] == 30
    full = session_summary(session.evaluations)
    assert full["sarvam_conversational"]["avg_latency"] == 20.0


def test_missing_response_is_not_silently_dropped():
    session = _session()
    session.generate_turn("Where is it located?")
    session.turns[0]["responses"] = [
        item for item in session.turns[0]["responses"] if item["alias"] != "claude_flagship"
    ]
    coverage = session.evaluate_session()
    records = _records(session)
    assert len(records) == len(DISPLAY_ORDER)
    missing = [row for row in records if row["alias"] == "claude_flagship"]
    assert len(missing) == 1
    assert missing[0]["evaluation_status"] == EVAL_STATUS_MISSING_RESPONSE
    assert missing[0]["intent_correct"] is None
    assert coverage["skipped"] >= 1
    assert coverage["expected"] == len(DISPLAY_ORDER)


def test_final_turn_next_step_is_not_faq():
    stage, allowed = outbound_expected_stage("Okay what is the next step?", 8)
    assert STAGE_LABELS[stage] != "FAQ"
    assert STAGE_LABELS[stage] in {"Closure", "Slot", "Visit"}
    assert "FAQ" not in [STAGE_LABELS.get(key, key) for key in allowed]
    session = _session(mode="outbound")
    ev = evaluate_turn(
        "Okay what is the next step?",
        [_response(alias, "We will send the visit confirmation shortly.") for alias in DISPLAY_ORDER],
        session.dataset,
        {alias: [] for alias in DISPLAY_ORDER},
        8,
        session.retrieved_context,
        outbound=True,
    )
    assert ev["models"][0]["expected_stage"] != "FAQ"
    assert ev["models"][0]["expected_stage"] in {"Closure", "Slot", "Visit"}


def test_missing_ground_truth_is_not_model_failure():
    session = _session()
    session.generate_turn("Can you sing a song about cricket?")
    ev = session.evaluate_latest()
    for model in ev["models"]:
        assert model["evaluation_status"] == EVAL_STATUS_MISSING_GROUND_TRUTH
        assert model["intent_correct"] is None
        assert model["action_correct"] is None
        assert model.get("api_error") is not True


def test_evaluate_session_command_covers_all_turns(tmp_path):
    session = _session()
    session.generate_turn("Hi, I am interested in Sobha Townpark.")
    session.generate_turn("Where is it located?")
    session.generate_turn("How much does a 3 BHK cost?")
    buf = StringIO()
    handle_command(session, "/evaluate", "session", buf.write)
    records = _records(session)
    assert len(records) == len(session.turns) * len(DISPLAY_ORDER)
    text = buf.getvalue()
    assert "Expected:" in text
    assert "Evaluated:" in text
    assert "Skipped:" in text
    exported_dir = session.export(tmp_path)
    lines = (exported_dir / "evaluations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(records)
    parsed = [json.loads(line) for line in lines]
    assert all("session_id" in row and "turn_id" in row and "model_id" in row for row in parsed)

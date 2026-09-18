from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from interactive.cli import dry_run_report, handle_command, parse_args
from interactive.commands import parse_command
from interactive.conversation import detect_language, match_golden_case
from interactive.evaluator import _classification_prf, evaluate_turn, session_summary
from interactive.renderer import banner, redact, render_debug, render_turn
from interactive.schemas import DISPLAY_ORDER, TurnResponse
from interactive.session import InteractiveSession
from models.parsing import extract_json_object, validate_structured_output

ROOT = Path(__file__).resolve().parents[1]


class FakeProvider:
    def __init__(self, alias: str, answer: str, intent="PROJECT_FAQ", action="ANSWER", language="en", error=None, raise_exc=False):
        self.alias = alias
        self.answer = answer
        self.intent = intent
        self.action = action
        self.language = language
        self.error = error
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc:
            raise RuntimeError("provider exploded")
        if self.error:
            return {
                "provider": "fake",
                "model_id": f"fake-{self.alias}",
                "answer": None,
                "intent": None,
                "action": None,
                "language": None,
                "raw_response": "",
                "schema_valid": False,
                "usage": {},
                "latency_ms": 12,
                "error": self.error,
                "error_type": "API_ERROR",
                "error_message": self.error,
            }
        return {
            "provider": "fake",
            "model_id": f"fake-{self.alias}",
            "answer": self.answer,
            "intent": self.intent,
            "action": self.action,
            "language": self.language,
            "raw_response": json.dumps(
                {"intent": self.intent, "language": self.language, "answer": self.answer, "action": self.action}
            ),
            "schema_valid": True,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "latency_ms": 42,
            "error": None,
            "error_type": None,
            "error_message": None,
            "confidence": None,
        }


def _providers(**answers) -> dict[str, FakeProvider]:
    defaults = {
        "sarvam_conversational": "Townpark is near Electronic City.",
        "sarvam_flagship": "Located on Hosur Road, Bengaluru.",
        "claude_sonnet": "SOBHA Townpark is in Bengaluru near Electronic City.",
        "claude_flagship": "It is near Electronic City on Hosur Road.",
    }
    defaults.update(answers)
    return {alias: FakeProvider(alias, text) for alias, text in defaults.items()}


def make_session(providers=None, dry_run=False) -> InteractiveSession:
    return InteractiveSession(dry_run=dry_run, providers=providers if providers is not None else _providers())


def test_conversation_history_isolation():
    providers = _providers(
        sarvam_conversational="Answer A",
        sarvam_flagship="Answer B",
        claude_sonnet="Answer C",
        claude_flagship="Answer D",
    )
    session = make_session(providers)
    session.generate_turn("Hi, I am interested in Sobha Townpark.")
    session.generate_turn("Where is it located?")
    for alias, provider in providers.items():
        second = provider.calls[1]
        history = second["conversation_history"]
        texts = " ".join(item["content"] for item in history)
        own = provider.answer
        others = [p.answer for a, p in providers.items() if a != alias]
        assert own in texts
        for other in others:
            assert other not in texts


def test_same_user_message_sent_to_all_models():
    providers = _providers()
    session = make_session(providers)
    msg = "Kannada alli location heli."
    session.generate_turn(msg)
    utterances = [p.calls[0]["customer_utterance"] for p in providers.values()]
    assert utterances == [msg, msg, msg, msg]


def test_same_knowledge_context_sent_to_all_models():
    providers = _providers()
    session = make_session(providers)
    session.generate_turn("Where is it located?")
    contexts = [p.calls[0]["retrieved_context"] for p in providers.values()]
    assert len(set(contexts)) == 1
    assert "SOBHA Townpark" in contexts[0]
    payloads = session.preview_payloads("What configurations are available?")
    assert all("SOBHA Townpark" in text for text in payloads.values())


def test_turn_numbering():
    session = make_session()
    assert session.next_turn_index() == 1
    session.generate_turn("Hi")
    session.generate_turn("Where is it located?")
    assert [t["index"] for t in session.turns] == [1, 2]
    assert session.next_turn_index() == 3


def test_reset_clears_histories():
    session = make_session()
    session.generate_turn("Hi, I am interested in Sobha Townpark.")
    session.reset()
    assert session.turns == []
    assert all(hist == [] for hist in session.histories.values())
    assert session.next_turn_index() == 1


def test_command_parsing():
    assert parse_command("/help") == ("/help", "")
    assert parse_command("/evaluate session") == ("/evaluate", "session")
    assert parse_command("/exit") == ("/quit", "")
    assert parse_command("Hello there") == (None, "Hello there")


def test_error_isolation():
    providers = _providers()
    providers["claude_sonnet"].raise_exc = True
    session = make_session(providers)
    responses = session.generate_turn("Where is it located?")
    by_alias = {r.alias: r for r in responses}
    assert by_alias["claude_sonnet"].error_type == "API_ERROR"
    assert by_alias["sarvam_conversational"].answer
    assert by_alias["sarvam_flagship"].answer
    assert by_alias["claude_flagship"].answer
    rendered = render_turn(1, "Where is it located?", responses)
    assert "STATUS: ERROR" in rendered
    assert "SARVAM — CONVERSATIONAL" in rendered


def test_response_schema_parsing():
    parsed = extract_json_object(
        '{"intent":"LOCATION_QUERY","language":"en","answer":"Near Electronic City.","action":"ANSWER"}'
    )
    valid, fields = validate_structured_output(parsed)
    assert valid is True
    assert fields["intent"] == "LOCATION_QUERY"
    assert fields["action"] == "ANSWER"


def test_session_export(tmp_path):
    session = make_session()
    session.generate_turn("Where is the project located?")
    session.evaluate_latest()
    out = session.export(tmp_path)
    assert (out / "conversation.json").exists()
    assert (out / "model_responses.jsonl").exists()
    assert (out / "evaluations.jsonl").exists()
    assert (out / "session_summary.csv").exists()
    assert (out / "session_summary.xlsx").exists()
    data = json.loads((out / "conversation.json").read_text(encoding="utf-8"))
    assert data["twilio"] == "DISABLED"


def test_ground_truth_matching():
    session = make_session()
    gold = match_golden_case("Where is the project located?", session.dataset)
    assert gold is not None
    assert gold["test_id"] == "L003"
    session.generate_turn("Where is the project located?")
    ev = session.evaluate_latest()
    assert ev["ground_truth"] == "L003"


def test_missing_ground_truth_handling():
    session = make_session()
    assert match_golden_case("Can you sing a song about cricket?", session.dataset) is None
    session.generate_turn("Can you sing a song about cricket?")
    ev = session.evaluate_latest()
    assert ev["ground_truth"] == "NOT AVAILABLE"
    for model in ev["models"]:
        if not model.get("api_error"):
            assert model["intent_correct"] is None
            assert model["action_correct"] is None


def test_metric_calculation():
    pairs = [("LOCATION_QUERY", "LOCATION_QUERY"), ("PRICE_QUERY", "PRICE_QUERY"), ("PRICE_QUERY", "PROJECT_FAQ")]
    prf = _classification_prf(pairs)
    assert prf["accuracy"] == pytest.approx(0.6667, rel=1e-3)
    assert prf["f1"] is not None
    responses = [
        TurnResponse(
            alias=alias,
            provider="fake",
            model_id="x",
            answer="Near Electronic City on Hosur Road, Bengaluru.",
            intent="LOCATION_QUERY",
            action="ANSWER",
            language="en",
            stage=None,
            raw_response="{}",
            latency_ms=10,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            error_type=None,
            error_message=None,
            schema_valid=True,
        )
        for alias in DISPLAY_ORDER
    ]
    session = make_session()
    ev = evaluate_turn(
        "Where is the project located?",
        responses,
        session.dataset,
        {a: [] for a in DISPLAY_ORDER},
        1,
        session.retrieved_context,
    )
    summary = session_summary([ev])
    assert summary["claude_sonnet"]["intent"]["accuracy"] == 1.0


def test_multilingual_input_handling():
    providers = _providers()
    session = make_session(providers)
    msg = "Townpark ka starting price kya hai for 2 BHK?"
    assert detect_language(msg) == "mixed"
    session.generate_turn(msg)
    assert all(p.calls[0]["customer_utterance"] == msg for p in providers.values())
    ev = session.evaluate_latest()
    assert ev["ground_truth"] == "L010"
    assert ev["detected_language"] == "mixed"


def test_no_api_key_leakage():
    text = redact("Authorization Bearer sk-secret-token-value and api_key=abcd1234")
    assert "sk-secret-token-value" not in text
    assert "[REDACTED]" in text
    payload = render_debug({"headers": {"Authorization": "Bearer sk-abc123456789"}, "api_key": "sk-abc123456789"})
    assert "sk-abc123456789" not in payload


def test_no_twilio_invocation():
    interactive_dir = ROOT / "interactive"
    for path in interactive_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from twilio" not in source
        assert "import twilio" not in source
        assert "Client(" not in source
    session = make_session()
    assert "DISABLED" in banner()
    assert "INTERACTIVE EVALUATION" in banner()


def test_handle_commands_and_models_status():
    session = make_session()
    session.generate_turn("Hi")
    buf = StringIO()
    handle_command(session, "/models", "", buf.write)
    assert "Model ID:" in buf.getvalue()
    captured = StringIO()
    handle_command(session, "/history", "", captured.write)
    assert "Turn 01" in captured.getvalue()


def test_dry_run_zero_api_calls():
    providers = _providers()
    session = InteractiveSession(dry_run=True, providers=providers)
    report = dry_run_report(session)
    assert "API calls made: 0" in report
    assert all(p.calls == [] for p in providers.values())
    assert session.turns  # dry-run still exercises the turn pipeline without live APIs


def test_cli_parse_flags():
    args = parse_args(["--dry-run", "--check-models", "--demo"])
    assert args.dry_run and args.check_models and args.demo

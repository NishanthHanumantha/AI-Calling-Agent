from __future__ import annotations

import json

from interactive.cli import dry_run_report, parse_args
from interactive.conversation import detect_language
from interactive.evaluator import evaluate_turn, session_summary
from interactive.outbound import (
    FIXED_OPENING,
    OPENING_INSTRUCTION,
    OUTBOUND_BASELINE,
    evaluate_opening,
    exception_notes,
    opening_elements,
)
from interactive.renderer import banner, render_outbound_opening, render_turn
from interactive.schemas import DISPLAY_ORDER, TurnResponse
from interactive.session import InteractiveSession


class FakeProvider:
    def __init__(self, alias: str, answer: str, intent="GENERAL", action="ANSWER", raise_exc=False):
        self.alias = alias
        self.answer = answer
        self.intent = intent
        self.action = action
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc:
            raise RuntimeError("provider exploded")
        return {
            "provider": "fake",
            "model_id": f"fake-{self.alias}",
            "answer": self.answer,
            "intent": self.intent,
            "action": self.action,
            "language": "en",
            "raw_response": self.answer,
            "schema_valid": True,
            "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
            "latency_ms": 30,
            "error": None,
            "error_type": None,
            "error_message": None,
        }


def _openings() -> dict[str, str]:
    return {
        "sarvam_conversational": (
            "Hi, this is Sobha Limited calling about SOBHA Townpark, a New York-inspired luxury "
            "community near Electronic City on Hosur Road in Bengaluru. Is now a good time?"
        ),
        "deepseek_flagship": (
            "Hello, I'm calling from Sobha Limited regarding SOBHA Townpark, a luxury New York-themed "
            "community on Hosur Road near Electronic City, Bengaluru. Do you have a moment?"
        ),
        "claude_sonnet": (
            "Hi, Sobha Limited here. I'm calling about SOBHA Townpark, a New York-inspired luxury "
            "project near Electronic City, Hosur Road, Bengaluru. Is this a good time for a quick call?"
        ),
        "claude_flagship": (
            "Good morning, this is Sobha Limited. I'm reaching out about SOBHA Townpark, a luxury "
            "New York-inspired community near Electronic City on Hosur Road in Bengaluru. Is now a good time?"
        ),
    }


def _providers(**overrides) -> dict[str, FakeProvider]:
    answers = _openings()
    providers = {alias: FakeProvider(alias, text) for alias, text in answers.items()}
    for alias, answer in overrides.items():
        providers[alias].answer = answer
    return providers


def _session(mode="outbound", opening="generated", providers=None, dry_run=False) -> InteractiveSession:
    return InteractiveSession(
        dry_run=dry_run,
        providers=providers if providers is not None else _providers(),
        conversation_mode=mode,
        opening_mode=opening,
    )


def _response(alias: str, answer: str, intent=None, action=None, error_type=None) -> TurnResponse:
    return TurnResponse(
        alias=alias,
        provider="fake",
        model_id="x",
        answer=answer,
        intent=intent,
        action=action,
        language="en",
        stage=None,
        raw_response=answer or "",
        latency_ms=None if error_type else 15,
        input_tokens=None if error_type else 3,
        output_tokens=None if error_type else 2,
        total_tokens=None if error_type else 5,
        error_type=error_type,
        error_message="HTTP 500" if error_type else None,
        schema_valid=error_type is None,
    )


def test_customer_mode_remains_default():
    args = parse_args([])
    assert args.mode == "customer"
    assert args.opening == "generated"
    session = InteractiveSession(dry_run=True)
    assert session.conversation_mode == "customer"
    assert "OUTBOUND" not in banner()
    assert "INTERACTIVE EVALUATION" in banner()
    assert "DeepSeek — V4.1 Flash" in banner()


def test_generated_opening_is_independent():
    providers = _providers()
    session = _session(providers=providers)
    opening = session.begin_outbound()
    instructions = [item.calls[0]["customer_utterance"] for item in providers.values()]
    systems = [item.calls[0]["system_prompt"] for item in providers.values()]
    contexts = [item.calls[0]["retrieved_context"] for item in providers.values()]
    histories = [item.calls[0]["conversation_history"] for item in providers.values()]
    assert instructions == [OPENING_INSTRUCTION] * 4
    assert len(set(systems)) == 1
    assert len(set(contexts)) == 1
    assert histories == [[], [], [], []]
    texts = [resp.answer for resp in opening]
    assert len(set(texts)) == 4
    for alias, provider in providers.items():
        stored = session.histories[alias]
        assert stored == [{"role": "assistant", "content": provider.answer}]
        assert OPENING_INSTRUCTION not in stored[0]["content"]
        for other in providers.values():
            if other is not provider:
                assert other.answer not in stored[0]["content"]
    rendered = render_outbound_opening(opening)
    assert "OUTBOUND OPENING" in rendered
    assert "SARVAM — CONVERSATIONAL" in rendered
    assert "DEEPSEEK — V4.1 FLASH" in rendered
    assert "CLAUDE — OPUS" in rendered


def test_fixed_opening_is_identical_and_offline():
    providers = _providers()
    session = _session(opening="fixed", providers=providers)
    opening = session.begin_outbound()
    assert all(provider.calls == [] for provider in providers.values())
    assert [resp.answer for resp in opening] == [FIXED_OPENING] * 4
    assert [session.histories[alias][0]["content"] for alias in DISPLAY_ORDER] == [FIXED_OPENING] * 4
    elements = opening_elements(FIXED_OPENING)
    assert set(elements.values()) == {"PASS"}
    rendered = render_outbound_opening(opening)
    assert FIXED_OPENING in rendered


def test_same_customer_message_after_opening():
    providers = _providers()
    session = _session(providers=providers)
    session.begin_outbound()
    message = "Yes, I have a couple of minutes."
    session.generate_turn(message)
    utterances = [provider.calls[1]["customer_utterance"] for provider in providers.values()]
    contexts = [provider.calls[1]["retrieved_context"] for provider in providers.values()]
    systems = [provider.calls[1]["system_prompt"] for provider in providers.values()]
    assert utterances == [message] * 4
    assert len(set(contexts)) == 1
    assert len(set(systems)) == 1
    for alias, provider in providers.items():
        history = provider.calls[1]["conversation_history"]
        assert history[0]["content"] == provider.answer
        for other in providers.values():
            if other is not provider:
                assert other.answer not in history[0]["content"]


def test_opening_evaluation_and_chatbot_penalty():
    good = [_response(alias, _openings()[alias]) for alias in DISPLAY_ORDER]
    ev = evaluate_opening(good, "SOBHA Townpark near Electronic City", "generated")
    first = ev["models"][0]
    assert first["opening_element_company"] == "PASS"
    assert first["opening_element_project"] == "PASS"
    assert first["opening_element_description"] == "PASS"
    assert first["opening_element_location"] == "PASS"
    assert first["opening_element_permission"] == "PASS"
    assert first["intent_correct"] is None
    assert 1 <= first["opening_conciseness"] <= 5
    assert first["outbound_appropriateness"] >= 4
    chatbot = evaluate_opening(
        [_response("sarvam_conversational", "How can I help you?")],
        "kb",
        "generated",
    )
    assert chatbot["models"][0]["outbound_appropriateness"] == 1
    failed = evaluate_opening(
        [_response("deepseek_flagship", None, error_type="API_ERROR")],
        "kb",
        "generated",
    )
    assert failed["models"][0]["api_error"] is True
    assert failed["models"][0]["hallucination"] is None


def test_intent_and_action_metrics_when_labelled():
    session = _session(dry_run=True)
    utterance = "Where is the project located?"
    responses = [
        _response(alias, "It is near Electronic City on Hosur Road, Bengaluru.", "LOCATION_QUERY", "ANSWER")
        for alias in DISPLAY_ORDER
    ]
    ev = evaluate_turn(utterance, responses, session.dataset, {alias: [] for alias in DISPLAY_ORDER}, 2, session.retrieved_context, outbound=True)
    assert ev["ground_truth"] != "NOT AVAILABLE"
    summary = session_summary([ev])
    stats = summary["sarvam_conversational"]
    assert stats["intent"]["accuracy"] == 1.0
    assert stats["intent"]["precision"] is not None
    assert stats["intent"]["recall"] is not None
    assert stats["intent"]["f1"] is not None
    assert stats["intent"]["confusion"]
    assert stats["action"]["accuracy"] == 1.0
    assert stats["action"]["f1"] is not None
    assert stats["action"]["confusion"]
    unlabeled = evaluate_turn(
        "Can you sing a song about cricket?",
        responses,
        session.dataset,
        {alias: [] for alias in DISPLAY_ORDER},
        3,
        session.retrieved_context,
        outbound=True,
    )
    assert unlabeled["ground_truth"] == "NOT AVAILABLE"
    assert unlabeled["models"][0]["intent_correct"] is None


def test_stage_context_grounding_and_exceptions():
    session = _session(dry_run=True)
    visit = evaluate_turn(
        "Can I visit this Saturday?",
        [_response(alias, "Yes, we can arrange a site visit on Saturday.", "SITE_VISIT", "BOOK_VISIT") for alias in DISPLAY_ORDER],
        session.dataset,
        {alias: [] for alias in DISPLAY_ORDER},
        5,
        session.retrieved_context,
        outbound=True,
    )
    assert visit["models"][0]["expected_stage"] == "Visit"
    assert visit["models"][0]["stage_correct"] is True
    slot = evaluate_turn(
        "Morning would be better.",
        [_response(alias, "Saturday morning at 10:00 AM works for the visit.", "CONFIRMATION", "CONFIRM_SLOT") for alias in DISPLAY_ORDER],
        session.dataset,
        {alias: [] for alias in DISPLAY_ORDER},
        6,
        session.retrieved_context,
        outbound=True,
    )
    assert "Slot" in slot["models"][0]["allowed_stages"]
    assert slot["models"][0]["stage_correct"] is True
    history = {
        alias: [
            {"role": "user", "content": "I'm looking for a 3 BHK."},
            {"role": "assistant", "content": "A 3 BHK is available."},
        ]
        for alias in DISPLAY_ORDER
    }
    kept = evaluate_turn(
        "How much does it cost?",
        [_response(alias, "The 3 BHK starts at INR 1.8 Crore onwards.", "PRICE_QUERY", "ANSWER") for alias in DISPLAY_ORDER],
        session.dataset,
        history,
        4,
        session.retrieved_context,
        outbound=True,
    )
    assert kept["models"][0]["context_used"] is True
    assert kept["models"][0]["context_error"] is False
    invented = evaluate_turn(
        "What appreciation will I get over the next five years?",
        [_response(alias, "You will get 25% appreciation guaranteed.", "UNKNOWN", "ANSWER") for alias in DISPLAY_ORDER],
        session.dataset,
        {alias: [] for alias in DISPLAY_ORDER},
        4,
        session.retrieved_context,
        outbound=True,
    )
    assert invented["models"][0]["hallucination"] is True
    assert invented["models"][0]["unsupported_claims"]
    empty = evaluate_turn(
        "What appreciation will I get over the next five years?",
        [_response(alias, "", "UNKNOWN", "ANSWER") for alias in DISPLAY_ORDER],
        session.dataset,
        {alias: [] for alias in DISPLAY_ORDER},
        4,
        session.retrieved_context,
        outbound=True,
    )
    assert empty["models"][0]["hallucination"] is False
    busy = exception_notes("I'm actually busy right now.", "No problem, I can call you back later.")
    assert busy["branch"] == "busy"
    assert busy["callback_handling"] is True
    assert busy["continued_sales_pitch"] is False
    declined = exception_notes("I'm not interested.", "No problem, thank you for your time.")
    assert declined["acknowledged"] is True
    assert declined["aggressive_persuasion"] is False
    competitor = exception_notes("Another project is offering a lower price. Why should I consider Townpark?", "Townpark is near Electronic City.")
    assert competitor["branch"] == "competitor"


def test_multilingual_customer_text_is_not_translated():
    providers = _providers()
    session = _session(opening="fixed", providers=providers)
    session.begin_outbound()
    message = "Location yelli ide?"
    assert detect_language(message) == "mixed"
    assert detect_language("3 BHK price eshtu?") == "mixed"
    session.generate_turn(message)
    assert all(provider.calls[0]["customer_utterance"] == message for provider in providers.values())


def test_error_isolation_and_export(tmp_path):
    providers = _providers()
    providers["claude_sonnet"].raise_exc = True
    session = _session(providers=providers)
    opening = session.begin_outbound()
    by_alias = {item.alias: item for item in opening}
    assert by_alias["claude_sonnet"].error_type == "API_ERROR"
    assert by_alias["sarvam_conversational"].answer
    assert by_alias["deepseek_flagship"].answer
    assert by_alias["claude_flagship"].answer
    session.generate_turn("Yes, I have a couple of minutes.")
    session.evaluations.clear()
    session.evaluations.append(evaluate_opening(
        [TurnResponse(**item) for item in session.turns[0]["responses"]],
        session.retrieved_context,
        "generated",
    ))
    out = session.export(tmp_path)
    conversation = json.loads((out / "conversation.json").read_text(encoding="utf-8"))
    assert conversation["conversation_mode"] == "outbound"
    assert conversation["opening_mode"] == "generated"
    assert conversation["twilio"] == "DISABLED"
    assert (out / "model_responses.jsonl").exists()
    assert (out / "evaluations.jsonl").exists()
    assert (out / "session_summary.csv").exists()
    csv_text = (out / "session_summary.csv").read_text(encoding="utf-8")
    assert "opening_element_company" in csv_text
    rendered = render_turn(2, "Yes, I have a couple of minutes.", opening)
    assert "STATUS: ERROR" in rendered


def test_outbound_dry_run_makes_no_api_calls():
    providers = _providers()
    session = _session(dry_run=True, providers=providers)
    report = dry_run_report(session)
    assert "API calls made: 0" in report
    assert "OUTBOUND INTERACTIVE EVALUATION" in report
    assert "Opening mode: generated" in report
    assert all(provider.calls == [] for provider in providers.values())
    fixed = _session(opening="fixed", dry_run=True, providers=_providers())
    fixed_report = dry_run_report(fixed)
    assert "API calls made: 0" in fixed_report
    assert "Fixed opening identical: True" in fixed_report


def test_no_twilio_and_cli_flags():
    args = parse_args(["--mode", "outbound", "--opening", "fixed", "--dry-run"])
    assert args.mode == "outbound" and args.opening == "fixed" and args.dry_run
    text = banner("outbound", "fixed")
    assert "MODE: OFFLINE / OUTBOUND INTERACTIVE EVALUATION" in text
    assert "TWILIO: DISABLED" in text
    assert "Sarvam — Flagship" not in text
    source = (OUTBOUND_BASELINE and "busy") or ""
    assert "I'm actually busy right now." not in source
    interactive = __import__("pathlib").Path(__file__).resolve().parents[1] / "interactive"
    for path in interactive.glob("*.py"):
        code = path.read_text(encoding="utf-8")
        assert "import twilio" not in code
        assert "from twilio" not in code


def test_baseline_is_not_auto_executed():
    assert len(OUTBOUND_BASELINE) == 7
    assert OUTBOUND_BASELINE[0] == "Yes, I have a couple of minutes."
    assert OUTBOUND_BASELINE[-1].startswith("Okay, I'll discuss it")
    assert parse_args(["--mode", "outbound"]).demo is False

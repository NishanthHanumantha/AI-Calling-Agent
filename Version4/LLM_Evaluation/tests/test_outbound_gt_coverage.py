"""LLM-EVAL.3.1D — latest 11-turn outbound ground-truth coverage."""

from __future__ import annotations

from interactive.evaluator import (
    EVAL_STATUS_MISSING_GROUND_TRUTH,
    evaluate_turn,
    flatten_evaluation_records,
    session_summary,
)
from interactive.outbound_ground_truth import (
    LATEST_BENCHMARK_CUSTOMER_TURNS,
    infer_outbound_case,
    lookup_calibration_case,
    requires_outbound_ground_truth,
)
from interactive.renderer import render_session_summary
from interactive.schemas import DISPLAY_ORDER, TurnResponse
from interactive.session import InteractiveSession
from interactive.visit_policy import CORRECT_DATE_REQUEST, CORRECT_FINAL_CONFIRMATION, CORRECT_SLOT_OFFER


class FakeProvider:
    def __init__(self, alias: str, answer: str, intent="QUALIFICATION", action="QUALIFY"):
        self.alias = alias
        self.answer = answer
        self.intent = intent
        self.action = action
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
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
            "latency_ms": 25,
            "error": None,
            "error_type": None,
            "error_message": None,
        }


def _session() -> InteractiveSession:
    providers = {alias: FakeProvider(alias, "Noted.") for alias in DISPLAY_ORDER}
    return InteractiveSession(providers=providers, conversation_mode="outbound", opening_mode="fixed")


def _response(alias: str, answer: str, intent: str, action: str) -> TurnResponse:
    return TurnResponse(
        alias=alias,
        provider=alias.split("_")[0],
        model_id=f"fake-{alias}",
        answer=answer,
        intent=intent,
        action=action,
        language="en",
        stage=None,
        raw_response=answer,
        latency_ms=12,
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
        error_type=None,
        error_message=None,
        schema_valid=True,
    )


def _all(answer: str, intent: str, action: str) -> list[TurnResponse]:
    return [_response(alias, answer, intent, action) for alias in DISPLAY_ORDER]


def _hist(utterances: list[str]) -> dict[str, list[dict[str, str]]]:
    hist = []
    for text in utterances:
        hist.append({"role": "user", "content": text})
        hist.append({"role": "assistant", "content": "Noted."})
    return {alias: list(hist) for alias in DISPLAY_ORDER}


def _eval(utterance: str, turn: int, answer: str, intent: str, action: str, prior: list[str]):
    session = _session()
    return evaluate_turn(
        utterance,
        _all(answer, intent, action),
        session.dataset,
        _hist(prior),
        turn,
        session.retrieved_context,
        outbound=True,
        required_aliases=list(DISPLAY_ORDER),
    )


LATEST_EXPECTED = {
    2: ("QUALIFICATION", "QUALIFY", "Qualification"),
    3: ("QUALIFICATION", "QUALIFY", "Qualification"),
    4: ("QUALIFICATION", "QUALIFY", "Qualification"),
    5: ("QUALIFICATION", "QUALIFY", "Qualification"),
    6: ("AMENITIES_QUERY", "ANSWER", "FAQ"),
    7: ("GENERAL", "QUALIFY", "Qualification"),
    8: ("SITE_VISIT", "CLARIFY", "Visit"),
    9: ("SITE_VISIT", "BOOK_VISIT", "Visit"),
    10: ("SITE_VISIT", "CONFIRM_SLOT", "Slot"),
    11: ("CONFIRMATION", "ANSWER", "Closure"),
}


def test_latest_eleven_turn_sequence_labels_and_coverage():
    assert len(LATEST_BENCHMARK_CUSTOMER_TURNS) == 10
    session = _session()
    session.begin_outbound()
    for utterance in LATEST_BENCHMARK_CUSTOMER_TURNS:
        session.generate_turn(utterance)
    coverage = session.evaluate_session()
    records = flatten_evaluation_records(session.evaluations, session_id=session.run_id)
    assert coverage["expected"] == 11 * 4
    assert coverage["evaluated"] == 11 * 4
    assert coverage["skipped"] == 0
    assert len(records) == 44

    opening = [row for row in records if row.get("turn_id") == 1]
    customer = [row for row in records if row.get("turn_id") != 1]
    assert len(opening) == 4
    assert all(row.get("ground_truth_applicable") is False for row in opening)
    assert all(row.get("expected_intent") is None for row in opening)
    assert len(customer) == 40
    assert all(row.get("ground_truth_applicable") is True for row in customer)
    assert all(row.get("evaluation_status") != EVAL_STATUS_MISSING_GROUND_TRUTH for row in customer)

    for row in customer:
        intent, action, stage = LATEST_EXPECTED[row["turn_id"]]
        assert row["expected_intent"] == intent
        assert row["expected_action"] == action
        assert row["expected_stage"] == stage

    for turn_id in range(2, 12):
        labels = {
            (row["expected_intent"], row["expected_action"], row["expected_stage"])
            for row in customer
            if row["turn_id"] == turn_id
        }
        assert len(labels) == 1

    summary = session_summary(session.evaluations)
    applicable = 10 * 4
    for alias in DISPLAY_ORDER:
        stats = summary[alias]
        assert stats["evaluation_coverage"] == 1.0
        assert stats["ground_truth_coverage"] == 1.0
        assert stats["intentionally_unlabelled_turns"] == 1
        assert stats["unexpected_missing_ground_truth"] == 0
        assert stats["n_turns"] == 11
    assert applicable == len(customer)

    rendered = render_session_summary(session.evaluations)
    assert "Evaluation Coverage" in rendered
    assert "Ground Truth Coverage" in rendered
    assert "Intentionally Unlabelled Turns" in rendered
    assert "Unexpected Missing Ground Truth" in rendered
    assert "No winner ranking" in rendered


def test_latest_turn_id_binding_survives_punctuation():
    assert lookup_calibration_case("Yes.", 2)["expected_intent"] == "QUALIFICATION"
    gold = infer_outbound_case("2Cr.", 3, ["Yes"])
    assert gold["expected_intent"] == "QUALIFICATION"
    gold = infer_outbound_case("When can I visit?", 8, LATEST_BENCHMARK_CUSTOMER_TURNS[:6])
    assert gold["expected_action"] == "CLARIFY"
    gold = infer_outbound_case("Friday.", 9, LATEST_BENCHMARK_CUSTOMER_TURNS[:7])
    assert gold["expected_action"] == "BOOK_VISIT"
    gold = infer_outbound_case("4pm.", 10, LATEST_BENCHMARK_CUSTOMER_TURNS[:8])
    assert gold["expected_action"] == "CONFIRM_SLOT"


def test_stateful_ground_truth_does_not_copy_previous_turn():
    amenities = infer_outbound_case(
        "Okay I would like to know about project amenities",
        6,
        LATEST_BENCHMARK_CUSTOMER_TURNS[:4],
    )
    after = infer_outbound_case("Okay", 7, LATEST_BENCHMARK_CUSTOMER_TURNS[:5])
    assert amenities["expected_intent"] == "AMENITIES_QUERY"
    assert after["expected_intent"] != "AMENITIES_QUERY"
    assert after["expected_intent"] == "GENERAL"
    closed = infer_outbound_case("Okay", 11, LATEST_BENCHMARK_CUSTOMER_TURNS[:9])
    confirm = infer_outbound_case("4pm", 10, LATEST_BENCHMARK_CUSTOMER_TURNS[:8])
    assert confirm["expected_action"] == "CONFIRM_SLOT"
    assert closed["expected_action"] == "ANSWER"
    assert closed["conversation_stage"] == "closed"


def test_visit_flow_for_latest_benchmark():
    visit = _eval(
        "When can I visit?",
        8,
        "Which day or date would you prefer for the visit?",
        "SITE_VISIT",
        "CLARIFY",
        LATEST_BENCHMARK_CUSTOMER_TURNS[:6],
    )
    assert visit["models"][0]["visit_sequence_code"] == CORRECT_DATE_REQUEST
    friday = _eval(
        "Friday",
        9,
        "Friday works. We have 10 AM, 11:30 AM, 4 PM and 6 PM. Which time would you prefer?",
        "SITE_VISIT",
        "BOOK_VISIT",
        LATEST_BENCHMARK_CUSTOMER_TURNS[:7],
    )
    assert friday["models"][0]["visit_sequence_code"] == CORRECT_SLOT_OFFER
    four = _eval(
        "4pm",
        10,
        "Your visit is confirmed for Friday at 4:00 PM.",
        "CONFIRMATION",
        "CONFIRM_SLOT",
        LATEST_BENCHMARK_CUSTOMER_TURNS[:8],
    )
    assert four["models"][0]["visit_sequence_code"] == CORRECT_FINAL_CONFIRMATION


def test_cricket_remains_missing_and_is_not_applicable():
    session = _session()
    unlabeled = evaluate_turn(
        "Can you sing a song about cricket?",
        _all("I can talk about Townpark instead.", "GENERAL", "ANSWER"),
        session.dataset,
        {alias: [] for alias in DISPLAY_ORDER},
        3,
        session.retrieved_context,
        outbound=True,
    )
    assert unlabeled["ground_truth"] == "NOT AVAILABLE"
    assert unlabeled["models"][0]["evaluation_status"] == EVAL_STATUS_MISSING_GROUND_TRUTH
    assert unlabeled["models"][0]["intent_correct"] is None
    assert unlabeled["models"][0]["ground_truth_applicable"] is False
    assert requires_outbound_ground_truth("Can you sing a song about cricket?") is False
    assert requires_outbound_ground_truth("2Cr") is True
    assert requires_outbound_ground_truth("Timeline is not yet decided") is True
    assert requires_outbound_ground_truth("Okay") is True


SEQUENCE_A = [
    "Yes",
    "2Cr",
    "3BHK",
    "I would like to know about amenities",
    "Yes",
    "Friday",
    "4pm",
    "Okay",
]
SEQUENCE_B = [
    "Yes",
    "2Cr",
    "Okay",
    "3BHK",
    "I would like to know about amenities",
    "Okay",
    "Yes",
    "Friday",
    "4pm",
    "Okay",
]
SEQUENCE_C = [
    "Yes",
    "2Cr",
    "3BHK",
    "When can I visit?",
    "Friday",
    "4pm",
    "Okay",
]


def _run_sequence(turns: list[str]):
    session = _session()
    session.begin_outbound()
    for utterance in turns:
        session.generate_turn(utterance)
    coverage = session.evaluate_session()
    records = flatten_evaluation_records(session.evaluations, session_id=session.run_id)
    return session, coverage, records


def _customer_rows(records):
    return [row for row in records if row.get("turn_id") != 1]


def _labels_for_utterance(records, utterance: str) -> tuple[str, str, str]:
    rows = [row for row in _customer_rows(records) if (row.get("customer_message") or "") == utterance]
    assert rows, utterance
    sample = rows[-1]
    labels = {
        (row["expected_intent"], row["expected_action"], row["expected_stage"])
        for row in rows
        if row["turn_id"] == sample["turn_id"]
    }
    assert len(labels) == 1
    return next(iter(labels))


def test_sequence_a_variable_length_visit_states():
    session, coverage, records = _run_sequence(SEQUENCE_A)
    customer = _customer_rows(records)
    assert coverage["expected"] == 9 * 4
    assert coverage["evaluated"] == 36
    assert coverage["skipped"] == 0
    assert all(row.get("ground_truth_applicable") for row in customer)
    assert all(row.get("expected_intent") for row in customer)
    assert _labels_for_utterance(records, "Friday") == ("SITE_VISIT", "BOOK_VISIT", "Visit")
    assert _labels_for_utterance(records, "4pm") == ("SITE_VISIT", "CONFIRM_SLOT", "Slot")
    assert _labels_for_utterance(records, "Okay") == ("CONFIRMATION", "ANSWER", "Closure")
    summary = session_summary(session.evaluations)
    for alias in DISPLAY_ORDER:
        assert summary[alias]["ground_truth_coverage"] == 1.0
        assert summary[alias]["unexpected_missing_ground_truth"] == 0
        assert summary[alias]["evaluation_coverage"] == 1.0


def test_sequence_b_extra_acknowledgements_same_visit_gt():
    session, coverage, records = _run_sequence(SEQUENCE_B)
    assert coverage["skipped"] == 0
    assert _labels_for_utterance(records, "Friday") == ("SITE_VISIT", "BOOK_VISIT", "Visit")
    assert _labels_for_utterance(records, "4pm") == ("SITE_VISIT", "CONFIRM_SLOT", "Slot")
    assert _labels_for_utterance(records, "Okay") == ("CONFIRMATION", "ANSWER", "Closure")
    summary = session_summary(session.evaluations)
    for alias in DISPLAY_ORDER:
        assert summary[alias]["ground_truth_coverage"] == 1.0
        assert summary[alias]["unexpected_missing_ground_truth"] == 0


def test_sequence_c_explicit_visit_question_same_visit_gt():
    session, coverage, records = _run_sequence(SEQUENCE_C)
    assert coverage["skipped"] == 0
    assert _labels_for_utterance(records, "Friday") == ("SITE_VISIT", "BOOK_VISIT", "Visit")
    assert _labels_for_utterance(records, "4pm") == ("SITE_VISIT", "CONFIRM_SLOT", "Slot")
    assert _labels_for_utterance(records, "Okay") == ("CONFIRMATION", "ANSWER", "Closure")
    summary = session_summary(session.evaluations)
    for alias in DISPLAY_ORDER:
        assert summary[alias]["ground_truth_coverage"] == 1.0
        assert summary[alias]["unexpected_missing_ground_truth"] == 0


def test_action_scoring_4pm_after_friday_is_confirm_slot():
    prior = ["Yes", "2Cr", "3BHK", "I would like to know about amenities", "Yes", "Friday"]
    ev = _eval("4pm", 8, "Your visit is confirmed for Friday at 4:00 PM.", "SITE_VISIT", "CONFIRM_SLOT", prior)
    model = ev["models"][0]
    assert model["expected_intent"] == "SITE_VISIT"
    assert model["expected_action"] == "CONFIRM_SLOT"
    assert model["expected_stage"] == "Slot"
    assert model["ground_truth"] != "OB-T08"
    assert model["action_correct"] is True
    assert model["intent_correct"] is True


def test_action_scoring_okay_after_confirmed_visit_allows_restated_confirm():
    prior = ["Yes", "2Cr", "3BHK", "I would like to know about amenities", "Yes", "Friday", "4pm"]
    restated = _eval("Okay", 9, "Your visit is confirmed for Friday at 4:00 PM.", "CONFIRMATION", "CONFIRM_SLOT", prior)
    model = restated["models"][0]
    assert model["expected_intent"] == "CONFIRMATION"
    assert model["expected_action"] == "ANSWER"
    assert model["expected_stage"] == "Closure"
    assert model["ground_truth"] != "OB-T09"
    assert model["action_correct"] is True
    reopen = _eval(
        "Okay",
        9,
        "We have 10 AM, 11:30 AM, 4 PM and 6 PM. Which one works?",
        "SITE_VISIT",
        "BOOK_VISIT",
        prior,
    )
    assert reopen["models"][0]["expected_action"] == "ANSWER"
    assert reopen["models"][0]["action_correct"] is False

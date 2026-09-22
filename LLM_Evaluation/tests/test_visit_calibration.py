from __future__ import annotations

from interactive.conversation import match_golden_case
from interactive.evaluator import (
    EVAL_STATUS_MISSING_GROUND_TRUTH,
    evaluate_turn,
    flatten_evaluation_records,
    session_summary,
)
from interactive.outbound_ground_truth import (
    CALIBRATION_CUSTOMER_TURNS,
    infer_outbound_case,
    lookup_calibration_case,
)
from interactive.schemas import DISPLAY_ORDER, TurnResponse
from interactive.session import InteractiveSession
from interactive.visit_policy import (
    SITE_VISIT_SCHEDULING_RULE,
    evaluate_visit_sequence,
    offers_time_slots,
    visit_booking_state,
)


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


def _providers() -> dict[str, FakeProvider]:
    return {
        alias: FakeProvider(alias, "A 3 BHK starts at INR 1.8 Crore onwards.")
        for alias in DISPLAY_ORDER
    }


def _session() -> InteractiveSession:
    return InteractiveSession(
        providers=_providers(),
        conversation_mode="outbound",
        opening_mode="fixed",
    )


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


def test_calibration_ground_truth_covers_every_customer_turn():
    assert len(CALIBRATION_CUSTOMER_TURNS) == 9
    for index, utterance in enumerate(CALIBRATION_CUSTOMER_TURNS, start=2):
        gold = lookup_calibration_case(utterance, index)
        assert gold is not None, utterance
        assert gold["expected_intent"]
        assert gold["expected_action"]
        assert gold["conversation_stage"]


def test_turn_2_qualification():
    ev = _eval("Yes", 2, "Great, are you looking at a particular configuration?", "QUALIFICATION", "QUALIFY", [])
    model = ev["models"][0]
    assert model["ground_truth"] == "OB-T02"
    assert model["evaluation_status"] != EVAL_STATUS_MISSING_GROUND_TRUTH
    assert model["expected_intent"] == "QUALIFICATION"
    assert model["intent_correct"] is True
    assert model["action_correct"] is True
    assert model["expected_stage"] == "Qualification"


def test_turn_3_configuration_and_budget():
    ev = _eval(
        "3BHK around 1.5-2Cr",
        3,
        "A 3 BHK is available, and Townpark starts at INR 1.8 Crore onwards.",
        "QUALIFICATION",
        "QUALIFY",
        ["Yes"],
    )
    model = ev["models"][0]
    assert model["ground_truth"] == "OB-T03"
    assert model["intent_correct"] is True
    assert model["action_correct"] is True
    assert model["factual_accuracy"] == 1.0
    assert model["expected_stage"] == "Qualification"


def test_turn_6_amenities_query():
    ev = _eval(
        "I would like to know more about project amenities",
        6,
        "Townpark has clubhouses, swimming pools, and landscaped gardens.",
        "AMENITIES_QUERY",
        "ANSWER",
        ["Yes", "3BHK around 1.5-2Cr", "Okay", "Okay"],
    )
    model = ev["models"][0]
    assert model["ground_truth"] == "OB-T06"
    assert model["expected_intent"] == "AMENITIES_QUERY"
    assert model["intent_correct"] is True
    assert model["action_correct"] is True
    assert model["expected_stage"] == "FAQ"
    assert model["factual_accuracy"] == 1.0
    assert model["grounded"] is True


def test_turn_7_visit_agreement_asks_date_not_slots():
    prior = CALIBRATION_CUSTOMER_TURNS[:5]
    good = _eval(
        "Yes I would like to Visit",
        7,
        "Certainly. Which day or date would you prefer for the visit?",
        "SITE_VISIT",
        "CLARIFY",
        prior,
    )
    model = good["models"][0]
    assert model["ground_truth"] == "OB-T07"
    assert model["expected_action"] == "CLARIFY"
    assert model["intent_correct"] is True
    assert model["action_correct"] is True
    assert model["visit_rule"] == "ASK_DATE"
    assert model["visit_sequence_pass"] is True
    assert model["asked_for_date"] is True
    assert model["offered_time_slots"] is False
    bad = _eval(
        "Yes I would like to Visit",
        7,
        "We have slots at 10 AM, 11:30 AM, 4 PM and 6 PM. Which one works?",
        "SITE_VISIT",
        "BOOK_VISIT",
        prior,
    )
    assert bad["models"][0]["offered_time_slots"] is True
    assert bad["models"][0]["visit_sequence_pass"] is False


def test_turn_8_time_before_date_asks_date():
    prior = CALIBRATION_CUSTOMER_TURNS[:6]
    good = _eval(
        "4pm",
        8,
        "4 PM noted. Which day or date would you prefer?",
        "SITE_VISIT",
        "CLARIFY",
        prior,
    )
    model = good["models"][0]
    assert model["ground_truth"] == "OB-T08"
    assert model["visit_rule"] == "ASK_DATE"
    assert model["visit_sequence_pass"] is True
    assert model["premature_confirmation"] is False
    confirmed = _eval(
        "4pm",
        8,
        "Perfect, your 4 PM visit is confirmed.",
        "CONFIRMATION",
        "CONFIRM_SLOT",
        prior,
    )
    assert confirmed["models"][0]["premature_confirmation"] is True
    assert confirmed["models"][0]["visit_sequence_pass"] is False
    assert confirmed["models"][0]["action_correct"] is False


def test_turn_9_missing_date_no_premature_confirmation():
    prior = CALIBRATION_CUSTOMER_TURNS[:7]
    good = _eval(
        "Okay",
        9,
        "Just to lock this in, which day would you like to visit?",
        "SITE_VISIT",
        "CLARIFY",
        prior,
    )
    assert good["models"][0]["ground_truth"] == "OB-T09"
    assert good["models"][0]["visit_sequence_pass"] is True
    assert good["models"][0]["premature_confirmation"] is False
    bad = _eval(
        "Okay",
        9,
        "Great, you are booked for 4 PM.",
        "CONFIRMATION",
        "CONFIRM_SLOT",
        prior,
    )
    assert bad["models"][0]["premature_confirmation"] is True
    assert bad["models"][0]["visit_sequence_pass"] is False


def test_turn_10_date_then_offer_slots():
    prior = CALIBRATION_CUSTOMER_TURNS[:8]
    good = _eval(
        "Thursday",
        10,
        "Thursday works. We have 10 AM, 11:30 AM, 4 PM and 6 PM available. Which time would you prefer?",
        "SITE_VISIT",
        "BOOK_VISIT",
        prior,
    )
    model = good["models"][0]
    assert model["ground_truth"] == "OB-T10"
    assert model["visit_rule"] == "OFFER_SLOTS"
    assert model["visit_sequence_pass"] is True
    assert model["offered_time_slots"] is True
    assert model["premature_confirmation"] is False
    confirmed = _eval(
        "Thursday",
        10,
        "Perfect, your Thursday 4 PM visit is confirmed.",
        "CONFIRMATION",
        "CONFIRM_SLOT",
        prior,
    )
    assert confirmed["models"][0]["visit_sequence_pass"] is False
    assert confirmed["models"][0]["premature_confirmation"] is True


def test_slots_not_offered_immediately_after_visit_agreement():
    answer = "We have 10 AM, 11:30 AM, 4 PM and 6 PM. Which one works?"
    assert offers_time_slots(answer) is True
    state = visit_booking_state(["Yes I would like to Visit"], "Yes I would like to Visit")
    scored = evaluate_visit_sequence(state, answer)
    assert scored["visit_rule"] == "ASK_DATE"
    assert scored["visit_sequence_pass"] is False


def test_confirmation_requires_date_then_time():
    history = [
        "Yes I would like to Visit",
        "Thursday",
        "4pm",
    ]
    state = visit_booking_state(history[:-1], "4pm")
    assert state.ready_to_confirm is True
    assert state.expected_rule == "CONFIRM"
    premature_state = visit_booking_state(["Yes I would like to Visit", "4pm"], "4pm")
    assert premature_state.ready_to_confirm is False
    assert premature_state.expected_rule == "ASK_DATE"


def test_short_yes_is_not_slot_confirmation_gold():
    session = _session()
    assert match_golden_case("Yes", session.dataset) is None


def test_full_calibration_session_has_labels_and_no_duplicates():
    session = _session()
    session.begin_outbound()
    for utterance in CALIBRATION_CUSTOMER_TURNS:
        session.generate_turn(utterance)
    coverage = session.evaluate_session()
    records = flatten_evaluation_records(session.evaluations, session_id=session.run_id)
    assert len(session.turns) == 10
    assert len(records) == 10 * 4
    assert coverage["expected"] == 40
    customer = [row for row in records if row.get("turn_id") != 1]
    assert len(customer) == 36
    assert all(row.get("ground_truth") not in {None, "NOT AVAILABLE"} for row in customer)
    assert all(row.get("evaluation_status") != EVAL_STATUS_MISSING_GROUND_TRUTH for row in customer)
    keys = [(row["turn_id"], row["alias"]) for row in records]
    assert len(keys) == len(set(keys))
    summary = session_summary(session.evaluations)
    sarvam = summary["sarvam_conversational"]
    assert sarvam["intent"]["accuracy"] is not None
    assert sarvam["n_turns"] == 10
    assert sarvam["avg_latency"] is not None


def test_same_policy_applied_to_all_four_models():
    session = _session()
    assert "SITE_VISIT_SCHEDULING_RULE" in session.system_prompt
    session.begin_outbound()
    session.generate_turn("Yes I would like to Visit")
    prompts = [provider.calls[-1]["system_prompt"] for provider in session._providers.values()]
    assert len(set(prompts)) == 1
    assert SITE_VISIT_SCHEDULING_RULE in prompts[0]


def test_missing_response_and_missing_ground_truth_still_work():
    session = _session()
    session.begin_outbound()
    session.generate_turn("Yes")
    session.turns[-1]["responses"] = [
        item for item in session.turns[-1]["responses"] if item["alias"] != "claude_flagship"
    ]
    session.evaluate_session()
    records = flatten_evaluation_records(session.evaluations, session_id=session.run_id)
    missing = [row for row in records if row["alias"] == "claude_flagship" and row["turn_id"] == 2]
    assert missing and missing[0]["evaluation_status"] == "MISSING_RESPONSE"
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


def test_book_visit_label_is_accepted_when_asking_for_date():
    ev = _eval(
        "Yes I would like to Visit",
        7,
        "Absolutely. What day would work best for you?",
        "SITE_VISIT",
        "BOOK_VISIT",
        CALIBRATION_CUSTOMER_TURNS[:5],
    )
    assert ev["models"][0]["action_correct"] is True
    assert ev["models"][0]["visit_sequence_pass"] is True


def test_infer_case_for_amenities_without_exact_turn_index():
    gold = infer_outbound_case("Tell me about the amenities", 4, [])
    assert gold is not None
    assert gold["expected_intent"] == "AMENITIES_QUERY"

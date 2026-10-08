"""Explicit outbound ground truth for interactive evaluation.

Uses existing intent/action/stage vocabulary. ASK_DATE is represented as CLARIFY.
OFFER_SLOTS is represented as BOOK_VISIT. CONFIRM remains CONFIRM_SLOT.
"""

from __future__ import annotations

import re
from typing import Any

from evaluator.deterministic import normalize_text

from .visit_policy import visit_booking_state

# Current calibrated outbound session (run 20260922_121834). Opening is turn 1.
CALIBRATION_CUSTOMER_TURNS = [
    "Yes",
    "3BHK around 1.5-2Cr",
    "Okay",
    "Okay",
    "I would like to know more about project amenities",
    "Yes I would like to Visit",
    "4pm",
    "Okay",
    "Thursday",
]

# Latest calibrated outbound session (run 20260923_054702). Opening is turn 1.
LATEST_BENCHMARK_CUSTOMER_TURNS = [
    "Yes",
    "2Cr",
    "Timeline is not yet decided",
    "Okay",
    "Okay I would like to know about project amenities",
    "Okay",
    "When can I visit?",
    "Friday",
    "4pm",
    "Okay",
]


_BUSINESS_RE = re.compile(
    r"\b(visit|appointment|amenit|bhk|budget|crore|price|cost|located|location|"
    r"timeline|possession|slot|configuration|looking for|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|"
    r"tomorrow|today|morning|evening|\d{1,2}(?::\d{2})?\s*(?:am|pm))\b"
)
_ACK = {"yes", "yeah", "yep", "ok", "okay", "sure", "alright", "all right", "great", "fine"}
_SHORT_KEYS = _ACK | {"thursday", "4 pm", "4pm"}


def _norm_key(utterance: str) -> str:
    text = normalize_text(utterance or "")
    text = text.replace("3bhk", "3 bhk").replace("2bhk", "2 bhk").replace("1bhk", "1 bhk").replace("4bhk", "4 bhk")
    text = re.sub(r"\b(\d{1,2})(am|pm)\b", r"\1 \2", text)
    text = re.sub(r"[,]+", " ", text)
    text = re.sub(r"[?!.;:]+$", "", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*cr(?:ore)?s?\b", r"\1 crore", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in {"ok", "k", "okk"}:
        return "okay"
    return text


def _compatible(actual: str, expected: str) -> bool:
    """Utterance may vary slightly; short acks must still match exactly."""
    if not actual or not expected:
        return False
    if actual == expected:
        return True
    if expected in _SHORT_KEYS or actual in _SHORT_KEYS or len(expected.split()) <= 2:
        return actual == expected
    ta, tb = set(actual.split()), set(expected.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.55


def _case(
    test_id: str,
    *,
    intent: str,
    action: str,
    stage: str,
    allowed_stages: list[str] | None = None,
    acceptable_intents: list[str] | None = None,
    acceptable_actions: list[str] | None = None,
    expected_facts: list[str] | None = None,
    facts_any: list[str] | None = None,
    criteria: str,
    category: str | None = None,
) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "category": category or intent,
        "language": "en",
        "conversation_stage": stage,
        "allowed_stage_keys": allowed_stages or [stage],
        "expected_intent": intent,
        "acceptable_intents": acceptable_intents or [intent],
        "expected_action": action,
        "acceptable_actions": acceptable_actions or [action],
        "expected_facts": expected_facts or [],
        "facts_any": facts_any or [],
        "acceptable_answer_criteria": criteria,
    }


# Turn index is 1-based including the opening turn.
CALIBRATION_CASES: dict[tuple[int, str], dict[str, Any]] = {
    (2, "yes"): _case(
        "OB-T02",
        intent="QUALIFICATION",
        action="QUALIFY",
        stage="qualify",
        allowed_stages=["qualify", "greeting"],
        acceptable_intents=["QUALIFICATION", "GENERAL", "CLARIFY"],
        acceptable_actions=["QUALIFY", "CLARIFY", "ANSWER"],
        criteria="After the customer accepts the opening, continue qualification. Do not treat this as slot confirmation.",
    ),
    (3, "3 bhk around 1.5-2cr"): _case(
        "OB-T03",
        intent="QUALIFICATION",
        action="QUALIFY",
        stage="qualify",
        allowed_stages=["qualify", "answer_faq"],
        acceptable_intents=["QUALIFICATION", "CONFIG_QUERY", "PRICE_QUERY"],
        acceptable_actions=["QUALIFY", "ANSWER"],
        expected_facts=["3 BHK"],
        facts_any=["3 BHK", "INR 1.8 Crore", "1.8 crore", "1.5", "2 crore"],
        criteria="Acknowledge 3 BHK and the stated 1.5–2 Cr budget. Starting price in context is INR 1.8 Crore onwards. Do not invent a rejection number.",
    ),
    (4, "okay"): _case(
        "OB-T04",
        intent="GENERAL",
        action="QUALIFY",
        stage="qualify",
        allowed_stages=["qualify", "answer_faq"],
        acceptable_intents=["GENERAL", "CLARIFY", "QUALIFICATION"],
        acceptable_actions=["QUALIFY", "CLARIFY", "ANSWER"],
        criteria="Ambiguous acknowledgement during qualification. Stay in qualification. Do not jump to an unrelated FAQ or confirm a visit.",
    ),
    (5, "okay"): _case(
        "OB-T05",
        intent="GENERAL",
        action="QUALIFY",
        stage="qualify",
        allowed_stages=["qualify", "answer_faq"],
        acceptable_intents=["GENERAL", "CLARIFY", "QUALIFICATION"],
        acceptable_actions=["QUALIFY", "CLARIFY", "ANSWER"],
        criteria="Second acknowledgement. Preserve qualification state. Do not force an unrelated FAQ intent.",
    ),
    (6, "i would like to know more about project amenities"): _case(
        "OB-T06",
        intent="AMENITIES_QUERY",
        action="ANSWER",
        stage="answer_faq",
        acceptable_intents=["AMENITIES_QUERY", "PROJECT_FAQ"],
        acceptable_actions=["ANSWER"],
        expected_facts=["clubhouses"],
        facts_any=[
            "clubhouses",
            "swimming pools",
            "sports courts",
            "landscaped gardens",
            "kids play areas",
            "forest grove",
            "camping grounds",
        ],
        criteria="Answer amenities from Townpark knowledge. Harmless wording differences are allowed. Do not invent amenities.",
        category="AMENITIES_QUERY",
    ),
    (7, "yes i would like to visit"): _case(
        "OB-T07",
        intent="SITE_VISIT",
        action="CLARIFY",
        stage="propose_slot",
        allowed_stages=["propose_slot", "visit_day"],
        acceptable_intents=["SITE_VISIT"],
        acceptable_actions=["CLARIFY", "BOOK_VISIT"],
        criteria="Customer agreed to a visit. ASK for day/date first. Do not offer time slots yet. ASK_DATE is the existing CLARIFY action.",
        category="SITE_VISIT",
    ),
    (8, "4 pm"): _case(
        "OB-T08",
        intent="SITE_VISIT",
        action="CLARIFY",
        stage="propose_slot",
        allowed_stages=["propose_slot", "visit_pick_slot"],
        acceptable_intents=["SITE_VISIT", "CLARIFY", "CONFIRMATION"],
        acceptable_actions=["CLARIFY", "BOOK_VISIT"],
        criteria="Time was given before day/date. Retain 4 PM and ASK for the day/date. Do not confirm the visit.",
        category="SITE_VISIT",
    ),
    (9, "okay"): _case(
        "OB-T09",
        intent="SITE_VISIT",
        action="CLARIFY",
        stage="propose_slot",
        allowed_stages=["propose_slot", "visit_pick_slot"],
        acceptable_intents=["SITE_VISIT", "CLARIFY", "GENERAL"],
        acceptable_actions=["CLARIFY", "BOOK_VISIT"],
        criteria="Day/date is still missing. Continue asking for it. Detect premature confirmation.",
        category="SITE_VISIT",
    ),
    (10, "thursday"): _case(
        "OB-T10",
        intent="SITE_VISIT",
        action="BOOK_VISIT",
        stage="propose_slot",
        allowed_stages=["propose_slot", "visit_pick_slot"],
        acceptable_intents=["SITE_VISIT", "CONFIRMATION"],
        acceptable_actions=["BOOK_VISIT", "CLARIFY"],
        expected_facts=["10:00 AM"],
        facts_any=["10:00 AM", "11:30 AM", "4:00 PM", "6:00 PM"],
        criteria="Thursday supplies the missing day. Offer configured slots. Do not confirm until a time is validly selected after the date.",
        category="SITE_VISIT",
    ),
}

CALIBRATION_BY_TURN = {turn_index: case for (turn_index, _utt), case in CALIBRATION_CASES.items()}
CALIBRATION_BY_ID = {case["test_id"]: case for case in CALIBRATION_BY_TURN.values()}


def lookup_calibration_case(utterance: str, turn_index: int) -> dict[str, Any] | None:
    """Bind OB-T0N by stable turn index, with a punctuation-tolerant utterance check."""
    return bind_outbound_ground_truth(utterance, turn_index)


def _customer_texts(history: list[dict[str, str]] | list[str] | None) -> list[str]:
    texts: list[str] = []
    for item in history or []:
        if isinstance(item, str):
            texts.append(item)
            continue
        role = item.get("role")
        if role and role != "user":
            continue
        content = item.get("content") or item.get("utterance") or ""
        if content:
            texts.append(content)
    return texts


def _calibration_prefix_matches(history: list[dict[str, str]] | list[str] | None, turn_index: int) -> bool:
    expected_prior = CALIBRATION_CUSTOMER_TURNS[: max(0, int(turn_index) - 2)]
    if not expected_prior:
        return True
    prior = _customer_texts(history)
    if len(prior) != len(expected_prior):
        return False
    return all(_compatible(_norm_key(actual), _norm_key(expected)) for actual, expected in zip(prior, expected_prior))


def _conflicts_with_state(case: dict[str, Any], state) -> bool:
    test_id = str(case.get("test_id") or "")
    if test_id in {"OB-T08", "OB-T09"} and (state.has_date or state.ready_to_confirm):
        return True
    if test_id == "OB-T07" and state.ready_to_confirm:
        return True
    if test_id == "OB-T10" and state.ready_to_confirm:
        return True
    return False


def bind_outbound_ground_truth(
    utterance: str,
    turn_index: int,
    history: list[dict[str, str]] | list[str] | None = None,
    ground_truth_id: str | None = None,
) -> dict[str, Any] | None:
    """Explicit OB-Txx, then stateful resolver. Never match on the model response text."""
    state = visit_booking_state(history, utterance)
    if ground_truth_id and ground_truth_id in CALIBRATION_BY_ID:
        stamped = dict(CALIBRATION_BY_ID[ground_truth_id])
        if not _conflicts_with_state(stamped, state):
            return stamped
    idx = int(turn_index)
    if idx in CALIBRATION_BY_TURN:
        expected_utt = CALIBRATION_CUSTOMER_TURNS[idx - 2]
        if _compatible(_norm_key(utterance), _norm_key(expected_utt)):
            prior = _customer_texts(history)
            prefix_ok = not prior or _calibration_prefix_matches(history, idx)
            if prefix_ok:
                case = dict(CALIBRATION_BY_TURN[idx])
                if not _conflicts_with_state(case, state):
                    return case
    return _infer_outbound_heuristics(utterance, idx, history)


def infer_outbound_case(
    utterance: str,
    turn_index: int,
    history: list[dict[str, str]] | list[str] | None = None,
    ground_truth_id: str | None = None,
) -> dict[str, Any] | None:
    """Scenario GT first, then visit-state heuristics. Never invent labels for unrelated talk."""
    gold, _applicable = resolve_outbound_ground_truth(utterance, turn_index, history, ground_truth_id)
    return gold


def resolve_outbound_ground_truth(
    utterance: str,
    turn_index: int,
    history: list[dict[str, str]] | list[str] | None = None,
    ground_truth_id: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Return (gold, applicable). Opening/unrelated talk is not applicable."""
    gold = bind_outbound_ground_truth(utterance, turn_index, history, ground_truth_id)
    if gold:
        return gold, True
    return None, requires_outbound_ground_truth(utterance, history, turn_index)


def requires_outbound_ground_truth(
    utterance: str,
    history: list[dict[str, str]] | list[str] | None = None,
    turn_index: int = 1,
) -> bool:
    """True when a customer turn has a deterministic business expectation."""
    del turn_index
    text = _norm_key(utterance)
    if not text:
        return False
    if text in _ACK:
        return True
    if _BUSINESS_RE.search(text):
        return True
    state = visit_booking_state(history, utterance)
    return bool(state.visit_requested or state.has_date or state.has_time_preference)


def stamped_ground_truth_id(
    utterance: str,
    turn_index: int,
    history: list[dict[str, str]] | list[str] | None = None,
) -> str | None:
    """Stable OB-T0N id for storage. Matched later by id, not by model response text."""
    gold = bind_outbound_ground_truth(utterance, turn_index, history)
    test_id = str((gold or {}).get("test_id") or "")
    if test_id.startswith("OB-T"):
        return test_id
    return None


def _prior_customer_text(history: list[dict[str, str]] | list[str] | None) -> str:
    if not history:
        return ""
    for item in reversed(list(history)):
        if isinstance(item, str):
            return item
        role = item.get("role")
        if role and role != "user":
            continue
        return item.get("content") or item.get("utterance") or ""
    return ""


def _infer_outbound_heuristics(
    utterance: str,
    turn_index: int,
    history: list[dict[str, str]] | list[str] | None = None,
) -> dict[str, Any] | None:
    text = _norm_key(utterance)
    state = visit_booking_state(history, utterance)
    if text in _ACK:
        if state.ready_to_confirm:
            return _case(
                "OB-ACK-CLOSED",
                intent="CONFIRMATION",
                action="ANSWER",
                stage="closed",
                allowed_stages=["closed", "visit_pick_slot"],
                acceptable_intents=["CONFIRMATION", "GENERAL", "SITE_VISIT"],
                acceptable_actions=["ANSWER", "QUALIFY", "CONFIRM_SLOT"],
                criteria="Visit is already dated and timed. Acknowledge and close. Do not re-offer slots or restart qualification.",
            )
        if state.visit_requested and not state.has_date:
            return _case(
                "OB-ACK-VISIT",
                intent="SITE_VISIT",
                action="CLARIFY",
                stage="propose_slot",
                allowed_stages=["propose_slot", "visit_pick_slot"],
                acceptable_intents=["SITE_VISIT", "CLARIFY", "GENERAL"],
                acceptable_actions=["CLARIFY", "BOOK_VISIT"],
                criteria="Acknowledgement while a visit date is still missing. Ask for day/date. Do not confirm.",
            )
        if state.visit_requested and state.has_date and not state.ready_to_confirm:
            return _case(
                "OB-ACK-OFFER-SLOTS",
                intent="SITE_VISIT",
                action="BOOK_VISIT",
                stage="propose_slot",
                allowed_stages=["propose_slot", "visit_pick_slot"],
                acceptable_intents=["SITE_VISIT", "CONFIRMATION"],
                acceptable_actions=["BOOK_VISIT", "CLARIFY"],
                facts_any=["10:00 AM", "11:30 AM", "4:00 PM", "6:00 PM"],
                criteria="Day/date is known. Continue offering configured slots. Do not confirm until a time is selected.",
            )
        if "amenit" in _norm_key(_prior_customer_text(history)):
            return _case(
                "OB-ACK-AFTER-FAQ",
                intent="GENERAL",
                action="QUALIFY",
                stage="qualify",
                allowed_stages=["qualify", "propose_slot", "answer_faq"],
                acceptable_intents=["GENERAL", "QUALIFICATION", "SITE_VISIT", "CLARIFY"],
                acceptable_actions=["QUALIFY", "CLARIFY", "BOOK_VISIT"],
                criteria="Acknowledgement after an amenities answer. Do not repeat the FAQ. Continue the call or invite a visit.",
            )
        if turn_index <= 2 or not state.visit_requested:
            return _case(
                "OB-ACK-QUALIFY",
                intent="QUALIFICATION",
                action="QUALIFY",
                stage="qualify",
                allowed_stages=["qualify", "greeting"],
                acceptable_intents=["QUALIFICATION", "GENERAL", "CLARIFY"],
                acceptable_actions=["QUALIFY", "CLARIFY", "ANSWER"],
                criteria="Opening acceptance or qualification continuation.",
            )
        return None
    if "amenit" in text:
        return _case(
            "OB-AMENITIES",
            intent="AMENITIES_QUERY",
            action="ANSWER",
            stage="answer_faq",
            acceptable_intents=["AMENITIES_QUERY", "PROJECT_FAQ"],
            facts_any=[
                "clubhouses",
                "swimming pools",
                "sports courts",
                "landscaped gardens",
                "kids play areas",
                "forest grove",
                "camping grounds",
            ],
            criteria="Answer amenities from retrieved Townpark knowledge.",
        )
    if re.search(r"\b(what|when is|possession date)\b", text) and re.search(r"\b(possession|timeline)\b", text):
        return _case(
            "OB-POSSESSION-UNKNOWN",
            intent="UNKNOWN",
            action="DECLINE_UNKNOWN",
            stage="answer_faq",
            allowed_stages=["answer_faq", "qualify"],
            acceptable_intents=["UNKNOWN", "PROJECT_FAQ"],
            acceptable_actions=["DECLINE_UNKNOWN", "ANSWER"],
            criteria="Do not invent a possession date. Say the detail is not in context and offer a sales follow-up.",
        )
    if re.search(r"\b(timeline|possession|not yet decided|not decided)\b", text) and not state.visit_requested:
        return _case(
            "OB-TIMELINE-QUALIFY",
            intent="QUALIFICATION",
            action="QUALIFY",
            stage="qualify",
            allowed_stages=["qualify", "answer_faq"],
            acceptable_intents=["QUALIFICATION", "GENERAL", "CLARIFY"],
            acceptable_actions=["QUALIFY", "CLARIFY", "ANSWER"],
            criteria="Customer buying timeline is undecided. Stay in qualification. Do not invent a possession date or jump to visit confirmation.",
        )
    if re.search(r"\b([1-4] bhk|budget|crore)\b", text) and not state.visit_requested:
        return _case(
            "OB-CONFIG",
            intent="QUALIFICATION",
            action="QUALIFY",
            stage="qualify",
            allowed_stages=["qualify", "answer_faq"],
            acceptable_intents=["QUALIFICATION", "CONFIG_QUERY", "PRICE_QUERY"],
            acceptable_actions=["QUALIFY", "ANSWER"],
            expected_facts=["3 BHK"] if "3 bhk" in text else [],
            facts_any=["INR 1.8 Crore", "1.8 crore", "1.8"],
            criteria="Acknowledge configuration/budget from the customer. Starting price in context is INR 1.8 Crore onwards.",
        )
    if state.expected_rule == "ASK_DATE":
        return _case(
            "OB-VISIT-ASK-DATE",
            intent="SITE_VISIT",
            action="CLARIFY",
            stage="propose_slot",
            allowed_stages=["propose_slot", "visit_day", "visit_pick_slot"],
            acceptable_intents=["SITE_VISIT", "CLARIFY"],
            acceptable_actions=["CLARIFY", "BOOK_VISIT"],
            criteria="Ask for day/date before offering time slots. Do not confirm.",
        )
    if state.expected_rule == "OFFER_SLOTS":
        return _case(
            "OB-VISIT-OFFER-SLOTS",
            intent="SITE_VISIT",
            action="BOOK_VISIT",
            stage="propose_slot",
            allowed_stages=["propose_slot", "visit_pick_slot"],
            acceptable_intents=["SITE_VISIT", "CONFIRMATION"],
            acceptable_actions=["BOOK_VISIT", "CLARIFY"],
            facts_any=["10:00 AM", "11:30 AM", "4:00 PM", "6:00 PM"],
            criteria="Day/date is known. Offer the configured time slots. Do not confirm yet.",
        )
    if state.expected_rule == "CONFIRM":
        return _case(
            "OB-VISIT-CONFIRM",
            intent="SITE_VISIT",
            action="CONFIRM_SLOT",
            stage="visit_pick_slot",
            allowed_stages=["visit_pick_slot", "propose_slot"],
            acceptable_intents=["SITE_VISIT", "CONFIRMATION"],
            acceptable_actions=["CONFIRM_SLOT", "BOOK_VISIT"],
            criteria="Both day/date and time are known in DATE-then-TIME order. Confirm the visit.",
        )
    return None


def customer_history_from_histories(
    histories: dict[str, list[dict[str, str]]] | None,
) -> list[dict[str, str]]:
    if not histories:
        return []
    sample = next(iter(histories.values()))
    return [{"role": item.get("role"), "content": item.get("content") or ""} for item in sample]

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

_ACK = {"yes", "yeah", "yep", "ok", "okay", "sure", "alright", "all right"}


def _norm_key(utterance: str) -> str:
    text = normalize_text(utterance or "")
    text = text.replace("3bhk", "3 bhk").replace("2bhk", "2 bhk").replace("1bhk", "1 bhk").replace("4bhk", "4 bhk")
    text = re.sub(r"\b(\d{1,2})(am|pm)\b", r"\1 \2", text)
    return text


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


def lookup_calibration_case(utterance: str, turn_index: int) -> dict[str, Any] | None:
    key = (int(turn_index), _norm_key(utterance))
    found = CALIBRATION_CASES.get(key)
    return dict(found) if found else None


def infer_outbound_case(
    utterance: str,
    turn_index: int,
    history: list[dict[str, str]] | list[str] | None = None,
) -> dict[str, Any] | None:
    """Scenario GT first, then visit-state heuristics. Never invent labels for unrelated talk."""
    calibrated = lookup_calibration_case(utterance, turn_index)
    if calibrated:
        return calibrated
    text = _norm_key(utterance)
    state = visit_booking_state(history, utterance)
    if text in _ACK:
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
            criteria="Acknowledge configuration/budget from the customer.",
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
            intent="CONFIRMATION",
            action="CONFIRM_SLOT",
            stage="visit_pick_slot",
            allowed_stages=["visit_pick_slot", "propose_slot"],
            acceptable_intents=["CONFIRMATION", "SITE_VISIT"],
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

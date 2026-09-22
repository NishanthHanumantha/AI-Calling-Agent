"""Shared site-visit scheduling policy for interactive outbound evaluation.

Used by all four interactive models via a single prompt appendix. Not a new
stage taxonomy and not a per-model instruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from evaluator.deterministic import normalize_text

SITE_VISIT_SCHEDULING_RULE = """
SITE_VISIT_SCHEDULING_RULE:
1. When the customer first agrees to a site visit: ASK FOR the preferred day/date.
2. Do not offer time slots before the day/date is known.
3. Once day/date is known: OFFER only the configured slots (10:00 AM, 11:30 AM, 4:00 PM, 6:00 PM).
4. Once a time is selected after the day/date: CONFIRM the complete visit (day/date + time).
5. If the customer provides a time before a day/date: retain the time preference and ASK FOR the missing day/date. Do not confirm.
6. Never claim a visit is fully confirmed until both day/date and time are known.
7. Never invent availability outside the configured slot list.

Action mapping (existing vocabulary):
- CLARIFY: ASK_DATE / missing visit field.
- BOOK_VISIT: OFFER_SLOTS after day/date is known.
- CONFIRM_SLOT: confirm only when both day/date and time are known.
""".strip()

BOOKABLE_SLOTS = ("10:00 AM", "11:30 AM", "4:00 PM", "6:00 PM")

_DAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|"
    r"tomorrow|today|tonight)\b"
)
_VISIT_AGREE_RE = re.compile(r"\b(visit|site visit|appointment|book a visit|like to visit)\b")
_ASK_DATE_RE = re.compile(
    r"\b(which day|what day|which date|what date|preferred day|prefer(?:red)? (?:day|date)|"
    r"day or date|what day would|which day would|when would you (?:like|prefer)|"
    r"when works|what(?:'s| is) a good day)\b"
)
_ASK_TIME_RE = re.compile(r"\b(which time|what time|which slot|what slot|preferred time)\b")
_CONFIRM_RE = re.compile(
    r"\b(confirmed|you're booked|you are booked|all set|appointment is (?:set|confirmed|booked)|"
    r"your visit is (?:confirmed|booked)|visit is confirmed|see you on)\b"
)
_SLOT_RE = re.compile(
    r"\b(10(?::00)?\s*am|11:30\s*am|11:30|4(?::00)?\s*pm|6(?::00)?\s*pm|10:00|4:00|6:00)\b"
)
_TIME_UTTERANCE_RE = re.compile(
    r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)|10:00|11:30|4:00|6:00|morning|evening)\b"
)


def _customer_utterances(history: list[dict[str, str]] | list[str] | None, utterance: str) -> list[str]:
    lines: list[str] = []
    for item in history or []:
        if isinstance(item, str):
            text = item
        else:
            if item.get("role") and item.get("role") != "user":
                continue
            text = item.get("content") or item.get("utterance") or ""
        if text:
            lines.append(text)
    if utterance:
        lines.append(utterance)
    return lines


def asks_for_day_or_date(answer: str | None) -> bool:
    return bool(_ASK_DATE_RE.search(normalize_text(answer or "")))


def offers_time_slots(answer: str | None) -> bool:
    """True when the reply presents a slot menu, not a single retained time."""
    text = normalize_text(answer or "")
    slots = _SLOT_RE.findall(text)
    if len(slots) >= 2:
        return True
    if slots and _ASK_TIME_RE.search(text):
        return True
    return bool(re.search(r"\b(available (?:times|slots)|time slots|we have (?:slots|timings))\b", text))


def claims_visit_confirmed(answer: str | None) -> bool:
    return bool(_CONFIRM_RE.search(normalize_text(answer or "")))


def mentions_unlisted_availability(answer: str | None) -> bool:
    text = normalize_text(answer or "")
    if not text:
        return False
    listed = {
        "10 am",
        "10:00 am",
        "11:30 am",
        "4 pm",
        "4:00 pm",
        "6 pm",
        "6:00 pm",
    }
    invented = re.findall(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", text)
    return any(normalize_text(item) not in listed for item in invented)


@dataclass
class VisitBookingState:
    visit_requested: bool
    has_date: bool
    has_time_preference: bool
    time_before_date: bool
    date_before_time_satisfied: bool
    ready_to_confirm: bool
    expected_rule: str | None


def visit_booking_state(
    history: list[dict[str, str]] | list[str] | None,
    utterance: str,
) -> VisitBookingState:
    """Derive visit booking progress from customer utterances in order."""
    visit_requested = False
    has_date = False
    has_time_preference = False
    saw_time_before_date = False
    valid_time = False
    for text in _customer_utterances(history, utterance):
        norm = normalize_text(text)
        if _VISIT_AGREE_RE.search(norm):
            visit_requested = True
        day = bool(_DAY_RE.search(norm))
        time = bool(_TIME_UTTERANCE_RE.search(norm))
        if visit_requested and time and not has_date:
            has_time_preference = True
            saw_time_before_date = True
        if visit_requested and day:
            has_date = True
        if visit_requested and has_date and time and not saw_time_before_date:
            valid_time = True
        if visit_requested and has_date and time and saw_time_before_date:
            # Out-of-order time is only a preference until date is known; do not auto-confirm.
            has_time_preference = True
    date_before_time = has_date and (valid_time or not has_time_preference)
    ready = bool(visit_requested and has_date and valid_time)
    rule = None
    if visit_requested and not has_date:
        rule = "ASK_DATE"
    elif visit_requested and has_date and not valid_time:
        rule = "OFFER_SLOTS"
    elif ready:
        rule = "CONFIRM"
    return VisitBookingState(
        visit_requested=visit_requested,
        has_date=has_date,
        has_time_preference=has_time_preference,
        time_before_date=saw_time_before_date and not valid_time,
        date_before_time_satisfied=date_before_time,
        ready_to_confirm=ready,
        expected_rule=rule,
    )


def evaluate_visit_sequence(state: VisitBookingState, answer: str | None) -> dict[str, Any]:
    """Score DATE-before-TIME behaviour. None when the rule does not apply."""
    asked_date = asks_for_day_or_date(answer)
    offered = offers_time_slots(answer)
    confirmed = claims_visit_confirmed(answer)
    if not state.expected_rule:
        return {
            "visit_rule": None,
            "visit_sequence_pass": None,
            "asked_for_date": asked_date,
            "offered_time_slots": offered,
            "premature_confirmation": confirmed,
        }
    if state.expected_rule == "ASK_DATE":
        passed = asked_date and not offered and not confirmed
    elif state.expected_rule == "OFFER_SLOTS":
        passed = offered and not confirmed
    else:
        passed = confirmed and state.ready_to_confirm
    premature = confirmed and not state.ready_to_confirm
    return {
        "visit_rule": state.expected_rule,
        "visit_sequence_pass": passed,
        "asked_for_date": asked_date,
        "offered_time_slots": offered,
        "premature_confirmation": premature,
    }

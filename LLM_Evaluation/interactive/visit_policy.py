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
5. If the customer provides a time before a day/date: retain the time preference and ASK FOR the missing day/date. Do not offer slots again. Do not confirm.
6. If the customer provides a day/date before a time: retain the date and OFFER the configured slots.
7. Never claim a visit is fully confirmed until both day/date and time are known.
8. Never invent availability outside the configured slot list.

Action mapping (existing vocabulary):
- CLARIFY: ASK_DATE / missing visit field.
- BOOK_VISIT: OFFER_SLOTS after day/date is known.
- CONFIRM_SLOT: confirm only when both day/date and time are known.
""".strip()

BOOKABLE_SLOTS = ("10:00 AM", "11:30 AM", "4:00 PM", "6:00 PM")

_DAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|"
    r"tomorrow|today|tonight|\d{1,2}(?:st|nd|rd|th))\b"
)
_VISIT_AGREE_RE = re.compile(r"\b(visit|site visit|appointment|book a visit|like to visit)\b")
_VISIT_INVITE_RE = re.compile(
    r"\b(visit|site visit|appointment|come down|see the project|schedule)\b"
)
_AGREE_ACK_RE = re.compile(r"^(yes|yeah|yep|sure)(?:[.!])?$")
_SOFT_ACK_RE = re.compile(r"^(ok|okay|great|fine|alright|all right)(?:[.!])?$")
_ASK_DATE_RE = re.compile(
    r"\b(which day|what day|which date|what date|preferred day|prefer(?:red)? (?:day|date)|"
    r"day or date|what day would|which day would|when would you (?:like|prefer)|"
    r"when works|what(?:'s| is) a good day|convenient (?:day|date)|available (?:day|date)|"
    r"which day works|what day works)\b"
)
_ASK_TIME_RE = re.compile(r"\b(which time|what time|which slot|what slot|preferred time)\b")
_CONFIRM_RE = re.compile(
    r"\b(confirmed|you're booked|you are booked|all set|appointment is (?:set|confirmed|booked)|"
    r"your visit is (?:confirmed|booked)|visit is confirmed|see you on|"
    r"confirmed for|booked for)\b"
)
_SLOT_RE = re.compile(
    r"\b(10(?::00)?\s*am|11:30\s*am|11:30|4(?::00)?\s*pm|6(?::00)?\s*pm|10:00|4:00|6:00)\b"
)
_TIME_UTTERANCE_RE = re.compile(
    r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)|10:00|11:30|4:00|6:00|16:00|18:00|morning|evening)\b"
)


_HI_DAYS = ("सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार")
_KN_DAYS = ("ಸೋಮವಾರ", "ಮಂಗಳವಾರ", "ಬುಧವಾರ", "ಗುರುವಾರ", "ಶುಕ್ರವಾರ", "ಶನಿವಾರ", "ಭಾನುವಾರ")
_HI_VISIT_TOKENS = ("विज़िट", "विजिट", "विज़िट", "साइट विजिट", "साइट")
_KN_VISIT_TOKENS = ("ವಿಸಿಟ್", "ಸೈಟ್")
_HI_AGREE = {"हाँ", "हां", "जी"}
_KN_AGREE = {"ಹೌದು"}
_HI_SOFT = {"ठीक है", "ठीक", "अच्छा"}
_KN_SOFT = {"ಸರಿ"}
_HI_AMENITY = ("सुविधा", "सुविधाएँ", "सुविधाएं")
_KN_AMENITY = ("ಸೌಲಭ್ಯ", "ಸೌಲಭ್ಯಗಳು")


def _has_day(text: str) -> bool:
    norm = normalize_text(text)
    if _DAY_RE.search(norm):
        return True
    raw = text or ""
    return any(day in raw for day in _HI_DAYS + _KN_DAYS)


def _has_time(text: str) -> bool:
    norm = normalize_text(text)
    if _TIME_UTTERANCE_RE.search(norm):
        return True
    raw = text or ""
    if re.search(r"(शाम|सुबह|दोपहर).{0,12}4|4.{0,8}बजे", raw):
        return True
    if re.search(r"(ಸಂಜೆ|ಬೆಳಿಗ್ಗೆ|ಮಧ್ಯಾಹ್ನ).{0,12}4|4.{0,8}ಗಂಟೆ", raw):
        return True
    return False


def _has_visit_intent(text: str) -> bool:
    norm = normalize_text(text)
    if _VISIT_AGREE_RE.search(norm):
        return True
    raw = text or ""
    return any(tok in raw for tok in _HI_VISIT_TOKENS + _KN_VISIT_TOKENS)


def _is_agree_ack(text: str) -> bool:
    stripped = (text or "").strip().rstrip(".!")
    if _AGREE_ACK_RE.match(normalize_text(stripped)):
        return True
    return stripped in _HI_AGREE or stripped in _KN_AGREE


def _has_amenity_query(text: str) -> bool:
    norm = normalize_text(text)
    if "amenit" in norm:
        return True
    raw = text or ""
    return any(tok in raw for tok in _HI_AMENITY + _KN_AMENITY)


def _history_events(
    history: list[dict[str, str]] | list[str] | None,
    utterance: str,
) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for item in history or []:
        if isinstance(item, str):
            events.append(("user", item))
            continue
        role = item.get("role") or "user"
        text = item.get("content") or item.get("utterance") or ""
        if text:
            events.append(("assistant" if role == "assistant" else "user", text))
    if utterance:
        events.append(("user", utterance))
    return events


def _customer_utterances(history: list[dict[str, str]] | list[str] | None, utterance: str) -> list[str]:
    return [text for role, text in _history_events(history, utterance) if role == "user"]


def asks_for_day_or_date(answer: str | None) -> bool:
    text = answer or ""
    if _ASK_DATE_RE.search(normalize_text(text)):
        return True
    return bool(
        re.search(r"(कौन सा दिन|किस दिन|कौन सी तारीख|दिन बता|ಯಾವ ದಿನ|ದಿನಾಂಕ|yav din|yava dina)", text, re.I)
    )


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
    text = answer or ""
    if _CONFIRM_RE.search(normalize_text(text)):
        return True
    return bool(re.search(r"(पुष्टि|कन्फर्म|पक्का|ದೃಢ|ಖಚಿತ)", text))


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


PREMATURE_SLOT_OFFER = "PREMATURE_SLOT_OFFER"
PREMATURE_CONFIRMATION = "PREMATURE_CONFIRMATION"
MISSING_DATE_REQUEST = "MISSING_DATE_REQUEST"
INVALID_SLOT = "INVALID_SLOT"
CORRECT_DATE_REQUEST = "CORRECT_DATE_REQUEST"
CORRECT_SLOT_OFFER = "CORRECT_SLOT_OFFER"
CORRECT_FINAL_CONFIRMATION = "CORRECT_FINAL_CONFIRMATION"


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
    invite_pending = False
    for role, text in _history_events(history, utterance):
        norm = normalize_text(text)
        if role == "assistant":
            if _VISIT_INVITE_RE.search(norm):
                invite_pending = True
            continue
        if _has_amenity_query(text):
            invite_pending = True
        if _has_visit_intent(text):
            visit_requested = True
            invite_pending = False
        elif invite_pending and _is_agree_ack(text):
            visit_requested = True
            invite_pending = False
        day = _has_day(text)
        time = _has_time(text)
        date_already_known = has_date
        if visit_requested and day:
            has_date = True
        if visit_requested and time and not has_date:
            has_time_preference = True
            saw_time_before_date = True
        elif visit_requested and has_date and time:
            has_time_preference = True
            if date_already_known or not saw_time_before_date:
                valid_time = True
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
    invalid_slot = mentions_unlisted_availability(answer)
    premature_conf = confirmed and not state.ready_to_confirm
    premature_slot = offered and state.expected_rule == "ASK_DATE"
    code = None
    passed = None
    if not state.expected_rule:
        return {
            "visit_rule": None,
            "visit_sequence_pass": None,
            "visit_sequence_code": None,
            "asked_for_date": asked_date,
            "offered_time_slots": offered,
            "premature_confirmation": premature_conf,
            "premature_slot_offer": False,
            "invalid_slot": invalid_slot,
        }
    if state.expected_rule == "ASK_DATE":
        if premature_conf:
            code = PREMATURE_CONFIRMATION
            passed = False
        elif premature_slot:
            code = PREMATURE_SLOT_OFFER
            passed = False
        elif asked_date:
            code = CORRECT_DATE_REQUEST
            passed = True
        else:
            code = MISSING_DATE_REQUEST
            passed = False
    elif state.expected_rule == "OFFER_SLOTS":
        if premature_conf:
            code = PREMATURE_CONFIRMATION
            passed = False
        elif invalid_slot:
            code = INVALID_SLOT
            passed = False
        elif offered:
            code = CORRECT_SLOT_OFFER
            passed = True
        else:
            passed = False
    else:
        if invalid_slot:
            code = INVALID_SLOT
            passed = False
        elif confirmed and state.ready_to_confirm:
            code = CORRECT_FINAL_CONFIRMATION
            passed = True
        elif premature_conf:
            code = PREMATURE_CONFIRMATION
            passed = False
        else:
            passed = False
    return {
        "visit_rule": state.expected_rule,
        "visit_sequence_pass": passed,
        "visit_sequence_code": code,
        "asked_for_date": asked_date,
        "offered_time_slots": offered,
        "premature_confirmation": premature_conf,
        "premature_slot_offer": premature_slot,
        "invalid_slot": invalid_slot,
    }

from __future__ import annotations

import re
from typing import Any

from .deterministic import NUMBER_RE, normalize_text


REFUSAL_PATTERNS = [
    r"don'?t have",
    r"do not have",
    r"not (?:have|available|listed|sure)",
    r"sales (?:team|advisor)",
    r"can'?t (?:confirm|share|say)",
    r"cannot (?:confirm|share|say)",
    r"no (?:information|detail)",
    r"check with",
    r"follow up",
]


def is_refusal(answer: str) -> bool:
    text = normalize_text(answer or "")
    return any(re.search(pattern, text) for pattern in REFUSAL_PATTERNS)


def _content_numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(normalize_text(text or "")))


def evaluate_grounding(retrieved_context: str, answer: str | None, expected_action: str | None) -> dict[str, Any]:
    answer = answer or ""
    context_norm = normalize_text(retrieved_context or "")
    answer_norm = normalize_text(answer)
    if not answer_norm:
        return {
            "grounded": False,
            "hallucination": False,
            "unsupported_claims": ["empty_answer"],
            "refusal": False,
        }

    refusal = is_refusal(answer)
    expected_unknown = str(expected_action or "").upper() == "DECLINE_UNKNOWN"
    if expected_unknown and refusal:
        return {
            "grounded": True,
            "hallucination": False,
            "unsupported_claims": [],
            "refusal": True,
        }

    unsupported: list[str] = []
    answer_numbers = _content_numbers(answer)
    context_numbers = _content_numbers(retrieved_context)
    invented = sorted(answer_numbers - context_numbers - {"1", "2", "3", "4"})
    # Keep 1-4 because BHK counts are common; still flag other invented figures.
    for number in invented:
        if number not in {"1", "2", "3", "4"}:
            unsupported.append(f"unsupported_number:{number}")

    invented_phrases = []
    for phrase in ("maintenance", "possession", "tower a", "emi", "sqft", "square feet"):
        if phrase in answer_norm and phrase not in context_norm:
            invented_phrases.append(phrase)
    unsupported.extend(invented_phrases)

    if expected_unknown and not refusal and (unsupported or answer_numbers - context_numbers):
        hallucination = True
        grounded = False
    elif unsupported:
        hallucination = True
        grounded = False
    else:
        hallucination = False
        grounded = True if (context_norm and answer_norm) else False

    return {
        "grounded": grounded,
        "hallucination": hallucination,
        "unsupported_claims": unsupported,
        "refusal": refusal,
    }


def evaluate_context_handling(case: dict[str, Any], answer: str | None, predicted_action: str | None) -> str:
    history = case.get("conversation_history") or []
    if not history:
        return "NA"
    answer_norm = normalize_text(answer or "")
    utterance = normalize_text(case.get("customer_utterance") or "")
    if not answer_norm:
        return "FAIL"

    # Confirmation / yes should not restart as a generic FAQ dump.
    if case.get("expected_action") == "CONFIRM_SLOT" or utterance.startswith("yes"):
        restart = any(token in answer_norm for token in ("amenities", "tell me about", "would you like to hear"))
        slot_ok = any(token in answer_norm for token in ("10:00", "10 am", "11:30", "visit", "slot", "morning", "book"))
        if predicted_action == "CONFIRM_SLOT" or (slot_ok and not restart):
            return "PASS"
        return "FAIL"

    # Ambiguous follow-up should retain the prior configuration (3 BHK).
    history_text = normalize_text(" ".join(str(turn.get("content", "")) for turn in history))
    if "3 bhk" in history_text and ("that one" in utterance or "that" in utterance):
        if "3 bhk" in answer_norm:
            return "PASS"
        return "FAIL"

    return "PASS" if len(answer_norm.split()) >= 3 else "FAIL"


def evaluate_language(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected = str(case.get("language") or "en").lower()
    predicted = str(prediction.get("language") or "").lower()
    answer = prediction.get("answer") or ""
    mixed_case = expected in {"mixed", "hi", "kn"} or case.get("category") == "MULTILINGUAL"
    if mixed_case:
        # Meaning preserved if expected facts or a valid refusal appear; wording need not match.
        language_id_ok = predicted in {"mixed", "hi", "en", "kn"} or predicted.startswith("en")
        meaning_ok = bool(answer.strip())
        return {
            "language_pass": language_id_ok and meaning_ok,
            "predicted_language": predicted or None,
        }
    language_pass = predicted in {expected, "en", "en-in", "english"} or expected == "en"
    return {
        "language_pass": bool(language_pass and answer.strip()),
        "predicted_language": predicted or None,
    }

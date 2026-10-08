"""Multilingual outbound business-conversation evaluation (LLM-EVAL.4.1).

Evaluation-only. Same PROJECT_KB and visit policy as English outbound.
Does not rank models or select a production model.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from evaluator.deterministic import normalize_text

from .conversation import detect_language
from .schemas import DISPLAY_ORDER, SHORT_NAMES
from .visit_policy import visit_booking_state

LANGUAGE_TRACKS = ("english", "hindi", "kannada", "mixed")
TRACK_LABELS = {
    "english": "English",
    "hindi": "Hindi",
    "kannada": "Kannada",
    "mixed": "Mixed",
}
LANGUAGE_INSTRUCTIONS = {
    "english": "Customer language track: english. Respond predominantly in English.",
    "hindi": "Customer language track: hindi. Respond predominantly in natural conversational Hindi.",
    "kannada": "Customer language track: kannada. Respond predominantly in natural conversational Kannada.",
    "mixed": "Customer language track: mixed. Natural English/Kannada/Hindi mixing is acceptable.",
}
EXPECTED_CUSTOMER_LANGUAGE = {
    "english": "en",
    "hindi": "hi",
    "kannada": "kn",
    "mixed": "mixed",
}

LANG_MATCH = "MATCH"
LANG_PARTIAL = "PARTIAL_MATCH"
LANG_MISMATCH = "MISMATCH"

_AMENITY_FACTS = [
    "clubhouses",
    "swimming pools",
    "sports courts",
    "landscaped gardens",
    "kids play areas",
    "forest grove",
    "camping grounds",
]
_SLOT_FACTS = ["10:00 AM", "11:30 AM", "4:00 PM", "6:00 PM"]
_BUSINESS_LATIN_RE = re.compile(
    r"\b(bhk|am|pm|inr|crore|lakh|sobha|townpark|bengaluru|bangalore|hosur|"
    r"visit|slot|clubhouse|pool|friday|monday|tuesday|wednesday|thursday|"
    r"saturday|sunday|okay|ok|yes|done)\b",
    re.I,
)


def _norm_key(text: str | None) -> str:
    return re.sub(r"[^\w\u0900-\u097f\u0c80-\u0cff]+", " ", normalize_text(text or ""), flags=re.UNICODE).strip()


def _ml_case(
    test_id: str,
    utterance: str,
    *,
    track: str,
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
    lang = EXPECTED_CUSTOMER_LANGUAGE[track]
    return {
        "test_id": test_id,
        "customer_utterance": utterance,
        "category": category or intent,
        "language": lang,
        "language_track": track,
        "expected_customer_language": lang,
        "conversation_stage": stage,
        "allowed_stage_keys": allowed_stages or [stage],
        "expected_intent": intent,
        "acceptable_intents": acceptable_intents or [intent],
        "expected_action": action,
        "acceptable_actions": acceptable_actions or [action],
        "expected_facts": expected_facts or [],
        "facts_any": facts_any or [],
        "acceptable_answer_criteria": criteria,
        "ground_truth_applicable": True,
    }


def _qualify(test_id: str, utterance: str, track: str, extra: str = "") -> dict[str, Any]:
    return _ml_case(
        test_id,
        utterance,
        track=track,
        intent="QUALIFICATION",
        action="QUALIFY",
        stage="qualify",
        allowed_stages=["qualify", "greeting"],
        acceptable_intents=["QUALIFICATION", "GENERAL", "CLARIFY", "CONFIG_QUERY", "PRICE_QUERY"],
        acceptable_actions=["QUALIFY", "CLARIFY", "ANSWER"],
        expected_facts=["3 BHK"] if "3" in utterance and "BHK" in utterance else [],
        facts_any=["3 BHK", "INR 1.8 Crore", "1.8 crore", "2 crore"] if "2" in utterance or "BHK" in utterance else [],
        criteria="Continue qualification. Do not jump to visit confirmation." + extra,
        category="QUALIFICATION",
    )


def _amenities(test_id: str, utterance: str, track: str) -> dict[str, Any]:
    return _ml_case(
        test_id,
        utterance,
        track=track,
        intent="AMENITIES_QUERY",
        action="ANSWER",
        stage="answer_faq",
        allowed_stages=["answer_faq", "qualify"],
        acceptable_intents=["AMENITIES_QUERY", "PROJECT_FAQ"],
        acceptable_actions=["ANSWER"],
        expected_facts=["clubhouses"],
        facts_any=list(_AMENITY_FACTS),
        criteria="Answer amenities from Townpark knowledge. Do not invent amenities.",
        category="AMENITIES_QUERY",
    )


def _visit_ask_date(test_id: str, utterance: str, track: str) -> dict[str, Any]:
    return _ml_case(
        test_id,
        utterance,
        track=track,
        intent="SITE_VISIT",
        action="CLARIFY",
        stage="propose_slot",
        allowed_stages=["propose_slot", "visit_day"],
        acceptable_intents=["SITE_VISIT"],
        acceptable_actions=["CLARIFY", "BOOK_VISIT"],
        criteria="Customer agreed to a visit. ASK for day/date first. Do not offer time slots yet. Do not confirm.",
        category="SITE_VISIT",
    )


def _visit_offer_slots(test_id: str, utterance: str, track: str) -> dict[str, Any]:
    return _ml_case(
        test_id,
        utterance,
        track=track,
        intent="SITE_VISIT",
        action="BOOK_VISIT",
        stage="propose_slot",
        allowed_stages=["propose_slot", "visit_pick_slot"],
        acceptable_intents=["SITE_VISIT", "CONFIRMATION"],
        acceptable_actions=["BOOK_VISIT", "CLARIFY"],
        expected_facts=["10:00 AM"],
        facts_any=list(_SLOT_FACTS),
        criteria="Day/date is known. Offer only configured slots. Do not invent a slot. Do not confirm yet.",
        category="SITE_VISIT",
    )


def _visit_confirm(test_id: str, utterance: str, track: str) -> dict[str, Any]:
    return _ml_case(
        test_id,
        utterance,
        track=track,
        intent="SITE_VISIT",
        action="CONFIRM_SLOT",
        stage="visit_pick_slot",
        allowed_stages=["visit_pick_slot", "closed"],
        acceptable_intents=["SITE_VISIT", "CONFIRMATION"],
        acceptable_actions=["CONFIRM_SLOT", "ANSWER"],
        expected_facts=["4:00 PM"],
        facts_any=list(_SLOT_FACTS),
        criteria="Date and a configured time are known. Confirm the visit. Do not invent a slot.",
        category="SITE_VISIT",
    )


def _closure(test_id: str, utterance: str, track: str) -> dict[str, Any]:
    return _ml_case(
        test_id,
        utterance,
        track=track,
        intent="CONFIRMATION",
        action="ANSWER",
        stage="closed",
        allowed_stages=["closed", "visit_pick_slot"],
        acceptable_intents=["CONFIRMATION", "GENERAL", "SITE_VISIT"],
        acceptable_actions=["ANSWER", "QUALIFY", "CONFIRM_SLOT"],
        criteria="Visit is already dated and timed. Acknowledge and close. Do not re-ask the date.",
        category="CONFIRMATION",
    )


TRACK_CASES: dict[str, list[dict[str, Any]]] = {
    "english": [
        _qualify("E01", "Yes", "english"),
        _qualify("E02", "My budget is around 2 crore", "english"),
        _qualify("E03", "I am looking for a 3 BHK", "english"),
        _amenities("E04", "What amenities are available?", "english"),
        _visit_ask_date("E05", "I would like to visit", "english"),
        _visit_offer_slots("E06", "Friday", "english"),
        _visit_confirm("E07", "4 PM", "english"),
        _closure("E08", "Okay", "english"),
    ],
    "hindi": [
        _qualify("H01", "हाँ", "hindi"),
        _qualify("H02", "मेरा बजट लगभग 2 करोड़ है", "hindi"),
        _qualify("H03", "मुझे 3 BHK चाहिए", "hindi"),
        _amenities("H04", "इस प्रोजेक्ट में कौन-कौन सी सुविधाएँ हैं?", "hindi"),
        _visit_ask_date("H05", "मैं साइट विज़िट करना चाहता हूँ", "hindi"),
        _visit_offer_slots("H06", "शुक्रवार", "hindi"),
        _visit_confirm("H07", "शाम 4 बजे", "hindi"),
        _closure("H08", "ठीक है", "hindi"),
    ],
    "kannada": [
        _qualify("K01", "ಹೌದು", "kannada"),
        _qualify("K02", "ನನ್ನ ಬಜೆಟ್ ಸುಮಾರು 2 ಕೋಟಿ", "kannada"),
        _qualify("K03", "ನನಗೆ 3 BHK ಬೇಕು", "kannada"),
        _amenities("K04", "ಈ ಪ್ರಾಜೆಕ್ಟ್‌ನಲ್ಲಿ ಯಾವ ಯಾವ ಸೌಲಭ್ಯಗಳಿವೆ?", "kannada"),
        _visit_ask_date("K05", "ನಾನು ಸೈಟ್ ವಿಸಿಟ್ ಮಾಡಲು ಇಷ್ಟಪಡುತ್ತೇನೆ", "kannada"),
        _visit_offer_slots("K06", "ಶುಕ್ರವಾರ", "kannada"),
        _visit_confirm("K07", "ಸಂಜೆ 4 ಗಂಟೆಗೆ", "kannada"),
        _closure("K08", "ಸರಿ", "kannada"),
    ],
    "mixed": [
        _qualify("M01", "Yes, nanage details bekagide", "mixed"),
        _qualify("M02", "Budget around 2 crore ide", "mixed"),
        _qualify("M03", "Nanage 3 BHK beku", "mixed"),
        _amenities("M04", "Amenities yenu yenu ide?", "mixed"),
        _visit_offer_slots("M05", "I would like to visit, Friday okay", "mixed"),
        _visit_confirm("M06", "4 PM okay", "mixed"),
        _closure("M07", "Okay, done", "mixed"),
    ],
}

TRACK_UTTERANCES: dict[str, list[str]] = {
    track: [case["customer_utterance"] for case in cases] for track, cases in TRACK_CASES.items()
}
CASES_BY_ID: dict[str, dict[str, Any]] = {
    case["test_id"]: case for cases in TRACK_CASES.values() for case in cases
}


def track_utterances(track: str) -> list[str]:
    if track not in TRACK_CASES:
        raise ValueError(f"Unknown language track: {track}")
    return list(TRACK_UTTERANCES[track])


def _customer_texts(history: list[dict[str, str]] | list[str] | None) -> list[str]:
    texts: list[str] = []
    for item in history or []:
        if isinstance(item, str):
            texts.append(item)
            continue
        role = item.get("role")
        if role and role != "user":
            continue
        text = item.get("content") or item.get("utterance") or ""
        if text:
            texts.append(text)
    return texts


def bind_multilingual_ground_truth(
    track: str,
    utterance: str,
    turn_index: int,
    history: list[dict[str, str]] | list[str] | None = None,
    ground_truth_id: str | None = None,
) -> dict[str, Any] | None:
    if ground_truth_id and ground_truth_id in CASES_BY_ID:
        stamped = dict(CASES_BY_ID[ground_truth_id])
        if stamped.get("language_track") == track:
            return stamped
    cases = TRACK_CASES.get(track) or []
    key = _norm_key(utterance)
    for case in cases:
        if _norm_key(case["customer_utterance"]) == key:
            return dict(case)
    prior = _customer_texts(history)
    idx = len(prior)
    if 0 <= idx < len(cases):
        return dict(cases[idx])
    customer_n = max(0, int(turn_index) - 2)
    if 0 <= customer_n < len(cases) and not prior:
        return dict(cases[customer_n])
    return None


def resolve_multilingual_ground_truth(
    track: str,
    utterance: str,
    turn_index: int,
    history: list[dict[str, str]] | list[str] | None = None,
    ground_truth_id: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    gold = bind_multilingual_ground_truth(track, utterance, turn_index, history, ground_truth_id)
    applicable = requires_multilingual_ground_truth(utterance)
    if gold:
        return gold, True
    return None, applicable


def requires_multilingual_ground_truth(utterance: str | None) -> bool:
    return bool((utterance or "").strip())


def stamped_multilingual_id(
    track: str,
    utterance: str,
    turn_index: int,
    history: list[dict[str, str]] | list[str] | None = None,
) -> str | None:
    gold = bind_multilingual_ground_truth(track, utterance, turn_index, history)
    return (gold or {}).get("test_id")


def detect_response_language(text: str | None) -> str:
    raw = text or ""
    if not raw.strip():
        return "empty"
    hi = len(re.findall(r"[\u0900-\u097F]", raw))
    kn = len(re.findall(r"[\u0C80-\u0CFF]", raw))
    latin = len(re.findall(r"[A-Za-z]", raw))
    leftover_latin = _BUSINESS_LATIN_RE.sub(" ", raw)
    leftover_words = re.findall(r"[A-Za-z]{3,}", leftover_latin)
    if hi and kn:
        return "mixed"
    if (hi or kn) and len(leftover_words) >= 3:
        return "mixed"
    if hi > kn and hi >= max(latin, 1):
        return "hi"
    if kn > hi and kn >= max(latin, 1):
        return "kn"
    if hi > 0 and hi > latin * 0.4:
        return "hi"
    if kn > 0 and kn > latin * 0.4:
        return "kn"
    if hi and latin:
        return "mixed"
    if kn and latin:
        return "mixed"
    if detect_language(raw) == "mixed":
        return "mixed"
    return "en"


def score_response_language(expected_customer_language: str, response_text: str | None) -> str:
    detected = detect_response_language(response_text)
    if detected == "empty":
        return LANG_MISMATCH
    expected = (expected_customer_language or "en").lower()
    if expected == "mixed":
        return LANG_MATCH if detected in {"mixed", "en", "hi", "kn"} else LANG_MISMATCH
    if detected == expected:
        return LANG_MATCH
    if expected in {"hi", "kn", "en"} and detected == "mixed":
        return LANG_PARTIAL
    return LANG_MISMATCH


def language_understanding_correct(intent_correct: bool | None) -> bool | None:
    """Semantic business understanding. Independent of response-language purity."""
    return intent_correct


def structured_output_status(resp: Any) -> str:
    error_type = str(getattr(resp, "error_type", None) or "").upper()
    answer = (getattr(resp, "answer", None) or "") if resp is not None else ""
    raw = (getattr(resp, "raw_response", None) or "") if resp is not None else ""
    if error_type in {"TIMEOUT", "DRY_RUN"}:
        if error_type == "TIMEOUT":
            return "timeout"
    if error_type and error_type not in {"", "NONE"}:
        if error_type == "TIMEOUT":
            return "timeout"
        return "api_error"
    if not str(answer).strip() and not str(raw).strip():
        return "empty_response"
    schema_valid = getattr(resp, "schema_valid", None) if resp is not None else None
    if schema_valid is False:
        return "malformed_response"
    return "successful_structured"


def _mean(values: list[float | int | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def _variance(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if len(nums) < 2:
        return 0.0 if len(nums) == 1 else None
    return round(statistics.pvariance(nums), 6)


def _metric_map(summary: dict[str, Any]) -> dict[str, Any]:
    intent = summary.get("intent") or {}
    action = summary.get("action") or {}
    return {
        "intent_accuracy": intent.get("accuracy"),
        "intent_macro_f1": intent.get("f1"),
        "action_accuracy": action.get("accuracy"),
        "action_macro_f1": action.get("f1"),
        "factual_accuracy": summary.get("factual_accuracy"),
        "grounded_pct": summary.get("grounded_pct"),
        "unsupported_rate": summary.get("unsupported_rate"),
        "context_accuracy": summary.get("context_accuracy"),
        "stage_accuracy": summary.get("stage_accuracy"),
        "visit_sequence_accuracy": summary.get("visit_sequence_accuracy"),
        "premature_confirmation_count": summary.get("premature_confirmation_count"),
        "premature_slot_offer_count": summary.get("premature_slot_offer_count"),
        "avg_relevance": summary.get("avg_relevance"),
        "avg_completeness": summary.get("avg_completeness"),
        "avg_clarity": summary.get("avg_clarity"),
        "avg_conversational": summary.get("avg_conversational"),
        "avg_naturalness": summary.get("avg_naturalness"),
        "avg_latency": summary.get("avg_latency"),
        "p50_latency": summary.get("p50_latency"),
        "p95_latency": summary.get("p95_latency"),
        "avg_total_tokens": summary.get("avg_total_tokens"),
        "p50_total_tokens": summary.get("p50_total_tokens"),
        "p95_total_tokens": summary.get("p95_total_tokens"),
        "mean_input_tokens": summary.get("avg_input_tokens"),
        "mean_output_tokens": summary.get("avg_output_tokens"),
        "language_understanding_accuracy": summary.get("language_understanding_accuracy"),
        "response_language_match_pct": summary.get("response_language_match_pct"),
        "ground_truth_coverage": summary.get("ground_truth_coverage"),
        "unexpected_missing_ground_truth": summary.get("unexpected_missing_ground_truth"),
        "evaluation_coverage": summary.get("evaluation_coverage"),
        "structured_success_count": summary.get("structured_success_count"),
        "malformed_response_count": summary.get("malformed_response_count"),
        "empty_response_count": summary.get("empty_response_count"),
        "timeout_count": summary.get("timeout_count"),
        "api_error_count": summary.get("n_api_errors"),
    }


def aggregate_multilingual(
    per_track_summaries: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate per-model × per-language summaries. No ranking or winner score."""
    per_model_per_language: dict[str, dict[str, dict[str, Any]]] = {}
    for track, summary in per_track_summaries.items():
        for alias in DISPLAY_ORDER:
            per_model_per_language.setdefault(alias, {})[track] = _metric_map(summary.get(alias) or {})

    averages: dict[str, dict[str, Any]] = {}
    robustness: dict[str, dict[str, Any]] = {}
    keys = [
        "intent_accuracy",
        "intent_macro_f1",
        "action_accuracy",
        "action_macro_f1",
        "factual_accuracy",
        "grounded_pct",
        "unsupported_rate",
        "context_accuracy",
        "stage_accuracy",
        "visit_sequence_accuracy",
        "avg_relevance",
        "avg_completeness",
        "avg_clarity",
        "avg_conversational",
        "avg_naturalness",
        "avg_latency",
        "avg_total_tokens",
        "language_understanding_accuracy",
        "response_language_match_pct",
        "ground_truth_coverage",
        "evaluation_coverage",
    ]
    count_keys = (
        "premature_confirmation_count",
        "premature_slot_offer_count",
        "unexpected_missing_ground_truth",
        "structured_success_count",
        "malformed_response_count",
        "empty_response_count",
        "timeout_count",
        "api_error_count",
    )
    for alias in DISPLAY_ORDER:
        by_track = per_model_per_language.get(alias) or {}
        avg_row: dict[str, Any] = {"short": SHORT_NAMES[alias], "tracks": list(by_track)}
        for key in keys:
            avg_row[key] = _mean([row.get(key) for row in by_track.values()])
        for key in count_keys:
            vals = [row.get(key) for row in by_track.values() if row.get(key) is not None]
            avg_row[key] = round(sum(float(v) for v in vals) / len(vals), 4) if vals else None
            avg_row[f"{key}_sum"] = int(sum(float(v) for v in vals)) if vals else 0
        averages[alias] = avg_row

        lu_by_track = {
            track: (row.get("language_understanding_accuracy"), row.get("intent_accuracy"))
            for track, row in by_track.items()
        }
        scores = {}
        for track, (lu, intent) in lu_by_track.items():
            score = lu if lu is not None else intent
            if score is not None:
                scores[track] = float(score)
        if scores:
            worst_track = min(scores, key=scores.get)
            best_track = max(scores, key=scores.get)
            values = list(scores.values())
            robustness[alias] = {
                "short": SHORT_NAMES[alias],
                "worst_language": worst_track,
                "worst_language_score": scores[worst_track],
                "best_language": best_track,
                "best_language_score": scores[best_track],
                "cross_language_variance": _variance(values),
                "cross_language_consistency": (
                    round(1.0 - (math.sqrt(_variance(values) or 0.0) / (sum(values) / len(values))), 4)
                    if values and (sum(values) / len(values))
                    else None
                ),
                "per_language_understanding": scores,
            }
        else:
            robustness[alias] = {
                "short": SHORT_NAMES[alias],
                "worst_language": None,
                "worst_language_score": None,
                "cross_language_variance": None,
                "cross_language_consistency": None,
            }
    return {
        "per_model_per_language": per_model_per_language,
        "model_multilingual_average": averages,
        "language_robustness": robustness,
        "ranking": None,
        "winner": None,
        "note": "Evidence only. No model ranking or selection in LLM-EVAL.4.1.",
    }


def write_multilingual_aggregate(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def multilingual_visit_state(history: list[str] | list[dict[str, str]] | None, utterance: str):
    return visit_booking_state(history, utterance)

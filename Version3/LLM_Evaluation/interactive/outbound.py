"""Outbound conversation simulation for the interactive lab.

Customer-first LLM-EVAL.2 is unchanged. This module only adds the
AI-initiated path, opening checks, and outbound evaluation notes.
"""

from __future__ import annotations

import re
from typing import Any

from evaluator.deterministic import normalize_text
from evaluator.grounding import evaluate_grounding

from .conversation import STAGE_LABELS, infer_stage
from .schemas import DISPLAY_ORDER, TurnResponse

FIXED_OPENING = (
    "Hi, this is Sobha Limited calling. I'm getting in touch about "
    "SOBHA Townpark — a New York-inspired luxury community near "
    "Electronic City on Hosur Road in Bengaluru. Is now a good time "
    "for a quick call?"
)

OPENING_INSTRUCTION = (
    "The customer has not spoken. You are placing the outbound call. "
    "Speak first as a Sobha Limited sales agent for SOBHA Townpark. "
    "In one or two short spoken sentences, identify the company, say why you are calling, "
    "name the project, give a brief description and the location using only the retrieved context, "
    "and ask whether now is a good time. "
    "Sound like an outbound caller. Do not say 'How can I help you?'. Do not invent facts."
)

# Standard manual scenario. Not executed unless --demo is requested.
OUTBOUND_BASELINE = [
    "Yes, I have a couple of minutes.",
    "I'm looking for a 3 BHK.",
    "Where exactly is the project located?",
    "How much does it cost?",
    "Can I visit this Saturday?",
    "Morning would be better.",
    "Okay, I'll discuss it with my wife and get back to you.",
]

OUTBOUND_NAMES = {
    "sarvam_conversational": "SARVAM — CONVERSATIONAL",
    "deepseek_flagship": "DEEPSEEK — V4.1 FLASH",
    "claude_sonnet": "CLAUDE — SONNET",
    "claude_flagship": "CLAUDE — OPUS",
}

_CHATBOT_OPENING = re.compile(r"how can i help you", re.I)


def _pass_fail(ok: bool | None) -> str | None:
    if ok is None:
        return None
    return "PASS" if ok else "FAIL"


def opening_elements(text: str | None) -> dict[str, str | None]:
    """Checklist for an outbound opening. Wording does not need to match the reference."""
    if not text or not str(text).strip():
        return {
            "opening_element_company": "FAIL",
            "opening_element_project": "FAIL",
            "opening_element_description": "FAIL",
            "opening_element_location": "FAIL",
            "opening_element_permission": "FAIL",
        }
    norm = normalize_text(text)
    company = "sobha" in norm
    project = "townpark" in norm
    description = "new york" in norm or "luxury" in norm
    location = any(token in norm for token in ("electronic city", "hosur", "bengaluru", "bangalore"))
    permission = bool(
        re.search(
            r"\b(good time|right time|quick call|have a (minute|moment|couple)|is now|free to talk|spare a)\b",
            norm,
        )
    )
    return {
        "opening_element_company": _pass_fail(company),
        "opening_element_project": _pass_fail(project),
        "opening_element_description": _pass_fail(description),
        "opening_element_location": _pass_fail(location),
        "opening_element_permission": _pass_fail(permission),
    }


def _clip(value: int) -> int:
    return max(1, min(5, value))


def opening_quality(text: str | None) -> dict[str, int | None]:
    if not text or not str(text).strip():
        return {"opening_conciseness": 1, "outbound_appropriateness": 1, "naturalness": 1}
    words = [w for w in str(text).split() if w]
    count = len(words)
    if 20 <= count <= 55:
        concise = 5
    elif 12 <= count <= 80:
        concise = 4
    elif count < 12:
        concise = 2
    else:
        concise = 2
    norm = normalize_text(text)
    elements = opening_elements(text)
    present = sum(1 for key, value in elements.items() if value == "PASS")
    if _CHATBOT_OPENING.search(text or ""):
        appropriate = 1
    elif present >= 4 and elements["opening_element_permission"] == "PASS":
        appropriate = 5
    elif elements["opening_element_permission"] == "PASS":
        appropriate = 4
    else:
        appropriate = 2
    natural = 4
    if re.search(r"as an ai|retrieved context|```|\{", text or "", re.I):
        natural = 2
    if _CHATBOT_OPENING.search(text or ""):
        natural = 2
    if 12 <= count <= 70 and "." in (text or "") and natural > 2:
        natural = 5
    return {
        "opening_conciseness": _clip(concise),
        "outbound_appropriateness": _clip(appropriate),
        "naturalness": _clip(natural),
    }


def factual_claims(answer: str | None, retrieved_context: str) -> dict[str, Any]:
    """Grounding wrapper. An empty answer is not treated as a hallucination."""
    grounding = evaluate_grounding(retrieved_context, answer, None)
    claims = [item for item in (grounding.get("unsupported_claims") or []) if item != "empty_answer"]
    has_text = bool(normalize_text(answer or ""))
    hallucination = bool(claims) if has_text else False
    grounded = bool(grounding.get("grounded")) if has_text and not claims else (False if has_text else None)
    if has_text and not claims:
        grounded = True
    return {
        "grounded": grounded if has_text else None,
        "hallucination": hallucination if has_text else False,
        "unsupported_claims": claims,
        "supported_by_context": has_text and not claims,
    }


def evaluate_opening(
    responses: list[TurnResponse],
    retrieved_context: str,
    opening_mode: str,
    turn_index: int = 1,
    required_aliases: list[str] | None = None,
) -> dict[str, Any]:
    by_alias = {resp.alias: resp for resp in responses}
    aliases = list(required_aliases) if required_aliases is not None else [resp.alias for resp in responses]
    models = []
    for alias in aliases:
        resp = by_alias.get(alias)
        if resp is None:
            models.append(
                {
                    "alias": alias,
                    "provider": alias.split("_")[0],
                    "model_id": "",
                    "evaluation_status": "MISSING_RESPONSE",
                    "customer_message": "",
                    "model_response": None,
                    "api_error": False,
                    "error_type": "MISSING_RESPONSE",
                    "error_message": None,
                    "hallucination": None,
                    "unsupported_claims": None,
                    "intent_correct": None,
                    "action_correct": None,
                    "stage_correct": None,
                    "latency_ms": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "note": "Response missing — not scored as model quality failure",
                    "ground_truth_applicable": False,
                }
            )
            continue
        if resp.error_type:
            models.append(
                {
                    "alias": resp.alias,
                    "provider": resp.provider,
                    "model_id": resp.model_id,
                    "evaluation_status": "OK",
                    "customer_message": "",
                    "model_response": resp.answer or resp.raw_response,
                    "api_error": True,
                    "error_type": resp.error_type,
                    "error_message": resp.error_message,
                    "hallucination": None,
                    "unsupported_claims": None,
                    "intent_correct": None,
                    "action_correct": None,
                    "opening_element_company": None,
                    "opening_element_project": None,
                    "opening_element_description": None,
                    "opening_element_location": None,
                    "opening_element_permission": None,
                    "opening_conciseness": None,
                    "outbound_appropriateness": None,
                    "naturalness": None,
                    "note": "API/model error — not scored as quality failure",
                }
            )
            continue
        text = resp.answer or resp.raw_response or ""
        elements = opening_elements(text)
        quality = opening_quality(text)
        facts = factual_claims(text, retrieved_context)
        models.append(
            {
                "alias": resp.alias,
                "provider": resp.provider,
                "model_id": resp.model_id,
                "evaluation_status": "OK",
                "customer_message": "",
                "model_response": text,
                "api_error": False,
                "ground_truth": "NOT AVAILABLE",
                "ground_truth_applicable": False,
                "predicted_intent": resp.intent,
                "intent_correct": None,
                "predicted_action": resp.action,
                "action_correct": None,
                "expected_stage": "Greeting",
                "model_stage": "Greeting",
                "allowed_stages": ["Greeting"],
                "stage_correct": True if text.strip() else None,
                "grounded": facts["grounded"],
                "hallucination": facts["hallucination"],
                "unsupported_claims": facts["unsupported_claims"],
                "factual_accuracy": 1.0 if facts["supported_by_context"] else (0.0 if text.strip() else None),
                "context_used": None,
                "context_error": False,
                "latency_ms": resp.latency_ms,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "total_tokens": resp.total_tokens,
                **elements,
                **quality,
            }
        )
    return {
        "kind": "opening",
        "turn_index": turn_index,
        "turn_id": turn_index,
        "utterance": "",
        "opening_mode": opening_mode,
        "ground_truth": "NOT AVAILABLE",
        "ground_truth_applicable": False,
        "detected_language": "en",
        "models": models,
    }


def outbound_expected_stage(
    utterance: str,
    turn_index: int,
    customer_history: list[dict[str, str]] | list[str] | None = None,
    ground_truth_id: str | None = None,
) -> tuple[str, list[str]]:
    """Primary stage key plus allowed keys. Uses existing stage names."""
    from .outbound_ground_truth import infer_outbound_case

    labelled = infer_outbound_case(utterance, turn_index, customer_history, ground_truth_id)
    if labelled:
        stage = labelled["conversation_stage"]
        return stage, labelled.get("allowed_stage_keys") or [stage]
    text = normalize_text(utterance)
    if re.search(r"\bnot interested\b", text):
        return "closed", ["closed"]
    if re.search(r"\bbusy\b", text):
        return "greeting", ["greeting", "closed"]
    if re.search(r"\b(wife|get back|call you later|think about it)\b", text):
        return "closed", ["closed"]
    if re.search(r"\b(morning|evening|slot|10|11|4|6)\b", text) and re.search(
        r"\b(better|prefer|am|pm|morning|evening|slot)\b", text
    ):
        return "visit_pick_slot", ["visit_pick_slot", "propose_slot"]
    if re.search(r"\b(visit|saturday|sunday|weekend|appointment)\b", text):
        return "propose_slot", ["propose_slot"]
    if re.search(r"\b(next step|what(?: is|'s)? next|what happens next)\b", text):
        return "closed", ["closed", "visit_pick_slot"]
    if re.search(r"\b(3 bhk|2 bhk|1 bhk|4 bhk|looking for|configuration)\b", text):
        return "qualify", ["qualify", "answer_faq"]
    if re.search(r"\b(where|located|location|price|cost|how much|appreciation|another project)\b", text):
        allowed = ["answer_faq"]
        if "how much" in text or text.strip() in {"how much", "how much?"}:
            allowed = ["answer_faq", "qualify"]
        return "answer_faq", allowed
    if re.search(r"\b(yes|yeah|sure|couple of minutes|good time)\b", text):
        return "qualify", ["greeting", "qualify"]
    fallback = infer_stage(utterance, turn_index)
    return fallback, [fallback]


def shared_stage(
    utterance: str,
    turn_index: int,
    conversation_mode: str,
    customer_history: list[dict[str, str]] | list[str] | None = None,
) -> str:
    if conversation_mode == "outbound":
        stage, _allowed = outbound_expected_stage(utterance, turn_index, customer_history)
        return stage
    return infer_stage(utterance, turn_index)


def stage_labels(keys: list[str]) -> list[str]:
    return [STAGE_LABELS.get(key, key) for key in keys]


def model_stage_key(action: str | None, answer: str | None, fallback: str) -> str:
    from .visit_policy import asks_for_day_or_date, offers_time_slots

    text = normalize_text(answer or "")
    if any(token in text for token in ("call you back", "call back", "another time", "goodbye", "no problem")):
        return "closed"
    if asks_for_day_or_date(answer):
        return "propose_slot"
    if offers_time_slots(answer):
        return "propose_slot"
    if any(token in text for token in ("10:00", "11:30", "4:00", "6:00", "morning")) and "visit" in text:
        return "visit_pick_slot"
    if "visit" in text or "saturday" in text:
        return "propose_slot"
    mapped = {
        "QUALIFY": "qualify",
        "BOOK_VISIT": "propose_slot",
        "CONFIRM_SLOT": "visit_pick_slot",
        "ANSWER": "answer_faq",
        "DECLINE_UNKNOWN": "answer_faq",
        "CLARIFY": "qualify",
    }
    return mapped.get((action or "").upper(), fallback)


def response_naturalness(answer: str | None) -> int | None:
    if not answer or not str(answer).strip():
        return None
    words = normalize_text(answer).split()
    score = 4
    if re.search(r"as an ai|retrieved context|```|how can i help you", answer, re.I):
        score = 2
    if 4 <= len(words) <= 60 and "." in answer:
        score = 5
    if len(words) > 90:
        score = 2
    return _clip(score)


def customer_experience(utterance: str, answer: str | None, previous_answer: str | None = None) -> dict[str, bool | None]:
    text = answer or ""
    norm = normalize_text(text)
    utt = normalize_text(utterance)
    if not norm:
        return {
            "appropriate_follow_up": None,
            "unnecessary_repetition": None,
            "excessive_information": None,
            "unnatural_response": None,
            "customer_intent_addressed": None,
        }
    words = norm.split()
    utt_tokens = [tok for tok in utt.split() if len(tok) > 3]
    overlap = sum(1 for tok in utt_tokens if tok in norm)
    previous = normalize_text(previous_answer or "")
    repeated = False
    if previous and norm:
        prev_tokens = set(previous.split())
        now_tokens = set(words)
        if prev_tokens and len(prev_tokens & now_tokens) / len(prev_tokens | now_tokens) >= 0.72:
            repeated = True
    return {
        "appropriate_follow_up": "?" in text,
        "unnecessary_repetition": repeated,
        "excessive_information": len(words) > 80,
        "unnatural_response": bool(re.search(r"how can i help you|as an ai|retrieved context", text, re.I)),
        "customer_intent_addressed": bool(overlap or (utt_tokens and len(words) >= 4)),
    }


def exception_notes(utterance: str, answer: str | None) -> dict[str, Any] | None:
    """Evaluation dimensions for manual branches. These are not scripted replies."""
    text = normalize_text(utterance)
    ans = normalize_text(answer or "")
    if "busy" in text:
        return {
            "branch": "busy",
            "permission_handling": any(token in ans for token in ("busy", "time", "later", "moment")),
            "callback_handling": any(token in ans for token in ("call back", "callback", "later", "another time")),
            "continued_sales_pitch": any(token in ans for token in ("crore", "amenities", "clubhouse"))
            and "later" not in ans
            and "call back" not in ans,
        }
    if "not interested" in text:
        return {
            "branch": "not_interested",
            "acknowledged": any(token in ans for token in ("understand", "no problem", "alright", "thank")),
            "polite_closure": any(token in ans for token in ("thank", "goodbye", "no problem", "have a")),
            "aggressive_persuasion": any(
                token in ans for token in ("you should", "don't miss", "do not miss", "last chance", "why not")
            ),
        }
    if text.strip() in {"how much", "how much?"} or text.strip() == "how much":
        return {
            "branch": "ambiguous",
            "uses_context_or_clarifies": ("?" in (answer or ""))
            or any(token in ans for token in ("3 bhk", "2 bhk", "configuration", "which")),
        }
    if "appreciation" in text or "next five years" in text:
        return {
            "branch": "unknown_information",
            "note": "Do not invent appreciation, guarantees, or future returns.",
        }
    if "another project" in text or "lower price" in text:
        return {
            "branch": "competitor",
            "note": "Score factual grounding and unsupported claims. No scripted rebuttal.",
        }
    return None


def context_resolves_requirement(history_text: str, utterance: str, answer: str | None) -> tuple[bool | None, bool]:
    """Whether a follow-up such as 'that' or 'it' keeps the earlier configuration."""
    history = normalize_text(history_text)
    utt = normalize_text(utterance)
    ans = normalize_text(answer or "")
    configs = [token for token in ("3 bhk", "2 bhk", "1 bhk", "4 bhk") if token in history]
    if not configs:
        return None, False
    refers = bool(re.search(r"\b(that|it|this|one)\b", utt)) and bool(
        re.search(r"\b(cost|price|how much|visit|where)\b", utt)
    )
    if not refers:
        return None, False
    kept = any(config in ans for config in configs)
    asked_again = "which configuration" in ans or "which one" in ans
    return kept and not asked_again, asked_again or not kept


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]] | None:
    labelled = [(gold, pred) for gold, pred in pairs if gold and pred]
    if not labelled:
        return None
    labels = sorted({gold for gold, _ in labelled} | {pred for _, pred in labelled})
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for gold, pred in labelled:
        matrix[gold][pred] = matrix[gold].get(pred, 0) + 1
    return matrix


def fixed_opening_responses(slots: list[Any]) -> list[TurnResponse]:
    ordered = []
    by_alias = {slot.alias: slot for slot in slots}
    for alias in DISPLAY_ORDER:
        slot = by_alias.get(alias)
        ordered.append(
            TurnResponse(
                alias=alias,
                provider=slot.provider if slot else alias.split("_")[0],
                model_id=slot.model_id if slot else "",
                answer=FIXED_OPENING,
                intent=None,
                action=None,
                language="en",
                stage="Greeting",
                raw_response=FIXED_OPENING,
                latency_ms=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                error_type=None,
                error_message=None,
                schema_valid=None,
            )
        )
    return ordered

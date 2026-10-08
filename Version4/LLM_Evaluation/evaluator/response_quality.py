from __future__ import annotations

import json
import re
from typing import Any, Callable

from .deterministic import evaluate_facts, normalize_text
from .grounding import is_refusal


def _clip_score(value: float) -> int:
    return max(1, min(5, int(round(value))))


def score_response_quality(case: dict[str, Any], prediction: dict[str, Any], grounding: dict[str, Any]) -> dict[str, Any]:
    """Deterministic 1-5 rubric. Does not receive provider/model names."""
    answer = prediction.get("answer") or ""
    utterance = case.get("customer_utterance") or ""
    facts = evaluate_facts(case.get("expected_facts") or [], answer)
    words = normalize_text(answer).split()
    utt_tokens = [tok for tok in re.findall(r"[a-z0-9]+", normalize_text(utterance)) if len(tok) > 2]
    overlap = sum(1 for tok in utt_tokens if tok in normalize_text(answer)) if utt_tokens else 0
    overlap_ratio = overlap / max(1, len(set(utt_tokens)))

    relevance = 1
    if answer.strip():
        relevance = 3 + (2 if overlap_ratio >= 0.25 else 1 if overlap_ratio > 0 else 0)
        if case.get("expected_action") == "DECLINE_UNKNOWN" and is_refusal(answer):
            relevance = 5

    grounding_score = 5 if grounding.get("grounded") and not grounding.get("hallucination") else 2
    if grounding.get("hallucination"):
        grounding_score = 1

    completeness = 1
    if case.get("expected_action") == "DECLINE_UNKNOWN":
        completeness = 5 if is_refusal(answer) else 2
    elif facts["facts_expected"]:
        completeness = _clip_score(1 + facts["fact_accuracy"] * 4)
    elif answer.strip():
        completeness = 3

    clarity = 4 if answer.strip() and not re.search(r"```|as an ai|bullet", answer, re.I) else 2
    if len(words) > 80:
        clarity = max(1, clarity - 1)
    if not answer.strip():
        clarity = 1

    conversational = 4
    stage = str(case.get("conversation_stage") or "")
    if any(token in normalize_text(answer) for token in ("brochure", "retrieved context", "as an ai")):
        conversational = 2
    if 4 <= len(words) <= 60:
        conversational = min(5, conversational + 1)
    if stage == "propose_slot" and "amenities" in normalize_text(answer) and "visit" not in normalize_text(answer):
        conversational = 2
    if not answer.strip():
        conversational = 1

    stage_score = conversational
    expected_action = str(case.get("expected_action") or "")
    predicted_action = str(prediction.get("action") or "")
    if expected_action and predicted_action and expected_action == predicted_action:
        stage_score = min(5, stage_score + 1)
    elif expected_action and predicted_action and expected_action != predicted_action:
        stage_score = max(1, stage_score - 2)

    word_count = len([w for w in (answer or "").split() if w])
    conciseness = 5 if 4 <= word_count <= 45 else 4 if word_count <= 70 else 2 if word_count else 1
    actionability = 4
    if predicted_action in {"BOOK_VISIT", "CONFIRM_SLOT", "QUALIFY", "CLARIFY"}:
        actionability = 5
    if predicted_action == "DECLINE_UNKNOWN" and is_refusal(answer):
        actionability = 4
    if not answer.strip():
        actionability = 1

    utterance_norm = normalize_text(utterance)
    short_sim = None
    if len(utterance_norm.split()) <= 4 or utterance_norm in {"yes", "no", "yeah", "ok", "okay"}:
        short_sim = "TEXT-LEVEL SHORT-UTTERANCE SIMULATION"
        if predicted_action in {"CONFIRM_SLOT", "BOOK_VISIT", "CLARIFY", "ANSWER", "DECLINE_UNKNOWN", "QUALIFY"}:
            short_sim_pass = True
        else:
            short_sim_pass = bool(answer.strip())
    else:
        short_sim_pass = None

    scores = {
        "relevance_score": _clip_score(relevance),
        "grounding_quality_score": _clip_score(grounding_score),
        "completeness_score": _clip_score(completeness),
        "clarity_score": _clip_score(clarity),
        "conversation_quality_score": _clip_score(conversational),
        "stage_appropriateness_score": _clip_score(stage_score),
        "response_word_count": word_count,
        "conciseness_score": conciseness,
        "actionability_score": actionability,
        "short_utterance_label": short_sim,
        "short_utterance_pass": short_sim_pass,
    }
    quality_keys = [
        "relevance_score",
        "clarity_score",
        "completeness_score",
        "conversation_quality_score",
        "stage_appropriateness_score",
    ]
    overall = sum(scores[k] for k in quality_keys) / len(quality_keys)
    scores["overall_response_quality"] = round(overall, 2)
    scores["semantic_eval_status"] = "NOT_RUN"
    scores["judge_notes"] = None
    return scores


def maybe_run_judge(
    case: dict[str, Any],
    prediction: dict[str, Any],
    judge_fn: Callable[..., dict[str, Any]] | None,
    judge_system_prompt: str | None,
) -> dict[str, Any]:
    """Optional LLM-as-a-judge. Must not be given provider/model names."""
    if judge_fn is None:
        return {"semantic_eval_status": "NOT_RUN"}
    payload = {
        "customer_utterance": case.get("customer_utterance"),
        "conversation_history": case.get("conversation_history"),
        "retrieved_context": case.get("retrieved_context"),
        "expected_facts": case.get("expected_facts"),
        "acceptable_answer_criteria": case.get("acceptable_answer_criteria"),
        "candidate_answer": prediction.get("answer"),
        "candidate_intent": prediction.get("intent"),
        "candidate_action": prediction.get("action"),
    }
    try:
        judged = judge_fn(
            system_prompt=judge_system_prompt or "",
            conversation_history=[],
            customer_utterance=json.dumps(payload, ensure_ascii=False),
            retrieved_context=case.get("retrieved_context") or "",
            extra={"conversation_stage": "evaluation"},
        )
        if judged.get("error"):
            return {"semantic_eval_status": "NOT_RUN", "judge_error": "judge_call_failed"}
        return {
            "semantic_eval_status": "RUN",
            "judge_raw": judged.get("raw_response"),
        }
    except Exception:
        return {"semantic_eval_status": "NOT_RUN", "judge_error": "judge_call_failed"}

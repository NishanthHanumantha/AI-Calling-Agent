from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluator.aggregation import macro_f1
from evaluator.deterministic import normalize_text
from evaluator.runner import evaluate_one

from .conversation import STAGE_LABELS, detect_language, infer_stage, match_golden_case
from .schemas import DISPLAY_ORDER, SHORT_NAMES, TurnResponse

ACTION_TO_STAGE = {
    "QUALIFY": "qualify",
    "BOOK_VISIT": "propose_slot",
    "CONFIRM_SLOT": "visit_pick_slot",
    "ANSWER": "answer_faq",
    "DECLINE_UNKNOWN": "answer_faq",
    "CLARIFY": "qualify",
}


def _classification_prf(pairs: list[tuple[str, str]]) -> dict[str, float | None]:
    if not pairs:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}
    correct = sum(1 for gold, pred in pairs if gold == pred)
    accuracy = round(correct / len(pairs), 4)
    rows = [{"expected": g, "predicted": p} for g, p in pairs]
    labels = sorted({g for g, _ in pairs})
    precs, recs = [], []
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precs.append(precision)
        recs.append(recall)
    precision = round(sum(precs) / len(precs), 4) if precs else None
    recall = round(sum(recs) / len(recs), 4) if recs else None
    f1 = macro_f1(rows, "expected", "predicted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _mark(value: bool | None) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "N/A"


def evaluate_turn(
    utterance: str,
    responses: list[TurnResponse],
    dataset: list[dict[str, Any]],
    histories: dict[str, list[dict[str, str]]],
    turn_index: int,
    retrieved_context: str,
) -> dict[str, Any]:
    gold = match_golden_case(utterance, dataset)
    expected_stage = infer_stage(utterance, turn_index)
    detected_lang = detect_language(utterance)
    per_model = []
    for resp in responses:
        if resp.error_type:
            per_model.append(
                {
                    "alias": resp.alias,
                    "api_error": True,
                    "error_type": resp.error_type,
                    "error_message": resp.error_message,
                    "ground_truth": gold.get("test_id") if gold else "NOT AVAILABLE",
                    "intent_correct": None,
                    "action_correct": None,
                    "grounded": None,
                    "hallucination": None,
                    "unsupported_claims": None,
                    "factual_accuracy": None,
                    "relevance": None,
                    "completeness": None,
                    "clarity": None,
                    "conversational": None,
                    "stage_quality": None,
                    "note": "API/model error — not scored as quality failure",
                }
            )
            continue
        unlabeled = gold is None
        case = {
            "test_id": None if unlabeled else gold.get("test_id"),
            "category": "UNLABELED" if unlabeled else gold.get("category"),
            "language": detected_lang if unlabeled else gold.get("language"),
            "conversation_stage": expected_stage if unlabeled else (gold.get("conversation_stage") or expected_stage),
            "conversation_history": histories.get(resp.alias) or [],
            "customer_utterance": utterance,
            "retrieved_context": retrieved_context,
            "expected_intent": None if unlabeled else gold.get("expected_intent"),
            "expected_facts": [] if unlabeled else (gold.get("expected_facts") or []),
            "expected_action": None if unlabeled else gold.get("expected_action"),
            "acceptable_answer_criteria": "" if unlabeled else gold.get("acceptable_answer_criteria"),
        }
        generation = {
            "provider": resp.provider,
            "model_alias": resp.alias,
            "model_id": resp.model_id,
            "model": resp.model_id,
            "intent": resp.intent,
            "action": resp.action,
            "language": resp.language,
            "answer": resp.answer,
            "raw_response": resp.raw_response,
            "schema_valid": resp.schema_valid,
            "usage": {
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "total_tokens": resp.total_tokens,
            },
            "latency_ms": resp.latency_ms,
            "error": None,
            "error_type": None,
        }
        row = evaluate_one(case, generation)
        history_text = " ".join(t.get("content", "") for t in (histories.get(resp.alias) or []))
        context_used = False
        if histories.get(resp.alias):
            prev = normalize_text(history_text)
            ans = normalize_text(resp.answer or "")
            clues = [tok for tok in ("3 bhk", "2 bhk", "visit", "price", "townpark") if tok in prev]
            context_used = any(clue in ans for clue in clues) or bool(ans and clues)
            if not clues:
                context_used = bool(ans)
        context_error = bool(histories.get(resp.alias)) and any(
            phrase in normalize_text(resp.answer or "")
            for phrase in ("which project", "what project", "which one are you referring")
        )
        model_stage_key = ACTION_TO_STAGE.get((resp.action or "").upper(), expected_stage)
        stage_correct = STAGE_LABELS.get(model_stage_key) == STAGE_LABELS.get(expected_stage)
        if unlabeled:
            intent_correct = None
            action_correct = None
            factual_accuracy = None
        else:
            intent_correct = row.get("intent_pass")
            action_correct = row.get("action_pass")
            factual_accuracy = row.get("fact_accuracy")
            stage_correct = bool(action_correct) if action_correct is not None else stage_correct
        per_model.append(
            {
                "alias": resp.alias,
                "api_error": False,
                "ground_truth": gold.get("test_id") if gold else "NOT AVAILABLE",
                "expected_intent": None if unlabeled else gold.get("expected_intent"),
                "predicted_intent": resp.intent,
                "intent_correct": intent_correct,
                "expected_action": None if unlabeled else gold.get("expected_action"),
                "predicted_action": resp.action,
                "action_correct": action_correct,
                "supported_claims": row.get("facts_found") or [],
                "unsupported_claims": row.get("unsupported_claims") or [],
                "factual_accuracy": factual_accuracy,
                "grounded": row.get("grounded"),
                "grounding_score": row.get("grounding_quality_score"),
                "hallucination": row.get("hallucination"),
                "context_used": context_used,
                "context_error": context_error,
                "context_handling": row.get("context_handling"),
                "expected_stage": STAGE_LABELS.get(expected_stage, expected_stage),
                "model_stage": STAGE_LABELS.get(model_stage_key, model_stage_key),
                "stage_correct": stage_correct,
                "detected_language": resp.language or detect_language(resp.answer or utterance),
                "expected_language": None if unlabeled else gold.get("language"),
                "language_correct": None if unlabeled else row.get("language_pass"),
                "relevance": row.get("relevance_score"),
                "completeness": row.get("completeness_score"),
                "clarity": row.get("clarity_score"),
                "conversational": row.get("conversation_quality_score"),
                "stage_quality": row.get("stage_appropriateness_score"),
                "latency_ms": resp.latency_ms,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "raw_eval": row,
            }
        )
    return {
        "utterance": utterance,
        "turn_index": turn_index,
        "ground_truth": gold.get("test_id") if gold else "NOT AVAILABLE",
        "detected_language": detected_lang,
        "models": per_model,
    }


def session_summary(evaluations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_alias: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in evaluations:
        for model in ev.get("models") or []:
            by_alias[model["alias"]].append(model)
    out: dict[str, dict[str, Any]] = {}
    for alias in DISPLAY_ORDER:
        rows = by_alias.get(alias) or []
        quality = [r for r in rows if not r.get("api_error")]
        intent_pairs = [
            (str(r["expected_intent"]).upper(), str(r["predicted_intent"] or "").upper())
            for r in quality
            if r.get("expected_intent")
        ]
        action_pairs = [
            (str(r["expected_action"]).upper(), str(r["predicted_action"] or "").upper())
            for r in quality
            if r.get("expected_action")
        ]
        facts = [r.get("factual_accuracy") for r in quality if r.get("factual_accuracy") is not None]
        grounded = [r for r in quality if r.get("grounded") is not None]
        lat = [r.get("latency_ms") for r in quality if r.get("latency_ms") is not None]
        lat_sorted = sorted(lat)

        def pct(p: float) -> float | None:
            if not lat_sorted:
                return None
            idx = int(round((len(lat_sorted) - 1) * p))
            return lat_sorted[idx]

        stage_labeled = [r for r in quality if r.get("stage_correct") is not None]
        out[alias] = {
            "short": SHORT_NAMES[alias],
            "n_turns": len(rows),
            "n_scored": len(quality),
            "n_api_errors": sum(1 for r in rows if r.get("api_error")),
            "intent": _classification_prf(intent_pairs),
            "action": _classification_prf(action_pairs),
            "factual_accuracy": round(sum(facts) / len(facts), 4) if facts else None,
            "grounded_pct": round(sum(1 for r in grounded if r.get("grounded")) / len(grounded), 4) if grounded else None,
            "unsupported_rate": (
                round(sum(1 for r in quality if r.get("unsupported_claims")) / len(quality), 4) if quality else None
            ),
            "context_accuracy": (
                round(
                    sum(1 for r in quality if r.get("context_used") and not r.get("context_error")) / len(quality),
                    4,
                )
                if quality
                else None
            ),
            "stage_accuracy": (
                round(sum(1 for r in stage_labeled if r.get("stage_correct")) / len(stage_labeled), 4)
                if stage_labeled
                else None
            ),
            "avg_relevance": _avg([r.get("relevance") for r in quality]),
            "avg_completeness": _avg([r.get("completeness") for r in quality]),
            "avg_clarity": _avg([r.get("clarity") for r in quality]),
            "avg_conversational": _avg([r.get("conversational") for r in quality]),
            "avg_latency": _avg(lat),
            "p50_latency": pct(0.5),
            "p95_latency": pct(0.95),
        }
    return out


def _avg(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value <= 1:
        return f"{round(value * 100):.0f}%"
    return str(value)


def format_num(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)

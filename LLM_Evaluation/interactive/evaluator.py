from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluator.aggregation import macro_f1
from evaluator.deterministic import normalize_text
from evaluator.runner import evaluate_one

from .conversation import STAGE_LABELS, detect_language, infer_stage, match_golden_case
from .outbound import (
    confusion_matrix,
    context_resolves_requirement,
    customer_experience,
    exception_notes,
    model_stage_key,
    outbound_expected_stage,
    response_naturalness,
    stage_labels,
)
from .multilingual import (
    EXPECTED_CUSTOMER_LANGUAGE,
    detect_response_language,
    language_understanding_correct,
    resolve_multilingual_ground_truth,
    score_response_language,
    structured_output_status,
)
from .outbound_ground_truth import customer_history_from_histories, resolve_outbound_ground_truth
from .schemas import DISPLAY_ORDER, SHORT_NAMES, TurnResponse
from .visit_policy import evaluate_visit_sequence, visit_booking_state

ACTION_TO_STAGE = {
    "QUALIFY": "qualify",
    "BOOK_VISIT": "propose_slot",
    "CONFIRM_SLOT": "visit_pick_slot",
    "ANSWER": "answer_faq",
    "DECLINE_UNKNOWN": "answer_faq",
    "CLARIFY": "qualify",
}

EVAL_STATUS_OK = "OK"
EVAL_STATUS_MISSING_RESPONSE = "MISSING_RESPONSE"
EVAL_STATUS_MISSING_GROUND_TRUTH = "MISSING_GROUND_TRUTH"
EVAL_STATUS_EVALUATION_ERROR = "EVALUATION_ERROR"
_SKIPPED_STATUSES = {EVAL_STATUS_MISSING_RESPONSE, EVAL_STATUS_EVALUATION_ERROR}


def _provider_from_alias(alias: str) -> str:
    return (alias or "").split("_")[0]


def _unscored_model_record(
    *,
    alias: str,
    status: str,
    utterance: str,
    provider: str | None = None,
    model_id: str | None = None,
    model_response: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    ground_truth: str = "NOT AVAILABLE",
    expected_intent: str | None = None,
    expected_action: str | None = None,
    expected_stage: str | None = None,
    ground_truth_applicable: bool = False,
    api_error: bool = False,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "alias": alias,
        "provider": provider or _provider_from_alias(alias),
        "model_id": model_id or "",
        "evaluation_status": status,
        "customer_message": utterance,
        "model_response": model_response,
        "api_error": api_error,
        "error_type": error_type,
        "error_message": error_message,
        "ground_truth": ground_truth,
        "expected_intent": expected_intent,
        "expected_action": expected_action,
        "expected_stage": expected_stage,
        "ground_truth_applicable": ground_truth_applicable,
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
        "stage_correct": None,
        "latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "schema_valid": None,
        "structured_output_status": status if status != EVAL_STATUS_MISSING_RESPONSE else "empty_response",
        "note": note,
    }


def flatten_evaluation_records(
    evaluations: list[dict[str, Any]],
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """One record per (turn_id, model alias). Later evaluations replace earlier duplicates."""
    by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    order: list[tuple[Any, Any]] = []
    for ev in evaluations:
        turn_id = ev.get("turn_id")
        if turn_id is None:
            turn_id = ev.get("turn_index")
        sid = ev.get("session_id") or session_id
        utterance = ev.get("utterance")
        for model in ev.get("models") or []:
            alias = model.get("alias")
            key = (turn_id, alias)
            row = {
                **model,
                "session_id": model.get("session_id") or sid,
                "turn_id": model.get("turn_id") if model.get("turn_id") is not None else turn_id,
                "model_id": model.get("model_id") or alias,
                "model_alias": alias,
                "provider": model.get("provider") or _provider_from_alias(alias or ""),
                "customer_message": (
                    model.get("customer_message") if model.get("customer_message") is not None else utterance
                ),
                "model_response": model.get("model_response"),
                "evaluation_status": model.get("evaluation_status") or EVAL_STATUS_OK,
            }
            if key not in by_key:
                order.append(key)
            by_key[key] = row
    return [by_key[key] for key in order]


def evaluation_coverage(
    records: list[dict[str, Any]],
    expected: int | None = None,
) -> dict[str, Any]:
    expected_count = len(records) if expected is None else expected
    skipped_rows = [row for row in records if row.get("evaluation_status") in _SKIPPED_STATUSES]
    evaluated_count = sum(1 for row in records if row.get("evaluation_status") not in _SKIPPED_STATUSES)
    missing_records = max(0, expected_count - len(records))
    reasons = [
        (
            f"turn_id={row.get('turn_id')} model={row.get('alias') or row.get('model_id')}: "
            f"{row.get('evaluation_status')}"
            + (f" ({row.get('error_message')})" if row.get("error_message") else "")
        )
        for row in skipped_rows
    ]
    if missing_records:
        reasons.append(f"{missing_records} turn×model record(s) were not produced")
    return {
        "expected": expected_count,
        "evaluated": evaluated_count,
        "skipped": expected_count - evaluated_count,
        "reasons": reasons,
        "record_count": len(records),
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


def _label_allowed(expected: str | None, predicted: str | None, acceptable: list[str] | None) -> bool | None:
    if not expected:
        return None
    pred = str(predicted or "").strip().upper()
    if not pred:
        return False
    allowed = {str(expected).strip().upper()}
    allowed.update(str(item).strip().upper() for item in (acceptable or []) if item)
    return pred in allowed


def _fact_score(gold: dict[str, Any], answer: str | None, row: dict[str, Any]) -> float | None:
    from evaluator.deterministic import fact_present

    any_facts = gold.get("facts_any") or []
    if any_facts:
        found = [fact for fact in any_facts if fact_present(fact, answer or "")]
        return 1.0 if found else 0.0
    if gold.get("expected_facts"):
        return row.get("fact_accuracy")
    return row.get("fact_accuracy")


def evaluate_turn(
    utterance: str,
    responses: list[TurnResponse],
    dataset: list[dict[str, Any]],
    histories: dict[str, list[dict[str, str]]],
    turn_index: int,
    retrieved_context: str,
    *,
    outbound: bool = False,
    required_aliases: list[str] | None = None,
    ground_truth_id: str | None = None,
    language_track: str | None = None,
) -> dict[str, Any]:
    customer_history = customer_history_from_histories(histories)
    gold = None
    gt_applicable = False
    if language_track:
        gold, gt_applicable = resolve_multilingual_ground_truth(
            language_track, utterance, turn_index, customer_history, ground_truth_id
        )
        expected_stage, allowed_keys = outbound_expected_stage(
            utterance, turn_index, customer_history, ground_truth_id
        )
    elif outbound:
        gold, gt_applicable = resolve_outbound_ground_truth(
            utterance, turn_index, customer_history, ground_truth_id
        )
        expected_stage, allowed_keys = outbound_expected_stage(
            utterance, turn_index, customer_history, ground_truth_id
        )
    else:
        expected_stage = infer_stage(utterance, turn_index)
        allowed_keys = [expected_stage]
    if gold is None and not language_track:
        gold = match_golden_case(utterance, dataset)
        if gold:
            gt_applicable = True
    if gold and gold.get("conversation_stage"):
        expected_stage = gold["conversation_stage"]
        allowed_keys = gold.get("allowed_stage_keys") or [expected_stage]
    detected_lang = detect_language(utterance)
    expected_customer_language = None
    if language_track:
        expected_customer_language = (gold or {}).get("expected_customer_language") or EXPECTED_CUSTOMER_LANGUAGE.get(
            language_track
        )
    unlabeled = gold is None
    expected_stage_label = STAGE_LABELS.get(expected_stage, expected_stage)
    gt_id = gold.get("test_id") if gold else "NOT AVAILABLE"
    gt_intent = None if unlabeled else gold.get("expected_intent")
    gt_action = None if unlabeled else gold.get("expected_action")
    by_alias = {resp.alias: resp for resp in responses}
    aliases = list(required_aliases) if required_aliases is not None else [resp.alias for resp in responses]
    per_model = []
    for alias in aliases:
        resp = by_alias.get(alias)
        if resp is None:
            per_model.append(
                _unscored_model_record(
                    alias=alias,
                    status=EVAL_STATUS_MISSING_RESPONSE,
                    utterance=utterance,
                    ground_truth=gt_id,
                    expected_intent=gt_intent,
                    expected_action=gt_action,
                    expected_stage=expected_stage_label,
                    ground_truth_applicable=gt_applicable,
                    note="Response missing — not scored as model quality failure",
                )
            )
            continue
        if resp.error_type:
            per_model.append(
                {
                    "alias": resp.alias,
                    "provider": resp.provider,
                    "model_id": resp.model_id,
                    "evaluation_status": EVAL_STATUS_OK,
                    "customer_message": utterance,
                    "model_response": resp.answer or resp.raw_response,
                    "api_error": True,
                    "error_type": resp.error_type,
                    "error_message": resp.error_message,
                    "ground_truth": gt_id,
                    "expected_intent": gt_intent,
                    "expected_action": gt_action,
                    "expected_stage": expected_stage_label,
                    "ground_truth_applicable": gt_applicable,
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
                    "schema_valid": resp.schema_valid,
                    "structured_output_status": structured_output_status(resp),
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "total_tokens": resp.total_tokens,
                    "latency_ms": resp.latency_ms,
                    "note": "API/model error — not scored as quality failure",
                }
            )
            continue
        case = {
            "test_id": None if unlabeled else gold.get("test_id"),
            "category": "UNLABELED" if unlabeled else gold.get("category"),
            "language": detected_lang if unlabeled else gold.get("language"),
            "conversation_stage": expected_stage if unlabeled else (gold.get("conversation_stage") or expected_stage),
            "conversation_history": histories.get(resp.alias) or [],
            "customer_utterance": utterance,
            "retrieved_context": retrieved_context,
            "expected_intent": gt_intent,
            "expected_facts": [] if unlabeled else (gold.get("expected_facts") or gold.get("facts_any") or []),
            "expected_action": gt_action,
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
        try:
            row = evaluate_one(case, generation)
        except Exception as exc:
            per_model.append(
                _unscored_model_record(
                    alias=resp.alias,
                    status=EVAL_STATUS_EVALUATION_ERROR,
                    utterance=utterance,
                    provider=resp.provider,
                    model_id=resp.model_id,
                    model_response=resp.answer or resp.raw_response,
                    error_type=EVAL_STATUS_EVALUATION_ERROR,
                    error_message=str(exc),
                    ground_truth=gt_id,
                    expected_intent=gt_intent,
                    expected_action=gt_action,
                    expected_stage=expected_stage_label,
                    ground_truth_applicable=gt_applicable,
                    note="Evaluator exception — not scored as model failure",
                )
            )
            continue
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
        model_stage = ACTION_TO_STAGE.get((resp.action or "").upper(), expected_stage)
        if outbound:
            model_stage = model_stage_key(resp.action, resp.answer, expected_stage)
        stage_correct = STAGE_LABELS.get(model_stage) == STAGE_LABELS.get(expected_stage)
        if outbound:
            stage_correct = STAGE_LABELS.get(model_stage) in stage_labels(allowed_keys)
        if unlabeled:
            intent_correct = None
            action_correct = None
            factual_accuracy = None
            eval_status = EVAL_STATUS_MISSING_GROUND_TRUTH
        else:
            intent_correct = _label_allowed(
                gold.get("expected_intent"),
                resp.intent,
                gold.get("acceptable_intents"),
            )
            action_correct = _label_allowed(
                gold.get("expected_action"),
                resp.action,
                gold.get("acceptable_actions"),
            )
            factual_accuracy = _fact_score(gold, resp.answer, row)
            eval_status = EVAL_STATUS_OK
            if not outbound:
                stage_correct = bool(action_correct) if action_correct is not None else stage_correct
        claims = row.get("unsupported_claims") or []
        if isinstance(claims, list):
            claims = [item for item in claims if item != "empty_answer"]
        hallucination = None if not (resp.answer or resp.raw_response) else row.get("hallucination")
        if not (resp.answer or resp.raw_response):
            hallucination = False
        previous = ""
        for turn in reversed(histories.get(resp.alias) or []):
            if turn.get("role") == "assistant":
                previous = turn.get("content") or ""
                break
        history_follow, history_miss = context_resolves_requirement(history_text, utterance, resp.answer)
        if history_follow is not None:
            context_used = history_follow
            context_error = history_miss
        extras: dict[str, Any] = {}
        if outbound or language_track:
            visit_state = visit_booking_state(customer_history, utterance)
            visit_eval = evaluate_visit_sequence(visit_state, resp.answer)
            extras = {
                "allowed_stages": stage_labels(allowed_keys),
                "naturalness": response_naturalness(resp.answer),
                "customer_experience": customer_experience(utterance, resp.answer, previous),
                "exception_branch": exception_notes(utterance, resp.answer),
                **visit_eval,
            }
        if language_track:
            extras.update(
                {
                    "language_track": language_track,
                    "expected_customer_language": expected_customer_language,
                    "response_language": detect_response_language(resp.answer),
                    "response_language_match": score_response_language(
                        expected_customer_language or detected_lang, resp.answer
                    ),
                    "language_understanding_correct": language_understanding_correct(intent_correct),
                }
            )
        per_model.append(
            {
                "alias": resp.alias,
                "provider": resp.provider,
                "model_id": resp.model_id,
                "evaluation_status": eval_status,
                "customer_message": utterance,
                "model_response": resp.answer or resp.raw_response,
                "api_error": False,
                "ground_truth": gt_id,
                "ground_truth_applicable": gt_applicable,
                "expected_intent": gt_intent,
                "predicted_intent": resp.intent,
                "intent_correct": intent_correct,
                "expected_action": gt_action,
                "predicted_action": resp.action,
                "action_correct": action_correct,
                "supported_claims": row.get("facts_found") or [],
                "unsupported_claims": claims,
                "factual_accuracy": factual_accuracy,
                "grounded": row.get("grounded"),
                "grounding_score": row.get("grounding_quality_score"),
                "hallucination": hallucination,
                "context_used": context_used,
                "context_error": context_error,
                "context_handling": row.get("context_handling"),
                "expected_stage": expected_stage_label,
                "model_stage": STAGE_LABELS.get(model_stage, model_stage),
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
                "total_tokens": resp.total_tokens,
                "schema_valid": resp.schema_valid,
                "structured_output_status": structured_output_status(resp),
                "raw_eval": row,
                **extras,
            }
        )
    return {
        "utterance": utterance,
        "turn_index": turn_index,
        "turn_id": turn_index,
        "ground_truth": gt_id,
        "ground_truth_applicable": gt_applicable,
        "expected_intent": gt_intent,
        "expected_action": gt_action,
        "expected_stage": expected_stage_label,
        "detected_language": detected_lang,
        "language_track": language_track,
        "expected_customer_language": expected_customer_language,
        "conversation_mode": "outbound" if outbound else "customer",
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
        quality = [
            r
            for r in rows
            if not r.get("api_error") and r.get("evaluation_status", EVAL_STATUS_OK) not in _SKIPPED_STATUSES
        ]
        intent_pairs = []
        action_pairs = []
        for r in quality:
            if r.get("expected_intent"):
                gold_i = str(r["expected_intent"]).upper()
                pred_i = gold_i if r.get("intent_correct") else str(r.get("predicted_intent") or "").upper()
                intent_pairs.append((gold_i, pred_i))
            if r.get("expected_action"):
                gold_a = str(r["expected_action"]).upper()
                pred_a = gold_a if r.get("action_correct") else str(r.get("predicted_action") or "").upper()
                action_pairs.append((gold_a, pred_a))
        facts = [r.get("factual_accuracy") for r in quality if r.get("factual_accuracy") is not None]
        grounded = [r for r in quality if r.get("grounded") is not None]
        lat = [r.get("latency_ms") for r in quality if r.get("latency_ms") is not None]
        lat_sorted = sorted(lat)

        def pct(p: float) -> float | None:
            if not lat_sorted:
                return None
            idx = int(round((len(lat_sorted) - 1) * p))
            return lat_sorted[idx]

        visit_rows = [r for r in quality if r.get("visit_sequence_pass") is not None]
        stage_labeled = [r for r in quality if r.get("stage_correct") is not None]
        tokens = [r.get("total_tokens") for r in quality if r.get("total_tokens") is not None]
        tokens_sorted = sorted(tokens)
        input_tokens = [r.get("input_tokens") for r in quality if r.get("input_tokens") is not None]
        output_tokens = [r.get("output_tokens") for r in quality if r.get("output_tokens") is not None]
        intent_stats = _classification_prf(intent_pairs)
        action_stats = _classification_prf(action_pairs)
        intent_stats["confusion"] = confusion_matrix(intent_pairs)
        action_stats["confusion"] = confusion_matrix(action_pairs)
        gt_applicable_rows = [r for r in quality if r.get("ground_truth_applicable")]
        gt_covered = [r for r in gt_applicable_rows if r.get("expected_intent")]
        unexpected_missing = [
            r for r in gt_applicable_rows if not r.get("expected_intent")
        ]
        intentionally_unlabelled = [
            r for r in quality if not r.get("ground_truth_applicable")
        ]
        visit_compliance = (
            round(sum(1 for r in visit_rows if r.get("visit_sequence_pass")) / len(visit_rows), 4)
            if visit_rows
            else None
        )
        lu_rows = [r for r in quality if r.get("language_understanding_correct") is not None]
        lang_match_rows = [r for r in quality if r.get("response_language_match")]
        statuses = [r.get("structured_output_status") for r in rows if r.get("structured_output_status")]
        out[alias] = {
            "short": SHORT_NAMES[alias],
            "n_turns": len(rows),
            "n_scored": len(quality),
            "n_api_errors": sum(1 for r in rows if r.get("api_error")),
            "evaluation_coverage": round(len(quality) / len(rows), 4) if rows else None,
            "intent": intent_stats,
            "action": action_stats,
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
            "visit_sequence_accuracy": visit_compliance,
            "visit_sequence_compliance": visit_compliance,
            "premature_confirmation_count": sum(1 for r in quality if r.get("premature_confirmation")),
            "premature_slot_offer_count": sum(1 for r in quality if r.get("premature_slot_offer")),
            "ground_truth_coverage": (
                round(len(gt_covered) / len(gt_applicable_rows), 4) if gt_applicable_rows else None
            ),
            "intentionally_unlabelled_turns": len({r.get("turn_id") for r in intentionally_unlabelled}),
            "unexpected_missing_ground_truth": len({r.get("turn_id") for r in unexpected_missing}),
            "avg_relevance": _avg([r.get("relevance") for r in quality]),
            "avg_completeness": _avg([r.get("completeness") for r in quality]),
            "avg_clarity": _avg([r.get("clarity") for r in quality]),
            "avg_conversational": _avg([r.get("conversational") for r in quality]),
            "avg_naturalness": _avg([r.get("naturalness") for r in quality]),
            "avg_latency": _avg(lat),
            "p50_latency": pct(0.5),
            "p95_latency": pct(0.95),
            "avg_total_tokens": _avg(tokens),
            "p50_total_tokens": tokens_sorted[int(round((len(tokens_sorted) - 1) * 0.5))] if tokens_sorted else None,
            "p95_total_tokens": tokens_sorted[int(round((len(tokens_sorted) - 1) * 0.95))] if tokens_sorted else None,
            "avg_input_tokens": _avg(input_tokens),
            "avg_output_tokens": _avg(output_tokens),
            "language_understanding_accuracy": (
                round(sum(1 for r in lu_rows if r.get("language_understanding_correct")) / len(lu_rows), 4)
                if lu_rows
                else None
            ),
            "response_language_match_pct": (
                round(sum(1 for r in lang_match_rows if r.get("response_language_match") == "MATCH") / len(lang_match_rows), 4)
                if lang_match_rows
                else None
            ),
            "structured_success_count": sum(1 for s in statuses if s == "successful_structured"),
            "malformed_response_count": sum(1 for s in statuses if s == "malformed_response"),
            "empty_response_count": sum(1 for s in statuses if s == "empty_response"),
            "timeout_count": sum(1 for s in statuses if s == "timeout"),
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

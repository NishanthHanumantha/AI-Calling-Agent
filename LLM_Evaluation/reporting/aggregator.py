"""Aggregate independent per-model metrics from canonical observations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluator.aggregation import _percentile, _safe_mean

from interactive.evaluator import _classification_prf

from .data_loader import (
    CLASS_FRAMEWORK,
    CLASS_MODEL,
    CLASS_MULTILINGUAL,
    EVIDENCE_CONTROLLED,
    EVIDENCE_EXPLORATORY,
    EVIDENCE_FRAMEWORK,
    MODEL_DISPLAY,
    RunBundle,
    language_label,
)
from .pricing import PRICING_ASSUMPTIONS, PRICING_NOTE, llm_cost, lookup_pricing

MODEL_EVAL_CLASSES = {CLASS_MODEL, CLASS_MULTILINGUAL}

ERROR_CHECKS = [
    ("Wrong Intent", lambda r: r.get("intent_correct") is False),
    ("Wrong Action", lambda r: r.get("action_correct") is False),
    ("Factual Error", lambda r: r.get("factual_correct") is False),
    ("Unsupported Claim", lambda r: r.get("unsupported_claim") is True),
    ("Context Error", lambda r: r.get("context_correct") is False),
    ("Stage Error", lambda r: r.get("stage_correct") is False),
    ("Language Understanding Error", lambda r: r.get("language_understanding_correct") is False),
    ("Response Language Mismatch", lambda r: str(r.get("response_language_match") or "") == "MISMATCH"),
    ("Visit Sequence Error", lambda r: r.get("visit_sequence_correct") is False),
    ("Premature Confirmation", lambda r: r.get("premature_confirmation") is True),
    ("Premature Slot Offer", lambda r: r.get("premature_slot_offer") is True),
    ("Timeout", lambda r: str(r.get("error_type") or "").upper() == "TIMEOUT" or str(r.get("evaluation_status") or "") == "TIMEOUT"),
    ("Malformed Response", lambda r: str(r.get("error_type") or "").upper() in {"INVALID_JSON"} or r.get("structured_success") is False),
    ("Empty Response", lambda r: "empty" in str(r.get("error_message") or "").lower()),
]


def is_model_eval_run(bundle: RunBundle) -> bool:
    return bundle.classification in MODEL_EVAL_CLASSES


def model_eval_observations(runs: list[RunBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for bundle in runs:
        if not is_model_eval_run(bundle):
            continue
        for row in bundle.observations:
            if row.get("is_fixture"):
                continue
            key = (row.get("run_id"), row.get("turn"), row.get("model"))
            if key in seen:
                continue
            seen.add(key)
            enriched = dict(row)
            enriched["run_classification"] = bundle.classification
            enriched["evidence_class"] = bundle.evidence_class
            rows.append(enriched)
    return rows


def _rate(values: list[bool | None]) -> tuple[float | None, int]:
    flagged = [v for v in values if v is not None]
    if not flagged:
        return None, 0
    return round(sum(1 for v in flagged if v) / len(flagged), 4), len(flagged)


def _mean(values: list[Any]) -> tuple[float | None, int]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None, 0
    return _safe_mean(nums), len(nums)


def _pct(values: list[Any], q: float) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return _percentile(nums, q)


def _intent_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs = []
    for row in rows:
        if not row.get("expected_intent"):
            continue
        if row.get("intent_correct") is None and not row.get("predicted_intent"):
            continue
        gold = str(row["expected_intent"]).upper()
        if row.get("intent_correct") is True:
            pred = gold
        else:
            pred = str(row.get("predicted_intent") or "").upper()
        pairs.append((gold, pred))
    return pairs


def _action_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs = []
    for row in rows:
        if not row.get("expected_action"):
            continue
        if row.get("action_correct") is None and not row.get("predicted_action"):
            continue
        gold = str(row["expected_action"]).upper()
        if row.get("action_correct") is True:
            pred = gold
        else:
            pred = str(row.get("predicted_action") or "").upper()
        pairs.append((gold, pred))
    return pairs


def _metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intent = _classification_prf(_intent_pairs(rows))
    action = _classification_prf(_action_pairs(rows))
    fact_mean, fact_n = _mean([r.get("factual_accuracy") for r in rows])
    grounded, grounded_n = _rate([r.get("grounded") for r in rows])
    unsupported, unsupported_n = _rate([r.get("unsupported_claim") for r in rows])
    context, context_n = _rate([r.get("context_correct") for r in rows])
    stage, stage_n = _rate([r.get("stage_correct") for r in rows])
    rel, rel_n = _mean([r.get("relevance") for r in rows])
    comp, comp_n = _mean([r.get("completeness") for r in rows])
    clar, clar_n = _mean([r.get("clarity") for r in rows])
    conv, conv_n = _mean([r.get("conversational_quality") for r in rows])
    nat, nat_n = _mean([r.get("naturalness") for r in rows])
    lat = [r.get("latency_ms") for r in rows]
    tok_in = [r.get("input_tokens") for r in rows]
    tok_out = [r.get("output_tokens") for r in rows]
    tok_tot = [r.get("total_tokens") for r in rows]
    lu, lu_n = _rate([r.get("language_understanding_correct") for r in rows])
    match_rows = [r for r in rows if r.get("response_language_match")]
    match_rate = None
    match_n = len(match_rows)
    if match_rows:
        match_rate = round(sum(1 for r in match_rows if r.get("response_language_match") == "MATCH") / match_n, 4)
    intent_n = len(_intent_pairs(rows))
    action_n = len(_action_pairs(rows))
    lat_mean, lat_n = _mean(lat)
    return {
        "n": len(rows),
        "intent_accuracy": intent["accuracy"],
        "intent_precision": intent["precision"],
        "intent_recall": intent["recall"],
        "intent_macro_f1": intent["f1"],
        "intent_n": intent_n,
        "action_accuracy": action["accuracy"],
        "action_precision": action["precision"],
        "action_recall": action["recall"],
        "action_macro_f1": action["f1"],
        "action_n": action_n,
        "factual_accuracy": fact_mean,
        "factual_n": fact_n,
        "grounded_pct": grounded,
        "grounded_n": grounded_n,
        "unsupported_claim_rate": unsupported,
        "unsupported_n": unsupported_n,
        "context_accuracy": context,
        "context_n": context_n,
        "stage_accuracy": stage,
        "stage_n": stage_n,
        "avg_relevance": rel,
        "relevance_n": rel_n,
        "avg_completeness": comp,
        "completeness_n": comp_n,
        "avg_clarity": clar,
        "clarity_n": clar_n,
        "avg_conversational_quality": conv,
        "conversational_n": conv_n,
        "avg_naturalness": nat,
        "naturalness_n": nat_n,
        "language_understanding_accuracy": lu,
        "language_understanding_n": lu_n,
        "response_language_match_pct": match_rate,
        "response_language_match_n": match_n,
        "avg_latency_ms": lat_mean,
        "latency_n": lat_n,
        "p25_latency_ms": _pct(lat, 0.25),
        "p50_latency_ms": _pct(lat, 0.50),
        "p75_latency_ms": _pct(lat, 0.75),
        "p95_latency_ms": _pct(lat, 0.95),
        "min_latency_ms": min((float(v) for v in lat if v is not None), default=None),
        "max_latency_ms": max((float(v) for v in lat if v is not None), default=None),
        "mean_input_tokens": _mean(tok_in)[0],
        "input_token_n": _mean(tok_in)[1],
        "p50_input_tokens": _pct(tok_in, 0.50),
        "p95_input_tokens": _pct(tok_in, 0.95),
        "mean_output_tokens": _mean(tok_out)[0],
        "output_token_n": _mean(tok_out)[1],
        "p50_output_tokens": _pct(tok_out, 0.50),
        "p95_output_tokens": _pct(tok_out, 0.95),
        "mean_total_tokens": _mean(tok_tot)[0],
        "total_token_n": _mean(tok_tot)[1],
        "p50_total_tokens": _pct(tok_tot, 0.50),
        "p95_total_tokens": _pct(tok_tot, 0.95),
        "sum_input_tokens": int(sum(int(v) for v in tok_in if v is not None)),
        "sum_output_tokens": int(sum(int(v) for v in tok_out if v is not None)),
        "sum_total_tokens": int(sum(int(v) for v in tok_tot if v is not None)),
    }


def _model_sort_key(alias: str) -> tuple[int, str]:
    order = list(MODEL_DISPLAY)
    if alias in order:
        return (order.index(alias), alias)
    return (len(order), alias)


def per_model_metrics(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_model[str(row.get("model") or row.get("model_id") or "unknown")].append(row)
    out = []
    for alias in sorted(by_model, key=_model_sort_key):
        rows = by_model[alias]
        sample = rows[0]
        block = _metric_block(rows)
        spec = lookup_pricing(str(sample.get("model_id") or ""), alias)
        cost_per = None
        if spec and block["mean_input_tokens"] is not None:
            cost_per = llm_cost(block["mean_input_tokens"], block["mean_output_tokens"], spec)
        out.append(
            {
                "model": alias,
                "display_model": sample.get("display_model") or MODEL_DISPLAY.get(alias, alias),
                "provider": sample.get("provider"),
                "model_id": sample.get("model_id"),
                "pricing": spec,
                "cost_per_model_turn": cost_per,
                "cost_currency": spec["currency"] if spec else None,
                **block,
            }
        )
    return out


def language_coverage(runs: list[RunBundle], observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = ["English", "Hindi", "Kannada", "Mixed"]
    eval_runs = [r for r in runs if is_model_eval_run(r)]
    by_lang_runs: dict[str, set[str]] = defaultdict(set)
    by_lang_turns: dict[str, int] = defaultdict(int)
    by_lang_obs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lang_models: dict[str, set[str]] = defaultdict(set)
    for bundle in eval_runs:
        seen_turns: set[tuple[Any, str]] = set()
        for row in bundle.observations:
            if row.get("is_fixture"):
                continue
            lang = row.get("language") or "English"
            by_lang_runs[lang].add(bundle.run_id)
            key = (row.get("turn"), lang)
            if key in seen_turns:
                continue
            seen_turns.add(key)
            if str(row.get("customer_input") or "").strip():
                by_lang_turns[lang] += 1
        if bundle.language_track:
            by_lang_runs[language_label(bundle.language_track)].add(bundle.run_id)
    for row in observations:
        lang = row.get("language") or "English"
        by_lang_obs[lang].append(row)
        by_lang_models[lang].add(str(row.get("display_model") or row.get("model")))
    rows = []
    for lang in wanted:
        n_runs = len(by_lang_runs.get(lang, set()))
        n_obs = len(by_lang_obs.get(lang, []))
        n_turns = by_lang_turns.get(lang, 0)
        if n_obs == 0:
            status = "Pending / Not yet evaluated"
        elif lang == "English" and any(
            r.evidence_class == EVIDENCE_CONTROLLED and (r.language_track in (None, "english") or r.mode == "BATCH GOLDEN DATASET")
            for r in eval_runs
        ):
            status = "Completed — controlled English benchmark present; exploratory English runs kept separate"
        elif lang == "Hindi":
            h01 = any(r.language_track == "hindi" and r.evidence_class == EVIDENCE_CONTROLLED for r in eval_runs)
            exploratory = any(r.language_track == "hindi" and r.evidence_class == EVIDENCE_EXPLORATORY for r in eval_runs)
            if h01 and exploratory:
                status = "Partial — H01–H08 sequence exists and is kept separate from later exploratory Hindi; not production-validated"
            elif h01:
                status = "Partial — H01–H08 interactive sequence exists; not production-validated"
            else:
                status = "Partial — interactive Hindi evaluation exists; not H01–H08 standardized benchmark"
        elif lang == "Kannada":
            status = (
                "Partial — K01–K08 interactive sequence exists; not production-validated"
                if n_obs
                else "Pending / Not yet evaluated"
            )
        elif lang == "Mixed":
            status = "Partial — limited mixed-language evidence (not production-validated)"
        else:
            status = "Partial"
        rows.append(
            {
                "language": lang,
                "evaluation_runs": n_runs,
                "customer_turns": n_turns,
                "model_turn_evaluations": n_obs,
                "models": ", ".join(sorted(by_lang_models.get(lang, set()))) or "unavailable",
                "status": status,
            }
        )
    return rows


def error_rows(observations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail = []
    summary_counts: dict[tuple[str, str], int] = defaultdict(int)
    model_totals: dict[str, int] = defaultdict(int)
    for row in observations:
        model_totals[str(row.get("display_model") or row.get("model"))] += 1
    for row in observations:
        model = str(row.get("display_model") or row.get("model"))
        lang = row.get("language") or "English"
        for label, predicate in ERROR_CHECKS:
            if not predicate(row):
                continue
            expected = row.get("expected_intent") or row.get("expected_action")
            predicted = row.get("predicted_intent") or row.get("predicted_action")
            if label == "Wrong Action":
                expected = row.get("expected_action")
                predicted = row.get("predicted_action")
            detail.append(
                {
                    "model": model,
                    "language": lang,
                    "error_type": label,
                    "run_id": row.get("run_id"),
                    "turn": row.get("turn"),
                    "customer_input": row.get("customer_input"),
                    "expected": expected,
                    "predicted": predicted,
                    "notes": row.get("error_message") or row.get("ground_truth") or "",
                }
            )
            summary_counts[(model, label)] += 1
    summary = []
    for (model, label), count in sorted(summary_counts.items()):
        total = model_totals.get(model) or 1
        summary.append(
            {
                "model": model,
                "error_type": label,
                "count": count,
                "percentage": round(count / total, 4),
            }
        )
    for row in detail:
        model = row["model"]
        total = model_totals.get(model) or 1
        matching = next(
            (s for s in summary if s["model"] == model and s["error_type"] == row["error_type"]),
            None,
        )
        row["count"] = matching["count"] if matching else 1
        row["percentage"] = matching["percentage"] if matching else round(1 / total, 4)
    return detail, summary


def cost_rows(model_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in model_metrics:
        spec = item.get("pricing")
        n = item.get("n") or 0
        mean_in = item.get("mean_input_tokens")
        mean_out = item.get("mean_output_tokens")
        unit = item.get("cost_per_model_turn")
        rows.append(
            {
                "model": item.get("display_model"),
                "evaluation_observations": n,
                "mean_input_tokens": mean_in,
                "mean_output_tokens": mean_out,
                "input_price_per_1m": spec["input_per_1m"] if spec else None,
                "output_price_per_1m": spec["output_per_1m"] if spec else None,
                "currency": spec["currency"] if spec else "unavailable",
                "estimated_llm_cost_per_model_turn": unit,
                "estimated_cost_per_1000": None if unit is None else unit * 1000,
                "estimated_cost_per_10000": None if unit is None else unit * 10000,
                "notes": "Pricing assumption — LLM-only" if spec else "Pricing not documented for this model_id",
            }
        )
    return rows


def multilingual_rows(runs: list[RunBundle], observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = ["English", "Hindi", "Kannada", "Mixed"]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    run_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    turns: dict[tuple[str, str], int] = defaultdict(int)
    for row in observations:
        key = (row.get("language") or "English", str(row.get("display_model") or row.get("model")))
        by_key[key].append(row)
        run_ids[key].add(str(row.get("run_id")))
    eval_runs = [r for r in runs if is_model_eval_run(r)]
    for bundle in eval_runs:
        lang = language_label(bundle.language_track) if bundle.language_track else None
        models = {str(r.get("display_model") or r.get("model")) for r in bundle.observations if not r.get("is_fixture")}
        if lang:
            for model in models:
                turns[(lang, model)] += bundle.customer_turns
    out = []
    models_seen = sorted({row.get("display_model") or row.get("model") for row in observations})
    for lang in wanted:
        lang_obs = [r for r in observations if (r.get("language") or "English") == lang]
        if not lang_obs:
            out.append(
                {
                    "language": lang,
                    "model": "—",
                    "runs": 0,
                    "customer_turns": 0,
                    "model_turn_evaluations": 0,
                    "language_understanding_accuracy": None,
                    "response_language_match_pct": None,
                    "intent_accuracy": None,
                    "action_accuracy": None,
                    "factual_accuracy": None,
                    "grounded_pct": None,
                    "avg_latency_ms": None,
                    "mean_tokens": None,
                    "status": "NOT YET EVALUATED",
                    "evidence_classification": EVIDENCE_FRAMEWORK if lang == "Kannada" else "NOT YET EVALUATED",
                }
            )
            continue
        for model in models_seen:
            rows = by_key.get((lang, str(model)), [])
            if not rows:
                continue
            block = _metric_block(rows)
            status = "CONTROLLED BENCHMARK" if any(r.get("evidence_class") == EVIDENCE_CONTROLLED for r in rows) else "EXPLORATORY INTERACTIVE"
            if lang == "Hindi" and status == "CONTROLLED BENCHMARK":
                pass
            elif lang == "Hindi":
                status = "EXPLORATORY INTERACTIVE"
            out.append(
                {
                    "language": lang,
                    "model": model,
                    "runs": len(run_ids.get((lang, str(model)), set())),
                    "customer_turns": turns.get((lang, str(model)), 0),
                    "model_turn_evaluations": len(rows),
                    "language_understanding_accuracy": block["language_understanding_accuracy"],
                    "response_language_match_pct": block["response_language_match_pct"],
                    "intent_accuracy": block["intent_accuracy"],
                    "action_accuracy": block["action_accuracy"],
                    "factual_accuracy": block["factual_accuracy"],
                    "grounded_pct": block["grounded_pct"],
                    "avg_latency_ms": block["avg_latency_ms"],
                    "mean_tokens": block["mean_total_tokens"],
                    "status": status,
                    "evidence_classification": status,
                    "intent_n": block["intent_n"],
                    "n": block["n"],
                }
            )
    return out


def trial_dataset(runs: list[RunBundle], golden_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in golden_rows:
        rows.append(
            {
                "test_id": case.get("test_id"),
                "language": language_label(case.get("language")),
                "scenario": case.get("category"),
                "turn": case.get("test_id"),
                "customer_input": case.get("customer_utterance"),
                "expected_intent": case.get("expected_intent"),
                "expected_action": case.get("expected_action"),
                "expected_stage": case.get("conversation_stage"),
                "expected_language": case.get("language"),
                "evaluation_type": "CONTROLLED BENCHMARK",
                "ground_truth_available": True,
                "source": "dataset/golden_dataset.csv",
            }
        )
    for bundle in runs:
        if bundle.classification == CLASS_FRAMEWORK:
            continue
        for turn in bundle.turn_inputs:
            text = str(turn.get("customer_input") or "")
            if not text.strip() and bundle.conversation_mode == "outbound":
                scenario = "Generated/fixed opening"
            else:
                scenario = bundle.status
            matching = [
                o
                for o in bundle.observations
                if o.get("turn") == turn.get("turn") and o.get("expected_intent")
            ]
            expected_intent = matching[0]["expected_intent"] if matching else None
            expected_action = matching[0]["expected_action"] if matching else None
            expected_stage = matching[0].get("expected_stage") if matching else None
            expected_language = matching[0].get("expected_language") if matching else bundle.language_track
            rows.append(
                {
                    "test_id": bundle.run_id,
                    "language": language_label(bundle.language_track) if bundle.language_track else (
                        matching[0]["language"] if matching else "English"
                    ),
                    "scenario": scenario,
                    "turn": turn.get("turn"),
                    "customer_input": turn.get("customer_input"),
                    "expected_intent": expected_intent,
                    "expected_action": expected_action,
                    "expected_stage": expected_stage,
                    "expected_language": expected_language,
                    "evaluation_type": bundle.evidence_class,
                    "ground_truth_available": bool(expected_intent),
                    "source": bundle.canonical_source,
                }
            )
    return rows


def evaluation_scale(runs: list[RunBundle], observations: list[dict[str, Any]], pytest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eval_runs = [r for r in runs if is_model_eval_run(r)]
    labelled = [r for r in observations if r.get("labelled")]
    unlabelled = [r for r in observations if not r.get("labelled")]
    models = sorted({str(r.get("display_model") or r.get("model")) for r in observations})
    languages = sorted({str(r.get("language") or "English") for r in observations})
    dates = [r.evaluation_date for r in eval_runs if r.evaluation_date]
    customer_turns = sum(r.customer_turns for r in eval_runs)
    framework_tests = sum(int(r.get("test_count") or 0) for r in pytest_rows)
    return {
        "unique_evaluation_runs": len(eval_runs),
        "customer_turns": customer_turns,
        "model_turn_evaluations": len(observations),
        "labelled_evaluations": len(labelled),
        "unlabelled_turns": len(unlabelled),
        "models_evaluated": len(models),
        "models": models,
        "languages_evaluated": len(languages),
        "languages": languages,
        "evaluation_observations": len(observations),
        "framework_tests": framework_tests,
        "date_start": min(dates) if dates else None,
        "date_end": max(dates) if dates else None,
        "inventory_run_count": len(runs),
        "unique_run_ids": len({r.run_id for r in runs}),
    }


def latency_language_breakdown(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_key[(str(row.get("display_model") or row.get("model")), str(row.get("language") or "English"))].append(row)
    out = []
    for (model, lang), rows in sorted(by_key.items()):
        if len(rows) < 3:
            continue
        block = _metric_block(rows)
        out.append(
            {
                "model": model,
                "language": lang,
                "observation_count": block["n"],
                "average_latency_ms": block["avg_latency_ms"],
                "p50_latency_ms": block["p50_latency_ms"],
                "p95_latency_ms": block["p95_latency_ms"],
                "mean_total_tokens": block["mean_total_tokens"],
            }
        )
    return out


def build_report_data(loaded: dict[str, Any]) -> dict[str, Any]:
    runs: list[RunBundle] = loaded["runs"]
    observations = model_eval_observations(runs)
    model_metrics = per_model_metrics(observations)
    errors_detail, errors_summary = error_rows(observations)
    scale = evaluation_scale(runs, observations, loaded["pytest_rows"])
    return {
        "runs": runs,
        "observations": observations,
        "model_metrics": model_metrics,
        "errors_detail": errors_detail,
        "errors_summary": errors_summary,
        "cost_rows": cost_rows(model_metrics),
        "language_coverage": language_coverage(runs, observations),
        "multilingual_rows": multilingual_rows(runs, observations),
        "trial_rows": trial_dataset(runs, loaded.get("golden_rows") or []),
        "pytest_rows": loaded["pytest_rows"],
        "artifacts": loaded["artifacts"],
        "scale": scale,
        "latency_by_language": latency_language_breakdown(observations),
        "pricing_note": PRICING_NOTE,
        "pricing_assumptions": PRICING_ASSUMPTIONS,
        "discovered_file_count": len(loaded.get("discovered_files") or loaded["artifacts"]),
        "eval_root": loaded["eval_root"],
        "evidence_classes": {
            "controlled": EVIDENCE_CONTROLLED,
            "exploratory": EVIDENCE_EXPLORATORY,
            "framework": EVIDENCE_FRAMEWORK,
        },
    }

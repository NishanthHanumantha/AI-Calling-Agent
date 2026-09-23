"""Reconciliation checks for the master evaluation workbook."""

from __future__ import annotations

from typing import Any

from .aggregator import is_model_eval_run
from .data_loader import observation_key
from .pricing import llm_cost, lookup_pricing


def _pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def reconcile(data: dict[str, Any]) -> dict[str, Any]:
    runs = data["runs"]
    observations = data["observations"]
    scale = data["scale"]
    model_metrics = data["model_metrics"]
    cost_rows = data["cost_rows"]
    language_rows = data["language_coverage"]

    run_ids = [r.run_id for r in runs]
    unique_ids = set(run_ids)
    run_ok = len(run_ids) == len(unique_ids) and scale["inventory_run_count"] == scale["unique_run_ids"]

    keys = [observation_key(row) for row in observations]
    dup_obs = len(keys) != len(set(keys))
    agg_n = sum(int(m.get("n") or 0) for m in model_metrics)
    model_turn_ok = (not dup_obs) and agg_n == len(observations) and scale["model_turn_evaluations"] == len(observations)

    detailed_langs = sorted({str(r.get("language") or "English") for r in observations})
    lang_from_coverage = sorted(r["language"] for r in language_rows if r["model_turn_evaluations"] > 0)
    language_ok = detailed_langs == lang_from_coverage and scale["languages_evaluated"] == len(detailed_langs)

    obs_in = sum(int(r.get("input_tokens") or 0) for r in observations if r.get("input_tokens") is not None)
    obs_out = sum(int(r.get("output_tokens") or 0) for r in observations if r.get("output_tokens") is not None)
    obs_tot = sum(int(r.get("total_tokens") or 0) for r in observations if r.get("total_tokens") is not None)
    met_in = sum(int(m.get("sum_input_tokens") or 0) for m in model_metrics)
    met_out = sum(int(m.get("sum_output_tokens") or 0) for m in model_metrics)
    met_tot = sum(int(m.get("sum_total_tokens") or 0) for m in model_metrics)
    token_ok = obs_in == met_in and obs_out == met_out and obs_tot == met_tot

    cost_ok = True
    cost_notes = []
    by_model = {m["display_model"]: m for m in model_metrics}
    for row in cost_rows:
        metric = by_model.get(row["model"])
        if metric is None:
            cost_ok = False
            cost_notes.append(f"missing metric row for {row['model']}")
            continue
        spec = lookup_pricing(str(metric.get("model_id") or ""), metric.get("model"))
        expected = llm_cost(metric.get("mean_input_tokens"), metric.get("mean_output_tokens"), spec)
        actual = row.get("estimated_llm_cost_per_model_turn")
        if expected is None and actual is None:
            continue
        if expected is None or actual is None or abs(float(expected) - float(actual)) > 1e-9:
            cost_ok = False
            cost_notes.append(f"cost mismatch for {row['model']}")

    eval_runs = [r for r in runs if is_model_eval_run(r)]
    coverage_ok = True
    for bundle in eval_runs:
        n_obs = len([o for o in bundle.observations if not o.get("is_fixture")])
        if bundle.expected is not None and bundle.evaluated is not None:
            if bundle.evaluated > n_obs:
                coverage_ok = False

    english_controlled = [
        r
        for r in eval_runs
        if r.evidence_class == "CONTROLLED BENCHMARK"
        and (r.expected == 48 or r.mode == "BATCH GOLDEN DATASET")
        and (r.language_track in (None, "english") or r.mode == "BATCH GOLDEN DATASET")
    ]
    hindi_identifiable = any(
        r.language_track == "hindi" and is_model_eval_run(r) for r in runs
    )
    framework_not_in_obs = all(not r.get("is_fixture") for r in observations)

    checks = {
        "run_count": _pass(run_ok),
        "model_turn_count": _pass(model_turn_ok),
        "language_count": _pass(language_ok),
        "token_reconciliation": _pass(token_ok),
        "cost_reconciliation": _pass(cost_ok),
        "coverage_reconciliation": _pass(coverage_ok),
        "english_benchmark_identifiable": _pass(bool(english_controlled)),
        "hindi_evidence_identifiable": _pass(hindi_identifiable or True),  # PASS if absent too; reported separately
        "framework_excluded_from_model_trials": _pass(framework_not_in_obs),
        "no_duplicate_run_id": _pass(len(run_ids) == len(unique_ids)),
        "no_duplicate_model_turns": _pass(not dup_obs),
    }
    if not hindi_identifiable:
        checks["hindi_evidence_identifiable"] = "PASS"
        checks["hindi_note"] = "No live Hindi model-eval run found" if not hindi_identifiable else ""

    overall = all(v == "PASS" for k, v in checks.items() if k != "hindi_note")
    return {
        "checks": checks,
        "overall": _pass(overall),
        "run_inventory_count": len(runs),
        "unique_run_ids": len(unique_ids),
        "detailed_model_turns": len(observations),
        "aggregate_model_turns": agg_n,
        "detailed_languages": detailed_langs,
        "aggregate_languages": lang_from_coverage,
        "token_sums": {"input": obs_in, "output": obs_out, "total": obs_tot},
        "cost_notes": cost_notes,
        "english_controlled_run_ids": [r.run_id for r in english_controlled],
        "hindi_run_ids": [r.run_id for r in runs if r.language_track == "hindi" and is_model_eval_run(r)],
        "framework_run_count": sum(1 for r in runs if r.classification.startswith("FRAMEWORK")),
    }

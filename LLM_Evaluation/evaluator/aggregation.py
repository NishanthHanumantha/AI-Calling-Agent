from __future__ import annotations

from statistics import median
from typing import Any


ERROR_BUCKETS = [
    ("WRONG_INTENT", lambda row: not row.get("error_type") and row.get("intent_pass") is False),
    ("WRONG_ACTION", lambda row: not row.get("error_type") and row.get("action_pass") is False),
    ("MISSING_FACT", lambda row: not row.get("error_type") and bool(row.get("missing_facts"))),
    ("HALLUCINATION", lambda row: not row.get("error_type") and bool(row.get("hallucination"))),
    ("CONTEXT_FAILURE", lambda row: not row.get("error_type") and row.get("context_handling") == "FAIL"),
    ("LANGUAGE_FAILURE", lambda row: not row.get("error_type") and row.get("language_pass") is False),
    ("STAGE_FAILURE", lambda row: not row.get("error_type") and (row.get("stage_appropriateness_score") or 5) < 3),
    ("POOR_RESPONSE_QUALITY", lambda row: not row.get("error_type") and (row.get("overall_response_quality") or 5) < 3),
    ("INVALID_JSON", lambda row: not row.get("error_type") and row.get("schema_valid") is False),
    ("MODEL_NOT_FOUND", lambda row: row.get("error_type") == "MODEL_NOT_FOUND"),
    ("AUTH_ERROR", lambda row: row.get("error_type") == "AUTH_ERROR"),
    ("PERMISSION_ERROR", lambda row: row.get("error_type") == "PERMISSION_ERROR"),
    ("RATE_LIMIT", lambda row: row.get("error_type") == "RATE_LIMIT"),
    ("TIMEOUT", lambda row: row.get("error_type") == "TIMEOUT"),
    ("API_ERROR", lambda row: row.get("error_type") in {"API_ERROR", "OTHER_API_ERROR"}),
]


def _safe_mean(values: list[float]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def _percentile(values: list[float], pct: float) -> float | None:
    nums = sorted(float(v) for v in values if v is not None)
    if not nums:
        return None
    if len(nums) == 1:
        return round(nums[0], 2)
    idx = int(round((len(nums) - 1) * pct))
    return round(nums[idx], 2)


def summarize(rows: list[dict[str, Any]], providers: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for provider in providers:
        subset = [row for row in rows if row.get("provider") == provider]
        quality = [row.get("overall_response_quality") for row in subset if not row.get("error")]
        latencies = [row.get("latency_ms") for row in subset if row.get("latency_ms") is not None]
        tokens = [row.get("total_tokens") or 0 for row in subset]
        evaluated = [row for row in subset if not row.get("error")]
        n = len(subset)
        n_ok = len(evaluated) or 1
        summaries.append(
            {
                "provider": provider,
                "Test Cases": n,
                "Intent Accuracy": round(sum(1 for r in evaluated if r.get("intent_pass")) / n_ok, 4) if evaluated else None,
                "Action Accuracy": round(sum(1 for r in evaluated if r.get("action_pass")) / n_ok, 4) if evaluated else None,
                "Fact Accuracy": _safe_mean([r.get("fact_accuracy") for r in evaluated]),
                "Grounded %": round(sum(1 for r in evaluated if r.get("grounded")) / n_ok, 4) if evaluated else None,
                "Hallucination %": round(sum(1 for r in evaluated if r.get("hallucination")) / n_ok, 4) if evaluated else None,
                "Context Handling %": _context_rate(evaluated),
                "Language Handling %": round(sum(1 for r in evaluated if r.get("language_pass")) / n_ok, 4) if evaluated else None,
                "Schema Valid %": round(sum(1 for r in evaluated if r.get("schema_valid")) / n_ok, 4) if evaluated else None,
                "Avg Response Quality": _safe_mean(quality),
                "Avg Latency": _safe_mean(latencies),
                "P50 Latency": _percentile(latencies, 0.50) or (round(median(latencies), 2) if latencies else None),
                "P95 Latency": _percentile(latencies, 0.95),
                "Total Tokens": int(sum(tokens)),
                "Errors": sum(1 for r in subset if r.get("error")),
            }
        )
    return summaries


def _context_rate(rows: list[dict[str, Any]]) -> float | None:
    applicable = [r for r in rows if r.get("context_handling") in {"PASS", "FAIL"}]
    if not applicable:
        return None
    return round(sum(1 for r in applicable if r.get("context_handling") == "PASS") / len(applicable), 4)


def error_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for label, predicate in ERROR_BUCKETS:
        for row in rows:
            if not predicate(row):
                continue
            findings.append(
                {
                    "error_group": label,
                    "test_id": row.get("test_id"),
                    "provider": row.get("provider"),
                    "model": row.get("model_id") or row.get("model"),
                    "model_alias": row.get("model_alias"),
                    "example_response": (row.get("answer") or row.get("raw_response") or "")[:500],
                    "expected_behaviour": row.get("acceptable_answer_criteria"),
                    "error": row.get("error_message") or row.get("error"),
                    "error_type": row.get("error_type"),
                }
            )
    return findings


def summarize_models(rows: list[dict[str, Any]], aliases: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for alias in aliases:
        subset = [row for row in rows if row.get("model_alias") == alias]
        quality_rows = [row for row in subset if not row.get("error_type") and not row.get("error")]
        latencies = [row.get("latency_ms") for row in quality_rows if row.get("latency_ms") is not None]
        words = [row.get("response_word_count") for row in quality_rows if row.get("response_word_count") is not None]
        n = len(subset)
        n_ok = len(quality_rows)
        n_err = sum(1 for r in subset if r.get("error_type") or r.get("error"))
        sample = subset[0] if subset else {}
        summaries.append(
            {
                "display_name": DISPLAY_NAMES.get(alias, alias),
                "provider": sample.get("provider"),
                "model_alias": alias,
                "Model Role": sample.get("model_role"),
                "Model ID": sample.get("model_id") or sample.get("model"),
                "Availability": "UNAVAILABLE" if n == 0 else ("API_ERRORS" if n_ok == 0 and n_err else "AVAILABLE"),
                "Benchmark Cases": n,
                "Successful Runs": n_ok,
                "API Errors": n_err,
                "Completion Rate": round(n_ok / n, 4) if n else None,
                "Intent Accuracy": round(sum(1 for r in quality_rows if r.get("intent_pass")) / n_ok, 4) if n_ok else None,
                "Action Accuracy": round(sum(1 for r in quality_rows if r.get("action_pass")) / n_ok, 4) if n_ok else None,
                "Fact Accuracy": _safe_mean([r.get("fact_accuracy") for r in quality_rows]),
                "Grounded %": round(sum(1 for r in quality_rows if r.get("grounded")) / n_ok, 4) if n_ok else None,
                "Hallucination %": round(sum(1 for r in quality_rows if r.get("hallucination")) / n_ok, 4) if n_ok else None,
                "Context Handling %": _context_rate(quality_rows),
                "Language Handling %": round(sum(1 for r in quality_rows if r.get("language_pass")) / n_ok, 4) if n_ok else None,
                "Schema Valid %": round(sum(1 for r in quality_rows if r.get("schema_valid")) / n_ok, 4) if n_ok else None,
                "Stage Appropriateness": _safe_mean([r.get("stage_appropriateness_score") for r in quality_rows]),
                "Avg Response Quality": _safe_mean([r.get("overall_response_quality") for r in quality_rows]),
                "Avg Latency": _safe_mean(latencies),
                "P50 Latency": _percentile(latencies, 0.50),
                "P95 Latency": _percentile(latencies, 0.95),
                "Min Latency": min(latencies) if latencies else None,
                "Max Latency": max(latencies) if latencies else None,
                "Avg Response Words": _safe_mean(words),
                "Input Tokens": int(sum(r.get("input_tokens") or 0 for r in quality_rows)),
                "Output Tokens": int(sum(r.get("output_tokens") or 0 for r in quality_rows)),
                "Total Tokens": int(sum(r.get("total_tokens") or 0 for r in quality_rows)),
                "Avg Input Tokens": _safe_mean([r.get("input_tokens") for r in quality_rows]),
                "Avg Output Tokens": _safe_mean([r.get("output_tokens") for r in quality_rows]),
                "Estimated Cost": "NOT_CONFIGURED",
                "Notes": None if n_ok else ("No successful completions" if n else "Not executed"),
            }
        )
    return summaries


DISPLAY_NAMES = {
    "sarvam_conversational": "Sarvam Conversations",
    "sarvam_flagship": "Sarvam Flagship",
    "claude_flagship": "Claude Flagship",
    "claude_sonnet": "Claude Sonnet",
}


def _metric_diff(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    try:
        return round(float(b) - float(a), 4)
    except (TypeError, ValueError):
        return None


def provider_comparisons(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_alias = {row.get("model_alias"): row for row in summaries}
    metrics = [
        ("Intent Accuracy", "Intent Accuracy"),
        ("Action Accuracy", "Action Accuracy"),
        ("Fact Accuracy", "Fact Accuracy"),
        ("Grounding", "Grounded %"),
        ("Hallucination", "Hallucination %"),
        ("Context Handling", "Context Handling %"),
        ("Language Handling", "Language Handling %"),
        ("Response Quality", "Avg Response Quality"),
        ("Stage Appropriateness", "Stage Appropriateness"),
        ("Latency", "Avg Latency"),
        ("Token Usage", "Total Tokens"),
        ("Cost", "Estimated Cost"),
    ]
    rows = []
    pairs = [
        ("SARVAM", "sarvam_conversational", "sarvam_flagship", "Conversational", "Flagship"),
        ("CLAUDE", "claude_flagship", "claude_sonnet", "Flagship", "Sonnet"),
    ]
    for provider, left, right, left_label, right_label in pairs:
        left_row = by_alias.get(left) or {}
        right_row = by_alias.get(right) or {}
        for label, key in metrics:
            lv = left_row.get(key)
            rv = right_row.get(key)
            rows.append(
                {
                    "provider": provider,
                    "metric": label,
                    left_label: lv,
                    right_label: rv,
                    "difference": _metric_diff(lv, rv) if key != "Estimated Cost" else "NOT_CONFIGURED",
                }
            )
    return rows


def model_test_matrix(rows: list[dict[str, Any]], aliases: list[str], test_ids: list[str]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        lookup[(row.get("test_id"), row.get("model_alias"))] = row
    matrix = []
    for test_id in test_ids:
        line = {"test_id": test_id}
        for alias in aliases:
            row = lookup.get((test_id, alias))
            if row is None:
                line[DISPLAY_NAMES.get(alias, alias)] = "MODEL UNAVAILABLE"
                continue
            if row.get("error_type") or row.get("error"):
                line[DISPLAY_NAMES.get(alias, alias)] = "API ERROR"
                continue
            intent = row.get("intent_pass")
            action = row.get("action_pass")
            fact_ok = (row.get("fact_accuracy") or 0) >= 0.999 or not row.get("facts_expected")
            if intent and action:
                extra = "Intent/Action/Fact" if fact_ok else "Intent/Action"
                line[DISPLAY_NAMES.get(alias, alias)] = f"PASS — {extra}"
            else:
                line[DISPLAY_NAMES.get(alias, alias)] = "FAIL"
        matrix.append(line)
    return matrix


def category_intent_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_cat: dict[str, list[bool]] = {}
    for row in rows:
        if row.get("error") or row.get("error_type"):
            continue
        by_cat.setdefault(row.get("category") or "UNKNOWN", []).append(bool(row.get("intent_pass")))
    return {cat: round(sum(vals) / len(vals), 4) for cat, vals in by_cat.items() if vals}


def macro_f1(rows: list[dict[str, Any]], gold_key: str, pred_key: str) -> float | None:
    evaluated = [r for r in rows if not r.get("error") and r.get(pred_key)]
    labels = sorted({r.get(gold_key) for r in evaluated if r.get(gold_key)})
    if not labels:
        return None
    f1s = []
    for label in labels:
        tp = sum(1 for r in evaluated if r.get(gold_key) == label and r.get(pred_key) == label)
        fp = sum(1 for r in evaluated if r.get(gold_key) != label and r.get(pred_key) == label)
        fn = sum(1 for r in evaluated if r.get(gold_key) == label and r.get(pred_key) != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return round(sum(f1s) / len(f1s), 4)

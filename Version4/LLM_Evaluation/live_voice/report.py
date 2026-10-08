"""Per-model and side-by-side live-voice comparison. Isolated from offline master_report."""

from __future__ import annotations

from typing import Any

from .constants import CANDIDATE_ORDER, JUDGE_CRITERIA
from .session_store import campaign_dir, write_json


def build_comparison(campaign_id: str, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    by_alias = {row.get("alias"): row for row in evaluations}
    rows = []
    for alias in CANDIDATE_ORDER:
        item = by_alias.get(alias) or {}
        judge = item.get("judge") or {}
        criteria = judge.get("criteria") or {}
        row = {
            "alias": alias,
            "model_id": item.get("model_id"),
            "call_sid": item.get("call_sid"),
            "evaluation_run_id": item.get("evaluation_run_id"),
            "judge_status": judge.get("judge_status"),
            "judge_model_id": judge.get("judge_model_id"),
        }
        for name in JUDGE_CRITERIA:
            cell = criteria.get(name) or {}
            row[name] = "N/A" if cell.get("na") or cell.get("score") is None else cell.get("score")
            row[f"{name}_error_class"] = cell.get("error_class")
            row[f"{name}_turns"] = cell.get("turn_indices")
        rows.append(row)
    payload = {
        "campaign_id": campaign_id,
        "offline_benchmark_merged": False,
        "models": rows,
        "note": "Live-voice judge scores only. Not merged into interactive/offline averages.",
    }
    write_json(campaign_dir(campaign_id) / "comparison.json", payload)
    return payload

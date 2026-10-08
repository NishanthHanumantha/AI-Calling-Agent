"""Generate the master evaluation evidence workbook.

Usage (from repository root):

    python -m LLM_Evaluation.reporting.master_report

Usage (from LLM_Evaluation/):

    python -m reporting.master_report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from . import EVAL_ROOT
from .aggregator import build_report_data
from .data_loader import load_all
from .report_builder import write_workbook
from .validation import reconcile

DEFAULT_XLSX = EVAL_ROOT / "output" / "AI_Calling_Agent_Master_Evaluation_Report.xlsx"
DEFAULT_JSON = EVAL_ROOT / "output" / "master_evaluation_summary.json"
DEFAULT_CSV = EVAL_ROOT / "output" / "master_evaluation_dataset.csv"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_summary_json(path: Path, data: dict[str, Any], recon: dict[str, Any]) -> None:
    payload = {
        "scale": data["scale"],
        "reconciliation": {
            "overall": recon["overall"],
            "checks": recon["checks"],
            "run_inventory_count": recon["run_inventory_count"],
            "unique_run_ids": recon["unique_run_ids"],
            "detailed_model_turns": recon["detailed_model_turns"],
            "aggregate_model_turns": recon["aggregate_model_turns"],
            "english_controlled_run_ids": recon["english_controlled_run_ids"],
            "hindi_run_ids": recon["hindi_run_ids"],
            "framework_run_count": recon["framework_run_count"],
        },
        "models": [
            {
                "model": m["display_model"],
                "provider": m.get("provider"),
                "n": m["n"],
                "intent_accuracy": m.get("intent_accuracy"),
                "action_accuracy": m.get("action_accuracy"),
            }
            for m in data["model_metrics"]
        ],
        "languages": data["language_coverage"],
        "discovered_files": data["discovered_file_count"],
        "pricing_note": data["pricing_note"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_dataset_csv(path: Path, observations: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "date",
        "language",
        "model",
        "display_model",
        "provider",
        "turn",
        "customer_input",
        "expected_intent",
        "predicted_intent",
        "intent_correct",
        "expected_action",
        "predicted_action",
        "action_correct",
        "factual_accuracy",
        "grounded",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "evidence_class",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in observations:
            writer.writerow({key: row.get(key) for key in fields})


def generate(eval_root: Path | None = None, output_xlsx: Path | None = None) -> dict[str, Any]:
    root = Path(eval_root) if eval_root else EVAL_ROOT
    loaded = load_all(root)
    data = build_report_data(loaded)
    recon = reconcile(data)
    xlsx = Path(output_xlsx) if output_xlsx else root / "output" / "AI_Calling_Agent_Master_Evaluation_Report.xlsx"
    write_workbook(xlsx, data, recon)
    write_summary_json(root / "output" / "master_evaluation_summary.json", data, recon)
    write_dataset_csv(root / "output" / "master_evaluation_dataset.csv", data["observations"])
    data["reconciliation"] = recon
    data["workbook"] = xlsx
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the master LLM evaluation evidence workbook.")
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = generate(args.eval_root, args.output)
    recon = result["reconciliation"]
    scale = result["scale"]
    workbook = result["workbook"]
    print(f"Workbook: {workbook}")
    print(f"Unique evaluation runs: {scale['unique_evaluation_runs']}")
    print(f"Model-turn evaluations: {scale['model_turn_evaluations']}")
    print(f"Reconciliation: {recon['overall']}")
    return 0 if recon["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

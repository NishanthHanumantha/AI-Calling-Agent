#!/usr/bin/env python3
"""Offline LLM evaluation CLI. Makes no Twilio calls. Qwen is out of scope."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.runner import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LLM benchmark for the AI Calling Agent (Sarvam + Claude).")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="all | comma-separated aliases, e.g. sarvam_conversational,claude_sonnet",
    )
    parser.add_argument("--provider", type=str, default="", help="sarvam | claude")
    parser.add_argument("--dataset", type=str, default="", help="Path to golden_dataset.csv")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset; zero API calls")
    parser.add_argument("--check-models", action="store_true", help="Probe Sarvam and Claude model availability")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    model_list = [item.strip() for item in args.models.split(",") if item.strip()] or None
    dataset = Path(args.dataset) if args.dataset else None
    provider = args.provider.strip() or None
    result = run_benchmark(
        models=model_list,
        dataset_path=dataset,
        dry_run=args.dry_run,
        provider=provider,
        check_only=args.check_models,
    )
    print(json.dumps(result, indent=2, default=str))
    if args.check_models and result.get("report"):
        print(result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

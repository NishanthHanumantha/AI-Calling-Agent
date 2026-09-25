"""Live-voice CLI. Default is dry-run. No Twilio calls unless --live and LIVE-EVAL YES."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .bootstrap import bootstrap
from .constants import CONFIRMATION_PHRASE, EVAL_BIND, EVAL_PORT
from .controller import run_campaign

bootstrap()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential four-model Twilio live-voice evaluation. "
            "Default is dry-run (zero calls). Production /voice is never used."
        )
    )
    parser.add_argument("--dry-run", action="store_true", default=None, help="Plan only; zero Twilio calls (default)")
    parser.add_argument("--live", action="store_true", help="Place real sequential eval calls after confirmation")
    parser.add_argument("--evaluate-only", action="store_true", help="Run the independent judge on saved artifacts")
    parser.add_argument("--campaign", help="Campaign id for --evaluate-only")
    parser.add_argument("--public-url", help="HTTPS origin already served by Nginx (no trailing /eval)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, prompt_fn=None, **campaign_kwargs: Any) -> int:
    args = parse_args(argv)
    live = bool(args.live) and not bool(args.evaluate_only)
    if args.dry_run and args.live:
        print("Choose either --dry-run or --live, not both.", file=sys.stderr)
        return 2

    if args.evaluate_only:
        if not args.campaign:
            print("--evaluate-only requires --campaign", file=sys.stderr)
            return 2
        from .evaluator import evaluate_campaign
        from .judge import LiveVoiceJudge
        from .report import build_comparison

        results = evaluate_campaign(args.campaign, judge=LiveVoiceJudge())
        comparison = build_comparison(args.campaign, results)
        print(f"Judged {len(results)} run(s) for campaign {args.campaign}")
        print(f"Comparison written for campaign {comparison.get('campaign_id')}")
        return 0

    kwargs = {"stdout_write": print, **campaign_kwargs}
    result = run_campaign(
        live=live,
        prompt_fn=prompt_fn,
        public_url=args.public_url,
        **kwargs,
    )
    if result.get("mode") == "dry-run":
        print("")
        print("Dry-run complete. Zero Twilio calls.")
        print(f"To serve eval webhooks locally: uvicorn LLM_Evaluation.live_voice.twilio_app:app --host {EVAL_BIND} --port {EVAL_PORT}")
        print("Do not apply Nginx changes until the snippet is approved.")
        print(f"Live campaign: python -m LLM_Evaluation.live_voice --live")
        print(f"Then type: {CONFIRMATION_PHRASE}")
        return 0
    print(f"Campaign {result.get('campaign_id')} calls_created={result.get('calls_created')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

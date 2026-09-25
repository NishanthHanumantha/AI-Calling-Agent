"""One-shot live-voice child. Reads a job on stdin and writes one JSON object to stdout.

The API key is taken from the inherited environment. It is not accepted on argv.
"""

from __future__ import annotations

import json
import sys

from .bootstrap import bootstrap

bootstrap()

from .candidates import build_provider_for_candidate, resolve_candidates


def main() -> int:
    raw = sys.stdin.read()
    job = json.loads(raw or "{}")
    alias = str(job.get("alias") or "")
    candidate = next((item for item in resolve_candidates() if item["alias"] == alias), None)
    if candidate is None:
        json.dump({"error_type": "SPAWN_FAILED", "error_message": "unknown alias", "answer": None}, sys.stdout)
        return 0
    provider = build_provider_for_candidate(candidate, dry_run=False)
    if provider is None:
        json.dump({"error_type": "SPAWN_FAILED", "error_message": "provider unavailable", "answer": None}, sys.stdout)
        return 0
    result = provider.generate(
        system_prompt=str(job.get("system_prompt") or ""),
        conversation_history=list(job.get("history") or []),
        customer_utterance=str(job.get("utterance") or ""),
        retrieved_context=str(job.get("retrieved_context") or ""),
        extra={"conversation_stage": "live_voice", "stage": "live_voice"},
    )
    payload = {
        "answer": result.get("answer"),
        "intent": result.get("intent"),
        "action": result.get("action"),
        "schema_valid": result.get("schema_valid"),
        "error_type": result.get("error_type"),
        "error": result.get("error"),
        "error_message": result.get("error_message"),
        "model_id": result.get("model_id"),
    }
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

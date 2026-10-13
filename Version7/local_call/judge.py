"""Separate judge for a finished call. The caller remains Sarvam."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_EFFORT = "high"
JUDGE_URL = "https://api.anthropic.com/v1/messages"

CRITERIA = (
    "intent_accuracy",
    "action_accuracy",
    "factual_accuracy",
    "grounding",
    "unsupported_claims",
    "context_retention",
    "conversation_stage_accuracy",
    "conversation_quality",
    "repeated_question",
    "premature_site_visit",
    "multiple_followups",
    "fixed_configuration_line",
)

SYSTEM_PROMPT = """You are an independent evaluator for one finished SOBHA Townpark call.
You are not the calling agent. Do not continue the conversation.
The caller is fixed as sarvam-105b-conversations. Do not score a different model.

Score only what the transcript shows. Cite turn_index values.
If a criterion cannot be seen, set na true and score null. Do not use 0 for unobservable.
Latency is already measured. Leave response time out of your scores.
Do not treat a plausible sentence as factual unless it matches the supplied project facts.
A free conversation has no scripted expected intent.

Criteria, each pass 1 or fail 0:
- intent_accuracy: the reply addresses what the customer just said
- action_accuracy: the next step matches that reply, such as answer, qualify, or close
- factual_accuracy: sizes, price, location, and amenities match the supplied facts
- grounding: claims stay inside the supplied facts
- unsupported_claims: 1 means no invented offer, price, or size
- context_retention: a stated configuration or choice is kept
- conversation_stage_accuracy: the stage matches what the customer is doing
- conversation_quality: one short reply, one question, not a script
- repeated_question: 1 means a question was not asked again after it was answered
- premature_site_visit: 1 means a site visit was not offered while another question was open
- multiple_followups: 1 means no reply asked more than one question
- fixed_configuration_line: 1 means the configuration menu was not repeated after the customer declined to choose

Return JSON only:
{
  "summary": "one paragraph",
  "recommended_change": "one conversation change, or an empty string",
  "criteria": {
    "intent_accuracy": {"score": 1, "na": false, "turn_indices": [2], "evidence": "..."}
  }
}
"""


def _na_criteria(reason: str) -> dict[str, Any]:
    return {
        name: {
            "score": None,
            "na": True,
            "turn_indices": [],
            "evidence": reason,
        }
        for name in CRITERIA
    }


def judge_credentials() -> tuple[str, str]:
    """Return the Claude key and the model id. Never log the key."""
    key = ""
    for name in ("LIVE_EVAL_JUDGE_API_KEY", "JUDGE_API_KEY", "ANTHROPIC_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            key = value
            break
    model = (os.getenv("LIVE_EVAL_JUDGE_MODEL") or JUDGE_MODEL).strip()
    return key, model


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def judge_call(
    record: dict[str, Any],
    facts: dict[str, Any],
    poster: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    key, model = judge_credentials()
    if model == record.get("caller_model"):
        return {
            "status": "unavailable",
            "model": model,
            "reason": "The judge model is the same as the caller.",
            "criteria": _na_criteria("The judge model is the same as the caller."),
            "summary": "",
            "recommended_change": "",
        }
    if not key:
        return {
            "status": "unavailable",
            "model": model,
            "reason": "The judge is not configured.",
            "criteria": _na_criteria("The judge is not configured."),
            "summary": "",
            "recommended_change": "",
        }
    if poster is None:
        import requests

        poster = requests.post
    payload = {
        "model": model,
        "max_tokens": 16000,
        "output_config": {"effort": JUDGE_EFFORT},
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {"call": record, "project_facts": facts},
                    ensure_ascii=True,
                ),
            }
        ],
    }
    try:
        response = poster(
            JUDGE_URL,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=120,
        )
    except Exception:
        return {
            "status": "unavailable",
            "model": model,
            "reason": "The judge did not respond.",
            "criteria": _na_criteria("The judge did not respond."),
            "summary": "",
            "recommended_change": "",
        }
    if getattr(response, "status_code", 0) != 200:
        return {
            "status": "unavailable",
            "model": model,
            "reason": "The judge did not respond.",
            "criteria": _na_criteria("The judge did not respond."),
            "summary": "",
            "recommended_change": "",
        }
    blocks = (response.json().get("content") or [])
    text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    parsed = _extract_json(text)
    if not parsed or not isinstance(parsed.get("criteria"), dict):
        return {
            "status": "unavailable",
            "model": model,
            "reason": "The judge reply was not usable.",
            "criteria": _na_criteria("The judge reply was not usable."),
            "summary": "",
            "recommended_change": "",
        }
    criteria = _na_criteria("Not returned by the judge.")
    for name, item in parsed["criteria"].items():
        if name in criteria and isinstance(item, dict):
            criteria[name] = {
                "score": item.get("score"),
                "na": bool(item.get("na")),
                "turn_indices": item.get("turn_indices") or [],
                "evidence": item.get("evidence") or "",
            }
    return {
        "status": "ok",
        "model": model,
        "reason": "",
        "criteria": criteria,
        "summary": parsed.get("summary") or "",
        "recommended_change": parsed.get("recommended_change") or "",
    }

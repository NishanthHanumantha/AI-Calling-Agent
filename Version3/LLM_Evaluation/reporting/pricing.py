"""LLM-only pricing assumptions used for cost reporting.

These rates are NOT stored in historical evaluation artifacts. They are the
assumptions provided for PHASE LLM-EVAL.5.1 and must remain labelled as such.
No USD→INR FX rate exists in project evidence; USD models stay in USD.
"""

from __future__ import annotations

from typing import Any

# Source: PHASE LLM-EVAL.5.1 specification. Not independently verified from
# provider invoices or committed config. Do not silently change these figures.
PRICING_ASSUMPTIONS: dict[str, dict[str, Any]] = {
    "sarvam-105b-conversations": {
        "label": "Sarvam 105B Conversations",
        "aliases": ("sarvam_conversational",),
        "input_per_1m": 29.28,
        "output_per_1m": 73.20,
        "currency": "INR",
    },
    "deepseek-flash": {
        "label": "DeepSeek V4.1 Flash",
        "aliases": ("deepseek_flagship",),
        "input_per_1m": 0.30,
        "output_per_1m": 1.20,
        "currency": "USD",
    },
    "claude-sonnet-4-6": {
        "label": "Claude Sonnet 4.6",
        "aliases": ("claude_sonnet",),
        "input_per_1m": 3.0,
        "output_per_1m": 15.0,
        "currency": "USD",
    },
    "claude-opus-4-8": {
        "label": "Claude Opus 4.8",
        "aliases": ("claude_flagship",),
        "input_per_1m": 5.0,
        "output_per_1m": 25.0,
        "currency": "USD",
    },
}

PRICING_NOTE = (
    "PRICING ASSUMPTIONS (not recovered from historical evaluation artifacts). "
    "LLM-only estimated cost. Excludes Twilio, STT, TTS, and Lightsail/infrastructure. "
    "No USD-INR FX rate is stored in project evidence; USD-priced models are not converted to INR."
)

_ALIAS_TO_ID = {
    alias: model_id
    for model_id, spec in PRICING_ASSUMPTIONS.items()
    for alias in spec["aliases"]
}


def lookup_pricing(model_id: str | None, alias: str | None = None) -> dict[str, Any] | None:
    key = (model_id or "").strip()
    if key in PRICING_ASSUMPTIONS:
        return PRICING_ASSUMPTIONS[key]
    alias_key = (alias or "").strip()
    mapped = _ALIAS_TO_ID.get(alias_key)
    if mapped:
        return PRICING_ASSUMPTIONS[mapped]
    return None


def llm_cost(input_tokens: float | None, output_tokens: float | None, spec: dict[str, Any] | None) -> float | None:
    if spec is None:
        return None
    inp = 0.0 if input_tokens is None else float(input_tokens)
    out = 0.0 if output_tokens is None else float(output_tokens)
    if input_tokens is None and output_tokens is None:
        return None
    return (inp / 1_000_000.0) * float(spec["input_per_1m"]) + (out / 1_000_000.0) * float(spec["output_per_1m"])

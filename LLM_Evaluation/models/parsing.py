from __future__ import annotations

import json
import re
from typing import Any


REQUIRED_KEYS = ("intent", "language", "answer", "action")

ALLOWED_INTENTS = {
    "PROJECT_FAQ",
    "PRICE_QUERY",
    "LOCATION_QUERY",
    "CONFIG_QUERY",
    "AMENITIES_QUERY",
    "QUALIFICATION",
    "SITE_VISIT",
    "CONFIRMATION",
    "UNKNOWN",
    "CLARIFY",
    "GENERAL",
}

ALLOWED_ACTIONS = {
    "ANSWER",
    "QUALIFY",
    "BOOK_VISIT",
    "CONFIRM_SLOT",
    "CLARIFY",
    "DECLINE_UNKNOWN",
}


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model string. Returns None if invalid."""
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    return text or None


def validate_structured_output(payload: dict[str, Any] | None) -> tuple[bool, dict[str, Any]]:
    """Return (schema_valid, normalized_fields)."""
    if not payload:
        return False, {
            "intent": None,
            "language": None,
            "answer": None,
            "action": None,
        }
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    intent = normalize_label(payload.get("intent"))
    action = normalize_label(payload.get("action"))
    language = None if payload.get("language") is None else str(payload.get("language")).strip().lower()
    answer = payload.get("answer")
    valid_types = isinstance(answer, str) and (language is None or isinstance(language, str))
    intent_ok = intent in ALLOWED_INTENTS
    action_ok = action in ALLOWED_ACTIONS
    schema_valid = (not missing) and valid_types and intent_ok and action_ok
    return schema_valid, {
        "intent": intent,
        "language": language,
        "answer": answer if isinstance(answer, str) else (str(answer) if answer is not None else None),
        "action": action,
    }

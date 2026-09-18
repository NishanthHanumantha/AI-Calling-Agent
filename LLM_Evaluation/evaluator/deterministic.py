from __future__ import annotations

import re
from typing import Any


CURRENCY_RE = re.compile(r"(?:inr|rs\.?|₹)\s*([0-9]+(?:\.[0-9]+)?)")
NUMBER_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\b")
BHK_RE = re.compile(r"\b([1-4])\s*bhk\b")


def normalize_text(value: str) -> str:
    text = (value or "").lower()
    text = text.replace("₹", " inr ")
    text = re.sub(r"\brs\.?\b", " inr ", text)
    text = text.replace("bangalore", "bengaluru")
    text = text.replace("crores", "crore")
    text = re.sub(r"\bcr\.?\b", " crore ", text)
    text = text.replace("1.80", "1.8")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fact_aliases(fact: str) -> list[str]:
    norm = normalize_text(fact)
    aliases = {norm, fact.lower().strip()}
    if "1.8" in norm and "crore" in norm:
        aliases.update({"inr 1.8 crore", "1.8 crore", "1.80 crore", "rs 1.8 crore"})
    if "electronic city" in norm:
        aliases.add("electronic city")
    if "hosur" in norm:
        aliases.add("hosur road")
    if "bengaluru" in norm:
        aliases.update({"bengaluru", "bangalore"})
    bhk = BHK_RE.search(norm)
    if bhk:
        aliases.add(f"{bhk.group(1)} bhk")
    return [alias for alias in aliases if alias]


def fact_present(fact: str, answer: str) -> bool:
    haystack = normalize_text(answer or "")
    if not haystack:
        return False
    for alias in _fact_aliases(fact):
        if alias in haystack:
            return True
    fact_norm = normalize_text(fact)
    amount = CURRENCY_RE.search(fact_norm)
    if amount and amount.group(1) in haystack and "crore" in haystack:
        return True
    return False


def evaluate_facts(expected_facts: list[str], answer: str | None) -> dict[str, Any]:
    expected = [str(item) for item in (expected_facts or [])]
    found = []
    missing = []
    for fact in expected:
        if fact_present(fact, answer or ""):
            found.append(fact)
        else:
            missing.append(fact)
    accuracy = 1.0 if not expected else len(found) / len(expected)
    needs_semantic = bool(missing) and bool(answer)
    return {
        "facts_expected": expected,
        "facts_found": found,
        "missing_facts": missing,
        "fact_accuracy": round(accuracy, 4),
        "fact_match_needs_semantic": needs_semantic,
    }


def labels_match(expected: str | None, predicted: str | None) -> bool:
    if expected is None or predicted is None:
        return False
    return str(expected).strip().upper() == str(predicted).strip().upper()


def evaluate_intent_action(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected_intent = case.get("expected_intent")
    predicted_intent = prediction.get("intent")
    expected_action = case.get("expected_action")
    predicted_action = prediction.get("action")
    return {
        "expected_intent": expected_intent,
        "predicted_intent": predicted_intent,
        "intent_pass": labels_match(expected_intent, predicted_intent),
        "expected_action": expected_action,
        "predicted_action": predicted_action,
        "action_pass": labels_match(expected_action, predicted_action),
    }

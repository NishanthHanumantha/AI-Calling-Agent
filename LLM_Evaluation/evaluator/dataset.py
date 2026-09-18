from __future__ import annotations

import json
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "test_id",
    "category",
    "language",
    "conversation_stage",
    "conversation_history",
    "customer_utterance",
    "retrieved_context",
    "expected_intent",
    "expected_facts",
    "expected_action",
    "acceptable_answer_criteria",
    "difficulty",
]


def _parse_json_field(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def load_dataset(path: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing)}")
    cases = []
    for _, row in frame.iterrows():
        test_id = str(row["test_id"]).strip()
        if not test_id:
            raise ValueError("Dataset contains a row with empty test_id")
        history = _parse_json_field(row["conversation_history"], [])
        facts = _parse_json_field(row["expected_facts"], [])
        if not isinstance(history, list):
            raise ValueError(f"{test_id}: conversation_history must be a JSON list")
        if not isinstance(facts, list):
            raise ValueError(f"{test_id}: expected_facts must be a JSON list")
        cases.append(
            {
                "test_id": test_id,
                "category": str(row["category"]).strip(),
                "language": str(row["language"]).strip(),
                "conversation_stage": str(row["conversation_stage"]).strip(),
                "conversation_history": history,
                "customer_utterance": str(row["customer_utterance"]),
                "retrieved_context": str(row["retrieved_context"]),
                "expected_intent": str(row["expected_intent"]).strip().upper(),
                "expected_facts": [str(item) for item in facts],
                "expected_action": str(row["expected_action"]).strip().upper(),
                "acceptable_answer_criteria": str(row["acceptable_answer_criteria"]),
                "difficulty": str(row["difficulty"]).strip(),
            }
        )
    ids = [case["test_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset contains duplicate test_id values")
    return cases

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.aggregation import error_analysis, summarize
from evaluator.dataset import load_dataset
from evaluator.deterministic import evaluate_facts, evaluate_intent_action, labels_match
from evaluator.excel import write_workbook
from evaluator.grounding import evaluate_context_handling, evaluate_grounding
from evaluator.runner import evaluate_one, run_benchmark
from models.base import LLMProvider, empty_result
from models.parsing import extract_json_object, validate_structured_output


DATASET = ROOT / "dataset" / "golden_dataset.csv"


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, model: str = "mock-1", fail: bool = False, payload: dict[str, Any] | None = None):
        self.model = model
        self.fail = fail
        self.payload = payload or {
            "intent": "PRICE_QUERY",
            "language": "en",
            "answer": "SOBHA Townpark starts at INR 1.8 Crore onwards for 3 BHK.",
            "action": "ANSWER",
        }

    def generate(self, system_prompt, conversation_history, customer_utterance, retrieved_context, extra=None):
        if self.fail:
            raise RuntimeError("simulated API failure")
        result = empty_result(self.name, self.model)
        result.update(self.payload)
        result["raw_response"] = json.dumps(self.payload)
        result["schema_valid"] = True
        result["parsed"] = True
        result["latency_ms"] = 12
        result["usage"] = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        return result


def test_dataset_loading():
    cases = load_dataset(str(DATASET))
    assert len(cases) >= 10
    ids = {case["test_id"] for case in cases}
    assert ids >= {"L001", "L002", "L003", "L004", "L005", "L006", "L007", "L008", "L009", "L010"}
    categories = {case["category"] for case in cases}
    assert "PRICE_QUERY" in categories
    assert "UNKNOWN" in categories
    assert "MULTILINGUAL" in categories
    assert "CONFIRMATION" in categories
    mixed = [case for case in cases if case["test_id"] == "L010"][0]
    assert mixed["language"] == "mixed"
    follow = [case for case in cases if case["test_id"] == "L009"][0]
    assert follow["conversation_history"]


def test_json_parsing():
    raw = '```json\n{"intent": "PRICE_QUERY", "language": "en", "answer": "Hi", "action": "ANSWER"}\n```'
    parsed = extract_json_object(raw)
    assert parsed["intent"] == "PRICE_QUERY"
    assert extract_json_object("not json") is None


def test_schema_validation():
    ok, fields = validate_structured_output(
        {"intent": "PRICE_QUERY", "language": "en", "answer": "ok", "action": "ANSWER"}
    )
    assert ok is True
    assert fields["intent"] == "PRICE_QUERY"
    bad, _ = validate_structured_output({"intent": "NOPE", "language": "en", "answer": "ok", "action": "ANSWER"})
    assert bad is False


def test_fact_matching():
    result = evaluate_facts(["INR 1.8 Crore", "3 BHK"], "Starting price is Rs 1.8 crore for 3 bhk.")
    assert result["missing_facts"] == []
    assert result["fact_accuracy"] == 1.0
    missing = evaluate_facts(["Electronic City"], "It is a nice project.")
    assert missing["missing_facts"] == ["Electronic City"]


def test_intent_comparison():
    assert labels_match("PRICE_QUERY", "price_query") is True
    row = evaluate_intent_action(
        {"expected_intent": "PRICE_QUERY", "expected_action": "ANSWER"},
        {"intent": "PRICE_QUERY", "action": "CLARIFY"},
    )
    assert row["intent_pass"] is True
    assert row["action_pass"] is False


def test_action_comparison():
    row = evaluate_intent_action(
        {"expected_intent": "UNKNOWN", "expected_action": "DECLINE_UNKNOWN"},
        {"intent": "UNKNOWN", "action": "DECLINE_UNKNOWN"},
    )
    assert row["action_pass"] is True


def test_grounding_evaluation():
    context = "Starting price: INR 1.8 Crore onwards. Amenities: clubhouses."
    grounded = evaluate_grounding(context, "Townpark starts at INR 1.8 Crore onwards.", "ANSWER")
    assert grounded["grounded"] is True
    assert grounded["hallucination"] is False
    halluc = evaluate_grounding(
        context,
        "Monthly maintenance is 25 rupees per sqft and possession is March 2021.",
        "DECLINE_UNKNOWN",
    )
    assert halluc["hallucination"] is True
    refuse = evaluate_grounding(
        context,
        "I don't have that detail; our sales team can follow up.",
        "DECLINE_UNKNOWN",
    )
    assert refuse["grounded"] is True
    assert refuse["hallucination"] is False


def test_context_handling():
    case = {
        "conversation_history": [{"role": "assistant", "content": "Would you like to schedule a site visit?"}],
        "customer_utterance": "Yes, tomorrow morning works.",
        "expected_action": "CONFIRM_SLOT",
    }
    assert evaluate_context_handling(case, "Great, I can book 10:00 AM tomorrow for the visit.", "CONFIRM_SLOT") == "PASS"
    follow = {
        "conversation_history": [{"role": "user", "content": "Tell me about the 3 BHK."}],
        "customer_utterance": "What is the starting price for that one?",
        "expected_action": "ANSWER",
    }
    assert evaluate_context_handling(follow, "For the 3 BHK, starting price is INR 1.8 Crore.", "ANSWER") == "PASS"


def test_summary_aggregation():
    rows = [
        {
            "provider": "sarvam",
            "intent_pass": True,
            "action_pass": True,
            "fact_accuracy": 1.0,
            "grounded": True,
            "hallucination": False,
            "context_handling": "PASS",
            "language_pass": True,
            "schema_valid": True,
            "overall_response_quality": 4.0,
            "latency_ms": 100,
            "total_tokens": 50,
            "error": None,
            "category": "PRICE_QUERY",
        },
        {
            "provider": "sarvam",
            "intent_pass": False,
            "action_pass": False,
            "fact_accuracy": 0.0,
            "grounded": False,
            "hallucination": True,
            "context_handling": "FAIL",
            "language_pass": False,
            "schema_valid": False,
            "overall_response_quality": 2.0,
            "latency_ms": 200,
            "total_tokens": 40,
            "error": None,
            "category": "UNKNOWN",
        },
    ]
    summary = summarize(rows, ["sarvam"])[0]
    assert summary["Test Cases"] == 2
    assert summary["Intent Accuracy"] == 0.5
    assert summary["Errors"] == 0


def test_api_failure_does_not_stop_benchmark():
    """One provider exception is recorded; the other provider still evaluates."""
    failing = MockProvider(fail=True)
    ok = MockProvider(fail=False)
    case = load_dataset(str(DATASET))[1]
    rows = []
    for provider in (failing, ok):
        try:
            generation = provider.generate("", [], case["customer_utterance"], case["retrieved_context"], {})
        except Exception as exc:
            generation = empty_result(provider.name, provider.model)
            generation["error"] = f"{type(exc).__name__}: provider failed"
        rows.append(evaluate_one(case, generation))
    assert rows[0]["error"]
    assert rows[1]["error"] is None
    assert rows[1]["intent_pass"] in {True, False}


def test_excel_generation(tmp_path: Path):
    path = tmp_path / "evaluation_summary.xlsx"
    write_workbook(
        path,
        [{"provider": "sarvam", "Test Cases": 1, "Intent Accuracy": 1, "Errors": 0}],
        [{"test_id": "L001", "provider": "sarvam", "language": "en", "latency_ms": 10, "expected_facts": ["clubhouses"]}],
        [{"error_group": "API Failure", "test_id": "L001", "provider": "sarvam"}],
        [{"test_id": "L001", "category": "FAQ", "conversation_history": [], "expected_facts": ["a"]}],
        extra_meta={"run_id": "test", "eval_version": "1.0.0"},
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_dry_run_makes_no_api_calls():
    result = run_benchmark(dry_run=True)
    assert result["dry_run"] is True
    assert result["api_calls"] == 0
    assert result["dataset_cases"] >= 10

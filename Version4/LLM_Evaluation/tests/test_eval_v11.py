from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.aggregation import model_test_matrix, provider_comparisons, summarize_models
from evaluator.excel import write_workbook
from evaluator.runner import (
    ACTIVE_ALIASES,
    EVAL_VERSION,
    check_models,
    evaluate_one,
    requested_aliases,
    run_benchmark,
    select_models,
    load_yaml_config,
)
from models.base import empty_result
from models.http_util import classify_http_error, request_with_retry, should_retry


def test_eval_version():
    assert EVAL_VERSION == "1.1.0"


def test_four_model_configuration():
    config = load_yaml_config()
    selected, skipped = select_models(config, ["all"])
    aliases = [item["name"] for item in selected]
    assert aliases == list(ACTIVE_ALIASES)
    assert len(selected) == 4
    assert all(item["provider"] in {"sarvam", "claude"} for item in selected)


def test_sarvam_two_model_loading():
    config = load_yaml_config()
    selected, _ = select_models(config, None, provider="sarvam")
    assert [item["name"] for item in selected] == ["sarvam_conversational", "sarvam_flagship"]
    assert {item["model"] for item in selected} == {"sarvam-105b-conversations", "sarvam-105b"}


def test_claude_two_model_loading():
    config = load_yaml_config()
    selected, _ = select_models(config, None, provider="claude")
    assert [item["name"] for item in selected] == ["claude_flagship", "claude_sonnet"]
    assert all(item["model"] for item in selected)


def test_qwen_excluded_from_active_benchmark():
    config = load_yaml_config()
    selected, skipped = select_models(config, ["all"])
    assert all(item["name"] != "qwen" for item in selected)
    selected_qwen, skipped_qwen = select_models(config, ["qwen"])
    assert selected_qwen == []
    assert any("OUT OF SCOPE" in note or "disabled" in note for note in skipped_qwen)


def test_missing_claude_model_id():
    config = load_yaml_config()
    spec = dict(config["models"]["claude_sonnet"])
    spec["default_model"] = ""
    with patch.dict("os.environ", {"CLAUDE_SONNET_MODEL": ""}, clear=False):
        from evaluator.runner import resolve_model_runtime
        runtime = resolve_model_runtime("claude_sonnet", spec, config)
        runtime["model"] = ""
        runtime["available"] = bool(runtime["api_key"] and runtime["model"] and runtime["base_url"])
        assert runtime["model"] == "" or not runtime["model"]


def test_claude_model_unavailable_classified():
    assert classify_http_error(404, '{"type":"not_found_error"}') == "MODEL_NOT_FOUND"
    row = evaluate_one(
        {
            "test_id": "L001",
            "category": "PROJECT_FAQ",
            "language": "en",
            "expected_intent": "AMENITIES_QUERY",
            "expected_action": "ANSWER",
            "expected_facts": ["clubhouses"],
            "conversation_stage": "answer_faq",
            "customer_utterance": "hi",
            "acceptable_answer_criteria": "x",
        },
        {
            **empty_result("claude", "missing-model"),
            "error": "HTTP 404",
            "error_type": "MODEL_NOT_FOUND",
            "error_message": "HTTP 404",
        },
    )
    assert row["error_type"] == "MODEL_NOT_FOUND"
    assert row["intent_pass"] is None
    assert row["overall_response_quality"] is None


def test_404_does_not_retry():
    assert should_retry(404, "MODEL_NOT_FOUND", None) is False
    calls = {"n": 0}

    class Resp:
        status_code = 404
        text = '{"type":"not_found_error"}'

        def json(self):
            return {"type": "not_found_error"}

    def fake_request(*args, **kwargs):
        calls["n"] += 1
        return Resp()

    with patch("models.http_util.requests.request", side_effect=fake_request):
        resp, err, msg = request_with_retry("POST", "https://example.test", max_retries=2, timeout=1)
    assert err == "MODEL_NOT_FOUND"
    assert calls["n"] == 1


def test_429_retries():
    calls = {"n": 0}

    class Resp:
        def __init__(self, code):
            self.status_code = code
            self.text = "rate"

        def json(self):
            return {}

    def fake_request(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return Resp(429)
        return Resp(200)

    with patch("models.http_util.requests.request", side_effect=fake_request):
        resp, err, msg = request_with_retry("POST", "https://example.test", max_retries=2, timeout=1)
    assert err == "AVAILABLE"
    assert calls["n"] == 3


def test_timeout_retries():
    import requests as req

    calls = {"n": 0}

    class Resp:
        status_code = 200
        text = "ok"

        def json(self):
            return {}

    def fake_request(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise req.Timeout()
        return Resp()

    with patch("models.http_util.requests.request", side_effect=fake_request):
        resp, err, msg = request_with_retry("GET", "https://example.test", max_retries=2, timeout=1)
    assert err == "AVAILABLE"
    assert calls["n"] == 2


def test_one_model_failure_does_not_zero_quality_of_others():
    case = {
        "test_id": "L002",
        "category": "PRICE_QUERY",
        "language": "en",
        "expected_intent": "PRICE_QUERY",
        "expected_action": "ANSWER",
        "expected_facts": ["INR 1.8 Crore"],
        "conversation_stage": "answer_faq",
        "customer_utterance": "price?",
        "retrieved_context": "Starting price: INR 1.8 Crore onwards.",
        "acceptable_answer_criteria": "state price",
    }
    fail = evaluate_one(case, {**empty_result("claude", "x"), "error": "HTTP 404", "error_type": "MODEL_NOT_FOUND"})
    ok = evaluate_one(
        case,
        {
            **empty_result("sarvam", "sarvam-105b-conversations", model_alias="sarvam_conversational", model_role="conversational"),
            "intent": "PRICE_QUERY",
            "language": "en",
            "answer": "Starting price is INR 1.8 Crore onwards.",
            "action": "ANSWER",
            "schema_valid": True,
        },
    )
    assert fail["intent_pass"] is None
    assert ok["intent_pass"] is True
    assert ok["error_type"] is None


def test_four_model_excel_summary(tmp_path: Path):
    path = tmp_path / "evaluation_summary.xlsx"
    summaries = summarize_models(
        [
            {
                "model_alias": "sarvam_conversational",
                "provider": "sarvam",
                "model_role": "conversational",
                "model_id": "sarvam-105b-conversations",
                "intent_pass": True,
                "action_pass": True,
                "fact_accuracy": 1,
                "grounded": True,
                "hallucination": False,
                "context_handling": "NA",
                "language_pass": True,
                "schema_valid": True,
                "stage_appropriateness_score": 4,
                "overall_response_quality": 4,
                "latency_ms": 10,
                "response_word_count": 12,
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            }
        ],
        list(ACTIVE_ALIASES),
    )
    write_workbook(
        path,
        summaries,
        [{"test_id": "L001", "provider": "sarvam", "model_alias": "sarvam_conversational", "language": "en"}],
        [],
        [{"test_id": "L001", "category": "FAQ", "conversation_history": [], "expected_facts": []}],
        extra_meta={"run_id": "t", "eval_version": "1.1.0"},
        comparison_rows=provider_comparisons(summaries),
        matrix_rows=model_test_matrix([], list(ACTIVE_ALIASES), ["L001"]),
        latency_rows=summaries,
        config_rows=[{"Provider": "sarvam", "Model Alias": "sarvam_conversational"}],
    )
    assert path.exists() and path.stat().st_size > 0


def test_provider_comparison_and_matrix():
    summaries = [
        {"model_alias": "sarvam_conversational", "Intent Accuracy": 0.7, "Action Accuracy": 0.8, "Fact Accuracy": 0.75, "Grounded %": 1, "Hallucination %": 0, "Context Handling %": 1, "Language Handling %": 1, "Avg Response Quality": 4, "Stage Appropriateness": 4, "Avg Latency": 100, "Total Tokens": 10, "Estimated Cost": "NOT_CONFIGURED"},
        {"model_alias": "sarvam_flagship", "Intent Accuracy": 0.8, "Action Accuracy": 0.8, "Fact Accuracy": 0.8, "Grounded %": 1, "Hallucination %": 0, "Context Handling %": 1, "Language Handling %": 1, "Avg Response Quality": 4.2, "Stage Appropriateness": 4, "Avg Latency": 120, "Total Tokens": 12, "Estimated Cost": "NOT_CONFIGURED"},
        {"model_alias": "claude_flagship", "Intent Accuracy": None, "Estimated Cost": "NOT_CONFIGURED"},
        {"model_alias": "claude_sonnet", "Intent Accuracy": None, "Estimated Cost": "NOT_CONFIGURED"},
    ]
    cmp_rows = provider_comparisons(summaries)
    assert any(r["provider"] == "SARVAM" and r["metric"] == "Intent Accuracy" for r in cmp_rows)
    matrix = model_test_matrix(
        [
            {"test_id": "L001", "model_alias": "sarvam_conversational", "intent_pass": True, "action_pass": True, "fact_accuracy": 1, "facts_expected": []},
            {"test_id": "L001", "model_alias": "claude_flagship", "error_type": "MODEL_NOT_FOUND", "error": "HTTP 404"},
        ],
        list(ACTIVE_ALIASES),
        ["L001"],
    )
    assert "PASS" in matrix[0]["Sarvam Conversations"]
    assert matrix[0]["Claude Flagship"] == "API ERROR"
    assert matrix[0]["Claude Sonnet"] == "MODEL UNAVAILABLE"


def test_run_metadata_shape():
    result = run_benchmark(dry_run=True)
    assert result["api_calls"] == 0
    assert result["models_configured"] == 4
    assert result["qwen"] == "OUT OF SCOPE"


def test_dry_run_zero_api_calls_v11():
    result = run_benchmark(dry_run=True)
    assert result["dry_run"] is True
    assert result["api_calls"] == 0
    assert result["potential_benchmark_executions"] == 40
    assert result["sarvam_models"] == 2
    assert result["claude_models"] == 2


def test_cli_model_filtering():
    config = load_yaml_config()
    assert requested_aliases(config, ["all"], None) == list(ACTIVE_ALIASES)
    assert requested_aliases(config, None, "sarvam") == ["sarvam_conversational", "sarvam_flagship"]
    assert requested_aliases(config, ["claude_sonnet"], None) == ["claude_sonnet"]


def test_api_errors_are_not_quality_failures():
    row = evaluate_one(
        {
            "test_id": "L001",
            "category": "X",
            "language": "en",
            "expected_intent": "PRICE_QUERY",
            "expected_action": "ANSWER",
            "expected_facts": ["INR 1.8 Crore"],
            "conversation_stage": "answer_faq",
            "customer_utterance": "price",
            "acceptable_answer_criteria": "x",
        },
        {"provider": "claude", "model": "x", "model_alias": "claude_flagship", "error_type": "MODEL_NOT_FOUND", "error": "HTTP 404"},
    )
    assert row["intent_pass"] is None
    assert row["fact_accuracy"] is None
    assert row["overall_response_quality"] is None
    summaries = summarize_models([row], ["claude_flagship"])
    assert summaries[0]["Intent Accuracy"] is None
    assert summaries[0]["API Errors"] == 1

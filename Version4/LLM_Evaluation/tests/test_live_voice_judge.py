"""Independent judge agent: never score from candidate self-assessment."""

from __future__ import annotations

import json
from pathlib import Path

from live_voice.evaluator import evaluate_run_dir
from live_voice.judge import LiveVoiceJudge, na_criteria, same_model_forbidden
from live_voice.session_store import write_json


def _complete_ok(_judge, _system, user: str):
    assert "untrusted" in user.lower() or "candidate_self_report" in user
    payload = json.loads(user.split("Return the rubric JSON only.\n\n", 1)[1])
    declared = payload["candidate_self_report"][0]["intent"]
    assert declared == "PRICE_QUERY"
    body = {
        "judge_status": "OK",
        "summary": "Location was asked; candidate self-report claimed PRICE_QUERY and is ignored.",
        "criteria": {
            "intent_accuracy": {
                "score": 1,
                "na": False,
                "turn_indices": [2],
                "error_class": "LLM",
                "evidence": "Turn 2 asked location; spoken answer named Electronic City.",
            },
            "action_accuracy": {
                "score": 1,
                "na": False,
                "turn_indices": [3],
                "error_class": "LLM",
                "evidence": "Turn 3 answered.",
            },
            "factual_accuracy": {
                "score": 1,
                "na": False,
                "turn_indices": [3],
                "error_class": "LLM",
                "evidence": "Electronic City is in ground truth.",
            },
            "grounding": {
                "score": 1,
                "na": False,
                "turn_indices": [3],
                "error_class": "LLM",
                "evidence": "Grounded in project facts.",
            },
            "unsupported_claims": {
                "score": 1,
                "na": False,
                "turn_indices": [3],
                "error_class": "LLM",
                "evidence": "No extra claims.",
            },
            "context_retention": {
                "score": None,
                "na": True,
                "turn_indices": [],
                "error_class": "UNOBSERVABLE",
                "evidence": "Only one customer turn.",
            },
            "conversation_stage_accuracy": {
                "score": 1,
                "na": False,
                "turn_indices": [3],
                "error_class": "LLM",
                "evidence": "FAQ answer.",
            },
            "site_visit_completion": {
                "score": None,
                "na": True,
                "turn_indices": [],
                "error_class": "UNOBSERVABLE",
                "evidence": "Visit stage not reached.",
            },
            "response_latency": {
                "score": None,
                "na": True,
                "turn_indices": [],
                "error_class": "UNOBSERVABLE",
                "evidence": "No latency samples.",
            },
            "stt_issues": {
                "score": 1,
                "na": False,
                "turn_indices": [2],
                "error_class": "STT",
                "evidence": "raw_stt matched corrected_text.",
            },
            "conversation_quality": {
                "score": 1,
                "na": False,
                "turn_indices": [3],
                "error_class": "LLM",
                "evidence": "Short spoken answer.",
            },
        },
    }
    return {"text": json.dumps(body), "error": None, "error_type": None}


def test_same_model_forbidden():
    assert same_model_forbidden("claude-sonnet-4-6", "claude-sonnet-4-6") is True
    assert same_model_forbidden("claude-sonnet-4-6", "claude-opus-4-8") is False


def test_judge_does_not_use_candidate_self_report(tmp_path):
    run = tmp_path / "camp" / "run1"
    write_json(
        run / "manifest.json",
        {
            "campaign_id": "camp",
            "evaluation_run_id": "run1",
            "alias": "deepseek_flagship",
            "provider": "deepseek",
            "model_id": "deepseek-flash",
            "call_sid": "CA123",
            "call_status": "completed",
            "duration_seconds": 40,
            "recording_status": "UNAVAILABLE",
        },
    )
    (run / "transcript.jsonl").write_text(
        json.dumps(
            {
                "turn_index": 1,
                "speaker": "agent",
                "text": "Hi, this is Sobha Limited calling about SOBHA Townpark.",
                "campaign_id": "camp",
                "evaluation_run_id": "run1",
                "call_sid": "CA123",
                "alias": "deepseek_flagship",
                "model_id": "deepseek-flash",
            }
        )
        + "\n"
        + json.dumps(
            {
                "turn_index": 2,
                "speaker": "customer",
                "text": "Where exactly is the project located?",
                "raw_stt": "Where exactly is the project located?",
                "corrected_text": "Where exactly is the project located?",
                "outbound_turn_index": 4,
                "campaign_id": "camp",
                "evaluation_run_id": "run1",
                "call_sid": "CA123",
                "alias": "deepseek_flagship",
                "model_id": "deepseek-flash",
            }
        )
        + "\n"
        + json.dumps(
            {
                "turn_index": 3,
                "speaker": "agent",
                "text": "SOBHA Townpark is near Electronic City on Hosur Road in Bengaluru.",
                "candidate_declared_intent": "PRICE_QUERY",
                "candidate_declared_action": "QUALIFY",
                "campaign_id": "camp",
                "evaluation_run_id": "run1",
                "call_sid": "CA123",
                "alias": "deepseek_flagship",
                "model_id": "deepseek-flash",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    judge = LiveVoiceJudge(
        complete_fn=_complete_ok,
        judge_runtime={
            "provider": "claude",
            "model_id": "claude-opus-4-8",
            "api_key": "not-used",
            "base_url": "https://api.anthropic.com/v1/messages",
            "temperature": 0,
            "max_tokens": 200,
            "timeout": 10,
        },
    )
    output = evaluate_run_dir(run, judge=judge)
    assert output["judge"]["judge_status"] == "OK"
    assert output["candidate_self_report"][0]["intent"] == "PRICE_QUERY"
    assert output["judge"]["criteria"]["intent_accuracy"]["score"] == 1
    assert output["judge"]["criteria"]["intent_accuracy"]["score"] != 0
    assert output["offline_benchmark_merged"] is False
    assert "interactive_runs" not in str(run)
    assert (run / "evaluator.json").exists()
    assert not list(Path(__file__).resolve().parents[1].joinpath("output", "interactive_runs").glob("evaluator.json"))


def test_same_model_id_skips_judge_scores():
    judge = LiveVoiceJudge(
        complete_fn=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("judge must not be called")),
        judge_runtime={
            "provider": "claude",
            "model_id": "deepseek-flash",
            "api_key": "x",
            "base_url": "https://example.test",
            "temperature": 0,
            "max_tokens": 10,
            "timeout": 5,
        },
    )
    result = judge.evaluate_packet(
        {
            "candidate_model_id": "deepseek-flash",
            "alias": "deepseek_flagship",
            "campaign_id": "c",
            "evaluation_run_id": "r",
            "call_sid": "CA",
        }
    )
    assert result["judge_status"] == "SAME_MODEL_FORBIDDEN"
    assert result["criteria"]["intent_accuracy"]["score"] is None
    assert result["criteria"]["intent_accuracy"]["na"] is True


def test_unobservable_is_na_not_zero():
    rows = na_criteria("missing latency")
    assert rows["response_latency"]["score"] is None
    assert rows["response_latency"]["na"] is True
    assert rows["response_latency"]["score"] != 0

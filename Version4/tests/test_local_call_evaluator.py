"""Offline checks for the Version4 full-call evaluator."""

from pathlib import Path

from local_call.call_record import CallLog
from local_call.evaluate import evaluate_saved_call
from local_call.judge import judge_call
from local_call.measured import recommended_change, score_measured
from local_call.scorecard import render_scorecard


def _record():
    return {
        "call_id": "sample",
        "caller_model": "sarvam-105b-conversations",
        "turns": [
            {
                "turn_index": 1,
                "customer": None,
                "reply": "Is now a good time for a quick call?",
                "stage": "greeting",
                "stt_ms": None,
                "llm_ms": None,
                "tts_ms": 400,
            },
            {
                "turn_index": 2,
                "customer": "Yes, I can speak now.",
                "reply": "Which configuration are you looking for?",
                "stage": "qualify",
                "stt_ms": 700,
                "llm_ms": None,
                "tts_ms": 500,
            },
            {
                "turn_index": 3,
                "customer": "I'm not sure.",
                "reply": "Which configuration are you looking for?",
                "stage": "qualify",
                "stt_ms": 650,
                "llm_ms": None,
                "tts_ms": 480,
            },
            {
                "turn_index": 4,
                "customer": "1 BHK.",
                "reply": "Our 1 BHK Luxe apartments are 754 sq. ft. Would you like to know more about the layout?",
                "stage": "answer_faq",
                "stt_ms": 800,
                "llm_ms": 8460,
                "tts_ms": 900,
            },
            {
                "turn_index": 5,
                "customer": "Yes, I would like to know.",
                "reply": "Would you like to know more about the layout?",
                "stage": "answer_faq",
                "stt_ms": 900,
                "llm_ms": 9000,
                "tts_ms": 700,
            },
            {
                "turn_index": 6,
                "customer": "No, I'm good.",
                "reply": "Would you like to schedule a site visit to SOBHA Townpark?",
                "stage": "propose_slot",
                "stt_ms": 600,
                "llm_ms": None,
                "tts_ms": 650,
            },
        ],
    }


def test_call_log_saves_timings(tmp_path: Path):
    log = CallLog()
    log.add_reply("Hello?", "greeting", None)
    log.note_tts(120)
    log.note_stt("Yes.", 300)
    log.add_reply("Which configuration?", "qualify", 2000, customer="Yes.")
    saved = log.save(tmp_path)
    text = saved.read_text(encoding="utf-8")
    assert "sarvam-105b-conversations" in text
    assert '"stt_ms": 300' in text
    assert '"llm_ms": 2000' in text
    assert '"tts_ms": 120' in text


def test_measured_scores_cover_the_call():
    measured = score_measured(_record())
    assert measured["latency"]["llm"]["n"] == 2
    assert measured["latency"]["llm"]["avg_ms"] == 8730
    assert measured["stage_path"] == ["greeting", "qualify", "answer_faq", "propose_slot"]
    assert [item["turn_index"] for item in measured["repeated_questions"]] == [3, 5]
    assert measured["premature_site_visits"] == [6]
    assert measured["repeated_configuration_prompts"] == [3]
    fit = {item["turn_index"]: item["fit"] for item in measured["last_answer_fit"]}
    assert fit[2] is True
    assert fit[3] is False
    assert fit[5] is True
    assert "Stop repeating the question at turn 3" in recommended_change(measured)


def test_judge_unavailable_without_a_key(monkeypatch):
    for name in ("LIVE_EVAL_JUDGE_API_KEY", "JUDGE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    judged = judge_call(_record(), {"project": {}})
    assert judged["status"] == "unavailable"
    assert judged["criteria"]["intent_accuracy"]["na"] is True
    assert judged["criteria"]["intent_accuracy"]["score"] is None


def test_judge_reads_claude_and_refuses_the_caller(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_EVAL_JUDGE_MODEL", "sarvam-105b-conversations")
    judged = judge_call(_record(), {"project": {}})
    assert judged["status"] == "unavailable"
    assert "same as the caller" in judged["reason"]


def test_scorecard_keeps_measured_scores_when_the_judge_is_absent(tmp_path: Path, monkeypatch):
    for name in ("LIVE_EVAL_JUDGE_API_KEY", "JUDGE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = evaluate_saved_call(_record(), tmp_path, {"project": {}})
    card = (tmp_path / result["scorecard"]).read_text(encoding="utf-8")
    assert "| Model | 8.7 s |" in card
    assert "Judged section unavailable" in card
    assert "Stop repeating the question at turn 3" in card
    assert "app_v3.py" not in card


def test_scorecard_uses_a_separate_judge_reply(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("LIVE_EVAL_JUDGE_MODEL", raising=False)

    class _Response:
        status_code = 200

        def json(self):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '{"summary":"The layout question was repeated.",'
                            '"recommended_change":"Answer the layout question.",'
                            '"criteria":{"intent_accuracy":{"score":0,"na":false,'
                            '"turn_indices":[5],"evidence":"Yes was not answered."}}}'
                        ),
                    }
                ]
            }

    def poster(url, headers, json, timeout):
        assert url.endswith("/messages")
        assert headers["x-api-key"] == "test-key"
        return _Response()

    result = evaluate_saved_call(_record(), tmp_path, {"sizes": {}}, poster=poster)
    card = (tmp_path / result["scorecard"]).read_text(encoding="utf-8")
    assert "| Judge | claude-opus-5-5 |" in card
    assert "| Intent | Fail | 5 |" in card
    assert "Stop repeating the question at turn 3" in card
    text = render_scorecard(_record(), result["measured"], result["judged"])
    assert "Answer the layout question." not in text

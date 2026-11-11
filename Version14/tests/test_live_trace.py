"""The live strip reports the same turn the call stores."""

import asyncio

import app_v3
from local_call import server
from local_call.usage_note import (
    BULBUL_INR_PER_CHARACTER,
    STT_INR_PER_SECOND,
    TRANSLATE_INR_PER_CHARACTER,
    priced,
)


class _Chat:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {"content": "The sales advisor can confirm that."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        }


def _quiet(monkeypatch):
    server.call_log.saved = True
    monkeypatch.setattr(server, "prepare_upcoming", lambda reply: None)


def _matches(trace_turn, stored):
    for key in ("source", "decision_ms", "llm_ms", "stage", "customer", "reply", "stt_ms", "tts_ms"):
        assert trace_turn[key] == stored[key], key


def _focus_birdsong():
    project = next(
        item for item in app_v3.catalog.PROJECTS if "birdsong" in item["spoken"].lower()
    )
    app_v3.conversation_memory.update(
        {
            "stage": "answer_faq",
            "catalog_flow": True,
            "catalog_done": True,
            "active_project_ids": [project["id"]],
            "customer_part": "north",
            "customer_purpose": "live",
            "last_agent_question": "Would you like to know anything else about the project?",
        }
    )


def test_one_call_shows_greeting_card_and_model(monkeypatch):
    _quiet(monkeypatch)
    app_v3.reset_memory()
    greeting = asyncio.run(server.start_call())
    stored = server.call_log.turns[-1]
    turn = greeting["trace"]["turn"]
    _matches(turn, stored)
    assert turn["source"] == "written greeting"
    assert turn["llm_ms"] is None
    assert isinstance(turn["decision_ms"], int)
    assert turn["prompt_tokens"] is None
    assert greeting["trace"]["averages"]["model_ms"] is None
    assert greeting["trace"]["averages"]["script_ms"] == turn["decision_ms"]
    assert greeting["trace"]["usage"]["rates_checked"] == "3 Nov 2026"
    assert greeting["trace"]["usage"]["language_model"]["requests"] == 0
    assert greeting["trace"]["usage"]["language_model"]["inr_ex_gst"] == 0.0

    _focus_birdsong()

    def no_chat(url, *args, **kwargs):
        if "chat/completions" in str(url):
            raise AssertionError("the price card called the model")
        raise AssertionError(url)

    monkeypatch.setattr(app_v3.requests, "post", no_chat)
    price = asyncio.run(server.take_turn(server.Turn(text="What is the price?")))
    stored = server.call_log.turns[-1]
    turn = price["trace"]["turn"]
    _matches(turn, stored)
    assert turn["source"] == "project card, price"
    assert turn["llm_ms"] is None
    assert isinstance(turn["decision_ms"], int)
    assert "1.2" in price["reply"]
    assert "crore" in price["reply"].lower()
    assert price["trace"]["averages"]["model_ms"] is None

    monkeypatch.setattr(app_v3.requests, "post", lambda url, *args, **kwargs: _Chat())
    asked = asyncio.run(server.take_turn(server.Turn(text="What is the vaastu of these homes?")))
    stored = server.call_log.turns[-1]
    turn = asked["trace"]["turn"]
    _matches(turn, stored)
    assert turn["source"] == "language model, generate_response"
    assert isinstance(turn["llm_ms"], int)
    assert turn["decision_ms"] >= turn["llm_ms"]
    assert turn["prompt_tokens"] == 30
    assert turn["completion_tokens"] == 8
    assert turn["cached_tokens"] == 2
    assert asked["trace"]["usage"]["language_model"]["prompt_tokens"] == 30
    assert asked["trace"]["usage"]["language_model"]["inr_ex_gst"] is not None
    assert asked["trace"]["averages"]["model_ms"] == turn["decision_ms"]
    script = [
        item["decision_ms"]
        for item in server.call_log.turns
        if not item.get("model_calls") and item.get("decision_ms") is not None
    ]
    assert asked["trace"]["averages"]["script_ms"] == int(round(sum(script) / len(script)))


def test_a_resumed_question_stays_on_the_script(monkeypatch):
    _quiet(monkeypatch)
    app_v3.reset_memory()
    asyncio.run(server.start_call())
    question = app_v3.catalog.AREA_QUESTION
    app_v3.conversation_memory["question_after_hold"] = question
    resumed = server.resume_after_hold()
    turn = resumed["trace"]["turn"]
    _matches(turn, server.call_log.turns[-1])
    assert turn["llm_ms"] is None
    assert isinstance(turn["decision_ms"], int)
    assert "AREA_QUESTION" in turn["source"]
    assert question in resumed["reply"]


def test_a_script_turn_prices_voice_without_model_tokens(monkeypatch):
    _quiet(monkeypatch)
    app_v3.reset_memory()
    greeting = asyncio.run(server.start_call())
    turn = greeting["trace"]["turn"]
    assert turn["prompt_tokens"] is None
    assert turn["speech_to_text"]["seconds"] is None
    assert turn["speech_to_text"]["inr_ex_gst"] is None
    assert turn["speech_to_text"]["inr_with_gst"] is None
    assert turn["bulbul"] == {"characters": 0, **priced(0, BULBUL_INR_PER_CHARACTER)}
    assert turn["translation"] == {"characters": 0, **priced(0, TRANSLATE_INR_PER_CHARACTER)}
    assert turn["neerja"] == {"characters": 0}
    assert "inr_ex_gst" not in turn["neerja"]

    _focus_birdsong()

    def no_chat(url, *args, **kwargs):
        if "chat/completions" in str(url):
            raise AssertionError("the price card called the model")
        raise AssertionError(url)

    monkeypatch.setattr(app_v3.requests, "post", no_chat)
    server.call_log.note_audio(2.0)
    price = asyncio.run(server.take_turn(server.Turn(text="What is the price?")))
    turn = price["trace"]["turn"]
    stored = server.call_log.turns[-1]
    assert turn["prompt_tokens"] is None
    assert turn["source"] == "project card, price"
    assert stored["stt_seconds"] == 2.0
    assert turn["speech_to_text"]["seconds"] == 2.0
    assert turn["speech_to_text"]["inr_ex_gst"] == priced(2.0, STT_INR_PER_SECOND)["inr_ex_gst"]
    assert turn["speech_to_text"]["inr_with_gst"] == priced(2.0, STT_INR_PER_SECOND)["inr_with_gst"]
    assert price["trace"]["previous"]["source"] == "written greeting"
    assert price["trace"]["previous"]["speech_to_text"]["seconds"] is None

    server.call_log.note_bulbul(100)
    server.call_log.note_translate(50)
    server.call_log.note_neerja(10)
    monkeypatch.setattr(
        server,
        "_save_and_score",
        lambda: {"saved": True, "call_id": "x", "scorecard": "ok"},
    )
    ended = server.end_call()
    latest = ended["trace"]["turn"]
    assert latest["prompt_tokens"] is None
    assert latest["bulbul"]["characters"] == stored["bulbul_characters"] == 100
    assert latest["bulbul"]["inr_ex_gst"] == priced(100, BULBUL_INR_PER_CHARACTER)["inr_ex_gst"]
    assert latest["translation"]["characters"] == stored["translation_characters"] == 50
    assert latest["translation"]["inr_ex_gst"] == priced(50, TRANSLATE_INR_PER_CHARACTER)["inr_ex_gst"]
    assert latest["neerja"] == {"characters": 10}
    assert ended["trace"]["previous"]["source"] == "written greeting"

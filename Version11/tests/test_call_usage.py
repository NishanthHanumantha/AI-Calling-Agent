"""Measured usage on one local call. The spoken reply is not changed."""

import asyncio
import io
import wave

import app_v3
from local_call import server
from local_call.usage_note import usage_from_body, wav_seconds


def _wav(seconds=1.0, rate=8000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class _Response:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _run_chat(body):
    def post(url, *args, **kwargs):
        return _Response(body)

    app_v3.requests.post = post

    async def action():
        return app_v3.requests.post("https://api.sarvam.ai/v1/chat/completions")

    _response, elapsed, calls, _decision_ms = asyncio.run(server._timed_chat(action))
    return _response, elapsed, calls


def test_a_usage_block_is_stored_on_the_turn():
    server.call_log.begin()
    from local_call.usage_note import tag_model

    tag_model("generate_response")
    _response, elapsed, calls = _run_chat(
        {
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 4},
            }
        }
    )
    server.call_log.add_model_calls(calls)
    server.call_log.add_reply("The detail is not confirmed.", "answer_faq", elapsed, customer="When is possession?")
    turn = server.call_log.turns[-1]
    assert turn["model_calls"] == [
        {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "cached_tokens": 4,
            "function": "generate_response",
        }
    ]
    assert turn["source"] == "language model, generate_response"
    summary = server.call_log.usage()["language_model"]
    assert summary["prompt_tokens"] == 20
    assert summary["completion_tokens"] == 5
    assert summary["cached_tokens"] == 4
    assert summary["inr_ex_gst"] is not None


def test_a_missing_usage_block_stays_blank():
    server.call_log.begin()
    from local_call.usage_note import tag_model

    tag_model("unmatched_model_answer")
    _response, elapsed, calls = _run_chat({"choices": [{"message": {"content": "Hello"}}]})
    server.call_log.add_model_calls(calls)
    server.call_log.add_reply("Hello", "answer_faq", elapsed, customer="Something new")
    recorded = server.call_log.turns[-1]["model_calls"][0]
    assert recorded["prompt_tokens"] is None
    assert recorded["completion_tokens"] is None
    assert recorded["cached_tokens"] is None
    assert recorded["function"] == "unmatched_model_answer"
    assert server.call_log.usage()["language_model"]["inr_ex_gst"] is None
    assert usage_from_body({})["prompt_tokens"] is None


def test_wav_length_is_the_uploaded_audio():
    audio = _wav(1.5, rate=8000)
    assert wav_seconds(audio) == 1.5
    server.call_log.begin()
    server.call_log.note_audio(wav_seconds(audio))
    server.call_log.add_reply("Which part of Bengaluru are you considering — North, East, or South?", "qualify", None, customer="North")
    assert server.call_log.turns[-1]["stt_seconds"] == 1.5
    assert server.call_log.usage()["speech_to_text"]["seconds"] == 1.5


def test_one_translate_post_records_its_characters(monkeypatch):
    class _Translated:
        status_code = 200

        def json(self):
            return {"translated_text": "namaste"}

    def post(url, **kwargs):
        return _Translated()

    monkeypatch.setattr(server.requests, "post", post)
    server.call_log.begin()
    server.call_log.add_reply("Hi.", "greeting", None)
    text = "Thanks for taking the call."
    assert server._request_translation(text, "hi-IN") == "namaste"
    assert server.call_log.turns[-1]["translation_characters"] == len(text)
    priced = server.call_log.usage()["translation"]
    assert priced["characters"] == len(text)
    assert priced["inr_ex_gst"] == round(len(text) * 2 / 1000, 4)


def test_one_bulbul_text_records_its_characters(monkeypatch):
    class _Spoken:
        status_code = 200

        def close(self):
            return None

    def post(url, **kwargs):
        return _Spoken()

    monkeypatch.setattr(server.requests, "post", post)
    previous = app_v3.SARVAM_API_KEY
    app_v3.SARVAM_API_KEY = previous or "test-key"
    server.call_log.begin()
    server.call_log.add_reply("Hello there.", "greeting", None)
    try:
        server.open_bulbul_stream("Hello there.")
    finally:
        app_v3.SARVAM_API_KEY = previous
    assert server.call_log.turns[-1]["bulbul_characters"] == len("Hello there.")
    assert server.call_log.usage()["bulbul"]["inr_ex_gst"] == round(len("Hello there.") * 1 / 1000, 4)


def test_a_price_question_is_labelled_from_the_project_card():
    app_v3.reset_memory()
    project = next(item for item in app_v3.catalog.PROJECTS if item["id"] != "sobha-townpark")
    app_v3.conversation_memory["active_project_ids"] = [project["id"]]
    body = app_v3.reply_from_active_cards("What is the price?")
    assert body
    assert app_v3.conversation_memory["reply_card"] == "price"
    server.call_log.begin()
    server.call_log.add_reply(body, "answer_faq", None, customer="What is the price?", card="price")
    assert server.call_log.turns[-1]["source"] == "project card, price"

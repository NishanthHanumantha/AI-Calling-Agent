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
    assert measured["premature_site_visits"] == []
    assert measured["repeated_configuration_prompts"] == [3]
    fit = {item["turn_index"]: item["fit"] for item in measured["last_answer_fit"]}
    assert fit[2] is True
    assert fit[3] is True
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


def _open_configuration_call():
    return {
        "call_id": "20261014T091409Z",
        "caller_model": "sarvam-105b-conversations",
        "turns": [
            {
                "turn_index": 1,
                "customer": None,
                "reply": (
                    "Hi, this is Sobha Limited calling. I'm getting in touch about "
                    "SOBHA Townpark - a New York-inspired luxury community near "
                    "Electronic City on Hosur Road in Bengaluru. "
                    "Is now a good time for a quick call?"
                ),
                "stage": "greeting",
                "stt_ms": None,
                "llm_ms": None,
                "tts_ms": 1100,
            },
            {
                "turn_index": 2,
                "customer": "Yes, I can speak now.",
                "reply": (
                    "Thanks for taking the call. This is regarding SOBHA Townpark. "
                    "We have one, two, three, and four BHK homes. "
                    "Which configuration are you looking for?"
                ),
                "stage": "qualify",
                "stt_ms": 900,
                "llm_ms": None,
                "tts_ms": 900,
            },
            {
                "turn_index": 3,
                "customer": "I haven't decided on it.",
                "reply": (
                    "That's alright, we can leave the configuration open. "
                    "Are you looking at this to live in, or as an investment?"
                ),
                "stage": "answer_faq",
                "stt_ms": 900,
                "llm_ms": None,
                "tts_ms": 1100,
            },
            {
                "turn_index": 4,
                "customer": "To live in.",
                "reply": (
                    "Living in is the perfect fit for SOBHA Townpark. "
                    "Would you like to know about the location?"
                ),
                "stage": "answer_faq",
                "stt_ms": 800,
                "llm_ms": 811,
                "tts_ms": 1000,
            },
            {
                "turn_index": 5,
                "customer": "No, I you already told about that.",
                "reply": (
                    "Understood. Since you are already aware of that, "
                    "shall we look at the available configurations?"
                ),
                "stage": "answer_faq",
                "stt_ms": 900,
                "llm_ms": 531,
                "tts_ms": 900,
            },
            {
                "turn_index": 6,
                "customer": "No, I'm not interested in that.",
                "reply": "No problem. Thank you for your time. Have a good day.",
                "stage": "closed",
                "stt_ms": 900,
                "llm_ms": None,
                "tts_ms": 800,
            },
        ],
    }


def test_open_answer_is_not_blamed_and_later_repeats_are():
    measured = score_measured(_open_configuration_call())
    fit = {item["turn_index"]: item["fit"] for item in measured["last_answer_fit"]}
    assert fit[3] is True
    assert fit[4] is True
    assert fit[6] is True
    assert measured["repeated_configuration_prompts"] == [5]
    repeated = measured["repeated_questions"]
    assert any(item["turn_index"] == 4 and item["first_turn_index"] == 1 for item in repeated)
    assert any(item["turn_index"] == 5 for item in repeated)
    assert all(item["turn_index"] != 3 for item in repeated)
    change = recommended_change(measured)
    assert change is not None
    assert "did not answer" not in change
    assert "turn 3" not in change


def test_for_living_fits_the_purpose_question():
    measured = score_measured({
        "turns": [
            {
                "turn_index": 4,
                "customer": "I'm fine with both.",
                "reply": "Are you looking at this to live in, or as an investment?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 5,
                "customer": "It's for living.",
                "reply": "Got it. You're looking at this for your own use.",
                "stage": "answer_faq",
            },
        ],
    })
    fit = {item["turn_index"]: item["fit"] for item in measured["last_answer_fit"]}
    assert fit[5] is True


def test_decide_later_fits_the_size_question():
    measured = score_measured({
        "turns": [
            {
                "turn_index": 3,
                "customer": "3 BHK.",
                "reply": "Which one interests you?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 4,
                "customer": "I will decide this later after seeing.",
                "reply": (
                    "That's alright, we can leave the size open. "
                    "Are you looking at this to live in, or as an investment?"
                ),
                "stage": "answer_faq",
            },
        ],
    })
    fit = {item["turn_index"]: item["fit"] for item in measured["last_answer_fit"]}
    assert fit[4] is True


def test_fine_with_both_fits_the_size_question():
    measured = score_measured({
        "turns": [
            {
                "turn_index": 3,
                "customer": "Maybe 3 BHK.",
                "reply": "Which of these two ranges interests you more?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 4,
                "customer": "I'm fine with both.",
                "reply": "That's alright. Are you looking at this to live in, or as an investment?",
                "stage": "answer_faq",
            },
        ],
    })
    fit = {item["turn_index"]: item["fit"] for item in measured["last_answer_fit"]}
    assert fit[4] is True


def test_judge_prompt_allows_an_open_answer():
    from local_call.judge import SYSTEM_PROMPT

    assert "Leaving configuration, size, or budget open is correct" in SYSTEM_PROMPT
    assert "stage answer_faq" in SYSTEM_PROMPT
    assert 'Not interested in that' in SYSTEM_PROMPT
    assert "Leave response time out of your scores." in SYSTEM_PROMPT
    assert "same topic asked again" in SYSTEM_PROMPT
    assert "one site-visit offer" in SYSTEM_PROMPT
    assert "yes to highlights or documents is not a yes to a visit" in SYSTEM_PROMPT
    assert "I don't need any details" in SYSTEM_PROMPT
    assert "A yes to a visit, a request to book one, a named day, or a clock time" in SYSTEM_PROMPT
    assert "Asking which day or listing those times is the wrong next step" in SYSTEM_PROMPT


def test_a_stated_menu_is_a_repeat_and_a_no_fits():
    record = {
        "call_id": "menu",
        "caller_model": "sarvam-105b-conversations",
        "turns": [
            {
                "turn_index": 2,
                "customer": "Yes.",
                "reply": "Which configuration are you looking for?",
                "stage": "qualify",
            },
            {
                "turn_index": 4,
                "customer": "As an investment.",
                "reply": "SOBHA Townpark offers 1 to 4 BHK apartments. Would you like to hear about the kind of home?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 7,
                "customer": "You already asked that.",
                "reply": "Would you like to hear about the amenities?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 12,
                "customer": "No, I am not.",
                "reply": "We have one to four bedroom apartments available.",
                "stage": "answer_faq",
            },
        ],
    }
    measured = score_measured(record)
    assert measured["repeated_configuration_prompts"] == [4, 12]
    fit = {item["turn_index"]: item["fit"] for item in measured["last_answer_fit"]}
    assert fit[7] is True
    assert fit[12] is True


def _wav(loud_seconds, quiet_seconds=0.0, rate=16000):
    import io
    import math
    import struct
    import wave

    samples = []
    for index in range(int(rate * loud_seconds)):
        samples.append(int(0.3 * 32767 * math.sin(2 * math.pi * 440 * index / rate)))
    samples.extend([0] * int(rate * quiet_seconds))
    raw = struct.pack("<" + "h" * len(samples), *samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(raw)
    return buffer.getvalue()


def test_short_noise_is_not_sent_to_stt(monkeypatch):
    import asyncio

    from local_call.server import clip_has_speech, listen

    short = _wav(0.15, 0.6)
    longer = _wav(0.6, 0.4)
    assert clip_has_speech(short) is False
    assert clip_has_speech(longer) is True

    def fail_post(*args, **kwargs):
        raise AssertionError("Sarvam was called")

    monkeypatch.setattr("local_call.server.requests.post", fail_post)

    class _Upload:
        async def read(self):
            return short

    assert asyncio.run(listen(_Upload())) == {"text": ""}


def test_a_lone_keyterm_is_not_a_turn(monkeypatch):
    import asyncio

    from local_call.server import listen, transcript_is_only_keyterms

    assert transcript_is_only_keyterms("BHK.") is True
    assert transcript_is_only_keyterms("SOBHA Townpark.") is True
    assert transcript_is_only_keyterms("2 BHK") is False
    assert transcript_is_only_keyterms("3 BHK in Whitefield") is False
    assert transcript_is_only_keyterms("Yes, I can speak now.") is False

    spoken = {"text": "BHK."}

    class _Response:
        status_code = 200

        def json(self):
            return {"transcript": spoken["text"]}

    monkeypatch.setattr("local_call.server.requests.post", lambda *args, **kwargs: _Response())
    monkeypatch.setattr("local_call.server.app_v3.SARVAM_API_KEY", "test")
    audio = _wav(0.6, 0.4)

    class _Upload:
        async def read(self):
            return audio

    assert asyncio.run(listen(_Upload())) == {"text": ""}
    spoken["text"] = "2 BHK."
    assert asyncio.run(listen(_Upload())) == {"text": "2 BHK."}


def test_a_missed_goodbye_outranks_a_repeated_question():
    record = {
        "call_id": "ending",
        "caller_model": "sarvam-105b-conversations",
        "turns": [
            {
                "turn_index": 3,
                "customer": "I haven't finalized it.",
                "reply": "Which configuration interests you?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 5,
                "customer": "No, I am good.",
                "reply": "Would you like to schedule a site visit to SOBHA Townpark?",
                "stage": "propose_slot",
            },
            {
                "turn_index": 7,
                "customer": "I will let you know later.",
                "reply": "Would you like me to share the highlights of the project?",
                "stage": "propose_slot",
            },
            {
                "turn_index": 12,
                "customer": "No, I'm not interested.",
                "reply": "Shall I share the details with you?",
                "stage": "visit_day",
            },
        ],
    }
    measured = score_measured(record)
    assert 5 not in measured["missed_closes"]
    assert measured["missed_closes"] == [7, 12]
    change = recommended_change(measured)
    assert change.startswith("Close the call at turn 7.")
    assert "Stop repeating" not in change


def test_a_missed_share_outranks_a_repeated_what_else():
    record = {
        "call_id": "share",
        "caller_model": "sarvam-105b-conversations",
        "turns": [
            {
                "turn_index": 7,
                "customer": "Yeah, BPK.",
                "reply": "SOBHA Townpark offers clubhouses and pools. What else can I help you with?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 8,
                "customer": "When will the project be completed?",
                "reply": "I do not have the confirmed completion date right now. Would you like me to share the other available details with you?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 9,
                "customer": "Yeah, sure you can share me.",
                "reply": "Sure. Would you like to know about the open space allocation?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 10,
                "customer": "No, I'm not interested in that.",
                "reply": "That's alright. What else can I help you with?",
                "stage": "answer_faq",
            },
        ],
    }
    measured = score_measured(record)
    assert measured["missed_shares"] == [9]
    change = recommended_change(measured)
    assert change.startswith("Share the offered details at turn 9.")
    assert "Stop repeating" not in change


def test_a_no_to_amenities_is_not_a_missed_close():
    record = {
        "call_id": "amenities",
        "caller_model": "sarvam-105b-conversations",
        "turns": [
            {
                "turn_index": 6,
                "customer": "Yeah, you can let me know.",
                "reply": "Flexible payment plans and home loan assistance are available. Would you like to hear about the amenities?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 7,
                "customer": "No, I am not interested to know about them.",
                "reply": "That's alright. Would you like to know anything else about the project?",
                "stage": "answer_faq",
            },
            {
                "turn_index": 8,
                "customer": "No.",
                "reply": "Would you like to schedule a site visit to SOBHA Townpark?",
                "stage": "propose_slot",
            },
        ],
    }
    measured = score_measured(record)
    assert measured["missed_closes"] == []
    assert measured["premature_site_visits"] == []
    change = recommended_change(measured)
    assert change is None or not change.startswith("Close the call")


def test_speaking_style_follows_the_clause():
    from local_call.server import BULBUL_SPEAKER, speaking_style

    assert BULBUL_SPEAKER == "ishita"
    lines = (
        "Hi, this is Sobha Limited calling.",
        "Thanks for taking the call.",
        "2 BHK Luxe is 1240 to 1248 square feet.",
        "Would you like to know the starting price?",
        "Would you like to schedule a site visit?",
        "I will have our sales advisor get in touch with you.",
        "Have a good day.",
        "Are you looking at this to live in, or as an investment?",
    )
    for line in lines:
        assert speaking_style(line) == (1.0, 0.6)


def test_spoken_size_says_square_feet():
    import app_v3
    from local_call.server import for_speech

    spoken = for_speech("2 BHK Luxe is 1240 to 1248 sq. ft.")
    assert spoken == "2 BHK Luxe is 1240 to 1248 square feet"
    assert for_speech("Which configuration are you looking for?") == (
        "Which configuration are you looking for?"
    )
    assert "sq. ft." in app_v3.UNIT_SIZES["2 BHK"]
    assert for_speech("Would you like to schedule a site visit to SOBHA Townpark?") == (
        "Would you like to schedule a site visit to Sobha Townpark?"
    )
    assert "SOBHA" in app_v3.visit_schedule_question()

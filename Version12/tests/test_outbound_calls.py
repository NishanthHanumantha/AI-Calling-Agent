"""Outbound calls stay on the existing script, with fakes instead of Exotel."""

import asyncio
import base64
import copy
import json
from datetime import datetime

import app_v3
from local_call import call_record, dialer, phone
from local_call.call_record import CallLog
from local_call.dialer import IST
from local_call.phone import PhoneSession, samples_to_wav, to_pcm16k
from local_call.server import call_log, clause_bytes, local_app, payload_from

FIRST = "+910000000001"
SECOND = "+910000000002"
DAY = datetime(2026, 10, 27, 10, 0, tzinfo=IST)
NIGHT = datetime(2026, 10, 27, 8, 30, tzinfo=IST)


class _Accepted:
    status_code = 200

    def __init__(self, sid="CA-fake"):
        self.sid = sid

    def json(self):
        return {"Call": {"Sid": self.sid}}


def _tone(rate, seconds, amplitude):
    count = int(rate * seconds)
    return struct_samples(amplitude, count)


def struct_samples(amplitude, count):
    return [amplitude] * count


def _pcm(samples):
    import struct

    return struct.pack("<" + "h" * len(samples), *samples)


def _media(samples):
    return {
        "event": "media",
        "stream_sid": "ST1",
        "media": {"payload": base64.b64encode(_pcm(samples)).decode("ascii")},
    }


def test_two_call_ids_keep_separate_stages():
    before = copy.deepcopy(app_v3._default_memory)

    async def run():
        token_a = app_v3.bind_call("CA-a")
        log_a = call_record.bind_log("CA-a")
        try:
            greeting = payload_from(await app_v3.voice())
            call_log.add_reply(greeting["reply"], greeting["stage"], None)
            reply = payload_from(
                await app_v3.handle_speech(
                    phone._Request("SOBHA Townpark, please speak in Hindi"),
                    SpeechResult="SOBHA Townpark, please speak in Hindi",
                )
            )
            call_log.add_reply(
                reply["reply"],
                reply["stage"],
                None,
                customer="SOBHA Townpark, please speak in Hindi",
            )
            stage_a = app_v3.conversation_memory["stage"]
            language_a = app_v3.conversation_memory["customer_language"]
            project_a = list(app_v3.conversation_memory["active_project_ids"])
        finally:
            app_v3.release_call(token_a)
            call_record.release_log(log_a)

        token_b = app_v3.bind_call("CA-b")
        log_b = call_record.bind_log("CA-b")
        try:
            greeting = payload_from(await app_v3.voice())
            call_log.add_reply(greeting["reply"], greeting["stage"], None)
            reply = payload_from(
                await app_v3.handle_speech(
                    phone._Request("I'm not interested"),
                    SpeechResult="I'm not interested",
                )
            )
            call_log.add_reply(reply["reply"], reply["stage"], None, customer="I'm not interested")
            stage_b = app_v3.conversation_memory["stage"]
            language_b = app_v3.conversation_memory["customer_language"]
            project_b = list(app_v3.conversation_memory["active_project_ids"])
        finally:
            app_v3.release_call(token_b)
            call_record.release_log(log_b)
        return stage_a, language_a, project_a, stage_b, language_b, project_b

    stage_a, language_a, project_a, stage_b, language_b, project_b = asyncio.run(run())
    assert stage_a == "answer_faq"
    assert language_a == "hi-IN"
    assert project_a == ["sobha-townpark"]
    assert stage_b == "closed"
    assert language_b == "en-IN"
    assert project_b == []
    again = app_v3.bind_call("CA-a")
    try:
        assert app_v3.conversation_memory["stage"] == "answer_faq"
        assert app_v3.conversation_memory["customer_language"] == "hi-IN"
        assert app_v3.conversation_memory["active_project_ids"] == ["sobha-townpark"]
    finally:
        app_v3.release_call(again)
    assert app_v3._default_memory == before
    assert call_record.log_for("CA-a").call_id == "CA-a"
    assert call_record.log_for("CA-b").call_id == "CA-b"
    assert call_record.log_for("CA-a").turns[1]["stage"] == "answer_faq"
    assert call_record.log_for("CA-b").turns[1]["stage"] == "closed"


def test_a_missed_clause_becomes_16k_pcm_after_one_bulbul(monkeypatch):
    app_v3.reset_memory()
    calls = []

    def opening(text, language_code="en-IN"):
        calls.append((text, language_code))
        return None

    async def neerja(text):
        return samples_to_wav(_tone(22050, 0.1, 1000), 22050)

    monkeypatch.setattr("local_call.server.open_bulbul_stream", opening)
    monkeypatch.setattr("local_call.server.neerja_audio", neerja)

    async def run():
        audio, media = await clause_bytes("Which part of Bengaluru?")
        pcm = to_pcm16k(audio, media)
        sent = []

        async def send(raw):
            sent.append(json.loads(raw))

        session = PhoneSession(send=send)
        session.stream_sid = "ST1"
        await session.speak("Which part of Bengaluru?")
        return pcm, sent

    pcm, sent = asyncio.run(run())
    assert len(calls) == 2
    assert all(language == "en-IN" for _text, language in calls)
    assert len(pcm) // 2 == 1600
    payload = base64.b64decode(sent[0]["media"]["payload"])
    assert sent[0]["event"] == "media"
    assert len(payload) % 320 == 0
    assert 3200 <= len(payload) <= 100000
    assert payload[: len(pcm)] == pcm


def test_customer_audio_is_sent_to_saaras(monkeypatch):
    monkeypatch.setattr(app_v3, "SARVAM_API_KEY", "test-key")
    posted = []

    def opening(text, language_code="en-IN"):
        return None

    async def neerja(text):
        return samples_to_wav(_tone(16000, 0.05, 1000), 16000)

    class _Response:
        status_code = 200

        def json(self):
            return {"transcript": "Yes", "language_code": "en-IN"}

    def post(url, *args, **kwargs):
        posted.append((url, kwargs.get("files"), kwargs.get("data")))
        return _Response()

    monkeypatch.setattr("local_call.server.open_bulbul_stream", opening)
    monkeypatch.setattr("local_call.server.neerja_audio", neerja)
    monkeypatch.setattr("local_call.server.requests.post", post)

    async def run():
        sent = []

        async def send(raw):
            sent.append(json.loads(raw))

        session = PhoneSession(send=send, scorer=lambda log: None)
        try:
            await session.on_message(
                {
                    "event": "start",
                    "stream_sid": "ST1",
                    "start": {
                        "call_sid": "CA-media",
                        "stream_sid": "ST1",
                        "media_format": {"encoding": "audio/x-raw", "sample_rate": 16000},
                    },
                }
            )
            await session.idle()
            await session.on_message(_media(_tone(16000, 0.4, 0)))
            utterance = _tone(16000, 0.5, 8000) + _tone(16000, 0.7, 0)
            await session.on_message(_media(utterance))
            await session.idle()
            return app_v3.conversation_memory["stage"]
        finally:
            await session.finish()

    stage = asyncio.run(run())
    assert stage == "qualify"
    assert posted
    url, files, data = posted[0]
    assert url.endswith("/speech-to-text")
    name, audio, content_type = files["file"]
    assert name == "speech.wav"
    assert content_type == "audio/wav"
    assert audio[:4] == b"RIFF"
    assert data["model"] == "saaras:v4"
    again = app_v3.bind_call("CA-media")
    try:
        assert app_v3.conversation_memory["stage"] == "qualify"
    finally:
        app_v3.release_call(again)


def test_talking_over_ishita_sends_clear():
    async def run():
        sent = []

        async def send(raw):
            sent.append(json.loads(raw))

        session = PhoneSession(send=send)
        session.stream_sid = "ST1"
        session.playing = True
        await session.on_message(_media(_tone(16000, 1.0, 12000)))
        return sent

    sent = asyncio.run(run())
    assert sent == [{"event": "clear", "stream_sid": "ST1"}]


def test_dialer_refuses_a_second_open_call_and_a_closed_call(tmp_path, monkeypatch):
    for name in ("LIVE_EVAL_JUDGE_API_KEY", "JUDGE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def poster(url, data, auth, timeout):
        calls.append((url, data, auth, timeout))
        return _Accepted("CA-1")

    campaign = dialer.Dialer([FIRST, SECOND])
    auth = ("key", "token")
    first = campaign.place(
        FIRST,
        DAY,
        poster,
        account_sid="account",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    second = campaign.place(
        SECOND,
        DAY,
        poster,
        account_sid="account",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert first["placed"] is True
    assert first["call_id"] == "CA-1"
    assert second == {"placed": False, "reason": "a call is already open"}
    assert len(calls) == 1
    url, form, used_auth, timeout = calls[0]
    assert url == "https://api.in.exotel.com/v1/Accounts/account/Calls/connect"
    assert form["From"] == FIRST
    assert form["StreamType"] == "bidirectional"
    assert form["StreamUrl"].endswith("/exotel/media?sample-rate=16000")
    assert used_auth == auth
    assert timeout == 30

    campaign.note_closed(FIRST)
    closed = campaign.place(
        FIRST,
        DAY,
        poster,
        account_sid="account",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert closed["reason"] == "that call already closed"
    assert len(calls) == 1

    missed = campaign.place(
        SECOND,
        DAY,
        poster,
        account_sid="account",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert missed["placed"] is True
    campaign.note_no_answer(SECOND)
    retry = campaign.place(
        SECOND,
        DAY,
        poster,
        account_sid="account",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert retry["placed"] is True
    campaign.note_no_answer(SECOND)
    again = campaign.place(
        SECOND,
        DAY,
        poster,
        account_sid="account",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert again["reason"] == "no-answer was already tried again"

    dark = campaign.place(
        SECOND,
        NIGHT,
        poster,
        account_sid="account",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert dark["reason"] == "outside daytime hours"
    blocked = campaign.place(
        FIRST,
        DAY,
        None,
        account_sid="account",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert blocked["reason"] == "that call already closed"

    listing = tmp_path / "customers.txt"
    listing.write_text(f"# keep\n{FIRST}\n\n{SECOND}\n", encoding="utf-8")
    assert dialer.load_numbers(listing) == [FIRST, SECOND]
    try:
        dialer.load_numbers(__file__)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("a list inside the repository was accepted")

    log = CallLog(call_id="CA-1")
    log.add_reply("Hi, this is Sobha Limited calling.", "greeting", None)
    log.save(tmp_path)
    result = campaign.write_scorecard(log.as_dict(), tmp_path, {"project": {}})
    assert (tmp_path / result["scorecard"]).is_file()
    assert (tmp_path / "CA-1.json").is_file()


def test_the_phone_socket_is_on_the_local_page():
    paths = [getattr(route, "path", "") for route in local_app.routes]
    assert "/" in paths
    assert "/exotel/media" in paths
    assert dialer.configured_stream_url("wss://example.test").endswith(
        "/exotel/media?sample-rate=16000"
    )

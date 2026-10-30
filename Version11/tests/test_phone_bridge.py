"""One Exotel socket on the Version 11 page. Fakes stand in for Saaras, Bulbul, and Exotel."""

import asyncio
import base64
import json
from datetime import datetime

import app_v3
from local_call import dialer, phone
from local_call.dialer import IST
from local_call.phone import PhoneSession, samples_to_wav, to_pcm16k
from local_call.server import call_log, clause_bytes, local_app

FIRST = "+910000000001"
DAY = datetime(2026, 10, 30, 11, 0, tzinfo=IST)
NIGHT = datetime(2026, 10, 30, 8, 30, tzinfo=IST)


class _Accepted:
    status_code = 200

    def json(self):
        return {"Call": {"Sid": "CA-fake"}}


def _tone(rate, seconds, amplitude):
    return [amplitude] * int(rate * seconds)


def _pcm(samples):
    import struct

    return struct.pack("<" + "h" * len(samples), *samples)


def _media(samples):
    return {
        "event": "media",
        "stream_sid": "ST1",
        "media": {"payload": base64.b64encode(_pcm(samples)).decode("ascii")},
    }


def _start_message():
    return {
        "event": "start",
        "stream_sid": "ST1",
        "start": {
            "call_sid": "CA-proof",
            "stream_sid": "ST1",
            "media_format": {"encoding": "audio/x-raw", "sample_rate": 16000},
        },
    }


def _patch_voice(monkeypatch, transcript):
    monkeypatch.setattr(app_v3, "SARVAM_API_KEY", "test-key")
    monkeypatch.setattr("local_call.server.prepare_upcoming", lambda reply: None)
    monkeypatch.setattr("local_call.server.open_bulbul_stream", lambda text, language_code="en-IN": None)

    async def neerja(text):
        return samples_to_wav(_tone(16000, 0.05, 1000), 16000)

    class _Response:
        status_code = 200

        def json(self):
            return {"transcript": transcript, "language_code": "en-IN"}

    posted = []

    def post(url, *args, **kwargs):
        posted.append((url, kwargs.get("files"), kwargs.get("data")))
        return _Response()

    monkeypatch.setattr("local_call.server.neerja_audio", neerja)
    monkeypatch.setattr("local_call.server.requests.post", post)
    return posted


async def _customer_turn(session, samples):
    await session.on_message(_media(_tone(16000, 0.4, 0)))
    await session.on_message(_media(samples))
    await session.idle()


def test_a_clause_is_sent_as_16k_pcm(monkeypatch):
    app_v3.reset_memory()
    calls = []

    def opening(text, language_code="en-IN"):
        calls.append((text, language_code))
        return None

    async def neerja(text):
        return samples_to_wav(_tone(22050, 0.1, 1000), 22050)

    monkeypatch.setattr("local_call.server.open_bulbul_stream", opening)
    monkeypatch.setattr("local_call.server.neerja_audio", neerja)
    monkeypatch.setattr("local_call.server.prepare_upcoming", lambda reply: None)

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
    assert calls
    assert all(language == "en-IN" for _text, language in calls)
    payload = base64.b64decode(sent[0]["media"]["payload"])
    assert sent[0]["event"] == "media"
    assert len(payload) % 320 == 0
    assert 3200 <= len(payload) <= 100000
    assert payload[: len(pcm)] == pcm


def test_customer_audio_is_sent_to_saaras(monkeypatch):
    call_log.begin()
    posted = _patch_voice(monkeypatch, "Yes")

    async def run():
        sent = []

        async def send(raw):
            sent.append(json.loads(raw))

        session = PhoneSession(send=send, scorer=lambda log: None)
        try:
            await session.on_message(_start_message())
            await session.idle()
            await _customer_turn(session, _tone(16000, 0.5, 8000) + _tone(16000, 0.7, 0))
            return app_v3.conversation_memory["stage"], sent
        finally:
            await session.finish()

    stage, sent = asyncio.run(run())
    assert stage == "qualify"
    assert posted
    url, files, data = posted[0]
    assert url.endswith("/speech-to-text")
    name, audio, content_type = files["file"]
    assert name == "speech.wav"
    assert content_type == "audio/wav"
    assert audio[:4] == b"RIFF"
    assert data["model"] == "saaras:v4"
    assert data["mode"] == "translate"
    assert any(item["event"] == "media" for item in sent)


def test_agreeing_to_talk_asks_the_area_and_not_the_greeting_again(monkeypatch):
    call_log.begin()
    _patch_voice(monkeypatch, "Yeah, you can tell me now")

    async def run():
        async def send(raw):
            return None

        session = PhoneSession(send=send, scorer=lambda log: None)
        try:
            await session.on_message(_start_message())
            await session.idle()
            greeting = call_log.turns[0]["reply"]
            await _customer_turn(session, _tone(16000, 0.5, 8000) + _tone(16000, 0.7, 0))
            return greeting, call_log.turns[-1]["reply"]
        finally:
            await session.finish()

    greeting, reply = asyncio.run(run())
    assert "good time" in greeting.lower()
    assert "good time" not in reply.lower()
    assert "which part of bengaluru" in reply.lower()
    assert app_v3.conversation_memory["stage"] == "qualify"


def test_a_hold_stays_quiet_for_45_seconds(monkeypatch):
    call_log.begin()
    _patch_voice(monkeypatch, "One minute")
    slept = []
    real_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        slept.append(seconds)
        if seconds:
            return None
        await real_sleep(0)

    monkeypatch.setattr(phone.asyncio, "sleep", fast_sleep)

    async def run():
        async def send(raw):
            return None

        session = PhoneSession(send=send, scorer=lambda log: None)
        try:
            await session.on_message(_start_message())
            await session.idle()
            await session.on_message(_media(_tone(16000, 0.4, 0)))
            await session.on_message(_media(_tone(16000, 0.5, 8000) + _tone(16000, 0.7, 0)))
            await session.hold_task
            await session.idle()
            return [turn["reply"] for turn in call_log.turns]
        finally:
            await session.finish()

    replies = asyncio.run(run())
    assert 45 in slept
    assert replies[1] == "Sure, I will wait."
    assert "good time" in replies[2].lower()


def test_speech_during_a_hold_cancels_the_quiet(monkeypatch):
    call_log.begin()
    transcripts = iter(["One minute", "Yeah, you can tell me now"])

    def opening(text, language_code="en-IN"):
        return None

    async def neerja(text):
        return samples_to_wav(_tone(16000, 0.05, 1000), 16000)

    class _Response:
        status_code = 200

        def json(self):
            return {"transcript": next(transcripts), "language_code": "en-IN"}

    monkeypatch.setattr(app_v3, "SARVAM_API_KEY", "test-key")
    monkeypatch.setattr("local_call.server.prepare_upcoming", lambda reply: None)
    monkeypatch.setattr("local_call.server.open_bulbul_stream", opening)
    monkeypatch.setattr("local_call.server.neerja_audio", neerja)
    monkeypatch.setattr("local_call.server.requests.post", lambda *args, **kwargs: _Response())
    started = asyncio.Event()
    release = asyncio.Event()
    real_sleep = asyncio.sleep

    async def parked(seconds):
        if seconds >= 45:
            started.set()
            await release.wait()
            return None
        await real_sleep(0)

    monkeypatch.setattr(phone.asyncio, "sleep", parked)

    async def run():
        async def send(raw):
            return None

        session = PhoneSession(send=send, scorer=lambda log: None)
        try:
            await session.on_message(_start_message())
            await session.idle()
            utterance = _tone(16000, 0.5, 8000) + _tone(16000, 0.7, 0)
            await session.on_message(_media(_tone(16000, 0.4, 0)))
            await session.on_message(_media(utterance))
            await started.wait()
            await session.on_message(_media(_tone(16000, 0.4, 0)))
            await session.on_message(_media(utterance))
            await session.idle()
            return [turn["reply"] for turn in call_log.turns], session.hold_task.done()
        finally:
            release.set()
            await session.finish()

    replies, finished = asyncio.run(run())
    assert finished
    assert replies[1] == "Sure, I will wait."
    assert "which part of bengaluru" in replies[-1].lower()
    assert sum("good time" in reply.lower() for reply in replies) == 1


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


def test_the_singapore_dial_uses_api_exotel_and_stays_quiet_without_a_poster(tmp_path):
    calls = []

    def poster(url, data, auth, timeout):
        calls.append((url, data, auth, timeout))
        return _Accepted()

    campaign = dialer.Dialer([FIRST])
    auth = ("key", "token")
    placed = campaign.place(
        FIRST,
        DAY,
        poster,
        account_sid="sobha10",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert placed["placed"] is True
    url, form, used_auth, timeout = calls[0]
    assert url == "https://api.exotel.com/v1/Accounts/sobha10/Calls/connect"
    assert "api.in.exotel.com" not in url
    assert form["From"] == FIRST
    assert form["CallerId"] == "01140000000"
    assert form["StreamType"] == "bidirectional"
    assert form["Record"] == "false"
    assert form["StreamUrl"].endswith("/exotel/media?sample-rate=16000")
    assert used_auth == auth
    assert timeout == 30

    blocked = campaign.place(
        FIRST,
        DAY,
        None,
        account_sid="sobha10",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert blocked["reason"] == "a call is already open"
    campaign.note_closed(FIRST)
    again = campaign.place(
        "+910000000002",
        DAY,
        None,
        account_sid="sobha10",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert again["reason"] == "live Exotel calls are not enabled in this build"
    dark = dialer.Dialer().place(
        FIRST,
        NIGHT,
        poster,
        account_sid="sobha10",
        caller_id="01140000000",
        stream_url="wss://example.test/exotel/media",
        auth=auth,
    )
    assert dark["reason"] == "outside daytime hours"
    assert len(calls) == 1

    listing = tmp_path / "customers.txt"
    listing.write_text(f"# keep\n{FIRST}\n", encoding="utf-8")
    assert dialer.load_numbers(listing) == [FIRST]
    try:
        dialer.load_numbers(__file__)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("a list inside the repository was accepted")


def test_a_local_stream_address_is_not_dialed(monkeypatch):
    from local_call.proof_call import place_one

    monkeypatch.setenv("EXOTEL_ACCOUNT_SID", "sobha10")
    monkeypatch.setenv("EXOTEL_API_KEY", "key")
    monkeypatch.setenv("EXOTEL_API_TOKEN", "token")
    monkeypatch.setenv("EXOTEL_CALLER_ID", "01140000000")
    monkeypatch.setenv("EXOTEL_PROOF_NUMBER", FIRST)
    monkeypatch.setenv("EXOTEL_STREAM_URL", "wss://127.0.0.1:8018/exotel/media")
    called = []
    result = place_one(lambda *args, **kwargs: called.append(args), DAY)
    assert called == []
    assert "public wss" in result["reason"]
    assert dialer.public_wss("wss://example.test/exotel/media") is True
    assert dialer.public_wss("ws://example.test/exotel/media") is False


def test_an_xml_acceptance_still_returns_the_call_id():
    class _Xml:
        status_code = 200
        text = "<TwilioResponse><Call><Sid>CA-xml</Sid></Call></TwilioResponse>"

        def json(self):
            raise ValueError("xml")

    assert dialer.call_sid(_Xml()) == "CA-xml"


def test_the_phone_socket_is_on_the_local_page():
    paths = [getattr(route, "path", "") for route in local_app.routes]
    assert "/" in paths
    assert "/exotel/media" in paths
    assert dialer.configured_stream_url("wss://example.test").endswith(
        "/exotel/media?sample-rate=16000"
    )

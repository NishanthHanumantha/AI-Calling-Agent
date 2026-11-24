"""A browser call becomes one recording, and the review link asks for a passphrase."""

import asyncio
import io
import struct
import wave
from pathlib import Path

from starlette.requests import Request

from local_call.call_record import CallLog
from local_call.recording import mix_to_wav
from local_call.server import (
    enter_review,
    stakeholder_allowed,
    stakeholder_gate,
)


def _tone(rate: int, seconds: float = 0.05) -> bytes:
    count = int(rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(struct.pack("<" + "h" * count, *([800] * count)))
    return buffer.getvalue()


def _request(cookie: str = "") -> Request:
    headers = []
    if cookie:
        headers.append((b"cookie", f"stakeholder={cookie}".encode()))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 5000),
            "server": ("127.0.0.1", 8021),
        }
    )


def test_two_sample_rates_become_one_recording():
    mixed = mix_to_wav([_tone(8000), _tone(48000)])
    assert mixed
    with wave.open(io.BytesIO(mixed), "rb") as clip:
        assert clip.getframerate() == 16000
        assert clip.getnchannels() == 1
        assert clip.getnframes() > int(16000 * 0.05)


def test_saving_a_call_writes_its_recording(tmp_path: Path):
    log = CallLog()
    log.note_clip(_tone(16000))
    log.add_reply("Hello?", "greeting", None)
    log.save(tmp_path)
    assert log.recording == f"{log.call_id}.wav"
    saved = tmp_path / log.recording
    assert saved.is_file()
    assert '"recording":' in (tmp_path / f"{log.call_id}.json").read_text(encoding="utf-8")


def test_the_page_stays_open_until_a_passphrase_is_set(monkeypatch):
    monkeypatch.delenv("STAKEHOLDER_PASSPHRASE", raising=False)
    assert stakeholder_allowed(_request()) is True


def test_the_passphrase_opens_the_review_and_rejects_another(monkeypatch):
    monkeypatch.setenv("STAKEHOLDER_PASSPHRASE", "review-secret")
    assert stakeholder_allowed(_request()) is False
    denied = asyncio.run(enter_review(_request(), passphrase="nope", next="/calls"))
    assert denied.status_code == 401
    opened = asyncio.run(enter_review(_request(), passphrase="review-secret", next="/calls"))
    assert opened.status_code == 303
    assert opened.headers["location"] == "/calls"
    cookie = opened.headers["set-cookie"]
    token = cookie.split("stakeholder=", 1)[1].split(";", 1)[0]
    assert stakeholder_allowed(_request(token)) is True

    async def page(_request):
        from fastapi.responses import Response

        return Response("open")

    blocked = asyncio.run(stakeholder_gate(_request(), page))
    assert blocked.status_code == 200
    assert b"passphrase" in blocked.body
    allowed = asyncio.run(stakeholder_gate(_request(token), page))
    assert allowed.body == b"open"

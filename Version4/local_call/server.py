"""Local microphone call for Version4.

Reuses app_v3.voice() and app_v3.handle_speech(). Twilio, Gather, and
Polly are not used. Bind is localhost only.
"""

import sys
import time
from pathlib import Path

import edge_tts
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

VOICE = "en-IN-NeerjaNeural"

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(__file__).resolve().parent / "runs"
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env", override=False)
sys.path.insert(0, str(ROOT))

import app_v3  # noqa: E402
from local_call.call_record import CallLog  # noqa: E402
from local_call.evaluate import evaluate_saved_call  # noqa: E402

local_app = FastAPI()
PAGE = Path(__file__).with_name("index.html")
call_log = CallLog()


class _Request:
    def __init__(self, speech):
        self._form = {"SpeechResult": speech, "UnstableSpeechResult": ""}

    async def form(self):
        return self._form


class Turn(BaseModel):
    text: str = ""


def spoken_reply(xml):
    import re

    parts = re.findall(r"<Say[^>]*>(.*?)</Say>", xml, flags=re.S)
    text = " ".join(part.strip() for part in parts if part.strip())
    return (
        text.replace("&apos;", "'")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def payload_from(response):
    xml = response.body.decode() if isinstance(response.body, bytes) else response.body
    return {
        "reply": spoken_reply(xml),
        "ended": "<Hangup" in xml,
        "stage": app_v3.conversation_memory.get("stage", ""),
    }


def project_facts():
    return {
        "project": app_v3.PROJECT_KB,
        "sizes": app_v3.UNIT_SIZES,
        "slots": list(app_v3.BOOKABLE_SLOTS),
    }


@local_app.get("/")
def page():
    return FileResponse(PAGE, headers={"Cache-Control": "no-store"})


@local_app.post("/local/start")
async def start_call():
    if call_log.turns and not call_log.saved:
        _save_and_score()
    call_log.begin()
    payload = payload_from(await app_v3.voice())
    call_log.add_reply(payload["reply"], payload["stage"], None)
    return payload


@local_app.post("/local/turn")
async def take_turn(turn: Turn):
    speech = (turn.text or "").strip()

    async def run():
        return await app_v3.handle_speech(_Request(speech), SpeechResult=speech)

    response, llm_ms = await _timed_chat(run)
    payload = payload_from(response)
    call_log.add_reply(payload["reply"], payload["stage"], llm_ms, customer=speech)
    return payload


async def _timed_chat(action):
    captured = {"ms": None}
    original = app_v3.requests.post

    def wrapped(url, *args, **kwargs):
        target = url if isinstance(url, str) else ""
        if "chat/completions" not in target:
            return original(url, *args, **kwargs)
        started = time.perf_counter()
        try:
            return original(url, *args, **kwargs)
        finally:
            captured["ms"] = int((time.perf_counter() - started) * 1000)

    app_v3.requests.post = wrapped
    try:
        return await action(), captured["ms"]
    finally:
        app_v3.requests.post = original


def _save_and_score():
    path = call_log.save(RUNS)
    record = call_log.as_dict()
    result = evaluate_saved_call(record, RUNS, project_facts())
    return {"saved": True, "call_id": record["call_id"], "scorecard": result["scorecard"], "path": path.name}


@local_app.post("/local/end")
def end_call():
    if not call_log.turns or call_log.saved:
        return {"saved": False, "scorecard": ""}
    result = _save_and_score()
    return {"saved": True, "call_id": result["call_id"], "scorecard": result["scorecard"]}


@local_app.post("/local/speak")
async def speak_reply(turn: Turn):
    """Speak one reply with Neerja. The browser voice is not used."""
    text = (turn.text or "").strip()
    if not text:
        return Response(status_code=400)
    started = time.perf_counter()
    communicate = edge_tts.Communicate(text, VOICE)
    parts = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            parts.append(chunk["data"])
    audio = b"".join(parts)
    if not audio:
        return Response(status_code=502)
    call_log.note_tts(int((time.perf_counter() - started) * 1000))
    return Response(content=audio, media_type="audio/mpeg")


@local_app.post("/local/listen")
async def listen(file: UploadFile = File(...)):
    """Transcribe one microphone clip. The browser speech service is not used."""
    audio = await file.read()
    if len(audio) < 800:
        return {"text": ""}
    if not app_v3.SARVAM_API_KEY:
        return {"text": "", "error": "Speech transcription is not configured."}
    started = time.perf_counter()
    response = requests.post(
        "https://api.sarvam.ai/speech-to-text",
        headers={"api-subscription-key": app_v3.SARVAM_API_KEY},
        files={"file": ("speech.wav", audio, "audio/wav")},
        data={
            "model": "saaras:v4",
            "mode": "transcribe",
            "language_code": "en-IN",
            "keyterms": '["SOBHA","Townpark","BHK","Whitefield","Electronic City","Hosur Road"]',
        },
        timeout=30,
    )
    if response.status_code != 200:
        detail = ""
        try:
            err = response.json().get("error") or {}
            detail = f"{err.get('code', '')} {(err.get('message') or '')[:160]}".strip()
        except Exception:
            detail = "unreadable"
        print("LOCAL STT STATUS:", response.status_code, detail)
        return {"text": "", "error": "Could not transcribe that. Please say it again."}
    transcript = (response.json().get("transcript") or "").strip()
    if transcript:
        call_log.note_stt(transcript, int((time.perf_counter() - started) * 1000))
    return {"text": transcript}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(local_app, host="127.0.0.1", port=8010)

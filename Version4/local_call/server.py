"""Local microphone call for Version4.

Reuses app_v3.voice() and app_v3.handle_speech(). Twilio, Gather, and
Polly are not used. Bind is localhost only.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

import app_v3  # noqa: E402

local_app = FastAPI()
PAGE = Path(__file__).with_name("index.html")


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


@local_app.get("/")
def page():
    return FileResponse(PAGE)


@local_app.post("/local/start")
async def start_call():
    return payload_from(await app_v3.voice())


@local_app.post("/local/turn")
async def take_turn(turn: Turn):
    speech = (turn.text or "").strip()
    response = await app_v3.handle_speech(_Request(speech), SpeechResult=speech)
    return payload_from(response)


@local_app.post("/local/listen")
async def listen(file: UploadFile = File(...)):
    """Transcribe one microphone clip. The browser speech service is not used."""
    audio = await file.read()
    if len(audio) < 800:
        return {"text": ""}
    if not app_v3.SARVAM_API_KEY:
        return {"text": "", "error": "Speech transcription is not configured."}
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
    return {"text": transcript}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(local_app, host="127.0.0.1", port=8010)

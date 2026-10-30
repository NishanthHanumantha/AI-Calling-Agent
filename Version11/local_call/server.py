"""Local microphone call for Version11.

Reuses app_v3.voice() and app_v3.handle_speech(). Twilio, Gather, and
Polly are not used. Bind is localhost only. Spoken replies use Sarvam
Bulbul, and fall back to Neerja if Bulbul does not return audio.
The page speaks the first clause before the later clauses are synthesized.
"""

import base64
import io
import json
import re
import struct
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import edge_tts
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

VOICE = "en-IN-NeerjaNeural"
BULBUL_URL = "https://api.sarvam.ai/text-to-speech"
BULBUL_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"
BULBUL_MODEL = "bulbul:v3"
BULBUL_SPEAKER = "ishita"
TRANSLATE_URL = "https://api.sarvam.ai/translate"

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
    payload = {
        "reply": spoken_reply(xml),
        "ended": "<Hangup" in xml,
        "stage": app_v3.conversation_memory.get("stage", ""),
    }
    follow = (app_v3.conversation_memory.get("question_after_hold") or "").strip()
    if follow and "i will wait" in payload["reply"].lower():
        payload["wait_seconds"] = 45
        payload["follow_up"] = follow
    return payload


def project_facts(record=None):
    """Townpark facts, plus the card for every project named in this call."""
    turns = (record or {}).get("turns") or []
    blob = " ".join(
        f"{turn.get('customer') or ''} {turn.get('reply') or ''}" for turn in turns
    )
    return {
        "project": app_v3.PROJECT_KB,
        "sizes": app_v3.UNIT_SIZES,
        "sizes_apply_to": "SOBHA Townpark only",
        "catalog_cards": app_v3.catalog.cards_named_in(blob),
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
    prepare_upcoming(payload["reply"])
    return payload


@local_app.post("/local/resume")
def resume_after_hold():
    """Speak the saved question after a quiet hold. This does not call the model."""
    question = (app_v3.conversation_memory.get("question_after_hold") or "").strip()
    app_v3.conversation_memory["question_after_hold"] = ""
    if question:
        app_v3.remember_spoken(question)
    stage = app_v3.conversation_memory.get("stage", "")
    call_log.add_reply(question, stage, None)
    prepare_upcoming(question)
    return {"reply": question, "ended": False, "stage": stage}


@local_app.post("/local/turn")
async def take_turn(turn: Turn):
    speech = (turn.text or "").strip()

    async def run():
        return await app_v3.handle_speech(_Request(speech), SpeechResult=speech)

    response, llm_ms = await _timed_chat(run)
    payload = payload_from(response)
    call_log.add_reply(payload["reply"], payload["stage"], llm_ms, customer=speech)
    prepare_upcoming(payload["reply"])
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
    call_log.callback_when = app_v3.conversation_memory.get("callback_when")
    path = call_log.save(RUNS)
    record = call_log.as_dict()
    result = evaluate_saved_call(record, RUNS, project_facts(record))
    return {"saved": True, "call_id": record["call_id"], "scorecard": result["scorecard"], "path": path.name}


@local_app.post("/local/end")
def end_call():
    if not call_log.turns or call_log.saved:
        return {"saved": False, "scorecard": ""}
    result = _save_and_score()
    return {"saved": True, "call_id": result["call_id"], "scorecard": result["scorecard"]}


def for_speech(text):
    """Say the company name and square feet. The written reply stays as written."""
    spoken = re.sub(r"\bSOBHA\b", "Sobha", text or "")
    return re.sub(r"\bsq\.?\s*ft\.?", "square feet", spoken, flags=re.IGNORECASE)


def speech_clauses(text):
    """Split a finished reply into spoken clauses. 'sq.' stays in one clause."""
    raw = (text or "").strip()
    if not raw:
        return []
    protected = re.sub(r"\bsq\.", lambda match: match.group(0)[:-1] + "\u2024", raw, flags=re.IGNORECASE)
    protected = re.sub(
        r"\b([AaPp])\.(?=[Mm]\.)",
        lambda match: match.group(1) + "\u2024",
        protected,
    )
    clauses = []
    for part in re.split(r"(?<=[.?!])\s+|\s*;\s*", protected):
        restored = part.replace("\u2024", ".").strip()
        if restored:
            clauses.append(restored)
    return clauses


@local_app.post("/local/clauses")
def list_clauses(turn: Turn):
    """Return the clauses of a finished reply, in speaking order."""
    return {"clauses": speech_clauses(turn.text or "")}


def speaking_style(text):
    """One pace and warmth for every clause. The speaker stays Ishita."""
    return 1.0, 0.6


_translation_cache = {}
_translation_pending = {}
_translation_lock = threading.Lock()
_translation_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="translate")


def clear_translations():
    """Drop cached clauses. Used by tests so one call does not answer the next."""
    with _translation_lock:
        _translation_cache.clear()
        _translation_pending.clear()


def _request_translation(text, target):
    """Ask Sarvam for one clause. None when the call does not return text."""
    try:
        response = requests.post(
            TRANSLATE_URL,
            headers={
                "api-subscription-key": app_v3.SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "input": text[:2000],
                "source_language_code": "en-IN",
                "target_language_code": target,
                "model": "sarvam-translate:v1",
                "mode": "formal",
                "speaker_gender": "Female",
                "numerals_format": "international",
            },
            timeout=2,
        )
    except requests.Timeout:
        print("LOCAL TRANSLATE STATUS: timeout")
        return None
    if response.status_code != 200:
        print("LOCAL TRANSLATE STATUS:", response.status_code)
        return None
    try:
        translated = (response.json().get("translated_text") or "").strip()
    except Exception:
        return None
    return translated or None


def _fetch_and_store(clause, language):
    translated = _request_translation(clause, language)
    if translated:
        with _translation_lock:
            _translation_cache[(language, clause)] = translated
    return translated


def _release_translation(key, future):
    with _translation_lock:
        if _translation_pending.get(key) is future:
            _translation_pending.pop(key, None)


def _enqueue(clause, language):
    """Start a translation if this clause is not already ready or in flight."""
    clause = (clause or "").strip()
    if (
        not clause
        or language not in app_v3.SPOKEN_LANGUAGES
        or not app_v3.SARVAM_API_KEY
    ):
        return
    key = (language, clause)
    with _translation_lock:
        if key in _translation_cache or key in _translation_pending:
            return
        future = _translation_pool.submit(_fetch_and_store, clause, language)
        _translation_pending[key] = future
    future.add_done_callback(lambda done, item=key: _release_translation(item, done))


def translate_text(text, target):
    """One English clause in the customer's language. A cached clause is not requested again."""
    clause = (text or "").strip()
    if target not in app_v3.SPOKEN_LANGUAGES or not app_v3.SARVAM_API_KEY or not clause:
        return None
    key = (target, clause)
    with _translation_lock:
        cached = _translation_cache.get(key)
        if cached:
            return cached
        future = _translation_pending.get(key)
        if future is None:
            future = _translation_pool.submit(_fetch_and_store, clause, target)
            _translation_pending[key] = future
            future.add_done_callback(lambda done, item=key: _release_translation(item, done))
    try:
        return future.result(timeout=2) or None
    except Exception:
        return None


def _fixed_lines():
    """Qualification lines spoken on every call, area question first."""
    return [
        "Thanks for taking the call. " + app_v3.catalog.AREA_QUESTION,
        app_v3.catalog.CONFIGURATION_QUESTION,
        app_v3.catalog.PURPOSE_QUESTION,
        "Got it. You're looking at this as an investment.",
        "Got it. You're looking at this for your own use.",
        "Would you be interested to know about the project?",
        app_v3.ADVISOR_CLOSE,
    ]


def _predicted_offer():
    """The catalog line that follows a purpose answer. It does not change the call."""
    matches = app_v3._current_matches()
    if not matches:
        return ""
    ceiling = app_v3.conversation_memory.get("price_ceiling")
    strict = bool(app_v3.conversation_memory.get("price_strict"))
    gave_part = bool(
        app_v3.conversation_memory.get("customer_part")
        or app_v3.conversation_memory.get("customer_place")
    )
    listed = app_v3._apartment_shortlist(matches)
    if (
        len(listed) > 2
        and gave_part
        and not app_v3.conversation_memory.get("narrow_asked")
        and not app_v3.conversation_memory.get("customer_place")
    ):
        return app_v3.catalog.suggestion(listed, bedrooms=app_v3._stated_bedrooms()) or ""
    closest = len(matches) > 2
    return app_v3.catalog.suggestion(
        matches,
        ceiling=ceiling,
        strict=strict,
        closest=closest,
        bedrooms=app_v3._stated_bedrooms(),
    ) or ""


def _preview_overview(projects):
    """The overview a yes would speak, without marking the project as opened."""
    chosen = list(projects)[:2]
    if not chosen:
        return ""
    shared = (
        len(chosen) == 2
        and chosen[0].get("place")
        and chosen[0]["place"] == chosen[1].get("place")
    )
    parts = [
        app_v3._theme_sentence(project, include_place=not shared)
        for project in chosen
    ]
    if shared:
        parts.append(app_v3.catalog.location_line(chosen))
    if not app_v3._configuration_already_chosen():
        parts.extend(app_v3._home_line(project, include_size=True) for project in chosen)
    question = (
        app_v3.OPEN_PROJECTS_QUESTION if len(chosen) > 1 else app_v3.OPEN_PROJECT_QUESTION
    )
    parts.append(question)
    return " ".join(part for part in parts if part)


def _offered_overviews():
    pending = [
        app_v3.catalog.BY_ID[item]
        for item in (app_v3.conversation_memory.get("pending_project_ids") or [])
        if item in app_v3.catalog.BY_ID
    ]
    if not pending:
        return []
    lines = []
    if len(pending) >= 2:
        lines.append(_preview_overview(pending[:2]))
    for project in pending[:2]:
        lines.append(_preview_overview([project]))
    return lines


def _upcoming_lines(reply):
    """The next spoken line, now that the call is in one language."""
    low = (reply or "").lower()
    lines = []
    if "part of bengaluru" in low or "which part" in low:
        lines.append(app_v3.catalog.CONFIGURATION_QUESTION)
    if "which configuration" in low:
        lines.append(app_v3.catalog.PURPOSE_QUESTION)
    if "live in" in low and "investment" in low:
        offer = _predicted_offer()
        if offer:
            lines.append("Got it. You're looking at this as an investment. " + offer)
            lines.append("Got it. You're looking at this for your own use. " + offer)
        else:
            lines.append("Got it. You're looking at this as an investment.")
            lines.append("Got it. You're looking at this for your own use.")
    if (
        "interested to know about the project" in low
        or "which one shall i tell you about" in low
        or "shall i cover both" in low
    ):
        lines.extend(_offered_overviews())
    return lines


def _enqueue_lines(lines, languages):
    for line in lines:
        for clause in speech_clauses(line):
            for language in languages:
                _enqueue(clause, language)


def prepare_upcoming(reply):
    """Translate the next line before the customer finishes answering."""
    try:
        language = app_v3.customer_language()
        if language == "en-IN":
            _enqueue_lines(_fixed_lines(), app_v3.SPOKEN_LANGUAGES)
            return
        if language in app_v3.SPOKEN_LANGUAGES:
            _enqueue_lines(_upcoming_lines(reply), [language])
    except Exception as exc:
        print("LOCAL TRANSLATE PREPARE:", type(exc).__name__)


def _bulbul_body(text, language_code):
    return {
        "text": text[:2500],
        "language_code": language_code,
        "model": BULBUL_MODEL,
        "speaker": BULBUL_SPEAKER,
        "pace": 1.0,
        "temperature": 0.6,
        "speech_sample_rate": 22050,
        "output_audio_codec": "wav",
    }


def open_bulbul_stream(text, language_code="en-IN"):
    """Start a Bulbul stream. None when the first audio does not arrive within 4 seconds."""
    if not app_v3.SARVAM_API_KEY or not (text or "").strip():
        return None
    try:
        response = requests.post(
            BULBUL_STREAM_URL,
            headers={
                "api-subscription-key": app_v3.SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json=_bulbul_body(text, language_code),
            stream=True,
            timeout=4,
        )
    except requests.RequestException as exc:
        print("LOCAL TTS STATUS:", type(exc).__name__)
        return None
    if response.status_code != 200:
        message = ""
        try:
            message = (response.json().get("error") or {}).get("message") or ""
        except Exception:
            message = ""
        print("LOCAL TTS STATUS:", response.status_code, message)
        response.close()
        return None
    return response


def first_audio_chunk(response):
    """The first bytes, and the same iterator still holding the rest of the stream."""
    if response is None:
        return None, None
    try:
        if hasattr(response, "iter_content"):
            pending = response.iter_content(chunk_size=4096)
            while True:
                try:
                    chunk = next(pending)
                except StopIteration:
                    return None, None
                if chunk:
                    return chunk, pending
        audios = response.json().get("audios") or []
    except requests.Timeout:
        print("LOCAL TTS STATUS: timeout")
        return None, None
    except Exception:
        return None, None
    if not audios:
        return None, None
    try:
        return base64.b64decode(audios[0]) or None, None
    except Exception:
        return None, None


def _remaining_chunks(response, first, pending):
    try:
        yield first
        if pending is None:
            return
        for chunk in pending:
            if chunk:
                yield chunk
    finally:
        if response is not None and hasattr(response, "close"):
            response.close()


def joined_audio(response, first, pending) -> bytes:
    """The full WAV from the same stream the page plays clause by clause."""
    return b"".join(_remaining_chunks(response, first, pending))


async def clause_bytes(text: str):
    """One clause as WAV or MP3, on the same Bulbul then Neerja path as the page."""
    written = (text or "").strip()
    english = for_speech(written)
    if not english:
        return None
    started = time.perf_counter()
    language = app_v3.customer_language()
    spoken = english
    if language in app_v3.SPOKEN_LANGUAGES:
        translated = translate_text(written, language)
        if translated:
            spoken = translated
        else:
            language = "en-IN"
    app_v3.conversation_memory["speech_language"] = language
    stream = open_bulbul_stream(spoken, language)
    first, pending = first_audio_chunk(stream)
    if not first:
        if stream is not None and hasattr(stream, "close"):
            stream.close()
        audio = await neerja_audio(english)
        if not audio:
            return None
        call_log.note_tts(int((time.perf_counter() - started) * 1000))
        return audio, "audio/mpeg"
    audio = joined_audio(stream, first, pending)
    call_log.note_tts(int((time.perf_counter() - started) * 1000))
    return audio, "audio/wav"


def lone_yes(transcript):
    """True when the clip is only a one-word yes, the sound a noisy room often becomes."""
    raw = re.sub(r"[^a-z\s]", " ", (transcript or "").lower())
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw in {"yeah", "yes", "ya", "ok", "okay", "yep"}


def _named_parts(text):
    """North, East, or South named as whole words."""
    raw = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    words = set(raw.split())
    return [part for part in ("north", "east", "south") if part in words]


def _one_offered_place(transcript, line):
    """One place chosen from a question that listed more than one."""
    offered = app_v3.catalog.places_mentioned(line or "")
    named = app_v3.catalog.places_mentioned(transcript or "")
    return len(offered) > 1 and len(named) == 1 and named[0] in offered


def mostly_echo(transcript, line):
    """True when most of the heard words are already in the line just spoken."""
    def words(text):
        raw = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        return [word for word in raw.split() if len(word) > 1]

    if len(_named_parts(transcript)) == 1:
        return False
    if _one_offered_place(transcript, line):
        return False
    said = words(transcript)
    agent = set(words(line))
    if len(said) < 2 or not agent:
        return False
    shared = sum(1 for word in said if word in agent)
    return shared / len(said) >= 0.6


async def neerja_audio(text):
    communicate = edge_tts.Communicate(text, VOICE)
    parts = []
    try:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                parts.append(chunk["data"])
    except Exception as exc:
        print("LOCAL FALLBACK VOICE FAILED:", type(exc).__name__)
        return None
    return b"".join(parts)


@local_app.post("/local/speak")
async def speak_reply(turn: Turn):
    """Speak one clause. Playback can start on the first Bulbul chunk."""
    written = (turn.text or "").strip()
    english = for_speech(written)
    if not english:
        return Response(status_code=400)
    started = time.perf_counter()
    language = app_v3.customer_language()
    spoken = english
    if language in app_v3.SPOKEN_LANGUAGES:
        translated = translate_text(written, language)
        if translated:
            spoken = translated
        else:
            language = "en-IN"
    app_v3.conversation_memory["speech_language"] = language
    stream = open_bulbul_stream(spoken, language)
    first, pending = first_audio_chunk(stream)
    if not first:
        if stream is not None and hasattr(stream, "close"):
            stream.close()
        audio = await neerja_audio(english)
        media = "audio/mpeg"
        voice = "neerja"
        if not audio:
            return Response(status_code=502)
        call_log.note_tts(int((time.perf_counter() - started) * 1000))
        return Response(content=audio, media_type=media, headers={"X-Voice": voice})
    call_log.note_tts(int((time.perf_counter() - started) * 1000))
    return StreamingResponse(
        _remaining_chunks(stream, first, pending),
        media_type="audio/wav",
        headers={
            "X-Voice": "bulbul-ishita",
            "X-Audio-Format": "wav",
        },
    )


def clip_has_speech(audio: bytes, min_seconds: float = 0.45) -> bool:
    """True when a clip rises above the room for long enough to be the customer."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError):
        return False
    if channels != 1 or width != 2 or rate <= 0 or not frames:
        return False
    count = len(frames) // 2
    samples = struct.unpack("<" + "h" * count, frames[: count * 2])
    frame_n = max(1, int(rate * 0.02))
    levels = []
    for start in range(0, count - frame_n + 1, frame_n):
        chunk = samples[start : start + frame_n]
        energy = sum(sample * sample for sample in chunk) / len(chunk)
        levels.append((energy ** 0.5) / 32768.0)
    if not levels:
        return False
    quiet = [level for level in levels if level < 0.05]
    if len(quiet) >= 3:
        ordered = sorted(quiet)
        floor = ordered[len(ordered) // 2]
        threshold = max(0.05, floor * 3)
    else:
        threshold = 0.05
    loud = sum(1 for level in levels if level > threshold)
    return (loud * frame_n / rate) >= min_seconds


def stt_keyterms():
    """Boost the places just asked. Townpark is boosted only for a South call."""
    memory = app_v3.conversation_memory
    last = memory.get("last_agent_question") or ""
    offered = app_v3.catalog.places_mentioned(last) if "closer to" in last.lower() else []
    terms = ["SOBHA", "BHK"]
    low = last.lower()
    part_question = "part of bengaluru" in low or "which part" in low
    if offered:
        terms.extend(offered)
    elif part_question:
        terms.extend(["Bengaluru", "Bangalore", "North", "East", "South", "West"])
    elif memory.get("customer_part") == "East":
        terms.extend(["Whitefield", "Panathur", "Sarjapur", "Koramangala", "Bhoganahalli"])
    elif memory.get("customer_part") == "South":
        terms.extend(["Electronic City", "Hosur Road", "Bannerghatta", "Banashankari"])
    south_places = {"electronic city", "bannerghatta", "banashankari"}
    east_places = {"panathur", "whitefield", "sarjapur", "koramangala", "bhoganahalli"}
    south_question = bool(offered) and any(place.lower() in south_places for place in offered) and not any(
        place.lower() in east_places for place in offered
    )
    if (
        "sobha-townpark" in (memory.get("active_project_ids") or [])
        or memory.get("customer_part") == "South"
        or south_question
        or "townpark" in last.lower()
    ):
        terms.append("Townpark")
    seen = []
    for term in terms:
        if term not in seen:
            seen.append(term)
    return json.dumps(seen)


BOOSTED_KEYTERMS = (
    "electronic city",
    "hosur road",
    "whitefield",
    "townpark",
    "sobha",
    "bhk",
)


def transcript_is_only_keyterms(text):
    """A clip that comes back as only a boosted word was not an answer."""
    raw = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return False
    remaining = raw
    for term in BOOSTED_KEYTERMS:
        remaining = re.sub(rf"\b{re.escape(term)}\b", " ", remaining)
    return re.search(r"[a-z0-9]", remaining) is None


def transcribe_clip(audio: bytes, heard: str = "1") -> dict:
    """Saaras on one clip. The browser speech service is not used."""
    if len(audio) < 800 or not clip_has_speech(audio):
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
            "mode": "translate",
            "language_code": app_v3.stt_language_code(),
            "keyterms": stt_keyterms(),
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
    body = response.json()
    transcript = (body.get("transcript") or "").strip()
    if transcript_is_only_keyterms(transcript):
        return {"text": ""}
    if mostly_echo(transcript, app_v3.conversation_memory.get("last_spoken_reply")):
        return {"text": ""}
    if str(heard) == "0" and lone_yes(transcript):
        return {"text": ""}
    if transcript:
        app_v3.note_detected_language(body.get("language_code"))
        call_log.note_stt(transcript, int((time.perf_counter() - started) * 1000))
    return {"text": transcript}


@local_app.post("/local/listen")
async def listen(file: UploadFile = File(...), heard: str = Form("1")):
    """Transcribe one microphone clip. The browser speech service is not used."""
    return transcribe_clip(await file.read(), heard)


@local_app.websocket("/exotel/media")
async def exotel_media(socket: WebSocket):
    """Exotel voice socket. The script and the voice stay the ones on this page."""
    from local_call.phone import phone_socket

    await phone_socket(socket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(local_app, host="127.0.0.1", port=8018)

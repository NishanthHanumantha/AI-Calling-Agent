"""Exotel voice socket. The script and the voice stay the ones on the page."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import struct
import subprocess
import wave

from fastapi import WebSocketDisconnect

STREAM_PATH = "/exotel/media"
SAMPLE_RATE = 16000
CHUNK_BYTES = 3200


class _Request:
    def __init__(self, speech):
        self._form = {"SpeechResult": speech, "UnstableSpeechResult": ""}

    async def form(self):
        return self._form


def samples_to_wav(samples, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = struct.pack("<" + "h" * len(samples), *samples) if samples else b""
        wav.writeframes(frames)
    return buffer.getvalue()


def wav_to_samples(data: bytes):
    """PCM samples from a WAV, including one whose chunk sizes are unset."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a wav")
    pos = 12
    rate = SAMPLE_RATE
    channels = 1
    pcm = b""
    while pos + 8 <= len(data):
        kind = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        start = pos + 8
        if size in (0xFFFFFFFF, 0):
            body = data[start:]
            pos = len(data)
        else:
            body = data[start : start + size]
            pos = start + size + (size % 2)
        if kind == b"fmt " and len(body) >= 16:
            channels = struct.unpack_from("<H", body, 2)[0] or 1
            rate = struct.unpack_from("<I", body, 4)[0] or rate
        elif kind == b"data":
            pcm = body
    count = len(pcm) // 2
    raw = struct.unpack("<" + "h" * count, pcm[: count * 2]) if count else ()
    if channels > 1:
        raw = raw[0::channels]
    return list(raw), rate


def resample(samples, source_rate: int, target_rate: int) -> bytes:
    if not samples:
        return b""
    if source_rate == target_rate:
        return struct.pack("<" + "h" * len(samples), *samples)
    count = max(1, int(round(len(samples) * target_rate / source_rate)))
    last = len(samples) - 1
    out = []
    for index in range(count):
        position = index * source_rate / target_rate
        left = min(int(position), last)
        right = min(left + 1, last)
        mix = position - left
        value = samples[left] * (1 - mix) + samples[right] * mix
        out.append(max(-32768, min(32767, int(round(value)))))
    return struct.pack("<" + "h" * len(out), *out)


def ffmpeg_to_pcm(audio: bytes, rate: int = SAMPLE_RATE) -> bytes:
    program = shutil.which("ffmpeg")
    if not program:
        raise RuntimeError("MP3 playback on the phone needs ffmpeg")
    result = subprocess.run(
        [
            program,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(rate),
            "pipe:1",
        ],
        input=audio,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError("could not convert the spoken clause")
    if len(result.stdout) % 2:
        return result.stdout[:-1]
    return result.stdout


def to_pcm16k(audio: bytes, media: str = "", rate: int = SAMPLE_RATE) -> bytes:
    """16-bit little-endian mono PCM at Exotel's rate. 16 kHz unless the call says otherwise."""
    if audio[:4] == b"RIFF" or "wav" in (media or ""):
        samples, source_rate = wav_to_samples(audio)
        return resample(samples, source_rate, rate)
    return ffmpeg_to_pcm(audio, rate)


def media_chunks(pcm: bytes) -> list[bytes]:
    """Exotel chunks: a multiple of 320 bytes, at least 3200 and at most 100000."""
    if len(pcm) % 320:
        pcm += b"\x00" * (320 - (len(pcm) % 320))
    if len(pcm) < 3200:
        pcm += b"\x00" * (3200 - len(pcm))
    return [pcm[index : index + CHUNK_BYTES] for index in range(0, len(pcm), CHUNK_BYTES)]


def _levels(samples, rate: int):
    frame_n = max(1, int(rate * 0.02))
    levels = []
    for start in range(0, len(samples) - frame_n + 1, frame_n):
        chunk = samples[start : start + frame_n]
        energy = sum(sample * sample for sample in chunk) / len(chunk)
        levels.append((energy ** 0.5) / 32768.0)
    return levels, frame_n


def _threshold(levels):
    quiet = [level for level in levels if level < 0.05]
    if len(quiet) >= 3:
        ordered = sorted(quiet)
        floor = ordered[len(ordered) // 2]
        return max(0.05, floor * 3), max(0.14, floor * 4)
    return 0.05, 0.14


class SpeechEar:
    """The page's 0.45 second speech gate, on 16-bit PCM instead of a microphone."""

    def __init__(self, rate: int = SAMPLE_RATE):
        self.rate = rate
        self.samples: list[int] = []

    def reset(self) -> None:
        self.samples = []

    def push(self, pcm: bytes, playing: bool):
        count = len(pcm) // 2
        if not count:
            return None
        self.samples.extend(struct.unpack("<" + "h" * count, pcm[: count * 2]))
        levels, frame_n = _levels(self.samples, self.rate)
        if not levels:
            return None
        threshold, interrupt = _threshold(levels)
        if playing:
            need = int(self.rate * 1.0)
            if len(self.samples) < need:
                return None
            recent_levels, _ = _levels(self.samples[-need:], self.rate)
            loud = sum(1 for level in recent_levels if level > interrupt)
            if recent_levels and loud / len(recent_levels) >= 0.6:
                self.reset()
                return ("barge", None)
            return None
        loud_time = sum(1 for level in levels if level > threshold) * frame_n / self.rate
        if loud_time < 0.45:
            return None
        trailing = 0
        for level in reversed(levels):
            if level > threshold:
                break
            trailing += 1
        if trailing * frame_n / self.rate < 0.6:
            return None
        wav = samples_to_wav(self.samples, self.rate)
        self.reset()
        return ("utterance", wav)


def score_phone_call(log) -> None:
    from local_call.evaluate import evaluate_saved_call
    from local_call.server import RUNS, project_facts

    record = log.as_dict()
    log.save(RUNS)
    evaluate_saved_call(record, RUNS, project_facts(record))


class PhoneSession:
    """One Exotel socket. Greeting, listen, and speech are the page's steps."""

    def __init__(self, send, rate: int = SAMPLE_RATE, scorer=None):
        self.send = send
        self.rate = rate
        self.ear = SpeechEar(rate)
        self.stream_sid = None
        self.call_sid = None
        self.playing = False
        self.stopped = False
        self.closed = False
        self.scorer = scorer
        self.task = None
        self.hold_task = None
        self.ignore_samples = 0

    async def _emit(self, message: dict) -> None:
        await self.send(json.dumps(message))

    async def on_message(self, raw) -> None:
        message = json.loads(raw) if isinstance(raw, str) else raw
        event = message.get("event")
        if event == "start":
            await self._start(message)
        elif event == "media":
            await self._media(message)
        elif event == "stop":
            await self.finish()

    async def _start(self, message: dict) -> None:
        import app_v3
        from local_call.server import _save_and_score, call_log, payload_from, prepare_upcoming

        start = message.get("start") or {}
        self.stream_sid = message.get("stream_sid") or start.get("stream_sid")
        self.call_sid = start.get("call_sid") or message.get("call_sid")
        media = start.get("media_format") or {}
        if media.get("sample_rate"):
            self.rate = int(media["sample_rate"])
            self.ear.rate = self.rate
        print("EXOTEL SAMPLE RATE:", self.rate)
        if call_log.turns and not call_log.saved:
            _save_and_score()
        response = await app_v3.voice()
        call_log.begin()
        payload = payload_from(response)
        call_log.add_reply(payload["reply"], payload["stage"], None)
        prepare_upcoming(payload["reply"])
        self.task = asyncio.create_task(self.speak(payload["reply"]))

    async def speak(self, text: str) -> None:
        from local_call.server import clause_bytes, speech_clauses

        self.playing = True
        self.stopped = False
        self.ear.reset()
        try:
            clauses = speech_clauses(text) or ([text] if (text or "").strip() else [])
            for clause in clauses:
                if self.stopped:
                    break
                spoken = await clause_bytes(clause)
                if not spoken or self.stopped:
                    continue
                audio, media = spoken
                pcm = to_pcm16k(audio, media, self.rate)
                for chunk in media_chunks(pcm):
                    if self.stopped:
                        break
                    await self._emit(
                        {
                            "event": "media",
                            "stream_sid": self.stream_sid,
                            "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                        }
                    )
                    await asyncio.sleep(len(chunk) / (2 * self.rate))
        finally:
            self.playing = False
            self.ignore_samples = int(self.rate * 0.4)

    async def _media(self, message: dict) -> None:
        payload = ((message.get("media") or {}).get("payload")) or ""
        if not payload:
            return
        pcm = base64.b64decode(payload)
        if not self.playing and self.ignore_samples:
            count = len(pcm) // 2
            if count <= self.ignore_samples:
                self.ignore_samples -= count
                return
            pcm = pcm[self.ignore_samples * 2 :]
            self.ignore_samples = 0
        heard = self.ear.push(pcm, playing=self.playing)
        if not heard:
            return
        kind, wav = heard
        if kind == "barge":
            self.stopped = True
            await self._emit({"event": "clear", "stream_sid": self.stream_sid})
            return
        await self._turn(wav)

    def _cancel_hold(self) -> None:
        if self.hold_task is not None and not self.hold_task.done():
            self.hold_task.cancel()

    async def _turn(self, wav: bytes) -> None:
        import app_v3
        from local_call.server import call_log, payload_from, transcribe_clip

        self._cancel_hold()
        await self._finish_speech()
        heard = transcribe_clip(wav)
        text = (heard.get("text") or "").strip()
        if not text:
            return
        response = await app_v3.handle_speech(_Request(text), SpeechResult=text)
        payload = payload_from(response)
        call_log.add_reply(payload["reply"], payload["stage"], None, customer=text)
        self.task = asyncio.create_task(self.speak(payload["reply"]))
        if payload.get("wait_seconds") and payload.get("follow_up") and not payload["ended"]:
            self.hold_task = asyncio.create_task(self._quiet_then_ask(payload))
        elif payload["ended"]:
            await self._finish_speech()
            await self.finish()

    async def _quiet_then_ask(self, payload: dict) -> None:
        """Stay quiet, then ask the saved question unless the customer speaks."""
        import app_v3
        from local_call.server import call_log, prepare_upcoming

        try:
            if self.task is not None:
                await asyncio.shield(self.task)
            await asyncio.sleep(int(payload["wait_seconds"]))
        except asyncio.CancelledError:
            return
        if self.closed:
            return
        follow = payload["follow_up"]
        app_v3.conversation_memory["question_after_hold"] = ""
        app_v3.remember_spoken(follow)
        call_log.add_reply(follow, app_v3.conversation_memory.get("stage", ""), None)
        prepare_upcoming(follow)
        self.task = asyncio.create_task(self.speak(follow))

    async def _finish_speech(self) -> None:
        """Wait out the line already speaking. A voice error must not drop the call."""
        if self.task is None:
            return
        try:
            await self.task
        except Exception as exc:
            print("PHONE SPEAK:", type(exc).__name__)

    async def idle(self) -> None:
        await self._finish_speech()

    async def finish(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stopped = True
        self._cancel_hold()
        await self._finish_speech()
        if self.scorer is not None:
            from local_call.server import call_log

            if call_log.turns and not call_log.saved:
                self.scorer(call_log)


async def phone_socket(socket) -> None:
    await socket.accept()
    rate = 8000
    raw_rate = socket.query_params.get("sample-rate") if socket.query_params else None
    if raw_rate:
        try:
            rate = int(raw_rate)
        except ValueError:
            rate = 8000
    session = PhoneSession(send=socket.send_text, rate=rate, scorer=score_phone_call)
    try:
        while not session.closed:
            await session.on_message(await socket.receive_text())
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print("PHONE SOCKET:", type(exc).__name__)
    finally:
        await session.finish()

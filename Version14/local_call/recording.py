"""One WAV for a browser call. The spoken reply is not chosen here."""

from __future__ import annotations

import io
import wave

from local_call.phone import ffmpeg_to_pcm, resample, wav_to_samples

RATE = 16000


def mix_to_wav(pieces: list[bytes]) -> bytes | None:
    """Customer clips and agent clauses, in the order they were spoken, at one rate."""
    pcm = bytearray()
    for piece in pieces:
        chunk = _pcm(piece)
        if chunk:
            pcm.extend(chunk)
    if not pcm:
        return None
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(bytes(pcm))
    return buffer.getvalue()


def _pcm(audio: bytes) -> bytes:
    if not audio:
        return b""
    if audio[:4] == b"RIFF":
        try:
            samples, rate = wav_to_samples(audio)
        except (ValueError, wave.Error):
            return b""
        return resample(samples, rate, RATE)
    try:
        return ffmpeg_to_pcm(audio, RATE)
    except Exception:
        return b""

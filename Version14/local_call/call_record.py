"""One Version7 localhost call. This does not change the conversation."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_call.usage_note import (
    BULBUL_INR_PER_CHARACTER,
    GST_RATE,
    RATES_CHECKED,
    STT_INR_PER_SECOND,
    TRANSLATE_INR_PER_CHARACTER,
    label_source,
    language_model_inr,
    priced,
)


CALLER_MODEL = "sarvam-105b-conversations"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class CallLog:
    """Collect turns for the single local call in this process."""

    def __init__(self) -> None:
        self.call_id = _now()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.turns: list[dict[str, Any]] = []
        self.callback_when: str | None = None
        self._pending_stt: dict[str, Any] | None = None
        self._pending_audio: float | None = None
        self._pending_calls: list[dict[str, Any]] = []
        self._awaiting_speech: dict[str, Any] | None = None
        self._speech_turn: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self.saved = False

    def begin(self) -> None:
        self.__init__()

    def add_model_calls(self, calls: list[dict[str, Any]]) -> None:
        with self._lock:
            self._pending_calls.extend(calls)

    def note_audio(self, seconds: float | None) -> None:
        with self._lock:
            self._pending_audio = seconds

    def add_reply(
        self,
        reply: str,
        stage: str,
        llm_ms: int | None,
        customer: str | None = None,
        card: str | None = None,
        decision_ms: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            stt_ms = None
            stt_seconds = None
            if customer is not None:
                if self._pending_stt is not None:
                    stt_ms = self._pending_stt.get("stt_ms")
                    self._pending_stt = None
                stt_seconds = self._pending_audio
                self._pending_audio = None
            calls = self._pending_calls
            self._pending_calls = []
            turn = {
                "turn_index": len(self.turns) + 1,
                "customer": customer,
                "reply": reply,
                "stage": stage,
                "source": label_source(reply, stage, calls, card),
                "stt_ms": stt_ms,
                "stt_seconds": stt_seconds,
                "decision_ms": decision_ms,
                "llm_ms": llm_ms,
                "tts_ms": None,
                "model_calls": calls,
                "bulbul_characters": 0,
                "neerja_characters": 0,
                "translation_characters": 0,
            }
            self.turns.append(turn)
            self._awaiting_speech = turn
            self._speech_turn = turn
            return turn

    def note_stt(self, text: str, stt_ms: int) -> None:
        with self._lock:
            self._pending_stt = {"text": text, "stt_ms": stt_ms}

    def note_tts(self, tts_ms: int) -> bool:
        """Store the wait until the first clause can play. Later clauses stay out."""
        with self._lock:
            if self._awaiting_speech is None:
                return False
            self._awaiting_speech["tts_ms"] = tts_ms
            self._awaiting_speech = None
            return True

    def note_bulbul(self, characters: int) -> None:
        self._add_characters("bulbul_characters", characters)

    def note_neerja(self, characters: int) -> None:
        self._add_characters("neerja_characters", characters)

    def note_translate(self, characters: int) -> None:
        self._add_characters("translation_characters", characters)

    def _add_characters(self, key: str, characters: int) -> None:
        with self._lock:
            turn = self._speech_turn
            if turn is None and self.turns:
                turn = self.turns[-1]
            if turn is not None:
                turn[key] = turn.get(key, 0) + characters

    def usage(self) -> dict[str, Any]:
        with self._lock:
            calls = [call for turn in self.turns for call in turn.get("model_calls") or []]
            seconds = [
                turn.get("stt_seconds")
                for turn in self.turns
                if turn.get("customer") is not None
            ]
            if seconds and all(isinstance(value, (int, float)) for value in seconds):
                audio_complete = round(sum(seconds), 3)
            else:
                audio_complete = None
            bulbul = sum(turn.get("bulbul_characters") or 0 for turn in self.turns)
            neerja = sum(turn.get("neerja_characters") or 0 for turn in self.turns)
            translated = sum(turn.get("translation_characters") or 0 for turn in self.turns)
        model = language_model_inr(calls)
        speech = {"seconds": audio_complete, **priced(audio_complete, STT_INR_PER_SECOND)}
        voice = {"characters": bulbul, **priced(bulbul, BULBUL_INR_PER_CHARACTER)}
        translation = {
            "characters": translated,
            **priced(translated, TRANSLATE_INR_PER_CHARACTER),
        }
        parts = [model["inr_ex_gst"], speech["inr_ex_gst"], voice["inr_ex_gst"], translation["inr_ex_gst"]]
        priced_parts = [part for part in parts if part is not None]
        total = round(sum(priced_parts), 4) if priced_parts else None
        return {
            "rates_checked": RATES_CHECKED,
            "gst_rate": GST_RATE,
            "language_model": model,
            "speech_to_text": speech,
            "bulbul": voice,
            "neerja": {"characters": neerja},
            "translation": translation,
            "inr_ex_gst": total,
            "inr_with_gst": None if total is None else round(total * (1 + GST_RATE), 4),
            "priced_complete": all(part is not None for part in parts),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "started_at": self.started_at,
            "caller_model": CALLER_MODEL,
            "turns": self.turns,
            "usage": self.usage(),
            **({"callback_when": self.callback_when} if self.callback_when else {}),
        }

    def save(self, runs_dir: Path) -> Path:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{self.call_id}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        self.saved = True
        return path

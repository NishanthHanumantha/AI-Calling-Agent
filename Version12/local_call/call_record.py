"""Turns for the call selected in this task. This does not change the conversation."""

from __future__ import annotations

import contextvars
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CALLER_MODEL = "sarvam-105b-conversations"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class CallLog:
    """Collect turns for one call."""

    def __init__(self, call_id: str | None = None) -> None:
        self.call_id = call_id or _now()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.turns: list[dict[str, Any]] = []
        self._pending_stt: dict[str, Any] | None = None
        self._awaiting_speech: dict[str, Any] | None = None
        self.saved = False

    def begin(self) -> None:
        self.__init__()

    def add_reply(
        self,
        reply: str,
        stage: str,
        llm_ms: int | None,
        customer: str | None = None,
    ) -> dict[str, Any]:
        stt_ms = None
        if customer is not None and self._pending_stt is not None:
            stt_ms = self._pending_stt.get("stt_ms")
            self._pending_stt = None
        turn = {
            "turn_index": len(self.turns) + 1,
            "customer": customer,
            "reply": reply,
            "stage": stage,
            "stt_ms": stt_ms,
            "llm_ms": llm_ms,
            "tts_ms": None,
        }
        self.turns.append(turn)
        self._awaiting_speech = turn
        return turn

    def note_stt(self, text: str, stt_ms: int) -> None:
        self._pending_stt = {"text": text, "stt_ms": stt_ms}

    def note_tts(self, tts_ms: int) -> None:
        if self._awaiting_speech is not None:
            self._awaiting_speech["tts_ms"] = tts_ms
            self._awaiting_speech = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "started_at": self.started_at,
            "caller_model": CALLER_MODEL,
            "turns": self.turns,
        }

    def save(self, runs_dir: Path) -> Path:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{self.call_id}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        self.saved = True
        return path


_active_log: contextvars.ContextVar[CallLog | None] = contextvars.ContextVar(
    "sobha_call_log", default=None
)
_logs: dict[str, CallLog] = {}
_default_log = CallLog()


class ActiveCallLog:
    """Forwards each note to the log selected for this task.

    The browser and the tests leave the selection unset, so they keep
    using the one default log.
    """

    def _current(self) -> CallLog:
        current = _active_log.get()
        return current if current is not None else _default_log

    def __getattr__(self, name: str):
        return getattr(self._current(), name)


call_log = ActiveCallLog()


def bind_log(call_id: str):
    """Select this call's log. Reset the returned token when the turn ends."""
    log = _logs.get(call_id)
    if log is None:
        log = CallLog(call_id=call_id)
        _logs[call_id] = log
    return _active_log.set(log)


def release_log(token) -> None:
    _active_log.reset(token)


def log_for(call_id: str) -> CallLog | None:
    return _logs.get(call_id)

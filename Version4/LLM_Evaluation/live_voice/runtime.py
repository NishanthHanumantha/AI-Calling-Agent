"""In-process eval runtime. One candidate provider per CallSid. Not production memory."""

from __future__ import annotations

from typing import Any

from .session_store import SessionStore


class EvalRuntime:
    def __init__(
        self,
        store: SessionStore | None = None,
        providers: dict[str, Any] | None = None,
        system_prompt: str = "",
        retrieved_context: str = "",
    ):
        self.store = store or SessionStore()
        self.providers: dict[str, Any] = providers or {}
        self.system_prompt = system_prompt
        self.retrieved_context = retrieved_context


_RUNTIME: EvalRuntime | None = None


def set_runtime(runtime: EvalRuntime | None) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def get_runtime() -> EvalRuntime:
    if _RUNTIME is None:
        raise RuntimeError("Live-voice eval runtime is not started")
    return _RUNTIME

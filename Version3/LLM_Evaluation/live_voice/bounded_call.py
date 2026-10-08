"""Async supervisor for one live-voice child process.

Limitation: asyncio.create_subprocess_exec has no timeout on Windows or Linux.
This module does not wrap that await in asyncio.wait_for, because cancelling it
can leave an OS process with no Process handle to kill or track.

LLM_SECTION_BUDGET_SECONDS (8.0) is an intended budget set before spawn. It is
not a guaranteed HTTP response time, not a hard end-to-end deadline, and not a
guarantee that Twilio receives TwiML inside its webhook window.

After a Process handle exists and is stored in the slot, wait, kill, and reap
use only remaining time on that same monotonic deadline. No second timer is started.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, Awaitable, Callable

from .constants import FALLBACK_SPOKEN, LLM_SECTION_BUDGET_SECONDS

SpawnFn = Callable[..., Awaitable[Any]]


class ChildSlot:
    """One in-flight child for this process. Not an asyncio.Lock."""

    def __init__(self) -> None:
        self.held = False
        self.proc: Any = None
        self.cleanup_failed = False

    def note_exit_if_reaped(self) -> None:
        proc = self.proc
        if proc is not None and getattr(proc, "returncode", None) is not None:
            self.proc = None
            self.held = False
            self.cleanup_failed = False

    def reset(self) -> None:
        self.held = False
        self.proc = None
        self.cleanup_failed = False

    def try_acquire(self) -> str | None:
        self.note_exit_if_reaped()
        if self.held:
            proc = self.proc
            if proc is not None and getattr(proc, "returncode", None) is None:
                _kill(proc)
            return "IN_FLIGHT"
        self.held = True
        self.proc = None
        self.cleanup_failed = False
        return None

    def release_if_exited(self) -> bool:
        proc = self.proc
        if proc is None or getattr(proc, "returncode", None) is not None:
            self.proc = None
            self.held = False
            self.cleanup_failed = False
            return True
        self.cleanup_failed = True
        self.held = True
        return False


SLOT = ChildSlot()


def loop_supports_subprocess() -> bool:
    if sys.platform != "win32":
        return True
    loop = asyncio.get_running_loop()
    return isinstance(loop, asyncio.ProactorEventLoop)


def _kill(proc: Any) -> None:
    if getattr(proc, "returncode", None) is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except OSError:
        pass


def _result(
    *,
    started: float,
    budget: float,
    spoken: str,
    fallback_used: bool,
    error_type: str | None,
    schema_valid: bool | None,
    intent: str | None,
    action: str | None,
    spawn_overrun: bool,
    error_message: str | None = None,
) -> dict[str, Any]:
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {
        "spoken": spoken,
        "fallback_used": fallback_used,
        "latency_ms": elapsed_ms,
        "error_type": error_type,
        "error_class": None if not fallback_used else "NETWORK",
        "schema_valid": schema_valid,
        "candidate_declared_intent": intent,
        "candidate_declared_action": action,
        "spawn_overrun": spawn_overrun,
        "deadline_exceeded": elapsed_ms > int(budget * 1000),
        "error_message": error_message,
        "slot_held": SLOT.held,
    }


def _fallback(
    *,
    started: float,
    budget: float,
    error_type: str,
    spawn_overrun: bool,
    error_message: str | None = None,
) -> dict[str, Any]:
    return _result(
        started=started,
        budget=budget,
        spoken=FALLBACK_SPOKEN,
        fallback_used=True,
        error_type=error_type,
        schema_valid=False,
        intent=None,
        action=None,
        spawn_overrun=spawn_overrun,
        error_message=error_message,
    )


async def _reap_until(proc: Any, deadline: float) -> bool:
    """Wait only until the original deadline. Never starts a new timer past it."""
    if getattr(proc, "returncode", None) is not None:
        return True
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return getattr(proc, "returncode", None) is not None
    try:
        await asyncio.wait_for(proc.wait(), timeout=remaining)
    except TimeoutError:
        return getattr(proc, "returncode", None) is not None
    return getattr(proc, "returncode", None) is not None


async def _reject_late_process(proc: Any, deadline: float, started: float, budget: float) -> dict[str, Any]:
    _kill(proc)
    confirmed = await _reap_until(proc, deadline)
    if confirmed:
        SLOT.release_if_exited()
        return _fallback(
            started=started,
            budget=budget,
            error_type="TIMEOUT",
            spawn_overrun=True,
            error_message="process handle arrived after the intended deadline",
        )
    SLOT.held = True
    SLOT.cleanup_failed = True
    return _fallback(
        started=started,
        budget=budget,
        error_type="CLEANUP_FAILED",
        spawn_overrun=True,
        error_message="late child could not be confirmed exited",
    )


def _accept_answer(payload: dict[str, Any], *, started: float, budget: float) -> dict[str, Any] | None:
    if payload.get("error_type") or payload.get("error"):
        return None
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        return None
    return _result(
        started=started,
        budget=budget,
        spoken=answer,
        fallback_used=False,
        error_type=None,
        schema_valid=payload.get("schema_valid"),
        intent=payload.get("intent"),
        action=payload.get("action"),
        spawn_overrun=False,
    )


async def run_bounded_turn(
    job: dict[str, Any],
    *,
    budget_seconds: float | None = None,
    spawn: SpawnFn | None = None,
    force_subprocess_support: bool | None = None,
) -> dict[str, Any]:
    """Run one child. Deadline is fixed at entry, before create_subprocess_exec."""
    budget = LLM_SECTION_BUDGET_SECONDS if budget_seconds is None else budget_seconds
    started = time.monotonic()
    deadline = started + budget

    if SLOT.try_acquire() == "IN_FLIGHT":
        return _fallback(
            started=started,
            budget=budget,
            error_type="IN_FLIGHT",
            spawn_overrun=False,
            error_message="another child is still tracked",
        )

    supported = loop_supports_subprocess() if force_subprocess_support is None else force_subprocess_support
    if not supported:
        SLOT.held = False
        SLOT.proc = None
        return _fallback(
            started=started,
            budget=budget,
            error_type="SUPERVISOR_UNSUPPORTED",
            spawn_overrun=False,
            error_message="this event loop cannot spawn subprocesses",
        )

    if time.monotonic() >= deadline:
        SLOT.held = False
        return _fallback(
            started=started,
            budget=budget,
            error_type="TIMEOUT",
            spawn_overrun=False,
            error_message="intended deadline already exhausted before spawn",
        )

    spawn_fn = spawn or _default_spawn
    payload = json.dumps(job).encode("utf-8")
    try:
        # Do not wrap this await in wait_for. Cancellation can orphan a child.
        proc = await spawn_fn(payload)
    except Exception as exc:
        SLOT.held = False
        SLOT.proc = None
        return _fallback(
            started=started,
            budget=budget,
            error_type="SPAWN_FAILED",
            spawn_overrun=False,
            error_message=type(exc).__name__,
        )

    SLOT.proc = proc
    if time.monotonic() > deadline:
        return await _reject_late_process(proc, deadline, started, budget)

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return await _reject_late_process(proc, deadline, started, budget)

    communicate = asyncio.create_task(proc.communicate(payload))
    done, pending = await asyncio.wait({communicate}, timeout=remaining)
    if pending or time.monotonic() > deadline:
        _kill(proc)
        confirmed = await _reap_until(proc, deadline)
        if not communicate.done():
            communicate.cancel()
        if confirmed:
            SLOT.release_if_exited()
            return _fallback(
                started=started,
                budget=budget,
                error_type="TIMEOUT",
                spawn_overrun=False,
                error_message="child still running at the intended deadline",
            )
        SLOT.held = True
        SLOT.cleanup_failed = True
        return _fallback(
            started=started,
            budget=budget,
            error_type="CLEANUP_FAILED",
            spawn_overrun=False,
            error_message="child exit was not confirmed",
        )

    stdout, _stderr = communicate.result()
    if time.monotonic() > deadline or getattr(proc, "returncode", None) != 0:
        _kill(proc)
        confirmed = await _reap_until(proc, deadline)
        if confirmed:
            SLOT.release_if_exited()
        else:
            SLOT.held = True
            SLOT.cleanup_failed = True
        return _fallback(
            started=started,
            budget=budget,
            error_type="TIMEOUT" if confirmed else "CLEANUP_FAILED",
            spawn_overrun=False,
            error_message="child output was not accepted",
        )

    try:
        parsed = json.loads((stdout or b"{}").decode("utf-8"))
    except json.JSONDecodeError:
        SLOT.release_if_exited()
        return _fallback(
            started=started,
            budget=budget,
            error_type="SPAWN_FAILED",
            spawn_overrun=False,
            error_message="child stdout was not JSON",
        )

    if time.monotonic() > deadline:
        SLOT.release_if_exited()
        return _fallback(
            started=started,
            budget=budget,
            error_type="TIMEOUT",
            spawn_overrun=True,
            error_message="result crossed the intended deadline",
        )

    accepted = _accept_answer(parsed, started=started, budget=budget)
    SLOT.release_if_exited()
    if accepted is not None:
        return accepted
    error_type = str(parsed.get("error_type") or parsed.get("error") or "API_ERROR")
    return _fallback(
        started=started,
        budget=budget,
        error_type=error_type,
        spawn_overrun=False,
        error_message=str(parsed.get("error_message") or "")[:500] or None,
    )


async def _default_spawn(payload: bytes) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "LLM_Evaluation.live_voice.llm_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

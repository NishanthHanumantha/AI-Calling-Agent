"""Spawn-overrun and post-spawn deadline tests. No Twilio calls."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

import pytest

from live_voice.bounded_call import SLOT, run_bounded_turn
from live_voice.constants import FALLBACK_SPOKEN, LLM_SECTION_BUDGET_SECONDS
from models.http_util import request_with_retry

REPO = Path(__file__).resolve().parents[2]


class FakeProc:
    def __init__(self, *, stdout: bytes = b"", die_on_kill: bool = True, hold_seconds: float = 0.0):
        self.returncode = None
        self.killed = False
        self.stdout = stdout
        self.die_on_kill = die_on_kill
        self.hold_seconds = hold_seconds

    def kill(self):
        self.killed = True
        if self.die_on_kill:
            self.returncode = -9

    async def wait(self):
        if self.returncode is None:
            await asyncio.sleep(30)
        return self.returncode

    async def communicate(self, input=None):
        if self.hold_seconds:
            await asyncio.sleep(self.hold_seconds)
        if self.returncode is None:
            self.returncode = 0
        return self.stdout, b""


@pytest.fixture(autouse=True)
def _clear_slot():
    SLOT.reset()
    yield
    proc = SLOT.proc
    if proc is not None and getattr(proc, "returncode", None) is None:
        try:
            proc.kill()
        except OSError:
            pass
    SLOT.reset()


def test_budget_constant_is_eight_seconds():
    assert LLM_SECTION_BUDGET_SECONDS == 8.0


def test_spawn_is_not_wrapped_in_wait_for():
    source = inspect.getsource(run_bounded_turn)
    spawn_line = next(line for line in source.splitlines() if "spawn_fn(" in line)
    assert "wait_for" not in spawn_line
    assert source.count("deadline = started + budget") == 1


def test_spawn_returns_after_deadline_discards_output():
    seen = {}

    async def late_spawn(_payload):
        seen["entered"] = True
        await asyncio.sleep(0.05)
        proc = FakeProc(stdout=b'{"answer":"late secret","error_type":null}', die_on_kill=True)
        seen["proc"] = proc
        return proc

    result = asyncio.run(
        run_bounded_turn({"alias": "sarvam_conversational"}, budget_seconds=0.01, spawn=late_spawn)
    )
    assert seen["entered"] is True
    assert seen["proc"].killed is True
    assert result["spoken"] == FALLBACK_SPOKEN
    assert result["fallback_used"] is True
    assert result["spawn_overrun"] is True
    assert result["candidate_declared_intent"] is None
    assert result["candidate_declared_action"] is None
    assert result["schema_valid"] is False
    assert "late secret" not in result["spoken"]
    assert result["latency_ms"] >= 10


def test_spawn_exception_does_not_occupy_slot():
    async def boom(_payload):
        raise RuntimeError("create failed")

    result = asyncio.run(run_bounded_turn({}, budget_seconds=1.0, spawn=boom))
    assert result["error_type"] == "SPAWN_FAILED"
    assert result["fallback_used"] is True
    assert result["spawn_overrun"] is False
    assert SLOT.held is False
    assert SLOT.proc is None


def test_late_spawn_cleanup_success_frees_slot():
    async def late_spawn(_payload):
        await asyncio.sleep(0.03)
        return FakeProc(die_on_kill=True)

    first = asyncio.run(run_bounded_turn({}, budget_seconds=0.01, spawn=late_spawn))
    assert first["spawn_overrun"] is True
    assert first["error_type"] == "TIMEOUT"
    assert SLOT.held is False

    async def quick(_payload):
        body = json.dumps({"answer": "Near Electronic City.", "intent": "LOCATION_QUERY", "action": "ANSWER", "schema_valid": True}).encode()
        return FakeProc(stdout=body, die_on_kill=True)

    second = asyncio.run(run_bounded_turn({}, budget_seconds=2.0, spawn=quick))
    assert second["fallback_used"] is False
    assert second["spoken"] == "Near Electronic City."


def test_late_spawn_cleanup_failure_blocks_next_child():
    spawned = {"count": 0}

    async def late_alive(_payload):
        spawned["count"] += 1
        await asyncio.sleep(0.03)
        return FakeProc(die_on_kill=False)

    first = asyncio.run(run_bounded_turn({}, budget_seconds=0.01, spawn=late_alive))
    assert first["error_type"] == "CLEANUP_FAILED"
    assert first["spawn_overrun"] is True
    assert SLOT.held is True

    async def should_not_run(_payload):
        spawned["count"] += 1
        return FakeProc()

    second = asyncio.run(run_bounded_turn({}, budget_seconds=1.0, spawn=should_not_run))
    assert second["error_type"] == "IN_FLIGHT"
    assert spawned["count"] == 1
    assert second["fallback_used"] is True


def test_prompt_success_before_deadline():
    async def quick(_payload):
        body = json.dumps(
            {"answer": "Townpark is near Electronic City.", "intent": "LOCATION_QUERY", "action": "ANSWER", "schema_valid": True}
        ).encode()
        return FakeProc(stdout=body)

    result = asyncio.run(run_bounded_turn({}, budget_seconds=2.0, spawn=quick))
    assert result["fallback_used"] is False
    assert result["spawn_overrun"] is False
    assert result["error_type"] is None
    assert result["candidate_declared_intent"] == "LOCATION_QUERY"


def test_provider_error_with_answer_text_is_not_spoken():
    async def quick(_payload):
        body = json.dumps(
            {
                "answer": "Invented price of 9 crore.",
                "intent": "PRICE_QUERY",
                "action": "ANSWER",
                "error_type": "API_ERROR",
                "schema_valid": False,
            }
        ).encode()
        return FakeProc(stdout=body)

    result = asyncio.run(run_bounded_turn({}, budget_seconds=2.0, spawn=quick))
    assert result["spoken"] == FALLBACK_SPOKEN
    assert "9 crore" not in result["spoken"]
    assert result["candidate_declared_intent"] is None
    assert result["candidate_declared_action"] is None
    assert result["error_type"] == "API_ERROR"
    assert result["spawn_overrun"] is False


def test_child_timeout_kills_process():
    async def hang(_payload):
        return FakeProc(hold_seconds=30, die_on_kill=True)

    result = asyncio.run(run_bounded_turn({}, budget_seconds=0.05, spawn=hang))
    assert result["error_type"] in {"TIMEOUT", "CLEANUP_FAILED"}
    assert result["fallback_used"] is True
    assert result["spawn_overrun"] is False


def test_concurrent_request_is_in_flight_without_second_spawn():
    started = {"n": 0}

    async def slow(_payload):
        started["n"] += 1
        return FakeProc(hold_seconds=0.2, die_on_kill=True)

    async def both():
        return await asyncio.gather(
            run_bounded_turn({}, budget_seconds=1.0, spawn=slow),
            run_bounded_turn({}, budget_seconds=1.0, spawn=slow),
        )

    first, second = asyncio.run(both())
    statuses = {first["error_type"], second["error_type"]}
    assert "IN_FLIGHT" in statuses
    assert started["n"] == 1


def test_repeated_timeouts_do_not_overlap_children():
    live = {"now": 0, "max": 0}

    async def hang(_payload):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        proc = FakeProc(hold_seconds=30, die_on_kill=True)
        original_kill = proc.kill

        def kill_and_drop():
            original_kill()
            live["now"] -= 1

        proc.kill = kill_and_drop
        return proc

    async def five():
        results = []
        for _ in range(5):
            results.append(await run_bounded_turn({}, budget_seconds=0.05, spawn=hang))
        return results

    results = asyncio.run(five())
    assert live["max"] == 1
    assert all(item["fallback_used"] for item in results)
    assert SLOT.held is False or results[-1]["error_type"] == "CLEANUP_FAILED"


def test_selector_loop_does_not_spawn():
    async def explode(_payload):
        raise AssertionError("must not spawn")

    result = asyncio.run(
        run_bounded_turn({}, budget_seconds=1.0, spawn=explode, force_subprocess_support=False)
    )
    assert result["error_type"] == "SUPERVISOR_UNSUPPORTED"
    assert SLOT.held is False


def test_proactor_loop_can_construct_supervisor_on_windows():
    if sys.platform != "win32":
        pytest.skip("proactor check is for the Windows development host")

    async def check():
        loop = asyncio.get_running_loop()
        assert isinstance(loop, asyncio.ProactorEventLoop)

    asyncio.run(check())


def test_offline_retry_defaults_unchanged():
    source = inspect.getsource(request_with_retry)
    assert "max_retries: int = 2" in source
    assert "timeout: int = 30" in source


def test_production_files_unchanged():
    app = (REPO / "app_v3.py").read_text(encoding="utf-8")
    make_call = (REPO / "make_call.py").read_text(encoding="utf-8")
    assert 'MODEL = "sarvam-105b-conversations"' in app
    assert "bounded_call" not in app
    assert "/eval/" not in make_call

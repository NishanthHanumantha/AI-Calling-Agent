"""CallSid-scoped live-voice handlers. No FastAPI dependency."""

from __future__ import annotations

from typing import Any

from .bounded_call import run_bounded_turn
from .constants import DONE_PHRASES, FIXED_OPENING, MAX_EMPTY_RETRIES, MAX_TURNS
from .runtime import get_runtime
from .session_store import utc_now
from .twiml import clean_spoken, gather_block, gather_twiml, hangup_twiml, twiml_response


def start_call(query: dict[str, str], form: dict[str, Any]) -> str:
    runtime = get_runtime()
    run_id = (query.get("run_id") or form.get("run_id") or "").strip()
    token = (query.get("token") or form.get("token") or "").strip()
    call_sid = str(form.get("CallSid") or "").strip()
    if not call_sid:
        return hangup_twiml("An application error occurred. Goodbye.")
    binding = runtime.store.consume_pending(run_id, token)
    if binding is None:
        return hangup_twiml("This evaluation session is not active. Goodbye.")
    session = runtime.store.start_session(call_sid, binding)
    runtime.store.append_turn(
        call_sid,
        {
            "turn_index": 1,
            "outbound_turn_index": 1,
            "speaker": "agent",
            "text": FIXED_OPENING,
            "raw_stt": None,
            "corrected_text": None,
            "timestamp": utc_now(),
            "latency_ms": None,
            "candidate_declared_intent": None,
            "candidate_declared_action": None,
            "schema_valid": None,
            "error_class": None,
            "error_message": None,
        },
    )
    session["history"] = []
    return gather_twiml(FIXED_OPENING)


def _customer_done(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in DONE_PHRASES)


async def handle_speech(form: dict[str, Any]) -> str:
    runtime = get_runtime()
    call_sid = str(form.get("CallSid") or "").strip()
    session = runtime.store.get_by_call_sid(call_sid)
    if session is None:
        return hangup_twiml("This evaluation session is not active. Goodbye.")
    if session.get("closed"):
        return hangup_twiml("Thank you. Goodbye.")

    raw = str(form.get("SpeechResult") or "").strip()
    if not raw:
        raw = str(form.get("UnstableSpeechResult") or "").strip()
    corrected = raw

    if not raw:
        session["empty_retries"] = int(session.get("empty_retries") or 0) + 1
        if session["empty_retries"] >= MAX_EMPTY_RETRIES:
            runtime.store.append_turn(
                call_sid,
                {
                    "turn_index": len(session["turns"]) + 1,
                    "outbound_turn_index": None,
                    "speaker": "system",
                    "text": "empty speech hangup",
                    "raw_stt": "",
                    "corrected_text": "",
                    "timestamp": utc_now(),
                    "latency_ms": None,
                    "candidate_declared_intent": None,
                    "candidate_declared_action": None,
                    "schema_valid": None,
                    "error_class": "STT",
                    "error_message": "empty SpeechResult retries exhausted",
                },
            )
            session["closed"] = True
            return hangup_twiml(
                "I'm having a little trouble hearing you clearly. Thank you for your time. Have a great day."
            )
        return gather_twiml("Sorry, I couldn't hear you clearly. Could you please repeat that?")

    session["empty_retries"] = 0
    customer_turns = [t for t in session["turns"] if t.get("speaker") == "customer"]
    outbound_turn_index = 2 + len(customer_turns)
    runtime.store.append_turn(
        call_sid,
        {
            "turn_index": len(session["turns"]) + 1,
            "outbound_turn_index": outbound_turn_index,
            "speaker": "customer",
            "text": corrected,
            "raw_stt": raw,
            "corrected_text": corrected,
            "timestamp": utc_now(),
            "latency_ms": None,
            "candidate_declared_intent": None,
            "candidate_declared_action": None,
            "schema_valid": None,
            "error_class": None,
            "error_message": None,
        },
    )

    hang_after = _customer_done(corrected) or len(session["turns"]) >= MAX_TURNS
    runner = getattr(runtime, "turn_runner", None)
    job = {
        "alias": session["alias"],
        "utterance": corrected,
        "history": list(session["history"]),
        "system_prompt": runtime.system_prompt,
        "retrieved_context": runtime.retrieved_context,
    }
    if runner is not None:
        outcome = await runner(job)
    else:
        outcome = await run_bounded_turn(job)
    spoken = clean_spoken(outcome.get("spoken")) or outcome.get("spoken") or ""
    declared_intent = outcome.get("candidate_declared_intent")
    declared_action = outcome.get("candidate_declared_action")
    schema_valid = outcome.get("schema_valid")
    latency_ms = outcome.get("latency_ms")
    error_class = outcome.get("error_class")
    error_message = outcome.get("error_message")

    runtime.store.append_turn(
        call_sid,
        {
            "turn_index": len(session["turns"]) + 1,
            "outbound_turn_index": None,
            "speaker": "agent",
            "text": spoken,
            "raw_stt": None,
            "corrected_text": None,
            "timestamp": utc_now(),
            "latency_ms": latency_ms,
            "candidate_declared_intent": declared_intent,
            "candidate_declared_action": declared_action,
            "schema_valid": schema_valid,
            "error_class": error_class,
            "error_message": error_message,
            "fallback_used": outcome.get("fallback_used"),
            "error_type": outcome.get("error_type"),
            "spawn_overrun": outcome.get("spawn_overrun"),
            "deadline_exceeded": outcome.get("deadline_exceeded"),
        },
    )
    session["history"].append({"role": "user", "content": corrected})
    session["history"].append({"role": "assistant", "content": spoken})

    if hang_after:
        session["closed"] = True
        return hangup_twiml(spoken)

    return twiml_response(gather_block(spoken))


def handle_recording(form: dict[str, Any]) -> dict[str, Any]:
    runtime = get_runtime()
    call_sid = str(form.get("CallSid") or "").strip()
    session = runtime.store.get_by_call_sid(call_sid)
    if session is None:
        return {"ok": False, "reason": "unknown CallSid"}
    status = str(form.get("RecordingStatus") or form.get("RecordingStatusCallbackEvent") or "").strip()
    recording_sid = str(form.get("RecordingSid") or "").strip() or None
    recording_url = str(form.get("RecordingUrl") or "").strip() or None
    if not recording_sid and status.lower() in {"absent", "failed", "canceled", "cancelled", ""}:
        ref = runtime.store.attach_recording(
            call_sid,
            recording_sid=None,
            recording_url=None,
            recording_status="UNAVAILABLE",
        )
        return {"ok": True, "recording_status": "UNAVAILABLE", "ref": ref}
    ref = runtime.store.attach_recording(
        call_sid,
        recording_sid=recording_sid,
        recording_url=recording_url,
        recording_status=status or "completed",
    )
    return {"ok": True, "ref": ref}

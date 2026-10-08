"""Sequential Twilio live-voice controller. One model per call. No automatic retry."""

from __future__ import annotations

import os
import secrets
import time
from typing import Any, Callable
from urllib.parse import urlencode

from .bootstrap import bootstrap
from .candidates import (
    build_provider_for_candidate,
    calling_system_prompt,
    load_live_config,
    project_context,
    resolve_candidates,
)
from .constants import (
    CANDIDATE_ORDER,
    CONFIRMATION_PHRASE,
    DEFAULT_CALL_TIMEOUT_SECONDS,
    DEFAULT_POLL_SECONDS,
    TERMINAL_CALL_STATUSES,
)
from .protocol import operator_script_text
from .runtime import EvalRuntime, set_runtime
from .session_store import SessionStore, redact_value, utc_now

bootstrap()
from evaluator.runner import new_run_id

PromptFn = Callable[[str], str]


def default_twilio_factory(account_sid: str, auth_token: str):
    from twilio.rest import Client

    return Client(account_sid, auth_token)


def is_terminal_status(status: str | None) -> bool:
    return str(status or "").strip().lower() in TERMINAL_CALL_STATUSES


def confirm_live(prompt_fn: PromptFn) -> bool:
    answer = (prompt_fn(f"Type {CONFIRMATION_PHRASE} to place real Twilio calls: ") or "").strip()
    return answer == CONFIRMATION_PHRASE


def after_call_choice(prompt_fn: PromptFn) -> str:
    raw = (prompt_fn("Next model: Continue / Skip / Abort: ") or "").strip().lower()
    if raw in {"continue", "c", "yes", "y"}:
        return "continue"
    if raw in {"skip", "s"}:
        return "skip"
    if raw in {"abort", "a", "quit", "q"}:
        return "abort"
    return "abort"


def poll_until_terminal(
    fetch_status: Callable[[str], dict[str, Any]],
    call_sid: str,
    *,
    timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {"sid": call_sid, "status": "queued", "duration": None}
    while time.monotonic() < deadline:
        last = fetch_status(call_sid)
        if is_terminal_status(last.get("status")):
            return last
        sleep_fn(poll_seconds)
    last = dict(last)
    last["status"] = "timeout"
    last["timed_out"] = True
    return last


def public_eval_url(public_url: str, path: str, params: dict[str, str]) -> str:
    base = public_url.rstrip("/") + path
    return base + "?" + urlencode(params)


def plan_text(candidates: list[dict[str, Any]], *, public_url: str, dry_run: bool) -> str:
    lines = [
        "FOUR-MODEL LIVE VOICE EVALUATION",
        "MODE: DRY-RUN (zero Twilio calls)" if dry_run else "MODE: LIVE (requires confirmation)",
        f"Eval service bind: 127.0.0.1:8001",
        f"Webhook base: {public_url or '(PUBLIC_URL unset)'}/eval/",
        "Production /voice is not used.",
        "",
        "Resolved candidates:",
    ]
    for index, item in enumerate(candidates, start=1):
        avail = "AVAILABLE" if item.get("available") else "UNAVAILABLE"
        err = f" ({item.get('error')})" if item.get("error") else ""
        lines.append(
            f"  {index}. {item['alias']}  provider={item['provider']}  "
            f"model_id={item.get('model_id') or '(none)'}  {avail}{err}"
        )
    lines.extend(["", operator_script_text()])
    return "\n".join(lines)


def _fetch_from_client(client: Any, call_sid: str) -> dict[str, Any]:
    call = client.calls(call_sid).fetch()
    return {
        "sid": getattr(call, "sid", call_sid),
        "status": getattr(call, "status", None),
        "duration": getattr(call, "duration", None),
    }


def _create_eval_call(
    client: Any,
    *,
    to_number: str,
    from_number: str,
    voice_url: str,
    recording_url: str,
    recording_channels: str | None = "dual",
) -> Any:
    kwargs: dict[str, Any] = {
        "to": to_number,
        "from_": from_number,
        "url": voice_url,
        "record": True,
        "recording_status_callback": recording_url,
        "recording_status_callback_method": "POST",
    }
    if recording_channels:
        kwargs["recording_channels"] = recording_channels
    try:
        return client.calls.create(**kwargs)
    except Exception:
        if not recording_channels:
            raise
        kwargs.pop("recording_channels", None)
        return client.calls.create(**kwargs)


def run_campaign(
    *,
    live: bool,
    prompt_fn: PromptFn | None = None,
    twilio_factory: Callable[[str, str], Any] | None = None,
    fetch_status: Callable[[str], dict[str, Any]] | None = None,
    store: SessionStore | None = None,
    providers: dict[str, Any] | None = None,
    campaign_id: str | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    call_timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    config: dict[str, Any] | None = None,
    public_url: str | None = None,
    stdout_write: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run dry-run or sequential live campaign. Never auto-retries a call."""

    log = stdout_write or (lambda text: print(text))
    cfg = config or load_live_config()
    candidates = resolve_candidates(cfg)
    public = (public_url or os.getenv("PUBLIC_URL") or "").rstrip("/")
    log(plan_text(candidates, public_url=public, dry_run=not live))

    if not live:
        return {
            "mode": "dry-run",
            "campaign_id": None,
            "calls_created": 0,
            "results": [],
            "candidates": [
                {
                    "alias": item["alias"],
                    "provider": item["provider"],
                    "model_id": item["model_id"],
                    "available": item["available"],
                }
                for item in candidates
            ],
        }

    prompt_fn = prompt_fn or input
    if not confirm_live(prompt_fn):
        log("Confirmation not given. Zero Twilio calls placed.")
        return {"mode": "live-aborted", "campaign_id": None, "calls_created": 0, "results": [], "reason": "no_confirmation"}

    account_sid = (os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
    auth_token = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    from_number = (os.getenv("TWILIO_PHONE_NUMBER") or "").strip()
    to_number = (os.getenv("TWILIO_TO_NUMBER") or "").strip()
    missing = [
        name
        for name, value in (
            ("TWILIO_ACCOUNT_SID", account_sid),
            ("TWILIO_AUTH_TOKEN", auth_token),
            ("TWILIO_PHONE_NUMBER", from_number),
            ("TWILIO_TO_NUMBER", to_number),
            ("PUBLIC_URL", public),
        )
        if not value
    ]
    if missing:
        raise ValueError("Missing required environment variable(s): " + ", ".join(missing))

    store = store or SessionStore()
    built_providers = dict(providers or {})
    for item in candidates:
        if item["alias"] not in built_providers:
            built_providers[item["alias"]] = build_provider_for_candidate(item, dry_run=False)
    set_runtime(
        EvalRuntime(
            store=store,
            providers=built_providers,
            system_prompt=calling_system_prompt(),
            retrieved_context=project_context(),
        )
    )

    factory = twilio_factory or default_twilio_factory
    client = factory(account_sid, auth_token)
    campaign = campaign_id or ("lv_" + new_run_id())
    results: list[dict[str, Any]] = []
    calls_created = 0
    skip_next = False
    abort_rest = False

    for item in candidates:
        alias = item["alias"]
        run_id = new_run_id() + "_" + alias
        if abort_rest:
            store.write_skipped(
                campaign_id=campaign,
                evaluation_run_id=run_id,
                alias=alias,
                provider=item["provider"],
                model_id=item["model_id"],
                reason="ABORTED_BY_OPERATOR",
            )
            results.append({"alias": alias, "status": "ABORTED_BY_OPERATOR", "calls_created": False})
            continue
        if skip_next:
            skip_next = False
            store.write_skipped(
                campaign_id=campaign,
                evaluation_run_id=run_id,
                alias=alias,
                provider=item["provider"],
                model_id=item["model_id"],
                reason="SKIPPED_BY_OPERATOR",
            )
            results.append({"alias": alias, "status": "SKIPPED_BY_OPERATOR", "calls_created": False})
            if alias != CANDIDATE_ORDER[-1]:
                choice = after_call_choice(prompt_fn)
                if choice == "abort":
                    abort_rest = True
                elif choice == "skip":
                    skip_next = True
            continue

        if not item.get("available") and built_providers.get(alias) is None:
            store.write_skipped(
                campaign_id=campaign,
                evaluation_run_id=run_id,
                alias=alias,
                provider=item["provider"],
                model_id=item["model_id"],
                reason="PROVIDER_UNAVAILABLE",
            )
            results.append({"alias": alias, "status": "PROVIDER_UNAVAILABLE", "calls_created": False})
            if alias != CANDIDATE_ORDER[-1]:
                choice = after_call_choice(prompt_fn)
                if choice == "abort":
                    abort_rest = True
                elif choice == "skip":
                    skip_next = True
            continue

        token = secrets.token_urlsafe(16)
        binding = {
            "campaign_id": campaign,
            "evaluation_run_id": run_id,
            "alias": alias,
            "provider": item["provider"],
            "model_id": item["model_id"],
            "token": token,
            "consumed": False,
        }
        store.register_pending(binding)
        params = {"run_id": run_id, "token": token}
        voice_url = public_eval_url(public, "/eval/voice", params)
        recording_url = public_eval_url(public, "/eval/recording", params)
        log(f"Placing eval call for {alias} model_id={item['model_id']} run_id={run_id}")
        try:
            call = _create_eval_call(
                client,
                to_number=to_number,
                from_number=from_number,
                voice_url=voice_url,
                recording_url=recording_url,
            )
        except Exception as exc:
            store.write_skipped(
                campaign_id=campaign,
                evaluation_run_id=run_id,
                alias=alias,
                provider=item["provider"],
                model_id=item["model_id"],
                reason="CREATE_FAILED",
            )
            results.append(
                {
                    "alias": alias,
                    "status": "CREATE_FAILED",
                    "error": redact_value(exc),
                    "calls_created": False,
                    "evaluation_run_id": run_id,
                }
            )
            log(f"Call create failed for {alias}: {redact_value(exc)}")
            log("No automatic retry.")
            if alias != CANDIDATE_ORDER[-1]:
                choice = after_call_choice(prompt_fn)
                if choice == "abort":
                    abort_rest = True
                elif choice == "skip":
                    skip_next = True
            continue

        calls_created += 1
        call_sid = getattr(call, "sid", None) or str(call)
        fetcher = fetch_status or (lambda sid, _client=client: _fetch_from_client(_client, sid))
        snapshot = poll_until_terminal(
            fetcher,
            call_sid,
            timeout_seconds=call_timeout_seconds,
            poll_seconds=poll_seconds,
            sleep_fn=sleep_fn,
        )
        status = str(snapshot.get("status") or "unknown")
        incomplete = status not in {"completed"}
        session = store.get_by_call_sid(call_sid)
        if session is not None:
            rec_status = None
            if incomplete or status == "timeout":
                store.mark_unavailable_recording(call_sid)
            store.finalize(
                call_sid,
                call_status=status,
                duration_seconds=snapshot.get("duration"),
                error="call timed out" if snapshot.get("timed_out") else None,
                incomplete=incomplete,
            )
        else:
            store.write_skipped(
                campaign_id=campaign,
                evaluation_run_id=run_id,
                alias=alias,
                provider=item["provider"],
                model_id=item["model_id"],
                reason=f"NO_WEBHOOK_{status}",
            )
        outcome = {
            "alias": alias,
            "provider": item["provider"],
            "model_id": item["model_id"],
            "evaluation_run_id": run_id,
            "campaign_id": campaign,
            "call_sid": call_sid,
            "status": status,
            "duration": snapshot.get("duration"),
            "incomplete": incomplete,
            "calls_created": True,
            "ended_at": utc_now(),
        }
        results.append(outcome)
        log(
            f"Outcome {alias}: status={status} CallSid={call_sid} "
            f"duration={snapshot.get('duration')} incomplete={incomplete}"
        )
        log("No automatic retry.")
        if alias != CANDIDATE_ORDER[-1]:
            choice = after_call_choice(prompt_fn)
            if choice == "abort":
                abort_rest = True
            elif choice == "skip":
                skip_next = True

    return {
        "mode": "live",
        "campaign_id": campaign,
        "calls_created": calls_created,
        "results": results,
        "to_number": "[REDACTED_PHONE]",
        "from_number": "[REDACTED_PHONE]",
    }

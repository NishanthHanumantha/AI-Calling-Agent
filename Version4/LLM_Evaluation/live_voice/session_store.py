"""Gitignored live-voice artifacts keyed by campaign, run, CallSid, alias, model_id."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap
from .constants import CANDIDATE_ORDER

PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")
SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+|api[_-]?key['\"]?\s*[:=]\s*['\"]?(?!NOT\b)[^'\"\s]+)",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_value(value: Any) -> str:
    text = "" if value is None else str(value)
    text = SECRET_RE.sub("[REDACTED]", text)
    return PHONE_RE.sub("[REDACTED_PHONE]", text)


def runs_root() -> Path:
    root = bootstrap()
    return root / "output" / "live_voice_runs"


def campaign_dir(campaign_id: str) -> Path:
    path = runs_root() / campaign_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(campaign_id: str, evaluation_run_id: str) -> Path:
    path = campaign_dir(campaign_id) / evaluation_run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def empty_manifest(
    *,
    campaign_id: str,
    evaluation_run_id: str,
    alias: str,
    provider: str,
    model_id: str,
) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "evaluation_run_id": evaluation_run_id,
        "alias": alias,
        "provider": provider,
        "model_id": model_id,
        "candidate_order": list(CANDIDATE_ORDER),
        "call_sid": None,
        "call_status": "pending",
        "duration_seconds": None,
        "started_at": None,
        "ended_at": None,
        "recording_sid": None,
        "recording_url": None,
        "recording_status": "NOT_YET",
        "to_number": "[REDACTED_PHONE]",
        "from_number": "[REDACTED_PHONE]",
        "errors": [],
        "operator_retry": False,
        "incomplete": False,
        "skip_reason": None,
    }


class SessionStore:
    """In-process CallSid isolation plus gitignored files."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or runs_root()
        self.by_call_sid: dict[str, dict[str, Any]] = {}
        self.by_run_id: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, dict[str, Any]] = {}

    def register_pending(self, binding: dict[str, Any]) -> None:
        run_id = binding["evaluation_run_id"]
        self.pending[run_id] = dict(binding)

    def consume_pending(self, run_id: str, token: str) -> dict[str, Any] | None:
        binding = self.pending.get(run_id)
        if not binding:
            return None
        if binding.get("token") != token:
            return None
        if binding.get("consumed"):
            return None
        binding["consumed"] = True
        return dict(binding)

    def start_session(self, call_sid: str, binding: dict[str, Any]) -> dict[str, Any]:
        session = {
            "campaign_id": binding["campaign_id"],
            "evaluation_run_id": binding["evaluation_run_id"],
            "alias": binding["alias"],
            "provider": binding["provider"],
            "model_id": binding["model_id"],
            "call_sid": call_sid,
            "history": [],
            "turns": [],
            "empty_retries": 0,
            "closed": False,
            "started_at": utc_now(),
        }
        if call_sid in self.by_call_sid:
            raise ValueError(f"CallSid already bound: {call_sid}")
        self.by_call_sid[call_sid] = session
        self.by_run_id[binding["evaluation_run_id"]] = session
        self._write_manifest(session, call_status="in-progress")
        return session

    def get_by_call_sid(self, call_sid: str) -> dict[str, Any] | None:
        return self.by_call_sid.get(call_sid)

    def get_by_run_id(self, run_id: str) -> dict[str, Any] | None:
        return self.by_run_id.get(run_id)

    def append_turn(self, call_sid: str, turn: dict[str, Any]) -> dict[str, Any]:
        session = self.by_call_sid[call_sid]
        turn = {
            "campaign_id": session["campaign_id"],
            "evaluation_run_id": session["evaluation_run_id"],
            "call_sid": call_sid,
            "alias": session["alias"],
            "model_id": session["model_id"],
            **turn,
        }
        session["turns"].append(turn)
        path = self._run_path(session) / "transcript.jsonl"
        append_jsonl(path, turn)
        return turn

    def attach_recording(
        self,
        call_sid: str,
        *,
        recording_sid: str | None,
        recording_url: str | None,
        recording_status: str,
    ) -> dict[str, Any] | None:
        session = self.by_call_sid.get(call_sid)
        if session is None:
            return None
        ref = {
            "campaign_id": session["campaign_id"],
            "evaluation_run_id": session["evaluation_run_id"],
            "call_sid": call_sid,
            "alias": session["alias"],
            "model_id": session["model_id"],
            "recording_sid": recording_sid,
            "recording_url": recording_url,
            "recording_status": recording_status,
        }
        write_json(self._run_path(session) / "recording.ref.json", ref)
        self._write_manifest(
            session,
            call_status=None,
            recording_sid=recording_sid,
            recording_url=recording_url,
            recording_status=recording_status,
        )
        return ref

    def mark_unavailable_recording(self, call_sid: str) -> None:
        self.attach_recording(
            call_sid,
            recording_sid=None,
            recording_url=None,
            recording_status="UNAVAILABLE",
        )

    def finalize(
        self,
        call_sid: str,
        *,
        call_status: str,
        duration_seconds: Any = None,
        error: str | None = None,
        incomplete: bool = False,
    ) -> dict[str, Any]:
        session = self.by_call_sid[call_sid]
        session["closed"] = True
        extra_errors = list(self._manifest(session).get("errors") or [])
        if error:
            extra_errors.append(redact_value(error))
        recording_status = self._manifest(session).get("recording_status") or "NOT_YET"
        if recording_status in {"NOT_YET", None}:
            recording_status = "UNAVAILABLE"
            self.mark_unavailable_recording(call_sid)
        return self._write_manifest(
            session,
            call_status=call_status,
            duration_seconds=duration_seconds,
            ended_at=utc_now(),
            errors=extra_errors,
            incomplete=incomplete,
            recording_status=recording_status,
        )

    def write_skipped(
        self,
        *,
        campaign_id: str,
        evaluation_run_id: str,
        alias: str,
        provider: str,
        model_id: str,
        reason: str,
    ) -> dict[str, Any]:
        manifest = empty_manifest(
            campaign_id=campaign_id,
            evaluation_run_id=evaluation_run_id,
            alias=alias,
            provider=provider,
            model_id=model_id,
        )
        manifest["call_status"] = reason
        manifest["incomplete"] = True
        manifest["skip_reason"] = reason
        write_json(
            Path(self.base_dir) / campaign_id / evaluation_run_id / "manifest.json",
            manifest,
        )
        return manifest

    def _run_path(self, session: dict[str, Any]) -> Path:
        path = Path(self.base_dir) / session["campaign_id"] / session["evaluation_run_id"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _manifest_path(self, session: dict[str, Any]) -> Path:
        return self._run_path(session) / "manifest.json"

    def _manifest(self, session: dict[str, Any]) -> dict[str, Any]:
        path = self._manifest_path(session)
        if path.exists():
            return load_json(path)
        return empty_manifest(
            campaign_id=session["campaign_id"],
            evaluation_run_id=session["evaluation_run_id"],
            alias=session["alias"],
            provider=session["provider"],
            model_id=session["model_id"],
        )

    def _write_manifest(self, session: dict[str, Any], **updates: Any) -> dict[str, Any]:
        manifest = self._manifest(session)
        for key, value in updates.items():
            if value is not None or key in {
                "recording_sid",
                "recording_url",
                "duration_seconds",
                "ended_at",
            }:
                manifest[key] = value
        manifest["call_sid"] = session["call_sid"]
        manifest["alias"] = session["alias"]
        manifest["model_id"] = session["model_id"]
        manifest["provider"] = session["provider"]
        if not manifest.get("started_at"):
            manifest["started_at"] = session.get("started_at")
        write_json(self._manifest_path(session), manifest)
        return manifest

"""One outbound call. Nothing is dialed unless a poster is passed in on purpose."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from local_call.evaluate import evaluate_saved_call
from local_call.phone import STREAM_PATH

IST = timezone(timedelta(hours=5, minutes=30))
CONNECT_URL = "https://api.exotel.com/v1/Accounts/{account_sid}/Calls/connect"
DAY_START = 9
DAY_END = 19


def in_daytime(moment: datetime) -> bool:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=IST)
    local = moment.astimezone(IST)
    return DAY_START <= local.hour < DAY_END


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_numbers(path) -> list[str]:
    """Read a customer list that lives outside this repository."""
    chosen = Path(path).resolve()
    repo = _repo_root()
    if chosen == repo or repo in chosen.parents:
        raise ValueError("the customer list must stay outside the repository")
    numbers = []
    for line in chosen.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        numbers.append(raw)
    return numbers


def with_sample_rate(url: str) -> str:
    if "sample-rate=" in url:
        return url
    join = "&" if "?" in url else "?"
    return f"{url}{join}sample-rate=16000"


class Dialer:
    """Places at most one call, retries a no-answer once, and skips a closed call."""

    def __init__(self, numbers=None):
        self.numbers = list(numbers or [])
        self.open_number = None
        self.open_id = None
        self.closed: set[str] = set()
        self.attempts: dict[str, int] = {}

    def refusal(self, number: str, moment: datetime) -> str | None:
        if not in_daytime(moment):
            return "outside daytime hours"
        if self.open_id:
            return "a call is already open"
        if number in self.closed:
            return "that call already closed"
        if self.attempts.get(number, 0) >= 2:
            return "no-answer was already tried again"
        return None

    def note_no_answer(self, number: str) -> None:
        self.open_id = None
        self.open_number = None

    def note_closed(self, number: str) -> None:
        self.closed.add(number)
        self.open_id = None
        self.open_number = None

    def place(self, number: str, moment: datetime, poster, *, account_sid, caller_id, stream_url, auth):
        """Ask Exotel to ring one number. `poster` is the only way a request leaves this process."""
        reason = self.refusal(number, moment)
        if reason:
            return {"placed": False, "reason": reason}
        if poster is None:
            return {"placed": False, "reason": "live Exotel calls are not enabled in this build"}
        form = {
            "From": number,
            "CallerId": caller_id,
            "StreamUrl": with_sample_rate(stream_url),
            "StreamType": "bidirectional",
            "Record": "false",
        }
        response = poster(
            CONNECT_URL.format(account_sid=account_sid),
            data=form,
            auth=auth,
            timeout=30,
        )
        if getattr(response, "status_code", 0) != 200:
            return {"placed": False, "reason": "Exotel did not accept the call"}
        call_id = call_sid(response)
        self.open_number = number
        self.open_id = call_id
        self.attempts[number] = self.attempts.get(number, 0) + 1
        return {"placed": True, "call_id": call_id, "reason": ""}

    def write_scorecard(self, record: dict, runs_dir: Path, facts: dict, poster=None) -> dict:
        return evaluate_saved_call(record, runs_dir, facts, poster=poster)


def call_sid(response) -> str:
    """Exotel returns the call id in JSON or in XML. The rest of the body stays unread."""
    body = None
    if hasattr(response, "json"):
        try:
            body = response.json()
        except Exception:
            body = None
    if isinstance(body, dict):
        call = body.get("Call") or body
        found = (call or {}).get("Sid") or (call or {}).get("CallSid") or ""
        if found:
            return str(found)
    text = getattr(response, "text", "") or ""
    match = re.search(r"<Sid>([^<]+)</Sid>", text)
    return match.group(1).strip() if match else ""


def credentials_from_env():
    """Read Exotel settings. Callers must not print the result."""
    return {
        "account_sid": (os.getenv("EXOTEL_ACCOUNT_SID") or "").strip(),
        "caller_id": (os.getenv("EXOTEL_CALLER_ID") or "").strip(),
        "stream_url": (os.getenv("EXOTEL_STREAM_URL") or "").strip(),
        "proof_number": (os.getenv("EXOTEL_PROOF_NUMBER") or "").strip(),
        "auth": (
            (os.getenv("EXOTEL_API_KEY") or "").strip(),
            (os.getenv("EXOTEL_API_TOKEN") or "").strip(),
        ),
    }


def configured_stream_url(public_base: str) -> str:
    return with_sample_rate(public_base.rstrip("/") + STREAM_PATH)


def public_wss(url: str) -> bool:
    """Exotel has to open this address. A localhost socket cannot be reached."""
    raw = (url or "").strip().lower()
    if not raw.startswith("wss://"):
        return False
    host = raw.split("://", 1)[1].split("/", 1)[0].split("@")[-1]
    hostname = host.split(":")[0]
    return hostname not in {"", "localhost", "127.0.0.1", "::1"}

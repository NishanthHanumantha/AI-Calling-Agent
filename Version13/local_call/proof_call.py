"""Place one Singapore trial call. This file does not print credentials or the number."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env", override=False)
sys.path.insert(0, str(ROOT))

from local_call.dialer import IST, Dialer, credentials_from_env, public_wss  # noqa: E402


def _short_reason(response) -> str:
    status = getattr(response, "status_code", 0)
    try:
        body = response.json()
    except Exception:
        return f"status {status}"
    if not isinstance(body, dict):
        return f"status {status}"
    detail = body.get("RestException") or body.get("error") or body
    if isinstance(detail, dict):
        text = detail.get("Message") or detail.get("message") or detail.get("Status") or ""
        text = str(text).strip()
        if text:
            return f"status {status}: {text[:180]}"
    return f"status {status}"


def place_one(poster, moment: datetime | None = None) -> dict:
    """Dial the one verified mobile. Missing settings return a status and do not post."""
    creds = credentials_from_env()
    missing = []
    if not creds["account_sid"]:
        missing.append("EXOTEL_ACCOUNT_SID")
    if not creds["auth"][0]:
        missing.append("EXOTEL_API_KEY")
    if not creds["auth"][1]:
        missing.append("EXOTEL_API_TOKEN")
    if not creds["caller_id"]:
        missing.append("EXOTEL_CALLER_ID")
    if not creds["proof_number"]:
        missing.append("EXOTEL_PROOF_NUMBER")
    if not creds["stream_url"]:
        missing.append("EXOTEL_STREAM_URL")
    if missing:
        return {"placed": False, "reason": "missing " + ", ".join(missing), "call_id": ""}
    if not public_wss(creds["stream_url"]):
        return {
            "placed": False,
            "reason": "EXOTEL_STREAM_URL must be a public wss address",
            "call_id": "",
        }
    when = moment or datetime.now(IST)
    result = Dialer([creds["proof_number"]]).place(
        creds["proof_number"],
        when,
        poster,
        account_sid=creds["account_sid"],
        caller_id=creds["caller_id"],
        stream_url=creds["stream_url"],
        auth=creds["auth"],
    )
    return result


def main() -> int:
    seen = {}

    def poster(url, data, auth, timeout):
        response = requests.post(url, data=data, auth=auth, timeout=timeout)
        seen["response"] = response
        return response

    result = place_one(poster)
    if result.get("placed"):
        print("proof call placed", result.get("call_id") or "")
        return 0
    response = seen.get("response")
    if response is not None and getattr(response, "status_code", 0) != 200:
        print("proof call not placed:", _short_reason(response))
        return 2
    print("proof call not placed:", result.get("reason") or "not placed")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

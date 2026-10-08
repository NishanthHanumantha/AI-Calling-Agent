"""TwiML helpers for the eval FastAPI app. Independent copy of production patterns."""

from __future__ import annotations

import html
import re

from .constants import GATHER_TIMEOUT, SAY_VOICE, SPEECH_HINTS


def escape_twiml(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def say_block(text: str) -> str:
    return f'<Say voice="{SAY_VOICE}">{escape_twiml(text)}</Say>'


def gather_block(prompt_text: str, action: str = "/eval/handle-speech") -> str:
    hints = html.escape(SPEECH_HINTS, quote=True)
    return (
        f'<Gather input="speech" action="{html.escape(action, quote=True)}" '
        f'method="POST" actionOnEmptyResult="true" timeout="{GATHER_TIMEOUT}" '
        f'speechTimeout="auto" hints="{hints}" language="en-IN">'
        f"{say_block(prompt_text)}"
        "</Gather>"
    )


def twiml_response(*inner: str) -> str:
    body = "".join(inner)
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>'


def hangup_twiml(message: str) -> str:
    return twiml_response(say_block(message), "<Hangup/>")


def gather_twiml(prompt_text: str) -> str:
    return twiml_response(gather_block(prompt_text))


def clean_spoken(text: str | None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.I)
    raw = re.sub(r"</?think>", "", raw, flags=re.I)
    raw = " ".join(raw.split())
    return raw[:500]

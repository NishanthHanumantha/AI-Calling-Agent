"""Names and prices for one measured call. This does not choose a reply."""

from __future__ import annotations

import contextvars
import io
import wave

MODEL_FUNCTION = contextvars.ContextVar("model_function", default="")

INPUT_INR_PER_TOKEN = 15 / 1_000_000
CACHED_INR_PER_TOKEN = 5 / 1_000_000
OUTPUT_INR_PER_TOKEN = 60 / 1_000_000
STT_INR_PER_SECOND = 0.49 / 60
BULBUL_INR_PER_CHARACTER = 3 / 1000
TRANSLATE_INR_PER_CHARACTER = 2 / 1000
GST_RATE = 0.18
RATES_CHECKED = "29 Oct 2026"


def tag_model(name: str) -> None:
    MODEL_FUNCTION.set(name)


def current_model() -> str:
    return MODEL_FUNCTION.get()


def _whole(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def usage_from_body(body) -> dict:
    """Token counts from a chat response. A missing field stays blank."""
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None}
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return {
        "prompt_tokens": _whole(usage.get("prompt_tokens")),
        "completion_tokens": _whole(usage.get("completion_tokens")),
        "cached_tokens": _whole(cached),
    }


def wav_seconds(audio: bytes) -> float | None:
    """Length of a WAV already about to be uploaded. None when the header is unreadable."""
    if not audio or audio[:4] != b"RIFF":
        return None
    try:
        with wave.open(io.BytesIO(audio), "rb") as clip:
            rate = clip.getframerate()
            frames = clip.getnframes()
    except (wave.Error, EOFError):
        return None
    if rate <= 0:
        return None
    return round(frames / rate, 3)


def _money(amount: float | None) -> float | None:
    if amount is None:
        return None
    return round(amount, 4)


def _with_gst(amount: float | None) -> float | None:
    if amount is None:
        return None
    return round(amount * (1 + GST_RATE), 4)


def _sum_optional(values) -> int | None:
    whole = [value for value in values if isinstance(value, int)]
    if len(whole) != len(list(values)):
        return None
    return sum(whole)


def language_model_inr(calls: list[dict]) -> dict:
    prompts = [call.get("prompt_tokens") for call in calls]
    completions = [call.get("completion_tokens") for call in calls]
    cached_values = [call.get("cached_tokens") for call in calls]
    prompt = _sum_optional(prompts)
    completion = _sum_optional(completions)
    cached_known = [value for value in cached_values if isinstance(value, int)]
    cached = sum(cached_known) if cached_known else None
    amount = None
    if prompt is not None and completion is not None:
        cached_billed = cached or 0
        fresh = max(prompt - cached_billed, 0)
        amount = (
            fresh * INPUT_INR_PER_TOKEN
            + cached_billed * CACHED_INR_PER_TOKEN
            + completion * OUTPUT_INR_PER_TOKEN
        )
    return {
        "requests": len(calls),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached if cached_known else None,
        "inr_ex_gst": _money(amount),
        "inr_with_gst": _with_gst(amount),
    }


def priced(quantity: float | None, rate: float) -> dict:
    amount = None if quantity is None else quantity * rate
    return {
        "inr_ex_gst": _money(amount),
        "inr_with_gst": _with_gst(amount),
    }


def label_source(reply: str, stage: str, model_calls: list[dict], card: str | None) -> str:
    """Model first, then a project card, then a fixed written line."""
    names = [call.get("function") or "chat" for call in model_calls]
    if names:
        return "language model, " + ", ".join(names)
    if card:
        return f"project card, {card}"
    text = (reply or "").strip()
    if text.startswith("Hi, this is Sobha Limited calling."):
        return "written greeting"
    from local_call import projects as catalog
    import app_v3

    if catalog.AREA_QUESTION in text:
        return "written, projects.py AREA_QUESTION"
    if catalog.CONFIGURATION_QUESTION in text:
        return "written, projects.py CONFIGURATION_QUESTION"
    if catalog.PURPOSE_QUESTION in text:
        return "written, projects.py PURPOSE_QUESTION"
    if text == app_v3.ADVISOR_CLOSE or "sales advisor get in touch" in text.lower():
        return "written, ADVISOR_CLOSE"
    return f"written script, {stage}"

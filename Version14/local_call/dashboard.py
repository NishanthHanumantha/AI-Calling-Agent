"""Saved-call list for the local page. This does not place a call or turn recording on."""

from __future__ import annotations

import json
from pathlib import Path

from local_call.sales_sheet import HEADERS, NOT_DISCUSSED, SHEET_PATH
from local_call.usage_note import (
    BULBUL_INR_PER_CHARACTER,
    CACHED_INR_PER_TOKEN,
    INPUT_INR_PER_TOKEN,
    OUTPUT_INR_PER_TOKEN,
    STT_INR_PER_SECOND,
    TRANSLATE_INR_PER_CHARACTER,
    priced,
)

_PROGRESSED = (
    "Part of Bengaluru",
    "Location",
    "Project",
    "Purpose",
    "Configuration",
    "Size",
    "Price range",
    "Site visit",
    "Visit day",
    "Visit time",
    "Callback",
)


def _filled(value) -> bool:
    text = "" if value is None else str(value).strip()
    return bool(text) and text != NOT_DISCUSSED


def _merged_details(record: dict, sheet_row: dict | None) -> dict:
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    row = dict(sheet_row or {})
    row.update({key: value for key, value in details.items() if value not in (None, "")})
    return row


def _reached_a_question(record: dict, sheet_row: dict | None) -> bool:
    """A name-only call, such as one saved as Written, stays off the stakeholder list."""
    row = _merged_details(record, sheet_row)
    return any(_filled(row.get(field)) for field in _PROGRESSED)


_DETAIL_FIELDS = (
    ("Customer name", "name"),
    ("Phone", "phone"),
    ("Language", "language"),
    ("Part of Bengaluru", "part"),
    ("Location", "place"),
    ("Project", "project"),
    ("Purpose", "purpose"),
    ("Configuration", "configuration"),
    ("Site visit", "visit"),
)


def list_calls(runs_dir: Path, sheet_path: Path | None = None) -> list[dict]:
    """Newest saved call first. A missing sales sheet leaves preference cells blank."""
    folder = Path(runs_dir)
    if not folder.is_dir():
        return []
    sheet = _sales_index(sheet_path)
    calls = []
    for path in sorted(folder.glob("*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or not record.get("call_id"):
            continue
        sheet_row = sheet.get(record["call_id"]) or {}
        if _reached_a_question(record, sheet_row):
            calls.append(summarize(record, sheet_row))
    return calls


def load_call(runs_dir: Path, call_id: str, sheet_path: Path | None = None) -> dict | None:
    """One call, with each spoken turn in order. The id cannot leave the runs folder."""
    safe = (call_id or "").strip()
    folder = Path(runs_dir).resolve()
    if not safe or Path(safe).name != safe or safe in {".", ".."}:
        return None
    path = (folder / f"{safe}.json").resolve()
    if path.parent != folder or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    summary = summarize(record, _sales_index(sheet_path).get(record.get("call_id")) or {})
    summary["turns"] = [
        line
        for turn in record.get("turns") or []
        if isinstance(turn, dict)
        for line in _turn(turn)["lines"]
    ]
    return summary


def summarize(record: dict, sheet_row: dict | None = None) -> dict:
    """The list row. A browser recording is the filename stored with the call."""
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    row = dict(sheet_row or {})
    row.update({key: value for key, value in details.items() if value not in (None, "")})
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    price, cost = _sarvam_price(usage)
    summary = {
        "call_id": record.get("call_id") or "",
        "started_at": record.get("started_at") or "",
        "date": row.get("Call date") or "",
        "time": row.get("Call time") or "",
        "cost": cost,
        "price": price,
        "recording": record.get("recording") or None,
    }
    for heading, key in _DETAIL_FIELDS:
        summary[key] = row.get(heading) or NOT_DISCUSSED
    summary["latency"] = _latency(record.get("turns") or [])
    if summary["phone"] == NOT_DISCUSSED:
        summary["phone"] = ""
    return summary


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _part_amount(quantity, rate, stored) -> float:
    """The current Sarvam price for a stored quantity. An old ex-GST amount is the fallback."""
    measured = _number(quantity)
    if measured is not None:
        return priced(measured, rate)["inr_ex_gst"] or 0.0
    if isinstance(stored, dict):
        saved = _number(stored.get("inr_ex_gst"))
        if saved is not None:
            return saved
    return 0.0


def _model_amount(model) -> float | None:
    """Sarvam 105B from the stored token counts. A call without tokens keeps its ex-GST amount."""
    if not isinstance(model, dict):
        return None
    prompt = model.get("prompt_tokens")
    completion = model.get("completion_tokens")
    if isinstance(prompt, int) and not isinstance(prompt, bool) and isinstance(completion, int) and not isinstance(completion, bool):
        cached = model.get("cached_tokens")
        cached_billed = cached if isinstance(cached, int) and not isinstance(cached, bool) else 0
        fresh = max(prompt - cached_billed, 0)
        return (
            fresh * INPUT_INR_PER_TOKEN
            + cached_billed * CACHED_INR_PER_TOKEN
            + completion * OUTPUT_INR_PER_TOKEN
        )
    return _number(model.get("inr_ex_gst"))


def _sarvam_price(usage: dict) -> tuple[dict | None, float | None]:
    """The ex-GST Sarvam amount. A call with no measured parts keeps its stored ex-GST total."""
    voice = usage.get("bulbul") if isinstance(usage.get("bulbul"), dict) else None
    heard = usage.get("speech_to_text") if isinstance(usage.get("speech_to_text"), dict) else None
    translated = usage.get("translation") if isinstance(usage.get("translation"), dict) else None
    if voice is None and heard is None and translated is None:
        saved = _number(usage.get("inr_ex_gst"))
        return None, None if saved is None else round(saved, 2)
    voice_amount = _part_amount((voice or {}).get("characters") or 0, BULBUL_INR_PER_CHARACTER, voice)
    speech_amount = _part_amount((heard or {}).get("seconds"), STT_INR_PER_SECOND, heard)
    translation_amount = _part_amount(
        (translated or {}).get("characters") or 0,
        TRANSLATE_INR_PER_CHARACTER,
        translated,
    )
    amounts = [voice_amount, speech_amount, translation_amount]
    price = {
        "voice": {
            "characters": int((voice or {}).get("characters") or 0),
            "inr": round(voice_amount, 2),
        },
        "speech_to_text": {
            "seconds": None if heard is None else heard.get("seconds"),
            "inr": round(speech_amount, 2),
        },
        "translation": {
            "characters": int((translated or {}).get("characters") or 0),
            "inr": round(translation_amount, 2),
        },
    }
    model = usage.get("language_model") if isinstance(usage.get("language_model"), dict) else None
    model_amount = _model_amount(model)
    if model_amount:
        amounts.append(model_amount)
        price["model"] = {"inr": round(model_amount, 2)}
    return price, round(sum(amounts), 2)


def _turn(turn: dict) -> dict:
    source = turn.get("source") or ""
    written_by_model = str(source).startswith("language model")
    lines = []
    customer = (turn.get("customer") or "").strip()
    reply = (turn.get("reply") or "").strip()
    if customer:
        lines.append({"speaker": "Customer", "text": customer, "origin": ""})
    if reply:
        lines.append(
            {
                "speaker": "Agent",
                "text": reply,
                "origin": "Written by Sarvam 105B" if written_by_model else "Written sentence",
                "decision_ms": _milliseconds(turn.get("decision_ms")),
                "first_voice_ms": _milliseconds(turn.get("tts_ms")),
            }
        )
    return {"lines": lines}


def _latency(turns: list) -> dict:
    """Averages after the customer has spoken. The greeting is not a response."""
    hearing = []
    decision = []
    first_voice = []
    response = []
    for turn in turns:
        if not isinstance(turn, dict) or not (turn.get("customer") or "").strip():
            continue
        heard = _milliseconds(turn.get("stt_ms"))
        chosen = _milliseconds(turn.get("decision_ms"))
        voice = _milliseconds(turn.get("tts_ms"))
        if heard is not None:
            hearing.append(heard)
        if chosen is not None:
            decision.append(chosen)
        if voice is not None:
            first_voice.append(voice)
        if chosen is not None and voice is not None:
            response.append(chosen + voice)
    return {
        "hearing_ms": _average(hearing),
        "decision_ms": _average(decision),
        "first_voice_ms": _average(first_voice),
        "response_ms": _average(response),
    }


def _milliseconds(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(round(value))


def _average(values: list[int]) -> int | None:
    if not values:
        return None
    return int(round(sum(values) / len(values)))


def _sales_index(sheet_path: Path | None) -> dict:
    target = Path(sheet_path) if sheet_path is not None else SHEET_PATH
    if not target.is_file():
        return {}
    try:
        from openpyxl import load_workbook

        book = load_workbook(target, read_only=True, data_only=True)
    except Exception:
        return {}
    try:
        sheet = book["Calls"] if "Calls" in book.sheetnames else book.active
        rows = sheet.iter_rows(values_only=True)
        headers = [str(cell or "") for cell in next(rows, ())]
        id_at = headers.index("Call id") if "Call id" in headers else None
        if id_at is None:
            return {}
        found = {}
        for values in rows:
            if not values or id_at >= len(values) or not values[id_at]:
                continue
            found[str(values[id_at])] = {
                header: "" if index >= len(values) or values[index] is None else str(values[index])
                for index, header in enumerate(headers)
                if header in HEADERS
            }
        return found
    finally:
        book.close()

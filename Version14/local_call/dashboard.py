"""Saved-call list for the local page. This does not place a call or turn recording on."""

from __future__ import annotations

import json
from pathlib import Path

from local_call.sales_sheet import HEADERS, NOT_DISCUSSED, SHEET_PATH

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
        if isinstance(record, dict) and record.get("call_id"):
            calls.append(summarize(record, sheet.get(record["call_id"]) or {}))
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
    """The list row. Recording stays empty until a telephony company supplies a file."""
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    row = dict(sheet_row or {})
    row.update({key: value for key, value in details.items() if value not in (None, "")})
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    cost = usage.get("inr_with_gst")
    summary = {
        "call_id": record.get("call_id") or "",
        "started_at": record.get("started_at") or "",
        "date": row.get("Call date") or "",
        "time": row.get("Call time") or "",
        "cost": None if cost is None else round(float(cost), 2),
        "recording": record.get("recording") or None,
    }
    for heading, key in _DETAIL_FIELDS:
        summary[key] = row.get(heading) or NOT_DISCUSSED
    summary["latency"] = _latency(record.get("turns") or [])
    if summary["phone"] == NOT_DISCUSSED:
        summary["phone"] = ""
    return summary


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

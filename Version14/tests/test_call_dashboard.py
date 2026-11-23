"""The call dashboard reads saved notes and leaves recording empty."""

import json
from pathlib import Path

from local_call.dashboard import list_calls, load_call


def _write(folder: Path, call_id: str, **extra) -> None:
    record = {
        "call_id": call_id,
        "started_at": "2026-11-18T12:24:11+00:00",
        "turns": [
            {
                "customer": None,
                "reply": "Hi, this is Sobha Limited calling.",
                "source": "written greeting",
            },
            {
                "customer": "I am Nishant.",
                "reply": "Okay Nishant. Which part of Bengaluru are you considering?",
                "source": "written, projects.py AREA_QUESTION",
            },
            {
                "customer": "What is the price?",
                "reply": "I will ask our sales advisor to share that.",
                "source": "language model, chat",
            },
        ],
        "usage": {"inr_with_gst": 0.44},
        "recording": None,
    }
    record.update(extra)
    (folder / f"{call_id}.json").write_text(json.dumps(record), encoding="utf-8")


def test_the_list_uses_the_sales_sheet_and_leaves_recording_empty(tmp_path):
    from openpyxl import Workbook

    _write(tmp_path, "20261118T122411Z")
    book = Workbook()
    sheet = book.active
    sheet.title = "Calls"
    sheet.append(["Call id", "Customer name", "Phone", "Project", "Site visit", "Part of Bengaluru", "Location", "Language", "Purpose", "Configuration", "Call date", "Call time"])
    sheet.append(["20261118T122411Z", "Nishant", "", "SOBHA Townpark", "Not discussed", "South", "Electronic City", "English", "Own use", "3 BHK", "18 Nov 2026", "5:54 PM"])
    sheet_path = tmp_path / "calls.xlsx"
    book.save(sheet_path)

    listed = list_calls(tmp_path, sheet_path)

    assert listed[0]["name"] == "Nishant"
    assert listed[0]["project"] == "SOBHA Townpark"
    assert listed[0]["visit"] == "Not discussed"
    assert listed[0]["cost"] == 0.44
    assert listed[0]["recording"] is None
    assert listed[0]["phone"] == ""


def test_opening_a_call_keeps_the_spoken_order_and_who_wrote_the_reply(tmp_path):
    _write(
        tmp_path,
        "20261118T130000Z",
        details={
            "Customer name": "Nishant",
            "Project": "SOBHA Townpark",
            "Site visit": "Will visit",
            "Call date": "18 Nov 2026",
            "Call time": "6:30 PM",
        },
    )

    opened = load_call(tmp_path, "20261118T130000Z", tmp_path / "missing.xlsx")

    assert [line["speaker"] for line in opened["turns"]] == ["Agent", "Customer", "Agent", "Customer", "Agent"]
    assert opened["turns"][0]["origin"] == "Written sentence"
    assert opened["turns"][-1]["origin"] == "Written by Sarvam 105B"
    assert opened["name"] == "Nishant"
    assert opened["recording"] is None


def test_a_response_average_uses_only_the_answered_turn(tmp_path):
    _write(
        tmp_path,
        "20261118T140000Z",
        turns=[
            {
                "customer": None,
                "reply": "Hi, this is Sobha Limited calling.",
                "source": "written greeting",
                "stt_ms": None,
                "decision_ms": 5,
                "tts_ms": 800,
            },
            {
                "customer": "I am Nishant.",
                "reply": "Okay Nishant.",
                "source": "written script, qualify",
                "stt_ms": 400,
                "decision_ms": 200,
                "tts_ms": 300,
            },
            {
                "customer": "What is the price?",
                "reply": "Our advisor can share that.",
                "source": "language model, chat",
                "stt_ms": 600,
                "decision_ms": 100,
                "tts_ms": None,
            },
        ],
    )

    opened = load_call(tmp_path, "20261118T140000Z", tmp_path / "missing.xlsx")

    assert opened["latency"]["response_ms"] == 500
    assert opened["latency"]["hearing_ms"] == 500
    assert opened["latency"]["decision_ms"] == 150
    assert opened["latency"]["first_voice_ms"] == 300
    agent = [line for line in opened["turns"] if line["speaker"] == "Agent"]
    assert agent[1]["decision_ms"] == 200
    assert agent[1]["first_voice_ms"] == 300
    assert agent[2]["first_voice_ms"] is None
    assert load_call(tmp_path, "../secret") is None
    assert load_call(tmp_path, "missing") is None


def test_a_saved_price_keeps_the_three_parts(tmp_path):
    _write(
        tmp_path,
        "20261123T122935Z",
        usage={
            "inr_with_gst": 1.2624,
            "bulbul": {"characters": 867, "inr_with_gst": 1.0231},
            "speech_to_text": {"seconds": 24.833, "inr_with_gst": 0.2393},
            "translation": {"characters": 0, "inr_with_gst": 0.0},
            "language_model": {"inr_with_gst": 0.0},
        },
        details={"Project": "SOBHA Valley View", "Location": "Not discussed"},
    )

    opened = load_call(tmp_path, "20261123T122935Z", tmp_path / "missing.xlsx")

    assert opened["cost"] == 1.26
    assert opened["price"]["voice"] == {"characters": 867, "inr": 1.02}
    assert opened["price"]["speech_to_text"]["seconds"] == 24.833
    assert opened["price"]["speech_to_text"]["inr"] == 0.24
    assert opened["price"]["translation"] == {"characters": 0, "inr": 0.0}
    assert "model" not in opened["price"]
    assert opened["project"] == "SOBHA Valley View"


def test_a_total_without_parts_has_no_breakdown(tmp_path):
    _write(tmp_path, "20261118T150000Z")

    opened = load_call(tmp_path, "20261118T150000Z", tmp_path / "missing.xlsx")

    assert opened["cost"] == 0.44
    assert opened["price"] is None

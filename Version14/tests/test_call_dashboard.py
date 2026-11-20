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


def test_a_call_id_cannot_read_outside_the_runs_folder(tmp_path):
    assert load_call(tmp_path, "../secret") is None
    assert load_call(tmp_path, "missing") is None

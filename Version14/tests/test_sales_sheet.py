"""The sales advisor workbook. Spoken lines stay as they were."""

import asyncio
import re

import app_v3
from local_call.sales_sheet import HEADERS, append_sales_row, fill_blank_preferences, sales_row


class _Request:
    def __init__(self, speech):
        self._form = {"SpeechResult": speech, "UnstableSpeechResult": ""}

    async def form(self):
        return self._form


def _spoken(response):
    xml = response.body.decode() if isinstance(response.body, bytes) else response.body
    parts = re.findall(r"<Say[^>]*>(.*?)</Say>", xml, flags=re.S)
    text = " ".join(parts)
    return (
        text.replace("&apos;", "'")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def _turn(speech):
    response = asyncio.run(app_v3.handle_speech(_Request(speech), SpeechResult=speech))
    return _spoken(response)


def _cells(path):
    from openpyxl import load_workbook

    sheet = load_workbook(path)["Calls"]
    rows = list(sheet.iter_rows(values_only=True))
    return rows[0], [dict(zip(rows[0], row)) for row in rows[1:]]


def test_a_known_call_is_written_for_the_advisor(tmp_path):
    memory = {
        "customer_name": "Nishanth",
        "dialed_number": "+910000000001",
        "customer_language": "hi-IN",
        "customer_part": "East",
        "customer_place": "Whitefield",
        "active_project_ids": ["sobha-townpark"],
        "customer_purpose": "live",
        "customer_configuration": "3 BHK",
        "customer_variant": "Grande",
        "price_ceiling": 2,
        "price_strict": True,
        "visit_outcome": "Clear yes",
        "visit_day": "tomorrow",
        "selected_slot": "4:00 PM",
        "callback_when": "6 PM",
        "wants_plot": True,
        "unstored_topics": ["parking", "RERA"],
    }
    path = append_sales_row(
        memory,
        call_id="call-1",
        started_at="2026-11-16T10:18:00+00:00",
        path=tmp_path / "Sales_Advisor_Calls.xlsx",
    )
    header, rows = _cells(path)
    assert header == HEADERS
    row = rows[0]
    assert row["Call date"] == "16 Nov 2026"
    assert row["Call time"] == "3:48 PM"
    assert row["Call id"] == "call-1"
    assert row["Customer name"] == "Nishanth"
    assert row["Phone"] == "+910000000001"
    assert row["Language"] == "Hindi"
    assert row["Part of Bengaluru"] == "East"
    assert row["Location"] == "Whitefield"
    assert "townpark" in row["Project"].lower()
    assert row["Purpose"] == "Own use"
    assert row["Configuration"] == "3 BHK"
    assert row["Size"] == "Grande"
    assert row["Price range"] == "Under 2 crore"
    assert row["Site visit"] == "Clear yes"
    assert row["Visit day"] == "tomorrow"
    assert row["Visit time"] == "4:00 PM"
    assert row["Callback"] == "6 PM"
    assert row["Advisor note"] == "Asked about a plot. Asked about parking and RERA."

    sheet = __import__("openpyxl").load_workbook(path)["Calls"]
    assert sheet.freeze_panes == "A2"
    assert sheet.auto_filter.ref.endswith(str(sheet.max_row))
    assert sheet["E2"].number_format == "@"


def test_a_browser_call_leaves_the_phone_blank_and_a_second_call_appends(tmp_path):
    path = tmp_path / "Sales_Advisor_Calls.xlsx"
    append_sales_row(
        {"customer_language": "en-IN", "customer_purpose": "investment", "price_ceiling": 1.65},
        call_id="call-a",
        started_at="2026-11-16T05:00:00+00:00",
        path=path,
    )
    append_sales_row(
        {
            "customer_language": "kn-IN",
            "pending_project_ids": ["sobha-townpark"],
            "price_ceiling": 1.65,
            "price_strict": False,
        },
        call_id="call-b",
        started_at="2026-11-16T06:00:00+00:00",
        path=path,
    )
    _header, rows = _cells(path)
    assert [row["Call id"] for row in rows] == ["call-a", "call-b"]
    assert not rows[0]["Customer name"]
    assert not rows[0]["Phone"]
    assert rows[0]["Language"] == "English"
    assert rows[0]["Purpose"] == "Investment"
    assert rows[0]["Price range"] == "Up to 1.65 crore"
    assert rows[0]["Part of Bengaluru"] == "Not discussed"
    assert rows[0]["Location"] == "Not discussed"
    assert rows[0]["Project"] == "Not discussed"
    assert rows[0]["Configuration"] == "Not discussed"
    assert rows[0]["Size"] == "Not discussed"
    assert rows[0]["Site visit"] == "Not discussed"
    assert rows[0]["Visit day"] == "Not discussed"
    assert rows[0]["Visit time"] == "Not discussed"
    assert rows[0]["Callback"] == "Not discussed"
    assert "townpark" in rows[1]["Project"].lower()
    assert rows[1]["Language"] == "Kannada"


def test_a_visit_yes_keeps_the_day_and_the_same_close():
    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "visit_offered": True,
        "active_project_ids": ["sobha-townpark"],
        "last_agent_question": "Would you like to schedule a site visit to SOBHA Townpark?",
    })
    spoken = _turn("Yes, tomorrow at 4 PM.")
    assert "visit to" in spoken.lower()
    assert "sobha townpark" in spoken.lower()
    assert app_v3.conversation_memory["visit_outcome"] == "Clear yes"
    assert app_v3.conversation_memory["visit_day"] == "tomorrow"
    assert app_v3.conversation_memory["selected_slot"] == "4:00 PM"


def test_a_visit_no_and_a_later_are_stored_without_a_new_question():
    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "visit_offered": True,
        "last_agent_question": "Would you like to schedule a site visit?",
    })
    spoken = _turn("No.")
    assert "sales advisor" in spoken.lower()
    assert "?" not in spoken
    assert app_v3.conversation_memory["visit_outcome"] == "Clear no"

    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "visit_offered": True,
        "last_agent_question": "Would you like to schedule a site visit?",
    })
    spoken = _turn("I will let you know later.")
    assert "sales advisor" in spoken.lower()
    assert app_v3.conversation_memory["visit_outcome"] == "Will check and confirm later"


def test_a_project_refusal_and_a_pause_are_not_visit_answers():
    app_v3.reset_memory()
    app_v3.conversation_memory.update({"stage": "qualify", "catalog_flow": True})
    _turn("I am not interested.")
    assert app_v3.conversation_memory["visit_outcome"] is None

    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "visit_offered": True,
        "last_agent_question": "Would you like to schedule a site visit?",
    })
    _turn("I need some time.")
    assert app_v3.conversation_memory["visit_outcome"] is None


def test_an_unstored_detail_is_kept_once():
    app_v3.reset_memory()
    app_v3.conversation_memory["stage"] = "answer_faq"
    spoken = _turn("Is parking available?")
    assert spoken.strip() == "That detail is not confirmed. Our sales advisor can share it."
    _turn("What about the parking again?")
    assert app_v3.conversation_memory["unstored_topics"] == ["parking"]


def test_an_older_blank_preference_cell_becomes_not_discussed(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "Sales_Advisor_Calls.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "Calls"
    sheet.append(list(HEADERS))
    sheet.append(["16 Nov 2026", "4:00 PM", "old", "Anita", None, "English", None, "Whitefield"] + [None] * 10)
    book.save(path)
    fill_blank_preferences(path)
    _header, rows = _cells(path)
    assert rows[0]["Customer name"] == "Anita"
    assert not rows[0]["Phone"]
    assert rows[0]["Location"] == "Whitefield"
    assert rows[0]["Configuration"] == "Not discussed"
    assert rows[0]["Site visit"] == "Not discussed"
    assert not rows[0]["Advisor note"]


def test_the_row_builder_leaves_a_missing_name_blank():
    from local_call.sales_sheet import NOT_DISCUSSED, _PREFERENCE_COLUMNS

    row = dict(zip(HEADERS, sales_row({}, call_id="x", started_at="")))
    assert row["Customer name"] == ""
    assert row["Phone"] == ""
    assert row["Advisor note"] == ""
    assert all(row[column] == NOT_DISCUSSED for column in _PREFERENCE_COLUMNS)

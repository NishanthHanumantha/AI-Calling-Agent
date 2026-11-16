"""One row per finished call, for the sales advisor. The spoken script is unchanged."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import app_v3


IST = timezone(timedelta(hours=5, minutes=30))
SHEET_PATH = Path(__file__).resolve().parent / "handoff" / "Sales_Advisor_Calls.xlsx"
SHEET_NAME = "Calls"

HEADERS = (
    "Call date",
    "Call time",
    "Call id",
    "Customer name",
    "Phone",
    "Language",
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
    "Advisor note",
)

_WIDTHS = (16, 12, 22, 22, 18, 14, 20, 20, 36, 14, 16, 14, 18, 30, 16, 14, 22, 42)

NOT_DISCUSSED = "Not discussed"
_PREFERENCE_COLUMNS = {
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
}

_LANGUAGES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "kn-IN": "Kannada",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "ml-IN": "Malayalam",
}

_HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
_HEADER_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
_BODY_FONT = Font(name="Calibri", size=11)
_WRAP = Alignment(wrap_text=True, vertical="center", horizontal="left")
_THIN = Border(
    left=Side(style="thin", color="D0D7E2"),
    right=Side(style="thin", color="D0D7E2"),
    top=Side(style="thin", color="D0D7E2"),
    bottom=Side(style="thin", color="D0D7E2"),
)


def record_finished_call(call_id: str, started_at: str, path: Path | None = None) -> None:
    """Append the current call. A sheet error must not drop the saved call."""
    try:
        append_sales_row(
            app_v3.conversation_memory,
            call_id=call_id,
            started_at=started_at,
            path=path,
        )
    except Exception as exc:
        print("SALES SHEET:", type(exc).__name__)


def append_sales_row(memory: dict, *, call_id: str, started_at: str, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else SHEET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        book = load_workbook(target)
        sheet = book[SHEET_NAME] if SHEET_NAME in book.sheetnames else book.active
    else:
        book = Workbook()
        sheet = book.active
        sheet.title = SHEET_NAME
    if sheet["A1"].value != HEADERS[0]:
        _write_header(sheet)
    values = sales_row(memory, call_id=call_id, started_at=started_at)
    sheet.append(values)
    _style_body(sheet, sheet.max_row)
    _finish_sheet(sheet)
    book.save(target)
    return target


def sales_row(memory: dict, *, call_id: str, started_at: str) -> list:
    when = _started(started_at)
    return [
        when.strftime("%d %b %Y"),
        when.strftime("%I:%M %p").lstrip("0"),
        call_id or "",
        (memory.get("customer_name") or "").strip(),
        (memory.get("dialed_number") or "").strip(),
        _LANGUAGES.get(memory.get("customer_language") or "", ""),
        _stated(memory.get("customer_part")),
        _stated(memory.get("customer_place")),
        _stated(_projects(memory)),
        _stated(_purpose(memory.get("customer_purpose"))),
        _stated(memory.get("customer_configuration")),
        _stated(memory.get("customer_variant")),
        _stated(_price(memory)),
        _stated(memory.get("visit_outcome")),
        _stated(memory.get("visit_day")),
        _stated(memory.get("selected_slot")),
        _stated(memory.get("callback_when")),
        _note(memory),
    ]


def _stated(value) -> str:
    text = "" if value is None else str(value).strip()
    return text or NOT_DISCUSSED


def fill_blank_preferences(path: Path | None = None) -> Path | None:
    """Put Not discussed into preference cells the earlier rows left empty."""
    target = Path(path) if path is not None else SHEET_PATH
    if not target.is_file():
        return None
    book = load_workbook(target)
    sheet = book[SHEET_NAME] if SHEET_NAME in book.sheetnames else book.active
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    columns = [
        index for index, title in enumerate(headers, start=1) if title in _PREFERENCE_COLUMNS
    ]
    for row in range(2, sheet.max_row + 1):
        for column in columns:
            cell = sheet.cell(row, column)
            if cell.value is None or str(cell.value).strip() == "":
                cell.value = NOT_DISCUSSED
    book.save(target)
    return target


def _started(started_at: str) -> datetime:
    text = (started_at or "").strip()
    if text:
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return moment.astimezone(IST)
        except ValueError:
            pass
    return datetime.now(IST)


def _projects(memory: dict) -> str:
    ids = memory.get("active_project_ids") or memory.get("pending_project_ids") or []
    names = []
    for item in ids:
        project = app_v3.catalog.BY_ID.get(item)
        if project and project.get("spoken"):
            names.append(project["spoken"])
    return ", ".join(names)


def _purpose(value) -> str:
    if value == "live":
        return "Own use"
    if value == "investment":
        return "Investment"
    return ""


def _price(memory: dict) -> str:
    ceiling = memory.get("price_ceiling")
    if ceiling is None:
        return ""
    amount = float(ceiling)
    shown = str(int(amount)) if amount == int(amount) else f"{amount:.2f}".rstrip("0").rstrip(".")
    if memory.get("price_strict"):
        return f"Under {shown} crore"
    return f"Up to {shown} crore"


def _note(memory: dict) -> str:
    parts = []
    if memory.get("wants_plot"):
        parts.append("Asked about a plot.")
    topics = [topic for topic in (memory.get("unstored_topics") or []) if topic]
    if len(topics) == 1:
        parts.append(f"Asked about {topics[0]}.")
    elif topics:
        parts.append("Asked about " + ", ".join(topics[:-1]) + f" and {topics[-1]}.")
    return " ".join(parts)


def _write_header(sheet) -> None:
    for index, title in enumerate(HEADERS, start=1):
        cell = sheet.cell(1, index, title)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = _WRAP
        cell.border = _THIN
    sheet.row_dimensions[1].height = 24
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}1"
    sheet.freeze_panes = "A2"
    sheet.sheet_view.showGridLines = False
    for index, width in enumerate(_WIDTHS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width


def _style_body(sheet, row: int) -> None:
    for index in range(1, len(HEADERS) + 1):
        cell = sheet.cell(row, index)
        cell.font = _BODY_FONT
        cell.alignment = _WRAP
        cell.border = _THIN
        if HEADERS[index - 1] == "Phone":
            cell.number_format = "@"
    sheet.row_dimensions[row].height = 32


def _finish_sheet(sheet) -> None:
    last = get_column_letter(len(HEADERS))
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{last}{sheet.max_row}"
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.print_title_rows = "1:1"
    sheet.oddHeader.left.text = "Sobha Bengaluru"
    sheet.oddFooter.right.text = "Page &P of &N"

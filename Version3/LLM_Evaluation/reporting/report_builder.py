"""Build AI_Calling_Agent_Master_Evaluation_Report.xlsx."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from .data_loader import CLASS_FRAMEWORK

NAVY = "1F4E79"
STEEL = "2E75B6"
LIGHT = "D6E3F0"
WHITE = "FFFFFF"
AMBER = "FFF2CC"
GREEN = "C6EFCE"
RED = "F8CBAD"
THIN = Border(
    left=Side(style="thin", color="BDD3E6"),
    right=Side(style="thin", color="BDD3E6"),
    top=Side(style="thin", color="BDD3E6"),
    bottom=Side(style="thin", color="BDD3E6"),
)
TITLE_FONT = Font(name="Calibri", size=16, bold=True, color=NAVY)
SECTION_FONT = Font(name="Calibri", size=12, bold=True, color=NAVY)
HEADER_FONT = Font(name="Calibri", size=10, bold=True, color=WHITE)
BODY_FONT = Font(name="Calibri", size=10)
NOTE_FONT = Font(name="Calibri", size=9, italic=True, color="666666")
HEADER_FILL = PatternFill("solid", fgColor=NAVY)
SECTION_FILL = PatternFill("solid", fgColor=LIGHT)
WRAP = Alignment(wrap_text=True, vertical="center")


def _header(ws: Worksheet, row: int, headers: list[str]) -> None:
    for col, title in enumerate(headers, start=1):
        cell = ws.cell(row, col, title)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = THIN
    ws.auto_filter.ref = f"A{row}:{get_column_letter(len(headers))}{row}"
    ws.freeze_panes = f"A{row + 1}"
    ws.row_dimensions[row].height = 32


def _write_rows(ws: Worksheet, start: int, headers: list[str], rows: list[list[Any]], formats: dict[int, str] | None = None) -> int:
    formats = formats or {}
    for offset, row in enumerate(rows):
        r = start + offset
        for col, value in enumerate(row, start=1):
            cell = ws.cell(r, col, value)
            cell.font = BODY_FONT
            cell.alignment = WRAP
            cell.border = THIN
            if col in formats and value is not None and value != "unavailable":
                cell.number_format = formats[col]
    return start + len(rows)


def _autosize(ws: Worksheet, max_width: int = 42) -> None:
    widths: dict[int, int] = {}
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row or 1, 80), max_col=ws.max_column or 1):
        for cell in row:
            if cell.value is None:
                continue
            width = min(max_width, max(12, len(str(cell.value)) + 2))
            widths[cell.column] = max(widths.get(cell.column, 12), width)
    for col, width in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = width


def _title(ws: Worksheet, title: str, subtitle: str | None = None) -> int:
    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=8)
    if subtitle:
        ws["A2"] = subtitle
        ws["A2"].font = NOTE_FONT
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=8)
        return 4
    return 3


def _unavailable(value: Any) -> Any:
    if value is None or value == "":
        return "unavailable"
    return value


def _pct(value: Any) -> Any:
    if value is None:
        return "unavailable"
    return float(value)


def _latency_sec(ms: Any) -> Any:
    if ms is None:
        return "unavailable"
    return round(float(ms) / 1000.0, 3)


def _bar_chart(ws: Worksheet, title: str, cats_col: int, data_col: int, header_row: int, data_rows: int, anchor: str, y_title: str) -> None:
    if data_rows <= 0:
        return
    chart = BarChart()
    chart.type = "col"
    chart.grouping = "clustered"
    chart.title = title
    chart.y_axis.title = y_title
    chart.x_axis.title = None
    data = Reference(ws, min_col=data_col, min_row=header_row, max_row=header_row + data_rows)
    cats = Reference(ws, min_col=cats_col, min_row=header_row + 1, max_row=header_row + data_rows)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.shape = 4
    chart.style = 10
    chart.legend = None
    chart.width = 15
    chart.height = 8
    ws.add_chart(chart, anchor)


def _sheet_exec(wb: Workbook, data: dict[str, Any], recon: dict[str, Any]) -> None:
    ws = wb.active
    ws.title = "Executive Summary"
    scale = data["scale"]
    period = "unavailable"
    if scale["date_start"] and scale["date_end"]:
        period = f"{scale['date_start']} to {scale['date_end']}"
    row = _title(
        ws,
        "AI Calling Agent — Master Evaluation Report",
        "Offline LLM evaluation evidence. Not Twilio voice-call counts. No overall winner or ranking.",
    )
    ws.cell(row, 1, "A. Evaluation Scale").font = SECTION_FONT
    row += 1
    _header(ws, row, ["Metric", "Value", "Notes"])
    header_row = row
    scale_rows = [
        ["Unique Evaluation Runs", scale["unique_evaluation_runs"], "MODEL EVALUATION + MULTILINGUAL MODEL EVALUATION only"],
        ["Customer Turns", scale["customer_turns"], "Non-empty customer utterances across unique model-eval runs"],
        ["Model-Turn Evaluations", scale["model_turn_evaluations"], "Actual model responses evaluated (not voice calls)"],
        ["Labelled Evaluations", scale["labelled_evaluations"], "Ground-truth applicable or expected intent present"],
        ["Unlabelled Turns", scale["unlabelled_turns"], "Openings / intentionally unlabelled / no GT"],
        ["Models Evaluated", scale["models_evaluated"], ", ".join(scale["models"])],
        ["Languages Evaluated", scale["languages_evaluated"], ", ".join(scale["languages"])],
        ["Evaluation Observations", scale["evaluation_observations"], "Same as model-turn evaluations"],
        ["Evaluation Period", period, "From canonical run timestamps"],
        ["Framework Tests (separate)", scale["framework_tests"], "pytest nodeids — not model trials"],
        ["Inventory Run IDs (all classes)", scale["inventory_run_count"], "Includes framework/dry-run dumps, not double-counted"],
    ]
    row = _write_rows(ws, row + 1, ["Metric", "Value", "Notes"], scale_rows)
    ws.auto_filter.ref = f"A{header_row}:C{row - 1}"

    row += 2
    ws.cell(row, 1, "B. Model Summary").font = SECTION_FONT
    row += 1
    headers = [
        "Model",
        "Provider",
        "Evaluation Observations",
        "Intent Accuracy",
        "Intent N",
        "Action Accuracy",
        "Action N",
        "Factual Accuracy",
        "Factual N",
        "Grounded Response %",
        "Grounded N",
        "Unsupported Claim Rate",
        "Unsupported N",
        "Context Accuracy",
        "Context N",
        "Stage Accuracy",
        "Stage N",
        "Avg Relevance",
        "Avg Completeness",
        "Avg Clarity",
        "Avg Conversational Quality",
        "Avg Naturalness",
        "Avg Latency (ms)",
        "Avg Latency (s)",
        "P50 Latency (ms)",
        "P95 Latency (ms)",
        "Latency N",
        "Mean Total Tokens",
        "LLM Cost / Model-Turn",
        "Cost Currency",
        "Cost N",
    ]
    _header(ws, row, headers)
    model_header = row
    body = []
    for item in data["model_metrics"]:
        body.append(
            [
                item["display_model"],
                item.get("provider"),
                item["n"],
                _pct(item.get("intent_accuracy")),
                item.get("intent_n") or 0,
                _pct(item.get("action_accuracy")),
                item.get("action_n") or 0,
                _unavailable(item.get("factual_accuracy")),
                item.get("factual_n") or 0,
                _pct(item.get("grounded_pct")),
                item.get("grounded_n") or 0,
                _pct(item.get("unsupported_claim_rate")),
                item.get("unsupported_n") or 0,
                _pct(item.get("context_accuracy")),
                item.get("context_n") or 0,
                _pct(item.get("stage_accuracy")),
                item.get("stage_n") or 0,
                _unavailable(item.get("avg_relevance")),
                _unavailable(item.get("avg_completeness")),
                _unavailable(item.get("avg_clarity")),
                _unavailable(item.get("avg_conversational_quality")),
                _unavailable(item.get("avg_naturalness")),
                _unavailable(item.get("avg_latency_ms")),
                _latency_sec(item.get("avg_latency_ms")),
                _unavailable(item.get("p50_latency_ms")),
                _unavailable(item.get("p95_latency_ms")),
                item.get("latency_n") or 0,
                _unavailable(item.get("mean_total_tokens")),
                _unavailable(item.get("cost_per_model_turn")),
                item.get("cost_currency") or "unavailable",
                item.get("n") or 0,
            ]
        )
    formats = {i: "0.00%" for i in (4, 6, 10, 12, 14, 16)}
    formats[8] = "0.00"
    formats[23] = "0.0"
    formats[24] = "0.000"
    formats[25] = "0.0"
    formats[26] = "0.0"
    formats[28] = "0.0"
    formats[29] = "#,##0.000000"
    row = _write_rows(ws, row + 1, headers, body, formats)
    n_models = len(body)
    ws.auto_filter.ref = f"A{model_header}:{get_column_letter(len(headers))}{max(model_header, row - 1)}"
    _bar_chart(ws, "Model Quality Metrics (Intent Accuracy)", 1, 4, model_header, n_models, "AH3", "Intent accuracy")
    _bar_chart(ws, "LLM/model response latency (P50, ms)", 1, 25, model_header, n_models, "AH18", "Milliseconds")

    row += 2
    ws.cell(row, 1, "C. Language Coverage").font = SECTION_FONT
    row += 1
    lang_headers = ["Language", "Evaluation Runs", "Customer Turns", "Model-Turn Evaluations", "Models", "Status"]
    _header(ws, row, lang_headers)
    lang_header = row
    lang_body = [
        [r["language"], r["evaluation_runs"], r["customer_turns"], r["model_turn_evaluations"], r["models"], r["status"]]
        for r in data["language_coverage"]
    ]
    row = _write_rows(ws, row + 1, lang_headers, lang_body)
    ws.auto_filter.ref = f"A{lang_header}:F{max(lang_header, row - 1)}"
    for r_idx in range(lang_header + 1, row):
        status = str(ws.cell(r_idx, 6).value or "")
        if status.startswith("Completed"):
            ws.cell(r_idx, 6).fill = PatternFill("solid", fgColor=GREEN)
        elif status.startswith("Partial"):
            ws.cell(r_idx, 6).fill = PatternFill("solid", fgColor=AMBER)
        else:
            ws.cell(r_idx, 6).fill = PatternFill("solid", fgColor=RED)
    _bar_chart(ws, "Language Evaluation Coverage (model-turns)", 1, 4, lang_header, len(lang_body), "AH33", "Model-turn evaluations")

    row += 2
    ws.cell(row, 1, "D. Evidence Status").font = SECTION_FONT
    row += 1
    ws.cell(row, 1, "Do not interpret limited exploratory testing as production multilingual validation.").font = NOTE_FONT
    row += 1
    _header(ws, row, ["Item", "Status", "Evidence"])
    hindi_ids = ", ".join(recon.get("hindi_run_ids") or []) or "none"
    eng_ids = ", ".join(recon.get("english_controlled_run_ids") or []) or "none"
    evidence_rows = [
        ["English controlled outbound benchmark (48/48)", "Completed" if any("48" in (r.status or "") for r in data["runs"]) else "Partial", eng_ids],
        ["English golden dataset L001–L010", "Completed" if any(r.mode == "BATCH GOLDEN DATASET" for r in data["runs"]) else "Pending / Not Evaluated", "batch evaluation_results.csv"],
        ["Hindi interactive model evaluation", "Partial" if recon.get("hindi_run_ids") else "Pending / Not Evaluated", hindi_ids],
        ["Hindi H01–H08 sequence", "Partial" if any(r.language_track == "hindi" and r.evidence_class == "CONTROLLED BENCHMARK" for r in data["runs"]) else "Pending / Not Evaluated", "Run identity preserved; not production-validated"],
        ["Kannada model evaluation", "Partial" if any((o.get("language") == "Kannada") for o in data["observations"]) else "Pending / Not Evaluated", "K01–K08 interactive evidence only if present; not production-validated"],
        ["Mixed language model evaluation", "Partial" if any(o.get("language") == "Mixed" for o in data["observations"]) else "Pending / Not Evaluated", "Includes batch L010 smoke if present"],
        ["Twilio / real voice calls", "Not Evaluated in this workbook", "All recorded twilio flags are DISABLED"],
        ["Framework / pytest validation", "Completed (separate sheet)", f"{scale['framework_tests']} collected tests"],
    ]
    row = _write_rows(ws, row + 1, ["Item", "Status", "Evidence"], evidence_rows)

    row += 2
    ws.cell(row, 1, "Reconciliation").font = SECTION_FONT
    row += 1
    _header(ws, row, ["Check", "Result", "Detail"])
    recon_rows = [
        ["RUNS: inventory count = unique run IDs", recon["checks"]["run_count"], f"{recon['run_inventory_count']} = {recon['unique_run_ids']}"],
        ["MODEL-TURNS: detailed = aggregate", recon["checks"]["model_turn_count"], f"{recon['detailed_model_turns']} = {recon['aggregate_model_turns']}"],
        ["LANGUAGES: detailed = aggregate", recon["checks"]["language_count"], f"{recon['detailed_languages']} vs {recon['aggregate_languages']}"],
        ["Token totals", recon["checks"]["token_reconciliation"], str(recon["token_sums"])],
        ["Cost calculations", recon["checks"]["cost_reconciliation"], "; ".join(recon.get("cost_notes") or []) or "native-currency LLM-only"],
        ["English controlled benchmark identifiable", recon["checks"]["english_benchmark_identifiable"], eng_ids],
        ["Hindi evidence identifiable", recon["checks"]["hindi_evidence_identifiable"], hindi_ids],
        ["Framework tests excluded from model trials", recon["checks"]["framework_excluded_from_model_trials"], "fixture observations omitted from model-turn table"],
    ]
    _write_rows(ws, row + 1, ["Check", "Result", "Detail"], recon_rows)
    _autosize(ws, 48)
    ws.sheet_view.showGridLines = False
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.oddHeader.left.text = "AI Calling Agent — Master Evaluation Report"
    ws.print_title_rows = "1:2"


def _sheet_inventory(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("Evaluation Run Inventory")
    _title(ws, "Evaluation Run Inventory", "One row per unique RUN_ID. Session /export copies are not additional trials.")
    headers = [
        "Run ID",
        "Created At",
        "Evaluation Date",
        "Mode",
        "Conversation Mode",
        "Language Track",
        "Opening Mode",
        "Twilio Status",
        "Models Available",
        "Customer Turns",
        "Expected Evaluations",
        "Evaluated",
        "Skipped",
        "GT Coverage",
        "Evaluation Coverage",
        "Intentionally Unlabelled",
        "Unexpected Missing GT",
        "Run Classification",
        "Evidence Class",
        "Source Artifact(s)",
        "Status",
    ]
    _header(ws, 4, headers)
    body = []
    for bundle in data["runs"]:
        body.append(
            [
                bundle.run_id,
                bundle.created_at or "unavailable",
                bundle.evaluation_date or "unavailable",
                bundle.mode or "unavailable",
                bundle.conversation_mode or "unavailable",
                bundle.language_track or "english (implicit)",
                bundle.opening_mode or "unavailable",
                bundle.twilio_status or "DISABLED",
                ", ".join(bundle.models_available) or "unavailable",
                bundle.customer_turns,
                _unavailable(bundle.expected),
                _unavailable(bundle.evaluated),
                _unavailable(bundle.skipped),
                _pct(bundle.gt_coverage) if bundle.gt_coverage is not None else "unavailable",
                _pct(bundle.evaluation_coverage) if bundle.evaluation_coverage is not None else "unavailable",
                _unavailable(bundle.intentionally_unlabelled),
                _unavailable(bundle.unexpected_missing_gt),
                bundle.classification,
                bundle.evidence_class,
                "; ".join(bundle.source_artifacts),
                bundle.status,
            ]
        )
    _write_rows(ws, 5, headers, body, {14: "0.00%", 15: "0.00%"})
    last = 4 + len(body)
    ws.auto_filter.ref = f"A4:U{max(4, last)}"
    for r_idx in range(5, last + 1):
        klass = str(ws.cell(r_idx, 18).value or "")
        if klass.startswith("FRAMEWORK"):
            ws.cell(r_idx, 18).fill = PatternFill("solid", fgColor=AMBER)
        elif "MULTILINGUAL" in klass:
            ws.cell(r_idx, 18).fill = PatternFill("solid", fgColor="C5D9F1")
        fill = PatternFill("solid", fgColor=GREEN)
        if str(ws.cell(r_idx, 19).value) == "CONTROLLED BENCHMARK":
            ws.cell(r_idx, 19).fill = fill
    _autosize(ws)


def _sheet_detail(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("Detailed Model-Turn Results")
    _title(ws, "Detailed Model-Turn Results", "Audit table: one row per actual model response in a unique evaluation run. Fixtures excluded.")
    headers = [
        "Run ID",
        "Date",
        "Language",
        "Model",
        "Provider",
        "Turn",
        "Customer Input",
        "Expected Intent",
        "Predicted Intent",
        "Intent Correct",
        "Expected Action",
        "Predicted Action",
        "Action Correct",
        "Factual Correct",
        "Grounded",
        "Unsupported Claim",
        "Context Correct",
        "Stage Correct",
        "Relevance",
        "Completeness",
        "Clarity",
        "Conversational Quality",
        "Naturalness",
        "Language Understanding Correct",
        "Response Language Match",
        "Visit Sequence Correct",
        "Premature Confirmation",
        "Premature Slot Offer",
        "Latency ms",
        "Input Tokens",
        "Output Tokens",
        "Total Tokens",
        "Structured Success",
        "Error Type",
        "Evidence Class",
    ]
    _header(ws, 4, headers)
    body = []
    for row in data["observations"]:
        body.append(
            [
                row.get("run_id"),
                row.get("date") or "unavailable",
                row.get("language"),
                row.get("display_model"),
                row.get("provider"),
                row.get("turn"),
                row.get("customer_input"),
                row.get("expected_intent") if row.get("expected_intent") is not None else "unavailable",
                row.get("predicted_intent") if row.get("predicted_intent") is not None else "unavailable",
                row.get("intent_correct") if row.get("intent_correct") is not None else "unavailable",
                row.get("expected_action") if row.get("expected_action") is not None else "unavailable",
                row.get("predicted_action") if row.get("predicted_action") is not None else "unavailable",
                row.get("action_correct") if row.get("action_correct") is not None else "unavailable",
                row.get("factual_correct") if row.get("factual_correct") is not None else "unavailable",
                row.get("grounded") if row.get("grounded") is not None else "unavailable",
                row.get("unsupported_claim") if row.get("unsupported_claim") is not None else "unavailable",
                row.get("context_correct") if row.get("context_correct") is not None else "unavailable",
                row.get("stage_correct") if row.get("stage_correct") is not None else "unavailable",
                _unavailable(row.get("relevance")),
                _unavailable(row.get("completeness")),
                _unavailable(row.get("clarity")),
                _unavailable(row.get("conversational_quality")),
                _unavailable(row.get("naturalness")),
                row.get("language_understanding_correct") if row.get("language_understanding_correct") is not None else "unavailable",
                row.get("response_language_match") or "unavailable",
                row.get("visit_sequence_correct") if row.get("visit_sequence_correct") is not None else "unavailable",
                row.get("premature_confirmation") if row.get("premature_confirmation") is not None else "unavailable",
                row.get("premature_slot_offer") if row.get("premature_slot_offer") is not None else "unavailable",
                _unavailable(row.get("latency_ms")),
                _unavailable(row.get("input_tokens")),
                _unavailable(row.get("output_tokens")),
                _unavailable(row.get("total_tokens")),
                row.get("structured_success") if row.get("structured_success") is not None else "unavailable",
                row.get("error_type") or "",
                row.get("evidence_class"),
            ]
        )
    _write_rows(ws, 5, headers, body)
    last = 4 + len(body)
    ws.auto_filter.ref = f"A4:{get_column_letter(len(headers))}{max(4, last)}"
    _autosize(ws, 36)
    ws.column_dimensions["G"].width = 48


def _sheet_model_metrics(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("Aggregate Model Metrics")
    _title(ws, "Aggregate Model Metrics", "Independent per-model metrics. No overall score, winner, ranking, or recommendation.")
    headers = [
        "Model",
        "Provider",
        "N / sample size",
        "Intent Accuracy",
        "Intent Precision",
        "Intent Recall",
        "Intent Macro F1",
        "Intent N",
        "Action Accuracy",
        "Action Precision",
        "Action Recall",
        "Action Macro F1",
        "Action N",
        "Factual Accuracy",
        "Factual N",
        "Grounded Response %",
        "Grounded N",
        "Unsupported Claim Rate",
        "Unsupported N",
        "Context Accuracy",
        "Context N",
        "Stage Accuracy",
        "Stage N",
        "Average Relevance",
        "Average Completeness",
        "Average Clarity",
        "Average Conversational Quality",
        "Average Naturalness",
        "Average Latency (ms)",
        "P50 Latency (ms)",
        "P95 Latency (ms)",
        "Latency N",
        "Mean Input Tokens",
        "Mean Output Tokens",
        "Mean Total Tokens",
        "P50 Total Tokens",
        "P95 Total Tokens",
        "Token N",
    ]
    _header(ws, 4, headers)
    body = []
    for item in data["model_metrics"]:
        body.append(
            [
                item["display_model"],
                item.get("provider"),
                item["n"],
                _pct(item.get("intent_accuracy")),
                _pct(item.get("intent_precision")),
                _pct(item.get("intent_recall")),
                _pct(item.get("intent_macro_f1")),
                item.get("intent_n") or 0,
                _pct(item.get("action_accuracy")),
                _pct(item.get("action_precision")),
                _pct(item.get("action_recall")),
                _pct(item.get("action_macro_f1")),
                item.get("action_n") or 0,
                _unavailable(item.get("factual_accuracy")),
                item.get("factual_n") or 0,
                _pct(item.get("grounded_pct")),
                item.get("grounded_n") or 0,
                _pct(item.get("unsupported_claim_rate")),
                item.get("unsupported_n") or 0,
                _pct(item.get("context_accuracy")),
                item.get("context_n") or 0,
                _pct(item.get("stage_accuracy")),
                item.get("stage_n") or 0,
                _unavailable(item.get("avg_relevance")),
                _unavailable(item.get("avg_completeness")),
                _unavailable(item.get("avg_clarity")),
                _unavailable(item.get("avg_conversational_quality")),
                _unavailable(item.get("avg_naturalness")),
                _unavailable(item.get("avg_latency_ms")),
                _unavailable(item.get("p50_latency_ms")),
                _unavailable(item.get("p95_latency_ms")),
                item.get("latency_n") or 0,
                _unavailable(item.get("mean_input_tokens")),
                _unavailable(item.get("mean_output_tokens")),
                _unavailable(item.get("mean_total_tokens")),
                _unavailable(item.get("p50_total_tokens")),
                _unavailable(item.get("p95_total_tokens")),
                item.get("total_token_n") or 0,
            ]
        )
    pct_cols = {i: "0.00%" for i in (4, 5, 6, 7, 9, 10, 11, 12, 16, 18, 20, 22)}
    _write_rows(ws, 5, headers, body, pct_cols)
    last = 4 + len(body)
    ws.auto_filter.ref = f"A4:{get_column_letter(len(headers))}{max(4, last)}"
    _autosize(ws)
    n_models = len(body)
    _bar_chart(ws, "Token Usage Comparison (mean total tokens)", 1, 35, 4, n_models, "A22", "Mean total tokens")


def _sheet_errors(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("Error Analysis")
    _title(ws, "Error Analysis", "Categories are emitted only when supported by actual evaluation fields. Not a ranking.")
    ws["A3"] = "Summary"
    ws["A3"].font = SECTION_FONT
    sum_headers = ["Model", "Error Type", "Count", "Percentage"]
    _header(ws, 4, sum_headers)
    sum_body = [
        [r["model"], r["error_type"], r["count"], r["percentage"]]
        for r in data["errors_summary"]
    ]
    _write_rows(ws, 5, sum_headers, sum_body, {4: "0.00%"})
    sum_last = 4 + max(len(sum_body), 1)
    ws.auto_filter.ref = f"A4:D{sum_last}"
    if sum_body:
        _bar_chart(ws, "Error Distribution (count)", 2, 3, 4, len(sum_body), "F3", "Count")

    start = sum_last + 3
    ws.cell(start, 1, "Examples").font = SECTION_FONT
    headers = [
        "Model",
        "Language",
        "Error Type",
        "Count",
        "Percentage",
        "Run ID",
        "Turn",
        "Customer Input",
        "Expected",
        "Predicted",
        "Notes",
    ]
    _header(ws, start + 1, headers)
    body = [
        [
            r["model"],
            r["language"],
            r["error_type"],
            r["count"],
            r["percentage"],
            r["run_id"],
            r["turn"],
            r["customer_input"],
            r.get("expected"),
            r.get("predicted"),
            r.get("notes"),
        ]
        for r in data["errors_detail"]
    ]
    _write_rows(ws, start + 2, headers, body, {5: "0.00%"})
    last = start + 1 + len(body)
    # second filter not supported on same sheet; keep freeze on summary
    _autosize(ws, 36)


def _sheet_latency(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("Latency & Token Usage")
    _title(
        ws,
        "Latency & Token Usage",
        "Latency is LLM/model response latency only. It is not total end-to-end voice latency (STT/TTS/Twilio excluded).",
    )
    headers = [
        "Model",
        "Observation Count",
        "Average Latency (ms)",
        "Average Latency (s)",
        "Minimum (ms)",
        "P25 (ms)",
        "P50 (ms)",
        "P75 (ms)",
        "P95 (ms)",
        "Maximum (ms)",
        "Input Tokens Mean",
        "Input Tokens P50",
        "Input Tokens P95",
        "Output Tokens Mean",
        "Output Tokens P50",
        "Output Tokens P95",
        "Total Tokens Mean",
        "Total Tokens P50",
        "Total Tokens P95",
    ]
    _header(ws, 4, headers)
    body = []
    for item in data["model_metrics"]:
        body.append(
            [
                item["display_model"],
                item.get("latency_n") or 0,
                _unavailable(item.get("avg_latency_ms")),
                _latency_sec(item.get("avg_latency_ms")),
                _unavailable(item.get("min_latency_ms")),
                _unavailable(item.get("p25_latency_ms")),
                _unavailable(item.get("p50_latency_ms")),
                _unavailable(item.get("p75_latency_ms")),
                _unavailable(item.get("p95_latency_ms")),
                _unavailable(item.get("max_latency_ms")),
                _unavailable(item.get("mean_input_tokens")),
                _unavailable(item.get("p50_input_tokens")),
                _unavailable(item.get("p95_input_tokens")),
                _unavailable(item.get("mean_output_tokens")),
                _unavailable(item.get("p50_output_tokens")),
                _unavailable(item.get("p95_output_tokens")),
                _unavailable(item.get("mean_total_tokens")),
                _unavailable(item.get("p50_total_tokens")),
                _unavailable(item.get("p95_total_tokens")),
            ]
        )
    _write_rows(ws, 5, headers, body)
    last = 4 + len(body)
    ws.auto_filter.ref = f"A4:S{max(4, last)}"
    _bar_chart(ws, "LLM/model response latency (average ms)", 1, 3, 4, len(body), "A18", "Milliseconds")
    _bar_chart(ws, "Token Usage Comparison (mean total)", 1, 17, 4, len(body), "J18", "Tokens")

    start = last + 12
    ws.cell(start, 1, "Breakdown by language (shown where at least 3 observations exist)").font = SECTION_FONT
    lang_headers = ["Model", "Language", "Observation Count", "Average Latency (ms)", "P50 (ms)", "P95 (ms)", "Mean Total Tokens"]
    _header(ws, start + 1, lang_headers)
    lang_body = [
        [r["model"], r["language"], r["observation_count"], r["average_latency_ms"], r["p50_latency_ms"], r["p95_latency_ms"], r["mean_total_tokens"]]
        for r in data["latency_by_language"]
    ]
    _write_rows(ws, start + 2, lang_headers, lang_body)
    _autosize(ws)


def _sheet_cost(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("Cost Analysis")
    _title(ws, "Cost Analysis — LLM-only estimated cost", data["pricing_note"])
    ws["A3"] = "CONFIGURATION / ASSUMPTION"
    ws["A3"].font = SECTION_FONT
    assume_headers = ["Model ID", "Label", "Input Price / 1M", "Output Price / 1M", "Currency", "Source"]
    _header(ws, 4, assume_headers)
    assume_body = [
        [mid, spec["label"], spec["input_per_1m"], spec["output_per_1m"], spec["currency"], "PHASE LLM-EVAL.5.1 assumption — not in historical artifacts"]
        for mid, spec in data["pricing_assumptions"].items()
    ]
    _write_rows(ws, 5, assume_headers, assume_body)
    start = 5 + len(assume_body) + 2
    ws.cell(start, 1, "Estimated cost from measured tokens").font = SECTION_FONT
    headers = [
        "Model",
        "Evaluation Observations",
        "Mean Input Tokens",
        "Mean Output Tokens",
        "Input Price / 1M Tokens",
        "Output Price / 1M Tokens",
        "Currency",
        "Estimated LLM Cost / Model-Turn",
        "Estimated Cost / 1,000 Model-Turns",
        "Estimated Cost / 10,000 Model-Turns",
        "Notes",
    ]
    _header(ws, start + 1, headers)
    body = [
        [
            r["model"],
            r["evaluation_observations"],
            _unavailable(r.get("mean_input_tokens")),
            _unavailable(r.get("mean_output_tokens")),
            _unavailable(r.get("input_price_per_1m")),
            _unavailable(r.get("output_price_per_1m")),
            r.get("currency") or "unavailable",
            _unavailable(r.get("estimated_llm_cost_per_model_turn")),
            _unavailable(r.get("estimated_cost_per_1000")),
            _unavailable(r.get("estimated_cost_per_10000")),
            r.get("notes"),
        ]
        for r in data["cost_rows"]
    ]
    _write_rows(ws, start + 2, headers, body, {8: "#,##0.000000", 9: "#,##0.00", 10: "#,##0.00"})
    _bar_chart(ws, "LLM Cost Comparison (native currency / model-turn)", 1, 8, start + 1, len(body), "A28", "Cost / model-turn")
    note_row = start + 2 + len(body) + 2
    ws.cell(note_row, 1, "INR is used for Sarvam. USD-priced models remain in USD because no FX rate is stored in project evidence. Do not compare INR and USD bars as a ranking.")
    ws.cell(note_row, 1).font = NOTE_FONT
    _autosize(ws)


def _sheet_multilingual(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("Multilingual Evaluation")
    _title(
        ws,
        "Multilingual Evaluation",
        "Conservative evidence split. Framework FakeProvider dumps are not converted into model performance. Not production-validated.",
    )
    headers = [
        "Language",
        "Model",
        "Runs",
        "Customer Turns",
        "Model-Turn Evaluations",
        "Language Understanding Accuracy",
        "Response Language Match %",
        "Intent Accuracy",
        "Action Accuracy",
        "Factual Accuracy",
        "Grounded Response %",
        "Avg Latency (ms)",
        "Mean Tokens",
        "Status",
        "Evidence Classification",
        "N",
    ]
    _header(ws, 4, headers)
    body = [
        [
            r["language"],
            r["model"],
            r["runs"],
            r["customer_turns"],
            r["model_turn_evaluations"],
            _pct(r.get("language_understanding_accuracy")),
            _pct(r.get("response_language_match_pct")),
            _pct(r.get("intent_accuracy")),
            _pct(r.get("action_accuracy")),
            _unavailable(r.get("factual_accuracy")),
            _pct(r.get("grounded_pct")),
            _unavailable(r.get("avg_latency_ms")),
            _unavailable(r.get("mean_tokens")),
            r["status"],
            r.get("evidence_classification"),
            r.get("n") if r.get("n") is not None else r["model_turn_evaluations"],
        ]
        for r in data["multilingual_rows"]
    ]
    pct = {i: "0.00%" for i in (6, 7, 8, 9, 11)}
    _write_rows(ws, 5, headers, body, pct)
    last = 4 + len(body)
    ws.auto_filter.ref = f"A4:P{max(4, last)}"
    for r_idx in range(5, last + 1):
        status = str(ws.cell(r_idx, 14).value or "")
        if "NOT YET" in status:
            ws.cell(r_idx, 14).fill = PatternFill("solid", fgColor=RED)
        elif "EXPLORATORY" in status:
            ws.cell(r_idx, 14).fill = PatternFill("solid", fgColor=AMBER)
        elif "CONTROLLED" in status:
            ws.cell(r_idx, 14).fill = PatternFill("solid", fgColor=GREEN)
    _autosize(ws)


def _sheet_trials(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("Trial Test Dataset")
    _title(ws, "Trial / Test Dataset", "Normalized inventory of what models were actually tested on.")
    headers = [
        "Test ID / Run ID",
        "Language",
        "Scenario",
        "Turn",
        "Customer Input",
        "Expected Intent",
        "Expected Action",
        "Expected Stage",
        "Expected Language",
        "Evaluation Type",
        "Ground Truth Available",
        "Source",
    ]
    _header(ws, 4, headers)
    body = [
        [
            r["test_id"],
            r["language"],
            r["scenario"],
            r["turn"],
            r["customer_input"],
            r.get("expected_intent") or "unavailable",
            r.get("expected_action") or "unavailable",
            r.get("expected_stage") or "unavailable",
            r.get("expected_language") or "unavailable",
            r["evaluation_type"],
            r["ground_truth_available"],
            r["source"],
        ]
        for r in data["trial_rows"]
    ]
    _write_rows(ws, 5, headers, body)
    last = 4 + len(body)
    ws.auto_filter.ref = f"A4:L{max(4, last)}"
    _autosize(ws, 40)


def _sheet_framework(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("Framework Validation")
    _title(
        ws,
        "Framework Validation",
        "Software/evaluation-framework evidence only. This sheet is NOT model accuracy and is not counted as model-turn evaluations.",
    )
    headers = ["Test Suite", "Test Count", "Passed", "Failed", "Date", "Scope", "Notes"]
    _header(ws, 4, headers)
    body = [
        [r["test_suite"], r["test_count"], r["passed"], r["failed"], r.get("date") or "unavailable", r["scope"], r["notes"]]
        for r in data["pytest_rows"]
    ]
    fw_runs = [r for r in data["runs"] if r.classification == CLASS_FRAMEWORK]
    body.append(
        [
            "interactive_runs (FakeProvider / DRY_RUN dumps)",
            len(fw_runs),
            len(fw_runs),
            0,
            "see inventory",
            "Pytest and dry-run session folders",
            "Excluded from unique evaluation-run and model-turn counts",
        ]
    )
    _write_rows(ws, 5, headers, body)
    last = 4 + len(body)
    ws.auto_filter.ref = f"A4:G{max(4, last)}"
    total = sum(int(r.get("test_count") or 0) for r in data["pytest_rows"])
    ws.cell(last + 2, 1, f"Total collected pytest tests: {total}. These MUST NOT be reported as model evaluation trials.")
    ws.cell(last + 2, 1).font = NOTE_FONT
    _autosize(ws)


def _sheet_lineage(wb: Workbook, data: dict[str, Any], recon: dict[str, Any]) -> None:
    ws = wb.create_sheet("Data Lineage Sources")
    _title(ws, "Data Lineage / Sources", "Audit trail of every discovered evaluation artifact.")
    headers = ["Source File", "Path", "File Type", "Run IDs Found", "Records", "Date Range", "Used For", "Included / Excluded", "Reason"]
    _header(ws, 4, headers)
    body = []
    for art in data["artifacts"]:
        dates = "unavailable"
        if art.date_start or art.date_end:
            dates = f"{art.date_start or '?'} to {art.date_end or '?'}"
        body.append(
            [
                Path(art.path).name,
                art.path,
                art.file_type,
                ", ".join(art.run_ids) or "—",
                art.records,
                dates,
                art.used_for,
                art.included,
                art.reason,
            ]
        )
    _write_rows(ws, 5, headers, body)
    last = 4 + len(body)
    ws.auto_filter.ref = f"A4:I{max(4, last)}"
    rec_row = last + 3
    ws.cell(rec_row, 1, "RECONCILIATION").font = SECTION_FONT
    rec_headers = ["Check", "Result", "Detail"]
    _header(ws, rec_row + 1, rec_headers)
    rec_body = [
        ["RUNS: Inventory count = Unique run IDs", recon["checks"]["run_count"], f"{recon['run_inventory_count']} = {recon['unique_run_ids']}"],
        ["MODEL-TURNS: Detailed count = Aggregate count", recon["checks"]["model_turn_count"], f"{recon['detailed_model_turns']} = {recon['aggregate_model_turns']}"],
        ["LANGUAGES: Detailed = Aggregate", recon["checks"]["language_count"], f"{recon['detailed_languages']} vs {recon['aggregate_languages']}"],
        ["Token reconciliation", recon["checks"]["token_reconciliation"], str(recon["token_sums"])],
        ["Cost reconciliation", recon["checks"]["cost_reconciliation"], "; ".join(recon.get("cost_notes") or []) or "PASS"],
        ["Overall", recon["overall"], "All required checks"],
    ]
    _write_rows(ws, rec_row + 2, rec_headers, rec_body)
    _autosize(ws, 50)


def write_workbook(path: Path, data: dict[str, Any], recon: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    _sheet_exec(wb, data, recon)
    _sheet_inventory(wb, data)
    _sheet_detail(wb, data)
    _sheet_model_metrics(wb, data)
    _sheet_errors(wb, data)
    _sheet_latency(wb, data)
    _sheet_cost(wb, data)
    _sheet_multilingual(wb, data)
    _sheet_trials(wb, data)
    _sheet_framework(wb, data)
    _sheet_lineage(wb, data, recon)
    wb.save(path)
    return path

"""Write the per-call scorecard. This does not edit the calling agent."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

IST = timezone(timedelta(hours=5, minutes=30))

from .measured import recommended_change

CRITERION_LABELS = {
    "intent_accuracy": "Intent",
    "action_accuracy": "Next step",
    "factual_accuracy": "Facts",
    "grounding": "Grounded in project facts",
    "unsupported_claims": "Unsupported claims",
    "context_retention": "Context",
    "conversation_stage_accuracy": "Stage",
    "conversation_quality": "Conversation quality",
    "repeated_question": "Repeated question",
    "premature_site_visit": "Premature site visit",
    "multiple_followups": "One question at a time",
    "fixed_configuration_line": "Configuration question repeated",
}

STAGE_LABELS = {
    "greeting": "Greeting",
    "qualify": "Qualify",
    "answer_faq": "Answer questions",
    "propose_slot": "Offer a visit",
    "visit_day": "Choose a day",
    "visit_pick_slot": "Choose a time",
    "closed": "Closed",
}


def _seconds(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value) / 1000:.1f} s"


def _label(name: str) -> str:
    return CRITERION_LABELS.get(name, name.replace("_", " ").capitalize())


def _when(record: dict[str, Any]) -> str:
    raw = record.get("started_at") or ""
    try:
        moment = datetime.fromisoformat(str(raw))
    except ValueError:
        return ""
    local = moment.astimezone(IST)
    hour = local.strftime("%I").lstrip("0") or "12"
    return f"{local.day} {local.strftime('%b %Y')}, {hour}:{local.strftime('%M %p')} IST"


def _latency_row(label: str, stats: dict[str, Any]) -> str:
    if not stats.get("n"):
        return f"| {label} | Not applicable | | | |"
    return (
        f"| {label} | {_seconds(stats.get('avg_ms'))} | "
        f"{_seconds(stats.get('p50_ms'))} | {_seconds(stats.get('p95_ms'))} | {stats['n']} |"
    )


def _turn_list(values: list[Any]) -> str:
    items = [str(item) for item in values or []]
    if not items:
        return "None"
    prefix = "Turn" if len(items) == 1 else "Turns"
    return f"{prefix} {', '.join(items)}"


def _stages(path: list[str]) -> str:
    if not path:
        return "None"
    return " → ".join(STAGE_LABELS.get(name, name) for name in path)


def _repeat_line(repeats: list[dict[str, Any]]) -> str:
    if not repeats:
        return "None"
    return "; ".join(
        f"Turn {item['turn_index']} repeated turn {item['first_turn_index']}"
        for item in repeats
    )


def _sentences(text: str) -> str:
    protected = text.replace("vs. ", "vs\u0000 ")
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
    restored = [part.replace("vs\u0000 ", "vs. ").strip() for part in parts if part.strip()]
    return "\n\n".join(restored)


def _tally(criteria: dict[str, Any]) -> str:
    failed = passed = 0
    for item in criteria.values():
        if item.get("na") or item.get("score") is None:
            continue
        if item.get("score"):
            passed += 1
        else:
            failed += 1
    if not failed and not passed:
        return ""
    return f"{failed} failed, {passed} passed"


def _cell(text: str) -> str:
    return " ".join((text or "").replace("|", "/").split())


def _result(item: dict[str, Any]) -> str:
    if item.get("na") or item.get("score") is None:
        return "Not applicable"
    return "Pass" if item.get("score") else "Fail"


def _ordered(criteria: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    def rank(item: dict[str, Any]) -> int:
        if item.get("na") or item.get("score") is None:
            return 2
        return 0 if not item.get("score") else 1

    return sorted((criteria or {}).items(), key=lambda pair: rank(pair[1]))


def _score_table(criteria: dict[str, Any]) -> list[str]:
    lines = [
        "## Scores",
        "",
        "| Score | Result | Turns |",
        "| --- | --- | --- |",
    ]
    for name, item in _ordered(criteria):
        turns = ", ".join(str(index) for index in item.get("turn_indices") or []) or "—"
        lines.append(f"| {_label(name)} | {_result(item)} | {turns} |")
    lines.append("")
    return lines


def _findings(criteria: dict[str, Any]) -> list[str]:
    blocks: list[str] = []
    for name, item in _ordered(criteria):
        evidence = (item.get("evidence") or "").strip()
        if not evidence:
            continue
        numbers = [str(index) for index in item.get("turn_indices") or []]
        if len(numbers) == 1:
            where = f"Turn {numbers[0]}. "
        elif numbers:
            where = f"Turns {', '.join(numbers)}. "
        else:
            where = ""
        blocks.extend([f"### {_label(name)} — {_result(item)}", "", f"{where}{evidence}", ""])
    if not blocks:
        return []
    return ["## Findings", "", *blocks]


def _conversation_table(record: dict[str, Any]) -> list[str]:
    turns = record.get("turns") or []
    if not turns:
        return []
    lines = [
        "## Conversation",
        "",
        "| Turn | Customer | Agent | Stage |",
        "| --- | --- | --- | --- |",
    ]
    for turn in turns:
        customer = _cell(turn.get("customer") or "—")
        reply = _cell(turn.get("reply") or "—")
        stage = STAGE_LABELS.get(turn.get("stage") or "", turn.get("stage") or "—")
        lines.append(f"| {turn.get('turn_index')} | {customer} | {reply} | {stage} |")
    lines.append("")
    return lines


def render_scorecard(
    record: dict[str, Any],
    measured: dict[str, Any],
    judged: dict[str, Any],
) -> str:
    latency = measured.get("latency") or {}
    fit = measured.get("last_answer_fit") or []
    applicable = [item for item in fit if not item.get("na")]
    fitted = [item for item in applicable if item.get("fit")]
    fit_line = "Not applicable" if not applicable else f"{len(fitted)} of {len(applicable)}"
    change = recommended_change(measured) or (judged.get("recommended_change") or "").strip()
    if not change:
        change = "No conversation change is indicated by this call."

    when = _when(record) or "—"
    tally = _tally(judged.get("criteria") or {}) if judged.get("status") == "ok" else "Judge unavailable"
    lines = [
        "# Call scorecard",
        "",
        "| Field | Detail |",
        "| --- | --- |",
        f"| When | {when} |",
        f"| Call | {record.get('call_id')} |",
        f"| Caller | {record.get('caller_model')} |",
        f"| Judge | {judged.get('model') or 'unavailable'} |",
        f"| Result | {tally or '—'} |",
        "",
        "## What to change",
        "",
        change,
        "",
        "## Timing",
        "",
        "| Step | Average | Median | 95th percentile | Count |",
        "| --- | ---: | ---: | ---: | ---: |",
        _latency_row("Speech to text", latency.get("stt") or {}),
        _latency_row("Model", latency.get("llm") or {}),
        _latency_row("Voice", latency.get("tts") or {}),
        _latency_row("Whole turn", latency.get("turn") or {}),
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Repeated questions | {_repeat_line(measured.get('repeated_questions') or [])} |",
        f"| Answers that fit the question | {fit_line} |",
        f"| Path | {_stages(measured.get('stage_path') or [])} |",
        f"| Site visit offered too early | {_turn_list(measured.get('premature_site_visits') or [])} |",
        f"| Extra question in one reply | {_turn_list(measured.get('extra_questions') or [])} |",
        f"| Configuration question repeated | {_turn_list(measured.get('repeated_configuration_prompts') or [])} |",
        "",
    ]
    if judged.get("status") != "ok":
        lines.extend(
            [
                "## Scores",
                "",
                f"Judged section unavailable: {judged.get('reason')}",
                "",
            ]
        )
    else:
        summary = (judged.get("summary") or "").strip()
        if summary:
            lines.extend(["## What happened", "", _sentences(summary), ""])
        criteria = judged.get("criteria") or {}
        lines.extend(_score_table(criteria))
        lines.extend(_findings(criteria))
    lines.extend(_conversation_table(record))
    return "\n".join(lines).rstrip() + "\n"


def write_scorecard(
    record: dict[str, Any],
    measured: dict[str, Any],
    judged: dict[str, Any],
    runs_dir: Path,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.get('call_id')}_scorecard.md"
    path.write_text(render_scorecard(record, measured, judged), encoding="utf-8")
    return path

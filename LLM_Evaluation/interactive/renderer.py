from __future__ import annotations

import json
import re
from typing import Any

from .evaluator import _mark, format_num, format_pct, session_summary
from .schemas import COMMANDS, DISPLAY_NAMES, DISPLAY_ORDER, SHORT_NAMES, TurnResponse

RULE = "=" * 60
LINE = "-" * 60
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+|api[_-]?key['\"]?\s*[:=]\s*['\"]?[^'\"\s]+)", re.I)


def redact(text: Any) -> str:
    value = "" if text is None else str(text)
    return SECRET_RE.sub("[REDACTED]", value)


def banner() -> str:
    return "\n".join(
        [
            RULE,
            "          AI CALLING AGENT",
            "      INTERACTIVE LLM COMPARISON LAB",
            RULE,
            "",
            "Models:",
            "",
            "[1] Sarvam — Conversational",
            "[2] Sarvam — Flagship",
            "[3] Claude — Sonnet",
            "[4] Claude — Flagship",
            "",
            "Mode:",
            "LOCAL INTERACTIVE EVALUATION",
            "",
            "Twilio:",
            "DISABLED",
            "",
            "MODE: OFFLINE / INTERACTIVE EVALUATION",
            "TWILIO: DISABLED",
            LINE,
        ]
    )


def help_text() -> str:
    lines = [RULE, "COMMANDS", RULE]
    for cmd, desc in COMMANDS.items():
        lines.append(f"    {cmd:<10} {desc}")
    return "\n".join(lines)


def render_turn(turn_index: int, utterance: str, responses: list[TurnResponse], max_chars: int = 0) -> str:
    blocks = [
        RULE,
        f"TURN {turn_index:02d}",
        "CUSTOMER",
        RULE,
        "",
        utterance,
        "",
    ]
    by_alias = {item.alias: item for item in responses}
    for i, alias in enumerate(DISPLAY_ORDER, start=1):
        resp = by_alias.get(alias)
        blocks.append(LINE)
        blocks.append(f"[{i}] {DISPLAY_NAMES[alias]}")
        blocks.append(LINE)
        blocks.append("")
        if resp is None:
            blocks.append("STATUS: UNAVAILABLE")
            blocks.append("")
            continue
        if resp.error_type:
            blocks.append("STATUS: ERROR")
            blocks.append(f"ERROR: {redact(resp.error_message or resp.error_type)}")
            blocks.append("")
            continue
        answer = resp.answer or resp.raw_response or ""
        if max_chars and len(answer) > max_chars:
            answer = answer[:max_chars] + "\n... [truncated]"
        blocks.append(answer)
        blocks.append("")
        meta = []
        if resp.intent:
            meta.append(f"intent={resp.intent}")
        if resp.action:
            meta.append(f"action={resp.action}")
        if resp.language:
            meta.append(f"language={resp.language}")
        if resp.latency_ms is not None:
            meta.append(f"latency={resp.latency_ms / 1000:.2f}s")
        if meta:
            blocks.append("(" + ", ".join(meta) + ")")
            blocks.append("")
    blocks.append(RULE)
    return "\n".join(blocks)


def render_history(turns: list[dict[str, Any]]) -> str:
    lines = [RULE, "CONVERSATION HISTORY", RULE, ""]
    if not turns:
        lines.append("(empty)")
        lines.append("")
        return "\n".join(lines)
    for turn in turns:
        lines.append(f"Turn {turn['index']:02d}")
        lines.append("Customer:")
        lines.append(turn["utterance"])
        lines.append("")
    lines.append(LINE)
    return "\n".join(lines)


def render_models(slots: list[Any]) -> str:
    lines = [RULE, "CONFIGURED MODELS", RULE, ""]
    for slot in slots:
        lines.append(DISPLAY_NAMES.get(slot.alias, slot.alias))
        lines.append(f"Provider: {slot.provider.title() if slot.provider != 'claude' else 'Anthropic'}")
        lines.append(f"Model ID: {slot.model_id or '(not configured)'}")
        lines.append(f"Status: {slot.status}")
        if slot.error:
            lines.append(f"Error: {redact(slot.error)}")
        lines.append("")
    return "\n".join(lines)


def render_status(slots: list[Any], last_responses: list[TurnResponse] | None) -> str:
    lines = [
        "Model                  Status       Latency",
        LINE,
    ]
    last = {item.alias: item for item in (last_responses or [])}
    labels = {
        "sarvam_conversational": "Sarvam Conversational",
        "sarvam_flagship": "Sarvam Flagship",
        "claude_sonnet": "Claude Sonnet",
        "claude_flagship": "Claude Flagship",
    }
    for slot in slots:
        resp = last.get(slot.alias)
        if resp is None:
            status = slot.status
            latency = "N/A"
        elif resp.error_type:
            status = "ERROR"
            latency = f"{resp.latency_ms / 1000:.2f} sec" if resp.latency_ms is not None else "N/A"
            if resp.error_type in {"MODEL_NOT_FOUND", "AUTH_ERROR"}:
                status = resp.error_type
        else:
            status = "SUCCESS"
            latency = f"{resp.latency_ms / 1000:.2f} sec" if resp.latency_ms is not None else "N/A"
        lines.append(f"{labels[slot.alias]:<22} {status:<12} {latency}")
    return "\n".join(lines)


def render_evaluation(ev: dict[str, Any]) -> str:
    lines = [
        RULE,
        f"TURN {ev['turn_index']:02d} — EVALUATION",
        RULE,
        "",
        "Customer:",
        ev["utterance"],
        "",
        f"Ground Truth: {ev.get('ground_truth') or 'NOT AVAILABLE'}",
        f"Detected Language: {ev.get('detected_language')}",
        "",
    ]
    by_alias = {m["alias"]: m for m in ev.get("models") or []}
    for alias in DISPLAY_ORDER:
        model = by_alias.get(alias) or {}
        lines.append(LINE)
        lines.append(DISPLAY_NAMES[alias])
        lines.append(LINE)
        lines.append("")
        if model.get("api_error"):
            lines.append("STATUS: ERROR")
            lines.append(f"ERROR: {redact(model.get('error_message'))}")
            lines.append("Note: API errors are not quality failures.")
            lines.append("")
            continue
        lines.append(f"Intent:")
        lines.append(f"{model.get('predicted_intent') or 'N/A':<22} {_check(model.get('intent_correct'))}")
        lines.append("")
        lines.append("Action:")
        lines.append(f"{model.get('predicted_action') or 'N/A':<22} {_check(model.get('action_correct'))}")
        lines.append("")
        lines.append("Stage:")
        lines.append(f"{model.get('model_stage') or 'N/A':<22} {_check(model.get('stage_correct'))}")
        lines.append("")
        lines.append("Grounding:")
        grounded = "Grounded" if model.get("grounded") else "Not grounded"
        lines.append(f"{grounded:<22} {_check(model.get('grounded'))}")
        lines.append(f"Grounding Score: {format_num(model.get('grounding_score'))}")
        lines.append("")
        unsupported = model.get("unsupported_claims") or []
        lines.append("Unsupported Claims:")
        lines.append(str(len(unsupported) if isinstance(unsupported, list) else unsupported))
        lines.append("")
        lines.append("Hallucination:")
        lines.append(f"{_mark(model.get('hallucination'))}")
        lines.append("")
        lines.append("Context Retention:")
        ctx_ok = model.get("context_used") and not model.get("context_error")
        lines.append(f"{'Correct' if ctx_ok else 'Check':<22} {_check(ctx_ok)}")
        lines.append("")
        lines.append("Relevance:")
        lines.append(_score(model.get("relevance")))
        lines.append("")
        lines.append("Completeness:")
        lines.append(_score(model.get("completeness")))
        lines.append("")
        lines.append("Clarity:")
        lines.append(_score(model.get("clarity")))
        lines.append("")
        lines.append("Conversational Appropriateness:")
        lines.append(_score(model.get("conversational")))
        lines.append("")
    lines.append(RULE)
    return "\n".join(lines)


def render_session_summary(evaluations: list[dict[str, Any]]) -> str:
    summary = session_summary(evaluations)
    cols = [SHORT_NAMES[a] for a in DISPLAY_ORDER]
    lines = [
        RULE,
        "SESSION EVALUATION SUMMARY",
        RULE,
        "",
        f"{'Metric':<30}" + "".join(f"{c:>12}" for c in cols),
        "-" * 78,
    ]

    def row(label: str, getter) -> None:
        cells = []
        for alias in DISPLAY_ORDER:
            cells.append(f"{getter(summary[alias]):>12}")
        lines.append(f"{label:<30}" + "".join(cells))

    row("Intent Accuracy", lambda s: format_pct(s["intent"]["accuracy"]))
    row("Intent Precision", lambda s: format_pct(s["intent"]["precision"]))
    row("Intent Recall", lambda s: format_pct(s["intent"]["recall"]))
    row("Intent Macro F1", lambda s: format_pct(s["intent"]["f1"]))
    row("Action Accuracy", lambda s: format_pct(s["action"]["accuracy"]))
    row("Action Precision", lambda s: format_pct(s["action"]["precision"]))
    row("Action Recall", lambda s: format_pct(s["action"]["recall"]))
    row("Action Macro F1", lambda s: format_pct(s["action"]["f1"]))
    row("Factual Accuracy", lambda s: format_pct(s["factual_accuracy"]))
    row("Grounded Response %", lambda s: format_pct(s["grounded_pct"]))
    row("Unsupported Claim Rate", lambda s: format_pct(s["unsupported_rate"]))
    row("Context Accuracy", lambda s: format_pct(s["context_accuracy"]))
    row("Stage Accuracy", lambda s: format_pct(s["stage_accuracy"]))
    row("Avg Relevance", lambda s: format_num(s["avg_relevance"]))
    row("Avg Completeness", lambda s: format_num(s["avg_completeness"]))
    row("Avg Clarity", lambda s: format_num(s["avg_clarity"]))
    row("Avg Conversational Quality", lambda s: format_num(s["avg_conversational"]))
    row("Avg Latency", lambda s: format_num(s["avg_latency"]))
    row("P50 Latency", lambda s: format_num(s["p50_latency"]))
    row("P95 Latency", lambda s: format_num(s["p95_latency"]))
    lines += ["", "Independent per-model metrics. No winner ranking.", RULE]
    return "\n".join(lines)


def render_debug(payload: dict[str, Any]) -> str:
    safe = json.loads(json.dumps(payload, default=str))
    text = json.dumps(safe, indent=2, ensure_ascii=False)
    return "\n".join([RULE, "DEBUG — LATEST TURN", RULE, redact(text), RULE])


def _check(value: bool | None) -> str:
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "N/A"


def _score(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{value}/5"

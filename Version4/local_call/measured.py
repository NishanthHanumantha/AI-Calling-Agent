"""Scores read from a saved call. These do not call a model."""

from __future__ import annotations

import re
from statistics import median
from typing import Any


def _numbers(values: list[int | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _percentile(values: list[float], pct: float) -> float | None:
    nums = sorted(values)
    if not nums:
        return None
    if len(nums) == 1:
        return round(nums[0], 2)
    index = int(round((len(nums) - 1) * pct))
    return round(nums[index], 2)


def _stats(values: list[int | None]) -> dict[str, Any]:
    nums = _numbers(values)
    if not nums:
        return {"n": 0, "avg_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "n": len(nums),
        "avg_ms": round(sum(nums) / len(nums), 2),
        "p50_ms": _percentile(nums, 0.50) if len(nums) > 1 else round(median(nums), 2),
        "p95_ms": _percentile(nums, 0.95),
    }


def _questions(text: str) -> list[str]:
    return [part.strip() for part in re.findall(r"[^?]*\?", text or "") if part.strip()]


def _norm(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _focus(question: str) -> str:
    parts = re.split(r"\.\s+", (question or "").strip())
    return parts[-1] if parts else question


def _question_kind(question: str) -> str:
    text = _focus(question).lower()
    if re.search(r"which configuration|one, two, three|bhk homes", text):
        return "configuration"
    if re.search(r"\b(luxe|grande|size|sizes|sq)\b", text):
        return "size"
    if re.search(r"what would you like", text):
        return "open"
    if re.search(r"\b(would you|do you|is now|shall i)\b", text):
        return "yes_no"
    return "other"


def _customer_kind(text: str) -> str:
    lowered = (text or "").lower()
    if re.search(r"\b(no|nope|not interested|i'm good|im good)\b", lowered):
        return "no"
    if re.search(r"\b(yes|yeah|yep|yup|sure|okay|ok)\b", lowered):
        return "yes"
    if re.search(r"\b([1-4]|one|two|three|four)\s*-?\s*bhk\b", lowered):
        return "bhk"
    if re.search(r"\b(luxe|grande|\d{3,4}|square|sq)\b", lowered):
        return "size"
    return "other"


_FIT = {
    ("yes_no", "yes"),
    ("yes_no", "no"),
    ("configuration", "bhk"),
    ("size", "size"),
    ("open", "yes"),
    ("open", "no"),
    ("open", "bhk"),
    ("open", "size"),
    ("open", "other"),
}


def latency_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    turn_totals: list[int | None] = []
    for turn in turns:
        parts = [turn.get("stt_ms"), turn.get("llm_ms"), turn.get("tts_ms")]
        present = [part for part in parts if part is not None]
        turn_totals.append(sum(present) if present else None)
    return {
        "stt": _stats([turn.get("stt_ms") for turn in turns]),
        "llm": _stats([turn.get("llm_ms") for turn in turns]),
        "tts": _stats([turn.get("tts_ms") for turn in turns]),
        "turn": _stats(turn_totals),
    }


def repeated_questions(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    repeats = []
    for turn in turns:
        for question in _questions(turn.get("reply") or ""):
            key = _norm(_focus(question))
            if not key:
                continue
            if key in seen:
                repeats.append(
                    {
                        "turn_index": turn["turn_index"],
                        "question": question.strip() + "?",
                        "first_turn_index": seen[key],
                    }
                )
            else:
                seen[key] = turn["turn_index"]
    return repeats


def last_answer_fit(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    previous_reply = ""
    for turn in turns:
        customer = turn.get("customer")
        if not customer:
            previous_reply = turn.get("reply") or previous_reply
            continue
        asked = _questions(previous_reply)
        if not asked:
            results.append({"turn_index": turn["turn_index"], "fit": None, "na": True})
        else:
            kind = _question_kind(asked[-1])
            answer = _customer_kind(customer)
            results.append(
                {
                    "turn_index": turn["turn_index"],
                    "fit": (kind, answer) in _FIT,
                    "na": False,
                    "question_kind": kind,
                    "answer_kind": answer,
                }
            )
        previous_reply = turn.get("reply") or previous_reply
    return results


def stage_path(turns: list[dict[str, Any]]) -> list[str]:
    path = []
    for turn in turns:
        stage = turn.get("stage") or ""
        if stage and (not path or path[-1] != stage):
            path.append(stage)
    return path


def premature_site_visits(turns: list[dict[str, Any]]) -> list[int]:
    hits = []
    previous_reply = ""
    for turn in turns:
        reply = turn.get("reply") or ""
        customer = turn.get("customer") or ""
        asked = _questions(previous_reply)
        previous_question = asked[-1].lower() if asked else ""
        visit_was_open = bool(re.search(r"\b(visit|schedule)\b", previous_question))
        if (
            re.search(r"site visit", reply, re.I)
            and not visit_was_open
            and _customer_kind(customer) == "no"
        ):
            hits.append(turn["turn_index"])
        previous_reply = reply
    return hits


def extra_questions(turns: list[dict[str, Any]]) -> list[int]:
    return [
        turn["turn_index"]
        for turn in turns
        if (turn.get("reply") or "").count("?") > 1
    ]


def repeated_configuration_prompts(turns: list[dict[str, Any]]) -> list[int]:
    hits = []
    seen = False
    for turn in turns:
        reply = (turn.get("reply") or "").lower()
        if "which configuration are you looking for" not in reply:
            continue
        customer = turn.get("customer") or ""
        if seen and _customer_kind(customer) != "bhk":
            hits.append(turn["turn_index"])
        seen = True
    return hits


def score_measured(record: dict[str, Any]) -> dict[str, Any]:
    turns = record.get("turns") or []
    fit = last_answer_fit(turns)
    applicable = [item for item in fit if not item.get("na")]
    fitted = [item for item in applicable if item.get("fit")]
    return {
        "latency": latency_summary(turns),
        "repeated_questions": repeated_questions(turns),
        "last_answer_fit": fit,
        "last_answer_fit_rate": (
            round(len(fitted) / len(applicable), 4) if applicable else None
        ),
        "stage_path": stage_path(turns),
        "premature_site_visits": premature_site_visits(turns),
        "extra_questions": extra_questions(turns),
        "repeated_configuration_prompts": repeated_configuration_prompts(turns),
    }


def recommended_change(measured: dict[str, Any]) -> str | None:
    repeats = measured.get("repeated_questions") or []
    if repeats:
        turn = repeats[0]["turn_index"]
        return f"Stop repeating the question at turn {turn}. Answer it or leave that topic."
    visits = measured.get("premature_site_visits") or []
    if visits:
        return (
            f"Do not offer a site visit at turn {visits[0]} "
            "while the previous question is still open."
        )
    extras = measured.get("extra_questions") or []
    if extras:
        return f"Keep one question in the reply at turn {extras[0]}."
    prompts = measured.get("repeated_configuration_prompts") or []
    if prompts:
        return (
            f"Do not repeat the configuration question at turn {prompts[0]} "
            "when the customer has not chosen one."
        )
    misses = [
        item
        for item in measured.get("last_answer_fit") or []
        if item.get("fit") is False
    ]
    if misses:
        return (
            f"At turn {misses[0]['turn_index']} the customer did not answer "
            "the question that was asked."
        )
    return None

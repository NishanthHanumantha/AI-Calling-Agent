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
    if re.search(r"part of bengaluru|north, east, or south", text):
        return "part"
    if re.search(r"which one shall i|cover both|closer to", text):
        return "choice"
    if re.search(r"which configuration|one, two, three|bhk homes", text):
        return "configuration"
    if re.search(r"\b(luxe|grande|size|sizes|sq|ranges?)\b", text):
        return "size"
    if re.search(r"what would you like", text):
        return "open"
    if re.search(r"\b(would you|do you|is now|shall i|shall we)\b", text):
        return "yes_no"
    if re.search(r"\b(live in|investment|own use)\b", text):
        return "purpose"
    return "other"


def _leaves_open(text: str) -> bool:
    """The customer left the last topic open. That is an answer."""
    return re.search(
        r"\b(haven'?t decided|not decided|haven'?t finalized|not finalized|"
        r"not sure|decide(?:\s+(?:it|this|that))?\s+later|"
        r"let you know later|let me know later|i will let you know|i'?ll let you know|"
        r"plan(?:\s+\w+){0,6}\s+later|"
        r"not interested in (?:that|it|those|them)|"
        r"already told|already said|already asked|already answered|"
        r"you told|you mentioned|you asked|heard that|didn'?t answer|repeating)\b",
        text or "",
        re.I,
    ) is not None


def _customer_kind(text: str) -> str:
    lowered = (text or "").lower()
    if _leaves_open(lowered):
        return "open_answer"
    if re.search(r"\b(no|nope|not interested|i'm good|im good)\b", lowered):
        return "no"
    if re.search(r"\b(yes|yeah|yep|yup|sure|okay|ok)\b", lowered):
        return "yes"
    if re.search(r"\b(live in|for living|investment)\b", lowered):
        return "purpose"
    if re.search(
        r"\b(fine with both|fine with either|both are fine|either is fine)\b",
        lowered,
    ):
        return "size"
    if re.search(r"\bboth\b", lowered):
        return "choice"
    if re.search(r"\b(north|east|south)\b", lowered):
        return "part"
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
    ("purpose", "purpose"),
    ("part", "part"),
    ("choice", "choice"),
    ("choice", "yes"),
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


def _states_location(text: str) -> bool:
    return re.search(r"\b(electronic city|hosur road)\b", text or "", re.I) is not None


def _configuration_question(reply: str) -> str | None:
    """A question that asks the customer to pick a configuration. Leaving it open does not."""
    for question in _questions(reply):
        asked = _focus(question)
        if re.search(r"\bleave the configuration\b", asked, re.I):
            continue
        if re.search(
            r"\b(configurations?|which configuration|which bhk|one, two, three|bhk homes)\b",
            asked,
            re.I,
        ):
            return question
    return None


def _location_question(reply: str) -> str | None:
    for question in _questions(reply):
        if re.search(r"\blocation\b", question, re.I):
            return question
    return None


def repeated_questions(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    repeats = []
    recorded: set[tuple[int, str]] = set()
    location_at: int | None = None
    configuration_at: int | None = None

    def add(turn_index: int, question: str, first_turn_index: int) -> None:
        key = (turn_index, _norm(question))
        if key in recorded:
            return
        recorded.add(key)
        text = question.strip()
        if not text.endswith("?"):
            text += "?"
        repeats.append(
            {
                "turn_index": turn_index,
                "question": text,
                "first_turn_index": first_turn_index,
            }
        )

    for turn in turns:
        reply = turn.get("reply") or ""
        index = turn["turn_index"]
        for question in _questions(reply):
            key = _norm(_focus(question))
            if not key:
                continue
            if key in seen:
                if re.search(r"\b(didn'?t catch|please repeat)\b", reply, re.I):
                    continue
                add(index, question.strip() + "?", seen[key])
            else:
                seen[key] = index
        if location_at is not None:
            question = _location_question(reply)
            if question:
                add(index, question, location_at)
        if configuration_at is not None:
            question = _configuration_question(reply)
            if question:
                add(index, question, configuration_at)
        if location_at is None and _states_location(reply):
            location_at = index
        if configuration_at is None and _configuration_question(reply):
            configuration_at = index
    return repeats


def _names_a_sellable_project(text: str) -> bool:
    """A project name answers the part question. North, East, or South are not required."""
    from local_call import projects as catalog

    return bool(catalog.project_named_in(text))


def _names_offered_place(question: str, customer: str) -> bool:
    """Panathur answers 'closer to Whitefield, Panathur, or Sarjapur?'"""
    low = (question or "").lower()
    if "closer to" not in low:
        return False
    tail = low.split("closer to", 1)[1]
    spoken = (customer or "").lower()
    for name in re.findall(r"[a-z][a-z'-]+", tail):
        if name in {"or", "are", "you", "looking"}:
            continue
        if re.search(rf"\b{re.escape(name)}\b", spoken):
            return True
    return False


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
            fit = answer in ("open_answer", "no") or (kind, answer) in _FIT
            if not fit and kind == "choice" and _names_offered_place(asked[-1], customer):
                fit = True
            if not fit and kind == "part" and _names_a_sellable_project(customer):
                fit = True
            if not fit and kind == "yes_no" and re.search(
                r"\b(?:tell me|let me know|you can tell me) about (?:that|it|this)\b",
                customer,
                re.I,
            ):
                fit = True
            if not fit and re.search(
                r"anything else about the project|what would you like to know",
                asked[-1],
                re.I,
            ) and re.search(
                r"\b(price|cost|how much|where|location|located|configuration|bhk|bedroom)\b",
                customer,
                re.I,
            ):
                fit = True
            if not fit and re.search(
                r"interested to know about the project",
                asked[-1],
                re.I,
            ) and re.search(r"\b(budget|crore|lakh)\b", customer, re.I):
                fit = True
            reply = turn.get("reply") or ""
            if not fit and re.search(r"\b(didn'?t catch|please repeat)\b", reply, re.I):
                fit = True
            if not fit and re.search(r"\bsales advisor\b", reply, re.I) and re.search(
                r"\b(let you know later|i will let you know|plan(?:\s+\w+){0,6}\s+later)\b",
                customer,
                re.I,
            ):
                fit = True
            results.append(
                {
                    "turn_index": turn["turn_index"],
                    "fit": fit,
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
    """One visit after a no to the previous topic is allowed."""
    del turns
    return []


def extra_questions(turns: list[dict[str, Any]]) -> list[int]:
    return [
        turn["turn_index"]
        for turn in turns
        if (turn.get("reply") or "").count("?") > 1
    ]


def _states_configuration_menu(reply: str) -> bool:
    """A statement of the 1 to 4 menu. Leaving configuration open does not count."""
    text = reply or ""
    if re.search(r"\bleave the configuration\b", text, re.I):
        return False
    if re.search(r"\b(1|one)\s+to\s+(4|four)\s+bhk\b", text, re.I):
        return True
    return re.search(r"\b(one|1)\s+to\s+(four|4)\s+bedrooms?\b", text, re.I) is not None


def repeated_configuration_prompts(turns: list[dict[str, Any]]) -> list[int]:
    hits = []
    seen = False
    for turn in turns:
        reply = turn.get("reply") or ""
        if not (_configuration_question(reply) or _states_configuration_menu(reply)):
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
        "missed_closes": missed_closes(turns),
        "missed_shares": missed_shares(turns),
        "extra_questions": extra_questions(turns),
        "repeated_configuration_prompts": repeated_configuration_prompts(turns),
    }


def missed_closes(turns: list[dict[str, Any]]) -> list[int]:
    """A project no, a deferred visit, or a refusal of details still got a question.

    A no to configuration or size, followed by the one visit offer, is not this.
    """
    visit_offered = False
    hits = []
    for turn in turns:
        customer = turn.get("customer") or ""
        reply = turn.get("reply") or ""
        stage = turn.get("stage") or ""
        asked_again = "?" in reply
        on_visit = visit_offered or stage in ("propose_slot", "visit_day", "visit_pick_slot")
        kind = _ending_kind(customer)
        if asked_again and kind == "project":
            hits.append(turn.get("turn_index"))
        elif asked_again and kind == "details":
            hits.append(turn.get("turn_index"))
        elif asked_again and kind == "visit" and on_visit:
            hits.append(turn.get("turn_index"))
        if re.search(r"\bsite visit\b", reply, re.I):
            visit_offered = True
    return hits


def _offered_details(text: str) -> bool:
    """A question that offers highlights, documents, or the other project details."""
    raw = text or ""
    if "?" not in raw:
        return False
    if re.search(r"\b(visit|which day|schedule)\b", raw, re.I):
        return False
    return bool(re.search(r"\b(highlights?|documents?|brochure|details|share)\b", raw, re.I))


def missed_shares(turns: list[dict[str, Any]]) -> list[int]:
    """The customer agreed to the details, and the reply asked something else."""
    hits = []
    previous = ""
    for turn in turns:
        customer = (turn.get("customer") or "").lower()
        reply = turn.get("reply") or ""
        agreed = bool(re.search(r"\b(yes|yeah|sure|please)\b", customer)) and not re.search(
            r"\b(no|not interested|don'?t|do not)\b", customer
        )
        if _offered_details(previous) and agreed and "?" in reply:
            hits.append(turn.get("turn_index"))
        previous = reply
    return hits


def _ending_kind(text: str) -> str | None:
    low = (text or "").lower()
    if re.search(r"\bnot interested to know\b", low):
        return None
    if re.search(r"\bnot interested in (?:that|it|those|them)\b", low):
        return None
    if re.search(r"\bnot interested\b", low):
        return "project"
    if re.search(
        r"\b(don'?t need|do not need|no need|not need|don'?t want|do not want)"
        r".{0,24}\bdetails\b|\bno details\b",
        low,
    ):
        return "details"
    if re.search(
        r"\b(let you know later|i will let you know|i'll let you know|"
        r"(?:come|get) back(?: to you)? later)\b",
        low,
    ):
        return "visit"
    return None


def recommended_change(measured: dict[str, Any]) -> str | None:
    closes = measured.get("missed_closes") or []
    if closes:
        return (
            f"Close the call at turn {closes[0]}. "
            "The customer declined the project, deferred the visit, or asked for no details."
        )
    shares = measured.get("missed_shares") or []
    if shares:
        return (
            f"Share the offered details at turn {shares[0]}. "
            "The customer agreed, and the reply asked something else."
        )
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
            f"At turn {misses[0]['turn_index']}, answer what the customer said. "
            "Change the agent's reply."
        )
    return None

from __future__ import annotations

import re

from evaluator.deterministic import normalize_text

# Copied from production PROJECT_KB for fair identical context. Not live PDF chunks.
SHARED_PROJECT_CONTEXT = """SOURCE: production PROJECT_KB (identical for all interactive models; not live PDF chunks).
Project: SOBHA Townpark
Brand: Sobha Limited
City: Bengaluru
Theme: New York-themed luxury apartments
Location: Near Electronic City on Hosur Road, Bengaluru
Starting price: INR 1.8 Crore onwards
Configurations: 1 BHK, 2 BHK, 3 BHK and 4 BHK apartments
Amenities: clubhouses, swimming pools, sports courts, landscaped gardens, kids play areas, forest grove and camping grounds
Possession: Please check with our sales team for the latest possession timeline
Payment: Flexible payment plans and home loan assistance are available
Bookable slots: 10:00 AM, 11:30 AM, 4:00 PM, 6:00 PM
"""

STAGE_LABELS = {
    "greeting": "Greeting",
    "qualify": "Qualification",
    "answer_faq": "FAQ",
    "propose_slot": "Visit",
    "visit_day": "Visit",
    "visit_pick_slot": "Slot",
    "closed": "Closure",
}


def infer_stage(utterance: str, turn_index: int) -> str:
    text = normalize_text(utterance)
    if re.search(r"\b(bye|goodbye|not interested)\b", text):
        return "closed"
    if re.search(r"\b(10|11|4|6|morning|evening|am|pm|slot)\b", text) and turn_index > 1:
        return "visit_pick_slot"
    if re.search(r"\b(visit|site visit|saturday|sunday|weekend|appointment)\b", text):
        return "propose_slot"
    if re.search(r"\b(budget|looking for|interested)\b", text) and turn_index <= 2:
        return "qualify"
    if turn_index <= 1 and re.search(r"\b(hi|hello|hey)\b", text):
        return "greeting"
    return "answer_faq"


def detect_language(utterance: str) -> str:
    text = utterance or ""
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kn"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    mixed_tokens = ("kya", "hai", "ka ", "ke ", "mujhe", "yelli", "ide", "heli", "alli", "sir", "eshtu")
    lower = text.lower()
    if any(tok in lower for tok in mixed_tokens) and re.search(r"[a-z]", lower):
        return "mixed"
    return "en"


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "it",
    "to",
    "for",
    "of",
    "and",
    "do",
    "you",
    "me",
    "i",
    "my",
    "in",
    "on",
    "at",
    "please",
}


def overlap_score(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def content_tokens(text: str) -> set[str]:
    return {tok for tok in normalize_text(text).split() if tok not in STOPWORDS and len(tok) > 1}


_SHORT_ACKS = {"yes", "yeah", "yep", "ok", "okay", "sure", "no", "hi", "hello", "hey"}


def match_golden_case(utterance: str, dataset: list[dict]) -> dict | None:
    """Return a golden case if the utterance is clearly the same question. Never invent GT."""
    norm = normalize_text(utterance)
    if not norm:
        return None
    tokens = content_tokens(norm)
    if norm in _SHORT_ACKS or len(tokens) < 2:
        for case in dataset:
            gold = normalize_text(case.get("customer_utterance") or "")
            if gold == norm:
                return case
        return None
    best = None
    best_score = 0.0
    for case in dataset:
        gold = normalize_text(case.get("customer_utterance") or "")
        if not gold:
            continue
        if gold == norm or gold in norm or norm in gold:
            return case
        ca, cb = content_tokens(norm), content_tokens(gold)
        if ca and cb:
            shorter, longer = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
            if shorter <= longer and len(shorter) >= 2:
                return case
        score = overlap_score(norm, gold)
        if score > best_score:
            best_score = score
            best = case
    if best is not None and best_score >= 0.72:
        return best
    return None

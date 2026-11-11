"""Score one saved call and write its scorecard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .judge import judge_call
from .measured import score_measured
from .scorecard import write_scorecard


def evaluate_saved_call(
    record: dict[str, Any],
    runs_dir: Path,
    facts: dict[str, Any],
    poster: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    measured = score_measured(record)
    judged = judge_call(record, facts, poster=poster)
    path = write_scorecard(record, measured, judged, runs_dir)
    return {"measured": measured, "judged": judged, "scorecard": path.name}

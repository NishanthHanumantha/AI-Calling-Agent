"""Master evaluation evidence reporting (LLM-EVAL.5.1).

Reporting / aggregation only. Does not modify evaluation scoring or production.
"""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

__all__ = ["EVAL_ROOT"]

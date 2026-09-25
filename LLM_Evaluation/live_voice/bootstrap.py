"""Put LLM_Evaluation and the repo root on sys.path. Does not load production app_v3."""

from __future__ import annotations

import sys
from pathlib import Path


def eval_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return eval_root().parent


def bootstrap() -> Path:
    root = eval_root()
    repo = repo_root()
    for path in (str(root), str(repo)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root

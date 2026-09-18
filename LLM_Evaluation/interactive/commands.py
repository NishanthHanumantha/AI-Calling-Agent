from __future__ import annotations

from typing import Any


def parse_command(raw: str) -> tuple[str | None, str]:
    text = (raw or "").strip()
    if not text.startswith("/"):
        return None, text
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    aliases = {"/exit": "/quit", "/q": "/quit"}
    cmd = aliases.get(cmd, cmd)
    return cmd, arg

"""Shared live-voice scenario. Same operator script for all four models."""

from __future__ import annotations

from .constants import FIXED_OPENING

OPERATOR_SCRIPT = [
    "Yes, I have a couple of minutes.",
    "I'm looking for a 3 BHK.",
    "Where exactly is the project located?",
    "How much does it cost?",
    "Can I visit this Saturday?",
    "Morning would be better.",
    "Okay, I'll discuss it with my wife and get back to you.",
]


def operator_script_text() -> str:
    lines = [
        "LIVE VOICE OPERATOR SCRIPT (same for every model)",
        "Opening is a fixed V3 greeting spoken by TTS. Do not ad-lib.",
        "",
        f"Agent (fixed): {FIXED_OPENING}",
        "",
    ]
    for index, line in enumerate(OPERATOR_SCRIPT, start=1):
        lines.append(f"You say ({index}): {line}")
    lines.extend(
        [
            "",
            "After hangup wait for the CLI. Type Continue, Skip, or Abort.",
            "Skip = do not place the next model call. Abort = end the campaign.",
        ]
    )
    return "\n".join(lines)

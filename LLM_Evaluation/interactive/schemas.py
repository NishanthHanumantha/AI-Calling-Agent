from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DISPLAY_ORDER = (
    "sarvam_conversational",
    "sarvam_flagship",
    "claude_sonnet",
    "claude_flagship",
)

DISPLAY_NAMES = {
    "sarvam_conversational": "SARVAM — CONVERSATIONAL",
    "sarvam_flagship": "SARVAM — FLAGSHIP",
    "claude_sonnet": "CLAUDE — SONNET",
    "claude_flagship": "CLAUDE — FLAGSHIP",
}

SHORT_NAMES = {
    "sarvam_conversational": "Sarvam C",
    "sarvam_flagship": "Sarvam F",
    "claude_sonnet": "Claude S",
    "claude_flagship": "Claude F",
}

COMMANDS = {
    "/help": "Show available commands",
    "/reset": "Start a new conversation",
    "/history": "Show conversation history",
    "/evaluate": "Evaluate the latest turn (or `/evaluate session`)",
    "/save": "Save current session",
    "/export": "Export session results",
    "/models": "Show configured models",
    "/status": "Show API/model status",
    "/debug": "Show latest-turn request context (no secrets)",
    "/quit": "Exit",
}

DEMO_TURNS = [
    "Hi, I am interested in Sobha Neopolis.",
    "Where is it located?",
    "What configurations are available?",
    "How much does a 3 BHK cost?",
    "Can I visit on Saturday morning?",
]


@dataclass
class ModelSlot:
    alias: str
    provider: str
    role: str
    model_id: str
    available: bool
    status: str
    error: str | None = None
    base_url: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResponse:
    alias: str
    provider: str
    model_id: str
    answer: str | None
    intent: str | None
    action: str | None
    language: str | None
    stage: str | None
    raw_response: str
    latency_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error_type: str | None
    error_message: str | None
    schema_valid: bool | None = None
    confidence: Any = None

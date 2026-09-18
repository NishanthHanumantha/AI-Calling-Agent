from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Common LLM interface for offline calling-agent evaluation."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        customer_utterance: str,
        retrieved_context: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a normalized generation result. Must not log API keys."""
        raise NotImplementedError


def empty_result(
    provider: str,
    model: str,
    *,
    model_alias: str | None = None,
    model_role: str | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "model_id": model,
        "model_alias": model_alias,
        "model_role": model_role,
        "provider": provider,
        "intent": None,
        "language": None,
        "answer": None,
        "action": None,
        "confidence": None,
        "raw_response": "",
        "usage": {},
        "latency_ms": 0,
        "error": None,
        "error_type": None,
        "error_message": None,
        "schema_valid": False,
        "parsed": False,
        "estimated_cost": "NOT_CONFIGURED",
    }

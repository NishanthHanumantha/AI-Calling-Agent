from __future__ import annotations

import time
from typing import Any

import requests

from .base import LLMProvider, empty_result
from .parsing import extract_json_object, validate_structured_output
from .prompting import chat_messages, usage_from_openai


class QwenProvider(LLMProvider):
    """OpenAI-compatible Qwen adapter. Base URL and model come from configuration."""

    name = "qwen"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: int = 30, **kwargs: Any):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = kwargs.get("temperature", 0.1)
        self.max_tokens = kwargs.get("max_tokens", 400)

    def _completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def generate(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        customer_utterance: str,
        retrieved_context: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = empty_result(self.name, self.model)
        messages = chat_messages(
            system_prompt, conversation_history, customer_utterance, retrieved_context, extra
        )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        try:
            response = requests.post(
                self._completions_url(), headers=headers, json=payload, timeout=self.timeout
            )
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            if response.status_code >= 400:
                result["error"] = f"HTTP {response.status_code}"
                result["raw_response"] = (response.text or "")[:2000]
                return result
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            result["raw_response"] = content if isinstance(content, str) else str(content)
            result["usage"] = usage_from_openai(data)
            parsed = extract_json_object(result["raw_response"])
            schema_valid, fields = validate_structured_output(parsed)
            result.update(fields)
            result["schema_valid"] = schema_valid
            result["parsed"] = parsed is not None
            result["confidence"] = None
            return result
        except Exception as exc:
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            result["error"] = f"{type(exc).__name__}: request failed"
            return result

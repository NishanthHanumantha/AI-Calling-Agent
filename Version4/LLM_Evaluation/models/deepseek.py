from __future__ import annotations

import time
from typing import Any

from .base import LLMProvider, empty_result
from .http_util import chat_completions_url, request_with_retry
from .parsing import extract_json_object, validate_structured_output
from .prompting import chat_messages, usage_from_openai

# Conversational benchmark default. Thinking stays available, but off unless configured.
_THINKING_ENABLED = {"enabled", "thinking", "on"}


def thinking_body(mode: str | None) -> dict[str, str]:
    """DeepSeek Chat Completions thinking toggle. Unknown values stay non-thinking."""
    normalized = (mode or "disabled").strip().lower().replace("_", "-")
    if normalized in _THINKING_ENABLED:
        return {"type": "enabled"}
    return {"type": "disabled"}


def _message_text(message: dict[str, Any]) -> str | None:
    """User-facing assistant text. Reasoning traces are not treated as the answer."""
    content = message.get("content")
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type in {None, "text"} and part.get("text"):
                    parts.append(str(part["text"]))
        return "\n".join(parts) if parts else None
    return str(content)


class DeepSeekProvider(LLMProvider):
    """OpenAI-compatible DeepSeek adapter for the interactive evaluation lab."""

    name = "deepseek"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: int = 30, **kwargs: Any):
        self.api_key = api_key
        self.model = model
        self.base_url = chat_completions_url(base_url)
        self.timeout = timeout
        self.temperature = kwargs.get("temperature", 0.1)
        self.max_tokens = kwargs.get("max_tokens", 400)
        self.max_retries = int(kwargs.get("max_retries", 2))
        self.model_alias = kwargs.get("model_alias")
        self.model_role = kwargs.get("model_role")
        self.thinking_mode = kwargs.get("thinking_mode") or "disabled"

    def generate(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        customer_utterance: str,
        retrieved_context: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        messages = chat_messages(
            system_prompt, conversation_history, customer_utterance, retrieved_context, extra
        )
        result = self.complete(messages)
        if result.get("error_type"):
            return result
        content = result.get("text") or ""
        result["raw_response"] = content
        parsed = extract_json_object(content)
        schema_valid, fields = validate_structured_output(parsed)
        result.update(fields)
        result["schema_valid"] = schema_valid
        result["parsed"] = parsed is not None
        result["confidence"] = None
        return result

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send chat messages and return a normalized provider result. Does not invent token counts."""
        result = empty_result(
            self.name, self.model, model_alias=self.model_alias, model_role=self.model_role
        )
        result["text"] = None
        result["input_tokens"] = None
        result["output_tokens"] = None
        result["total_tokens"] = None
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": thinking_body(self.thinking_mode),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        response, err, msg = request_with_retry(
            "POST",
            self.base_url,
            max_retries=self.max_retries,
            timeout=self.timeout,
            headers=headers,
            json=payload,
        )
        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        result["model_id"] = self.model
        result["model"] = self.model
        if err != "AVAILABLE":
            result["error"] = msg
            result["error_type"] = err
            result["error_message"] = msg
            result["raw_response"] = ((response.text if response is not None else "") or "")[:2000]
            return result
        data = response.json()
        returned_model = data.get("model")
        if isinstance(returned_model, str) and returned_model.strip():
            result["model"] = returned_model.strip()
            result["model_id"] = returned_model.strip()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        text = _message_text(message)
        result["text"] = text
        result["raw_response"] = text or ""
        usage = usage_from_openai(data if isinstance(data, dict) else {})
        result["usage"] = usage
        result["input_tokens"] = usage.get("input_tokens")
        result["output_tokens"] = usage.get("output_tokens")
        result["total_tokens"] = usage.get("total_tokens")
        return result

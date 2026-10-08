from __future__ import annotations

import time
from typing import Any

from .base import LLMProvider, empty_result
from .http_util import request_with_retry
from .parsing import extract_json_object, validate_structured_output
from .prompting import chat_messages, usage_from_openai


class SarvamProvider(LLMProvider):
    name = "sarvam"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: int = 30, **kwargs: Any):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = kwargs.get("temperature", 0.1)
        self.max_tokens = kwargs.get("max_tokens", 400)
        self.max_retries = int(kwargs.get("max_retries", 2))
        self.model_alias = kwargs.get("model_alias")
        self.model_role = kwargs.get("model_role")

    def generate(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        customer_utterance: str,
        retrieved_context: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = empty_result(
            self.name, self.model, model_alias=self.model_alias, model_role=self.model_role
        )
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
        if err != "AVAILABLE":
            result["error"] = msg
            result["error_type"] = err
            result["error_message"] = msg
            result["raw_response"] = ((response.text if response is not None else "") or "")[:2000]
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

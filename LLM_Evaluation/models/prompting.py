from __future__ import annotations

from typing import Any


def build_user_payload(
    conversation_history: list[dict[str, str]],
    customer_utterance: str,
    retrieved_context: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Identical logical user payload for every provider."""
    extra = extra or {}
    stage = extra.get("conversation_stage") or extra.get("stage") or "unknown"
    visit_rule = extra.get("visit_rule")
    visit_block = f"Visit sequence checkpoint: {visit_rule}\n\n" if visit_rule else ""
    language_block = ""
    language_track = extra.get("language_track")
    if language_track:
        instruction = extra.get("language_instruction") or f"Customer language track: {language_track}."
        language_block = f"{instruction}\n\n"
    history_lines = []
    for turn in conversation_history or []:
        role = str(turn.get("role", "user")).upper()
        content = str(turn.get("content", "")).strip()
        history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines) if history_lines else "(none)"
    return (
        f"Conversation stage: {stage}\n\n"
        f"{language_block}"
        f"{visit_block}"
        f"Retrieved context:\n{retrieved_context}\n\n"
        f"Conversation history:\n{history_block}\n\n"
        f"Customer utterance:\n{customer_utterance}\n\n"
        "Return the JSON object now."
    )


def chat_messages(
    system_prompt: str,
    conversation_history: list[dict[str, str]],
    customer_utterance: str,
    retrieved_context: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """OpenAI-style messages. History is included in the user payload so every model sees the same text."""
    user_content = build_user_payload(
        conversation_history,
        customer_utterance,
        retrieved_context,
        extra,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def usage_from_openai(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion = usage.get("completion_tokens") or usage.get("output_tokens")
    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
    }

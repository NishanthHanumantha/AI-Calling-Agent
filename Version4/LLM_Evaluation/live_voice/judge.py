"""Independent live-voice judge agent. Separate model, prompt, and protocol from candidates."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from .bootstrap import bootstrap
from .constants import JUDGE_CRITERIA

bootstrap()

from evaluator.runner import load_env, load_text, load_yaml_config, root_dir
from models.http_util import request_with_retry
from models.parsing import extract_json_object


def resolve_judge_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    load_env()
    cfg = config if config is not None else load_yaml_config()
    spec = (cfg.get("live_voice") or {}).get("judge") or {}
    provider = (
        os.getenv(spec.get("provider_env") or "LIVE_EVAL_JUDGE_PROVIDER")
        or spec.get("provider")
        or "claude"
    ).strip()
    model = (os.getenv(spec.get("model_env") or "LIVE_EVAL_JUDGE_MODEL") or "").strip()
    key_envs = [spec.get("api_key_env") or "LIVE_EVAL_JUDGE_API_KEY"]
    key_envs.extend(spec.get("fallback_api_key_env") or ["JUDGE_API_KEY", "ANTHROPIC_API_KEY"])
    api_key = ""
    api_key_env_used = None
    for env_name in key_envs:
        if not env_name:
            continue
        value = (os.getenv(env_name) or "").strip()
        if value:
            api_key = value
            api_key_env_used = env_name
            break
    base_url = (
        os.getenv(spec.get("base_url_env") or "LIVE_EVAL_JUDGE_BASE_URL")
        or spec.get("default_base_url")
        or "https://api.anthropic.com/v1/messages"
    ).strip()
    temperature = os.getenv("LIVE_EVAL_JUDGE_TEMPERATURE")
    return {
        "provider": provider,
        "model_id": model,
        "api_key": api_key,
        "api_key_env_used": api_key_env_used,
        "base_url": base_url,
        "temperature": float(temperature) if temperature not in (None, "") else float(spec.get("temperature") or 0),
        "max_tokens": int(spec.get("max_tokens") or 2000),
        "timeout": int(spec.get("timeout_seconds") or 60),
    }


def same_model_forbidden(candidate_model_id: str, judge_model_id: str) -> bool:
    left = (candidate_model_id or "").strip()
    right = (judge_model_id or "").strip()
    return bool(left and right and left == right)


def na_criteria(reason: str) -> dict[str, Any]:
    return {
        name: {
            "score": None,
            "na": True,
            "turn_indices": [],
            "error_class": "UNOBSERVABLE",
            "evidence": reason,
        }
        for name in JUDGE_CRITERIA
    }


def judge_system_prompt() -> str:
    return load_text(root_dir() / "prompts" / "live_voice_judge_system_prompt.txt")


def _claude_complete(judge: dict[str, Any], system: str, user: str) -> dict[str, Any]:
    payload = {
        "model": judge["model_id"],
        "max_tokens": judge["max_tokens"],
        "temperature": judge["temperature"],
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "x-api-key": judge["api_key"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response, err, msg = request_with_retry(
        "POST",
        judge["base_url"],
        max_retries=1,
        timeout=judge["timeout"],
        headers=headers,
        json=payload,
    )
    if err != "AVAILABLE" or response is None:
        return {"error": msg, "error_type": err, "text": ""}
    data = response.json()
    blocks = data.get("content") or []
    text_parts = [block.get("text", "") for block in blocks if block.get("type") == "text"]
    return {"text": "\n".join(part for part in text_parts if part), "error": None, "error_type": None}


def _openai_complete(judge: dict[str, Any], system: str, user: str) -> dict[str, Any]:
    url = judge["base_url"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    payload = {
        "model": judge["model_id"],
        "temperature": judge["temperature"],
        "max_tokens": judge["max_tokens"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {judge['api_key']}", "Content-Type": "application/json"}
    response, err, msg = request_with_retry(
        "POST",
        url,
        max_retries=1,
        timeout=judge["timeout"],
        headers=headers,
        json=payload,
    )
    if err != "AVAILABLE" or response is None:
        return {"error": msg, "error_type": err, "text": ""}
    data = response.json()
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    return {"text": text, "error": None, "error_type": None}


class LiveVoiceJudge:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        complete_fn: Callable[[dict[str, Any], str, str], dict[str, Any]] | None = None,
        judge_runtime: dict[str, Any] | None = None,
    ):
        self.judge = judge_runtime or resolve_judge_config(config)
        self.complete_fn = complete_fn
        self.system_prompt = judge_system_prompt()

    def evaluate_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(packet.get("candidate_model_id") or "")
        judge_id = str(self.judge.get("model_id") or "")
        base = {
            "judge_provider": self.judge.get("provider"),
            "judge_model_id": judge_id,
            "candidate_alias": packet.get("alias"),
            "candidate_model_id": candidate_id,
            "campaign_id": packet.get("campaign_id"),
            "evaluation_run_id": packet.get("evaluation_run_id"),
            "call_sid": packet.get("call_sid"),
        }
        if not judge_id:
            return {
                **base,
                "judge_status": "JUDGE_MODEL_UNSET",
                "criteria": na_criteria("LIVE_EVAL_JUDGE_MODEL is not set"),
                "notes": "Judge refused to score. Candidate self-report was not used.",
            }
        if same_model_forbidden(candidate_id, judge_id):
            return {
                **base,
                "judge_status": "SAME_MODEL_FORBIDDEN",
                "criteria": na_criteria("Judge model_id equals candidate model_id"),
                "notes": "Independent protocol requires a different model ID than the call under test.",
            }
        if not self.judge.get("api_key") and self.complete_fn is None:
            return {
                **base,
                "judge_status": "JUDGE_KEY_MISSING",
                "criteria": na_criteria("Judge API key is not configured"),
                "notes": "Judge refused to score. Candidate self-report was not used.",
            }
        user = (
            "Evaluate this completed live voice call. "
            "Do not trust candidate_self_report as the score. Return the rubric JSON only.\n\n"
            + json.dumps(packet, ensure_ascii=False, indent=2)
        )
        if self.complete_fn is not None:
            raw = self.complete_fn(self.judge, self.system_prompt, user)
        elif str(self.judge.get("provider") or "").lower() == "claude":
            raw = _claude_complete(self.judge, self.system_prompt, user)
        else:
            raw = _openai_complete(self.judge, self.system_prompt, user)
        if raw.get("error"):
            return {
                **base,
                "judge_status": "JUDGE_ERROR",
                "criteria": na_criteria(str(raw.get("error") or "judge request failed")),
                "error_type": raw.get("error_type"),
                "notes": "Judge call failed. Candidate self-report was not used.",
            }
        parsed = extract_json_object(raw.get("text") or "") or {}
        criteria = parsed.get("criteria") or {}
        normalized = {}
        for name in JUDGE_CRITERIA:
            row = criteria.get(name) or {}
            score = row.get("score")
            is_na = bool(row.get("na")) or score is None
            normalized[name] = {
                "score": None if is_na else score,
                "na": is_na,
                "turn_indices": row.get("turn_indices") or [],
                "error_class": row.get("error_class") or ("UNOBSERVABLE" if is_na else "LLM"),
                "evidence": row.get("evidence") or "",
            }
        return {
            **base,
            "judge_status": parsed.get("judge_status") or "OK",
            "criteria": normalized,
            "summary": parsed.get("summary") or "",
            "raw_judge_text": (raw.get("text") or "")[:8000],
        }

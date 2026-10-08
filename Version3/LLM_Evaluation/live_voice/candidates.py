"""Resolve the four live-voice candidates from existing eval config. No hardcoded keys."""

from __future__ import annotations

from typing import Any

from .bootstrap import bootstrap
from .constants import CANDIDATE_ORDER, DISPLAY_NAMES, LIVE_SPOKEN_APPENDIX

bootstrap()

from evaluator.runner import load_env, load_text, load_yaml_config, resolve_model_runtime, root_dir
from interactive.conversation import SHARED_PROJECT_CONTEXT
from interactive.visit_policy import SITE_VISIT_SCHEDULING_RULE
from models import build_provider


def load_live_config(config_path=None) -> dict[str, Any]:
    load_env()
    return load_yaml_config(config_path)


def calling_system_prompt() -> str:
    base = load_text(root_dir() / "prompts" / "calling_agent_system_prompt.txt")
    return base.rstrip() + "\n\n" + SITE_VISIT_SCHEDULING_RULE + "\n\n" + LIVE_SPOKEN_APPENDIX


def resolve_candidates(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or load_live_config()
    specs = cfg.get("models") or {}
    resolved = []
    for alias in CANDIDATE_ORDER:
        spec = specs.get(alias)
        if not spec:
            resolved.append(
                {
                    "alias": alias,
                    "display_name": DISPLAY_NAMES.get(alias, alias),
                    "provider": alias.split("_")[0],
                    "model_id": "",
                    "available": False,
                    "error": "unknown alias in models.yaml",
                    "runtime": None,
                }
            )
            continue
        runtime = resolve_model_runtime(alias, spec, cfg)
        error = None
        available = bool(runtime.get("available"))
        if not runtime.get("api_key"):
            available = False
            error = f"{runtime.get('api_key_env')}: NOT CONFIGURED"
        elif not runtime.get("model"):
            available = False
            error = f"missing model id ({runtime.get('model_env')})"
        resolved.append(
            {
                "alias": alias,
                "display_name": DISPLAY_NAMES.get(alias, alias),
                "provider": runtime["provider"],
                "model_id": runtime.get("model") or "",
                "available": available,
                "error": error,
                "runtime": runtime,
            }
        )
    return resolved


def build_provider_for_candidate(candidate: dict[str, Any], *, dry_run: bool = False):
    runtime = candidate.get("runtime") or {}
    if dry_run or not candidate.get("available"):
        return None
    kwargs = {
        "api_key": runtime["api_key"],
        "model": runtime["model"],
        "base_url": runtime["base_url"],
        "timeout": min(int(runtime.get("timeout") or 30), 12),
        "temperature": runtime.get("temperature", 0.1),
        "max_tokens": min(int(runtime.get("max_tokens") or 400), 180),
        "max_retries": 0,
        "model_alias": candidate["alias"],
        "model_role": runtime.get("role"),
    }
    if runtime.get("thinking_mode"):
        kwargs["thinking_mode"] = runtime["thinking_mode"]
    return build_provider(runtime["provider"], **kwargs)


def project_context() -> str:
    return SHARED_PROJECT_CONTEXT

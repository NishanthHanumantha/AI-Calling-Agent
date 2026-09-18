from __future__ import annotations

from typing import Any

import requests

from .http_util import classify_http_error, request_with_retry


def _safe_json(resp: requests.Response | None) -> dict[str, Any]:
    if resp is None:
        return {}
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def check_claude_model(api_key: str, model_id: str, models_url: str, timeout: int = 30) -> dict[str, Any]:
    """List Anthropic models, then verify the configured ID. Never logs the API key."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    list_url = models_url.rstrip("/")
    if not list_url.endswith("/models"):
        list_url = "https://api.anthropic.com/v1/models"
    resp, err, msg = request_with_retry("GET", list_url, max_retries=2, timeout=timeout, headers=headers)
    if err == "AUTH_ERROR":
        return {
            "status": "UNAVAILABLE",
            "auth": "FAIL",
            "model": "NOT_CHECKED",
            "error_type": "AUTH_ERROR",
            "error_message": msg,
            "available_model_ids": [],
        }
    if err != "AVAILABLE":
        return {
            "status": "UNAVAILABLE",
            "auth": "UNKNOWN",
            "model": "NOT_CHECKED",
            "error_type": err,
            "error_message": msg,
            "available_model_ids": [],
        }
    payload = _safe_json(resp)
    ids = [item.get("id") for item in payload.get("data") or [] if item.get("id")]
    auth = "PASS"
    if model_id in ids:
        return {
            "status": "AVAILABLE",
            "auth": auth,
            "model": "FOUND",
            "error_type": None,
            "error_message": "",
            "available_model_ids": ids,
        }
    detail_url = f"{list_url.rstrip('/')}/{model_id}"
    dresp, derr, dmsg = request_with_retry("GET", detail_url, max_retries=0, timeout=timeout, headers=headers)
    if derr == "AVAILABLE":
        return {
            "status": "AVAILABLE",
            "auth": auth,
            "model": "FOUND",
            "error_type": None,
            "error_message": "",
            "available_model_ids": ids,
        }
    model_state = "NOT FOUND" if derr in {"MODEL_NOT_FOUND", "API_ERROR"} and (dresp is None or dresp.status_code == 404) else derr
    if dresp is not None and dresp.status_code == 404:
        model_state = "NOT FOUND"
        derr = "MODEL_NOT_FOUND"
    return {
        "status": "UNAVAILABLE",
        "auth": auth,
        "model": model_state,
        "error_type": derr,
        "error_message": dmsg,
        "available_model_ids": ids,
    }


def check_sarvam_model(api_key: str, model_id: str, chat_url: str, timeout: int = 30) -> dict[str, Any]:
    """Minimal chat completion to verify the model ID. max_tokens=1."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    resp, err, msg = request_with_retry("POST", chat_url, max_retries=2, timeout=timeout, headers=headers, json=payload)
    if err == "AVAILABLE":
        return {
            "status": "AVAILABLE",
            "auth": "PASS",
            "model": "FOUND",
            "error_type": None,
            "error_message": "",
            "available_model_ids": [model_id],
        }
    if err == "AUTH_ERROR":
        return {
            "status": "UNAVAILABLE",
            "auth": "FAIL",
            "model": "NOT_CHECKED",
            "error_type": err,
            "error_message": msg,
            "available_model_ids": [],
        }
    body = (resp.text if resp is not None else "")[:500]
    if resp is not None:
        err = classify_http_error(resp.status_code, body)
    return {
        "status": "UNAVAILABLE",
        "auth": "PASS" if err not in {"AUTH_ERROR"} else "FAIL",
        "model": "NOT FOUND" if err == "MODEL_NOT_FOUND" else "UNAVAILABLE",
        "error_type": err,
        "error_message": msg,
        "available_model_ids": [],
    }

from __future__ import annotations

import time
from typing import Any

import requests

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def classify_http_error(status: int | None, body: str = "") -> str:
    text = (body or "").lower()
    if status is None:
        return "OTHER_API_ERROR"
    if status in (401,):
        return "AUTH_ERROR"
    if status in (403,):
        return "PERMISSION_ERROR"
    if status == 404:
        return "MODEL_NOT_FOUND"
    if status == 429:
        return "RATE_LIMIT"
    if status in RETRYABLE_STATUS:
        return "API_ERROR"
    if "not_found" in text or "model:" in text and status >= 400:
        return "MODEL_NOT_FOUND"
    if status >= 400:
        return "API_ERROR"
    return "AVAILABLE"


def should_retry(status: int | None, error_type: str | None, exc: BaseException | None) -> bool:
    if exc is not None and isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if error_type == "RATE_LIMIT":
        return True
    if status in RETRYABLE_STATUS:
        return True
    return False


def request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 2,
    timeout: int = 30,
    **kwargs: Any,
) -> tuple[requests.Response | None, str, str]:
    """
    Perform an HTTP request with at most `max_retries` retries (so 3 attempts total).
    Does not retry 401/403/404/invalid requests.
    Returns (response_or_none, error_type, error_message).
    error_type is AVAILABLE when HTTP 2xx.
    """
    attempts = 0
    last_error = "OTHER_API_ERROR"
    last_message = "request failed"
    last_resp: requests.Response | None = None
    while attempts <= max_retries:
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            last_resp = resp
            if 200 <= resp.status_code < 300:
                return resp, "AVAILABLE", ""
            err = classify_http_error(resp.status_code, resp.text or "")
            last_error, last_message = err, f"HTTP {resp.status_code}"
            if should_retry(resp.status_code, err, None) and attempts < max_retries:
                attempts += 1
                time.sleep(0.05 * (attempts + 1))
                continue
            return resp, err, last_message
        except requests.Timeout:
            last_error, last_message = "TIMEOUT", "timeout"
            if attempts < max_retries:
                attempts += 1
                continue
            return None, "TIMEOUT", "timeout"
        except requests.ConnectionError:
            last_error, last_message = "API_ERROR", "connection error"
            if attempts < max_retries:
                attempts += 1
                continue
            return None, "API_ERROR", "connection error"
        except Exception as exc:
            return None, "OTHER_API_ERROR", type(exc).__name__
    return last_resp, last_error, last_message


def chat_completions_url(base_url: str) -> str:
    """OpenAI-compatible chat URL. Accepts a host or a full /chat/completions URL."""
    base = (base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"

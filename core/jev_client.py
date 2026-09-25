"""
Thin client for the Jev AI decision API ("System One" / model jev-latest).
Docs: https://jev-ai.pro/docs

Auth: reads JEV_AI_API_KEY from the environment — never hardcode it here or
in config.py. See .env.example for local setup.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://jev-ai.pro/api/v1"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 15  # seconds


class JevError(Exception):
    """Base class for all Jev API errors."""


class JevAuthError(JevError):
    """401 — missing or invalid API key."""


class JevBillingError(JevError):
    """402 — insufficient balance/credits."""


class JevValidationError(JevError):
    """422 — malformed request body. Fix the request; do not retry unchanged."""


class JevRateLimitError(JevError):
    """429 — rate limited. `retry_after` is seconds to wait, when the API supplied it."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class JevServiceError(JevError):
    """
    502/503/504 — upstream unavailable.

    A POST's outcome is uncertain when this is raised: the request may or may
    not have been processed server-side. This client never auto-retries a
    POST on this error — the caller decides whether re-sending is safe.
    """

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class JevUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class JevResponse:
    model: str
    answers: dict[str, Any] = field(default_factory=dict)
    usage: JevUsage = field(default_factory=JevUsage)
    run_id: Optional[str] = None


@dataclass
class JevModel:
    name: str
    description: str = ""


def _api_key() -> str:
    key = os.environ.get("JEV_AI_API_KEY")
    if not key:
        raise JevAuthError(
            "JEV_AI_API_KEY is not set in the environment. See .env.example."
        )
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


def _error_detail(resp: requests.Response) -> str:
    try:
        payload = resp.json()
        return payload.get("error") or payload.get("message") or resp.text[:300]
    except ValueError:
        return resp.text[:300]


def _raise_for_status(resp: requests.Response) -> None:
    if resp.ok:
        return

    status = resp.status_code
    detail = _error_detail(resp)

    if status == 401:
        raise JevAuthError(f"Jev API rejected the API key (401): {detail}")
    if status == 402:
        raise JevBillingError(f"Jev API: insufficient balance (402): {detail}")
    if status == 422:
        raise JevValidationError(f"Jev API: invalid request (422): {detail}")
    if status == 429:
        retry_after_hdr = resp.headers.get("Retry-After")
        retry_after = float(retry_after_hdr) if retry_after_hdr else None
        raise JevRateLimitError(
            f"Jev API: rate limited (429): {detail}", retry_after=retry_after
        )
    if status in (502, 503, 504):
        raise JevServiceError(
            f"Jev API: upstream unavailable ({status}): {detail}", status_code=status
        )
    raise JevError(f"Jev API: unexpected error ({status}): {detail}")


def ask(
    state: Any,
    questions: dict[str, Any],
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
) -> JevResponse:
    """
    POST /systemone — ask 1-64 typed questions (noul/choice/score) about `state`.

    Never auto-retried by this function: on a network error or 502/503/504 the
    outcome is uncertain, and blindly re-sending a POST can double-charge or
    double-fire a decision. If the caller wants to retry, it must call `ask`
    again deliberately after deciding that's safe.
    """
    body = {"model": model, "state": state, "questions": questions}
    try:
        resp = requests.post(
            f"{BASE_URL}/systemone", json=body, headers=_headers(), timeout=timeout
        )
    except requests.RequestException as exc:
        raise JevError(f"Jev API request failed: {exc}") from exc

    _raise_for_status(resp)
    data = resp.json()
    usage = data.get("usage") or {}
    return JevResponse(
        model=data.get("model", model),
        answers=data.get("answers", {}),
        usage=JevUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        ),
        run_id=resp.headers.get("X-Jev-Run-Id"),
    )


def list_models(timeout: float = DEFAULT_TIMEOUT) -> list[JevModel]:
    """
    GET /models — idempotent, safe to retry. Returns only connected/available
    models (any model the API marks unavailable via a `connected`/`available`
    flag is filtered out; models with no such flag are assumed available).

    Not used in the trading loop — this project always pins model=jev-latest.
    Provided for ops/setup use (see scripts/jev_live_check.py).
    """
    try:
        resp = requests.get(f"{BASE_URL}/models", headers=_headers(), timeout=timeout)
    except requests.RequestException as exc:
        raise JevError(f"Jev API request failed: {exc}") from exc

    _raise_for_status(resp)
    data = resp.json()
    models = data.get("models", [])
    connected = [m for m in models if m.get("connected", m.get("available", True))]
    return [JevModel(name=m["name"], description=m.get("description", "")) for m in connected]

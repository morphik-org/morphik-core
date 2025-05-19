from __future__ import annotations

"""Dynamic routing of per-app LLM API keys (enterprise feature).

Fetches cached per-app ``llm_keys`` from the control-plane and exposes them
via a context-local accessor so that pipeline stages (completion, rules,
graphs…) can transparently pick up user overrides.

Endpoint contract (see *control-plane* service)::

    GET /api/apps/{app_id}/settings?user_id=XYZ → {
        "completion": {"openai_api_key": "sk-..."},
        "rules": {"anthropic_api_key": "..."}
    }

The helper flattens the nested dict into a single-level mapping so callers can
just look for, e.g. ``openai_api_key`` – precedence order is *pipeline-name*
→ global.
"""

import logging
import os
from contextvars import ContextVar
from time import time
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_CTX: ContextVar[Dict[str, Any]] = ContextVar("llm_overrides", default={})
_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
_TTL = 300  # seconds


async def _fetch(app_id: Optional[str], user_id: Optional[str]) -> Dict[str, Any]:
    if not app_id:
        return {}

    key = f"{app_id}:{user_id or ''}"
    now = time()
    if key in _CACHE and now - _CACHE[key][1] < _TTL:
        return _CACHE[key][0]

    base_url = os.getenv("CONTROL_PLANE_URL", "http://localhost:9000")
    url = f"{base_url.rstrip('/')}/api/apps/{app_id}/settings"
    params = {"user_id": user_id} if user_id else None
    data: Dict[str, Any] = {}  # Ensure data is initialized

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                logger.warning("Unexpected settings payload: %s", data)
                data = {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Settings fetch failed: %s", exc)
        data = {}  # Ensure data is a dict on failure

    # No flattening, return raw data from control plane
    _CACHE[key] = (data, now)
    return data


async def set_llm_overrides(app_id: Optional[str], user_id: Optional[str]):
    user_settings = await _fetch(app_id, user_id)

    # Handle registered_models (if any) from the top level of user_settings
    registered_models_override = user_settings.get("registered_models", {})
    if isinstance(registered_models_override, dict) and registered_models_override:
        try:
            from core.config import get_settings  # noqa: WPS433 – runtime import

            settings = get_settings()
            if hasattr(settings, "REGISTERED_MODELS") and isinstance(settings.REGISTERED_MODELS, dict):
                settings.REGISTERED_MODELS.update(registered_models_override)  # noqa: WPS437 – runtime patch
            else:
                settings.REGISTERED_MODELS = registered_models_override  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover – best-effort
            logger.debug("Failed to merge registered models: %s", exc)

    # Store the entire fetched user_settings (nested structure) in the context
    _CTX.set(user_settings)


def get_current_overrides() -> Dict[str, Any]:
    return _CTX.get()

"""Shared Redis connection settings for API startup and ARQ workers."""

from __future__ import annotations

import os
from typing import Any

from arq.connections import RedisSettings

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379

LOCAL_REDIS_HOSTS = {"localhost", "127.0.0.1", "::1"}
_MISSING = object()
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _allow_insecure_remote_redis(settings: Any) -> bool:
    setting_value = getattr(settings, "ALLOW_INSECURE_REMOTE_REDIS", None)
    if setting_value is None:
        setting_value = os.getenv("ALLOW_INSECURE_REMOTE_REDIS", "")
    return str(setting_value).strip().lower() in _TRUE_VALUES


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None or value is _MISSING:
        return None
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _is_default_local_redis_target(redis_settings: RedisSettings) -> bool:
    return (
        redis_settings.host in LOCAL_REDIS_HOSTS
        and redis_settings.port == DEFAULT_REDIS_PORT
        and redis_settings.database == 0
        and not redis_settings.unix_socket_path
        and not redis_settings.username
        and not redis_settings.password
        and not redis_settings.ssl
    )


def build_redis_settings(settings: Any) -> RedisSettings:
    """Build ARQ Redis settings from Morphik settings.

    `REDIS_URL` is the canonical source because it can carry authentication,
    database selection, and TLS. The host/port fallback preserves the existing
    `morphik.toml` behavior where deployments override only those fields while
    leaving the default localhost URL in place.
    """
    raw_redis_url = getattr(settings, "REDIS_URL", _MISSING)
    if raw_redis_url is _MISSING:
        redis_url = DEFAULT_REDIS_URL
    elif raw_redis_url is None:
        raise ValueError("REDIS_URL is set but empty; expected a valid Redis DSN")
    else:
        redis_url = str(raw_redis_url).strip()
        if not redis_url:
            raise ValueError("REDIS_URL is set but empty; expected a valid Redis DSN")

    redis_settings = RedisSettings.from_dsn(redis_url)
    if (
        not redis_settings.ssl
        and redis_settings.host not in LOCAL_REDIS_HOSTS
        and (redis_settings.username or redis_settings.password)
        and not _allow_insecure_remote_redis(settings)
    ):
        raise ValueError(
            "Authenticated remote Redis must use rediss://; set ALLOW_INSECURE_REMOTE_REDIS=true to allow "
            "plaintext Redis credentials"
        )
    ssl_check_hostname = _optional_bool(getattr(settings, "REDIS_SSL_CHECK_HOSTNAME", _MISSING), "REDIS_SSL_CHECK_HOSTNAME")
    if redis_settings.ssl and ssl_check_hostname is not None:
        redis_settings.ssl_check_hostname = ssl_check_hostname

    redis_host = getattr(settings, "REDIS_HOST", DEFAULT_REDIS_HOST)
    redis_port = getattr(settings, "REDIS_PORT", DEFAULT_REDIS_PORT)

    if _is_default_local_redis_target(redis_settings) and (
        redis_host != DEFAULT_REDIS_HOST or redis_port != DEFAULT_REDIS_PORT
    ):
        redis_settings.host = redis_host
        redis_settings.port = redis_port

    # Keep worker/API pool behavior aligned and stable under transient startup races.
    redis_settings.conn_timeout = 5
    redis_settings.conn_retries = 15
    redis_settings.conn_retry_delay = 1
    return redis_settings


def should_manage_local_redis(redis_settings: RedisSettings) -> bool:
    """Return true when `start_server.py` should manage a local Redis container."""
    return (
        redis_settings.host in LOCAL_REDIS_HOSTS
        and redis_settings.port == DEFAULT_REDIS_PORT
        and not redis_settings.unix_socket_path
        and not redis_settings.username
        and not redis_settings.password
        and not redis_settings.ssl
    )

import atexit
import importlib
import signal
import sys
import types
from types import SimpleNamespace

import pytest

from core.redis_settings import DEFAULT_REDIS_URL, build_redis_settings, should_manage_local_redis


def test_build_redis_settings_preserves_dsn_auth_database_and_tls():
    settings = SimpleNamespace(
        REDIS_URL="rediss://user:secret@redis.example.com:6380/3",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
    )

    redis_settings = build_redis_settings(settings)

    assert redis_settings.host == "redis.example.com"
    assert redis_settings.port == 6380
    assert redis_settings.database == 3
    assert redis_settings.username == "user"
    assert redis_settings.password == "secret"
    assert redis_settings.ssl is True
    assert redis_settings.ssl_check_hostname is False
    assert redis_settings.conn_timeout == 5
    assert redis_settings.conn_retries == 15
    assert redis_settings.conn_retry_delay == 1


def test_build_redis_settings_allows_hostname_verification_opt_in_for_tls():
    settings = SimpleNamespace(
        REDIS_URL="rediss://user:secret@redis.example.com:6380/3",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        REDIS_SSL_CHECK_HOSTNAME=True,
    )

    redis_settings = build_redis_settings(settings)

    assert redis_settings.ssl is True
    assert redis_settings.ssl_check_hostname is True


def test_build_redis_settings_rejects_empty_url():
    settings = SimpleNamespace(
        REDIS_URL=" ",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
    )

    with pytest.raises(ValueError, match="REDIS_URL is set but empty"):
        build_redis_settings(settings)


def test_build_redis_settings_rejects_plaintext_remote_credentials_by_default():
    settings = SimpleNamespace(
        REDIS_URL="redis://:secret@redis.example.com:6380/2",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
    )

    with pytest.raises(ValueError, match="Authenticated remote Redis must use rediss://"):
        build_redis_settings(settings)


def test_build_redis_settings_allows_plaintext_remote_credentials_with_explicit_override():
    settings = SimpleNamespace(
        REDIS_URL="redis://:secret@redis.example.com:6380/2",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        ALLOW_INSECURE_REMOTE_REDIS=True,
    )

    redis_settings = build_redis_settings(settings)

    assert redis_settings.host == "redis.example.com"
    assert redis_settings.port == 6380
    assert redis_settings.database == 2
    assert redis_settings.password == "secret"
    assert redis_settings.ssl is False


def test_build_redis_settings_preserves_host_port_fallback_for_default_url():
    settings = SimpleNamespace(
        REDIS_URL=DEFAULT_REDIS_URL,
        REDIS_HOST="redis",
        REDIS_PORT=6381,
    )

    redis_settings = build_redis_settings(settings)

    assert redis_settings.host == "redis"
    assert redis_settings.port == 6381
    assert redis_settings.database == 0
    assert redis_settings.password is None
    assert redis_settings.ssl is False


def test_build_redis_settings_applies_host_port_fallback_for_equivalent_default_url():
    settings = SimpleNamespace(
        REDIS_URL="redis://localhost:6379",
        REDIS_HOST="redis",
        REDIS_PORT=6381,
    )

    redis_settings = build_redis_settings(settings)

    assert redis_settings.host == "redis"
    assert redis_settings.port == 6381
    assert redis_settings.database == 0
    assert redis_settings.password is None
    assert redis_settings.ssl is False


def test_should_manage_local_redis_only_for_default_local_target():
    local_settings = build_redis_settings(
        SimpleNamespace(
            REDIS_URL=DEFAULT_REDIS_URL,
            REDIS_HOST="localhost",
            REDIS_PORT=6379,
        )
    )
    local_nonzero_db_settings = build_redis_settings(
        SimpleNamespace(
            REDIS_URL="redis://localhost:6379/2",
            REDIS_HOST="localhost",
            REDIS_PORT=6379,
        )
    )
    external_settings = build_redis_settings(
        SimpleNamespace(
            REDIS_URL="rediss://:secret@redis.example.com:6380/2",
            REDIS_HOST="localhost",
            REDIS_PORT=6379,
        )
    )

    assert should_manage_local_redis(local_settings) is True
    assert should_manage_local_redis(local_nonzero_db_settings) is True
    assert should_manage_local_redis(external_settings) is False


def test_should_not_manage_local_redis_for_unix_socket_dsn():
    redis_settings = build_redis_settings(
        SimpleNamespace(
            REDIS_URL="unix:///tmp/morphik-redis.sock",
            REDIS_HOST="localhost",
            REDIS_PORT=6379,
        )
    )

    assert redis_settings.unix_socket_path == "/tmp/morphik-redis.sock"
    assert should_manage_local_redis(redis_settings) is False


@pytest.mark.asyncio
async def test_app_lifespan_uses_shared_redis_settings(monkeypatch):
    import core.app_factory as app_factory

    captured = {}

    class FakePool:
        def close(self):
            captured["closed"] = True

    class AsyncInitializable:
        async def initialize(self):
            return True

    services_init = types.ModuleType("core.services_init")
    for name, value in {
        "database": AsyncInitializable(),
        "vector_store": AsyncInitializable(),
        "v2_chunk_store": AsyncInitializable(),
        "colpali_vector_store": None,
        "settings": SimpleNamespace(
            REDIS_URL="rediss://:secret@redis.example.com:6380/2",
            REDIS_HOST="localhost",
            REDIS_PORT=6379,
            TELEMETRY_ENABLED=False,
        ),
    }.items():
        setattr(services_init, name, value)
    monkeypatch.setitem(sys.modules, "core.services_init", services_init)

    async def fake_create_pool(redis_settings):
        captured["redis_settings"] = redis_settings
        return FakePool()

    monkeypatch.setattr(app_factory.arq, "create_pool", fake_create_pool)

    app = SimpleNamespace(state=SimpleNamespace())
    async with app_factory.lifespan(app):
        redis_settings = captured["redis_settings"]
        assert redis_settings.host == "redis.example.com"
        assert redis_settings.port == 6380
        assert redis_settings.database == 2
        assert redis_settings.password == "secret"
        assert redis_settings.ssl is True
        assert redis_settings.ssl_check_hostname is False
        assert app.state.redis_pool is not None

    assert captured["closed"] is True


def test_wait_for_redis_uses_dsn_settings_for_connection(monkeypatch):
    import start_server

    captured = {}
    redis_settings = build_redis_settings(
        SimpleNamespace(
            REDIS_URL="rediss://user:secret@redis.example.com:6380/2",
            REDIS_HOST="localhost",
            REDIS_PORT=6379,
        )
    )

    class FakePool:
        async def ping(self):
            captured["pinged"] = True

        async def aclose(self):
            captured["closed"] = True

    async def fake_create_pool(settings):
        captured["settings"] = settings
        return FakePool()

    monkeypatch.setattr(start_server.arq, "create_pool", fake_create_pool)

    assert start_server.wait_for_redis(redis_settings, timeout=3) is True

    readiness_settings = captured["settings"]
    assert readiness_settings.host == "redis.example.com"
    assert readiness_settings.port == 6380
    assert readiness_settings.database == 2
    assert readiness_settings.username == "user"
    assert readiness_settings.password == "secret"
    assert readiness_settings.ssl is True
    assert readiness_settings.ssl_check_hostname is False
    assert readiness_settings.conn_timeout == 1
    assert readiness_settings.conn_retries == 0
    assert captured["pinged"] is True
    assert captured["closed"] is True


def test_wait_for_redis_respects_timeout_deadline(monkeypatch):
    import start_server

    attempts = []
    sleeps = []
    now = {"value": 0.0}
    redis_settings = build_redis_settings(
        SimpleNamespace(
            REDIS_URL="rediss://redis.example.com:6380/2",
            REDIS_HOST="localhost",
            REDIS_PORT=6379,
        )
    )

    async def fake_create_pool(settings):
        attempts.append(settings)
        now["value"] += 0.4
        raise OSError("redis unavailable")

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now["value"] += seconds

    monkeypatch.setattr(start_server.arq, "create_pool", fake_create_pool)
    monkeypatch.setattr(start_server.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(start_server.time, "sleep", fake_sleep)

    assert start_server.wait_for_redis(redis_settings, timeout=1) is False
    assert len(attempts) == 1
    assert attempts[0].conn_retries == 0
    assert sleeps == [0.6]
    assert now["value"] == pytest.approx(1.0)


def test_start_server_import_has_no_shutdown_handler_side_effects(monkeypatch):
    int_handler = signal.getsignal(signal.SIGINT)
    term_handler = signal.getsignal(signal.SIGTERM)
    signal_calls = []
    atexit_calls = []

    monkeypatch.delitem(sys.modules, "start_server", raising=False)
    monkeypatch.setattr(signal, "signal", lambda *args: signal_calls.append(args))
    monkeypatch.setattr(atexit, "register", lambda *args: atexit_calls.append(args))
    start_server = importlib.import_module("start_server")

    assert signal.getsignal(signal.SIGINT) is int_handler
    assert signal.getsignal(signal.SIGTERM) is term_handler
    assert signal_calls == []
    assert atexit_calls == []
    monkeypatch.setitem(sys.modules, "start_server", start_server)


def test_start_server_skips_local_container_management_for_external_dsn(monkeypatch):
    import start_server

    calls = []
    settings = SimpleNamespace(
        REDIS_URL="rediss://:secret@redis.example.com:6380/2",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        HOST="127.0.0.1",
        PORT=8000,
    )

    monkeypatch.setattr(sys, "argv", ["start_server.py", "--skip-ollama-check"])
    monkeypatch.setattr(start_server, "register_shutdown_handlers", lambda: None)
    monkeypatch.setattr(start_server, "setup_logging", lambda log_level: None)
    monkeypatch.setattr(start_server, "load_local_env", lambda override: None)
    monkeypatch.setattr(start_server, "get_settings", lambda: settings)
    monkeypatch.setattr(start_server, "check_and_start_redis", lambda: calls.append("manage-redis"))
    monkeypatch.setattr(
        start_server,
        "wait_for_redis",
        lambda redis_settings: calls.append(("wait", redis_settings.host, redis_settings.port)) or True,
    )
    monkeypatch.setattr(start_server, "start_arq_worker", lambda: calls.append("worker"))
    monkeypatch.setattr(start_server.uvicorn, "run", lambda *args, **kwargs: calls.append(("uvicorn", kwargs)))

    start_server.main()

    assert "manage-redis" not in calls
    assert ("wait", "redis.example.com", 6380) in calls
    assert "worker" in calls


def test_start_server_manages_local_container_for_local_nonzero_db(monkeypatch):
    import start_server

    calls = []
    settings = SimpleNamespace(
        REDIS_URL="redis://localhost:6379/2",
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        HOST="127.0.0.1",
        PORT=8000,
    )

    monkeypatch.setattr(sys, "argv", ["start_server.py", "--skip-ollama-check"])
    monkeypatch.setattr(start_server, "register_shutdown_handlers", lambda: None)
    monkeypatch.setattr(start_server, "setup_logging", lambda log_level: None)
    monkeypatch.setattr(start_server, "load_local_env", lambda override: None)
    monkeypatch.setattr(start_server, "get_settings", lambda: settings)
    monkeypatch.setattr(start_server, "check_and_start_redis", lambda: calls.append("manage-redis"))
    monkeypatch.setattr(
        start_server,
        "wait_for_redis",
        lambda redis_settings: calls.append(("wait", redis_settings.host, redis_settings.port)) or True,
    )
    monkeypatch.setattr(start_server, "start_arq_worker", lambda: calls.append("worker"))
    monkeypatch.setattr(start_server.uvicorn, "run", lambda *args, **kwargs: calls.append(("uvicorn", kwargs)))

    start_server.main()

    assert "manage-redis" in calls
    assert ("wait", "localhost", 6379) in calls
    assert "worker" in calls

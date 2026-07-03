"""Unit tests for authentication secret settings."""

from pathlib import Path
import re

import pytest

from core.config import get_settings


ROOT_CONFIG = Path(__file__).resolve().parents[3] / "morphik.toml"
STRONG_JWT_SECRET = "jwt-secret-0123456789abcdef0123456789"
STRONG_SESSION_SECRET = "session-secret-0123456789abcdef0123456789"
STRONG_LOCAL_URI_PASSWORD = "local-uri-password-0123456789abcdef0123456789"


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)
    monkeypatch.delenv("LOCAL_URI_PASSWORD", raising=False)
    yield
    get_settings.cache_clear()


def _write_config(tmp_path, *, bypass_auth_mode):
    text = ROOT_CONFIG.read_text()
    replacement = f"bypass_auth_mode = {'true' if bypass_auth_mode else 'false'}"
    text, replacements = re.subn(r"bypass_auth_mode = (true|false)", replacement, text, count=1)
    assert replacements == 1
    (tmp_path / "morphik.toml").write_text(text)


def test_requires_session_secret_when_auth_bypass_is_disabled(tmp_path, monkeypatch):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", STRONG_JWT_SECRET)
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="SESSION_SECRET_KEY is required when bypass_auth_mode is disabled"):
        get_settings()


def test_requires_both_signing_secrets_when_auth_bypass_is_disabled(tmp_path, monkeypatch):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="JWT_SECRET_KEY, SESSION_SECRET_KEY are required when bypass_auth_mode is disabled",
    ):
        get_settings()


@pytest.mark.parametrize("secret_value", ["", "   ", '""', '"   "', "''", "'   '"])
def test_rejects_blank_session_secret_when_auth_bypass_is_disabled(tmp_path, monkeypatch, secret_value):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", STRONG_JWT_SECRET)
    monkeypatch.setenv("SESSION_SECRET_KEY", secret_value)

    with pytest.raises(ValueError, match="SESSION_SECRET_KEY is required when bypass_auth_mode is disabled"):
        get_settings()


@pytest.mark.parametrize("secret_value", ["", "   ", '""', '"   "', "''", "'   '"])
def test_rejects_blank_jwt_secret_when_auth_bypass_is_disabled(tmp_path, monkeypatch, secret_value):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", secret_value)
    monkeypatch.setenv("SESSION_SECRET_KEY", STRONG_SESSION_SECRET)

    with pytest.raises(ValueError, match="JWT_SECRET_KEY is required when bypass_auth_mode is disabled"):
        get_settings()


@pytest.mark.parametrize(
    ("jwt_secret", "session_secret", "error_match"),
    [
        (
            "your-super-secret-key-change-in-production",
            STRONG_SESSION_SECRET,
            "JWT_SECRET_KEY uses an example or development default value",
        ),
        (
            "dev-secret-key",
            STRONG_SESSION_SECRET,
            "JWT_SECRET_KEY uses an example or development default value",
        ),
        (
            STRONG_JWT_SECRET,
            "your-session-secret-key-change-in-production",
            "SESSION_SECRET_KEY uses an example or development default value",
        ),
        (
            STRONG_JWT_SECRET,
            "super-secret-dev-session-key",
            "SESSION_SECRET_KEY uses an example or development default value",
        ),
        (
            "your-secure-jwt-key-here",
            "your-secure-session-key-here",
            "JWT_SECRET_KEY, SESSION_SECRET_KEY use an example or development default value",
        ),
        (
            '"dev-secret-key"',
            STRONG_SESSION_SECRET,
            "JWT_SECRET_KEY uses an example or development default value",
        ),
        (
            STRONG_JWT_SECRET,
            "'your-secure-session-key-here'",
            "SESSION_SECRET_KEY uses an example or development default value",
        ),
    ],
)
def test_rejects_example_auth_secrets_when_auth_bypass_is_disabled(
    tmp_path, monkeypatch, jwt_secret, session_secret, error_match
):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", jwt_secret)
    monkeypatch.setenv("SESSION_SECRET_KEY", session_secret)

    with pytest.raises(ValueError, match=error_match):
        get_settings()


@pytest.mark.parametrize(
    ("jwt_secret", "session_secret", "error_match"),
    [
        ("short-jwt-secret", STRONG_SESSION_SECRET, "JWT_SECRET_KEY is too short"),
        (STRONG_JWT_SECRET, "short-session-secret", "SESSION_SECRET_KEY is too short"),
        ("short-jwt-secret", "short-session-secret", "JWT_SECRET_KEY, SESSION_SECRET_KEY are too short"),
    ],
)
def test_rejects_short_auth_secrets_when_auth_bypass_is_disabled(
    tmp_path, monkeypatch, jwt_secret, session_secret, error_match
):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", jwt_secret)
    monkeypatch.setenv("SESSION_SECRET_KEY", session_secret)

    with pytest.raises(ValueError, match=error_match):
        get_settings()


def test_allows_missing_local_uri_password_when_auth_bypass_is_disabled(tmp_path, monkeypatch):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", STRONG_JWT_SECRET)
    monkeypatch.setenv("SESSION_SECRET_KEY", STRONG_SESSION_SECRET)
    monkeypatch.delenv("LOCAL_URI_PASSWORD", raising=False)

    settings = get_settings()

    assert settings.LOCAL_URI_PASSWORD is None
    assert settings.bypass_auth_mode is False


@pytest.mark.parametrize("local_uri_password", ["", "   ", '""', '"   "', "''", "'   '"])
def test_treats_blank_local_uri_password_as_disabled(tmp_path, monkeypatch, local_uri_password):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", STRONG_JWT_SECRET)
    monkeypatch.setenv("SESSION_SECRET_KEY", STRONG_SESSION_SECRET)
    monkeypatch.setenv("LOCAL_URI_PASSWORD", local_uri_password)

    settings = get_settings()

    assert settings.LOCAL_URI_PASSWORD is None
    assert settings.bypass_auth_mode is False


def test_loads_strong_local_uri_password_when_configured(tmp_path, monkeypatch):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", STRONG_JWT_SECRET)
    monkeypatch.setenv("SESSION_SECRET_KEY", STRONG_SESSION_SECRET)
    monkeypatch.setenv("LOCAL_URI_PASSWORD", f'"{STRONG_LOCAL_URI_PASSWORD}"')

    settings = get_settings()

    assert settings.LOCAL_URI_PASSWORD == STRONG_LOCAL_URI_PASSWORD


@pytest.mark.parametrize(
    ("local_uri_password", "error_match"),
    [
        ("change-me-local-uri-password", "LOCAL_URI_PASSWORD uses an example or development default value"),
        ("your-local-uri-password-here", "LOCAL_URI_PASSWORD uses an example or development default value"),
        ("<replace-with-local-uri-password>", "LOCAL_URI_PASSWORD uses an example or development default value"),
        ("short-local-uri-password", "LOCAL_URI_PASSWORD is too short"),
    ],
)
def test_rejects_weak_local_uri_password_when_configured(
    tmp_path, monkeypatch, local_uri_password, error_match
):
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", STRONG_JWT_SECRET)
    monkeypatch.setenv("SESSION_SECRET_KEY", STRONG_SESSION_SECRET)
    monkeypatch.setenv("LOCAL_URI_PASSWORD", local_uri_password)

    with pytest.raises(ValueError, match=error_match):
        get_settings()


def test_rejects_weak_local_uri_password_when_auth_bypass_is_enabled(tmp_path, monkeypatch):
    _write_config(tmp_path, bypass_auth_mode=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("LOCAL_URI_PASSWORD", "short-local-uri-password")

    with pytest.raises(ValueError, match="LOCAL_URI_PASSWORD is too short"):
        get_settings()


def test_accepts_auth_secrets_at_minimum_length_when_auth_bypass_is_disabled(tmp_path, monkeypatch):
    jwt_secret = "j" * 32
    session_secret = "s" * 32
    _write_config(tmp_path, bypass_auth_mode=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", jwt_secret)
    monkeypatch.setenv("SESSION_SECRET_KEY", session_secret)
    monkeypatch.setenv("LOCAL_URI_PASSWORD", "")

    settings = get_settings()

    assert settings.JWT_SECRET_KEY == jwt_secret
    assert settings.SESSION_SECRET_KEY == session_secret
    assert settings.LOCAL_URI_PASSWORD is None
    assert settings.bypass_auth_mode is False


def test_allows_default_auth_secrets_when_auth_bypass_is_enabled(tmp_path, monkeypatch):
    _write_config(tmp_path, bypass_auth_mode=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)

    settings = get_settings()

    assert settings.JWT_SECRET_KEY == "dev-secret-key"
    assert settings.SESSION_SECRET_KEY == "super-secret-dev-session-key"
    assert settings.bypass_auth_mode is True


def test_uses_default_auth_secrets_for_blank_values_when_auth_bypass_is_enabled(tmp_path, monkeypatch):
    _write_config(tmp_path, bypass_auth_mode=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", "")
    monkeypatch.setenv("SESSION_SECRET_KEY", "   ")

    settings = get_settings()

    assert settings.JWT_SECRET_KEY == "dev-secret-key"
    assert settings.SESSION_SECRET_KEY == "super-secret-dev-session-key"
    assert settings.bypass_auth_mode is True


def test_uses_default_auth_secrets_for_quoted_blank_values_when_auth_bypass_is_enabled(tmp_path, monkeypatch):
    _write_config(tmp_path, bypass_auth_mode=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("JWT_SECRET_KEY", '""')
    monkeypatch.setenv("SESSION_SECRET_KEY", "'   '")

    settings = get_settings()

    assert settings.JWT_SECRET_KEY == "dev-secret-key"
    assert settings.SESSION_SECRET_KEY == "super-secret-dev-session-key"
    assert settings.bypass_auth_mode is True

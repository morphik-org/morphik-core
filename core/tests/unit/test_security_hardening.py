"""Token checks, session key, Drive credential storage and connector lookups."""

import asyncio
import json
import pickle
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException

import core.auth_utils as auth_utils
from core.config import get_settings

SECRET = "unit-test-signing-key-that-is-long-enough"


@pytest.fixture
def token_settings(monkeypatch):
    settings = SimpleNamespace(
        bypass_auth_mode=False,
        JWT_SECRET_KEY=SECRET,
        JWT_ALGORITHM="HS256",
        REQUIRE_APP_ID_IN_TOKEN=False,
    )
    monkeypatch.setattr(auth_utils, "settings", settings)

    async def app_is_active(app_id, token_version=None, redis_pool=None):
        return None

    monkeypatch.setattr(auth_utils, "ensure_app_is_active", app_is_active)
    return settings


def _verify(payload: dict):
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    return asyncio.run(auth_utils.verify_token(request, authorization=f"Bearer {token}"))


def _exp(days: int) -> int:
    return int((datetime.now(UTC) + timedelta(days=days)).timestamp())


def test_valid_token_is_accepted(token_settings):
    ctx = _verify({"user_id": "u1", "app_id": "a1", "exp": _exp(1)})

    assert (ctx.user_id, ctx.app_id) == ("u1", "a1")


def test_token_without_expiry_is_rejected_not_a_server_error(token_settings):
    with pytest.raises(HTTPException) as err:
        _verify({"user_id": "u1", "app_id": "a1"})

    assert err.value.status_code == 401


def test_expired_token_is_rejected(token_settings):
    with pytest.raises(HTTPException) as err:
        _verify({"user_id": "u1", "app_id": "a1", "exp": _exp(-1)})

    assert err.value.status_code == 401


def test_token_without_app_id_is_accepted_by_default(token_settings):
    assert _verify({"user_id": "u1", "exp": _exp(1)}).app_id is None


def test_token_without_app_id_is_rejected_when_required(token_settings):
    token_settings.REQUIRE_APP_ID_IN_TOKEN = True

    with pytest.raises(HTTPException) as err:
        _verify({"user_id": "u1", "exp": _exp(1)})

    assert err.value.status_code == 401
    assert _verify({"user_id": "u1", "app_id": "a1", "exp": _exp(1)}).app_id == "a1"


def test_session_key_is_never_the_public_default(monkeypatch):
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)
    monkeypatch.setenv("JWT_SECRET_KEY", SECRET)
    get_settings.cache_clear()
    try:
        derived = get_settings().SESSION_SECRET_KEY
        monkeypatch.setenv("SESSION_SECRET_KEY", "explicit-session-key")
        get_settings.cache_clear()
        explicit = get_settings().SESSION_SECRET_KEY
    finally:
        get_settings.cache_clear()

    assert derived != "super-secret-dev-session-key"
    assert SECRET not in derived and len(derived) == 64
    assert explicit == "explicit-session-key"


def test_drive_credentials_are_stored_as_json_and_pickles_are_not_loaded(tmp_path, monkeypatch):
    from google.oauth2.credentials import Credentials

    from ee.services.connectors import google_drive_connector as drive

    monkeypatch.setattr(drive, "get_ee_settings", lambda: SimpleNamespace(GOOGLE_TOKEN_STORAGE_PATH=str(tmp_path)))
    connector = drive.GoogleDriveConnector(user_morphik_id="user-1")
    connector.creds = Credentials(
        token="access",
        refresh_token="refresh",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="client",
        client_secret="secret",
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
        expiry=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
    )
    connector._save_credentials()

    token_path = connector._get_user_token_path()
    assert token_path.suffix == ".json"
    assert json.loads(token_path.read_text())["refresh_token"] == "refresh"

    monkeypatch.setattr(drive, "build", lambda *args, **kwargs: object())
    reloaded = drive.GoogleDriveConnector(user_morphik_id="user-1")
    assert reloaded.creds is not None and reloaded.creds.refresh_token == "refresh"

    # A pickle in the token file is treated as unreadable, never executed.
    token_path.write_bytes(pickle.dumps({"anything": True}))
    assert drive.GoogleDriveConnector(user_morphik_id="user-1").creds is None


def test_unknown_connector_type_is_a_404():
    from ee.routers import connectors_router
    from ee.services.connector_service import ConnectorService

    service = ConnectorService(auth_context=SimpleNamespace(user_id="user-1", entity_id=None))

    with pytest.raises(HTTPException) as err:
        asyncio.run(connectors_router._get_connector_or_404(service, "sharepoint"))

    assert err.value.status_code == 404

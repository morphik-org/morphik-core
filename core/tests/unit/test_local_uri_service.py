import uuid

import pytest

from core.models.apps import AppModel
from core.services.local_uri_service import persist_local_app_record


class _FakeSessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, existing=None):
        self.records = existing or {}
        self.committed = False

    async def get(self, _model, app_id):
        return self.records.get(app_id)

    def add(self, app_record):
        self.records[app_record.app_id] = app_record

    async def commit(self):
        self.committed = True


class _FakeDatabase:
    def __init__(self, session):
        self._session = session

    def async_session(self):
        return _FakeSessionContext(self._session)


@pytest.mark.asyncio
async def test_persist_local_app_record_creates_row_with_creator_and_uri():
    session = _FakeSession()
    database = _FakeDatabase(session)

    await persist_local_app_record(
        database=database,
        app_id="app-local-1",
        user_name="testuser",
        uri="morphik://testuser:token@127.0.0.1:8003",
        token_version=0,
    )

    assert session.committed is True
    app = session.records["app-local-1"]
    assert app.app_id == "app-local-1"
    assert app.user_id is None
    assert app.created_by_user_id == "testuser"
    assert app.name == "testuser"
    assert app.uri == "morphik://testuser:token@127.0.0.1:8003"
    assert app.token_version == 0


@pytest.mark.asyncio
async def test_persist_local_app_record_updates_existing_row():
    existing_app = AppModel(
        app_id="app-local-2",
        user_id=None,
        created_by_user_id="old-user",
        name="old-name",
        uri="morphik://old-name:old-token@127.0.0.1:8003",
        token_version=5,
    )
    session = _FakeSession(existing={"app-local-2": existing_app})
    database = _FakeDatabase(session)
    user_uuid = str(uuid.uuid4())

    await persist_local_app_record(
        database=database,
        app_id="app-local-2",
        user_name=user_uuid,
        uri="morphik://new-name:new-token@127.0.0.1:8003",
        token_version=0,
    )

    assert session.committed is True
    assert existing_app.created_by_user_id == user_uuid
    assert existing_app.name == user_uuid
    assert existing_app.uri == "morphik://new-name:new-token@127.0.0.1:8003"
    assert existing_app.token_version == 0
    assert str(existing_app.user_id) == user_uuid

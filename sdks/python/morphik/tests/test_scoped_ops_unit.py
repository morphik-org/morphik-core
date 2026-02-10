import httpx
import jwt
import pytest
from morphik.async_ import AsyncMorphik
from morphik.sync import Folder, Morphik


def _mock_document_response(filename="sample.txt"):
    return {
        "external_id": "doc-123",
        "content_type": "text/plain",
        "filename": filename,
    }


def _make_sync_client():
    client = Morphik()
    calls = []

    def fake_request(method, endpoint, data=None, files=None, params=None):
        calls.append(
            {
                "method": method,
                "endpoint": endpoint,
                "data": data,
                "params": params,
            }
        )
        if isinstance(endpoint, str) and endpoint.startswith("retrieve/chunks"):
            return []
        if endpoint == "batch/documents":
            return []
        if endpoint == "batch/chunks":
            return []
        if isinstance(endpoint, str) and endpoint.startswith("documents/filename/"):
            return _mock_document_response()
        # Return mock ListDocsResponse format
        return {
            "documents": [],
            "skip": data.get("skip", 0) if isinstance(data, dict) else 0,
            "limit": data.get("limit", 100) if isinstance(data, dict) else 100,
            "returned_count": 0,
            "has_more": False,
        }

    client._request = fake_request  # type: ignore[attr-defined]
    return client, calls


async def _make_async_client():
    client = AsyncMorphik()
    calls = []

    async def fake_request(method, endpoint, data=None, files=None, params=None):
        calls.append(
            {
                "method": method,
                "endpoint": endpoint,
                "data": data,
                "params": params,
            }
        )
        if isinstance(endpoint, str) and endpoint.startswith("retrieve/chunks"):
            return []
        if endpoint == "batch/documents":
            return []
        if endpoint == "batch/chunks":
            return []
        if isinstance(endpoint, str) and endpoint.startswith("documents/filename/"):
            return _mock_document_response()
        # Return mock ListDocsResponse format
        return {
            "documents": [],
            "skip": data.get("skip", 0) if isinstance(data, dict) else 0,
            "limit": data.get("limit", 100) if isinstance(data, dict) else 100,
            "returned_count": 0,
            "has_more": False,
        }

    client._request = fake_request  # type: ignore[attr-defined]
    return client, calls


def test_sync_list_documents_payloads_across_scopes():
    client, calls = _make_sync_client()
    try:
        client.list_documents(skip=5, limit=10, filters={"department": "ops"})
        base_call = calls.pop()
        assert base_call["method"] == "POST"
        assert base_call["endpoint"] == "documents/list_docs"
        assert base_call["params"] == {}
        assert base_call["data"]["skip"] == 5
        assert base_call["data"]["limit"] == 10
        assert base_call["data"]["document_filters"] == {"department": "ops"}
        assert base_call["data"]["return_documents"] is True

        folder = Folder(client, "alpha")
        folder.list_documents(filters={"project": "z"}, additional_folders=["beta"])
        folder_call = calls.pop()
        assert folder_call["params"]["folder_name"] == ["alpha", "beta"]
        assert folder_call["data"]["document_filters"] == {"project": "z"}

        user = client.signin("user-1")
        user.list_documents(limit=7, filters={"team": "blue"})
        user_call = calls.pop()
        assert user_call["params"]["end_user_id"] == "user-1"
        assert user_call["data"]["document_filters"] == {"team": "blue"}
        assert user_call["data"]["limit"] == 7

        folder_user = folder.signin("user-2")
        folder_user.list_documents(additional_folders=["shared"], filters=None)
        folder_user_call = calls.pop()
        assert folder_user_call["params"]["folder_name"] == ["alpha", "shared"]
        assert folder_user_call["params"]["end_user_id"] == "user-2"
        assert folder_user_call["data"]["document_filters"] is None
    finally:
        client.close()


def test_async_client_http2_toggle(monkeypatch):
    captured = []

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs.get("http2"))

        async def aclose(self):
            return None

    monkeypatch.setattr("morphik.async_.httpx.AsyncClient", DummyAsyncClient)

    AsyncMorphik()
    assert captured[-1] is False

    AsyncMorphik(http2=True)
    assert captured[-1] is True

    AsyncMorphik(http2=True, is_local=True)
    assert captured[-1] is False


def test_sync_client_http2_toggle(monkeypatch):
    captured = []

    class DummyClient:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs.get("http2"))

        def close(self):
            return None

    monkeypatch.setattr("morphik.sync.httpx.Client", DummyClient)

    Morphik()
    assert captured[-1] is False

    Morphik(http2=True)
    assert captured[-1] is True

    Morphik(http2=True, is_local=True)
    assert captured[-1] is False


def test_sync_client_accepts_plain_http_uri_without_auth():
    client = Morphik("http://0.0.0.0:8000")
    try:
        assert client._logic._base_url == "http://0.0.0.0:8000"
        assert client._logic._auth_token is None
        assert client._logic._is_local is True
    finally:
        client.close()


def test_sync_client_preserves_http_path_prefix():
    client = Morphik("https://api.example.com/morphik/v1/")
    try:
        assert client._logic._base_url == "https://api.example.com/morphik/v1"
        assert client._logic._get_url("ping") == "https://api.example.com/morphik/v1/ping"
    finally:
        client.close()


def test_sync_client_parses_morphik_uri_with_token():
    token = jwt.encode({"sub": "test-user"}, "test-secret", algorithm="HS256")
    client = Morphik(f"morphik://owner:{token}@api.morphik.ai")
    try:
        assert client._logic._base_url == "https://api.morphik.ai"
        assert client._logic._auth_token == token
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_client_accepts_plain_http_uri_without_auth():
    client = AsyncMorphik("http://0.0.0.0:8000")
    try:
        assert client._logic._base_url == "http://0.0.0.0:8000"
        assert client._logic._auth_token is None
        assert client._logic._is_local is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_client_preserves_http_path_prefix():
    client = AsyncMorphik("https://api.example.com/morphik/v1/")
    try:
        assert client._logic._base_url == "https://api.example.com/morphik/v1"
        assert client._logic._get_url("ping") == "https://api.example.com/morphik/v1/ping"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_http2_fallback_on_remote_protocol_error(monkeypatch):
    created_http2 = []

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self.http2 = kwargs.get("http2")
            created_http2.append(self.http2)

        async def request(self, method, url, headers=None, params=None, **kwargs):
            if self.http2:
                raise httpx.RemoteProtocolError("http2 failed")
            return DummyResponse({"ok": True})

        async def aclose(self):
            return None

    monkeypatch.setattr("morphik.async_.httpx.AsyncClient", DummyAsyncClient)

    client = AsyncMorphik(http2=True)
    try:
        response = await client._request("GET", "ping")
        assert response == {"ok": True}
        assert created_http2 == [True, False]
        assert client._http2 is False
    finally:
        await client.close()


def test_sync_http2_fallback_on_remote_protocol_error(monkeypatch):
    created_http2 = []

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.http2 = kwargs.get("http2")
            created_http2.append(self.http2)

        def request(self, method, url, headers=None, params=None, **kwargs):
            if self.http2:
                raise httpx.RemoteProtocolError("http2 failed")
            return DummyResponse({"ok": True})

        def close(self):
            return None

    monkeypatch.setattr("morphik.sync.httpx.Client", DummyClient)

    client = Morphik(http2=True)
    try:
        response = client._request("GET", "ping")
        assert response == {"ok": True}
        assert created_http2 == [True, False]
        assert client._http2 is False
    finally:
        client.close()


def test_sync_get_document_by_filename_scoped_params_and_encoding():
    client, calls = _make_sync_client()
    try:
        doc = client.get_document_by_filename(
            "folder/file name.txt",
            folder_name="/team/docs",
            folder_depth=2,
            end_user_id="user-1",
        )
        call = calls.pop()
        assert call["endpoint"] == "documents/filename/folder%2Ffile%20name.txt"
        assert call["params"] == {"folder_name": "/team/docs", "folder_depth": 2, "end_user_id": "user-1"}
        assert doc.external_id == "doc-123"
    finally:
        client.close()


def test_sync_folder_get_document_by_filename_scoped():
    client, calls = _make_sync_client()
    try:
        folder = Folder(client, "docs", full_path="/projects/docs")
        folder.get_document_by_filename("report.pdf")
        call = calls.pop()
        assert call["endpoint"] == "documents/filename/report.pdf"
        assert call["params"] == {"folder_name": "/projects/docs"}
    finally:
        client.close()


def test_sync_user_scope_get_document_by_filename_scoped():
    client, calls = _make_sync_client()
    try:
        user = client.signin("user-99")
        user.get_document_by_filename("note.txt")
        call = calls.pop()
        assert call["params"] == {"end_user_id": "user-99"}

        folder = Folder(client, "docs", full_path="/projects/docs")
        folder_user = folder.signin("user-42")
        folder_user.get_document_by_filename("plan.md")
        call = calls.pop()
        assert call["params"] == {"folder_name": "/projects/docs", "end_user_id": "user-42"}
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_list_documents_payloads_across_scopes():
    client, calls = await _make_async_client()
    try:
        await client.list_documents(skip=2, limit=4, filters={"region": "na"})
        base_call = calls.pop()
        assert base_call["params"] == {}
        assert base_call["data"]["skip"] == 2
        assert base_call["data"]["limit"] == 4
        assert base_call["data"]["document_filters"] == {"region": "na"}

        folder = client.get_folder_by_name("ops")
        await folder.list_documents(filters={"category": "a"}, additional_folders=["archive"])
        folder_call = calls.pop()
        assert folder_call["params"]["folder_name"] == ["ops", "archive"]
        assert folder_call["data"]["document_filters"] == {"category": "a"}

        user = client.signin("usr-5")
        await user.list_documents(filters={"tag": "beta"})
        user_call = calls.pop()
        assert user_call["params"]["end_user_id"] == "usr-5"
        assert user_call["data"]["document_filters"] == {"tag": "beta"}

        folder_user = folder.signin("usr-7")
        await folder_user.list_documents(additional_folders=["shared"], filters=None)
        folder_user_call = calls.pop()
        assert folder_user_call["params"]["folder_name"] == ["ops", "shared"]
        assert folder_user_call["params"]["end_user_id"] == "usr-7"
        assert folder_user_call["data"]["document_filters"] is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_get_document_by_filename_scoped_params_and_encoding():
    client, calls = await _make_async_client()
    try:
        doc = await client.get_document_by_filename(
            "folder/file name.txt",
            folder_name="/team/docs",
            folder_depth=2,
            end_user_id="user-1",
        )
        call = calls.pop()
        assert call["endpoint"] == "documents/filename/folder%2Ffile%20name.txt"
        assert call["params"] == {"folder_name": "/team/docs", "folder_depth": 2, "end_user_id": "user-1"}
        assert doc.external_id == "doc-123"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_folder_get_document_by_filename_scoped():
    from morphik.async_ import AsyncFolder

    client, calls = await _make_async_client()
    try:
        folder = AsyncFolder(client, "docs", full_path="/projects/docs")
        await folder.get_document_by_filename("report.pdf")
        call = calls.pop()
        assert call["endpoint"] == "documents/filename/report.pdf"
        assert call["params"] == {"folder_name": "/projects/docs"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_user_scope_get_document_by_filename_scoped():
    from morphik.async_ import AsyncFolder

    client, calls = await _make_async_client()
    try:
        user = client.signin("user-99")
        await user.get_document_by_filename("note.txt")
        call = calls.pop()
        assert call["params"] == {"end_user_id": "user-99"}

        folder = AsyncFolder(client, "docs", full_path="/projects/docs")
        folder_user = folder.signin("user-42")
        await folder_user.get_document_by_filename("plan.md")
        call = calls.pop()
        assert call["params"] == {"folder_name": "/projects/docs", "end_user_id": "user-42"}
    finally:
        await client.close()


def test_folder_depth_passthrough_sync():
    client, calls = _make_sync_client()
    try:
        client.list_documents(folder_name="/a/b", folder_depth=-1)
        call = calls.pop()
        assert call["params"]["folder_depth"] == -1

        folder = Folder(client, "/team/alpha")
        folder.retrieve_chunks(query="q", folder_depth=2)
        retrieve_call = calls.pop()
        assert retrieve_call["data"]["folder_depth"] == 2
    finally:
        client.close()


def test_batch_get_documents_folder_name_alias_sync():
    client, calls = _make_sync_client()
    try:
        user = client.signin("user-7")
        with pytest.warns(DeprecationWarning):
            user.batch_get_documents(["doc-1"], folder_name="legacy")
        call = calls.pop()
        assert call["endpoint"] == "batch/documents"
        assert call["data"]["folder_name"] == ["legacy"]
        assert call["data"]["end_user_id"] == "user-7"
    finally:
        client.close()


def test_batch_get_chunks_folder_name_alias_sync():
    client, calls = _make_sync_client()
    try:
        folder = Folder(client, "alpha", full_path="/alpha")
        with pytest.warns(DeprecationWarning):
            folder.batch_get_chunks([{"document_id": "d1", "chunk_number": 1}], folder_name=["beta"])
        call = calls.pop()
        assert call["endpoint"] == "batch/chunks"
        assert call["data"]["folder_name"] == ["/alpha", "beta"]
    finally:
        client.close()


@pytest.mark.asyncio
async def test_batch_get_documents_folder_name_alias_async():
    client, calls = await _make_async_client()
    try:
        user = client.signin("user-7")
        with pytest.warns(DeprecationWarning):
            await user.batch_get_documents(["doc-1"], folder_name=["legacy"])
        call = calls.pop()
        assert call["endpoint"] == "batch/documents"
        assert call["data"]["folder_name"] == ["legacy"]
        assert call["data"]["end_user_id"] == "user-7"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_batch_get_chunks_folder_name_alias_async():
    from morphik.async_ import AsyncFolder

    client, calls = await _make_async_client()
    try:
        folder = AsyncFolder(client, "alpha", full_path="/alpha")
        with pytest.warns(DeprecationWarning):
            await folder.batch_get_chunks([{"document_id": "d1", "chunk_number": 1}], folder_name="beta")
        call = calls.pop()
        assert call["endpoint"] == "batch/chunks"
        assert call["data"]["folder_name"] == ["/alpha", "beta"]
    finally:
        await client.close()


# =============================================================================
# Folder Nesting Tests - Sync
# =============================================================================


def test_folder_full_path_property():
    """Test that Folder.full_path returns the canonical path."""
    client, _ = _make_sync_client()
    try:
        # When full_path is provided, it should be used
        folder = Folder(client, "leaf", full_path="/projects/alpha/leaf")
        assert folder.full_path == "/projects/alpha/leaf"
        assert folder.name == "leaf"

        # When full_path is not provided, name is used
        folder2 = Folder(client, "/simple/path")
        assert folder2.full_path == "/simple/path"

        # Backward compatibility - simple name only
        folder3 = Folder(client, "legacy_folder")
        assert folder3.full_path == "legacy_folder"
    finally:
        client.close()


def test_folder_hierarchy_properties():
    """Test folder depth, parent_id, child_count properties."""
    client, _ = _make_sync_client()
    try:
        folder = Folder(
            client,
            "leaf",
            folder_id="folder-123",
            full_path="/projects/alpha/leaf",
            parent_id="parent-456",
            depth=3,
            child_count=5,
            description="Test folder",
        )

        assert folder.id == "folder-123"
        assert folder.full_path == "/projects/alpha/leaf"
        assert folder.parent_id == "parent-456"
        assert folder.depth == 3
        assert folder.child_count == 5
        assert folder.description == "Test folder"
    finally:
        client.close()


def test_sync_move_folder_calls_move_endpoint():
    client, calls = _make_sync_client()
    try:
        original_request = client._request

        def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if endpoint == "folders/folder-123/move":
                return {
                    "id": "folder-123",
                    "name": "archived",
                    "full_path": "/projects/archived",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "moved folder",
                }
            return original_request(method, endpoint, data, files, params)

        client._request = mock_request

        moved = client.move_folder("folder-123", "/projects/archived")
        move_call = calls.pop()
        assert move_call["method"] == "POST"
        assert move_call["endpoint"] == "folders/folder-123/move"
        assert move_call["data"] == {"new_path": "/projects/archived"}
        assert moved.id == "folder-123"
        assert moved.full_path == "/projects/archived"
        assert moved.name == "archived"
    finally:
        client.close()


def test_sync_move_folder_encodes_special_chars_in_identifier():
    client, calls = _make_sync_client()
    try:
        original_request = client._request

        def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if endpoint == "folders/team/a%231/move":
                return {
                    "id": "folder-123",
                    "name": "a1",
                    "full_path": "/team/a1",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "moved folder",
                }
            return original_request(method, endpoint, data, files, params)

        client._request = mock_request

        moved = client.move_folder("/team/a#1", "/team/a1")
        move_call = calls.pop()
        assert move_call["method"] == "POST"
        assert move_call["endpoint"] == "folders/team/a%231/move"
        assert move_call["data"] == {"new_path": "/team/a1"}
        assert moved.full_path == "/team/a1"
    finally:
        client.close()


def test_sync_rename_folder_derives_target_path_from_existing_folder():
    client, calls = _make_sync_client()
    try:
        original_request = client._request

        def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if method == "GET" and endpoint == "folders/folder-123":
                return {
                    "id": "folder-123",
                    "name": "old-name",
                    "full_path": "/team/old-name",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "original",
                }
            if method == "POST" and endpoint == "folders/folder-123/move":
                return {
                    "id": "folder-123",
                    "name": "new-name",
                    "full_path": "/team/new-name",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "renamed",
                }
            return original_request(method, endpoint, data, files, params)

        client._request = mock_request

        renamed = client.rename_folder("folder-123", "new-name")

        get_call = calls[-2]
        move_call = calls[-1]
        assert get_call["method"] == "GET"
        assert get_call["endpoint"] == "folders/folder-123"
        assert move_call["method"] == "POST"
        assert move_call["endpoint"] == "folders/folder-123/move"
        assert move_call["data"] == {"new_path": "/team/new-name"}
        assert renamed.name == "new-name"
        assert renamed.full_path == "/team/new-name"
    finally:
        client.close()


def test_sync_rename_folder_encodes_identifier_for_get_and_move():
    client, calls = _make_sync_client()
    try:
        original_request = client._request

        def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if method == "GET" and endpoint == "folders/team/a%231":
                return {
                    "id": "folder#123",
                    "name": "a#1",
                    "full_path": "/team/a#1",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "original",
                }
            if method == "POST" and endpoint == "folders/folder%23123/move":
                return {
                    "id": "folder#123",
                    "name": "a-new",
                    "full_path": "/team/a-new",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "renamed",
                }
            return original_request(method, endpoint, data, files, params)

        client._request = mock_request

        renamed = client.rename_folder("/team/a#1", "a-new")

        get_call = calls[-2]
        move_call = calls[-1]
        assert get_call["method"] == "GET"
        assert get_call["endpoint"] == "folders/team/a%231"
        assert move_call["method"] == "POST"
        assert move_call["endpoint"] == "folders/folder%23123/move"
        assert move_call["data"] == {"new_path": "/team/a-new"}
        assert renamed.full_path == "/team/a-new"
    finally:
        client.close()


def test_sync_folder_move_and_rename_refresh_local_metadata():
    client, _ = _make_sync_client()
    try:
        folder = Folder(client, "docs", folder_id="folder-123", full_path="/team/docs")

        def mock_move(folder_id_or_name, new_path):
            if new_path == "/archive/docs":
                return Folder(
                    client,
                    "docs",
                    folder_id="folder-123",
                    full_path="/archive/docs",
                    parent_id="parent-arch",
                    depth=2,
                    child_count=0,
                    description="moved",
                )
            return Folder(
                client,
                "docs-v2",
                folder_id="folder-123",
                full_path="/archive/docs-v2",
                parent_id="parent-arch",
                depth=2,
                child_count=0,
                description="renamed",
            )

        client.move_folder = mock_move  # type: ignore[method-assign]
        client.rename_folder = lambda folder_id_or_name, new_name: mock_move(  # type: ignore[method-assign]
            folder_id_or_name, "/archive/docs-v2"
        )

        moved = folder.move("/archive/docs")
        assert moved.full_path == "/archive/docs"
        assert folder.full_path == "/archive/docs"
        assert folder.parent_id == "parent-arch"

        renamed = folder.rename("docs-v2")
        assert renamed.name == "docs-v2"
        assert renamed.full_path == "/archive/docs-v2"
        assert folder.name == "docs-v2"
        assert folder.full_path == "/archive/docs-v2"
    finally:
        client.close()


def test_folder_uses_full_path_for_operations():
    """Test that Folder operations use full_path instead of name."""
    client, calls = _make_sync_client()
    try:
        folder = Folder(client, "specs", full_path="/projects/alpha/specs")

        # Test list_documents uses full_path
        folder.list_documents()
        call = calls.pop()
        assert call["params"]["folder_name"] == "/projects/alpha/specs"

        # Test retrieve_chunks uses full_path
        folder.retrieve_chunks(query="test")
        call = calls.pop()
        assert call["data"]["folder_name"] == "/projects/alpha/specs"
    finally:
        client.close()


def test_folder_depth_on_retrieve_docs_sync():
    """Test folder_depth parameter on retrieve_docs."""
    client, calls = _make_sync_client()
    try:
        # Add mock for retrieve/docs
        original_request = client._request

        def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if "retrieve/docs" in endpoint:
                return []
            return original_request(method, endpoint, data, files, params)

        client._request = mock_request

        folder = Folder(client, "test", full_path="/nested/folder")
        folder.retrieve_docs(query="search", folder_depth=-1)

        retrieve_call = calls.pop()
        assert retrieve_call["data"]["folder_name"] == "/nested/folder"
        assert retrieve_call["data"]["folder_depth"] == -1
    finally:
        client.close()


def test_folder_depth_on_query_sync():
    """Test folder_depth parameter on query."""
    client, calls = _make_sync_client()
    try:
        # Add mock for query
        original_request = client._request

        def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if endpoint == "query":
                return {
                    "completion": "test",
                    "sources": [],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            return original_request(method, endpoint, data, files, params)

        client._request = mock_request

        folder = Folder(client, "docs", full_path="/team/docs")
        folder.query(query="What is X?", folder_depth=2)

        query_call = calls.pop()
        assert query_call["data"]["folder_name"] == "/team/docs"
        assert query_call["data"]["folder_depth"] == 2
    finally:
        client.close()


def test_folder_additional_folders_combines_with_full_path():
    """Test that additional_folders combines correctly with folder's full_path."""
    client, calls = _make_sync_client()
    try:
        folder = Folder(client, "main", full_path="/projects/main")

        folder.list_documents(additional_folders=["/shared", "/archive"])
        call = calls.pop()

        # Should prepend the folder's full_path to the additional folders
        assert call["params"]["folder_name"] == ["/projects/main", "/shared", "/archive"]
    finally:
        client.close()


def test_folder_depth_values():
    """Test various folder_depth values: 0, 1, -1, 2."""
    client, calls = _make_sync_client()
    try:
        folder = Folder(client, "root", full_path="/root")

        # folder_depth=0 (exact match)
        folder.list_documents(folder_depth=0)
        call = calls.pop()
        assert call["params"]["folder_depth"] == 0

        # folder_depth=1 (direct children)
        folder.list_documents(folder_depth=1)
        call = calls.pop()
        assert call["params"]["folder_depth"] == 1

        # folder_depth=-1 (all descendants)
        folder.list_documents(folder_depth=-1)
        call = calls.pop()
        assert call["params"]["folder_depth"] == -1

        # folder_depth=2 (up to grandchildren)
        folder.list_documents(folder_depth=2)
        call = calls.pop()
        assert call["params"]["folder_depth"] == 2
    finally:
        client.close()


def test_user_scope_folder_depth_sync():
    """Test folder_depth on UserScope operations."""
    client, calls = _make_sync_client()
    try:
        folder = Folder(client, "team", full_path="/team/alpha")
        user_scope = folder.signin("user-123")

        user_scope.list_documents(folder_depth=-1)
        call = calls.pop()
        assert call["params"]["folder_name"] == "/team/alpha"
        assert call["params"]["folder_depth"] == -1
        assert call["params"]["end_user_id"] == "user-123"

        user_scope.retrieve_chunks(query="test", folder_depth=1)
        call = calls.pop()
        assert call["data"]["folder_name"] == "/team/alpha"
        assert call["data"]["folder_depth"] == 1
    finally:
        client.close()


# =============================================================================
# Folder Nesting Tests - Async
# =============================================================================


@pytest.mark.asyncio
async def test_async_folder_full_path_property():
    """Test that AsyncFolder.full_path returns the canonical path."""
    from morphik.async_ import AsyncFolder

    client, _ = await _make_async_client()
    try:
        folder = AsyncFolder(client, "leaf", full_path="/projects/beta/leaf")
        assert folder.full_path == "/projects/beta/leaf"
        assert folder.name == "leaf"

        folder2 = AsyncFolder(client, "/direct/path")
        assert folder2.full_path == "/direct/path"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_folder_hierarchy_properties():
    """Test async folder depth, parent_id, child_count properties."""
    from morphik.async_ import AsyncFolder

    client, _ = await _make_async_client()
    try:
        folder = AsyncFolder(
            client,
            "child",
            folder_id="async-folder-789",
            full_path="/org/team/child",
            parent_id="parent-abc",
            depth=3,
            child_count=2,
            description="Async test folder",
        )

        assert folder.id == "async-folder-789"
        assert folder.full_path == "/org/team/child"
        assert folder.parent_id == "parent-abc"
        assert folder.depth == 3
        assert folder.child_count == 2
        assert folder.description == "Async test folder"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_move_folder_calls_move_endpoint():
    client, calls = await _make_async_client()
    try:
        original_request = client._request

        async def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if endpoint == "folders/folder-123/move":
                return {
                    "id": "folder-123",
                    "name": "archived",
                    "full_path": "/projects/archived",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "moved folder",
                }
            return await original_request(method, endpoint, data, files, params)

        client._request = mock_request

        moved = await client.move_folder("folder-123", "/projects/archived")
        move_call = calls.pop()
        assert move_call["method"] == "POST"
        assert move_call["endpoint"] == "folders/folder-123/move"
        assert move_call["data"] == {"new_path": "/projects/archived"}
        assert moved.id == "folder-123"
        assert moved.full_path == "/projects/archived"
        assert moved.name == "archived"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_move_folder_encodes_special_chars_in_identifier():
    client, calls = await _make_async_client()
    try:
        original_request = client._request

        async def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if endpoint == "folders/team/a%231/move":
                return {
                    "id": "folder-123",
                    "name": "a1",
                    "full_path": "/team/a1",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "moved folder",
                }
            return await original_request(method, endpoint, data, files, params)

        client._request = mock_request

        moved = await client.move_folder("/team/a#1", "/team/a1")
        move_call = calls.pop()
        assert move_call["method"] == "POST"
        assert move_call["endpoint"] == "folders/team/a%231/move"
        assert move_call["data"] == {"new_path": "/team/a1"}
        assert moved.full_path == "/team/a1"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_rename_folder_derives_target_path_from_existing_folder():
    client, calls = await _make_async_client()
    try:
        original_request = client._request

        async def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if method == "GET" and endpoint == "folders/folder-123":
                return {
                    "id": "folder-123",
                    "name": "old-name",
                    "full_path": "/team/old-name",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "original",
                }
            if method == "POST" and endpoint == "folders/folder-123/move":
                return {
                    "id": "folder-123",
                    "name": "new-name",
                    "full_path": "/team/new-name",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "renamed",
                }
            return await original_request(method, endpoint, data, files, params)

        client._request = mock_request

        renamed = await client.rename_folder("folder-123", "new-name")
        get_call = calls[-2]
        move_call = calls[-1]
        assert get_call["method"] == "GET"
        assert get_call["endpoint"] == "folders/folder-123"
        assert move_call["method"] == "POST"
        assert move_call["endpoint"] == "folders/folder-123/move"
        assert move_call["data"] == {"new_path": "/team/new-name"}
        assert renamed.name == "new-name"
        assert renamed.full_path == "/team/new-name"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_rename_folder_encodes_identifier_for_get_and_move():
    client, calls = await _make_async_client()
    try:
        original_request = client._request

        async def mock_request(method, endpoint, data=None, files=None, params=None):
            calls.append({"method": method, "endpoint": endpoint, "data": data, "params": params})
            if method == "GET" and endpoint == "folders/team/a%231":
                return {
                    "id": "folder#123",
                    "name": "a#1",
                    "full_path": "/team/a#1",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "original",
                }
            if method == "POST" and endpoint == "folders/folder%23123/move":
                return {
                    "id": "folder#123",
                    "name": "a-new",
                    "full_path": "/team/a-new",
                    "parent_id": "parent-1",
                    "depth": 2,
                    "child_count": 0,
                    "description": "renamed",
                }
            return await original_request(method, endpoint, data, files, params)

        client._request = mock_request

        renamed = await client.rename_folder("/team/a#1", "a-new")
        get_call = calls[-2]
        move_call = calls[-1]
        assert get_call["method"] == "GET"
        assert get_call["endpoint"] == "folders/team/a%231"
        assert move_call["method"] == "POST"
        assert move_call["endpoint"] == "folders/folder%23123/move"
        assert move_call["data"] == {"new_path": "/team/a-new"}
        assert renamed.full_path == "/team/a-new"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_folder_move_and_rename_refresh_local_metadata():
    from morphik.async_ import AsyncFolder

    client, _ = await _make_async_client()
    try:
        folder = AsyncFolder(client, "docs", folder_id="folder-123", full_path="/team/docs")

        async def mock_move(folder_id_or_name, new_path):
            if new_path == "/archive/docs":
                return AsyncFolder(
                    client,
                    "docs",
                    folder_id="folder-123",
                    full_path="/archive/docs",
                    parent_id="parent-arch",
                    depth=2,
                    child_count=0,
                    description="moved",
                )
            return AsyncFolder(
                client,
                "docs-v2",
                folder_id="folder-123",
                full_path="/archive/docs-v2",
                parent_id="parent-arch",
                depth=2,
                child_count=0,
                description="renamed",
            )

        client.move_folder = mock_move  # type: ignore[method-assign]

        async def mock_rename(folder_id_or_name, new_name):
            return await mock_move(folder_id_or_name, "/archive/docs-v2")

        client.rename_folder = mock_rename  # type: ignore[method-assign]

        moved = await folder.move("/archive/docs")
        assert moved.full_path == "/archive/docs"
        assert folder.full_path == "/archive/docs"
        assert folder.parent_id == "parent-arch"

        renamed = await folder.rename("docs-v2")
        assert renamed.name == "docs-v2"
        assert renamed.full_path == "/archive/docs-v2"
        assert folder.name == "docs-v2"
        assert folder.full_path == "/archive/docs-v2"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_folder_uses_full_path():
    """Test that AsyncFolder operations use full_path."""
    from morphik.async_ import AsyncFolder

    client, calls = await _make_async_client()
    try:
        folder = AsyncFolder(client, "data", full_path="/analytics/data")

        await folder.list_documents()
        call = calls.pop()
        assert call["params"]["folder_name"] == "/analytics/data"

        await folder.retrieve_chunks(query="metrics")
        call = calls.pop()
        assert call["data"]["folder_name"] == "/analytics/data"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_folder_depth_on_operations():
    """Test folder_depth parameter on async folder operations."""
    from morphik.async_ import AsyncFolder

    client, calls = await _make_async_client()
    try:
        folder = AsyncFolder(client, "reports", full_path="/finance/reports")

        # Test folder_depth on list_documents
        await folder.list_documents(folder_depth=-1)
        call = calls.pop()
        assert call["params"]["folder_depth"] == -1

        # Test folder_depth on retrieve_chunks
        await folder.retrieve_chunks(query="quarterly", folder_depth=2)
        call = calls.pop()
        assert call["data"]["folder_depth"] == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_folder_additional_folders():
    """Test additional_folders combines with full_path in async."""
    from morphik.async_ import AsyncFolder

    client, calls = await _make_async_client()
    try:
        folder = AsyncFolder(client, "primary", full_path="/workspace/primary")

        await folder.list_documents(additional_folders=["/backup", "/archive"])
        call = calls.pop()
        assert call["params"]["folder_name"] == ["/workspace/primary", "/backup", "/archive"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_user_scope_folder_depth():
    """Test folder_depth on AsyncUserScope operations."""
    from morphik.async_ import AsyncFolder

    client, calls = await _make_async_client()
    try:
        folder = AsyncFolder(client, "shared", full_path="/dept/shared")
        user_scope = folder.signin("async-user-456")

        await user_scope.list_documents(folder_depth=1)
        call = calls.pop()
        assert call["params"]["folder_name"] == "/dept/shared"
        assert call["params"]["folder_depth"] == 1
        assert call["params"]["end_user_id"] == "async-user-456"

        await user_scope.retrieve_chunks(query="data", folder_depth=-1)
        call = calls.pop()
        assert call["data"]["folder_name"] == "/dept/shared"
        assert call["data"]["folder_depth"] == -1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_folder_depth_all_values():
    """Test all folder_depth values on async folder."""
    from morphik.async_ import AsyncFolder

    client, calls = await _make_async_client()
    try:
        folder = AsyncFolder(client, "base", full_path="/base")

        for depth_val in [0, 1, 2, -1]:
            await folder.list_documents(folder_depth=depth_val)
            call = calls.pop()
            assert call["params"]["folder_depth"] == depth_val
    finally:
        await client.close()

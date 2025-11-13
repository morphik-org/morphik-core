import pytest
from morphik.async_ import AsyncMorphik
from morphik.sync import Folder, Morphik


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

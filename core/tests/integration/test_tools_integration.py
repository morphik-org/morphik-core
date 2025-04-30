import json

import pytest

# Import application services and tool functions
from core.api import document_service
from core.models.auth import AuthContext, EntityType

# Import helper for API calls
from core.tests.integration.test_api import create_auth_header
from core.tools.document_tools import list_documents, retrieve_chunks, retrieve_document, save_to_memory
from core.tools.graph_tools import ToolError as GraphToolError
from core.tools.graph_tools import knowledge_graph_query, list_graphs


@pytest.mark.asyncio
async def test_document_tools_retrieve_and_list_integration(test_app, client):
    # Use API to ingest a document
    headers = create_auth_header()
    content = "Integration test content for tool-level retrieval"
    response = await client.post(
        "/ingest/text",
        json={"content": content, "metadata": {"int_test": True}},
        headers=headers,
    )
    assert response.status_code == 200
    doc_id = response.json()["external_id"]

    # Prepare auth context matching the test token
    auth = AuthContext(
        entity_type=EntityType.DEVELOPER,
        entity_id="test_user",
        permissions={"read", "write", "admin"},
    )

    # Test retrieve_document (text)
    text = await retrieve_document(
        document_id=doc_id,
        document_service=document_service,
        auth=auth,
    )
    assert content in text

    # Test retrieve_document (metadata)
    meta_str = await retrieve_document(
        document_id=doc_id,
        format="metadata",
        document_service=document_service,
        auth=auth,
    )
    meta = json.loads(meta_str)
    assert meta["metadata"]["int_test"] is True

    # Test list_documents without filters
    list_str = await list_documents(
        document_service=document_service,
        auth=auth,
    )
    listed = json.loads(list_str)
    assert any(d["id"] == doc_id for d in listed.get("documents", []))

    # Test retrieve_chunks tool
    chunks = await retrieve_chunks(
        query="integration",
        filters={"external_id": doc_id},
        document_service=document_service,
        auth=auth,
        min_relevance=0.0,
        k=1,
    )
    # Expect header plus at least one chunk
    assert any(content in item.get("text", "") and item["type"] == "text" for item in chunks)


@pytest.mark.asyncio
async def test_save_to_memory_and_retrieve_integration(test_app):
    # Save memory via tool
    auth = AuthContext(
        entity_type=EntityType.DEVELOPER,
        entity_id="test_user",
        permissions={"read", "write", "admin"},
    )
    mem_content = "Remember this integration memory"
    res_str = await save_to_memory(
        content=mem_content,
        memory_type="session",
        tags=["int_mem"],
        document_service=document_service,
        auth=auth,
    )
    res = json.loads(res_str)
    assert res.get("success") is True
    mem_id = res.get("memory_id")

    # Retrieve the saved memory as a document
    retrieved = await retrieve_document(
        document_id=mem_id,
        document_service=document_service,
        auth=auth,
    )
    assert mem_content in retrieved


@pytest.mark.asyncio
async def test_list_graphs_tool_integration(test_app, client):
    # Ensure no graphs initially
    auth = AuthContext(
        entity_type=EntityType.DEVELOPER,
        entity_id="test_user",
        permissions={"read", "write", "admin"},
    )
    out = await list_graphs(document_service=document_service, auth=auth)
    data = json.loads(out)
    assert data.get("graphs") == []

    # Ingest two documents for graph creation
    headers = create_auth_header()
    resp1 = await client.post(
        "/ingest/text",
        json={"content": "Graph A content", "metadata": {}},
        headers=headers,
    )
    resp2 = await client.post(
        "/ingest/text",
        json={"content": "Graph B content", "metadata": {}},
        headers=headers,
    )
    assert resp1.status_code == 200 and resp2.status_code == 200
    id1 = resp1.json()["external_id"]
    id2 = resp2.json()["external_id"]

    # Create a graph via API endpoint
    graph_name = "integration_graph"
    create_resp = await client.post(
        "/graph/create",
        json={"name": graph_name, "documents": [id1, id2]},
        headers=headers,
    )
    assert create_resp.status_code == 200

    # Now list graphs via tool
    out2 = await list_graphs(document_service=document_service, auth=auth)
    data2 = json.loads(out2)
    assert any(g.get("name") == graph_name for g in data2.get("graphs", []))


@pytest.mark.asyncio
async def test_knowledge_graph_query_error_integration(test_app):
    # Ensure knowledge_graph_query raises when no graphs exist
    auth = AuthContext(
        entity_type=EntityType.DEVELOPER,
        entity_id="test_user",
        permissions={"read", "write", "admin"},
    )
    with pytest.raises(GraphToolError):
        await knowledge_graph_query(
            query_type="entity", start_nodes=["NonExistent"], document_service=document_service, auth=auth
        )

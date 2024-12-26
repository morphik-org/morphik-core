import asyncio
import json
import pytest
from pathlib import Path
import jwt
from datetime import datetime, timedelta, UTC
from typing import AsyncGenerator, Dict
from httpx import AsyncClient
from fastapi import FastAPI
from core.api import app, get_settings
import mimetypes
import logging

logger = logging.getLogger(__name__)


# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
JWT_SECRET = "your-secret-key-for-signing-tokens"
TEST_USER_ID = "test_user"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(event_loop):
    """Setup test environment and create test files"""
    # Create test data directory if it doesn't exist
    TEST_DATA_DIR.mkdir(exist_ok=True)

    # Create a test text file
    text_file = TEST_DATA_DIR / "test.txt"
    if not text_file.exists():
        text_file.write_text("This is a test document for DataBridge testing.")

    # Create a small test PDF if it doesn't exist
    pdf_file = TEST_DATA_DIR / "test.pdf"
    if not pdf_file.exists():
        pytest.skip("PDF file not available, skipping PDF tests")


def create_test_token(
    entity_type: str = "developer",
    entity_id: str = TEST_USER_ID,
    permissions: list = None,
    app_id: str = None,
    expired: bool = False
) -> str:
    """Create a test JWT token"""
    if not permissions:
        permissions = ["read", "write", "admin"]

    payload = {
        "type": entity_type,
        "entity_id": entity_id,
        "permissions": permissions,
        "exp": datetime.now(UTC) + timedelta(days=-1 if expired else 1)
    }

    if app_id:
        payload["app_id"] = app_id

    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def create_auth_header(
    entity_type: str = "developer",
    permissions: list = None,
    expired: bool = False
) -> Dict[str, str]:
    """Create authorization header with test token"""
    token = create_test_token(entity_type, permissions=permissions, expired=expired)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def test_app(event_loop: asyncio.AbstractEventLoop) -> FastAPI:
    """Create test FastAPI application"""
    # Configure test settings
    settings = get_settings()
    settings.JWT_SECRET_KEY = JWT_SECRET
    return app


@pytest.fixture
async def client(
    test_app: FastAPI,
    event_loop: asyncio.AbstractEventLoop
) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_ingest_text_document(
    client: AsyncClient,
    content: str = "Test content for document ingestion"
):
    """Test ingesting a text document"""
    headers = create_auth_header()

    response = await client.post(
        "/ingest/text",
        json={
            "content": content,
            "metadata": {"test": True, "type": "text"}
        },
        headers=headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "external_id" in data
    assert data["content_type"] == "text/plain"
    assert data["metadata"]["test"] is True

    return data["external_id"]


@pytest.mark.asyncio
async def test_ingest_pdf(client: AsyncClient):
    """Test ingesting a pdf"""
    headers = create_auth_header()
    pdf_path = TEST_DATA_DIR / "test.pdf"

    if not pdf_path.exists():
        pytest.skip("Test PDF file not available")

    content_type, _ = mimetypes.guess_type(pdf_path)
    if not content_type:
        content_type = "application/octet-stream"

    with open(pdf_path, "rb") as f:
        response = await client.post(
            "/ingest/file",
            files={"file": (pdf_path.name, f, content_type)},
            data={"metadata": json.dumps({"test": True, "type": "pdf"})},
            headers=headers
        )

    assert response.status_code == 200
    data = response.json()
    assert "external_id" in data
    assert data["content_type"] == "application/pdf"
    assert "storage_info" in data

    return data["external_id"]


@pytest.mark.asyncio
async def test_ingest_invalid_text_request(client: AsyncClient):
    """Test ingestion with invalid text request missing required content field"""
    headers = create_auth_header()

    response = await client.post(
        "/ingest/text",
        json={
            "wrong_field": "Test content"  # Missing required content field
        },
        headers=headers
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_ingest_invalid_file_request(client: AsyncClient):
    """Test ingestion with invalid file request missing file"""
    headers = create_auth_header()

    response = await client.post(
        "/ingest/file",
        files={},  # Missing file
        data={"metadata": "{}"},
        headers=headers
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_ingest_invalid_metadata(client: AsyncClient):
    """Test ingestion with invalid metadata JSON"""
    headers = create_auth_header()

    pdf_path = TEST_DATA_DIR / "test.pdf"
    if pdf_path.exists():
        files = {
            "file": ("test.pdf", open(pdf_path, "rb"), "application/pdf")
        }
        response = await client.post(
            "/ingest/file",
            files=files,
            data={"metadata": "invalid json"},
            headers=headers
        )
        assert response.status_code == 400  # Bad request


@pytest.mark.asyncio
async def test_ingest_oversized_content(client: AsyncClient):
    """Test ingestion with oversized content"""
    headers = create_auth_header()

    large_content = "x" * (10 * 1024 * 1024)  # 10MB
    response = await client.post(
        "/ingest/text",
        json={
            "content": large_content,
            "metadata": {}
        },
        headers=headers
    )
    assert response.status_code == 400  # Bad request


@pytest.mark.asyncio
async def test_auth_missing_header(client: AsyncClient):
    """Test authentication with missing auth header"""
    response = await client.post("/ingest/text")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_invalid_token(client: AsyncClient):
    """Test authentication with invalid token"""
    headers = {"Authorization": "Bearer invalid_token"}
    response = await client.post("/ingest/file", headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_expired_token(client: AsyncClient):
    """Test authentication with expired token"""
    headers = create_auth_header(expired=True)
    response = await client.post("/ingest/text", headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_insufficient_permissions(client: AsyncClient):
    """Test authentication with insufficient permissions"""
    headers = create_auth_header(permissions=["read"])
    response = await client.post(
        "/ingest/text",
        json={
            "content": "Test content",
            "metadata": {}
        },
        headers=headers
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_documents(client: AsyncClient):
    """Test listing documents"""
    # First ingest some documents
    doc_id1 = await test_ingest_text_document(client)
    doc_id2 = await test_ingest_text_document(client)

    headers = create_auth_header()
    response = await client.get("/documents", headers=headers)

    assert response.status_code == 200
    docs = response.json()
    assert len(docs) >= 2
    doc_ids = [doc["external_id"] for doc in docs]
    assert doc_id1 in doc_ids
    assert doc_id2 in doc_ids


@pytest.mark.asyncio
async def test_get_document(client: AsyncClient):
    """Test getting a specific document"""
    # First ingest a document
    doc_id = await test_ingest_text_document(client)

    headers = create_auth_header()
    response = await client.get(f"/documents/{doc_id}", headers=headers)

    assert response.status_code == 200
    doc = response.json()
    assert doc["external_id"] == doc_id
    assert "metadata" in doc
    assert "content_type" in doc


@pytest.mark.asyncio
async def test_invalid_document_id(client: AsyncClient):
    """Test error handling for invalid document ID"""
    headers = create_auth_header()
    response = await client.get("/documents/invalid_id", headers=headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_retrieve_chunks(client: AsyncClient):
    """Test retrieving document chunks"""
    # First ingest a document to search
    doc_id = await test_ingest_text_document(
        client,
        content="The quick brown fox jumps over the lazy dog"
    )

    headers = create_auth_header()
    # Sleep to allow time for document to be indexed
    await asyncio.sleep(1)

    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "jumping fox",
            "k": 1,
            "min_score": 0.0
        },
        headers=headers
    )

    assert response.status_code == 200
    results = list(response.json())
    assert len(results) == 1
    assert results[0]["score"] > 0.5
    assert results[0]["document_id"] == doc_id


@pytest.mark.asyncio
async def test_retrieve_docs(client: AsyncClient):
    """Test retrieving full documents"""
    # First ingest a document to search
    content = (
        "Headaches can significantly impact daily life and wellbeing. "
        "Common triggers include stress, dehydration, and poor sleep habits. "
        "While over-the-counter pain relievers may provide temporary relief, "
        "it's important to identify and address the root causes. "
        "Maintaining good health through proper nutrition, regular exercise, "
        "and stress management can help prevent chronic headaches."
    )
    doc_id = await test_ingest_text_document(client, content=content)

    headers = create_auth_header()
    response = await client.post(
        "/retrieve/docs",
        json={
            "query": "Headaches, dehydration",
            "filters": {"test": True}
        },
        headers=headers
    )

    assert response.status_code == 200
    results = list(response.json())
    assert len(results) > 0
    assert results[0]["document_id"] == doc_id
    assert "score" in results[0]
    assert "metadata" in results[0]


@pytest.mark.asyncio
async def test_query_completion(client: AsyncClient):
    """Test generating completions from context"""
    # First ingest a document to use as context
    content = (
        "The benefits of exercise are numerous. Regular physical activity "
        "can improve cardiovascular health, strengthen muscles, enhance mental "
        "wellbeing, and help maintain a healthy weight. Studies show that "
        "even moderate exercise like walking can significantly reduce the risk "
        "of various health conditions."
    )
    await test_ingest_text_document(client, content=content)

    headers = create_auth_header()
    response = await client.post(
        "/query",
        json={
            "query": "What are the main benefits of exercise?",
            "k": 2,
            "temperature": 0.7,
            "max_tokens": 100
        },
        headers=headers
    )

    assert response.status_code == 200
    result = response.json()
    assert "completion" in result
    assert "usage" in result
    assert len(result["completion"]) > 0


@pytest.mark.asyncio
async def test_invalid_retrieve_params(client: AsyncClient):
    """Test error handling for invalid retrieve parameters"""
    headers = create_auth_header()

    # Test empty query
    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "",  # Empty query
            "k": 1
        },
        headers=headers
    )
    assert response.status_code == 422

    # Test invalid k
    response = await client.post(
        "/retrieve/docs",
        json={
            "query": "test",
            "k": -1  # Invalid k
        },
        headers=headers
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_completion_params(client: AsyncClient):
    """Test error handling for invalid completion parameters"""
    headers = create_auth_header()

    # Test empty query
    response = await client.post(
        "/query",
        json={
            "query": "",  # Empty query
            "temperature": 2.0  # Invalid temperature
        },
        headers=headers
    )
    assert response.status_code == 422

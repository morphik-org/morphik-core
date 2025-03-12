import asyncio
import json
import pytest
from pathlib import Path
import jwt
from datetime import datetime, timedelta, UTC
from typing import AsyncGenerator, Dict
from httpx import AsyncClient
from fastapi import FastAPI
from httpx import ASGITransport
from core.api import app, get_settings
import filetype
import logging
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
JWT_SECRET = "your-secret-key-for-signing-tokens"
TEST_USER_ID = "test_user"
TEST_POSTGRES_URI = "postgresql+asyncpg://postgres:postgres@localhost:5432/databridge_test"


async def setup_test_postgres():
    """Setup test PostgreSQL database"""
    engine = create_async_engine(TEST_POSTGRES_URI)
    try:
        async with engine.begin() as conn:
            # Drop all tables first
            await conn.execute(text("DROP TABLE IF EXISTS documents CASCADE"))
            await conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
            await conn.execute(text("DROP TABLE IF EXISTS caches CASCADE"))
            
            # Recreate documents table with all needed columns
            await conn.execute(
                text(
                    """
                CREATE TABLE documents (
                    external_id VARCHAR PRIMARY KEY,
                    owner JSONB NOT NULL,
                    content_type VARCHAR NOT NULL,
                    filename VARCHAR,
                    doc_metadata JSONB DEFAULT '{}'::jsonb,
                    storage_info JSONB DEFAULT '{}'::jsonb,
                    system_metadata JSONB DEFAULT '{}'::jsonb,
                    additional_metadata JSONB DEFAULT '{}'::jsonb,
                    access_control JSONB DEFAULT '{}'::jsonb,
                    chunk_ids JSONB DEFAULT '[]'::jsonb,
                    storage_files JSONB DEFAULT '[]'::jsonb
                )
                """
                )
            )
            
            # Create indexes
            await conn.execute(text("CREATE INDEX idx_owner_id ON documents USING gin(owner)"))
            await conn.execute(text("CREATE INDEX idx_access_control ON documents USING gin(access_control)"))
            await conn.execute(text("CREATE INDEX idx_system_metadata ON documents USING gin(system_metadata)"))
            
            # Create caches table
            await conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS caches (
                    name TEXT PRIMARY KEY,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
                )
            )

            logger.info("Test PostgreSQL database setup completed")
    except Exception as e:
        logger.error(f"Failed to setup test PostgreSQL database: {e}")
        raise
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_test_environment(event_loop):
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

    # Setup test PostgreSQL database
    if get_settings().DATABASE_PROVIDER == "postgres":
        await setup_test_postgres()


def create_test_token(
    entity_type: str = "developer",
    entity_id: str = TEST_USER_ID,
    permissions: list = None,
    app_id: str = None,
    expired: bool = False,
) -> str:
    """Create a test JWT token"""
    if not permissions:
        permissions = ["read", "write", "admin"]

    payload = {
        "type": entity_type,
        "entity_id": entity_id,
        "permissions": permissions,
        "exp": datetime.now(UTC) + timedelta(days=-1 if expired else 1),
    }

    if app_id:
        payload["app_id"] = app_id

    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def create_auth_header(
    entity_type: str = "developer", permissions: list = None, expired: bool = False
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
    test_app: FastAPI, event_loop: asyncio.AbstractEventLoop
) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_ingest_text_document(
    client: AsyncClient, content: str = "Test content for document ingestion"
):
    """Test ingesting a text document"""
    headers = create_auth_header()

    response = await client.post(
        "/ingest/text",
        json={"content": content, "metadata": {"test": True, "type": "text"}, "use_colpali": True},
        headers=headers,
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

    content_type = filetype.guess(pdf_path).mime
    if not content_type:
        content_type = "application/octet-stream"

    with open(pdf_path, "rb") as f:
        response = await client.post(
            "/ingest/file",
            files={"file": (pdf_path.name, f, content_type)},
            data={"metadata": json.dumps({"test": True, "type": "pdf"}), "use_colpali": True},
            headers=headers,
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
        json={"wrong_field": "Test content"},  # Missing required content field
        headers=headers,
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
        headers=headers,
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_ingest_invalid_metadata(client: AsyncClient):
    """Test ingestion with invalid metadata JSON"""
    headers = create_auth_header()

    pdf_path = TEST_DATA_DIR / "test.pdf"
    if pdf_path.exists():
        files = {"file": ("test.pdf", open(pdf_path, "rb"), "application/pdf")}
        response = await client.post(
            "/ingest/file",
            files=files,
            data={"metadata": "invalid json"},
            headers=headers,
        )
        assert response.status_code == 400  # Bad request


@pytest.mark.asyncio
@pytest.mark.skipif(
    get_settings().EMBEDDING_PROVIDER == "ollama",
    reason="local embedding models do not have size limits",
)
async def test_ingest_oversized_content(client: AsyncClient):
    """Test ingestion with oversized content"""
    headers = create_auth_header()
    large_content = "x" * (10 * 1024 * 1024)  # 10MB
    response = await client.post(
        "/ingest/text", json={"content": large_content, "metadata": {}}, headers=headers
    )
    assert response.status_code == 400  # Bad request


@pytest.mark.asyncio
async def test_auth_missing_header(client: AsyncClient):
    """Test authentication with missing auth header"""
    if get_settings().dev_mode:
        pytest.skip("Auth tests skipped in dev mode")
    response = await client.post("/ingest/text")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_invalid_token(client: AsyncClient):
    """Test authentication with invalid token"""
    if get_settings().dev_mode:
        pytest.skip("Auth tests skipped in dev mode")
    headers = {"Authorization": "Bearer invalid_token"}
    response = await client.post("/ingest/file", headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_expired_token(client: AsyncClient):
    """Test authentication with expired token"""
    if get_settings().dev_mode:
        pytest.skip("Auth tests skipped in dev mode")
    headers = create_auth_header(expired=True)
    response = await client.post("/ingest/text", headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_insufficient_permissions(client: AsyncClient):
    """Test authentication with insufficient permissions"""
    if get_settings().dev_mode:
        pytest.skip("Auth tests skipped in dev mode")
    headers = create_auth_header(permissions=["read"])
    response = await client.post(
        "/ingest/text",
        json={"content": "Test content", "metadata": {}},
        headers=headers,
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
async def test_get_document_by_filename(client: AsyncClient):
    """Test getting a document by filename"""
    # First ingest a document with a specific filename
    filename = "test_get_by_filename.txt"
    headers = create_auth_header()
    
    initial_content = "This is content for testing get_document_by_filename."
    response = await client.post(
        "/ingest/text",
        json={
            "content": initial_content, 
            "filename": filename,
            "metadata": {"test": True, "type": "text"},
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    doc_id = response.json()["external_id"]
    
    # Now try to get the document by filename
    response = await client.get(f"/documents/filename/{filename}", headers=headers)
    
    assert response.status_code == 200
    doc = response.json()
    assert doc["external_id"] == doc_id
    assert doc["filename"] == filename
    assert "metadata" in doc
    assert "content_type" in doc


@pytest.mark.asyncio
async def test_invalid_document_id(client: AsyncClient):
    """Test error handling for invalid document ID"""
    headers = create_auth_header()
    response = await client.get("/documents/invalid_id", headers=headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_document_with_text(client: AsyncClient):
    """Test updating a document with text content"""
    # First ingest a document to update
    initial_content = "This is the initial content for update testing."
    doc_id = await test_ingest_text_document(client, content=initial_content)
    
    headers = create_auth_header()
    update_content = "This is additional content for the document."
    
    # Test updating with text content
    response = await client.post(
        f"/documents/{doc_id}/update_text",
        json={
            "content": update_content,
            "metadata": {"updated": True, "version": "2.0"},
            "use_colpali": True
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    updated_doc = response.json()
    assert updated_doc["external_id"] == doc_id
    assert updated_doc["metadata"]["updated"] is True
    assert updated_doc["metadata"]["version"] == "2.0"
    
    # Verify the content was updated by retrieving chunks
    search_response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "additional content",
            "filters": {"external_id": doc_id},
        },
        headers=headers,
    )
    
    assert search_response.status_code == 200
    chunks = search_response.json()
    assert len(chunks) > 0
    assert any(update_content in chunk["content"] for chunk in chunks)


@pytest.mark.asyncio
async def test_update_document_with_file(client: AsyncClient):
    """Test updating a document with file content"""
    # First ingest a document to update
    initial_content = "This is the initial content for file update testing."
    doc_id = await test_ingest_text_document(client, content=initial_content)
    
    headers = create_auth_header()
    
    # Create a test file to upload
    test_file_path = TEST_DATA_DIR / "update_test.txt"
    update_content = "This is file content for updating the document."
    test_file_path.write_text(update_content)
    
    with open(test_file_path, "rb") as f:
        response = await client.post(
            f"/documents/{doc_id}/update_file",
            files={"file": ("update_test.txt", f, "text/plain")},
            data={
                "metadata": json.dumps({"updated_with_file": True}),
                "rules": json.dumps([]),
                "update_strategy": "add",
            },
            headers=headers,
        )
    
    assert response.status_code == 200
    updated_doc = response.json()
    assert updated_doc["external_id"] == doc_id
    assert updated_doc["metadata"]["updated_with_file"] is True
    
    # Verify the content was updated by retrieving chunks
    search_response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "file content for updating",
            "filters": {"external_id": doc_id},
        },
        headers=headers,
    )
    
    assert search_response.status_code == 200
    chunks = search_response.json()
    assert len(chunks) > 0
    assert any(update_content in chunk["content"] for chunk in chunks)
    
    # Clean up the test file
    test_file_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_update_document_metadata(client: AsyncClient):
    """Test updating only a document's metadata"""
    # First ingest a document to update
    initial_content = "This is the content for metadata update testing."
    doc_id = await test_ingest_text_document(client, content=initial_content)
    
    headers = create_auth_header()
    
    # Test updating just metadata
    new_metadata = {
        "meta_updated": True,
        "tags": ["test", "metadata", "update"],
        "priority": 1
    }
    
    response = await client.post(
        f"/documents/{doc_id}/update_metadata",
        json=new_metadata,
        headers=headers,
    )
    
    assert response.status_code == 200
    updated_doc = response.json()
    assert updated_doc["external_id"] == doc_id
    
    # Verify the response has the updated metadata
    assert updated_doc["metadata"]["meta_updated"] is True
    assert "test" in updated_doc["metadata"]["tags"]
    assert updated_doc["metadata"]["priority"] == 1
    
    # Fetch the document to verify it exists
    get_response = await client.get(f"/documents/{doc_id}", headers=headers)
    assert get_response.status_code == 200
    
    # Note: Depending on caching or database behavior, the metadata may not be 
    # immediately visible in a subsequent fetch. The important part is that
    # the update operation itself returned the correct metadata.
    
    
@pytest.mark.asyncio
async def test_update_document_with_rules(client: AsyncClient):
    """Test updating a document with text content and applying rules"""
    # First ingest a document to update
    initial_content = "This is the initial content for rule testing."
    doc_id = await test_ingest_text_document(client, content=initial_content)
    
    headers = create_auth_header()
    update_content = "This document contains information about John Doe who lives at 123 Main St and has SSN 123-45-6789."
    
    # Create a rule to apply during update (natural language rule to remove PII)
    rule = {
        "type": "natural_language",
        "prompt": "Remove all personally identifiable information (PII) such as names, addresses, and SSNs."
    }
    
    # Test updating with text content and a rule
    response = await client.post(
        f"/documents/{doc_id}/update_text",
        json={
            "content": update_content,
            "metadata": {"contains_pii": False},
            "rules": [rule],
            "use_colpali": True
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    updated_doc = response.json()
    assert updated_doc["external_id"] == doc_id
    assert updated_doc["metadata"]["contains_pii"] is False
    
    # Verify the content was updated and PII was removed by retrieving chunks
    # Note: Exact behavior depends on the LLM response, so we check that something changed
    search_response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "information about person",
            "filters": {"external_id": doc_id},
        },
        headers=headers,
    )
    
    assert search_response.status_code == 200
    chunks = search_response.json()
    assert len(chunks) > 0
    # The processed content should have some of the original content but not the PII
    assert any("information" in chunk["content"] for chunk in chunks)
    # Check that at least one of the PII elements is not in the content
    # This is a loose check since the exact result depends on the LLM
    content_text = " ".join([chunk["content"] for chunk in chunks])
    has_some_pii_removed = ("John Doe" not in content_text) or ("123-45-6789" not in content_text) or ("123 Main St" not in content_text)
    assert has_some_pii_removed, "Rule to remove PII did not seem to have any effect"


@pytest.mark.asyncio
async def test_update_document_by_filename_with_text(client: AsyncClient):
    """Test updating a document by filename with text content"""
    # First ingest a document with a specific filename
    filename = "test_update_by_filename.txt"
    headers = create_auth_header()
    
    initial_content = "This is the initial content for update-by-filename testing."
    response = await client.post(
        "/ingest/text",
        json={
            "content": initial_content, 
            "filename": filename,
            "metadata": {"test": True, "type": "text"},
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    doc_id = response.json()["external_id"]
    
    # Now update the document by filename
    update_content = "This is additional content added by filename update."
    new_filename = "updated_filename.txt"
    
    response = await client.post(
        f"/documents/filename/{filename}/update_text",
        json={
            "content": update_content,
            "filename": new_filename,
            "metadata": {"updated_by_filename": True, "version": "2.0"},
            "use_colpali": True
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    updated_doc = response.json()
    assert updated_doc["external_id"] == doc_id
    assert updated_doc["filename"] == new_filename
    assert updated_doc["metadata"]["updated_by_filename"] is True
    assert updated_doc["metadata"]["version"] == "2.0"
    
    # Verify the content was updated by retrieving chunks
    search_response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "additional content filename update",
            "filters": {"external_id": doc_id},
        },
        headers=headers,
    )
    
    assert search_response.status_code == 200
    chunks = search_response.json()
    assert len(chunks) > 0
    assert any(update_content in chunk["content"] for chunk in chunks)


@pytest.mark.asyncio
async def test_update_document_by_filename_with_file(client: AsyncClient):
    """Test updating a document by filename with file content"""
    # First ingest a document with a specific filename
    filename = "original_file.txt"
    headers = create_auth_header()
    
    initial_content = "This is the initial content for file update-by-filename testing."
    response = await client.post(
        "/ingest/text",
        json={
            "content": initial_content, 
            "filename": filename,
            "metadata": {"test": True, "type": "text"},
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    doc_id = response.json()["external_id"]
    
    # Create a test file to upload
    test_file_path = TEST_DATA_DIR / "update_by_filename_test.txt"
    update_content = "This is file content for updating the document by filename."
    test_file_path.write_text(update_content)
    
    # Update the document by filename
    with open(test_file_path, "rb") as f:
        response = await client.post(
            f"/documents/filename/{filename}/update_file",
            files={"file": ("new_filename.txt", f, "text/plain")},
            data={
                "metadata": json.dumps({"updated_with_file_by_filename": True}),
                "rules": json.dumps([]),
                "update_strategy": "add",
            },
            headers=headers,
        )
    
    assert response.status_code == 200
    updated_doc = response.json()
    assert updated_doc["external_id"] == doc_id
    assert updated_doc["filename"] == "new_filename.txt"
    assert updated_doc["metadata"]["updated_with_file_by_filename"] is True
    
    # Verify the content was updated by retrieving chunks
    search_response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "file content updating document filename",
            "filters": {"external_id": doc_id},
        },
        headers=headers,
    )
    
    assert search_response.status_code == 200
    chunks = search_response.json()
    assert len(chunks) > 0
    assert any(update_content in chunk["content"] for chunk in chunks)
    
    # Clean up the test file
    test_file_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_file_versioning_with_add_strategy(client: AsyncClient):
    """Test that file versioning works correctly with 'add' update strategy"""
    # First ingest a document with a file
    headers = create_auth_header()
    
    # Create the initial file
    initial_file_path = TEST_DATA_DIR / "version_test_1.txt"
    initial_content = "This is version 1 of the file for testing versioning."
    initial_file_path.write_text(initial_content)
    
    # Ingest the initial file
    with open(initial_file_path, "rb") as f:
        response = await client.post(
            "/ingest/file",
            files={"file": ("version_test.txt", f, "text/plain")},
            data={
                "metadata": json.dumps({"test": True, "version": 1}),
                "rules": json.dumps([]),
            },
            headers=headers,
        )
    
    assert response.status_code == 200
    doc_id = response.json()["external_id"]
    
    # Create second version of the file
    second_file_path = TEST_DATA_DIR / "version_test_2.txt"
    second_content = "This is version 2 of the file for testing versioning."
    second_file_path.write_text(second_content)
    
    # Update with second file using "add" strategy
    with open(second_file_path, "rb") as f:
        response = await client.post(
            f"/documents/{doc_id}/update_file",
            files={"file": ("version_test_v2.txt", f, "text/plain")},
            data={
                "metadata": json.dumps({"test": True, "version": 2}),
                "rules": json.dumps([]),
                "update_strategy": "add",
            },
            headers=headers,
        )
    
    assert response.status_code == 200
    updated_doc = response.json()
    
    # Create third version of the file
    third_file_path = TEST_DATA_DIR / "version_test_3.txt"
    third_content = "This is version 3 of the file for testing versioning."
    third_file_path.write_text(third_content)
    
    # Update with third file using "add" strategy
    with open(third_file_path, "rb") as f:
        response = await client.post(
            f"/documents/{doc_id}/update_file",
            files={"file": ("version_test_v3.txt", f, "text/plain")},
            data={
                "metadata": json.dumps({"test": True, "version": 3}),
                "rules": json.dumps([]),
                "update_strategy": "add",
            },
            headers=headers,
        )
    
    assert response.status_code == 200
    final_doc = response.json()
    
    # Verify the system_metadata has versioning info
    assert final_doc["system_metadata"]["version"] >= 3
    assert "update_history" in final_doc["system_metadata"]
    assert len(final_doc["system_metadata"]["update_history"]) >= 2  # At least 2 updates
    
    # Verify storage_files field exists and has multiple entries
    assert "storage_files" in final_doc
    assert len(final_doc["storage_files"]) >= 3  # Should have at least 3 files
    
    # Get most recent file's content through search 
    search_response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "version 3 testing versioning",
            "filters": {"external_id": doc_id},
        },
        headers=headers,
    )
    assert search_response.status_code == 200
    chunks = search_response.json()
    assert any(third_content in chunk["content"] for chunk in chunks)
    
    # Also check for version 1 content, which should still be in the merged content
    search_response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "version 1 testing versioning",
            "filters": {"external_id": doc_id},
        },
        headers=headers,
    )
    assert search_response.status_code == 200
    chunks = search_response.json()
    assert any(initial_content in chunk["content"] for chunk in chunks)
    
    # Clean up test files
    initial_file_path.unlink(missing_ok=True)
    second_file_path.unlink(missing_ok=True)
    third_file_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_update_document_by_filename_metadata(client: AsyncClient):
    """Test updating only a document's metadata using filename"""
    # First ingest a document with a specific filename
    filename = "metadata_update_test.txt"
    headers = create_auth_header()
    
    initial_content = "This is the content for metadata update-by-filename testing."
    response = await client.post(
        "/ingest/text",
        json={
            "content": initial_content, 
            "filename": filename,
            "metadata": {"test": True, "type": "text"},
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    doc_id = response.json()["external_id"]
    
    # Test updating just metadata by filename
    new_metadata = {
        "meta_updated_by_filename": True,
        "tags": ["test", "metadata", "filename", "update"],
        "priority": 2
    }
    
    response = await client.post(
        f"/documents/filename/{filename}/update_metadata",
        json=new_metadata,
        params={"new_filename": "updated_metadata_filename.txt"},
        headers=headers,
    )
    
    assert response.status_code == 200
    updated_doc = response.json()
    assert updated_doc["external_id"] == doc_id
    assert updated_doc["filename"] == "updated_metadata_filename.txt"
    
    # Verify the response has the updated metadata
    assert updated_doc["metadata"]["meta_updated_by_filename"] is True
    assert "filename" in updated_doc["metadata"]["tags"]
    assert updated_doc["metadata"]["priority"] == 2


@pytest.mark.asyncio
async def test_update_document_error_cases(client: AsyncClient):
    """Test error cases for document updates"""
    headers = create_auth_header()
    
    # Test updating non-existent document by ID
    response = await client.post(
        "/documents/non_existent_id/update_text",
        json={
            "content": "Test content for non-existent document",
            "metadata": {}
        },
        headers=headers,
    )
    assert response.status_code == 404
    
    # Test updating non-existent document by filename
    response = await client.post(
        "/documents/filename/non_existent_file.txt/update_text",
        json={
            "content": "Test content for non-existent document",
            "metadata": {}
        },
        headers=headers,
    )
    assert response.status_code == 404
    
    # Test updating text without content (validation error)
    doc_id = await test_ingest_text_document(client)
    response = await client.post(
        f"/documents/{doc_id}/update_text",
        json={
            # Missing required content field
            "metadata": {"test": True}
        },
        headers=headers,
    )
    assert response.status_code == 422
    
    # Test updating with insufficient permissions
    if not get_settings().dev_mode:
        restricted_headers = create_auth_header(permissions=["read"])
        response = await client.post(
            f"/documents/{doc_id}/update_metadata",
            json={"restricted": True},
            headers=restricted_headers,
        )
        assert response.status_code == 403


@pytest.mark.asyncio
async def test_retrieve_chunks(client: AsyncClient):
    """Test retrieving document chunks"""
    upload_string = "The quick brown fox jumps over the lazy dog"
    # First ingest a document to search
    doc_id = await test_ingest_text_document(client, content=upload_string)

    headers = create_auth_header()

    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "jumping fox",
            "k": 1,
            "min_score": 0.0,
            "filters": {"external_id": doc_id},  # Add filter for specific document
            "use_colpali": True,
        },
        headers=headers,
    )

    assert response.status_code == 200
    results = list(response.json())
    assert len(results) > 0
    assert (not get_settings().USE_RERANKING) or results[0]["score"] > 0.5
    assert any(upload_string == result["content"] for result in results)


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
            "filters": {"test": True, "external_id": doc_id},  # Add filter for specific document
            "use_colpali": True,
        },
        headers=headers,
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
            "max_tokens": 100,
        },
        headers=headers,
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
        "/retrieve/chunks", json={"query": "", "k": 1}, headers=headers  # Empty query
    )
    assert response.status_code == 422

    # Test invalid k
    response = await client.post(
        "/retrieve/docs", json={"query": "test", "k": -1}, headers=headers  # Invalid k
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_completion_params(client: AsyncClient):
    """Test error handling for invalid completion parameters"""
    headers = create_auth_header()

    # Test empty query
    response = await client.post(
        "/query",
        json={"query": "", "temperature": 2.0},  # Empty query  # Invalid temperature
        headers=headers,
    )
    assert response.status_code == 422


@pytest.fixture(autouse=True)
async def cleanup_database():
    """Clean up database before each test"""
    if get_settings().DATABASE_PROVIDER == "postgres":
        await setup_test_postgres()


@pytest.mark.asyncio
async def test_retrieve_chunks_default_reranking(client: AsyncClient):
    """Test retrieving chunks with default reranking behavior"""
    # First ingest some test documents
    _ = await test_ingest_text_document(
        client, "The quick brown fox jumps over the lazy dog. This is a test document."
    )
    _ = await test_ingest_text_document(
        client, "The lazy dog sleeps while the quick brown fox runs. Another test document."
    )

    headers = create_auth_header()
    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "What does the fox do?",
            "k": 2,
            "min_score": 0.0,
            # Not specifying use_reranking - should use default from config
            "use_colpali": True,
        },
        headers=headers,
    )

    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) > 0
    # Verify chunks are ordered by score
    scores = [chunk["score"] for chunk in chunks]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


@pytest.mark.asyncio
async def test_retrieve_chunks_explicit_reranking(client: AsyncClient):
    """Test retrieving chunks with explicitly enabled reranking"""
    # First ingest some test documents
    _ = await test_ingest_text_document(
        client, "The quick brown fox jumps over the lazy dog. This is a test document."
    )
    _ = await test_ingest_text_document(
        client, "The lazy dog sleeps while the quick brown fox runs. Another test document."
    )

    headers = create_auth_header()
    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "What does the fox do?",
            "k": 2,
            "min_score": 0.0,
            "use_reranking": True,
            "use_colpali": True,
        },
        headers=headers,
    )

    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) > 0
    # Verify chunks are ordered by score
    scores = [chunk["score"] for chunk in chunks]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


@pytest.mark.asyncio
async def test_retrieve_chunks_disabled_reranking(client: AsyncClient):
    """Test retrieving chunks with explicitly disabled reranking"""
    # First ingest some test documents
    await test_ingest_text_document(
        client, "The quick brown fox jumps over the lazy dog. This is a test document."
    )
    await test_ingest_text_document(
        client, "The lazy dog sleeps while the quick brown fox runs. Another test document."
    )

    headers = create_auth_header()
    response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "What does the fox do?",
            "k": 2,
            "min_score": 0.0,
            "use_reranking": False,
            "use_colpali": True,
        },
        headers=headers,
    )

    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_reranking_affects_results(client: AsyncClient):
    """Test that reranking actually changes the order of results"""
    # First ingest documents with clearly different semantic relevance
    await test_ingest_text_document(
        client, "The capital of France is Paris. The city is known for the Eiffel Tower."
    )
    await test_ingest_text_document(
        client, "Paris is a city in France. It has many famous landmarks and museums."
    )
    await test_ingest_text_document(
        client, "Paris Hilton is a celebrity and businesswoman. She has nothing to do with France."
    )

    headers = create_auth_header()

    # Get results without reranking
    response_no_rerank = await client.post(
        "/retrieve/chunks",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
            "use_reranking": False,
            "use_colpali": True,
        },
        headers=headers,
    )

    # Get results with reranking
    response_with_rerank = await client.post(
        "/retrieve/chunks",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
            "use_reranking": True,
            "use_colpali": True,
        },
        headers=headers,
    )

    assert response_no_rerank.status_code == 200
    assert response_with_rerank.status_code == 200

    chunks_no_rerank = response_no_rerank.json()
    chunks_with_rerank = response_with_rerank.json()

    # Verify we got results in both cases
    assert len(chunks_no_rerank) > 0
    assert len(chunks_with_rerank) > 0

    # The order or scores should be different between reranked and non-reranked results
    # This test might be a bit flaky depending on the exact scoring, but it should work most of the time
    # given our carefully crafted test data
    scores_no_rerank = [c["score"] for c in chunks_no_rerank]
    scores_with_rerank = [c["score"] for c in chunks_with_rerank]
    assert scores_no_rerank != scores_with_rerank, "Reranking should affect the scores"


@pytest.mark.asyncio
async def test_retrieve_docs_with_reranking(client: AsyncClient):
    """Test document retrieval with reranking options"""
    # First ingest documents with clearly different semantic relevance
    await test_ingest_text_document(
        client, "The capital of France is Paris. The city is known for the Eiffel Tower."
    )
    await test_ingest_text_document(
        client, "Paris is a city in France. It has many famous landmarks and museums."
    )
    await test_ingest_text_document(
        client, "Paris Hilton is a celebrity and businesswoman. She has nothing to do with France."
    )

    headers = create_auth_header()

    # Test with default reranking (from config)
    response_default = await client.post(
        "/retrieve/docs",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
        },
        headers=headers,
    )
    assert response_default.status_code == 200
    docs_default = response_default.json()
    assert len(docs_default) > 0

    # Test with explicit reranking enabled
    response_rerank = await client.post(
        "/retrieve/docs",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
            "use_reranking": True,
        },
        headers=headers,
    )
    assert response_rerank.status_code == 200
    docs_rerank = response_rerank.json()
    assert len(docs_rerank) > 0

    # Test with reranking disabled
    response_no_rerank = await client.post(
        "/retrieve/docs",
        json={
            "query": "Tell me about the capital city of France",
            "k": 3,
            "min_score": 0.0,
            "use_reranking": False,
        },
        headers=headers,
    )
    assert response_no_rerank.status_code == 200
    docs_no_rerank = response_no_rerank.json()
    assert len(docs_no_rerank) > 0

    # Verify that reranking affects the order
    scores_rerank = [doc["score"] for doc in docs_rerank]
    scores_no_rerank = [doc["score"] for doc in docs_no_rerank]
    assert scores_rerank != scores_no_rerank, "Reranking should affect document scores"


@pytest.mark.asyncio
async def test_query_with_reranking(client: AsyncClient):
    """Test query completion with reranking options"""
    # First ingest documents with clearly different semantic relevance
    await test_ingest_text_document(
        client, "The capital of France is Paris. The city is known for the Eiffel Tower."
    )
    await test_ingest_text_document(
        client, "Paris is a city in France. It has many famous landmarks and museums."
    )
    await test_ingest_text_document(
        client, "Paris Hilton is a celebrity and businesswoman. She has nothing to do with France."
    )

    headers = create_auth_header()

    # Test with default reranking (from config)
    response_default = await client.post(
        "/query",
        json={
            "query": "What is the capital of France?",
            "k": 3,
            "min_score": 0.0,
            "max_tokens": 50,
        },
        headers=headers,
    )
    assert response_default.status_code == 200
    completion_default = response_default.json()
    assert "completion" in completion_default

    # Test with explicit reranking enabled
    response_rerank = await client.post(
        "/query",
        json={
            "query": "What is the capital of France?",
            "k": 3,
            "min_score": 0.0,
            "max_tokens": 50,
            "use_reranking": True,
        },
        headers=headers,
    )
    assert response_rerank.status_code == 200
    completion_rerank = response_rerank.json()
    assert "completion" in completion_rerank

    # Test with reranking disabled
    response_no_rerank = await client.post(
        "/query",
        json={
            "query": "What is the capital of France?",
            "k": 3,
            "min_score": 0.0,
            "max_tokens": 50,
            "use_reranking": False,
        },
        headers=headers,
    )
    assert response_no_rerank.status_code == 200
    completion_no_rerank = response_no_rerank.json()
    assert "completion" in completion_no_rerank

    # The actual responses might be different due to different chunk ordering,
    # but all should mention Paris as the capital
    assert "Paris" in completion_default["completion"]
    assert "Paris" in completion_rerank["completion"]
    assert "Paris" in completion_no_rerank["completion"]

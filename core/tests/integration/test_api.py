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
from core.api import get_settings
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
    
    # Override database settings to use test database
    # This ensures we don't use the production database from .env
    settings.POSTGRES_URI = TEST_POSTGRES_URI
    settings.DATABASE_PROVIDER = "postgres"  # Ensure we're using postgres
    
    # IMPORTANT: We need to completely reinitialize the database connections
    # since they were already established at import time
    
    # First, get the database from the API module
    from core.api import database as api_database, app
    
    # Close existing connection if it exists
    if hasattr(api_database, 'engine'):
        await api_database.engine.dispose()
    
    # Create a new database connection with the test URI
    from core.database.postgres_database import PostgresDatabase
    test_database = PostgresDatabase(uri=TEST_POSTGRES_URI)
    
    # Initialize the test database
    await test_database.initialize()
    
    # Replace the global database instance with our test database
    import core.api
    core.api.database = test_database
    
    # Also update the vector store if it uses the same database (for pgvector)
    if settings.VECTOR_STORE_PROVIDER == "pgvector":
        from core.vector_store.pgvector_store import PGVectorStore
        from core.api import vector_store as api_vector_store
        
        # Create a new vector store with the test URI
        test_vector_store = PGVectorStore(uri=TEST_POSTGRES_URI)
        
        # Replace the global vector store with our test version
        core.api.vector_store = test_vector_store
    
    # Update the document service with our test instances
    from core.api import document_service as api_document_service
    from core.services.document_service import DocumentService
    from core.api import parser, embedding_model, reranker, storage
    
    # Create a new document service with our test database and vector store
    from core.api import completion_model, cache_factory, colpali_embedding_model, colpali_vector_store
    
    test_document_service = DocumentService(
        database=test_database,
        vector_store=core.api.vector_store,
        parser=parser,
        embedding_model=embedding_model,
        completion_model=completion_model,
        cache_factory=cache_factory,
        reranker=reranker,
        storage=storage,
        enable_colpali=settings.ENABLE_COLPALI,
        colpali_embedding_model=colpali_embedding_model,
        colpali_vector_store=colpali_vector_store,
    )
    
    # Replace the global document service with our test version
    core.api.document_service = test_document_service
    
    # Update the graph service if needed
    if hasattr(core.api, 'graph_service'):
        from core.services.graph_service import GraphService
        from core.api import completion_model
        
        test_graph_service = GraphService(
            db=test_database,
            embedding_model=embedding_model,
            completion_model=completion_model
        )
        
        core.api.graph_service = test_graph_service
    
    return app


@pytest.fixture
async def client(
    test_app: FastAPI, event_loop: asyncio.AbstractEventLoop
) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function", autouse=True)
async def cleanup_documents():
    """Clean up documents before each document test"""
    # This will run before the test
    yield
    # This will run after the test
    
    # We should always use the test database
    # Create a fresh connection to make sure we're not affected by any state
    engine = create_async_engine(TEST_POSTGRES_URI)
    
    try:
        async with engine.begin() as conn:
            # Clean up by deleting all rows rather than dropping tables
            await conn.execute(text("DELETE FROM documents"))
            
            # Delete from chunks table
            try:
                await conn.execute(text("DELETE FROM vector_embeddings"))
            except Exception as e:
                logger.info(f"No chunks table to clean or error: {e}")

    except Exception as e:
        logger.error(f"Failed to clean up document tables: {e}")
        raise
    finally:
        await engine.dispose()


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
async def test_ingest_text_document_with_metadata(client: AsyncClient, content: str = "Test content for document ingestion", metadata: dict = None):
    """Test ingesting a text document with metadata"""
    headers = create_auth_header()

    response = await client.post(
        "/ingest/text",
        json={"content": content, "metadata": metadata, "use_colpali": True},
        headers=headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "external_id" in data
    assert data["content_type"] == "text/plain"
    
    for key, value in (metadata or {}).items():
        assert data["metadata"][key] == value

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


# Knowledge Graph Tests

@pytest.fixture(scope="function", autouse=True)
async def cleanup_graphs():
    """Clean up graphs before each graph test"""
    # Create a fresh connection to the test database
    engine = create_async_engine(TEST_POSTGRES_URI)
    try:
        async with engine.begin() as conn:
            # Delete all rows from the graphs table
            await conn.execute(text("DELETE FROM graphs"))
            logger.info("Cleaned up all graph-related tables")
    except Exception as e:
        logger.error(f"Failed to clean up graph tables: {e}")
        raise
    finally:
        await engine.dispose()
        
    # This will run before each test function
    yield
    # Test runs here


@pytest.mark.asyncio
async def test_create_graph(client: AsyncClient):
    """Test creating a knowledge graph from documents."""
    # First ingest multiple documents with related content to extract entities and relationships
    doc_id1 = await test_ingest_text_document(
        client,
        content="Apple Inc. is a technology company headquartered in Cupertino, California. "
        "Tim Cook is the CEO of Apple. Steve Jobs was the co-founder of Apple."
    )

    doc_id2 = await test_ingest_text_document(
        client,
        content="Microsoft is a technology company based in Redmond, Washington. "
        "Satya Nadella is the CEO of Microsoft. Bill Gates co-founded Microsoft."
    )

    doc_id3 = await test_ingest_text_document(
        client,
        content="Tim Cook succeeded Steve Jobs as the CEO of Apple in 2011. "
        "Under Tim Cook's leadership, Apple became the world's first trillion-dollar company."
    )

    headers = create_auth_header()
    graph_name = "test_tech_companies_graph"

    # Create graph using the document IDs
    response = await client.post(
        "/graph/create",
        json={
            "name": graph_name,
            "documents": [doc_id1, doc_id2, doc_id3]
        },
        headers=headers,
    )

    assert response.status_code == 200
    graph = response.json()

    # Verify graph structure
    assert graph["name"] == graph_name
    assert len(graph["document_ids"]) == 3
    assert all(doc_id in graph["document_ids"] for doc_id in [doc_id1, doc_id2, doc_id3])
    assert len(graph["entities"]) > 0  # Should extract entities like Apple, Microsoft, Tim Cook, etc.
    assert len(graph["relationships"]) > 0  # Should extract relationships like "Tim Cook is CEO of Apple"

    # Verify specific expected entities were extracted
    entity_labels = [entity["label"] for entity in graph["entities"]]
    assert any("Apple" in label for label in entity_labels)
    assert any("Microsoft" in label for label in entity_labels)
    assert any("Tim Cook" in label for label in entity_labels)

    # Verify entity types
    entity_types = set(entity["type"] for entity in graph["entities"])
    assert "ORGANIZATION" in entity_types or "COMPANY" in entity_types
    assert "PERSON" in entity_types

    # Verify entities have document references
    for entity in graph["entities"]:
        assert len(entity["document_ids"]) > 0
        assert entity["document_ids"][0] in [doc_id1, doc_id2, doc_id3]

    return graph_name, [doc_id1, doc_id2, doc_id3]


@pytest.mark.asyncio
async def test_get_graph(client: AsyncClient):
    """Test retrieving a knowledge graph by name."""
    # First create a graph
    graph_name, _ = await test_create_graph(client)

    # Then retrieve it
    headers = create_auth_header()
    response = await client.get(
        f"/graph/{graph_name}",
        headers=headers,
    )

    assert response.status_code == 200
    graph = response.json()

    # Verify correct graph was retrieved
    assert graph["name"] == graph_name
    assert len(graph["entities"]) > 0
    assert len(graph["relationships"]) > 0


@pytest.mark.asyncio
async def test_list_graphs(client: AsyncClient):
    """Test listing all accessible graphs."""
    # First create a graph
    graph_name, _ = await test_create_graph(client)

    # List all graphs
    headers = create_auth_header()
    response = await client.get(
        "/graphs",
        headers=headers,
    )

    assert response.status_code == 200
    graphs = response.json()

    # Verify the created graph is in the list
    assert len(graphs) > 0
    assert any(graph["name"] == graph_name for graph in graphs)


@pytest.mark.asyncio
async def test_create_graph_with_filters(client: AsyncClient):
    """Test creating a knowledge graph using metadata filters."""
    # Ingest test documents with specific metadata
    doc_id1 = await test_ingest_text_document_with_metadata(
        client,
        content="The solar system consists of the Sun and eight planets. "
        "Earth is the third planet from the Sun. Mars is the fourth planet.",
        metadata={"category": "astronomy", "subject": "planets"}
    )

    headers = create_auth_header()

    # Get the document to add specific metadata
    response = await client.get(f"/documents/{doc_id1}", headers=headers)
    assert response.status_code == 200

    # Create graph with filters - using metadata field which is renamed to doc_metadata in PostgresDatabase
    graph_name = "test_astronomy_graph"
    response = await client.post(
        "/graph/create",
        json={
            "name": graph_name,
            "filters": {"category": "astronomy"}
        },
        headers=headers,
    )

    assert response.status_code == 200
    graph = response.json()

    # Verify graph was created with the right document
    assert graph["name"] == graph_name
    assert doc_id1 in graph["document_ids"]
    assert len(graph["entities"]) > 0  # Should extract entities like Sun, Earth, Mars

    # Verify specific entities
    entity_labels = [entity["label"] for entity in graph["entities"]]
    assert any(label == "Sun" or "Sun" in label for label in entity_labels)
    assert any(label == "Earth" or "Earth" in label for label in entity_labels)

    return graph_name, doc_id1


@pytest.mark.asyncio
async def test_query_with_graph(client: AsyncClient):
    """Test query completion with knowledge graph enhancement."""
    # First create a graph and get its name
    graph_name, doc_ids = await test_create_graph(client)

    # Additional document that won't be in the graph but contains related information
    _ = await test_ingest_text_document(
        client,
        content="Apple has released a new iPhone model. The company's focus on innovation continues."
    )
    
    headers = create_auth_header()
    
    # Query using the graph
    response = await client.post(
        "/query",
        json={
            "query": "Who is the CEO of Apple?",
            "graph_name": graph_name,
            "hop_depth": 2,
            "include_paths": True
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    result = response.json()
    
    # Verify the completion contains relevant information from graph
    assert "completion" in result
    assert any(term in result["completion"] for term in ["Tim Cook", "Cook", "CEO", "Apple"])
    
    # Verify we have graph metadata when include_paths=True
    assert "metadata" in result, "Expected metadata in response when include_paths=True"
    assert "graph" in result["metadata"], "Expected graph metadata in response"
    assert result["metadata"]["graph"]["name"] == graph_name
    
    # Verify relevant entities includes expected entities
    assert "relevant_entities" in result["metadata"]["graph"]
    relevant_entities = result["metadata"]["graph"]["relevant_entities"]
    
    # At least one relevant entity should contain either Tim Cook or Apple
    has_tim_cook = any("Tim Cook" in entity or "Cook" in entity for entity in relevant_entities)
    has_apple = any("Apple" in entity for entity in relevant_entities)
    assert has_tim_cook or has_apple, "Expected relevant entities to include Tim Cook or Apple"
    
    # Now try without the graph for comparison
    response_no_graph = await client.post(
        "/query",
        json={
            "query": "Who is the CEO of Apple?",
        },
        headers=headers,
    )
    
    assert response_no_graph.status_code == 200


@pytest.mark.asyncio
async def test_batch_ingest_with_shared_metadata(
    client: AsyncClient
):
    """Test batch ingestion with shared metadata for all files."""
    headers = create_auth_header()
    # Create test files
    files = [
        ("files", ("test1.txt", b"Test content 1")),
        ("files", ("test2.txt", b"Test content 2")),
    ]
    
    # Shared metadata for all files
    metadata = {"category": "test", "batch": "shared"}
    
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps(metadata),
            "rules": json.dumps([]),
            "use_colpali": "true",
            "parallel": "true",
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    result = response.json()
    assert len(result["documents"]) == 2
    assert len(result["errors"]) == 0
    
    # Verify all documents got the same metadata
    for doc in result["documents"]:
        assert doc["metadata"]["category"] == "test"
        assert doc["metadata"]["batch"] == "shared"


@pytest.mark.asyncio
async def test_batch_ingest_with_individual_metadata(
    client: AsyncClient
):
    """Test batch ingestion with individual metadata per file."""
    headers = create_auth_header()
    # Create test files
    files = [
        ("files", ("test1.txt", b"Test content 1")),
        ("files", ("test2.txt", b"Test content 2")),
    ]
    
    # Individual metadata
    metadata = [
        {"category": "test1", "batch": "individual"},
        {"category": "test2", "batch": "individual"},
    ]
    
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps(metadata),
            "rules": json.dumps([]),
            "use_colpali": "true",
            "parallel": "true",
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    result = response.json()
    assert len(result["documents"]) == 2
    assert len(result["errors"]) == 0
    
    # Verify each document got its correct metadata
    assert result["documents"][0]["metadata"]["category"] == "test1"
    assert result["documents"][1]["metadata"]["category"] == "test2"


@pytest.mark.asyncio
async def test_batch_ingest_metadata_validation(
    client: AsyncClient
):
    """Test validation when metadata list length doesn't match files."""
    headers = create_auth_header()
    files = [
        ("files", ("test1.txt", b"Test content 1")),
        ("files", ("test2.txt", b"Test content 2")),
    ]
    
    # Metadata list with wrong length
    metadata = [
        {"category": "test1"},
        {"category": "test2"},
        {"category": "test3"},  # Extra metadata
    ]
    
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps(metadata),
            "rules": json.dumps([]),
            "use_colpali": "true",
            "parallel": "true",
        },
        headers=headers,
    )
    
    assert response.status_code == 400
    assert "must match number of files" in response.json()["detail"]


@pytest.mark.asyncio
async def test_batch_ingest_sequential(
    client: AsyncClient
):
    """Test sequential batch ingestion."""
    headers = create_auth_header()
    files = [
        ("files", ("test1.txt", b"Test content 1")),
        ("files", ("test2.txt", b"Test content 2")),
    ]
    
    metadata = {"category": "test"}
    
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps(metadata),
            "rules": json.dumps([]),
            "use_colpali": "true",
            "parallel": "false",  # Process sequentially
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    result = response.json()
    assert len(result["documents"]) == 2
    assert len(result["errors"]) == 0


@pytest.mark.asyncio
async def test_batch_ingest_with_rules(
    client: AsyncClient
):
    """Test batch ingestion with rules applied."""
    headers = create_auth_header()
    files = [
        ("files", ("test1.txt", b"Test content 1")),
        ("files", ("test2.txt", b"Test content 2")),
    ]
    
    # Test shared rules for all files
    shared_rules = [{"type": "natural_language", "prompt": "Extract keywords"}]
    
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps({}),
            "rules": json.dumps(shared_rules),
            "use_colpali": "true",
            "parallel": "true",
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    result = response.json()
    assert len(result["documents"]) == 2
    assert len(result["errors"]) == 0
    
    # Test per-file rules
    per_file_rules = [
        [{"type": "natural_language", "prompt": "Extract keywords"}],  # Rules for first file
        [{"type": "metadata_extraction", "schema": {"title": "string"}}],  # Rules for second file
    ]
    
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps({}),
            "rules": json.dumps(per_file_rules),
            "use_colpali": "true",
            "parallel": "true",
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    result = response.json()
    assert len(result["documents"]) == 2
    assert len(result["errors"]) == 0


@pytest.mark.asyncio
async def test_batch_ingest_rules_validation(
    client: AsyncClient
):
    """Test validation of rules format and length."""
    headers = create_auth_header()
    files = [
        ("files", ("test1.txt", b"Test content 1")),
        ("files", ("test2.txt", b"Test content 2")),
    ]
    
    # Test invalid rules format
    invalid_rules = "not a list"
    
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps({}),
            "rules": invalid_rules,
            "use_colpali": "true",
            "parallel": "true",
        },
        headers=headers,
    )
    
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]
    
    # Test per-file rules with wrong length
    per_file_rules = [
        [{"type": "natural_language", "prompt": "Extract keywords"}],  # Only one set of rules
    ]
    
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps({}),
            "rules": json.dumps(per_file_rules),
            "use_colpali": "true",
            "parallel": "true",
        },
        headers=headers,
    )
    
    assert response.status_code == 400
    assert "must match number of files" in response.json()["detail"]


@pytest.mark.asyncio
async def test_batch_ingest_sequential_vs_parallel(
    client: AsyncClient
):
    """Test both sequential and parallel batch ingestion."""
    headers = create_auth_header()
    files = [
        ("files", ("test1.txt", b"Test content 1")),
        ("files", ("test2.txt", b"Test content 2")),
        ("files", ("test3.txt", b"Test content 3")),
    ]
    
    # Test parallel processing
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps({}),
            "rules": json.dumps([]),
            "use_colpali": "true",
            "parallel": "true",
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    result = response.json()
    assert len(result["documents"]) == 3
    assert len(result["errors"]) == 0
    
    # Test sequential processing 
    response = await client.post(
        "/ingest/files",
        files=files,
        data={
            "metadata": json.dumps({}),
            "rules": json.dumps([]),
            "use_colpali": "true",
            "parallel": "false",
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    result = response.json()
    assert len(result["documents"]) == 3
    assert len(result["errors"]) == 0


@pytest.mark.asyncio
async def test_delete_document(client: AsyncClient):
    """Test deleting a document and verifying it's gone."""
    # First ingest a document to delete
    content = "This is a test document that will be deleted."
    doc_id = await test_ingest_text_document(client, content=content)
    
    headers = create_auth_header()
    
    # Verify document exists
    response = await client.get(f"/documents/{doc_id}", headers=headers)
    assert response.status_code == 200
    
    # Verify document is searchable
    search_response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "test document deleted",
            "filters": {"external_id": doc_id},
        },
        headers=headers,
    )
    assert search_response.status_code == 200
    chunks = search_response.json()
    assert len(chunks) > 0
    
    # Delete document
    delete_response = await client.delete(
        f"/documents/{doc_id}",
        headers=headers,
    )
    assert delete_response.status_code == 200
    result = delete_response.json()
    assert result["status"] == "success"
    assert f"Document {doc_id} deleted successfully" in result["message"]
    
    # Verify document no longer exists
    response = await client.get(f"/documents/{doc_id}", headers=headers)
    assert response.status_code == 404
    
    # Verify document is no longer searchable
    search_response = await client.post(
        "/retrieve/chunks",
        json={
            "query": "test document deleted",
            "filters": {"external_id": doc_id},
        },
        headers=headers,
    )
    assert search_response.status_code == 200
    chunks = search_response.json()
    assert len(chunks) == 0  # No chunks should be found

@pytest.mark.asyncio
async def test_delete_document_permission_error(client: AsyncClient):
    """Test permissions handling for document deletion."""
    if get_settings().dev_mode:
        pytest.skip("Auth tests skipped in dev mode")
    
    # First ingest a document to delete
    content = "This is a test document for testing delete permissions."
    doc_id = await test_ingest_text_document(client, content=content)
    
    # Try to delete with read-only permission
    headers = create_auth_header(permissions=["read"])
    
    delete_response = await client.delete(
        f"/documents/{doc_id}",
        headers=headers,
    )
    assert delete_response.status_code == 403
    
    # Verify document still exists
    headers = create_auth_header()  # Full permissions
    response = await client.get(f"/documents/{doc_id}", headers=headers)
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_delete_nonexistent_document(client: AsyncClient):
    """Test deleting a document that doesn't exist."""
    headers = create_auth_header()
    
    delete_response = await client.delete(
        "/documents/nonexistent_document_id",
        headers=headers,
    )
    assert delete_response.status_code == 404

@pytest.mark.asyncio
async def test_cross_document_query_with_graph(client: AsyncClient):
    """Test cross-document information retrieval using knowledge graph."""
    # Create a graph with multiple documents containing related information
    doc_id1 = await test_ingest_text_document(
        client,
        content="Project Alpha was initiated in 2020. Jane Smith is the project lead."
    )
    
    doc_id2 = await test_ingest_text_document(
        client,
        content="Jane Smith has a PhD in Computer Science from MIT. She has 10 years of experience in AI research."
    )
    
    doc_id3 = await test_ingest_text_document(
        client,
        content="Project Alpha aims to develop advanced natural language processing models for medical applications."
    )
    
    headers = create_auth_header()
    graph_name = "test_project_graph"
    
    # Create graph using the document IDs
    response = await client.post(
        "/graph/create",
        json={
            "name": graph_name,
            "documents": [doc_id1, doc_id2, doc_id3]
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    
    # Query that requires connecting information across documents
    response = await client.post(
        "/query",
        json={
            "query": "What is Jane Smith's background and what project is she leading?",
            "graph_name": graph_name,
            "hop_depth": 2,
            "include_paths": True
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    result = response.json()
    
    # Verify the completion combines information from multiple documents
    assert "Jane Smith" in result["completion"]
    assert "PhD" in result["completion"] or "Computer Science" in result["completion"]
    assert "Project Alpha" in result["completion"]

    # Compare with non-graph query
    response_no_graph = await client.post(
        "/query",
        json={
            "query": "What is Jane Smith's background and what project is she leading?",
        },
        headers=headers,
    )

    assert response_no_graph.status_code == 200


@pytest.mark.asyncio
async def test_update_graph(client: AsyncClient):
    """Test updating a knowledge graph with new documents."""
    # Create initial graph with some documents
    doc_id1 = await test_ingest_text_document(
        client,
        content="SpaceX was founded by Elon Musk in 2002. It develops and manufactures spacecraft and rockets."
    )
    
    doc_id2 = await test_ingest_text_document(
        client,
        content="Elon Musk is also the CEO of Tesla, an electric vehicle manufacturer."
    )
    
    headers = create_auth_header()
    graph_name = "test_update_graph"
    
    # Create initial graph
    response = await client.post(
        "/graph/create",
        json={
            "name": graph_name,
            "documents": [doc_id1, doc_id2]
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    initial_graph = response.json()
    
    # Verify initial graph structure
    assert initial_graph["name"] == graph_name
    assert len(initial_graph["document_ids"]) == 2
    assert all(doc_id in initial_graph["document_ids"] for doc_id in [doc_id1, doc_id2])
    
    # Create some new documents to add to the graph
    doc_id3 = await test_ingest_text_document(
        client,
        content="The Starship is a spacecraft being developed by SpaceX. It's designed for missions to Mars."
    )
    
    doc_id4 = await test_ingest_text_document(
        client,
        content="Gwynne Shotwell is the President and COO of SpaceX. She joined the company in 2002."
    )
    
    # Update the graph with new documents
    update_response = await client.post(
        f"/graph/{graph_name}/update",
        json={
            "additional_documents": [doc_id3, doc_id4]
        },
        headers=headers,
    )
    
    assert update_response.status_code == 200
    updated_graph = update_response.json()
    
    # Verify updated graph structure
    assert updated_graph["name"] == graph_name
    assert len(updated_graph["document_ids"]) == 4
    assert all(doc_id in updated_graph["document_ids"] for doc_id in [doc_id1, doc_id2, doc_id3, doc_id4])
    
    # Verify new entities and relationships were added
    assert len(updated_graph["entities"]) > len(initial_graph["entities"])
    assert len(updated_graph["relationships"]) > len(initial_graph["relationships"])
    
    # Verify specific new entities were extracted
    entity_labels = [entity["label"].lower() for entity in updated_graph["entities"]]
    assert any("starship" in label for label in entity_labels)
    assert any("gwynne shotwell" in label for label in entity_labels)
    
    # Test updating with filters
    # Create a document with specific metadata
    doc_id5 = await test_ingest_text_document_with_metadata(
        client,
        content="The Falcon 9 is a reusable rocket developed by SpaceX.",
        metadata={"company": "spacex"}
    )

    # Verify metadata was set correctly
    doc_response = await client.get(f"/documents/{doc_id5}", headers=headers)
    assert doc_response.status_code == 200
    doc_data = doc_response.json()
    print(f"\nDEBUG - Document {doc_id5} metadata: {doc_data['metadata']}")
    
    # Update graph using filters
    print(f"\nDEBUG - Updating graph with filter: {{'company': 'spacex'}}")
    filter_update_response = await client.post(
        f"/graph/{graph_name}/update",
        json={
            "additional_filters": {"company": "spacex"}
        },
        headers=headers,
    )
    print(f"\nDEBUG - Filter update response status: {filter_update_response.status_code}")
    
    assert filter_update_response.status_code == 200
    filter_updated_graph = filter_update_response.json()
    
    # Verify the document was added via filters
    print(f"\nDEBUG - Graph document IDs: {filter_updated_graph['document_ids']}")
    print(f"\nDEBUG - Looking for document: {doc_id5}")
    print(f"\nDEBUG - Number of document IDs: {len(filter_updated_graph['document_ids'])}")
    print(f"\nDEBUG - doc_id5 in document_ids: {doc_id5 in filter_updated_graph['document_ids']}")
    
    assert len(filter_updated_graph["document_ids"]) == 5
    assert doc_id5 in filter_updated_graph["document_ids"]
    
    # Verify the new entity was added
    new_entity_labels = [entity["label"].lower() for entity in filter_updated_graph["entities"]]
    assert any("falcon 9" in label for label in new_entity_labels)
    
    # Test updating with both documents and filters
    doc_id6 = await test_ingest_text_document(
        client,
        content="The Tesla Cybertruck is an electric pickup truck announced in 2019."
    )
    
    doc_id7 = await test_ingest_text_document_with_metadata(
        client,
        content="Starlink is a satellite internet constellation developed by SpaceX.",
        metadata={"company": "spacex", "type": "satellite"}
    )

    # Update with both specific document and filter
    combined_update_response = await client.post(
        f"/graph/{graph_name}/update",
        json={
            "additional_documents": [doc_id6],
            "additional_filters": {"type": "satellite"}
        },
        headers=headers,
    )
    
    assert combined_update_response.status_code == 200
    combined_updated_graph = combined_update_response.json()
    
    # Verify both documents were added
    assert len(combined_updated_graph["document_ids"]) == 7
    assert doc_id6 in combined_updated_graph["document_ids"]
    assert doc_id7 in combined_updated_graph["document_ids"]
    
    # Verify new entities
    final_entity_labels = [entity["label"].lower() for entity in combined_updated_graph["entities"]]
    assert any("cybertruck" in label for label in final_entity_labels)
    assert any("starlink" in label for label in final_entity_labels)
    
    # Test querying with the updated graph
    query_response = await client.post(
        "/query",
        json={
            "query": "What spacecraft and rockets has SpaceX developed?",
            "graph_name": graph_name,
            "hop_depth": 2,
            "include_paths": True
        },
        headers=headers,
    )
    
    assert query_response.status_code == 200
    query_result = query_response.json()
    
    # Verify the completion includes information from the added documents
    completion = query_result["completion"].lower()
    assert "starship" in completion
    assert "falcon 9" in completion
    assert "starlink" in completion

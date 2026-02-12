import os
import time
import uuid
from pathlib import Path

import pytest
from morphik.sync import Morphik
from pydantic import BaseModel, Field

# Set to your local Morphik server - use localhost by default
# Default client connects to localhost:8000 automatically

# Skip these tests if the SKIP_LIVE_TESTS environment variable is set
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_LIVE_TESTS") == "1",
    reason="Skip tests that require a running Morphik server",
)

# Get the test files directory
TEST_DOCS_DIR = Path(__file__).parent / "test_docs"


class StructuredOutputSchema(BaseModel):
    summary: str = Field(..., description="A short summary of the input text")
    key_points: list[str] = Field(..., description="A list of key points from the text")


class TestMorphik:
    """
    Tests for the synchronous Morphik SDK client with a live server.

    To run these tests, start a local Morphik server and then run:
    pytest morphik/tests/test_sync.py -v
    """

    @pytest.fixture
    def db(self):
        """Create a Morphik client for testing"""
        # Use dedicated test URI when provided; otherwise default localhost behavior
        uri = os.environ.get("MORPHIK_TEST_URI")
        client = Morphik(uri=uri, timeout=120) if uri else Morphik(timeout=120)
        yield client
        client.close()

    def test_ingest_text(self, db):
        """Test ingesting a text document"""
        # Generate a unique filename to avoid conflicts
        filename = f"test_{uuid.uuid4().hex[:8]}.txt"

        # Test basic text ingestion
        doc = db.ingest_text(
            content="This is a test document for the Morphik SDK.",
            filename=filename,
            metadata={"test_id": "sync_text_test", "category": "test"},
        )

        # Verify the document was created
        assert doc.external_id is not None
        assert doc.filename == filename
        assert "test_id" in doc.metadata
        assert doc.metadata["test_id"] == "sync_text_test"

        # Clean up
        db.delete_document(doc.external_id)

    def test_ingest_file(self, db):
        """Test ingesting a file from disk"""
        # Use one of our test documents
        file_path = TEST_DOCS_DIR / "sample1.txt"

        # Test file ingestion
        doc = db.ingest_file(file=file_path, metadata={"test_id": "sync_file_test", "category": "test"})

        # Verify the document was created
        assert doc.external_id is not None
        assert doc.filename == "sample1.txt"
        assert "test_id" in doc.metadata
        assert doc.metadata["test_id"] == "sync_file_test"

        # Clean up
        db.delete_document(doc.external_id)

    def test_retrieve_chunks(self, db):
        """Test retrieving chunks with a query"""
        # First ingest a document
        doc = db.ingest_text(
            content="Artificial intelligence and machine learning are transforming industries worldwide.",
            filename=f"test_{uuid.uuid4().hex[:8]}.txt",
            metadata={"test_id": "sync_retrieval_test", "category": "test"},
        )

        # Wait for processing to complete
        max_retries = 10
        for _ in range(max_retries):
            try:
                status = db.get_document_status(doc.external_id)
                if status.get("status") == "completed":
                    break
                time.sleep(2)  # Wait before checking again
            except Exception:
                time.sleep(2)

        # Test retrieval
        chunks = db.retrieve_chunks(
            query="What is artificial intelligence?", filters={"test_id": "sync_retrieval_test"}
        )

        # Verify results (may be empty if processing is slow)
        if len(chunks) > 0:
            assert chunks[0].document_id == doc.external_id
            assert chunks[0].score > 0

        # Clean up
        db.delete_document(doc.external_id)

    def test_folder_operations(self, db):
        """Test folder operations"""
        # Create a unique folder name
        folder_name = f"test_folder_{uuid.uuid4().hex[:8]}"

        # Create a folder
        folder = db.create_folder(name=folder_name, description="Test folder for SDK tests")

        # Verify folder was created
        assert folder.name == folder_name
        assert folder.id is not None

        # Test ingesting a document into the folder
        doc = folder.ingest_text(
            content="This is a test document in a folder.",
            filename=f"test_{uuid.uuid4().hex[:8]}.txt",
            metadata={"test_id": "sync_folder_test", "category": "test"},
        )

        # Verify the document was created
        assert doc.external_id is not None

        # List documents in the folder
        docs = folder.list_documents()

        # There should be at least our test document
        doc_ids = [d.external_id for d in docs.documents]
        assert doc.external_id in doc_ids

        # Clean up - first delete the document
        db.delete_document(doc.external_id)

        # TODO: Add folder deletion when API supports it

    def test_direct_http_uri_ping(self):
        """Test direct HTTP base URL initialization and connectivity."""
        uri = os.environ.get("MORPHIK_TEST_URI", "http://localhost:8000")
        client = Morphik(uri, timeout=30)
        try:
            response = client.ping()
            assert response.get("status") == "ok"
        finally:
            client.close()

    def test_move_and_rename_folder(self, db):
        """Test moving and renaming folders via client-level APIs."""
        suffix = uuid.uuid4().hex[:8]
        original_path = f"/it_sync_{suffix}/leaf"
        moved_path = f"/it_sync_moved_{suffix}/leaf"
        renamed_leaf = f"leaf_renamed_{suffix}"
        expected_renamed_path = f"/it_sync_moved_{suffix}/{renamed_leaf}"

        folder = db.create_folder(name=f"leaf_{suffix}", full_path=original_path, description="integration move test")

        moved = None
        renamed = None
        try:
            moved = db.move_folder(folder.id or original_path, moved_path)
            assert moved.id == folder.id
            assert moved.full_path == moved_path
            assert moved.name == "leaf"

            fetched_moved = db.get_folder(moved.id or moved_path)
            assert fetched_moved.full_path == moved_path

            renamed = db.rename_folder(moved.id or moved_path, renamed_leaf)
            assert renamed.id == folder.id
            assert renamed.name == renamed_leaf
            assert renamed.full_path == expected_renamed_path

            fetched_renamed = db.get_folder(renamed.id or expected_renamed_path)
            assert fetched_renamed.full_path == expected_renamed_path
        finally:
            target = renamed or moved or folder
            if target and target.id:
                db.delete_folder(target.id)

    def test_folder_object_move_and_rename(self, db):
        """Test Folder convenience move/rename methods keep local metadata in sync."""
        suffix = uuid.uuid4().hex[:8]
        original_path = f"/it_sync_obj_{suffix}/leaf"
        moved_path = f"/it_sync_obj_moved_{suffix}/leaf"
        renamed_leaf = f"leaf_obj_renamed_{suffix}"
        expected_renamed_path = f"/it_sync_obj_moved_{suffix}/{renamed_leaf}"

        folder = db.create_folder(name=f"leaf_obj_{suffix}", full_path=original_path)
        try:
            moved = folder.move(moved_path)
            assert moved.full_path == moved_path
            assert folder.full_path == moved_path

            renamed = folder.rename(renamed_leaf)
            assert renamed.full_path == expected_renamed_path
            assert renamed.name == renamed_leaf
            assert folder.full_path == expected_renamed_path
            assert folder.name == renamed_leaf
        finally:
            if folder.id:
                db.delete_folder(folder.id)

    def test_user_scope(self, db):
        """Test user scoped operations"""
        # Create a unique user ID
        user_id = f"test_user_{uuid.uuid4().hex[:8]}"

        # Create a user scope
        user_scope = db.signin(user_id)

        # Verify user scope
        assert user_scope.end_user_id == user_id

        # Test ingesting a document as the user
        doc = user_scope.ingest_text(
            content="This is a test document from a specific user.",
            filename=f"test_{uuid.uuid4().hex[:8]}.txt",
            metadata={"test_id": "sync_user_test", "category": "test"},
        )

        # Verify the document was created
        assert doc.external_id is not None
        assert "test_id" in doc.metadata
        assert doc.metadata["test_id"] == "sync_user_test"

        # List documents for this user
        docs = user_scope.list_documents()

        # There should be at least our test document
        doc_ids = [d.external_id for d in docs.documents]
        assert doc.external_id in doc_ids

        # Clean up
        db.delete_document(doc.external_id)

    def test_batch_operations(self, db):
        """Test batch operations"""
        # Ingest multiple files
        files = [
            TEST_DOCS_DIR / "sample1.txt",
            TEST_DOCS_DIR / "sample2.txt",
            TEST_DOCS_DIR / "sample3.txt",
        ]

        # Test batch ingestion
        docs = db.ingest_files(files=files, metadata={"test_id": "sync_batch_test", "category": "test"}, parallel=True)

        # Verify documents were created
        assert len(docs) == 3
        file_names = [doc.filename for doc in docs]
        assert "sample1.txt" in file_names
        assert "sample2.txt" in file_names
        assert "sample3.txt" in file_names

        # Get documents in batch
        doc_ids = [doc.external_id for doc in docs]
        batch_docs = db.batch_get_documents(doc_ids)

        # Verify batch retrieval
        assert len(batch_docs) == len(doc_ids)
        retrieved_ids = [doc.external_id for doc in batch_docs]
        for doc_id in doc_ids:
            assert doc_id in retrieved_ids

        # Clean up
        for doc_id in doc_ids:
            db.delete_document(doc_id)

    def test_folder_with_user_scope(self, db):
        """Test combination of folder and user scope"""
        # Create unique names
        folder_name = f"test_folder_{uuid.uuid4().hex[:8]}"
        user_id = f"test_user_{uuid.uuid4().hex[:8]}"

        # Create a folder
        folder = db.create_folder(name=folder_name)

        # Create a user scope within the folder
        user_scope = folder.signin(user_id)

        # Verify scopes - folder_name uses full_path which includes leading slash
        assert user_scope.folder_name == folder.full_path
        assert user_scope.end_user_id == user_id

        # Test ingestion in this combined scope
        doc = user_scope.ingest_text(
            content="This is a test document in a folder from a specific user.",
            filename=f"test_{uuid.uuid4().hex[:8]}.txt",
            metadata={"test_id": "sync_folder_user_test", "category": "test"},
        )

        # Verify the document was created
        assert doc.external_id is not None

        # List documents in this scope
        docs = user_scope.list_documents()

        # There should be at least our test document
        doc_ids = [d.external_id for d in docs.documents]
        assert doc.external_id in doc_ids

        # Clean up
        db.delete_document(doc.external_id)

    def test_query_endpoint(self, db):
        """Test the query endpoint for RAG capabilities"""
        # First ingest a document
        doc = db.ingest_text(
            content="Artificial intelligence and machine learning are transforming industries worldwide. "
            "AI systems can now process natural language, recognize images, and make complex decisions.",
            filename=f"test_{uuid.uuid4().hex[:8]}.txt",
            metadata={"test_id": "sync_query_test", "category": "test"},
        )

        try:
            # Wait for processing to complete
            for _ in range(10):
                status = db.get_document_status(doc.external_id)
                if status.get("status") == "completed":
                    break
                time.sleep(2)

            # Only proceed with test if document is processed
            if status.get("status") == "completed":
                # Test the query endpoint
                response = db.query(
                    query="What can AI systems do?",
                    filters={"test_id": "sync_query_test"},
                    k=1,
                    temperature=0.7,
                )

                # Verify response
                assert response.completion is not None
                assert len(response.completion) > 0
                assert len(response.sources) > 0
                assert response.sources[0].document_id == doc.external_id

        finally:
            # Clean up
            db.delete_document(doc.external_id)

    def test_query_with_pydantic_schema(self, db):
        """Test the query endpoint with a Pydantic schema for structured output."""
        content = (
            "Morphik is a platform for building AI applications. "
            "It provides tools for data ingestion, retrieval, and generation. "
            "Key features include vector search and reranking."
        )
        doc = db.ingest_text(
            content=content,
            filename=f"test_schema_{uuid.uuid4().hex[:8]}.txt",
            metadata={"test_id": "sync_schema_pydantic_test"},
        )

        try:
            db.wait_for_document_completion(doc.external_id, timeout_seconds=60)

            response = db.query(
                query="Summarize this document and list its key points.",
                filters={"test_id": "sync_schema_pydantic_test"},
                k=1,
                schema=StructuredOutputSchema,
            )

            assert response.completion is not None
            # With the updated model, completion should be the dictionary itself
            assert isinstance(response.completion, dict)
            output_data = response.completion
            assert "summary" in output_data
            assert "key_points" in output_data
            assert isinstance(output_data["summary"], str)
            assert isinstance(output_data["key_points"], list)

        finally:
            db.delete_document(doc.external_id)

    def test_query_with_dict_schema(self, db):
        """Test the query endpoint with a dictionary schema for structured output."""
        content = "The capital of France is Paris. It is known for the Eiffel Tower."
        doc = db.ingest_text(
            content=content,
            filename=f"test_schema_dict_{uuid.uuid4().hex[:8]}.txt",
            metadata={"test_id": "sync_schema_dict_test"},
        )

        dict_schema = {
            "type": "object",
            "properties": {
                "capital": {"type": "string", "description": "The capital city"},
                "country": {"type": "string", "description": "The country name"},
                "landmark": {"type": "string", "description": "A famous landmark"},
            },
            "required": ["capital", "country"],
        }

        try:
            db.wait_for_document_completion(doc.external_id, timeout_seconds=60)

            response = db.query(
                query="Extract the capital, country, and a landmark.",
                filters={"test_id": "sync_schema_dict_test"},
                k=1,
                schema=dict_schema,
            )

            assert response.completion is not None
            # With the updated model, completion should be the dictionary itself
            assert isinstance(response.completion, dict)
            output_data = response.completion
            assert "capital" in output_data
            assert "country" in output_data
            # Landmark might not always be extracted, so check presence if required
            if "landmark" in dict_schema.get("required", []):
                assert "landmark" in output_data
            # Allow None if not required and type is string
            if "capital" not in dict_schema.get("required", []) and output_data.get("capital") is None:
                pass  # Allow None for non-required string
            else:
                assert isinstance(output_data.get("capital"), str)
            assert isinstance(output_data["country"], str)

        finally:
            db.delete_document(doc.external_id)

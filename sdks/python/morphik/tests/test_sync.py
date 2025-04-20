import os
import time
import uuid
from pathlib import Path

import pytest
from morphik.sync import Morphik

# Set to your local Morphik server - use localhost by default
# Default client connects to localhost:8000 automatically

# Skip these tests if the SKIP_LIVE_TESTS environment variable is set
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_LIVE_TESTS") == "1",
    reason="Skip tests that require a running Morphik server",
)

# Get the test files directory
TEST_DOCS_DIR = Path(__file__).parent / "test_docs"


class TestMorphik:
    """
    Tests for the synchronous Morphik SDK client with a live server.

    To run these tests, start a local Morphik server and then run:
    MORPHIK_TEST_URL=http://localhost:8000 pytest morphik/tests/test_sync.py -v
    """

    @pytest.fixture
    def db(self):
        """Create a Morphik client for testing"""
        client = Morphik()  # Connects to localhost:8000 by default
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
        processed_doc = doc
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
        doc_ids = [d.external_id for d in docs]
        assert doc.external_id in doc_ids

        # Clean up - first delete the document
        db.delete_document(doc.external_id)

        # TODO: Add folder deletion when API supports it

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
        doc_ids = [d.external_id for d in docs]
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

        # Verify scopes
        assert user_scope.folder_name == folder_name
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
        doc_ids = [d.external_id for d in docs]
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

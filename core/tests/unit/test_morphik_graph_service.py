import asyncio
import unittest
from datetime import UTC, datetime
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

from core.models.auth import AuthContext, EntityType
from core.models.documents import Document
from core.services.morphik_graph_service import MorphikGraphService


class MockStorage:
    """Mock storage service for testing"""

    def __init__(self):
        self.files = {}

    def set_file_content(self, bucket: str, key: str, content: bytes):
        """Set file content for testing"""
        self.files[f"{bucket}/{key}"] = content

    async def download_file(self, bucket: str, key: str) -> bytes:
        """Mock file download"""
        file_key = f"{bucket}/{key}"
        if file_key not in self.files:
            raise FileNotFoundError(f"File not found: {file_key}")
        return self.files[file_key]

    async def get_download_url(self, bucket: str, key: str) -> str:
        """Mock download URL generation"""
        return f"https://mock-storage.com/{bucket}/{key}"


class MockParser:
    """Mock parser for testing"""

    async def parse_file_to_text(self, file_content: bytes, filename: str) -> tuple[Dict[str, Any], str]:
        """Mock file parsing - returns simple text content"""
        # Simulate parsing different file types
        if filename.endswith(".txt"):
            return {}, file_content.decode("utf-8")
        elif filename.endswith(".pdf"):
            return {"pages": 1}, f"Parsed content from PDF: {filename}"
        elif filename.endswith(".docx"):
            return {"word_count": 100}, f"Parsed content from Word doc: {filename}"
        else:
            return {}, f"Generic parsed content from: {filename}"


class MockDatabase:
    """Mock database for testing"""

    def __init__(self):
        self.documents = {}
        self.update_calls = []

    def set_document(self, doc_id: str, document: Document):
        """Set a document for testing"""
        self.documents[doc_id] = document

    async def update_document(self, doc_id: str, updates: Dict[str, Any], auth: AuthContext) -> bool:
        """Mock document update"""
        self.update_calls.append({"doc_id": doc_id, "updates": updates, "auth": auth})

        # Simulate updating the document
        if doc_id in self.documents:
            doc = self.documents[doc_id]
            if "system_metadata" in updates:
                if not doc.system_metadata:
                    doc.system_metadata = {}
                doc.system_metadata.update(updates["system_metadata"])
            return True
        return False


class MockDocumentService:
    """Mock document service for testing"""

    def __init__(self, storage: MockStorage, parser: MockParser, db: MockDatabase):
        self.storage = storage
        self.parser = parser
        self.db = db


class TestMorphikGraphService(unittest.TestCase):
    """Test cases for MorphikGraphService document parsing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_storage = MockStorage()
        self.mock_parser = MockParser()
        self.mock_db = MockDatabase()
        self.mock_document_service = MockDocumentService(self.mock_storage, self.mock_parser, self.mock_db)

        # Create MorphikGraphService instance
        self.graph_service = MorphikGraphService(
            db=MagicMock(),
            embedding_model=MagicMock(),
            completion_model=MagicMock(),
            base_url="http://mock-graph-api.com",
            graph_api_key="mock-api-key",
        )

    def create_test_document(
        self,
        doc_id: str,
        filename: str,
        content: Optional[str] = None,
        bucket: str = "test-bucket",
        key: str = "test-key",
    ) -> Document:
        """Create a test document"""
        system_metadata = {
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "version": 1,
            "status": "processing",
        }
        if content is not None:
            system_metadata["content"] = content

        storage_info = {"bucket": bucket, "key": key}

        return Document(
            external_id=doc_id,
            filename=filename,
            content_type="text/plain",
            app_id="test-app",
            system_metadata=system_metadata,
            storage_info=storage_info,
        )

    async def test_prepare_document_content_already_parsed(self):
        """Test that documents with existing content are returned as-is"""
        # Create document with existing content
        doc = self.create_test_document(doc_id="doc-1", filename="test.txt", content="This document is already parsed")

        # Call the method
        result = await self.graph_service._prepare_document_content(doc, self.mock_document_service)

        # Should return existing content
        assert result == "This document is already parsed"

        # Database should not be updated
        assert len(self.mock_db.update_calls) == 0

    async def test_prepare_document_content_parse_text_file(self):
        """Test parsing a text file"""
        # Set up file content in mock storage
        file_content = b"This is the content of a test text file"
        self.mock_storage.set_file_content("test-bucket", "test.txt", file_content)

        # Create unparsed document
        doc = self.create_test_document(
            doc_id="doc-1", filename="test.txt", content="", bucket="test-bucket", key="test.txt"  # Empty content
        )
        self.mock_db.set_document("doc-1", doc)

        # Call the method
        result = await self.graph_service._prepare_document_content(doc, self.mock_document_service)

        # Should return parsed content
        assert result == "This is the content of a test text file"

        # Database should be updated
        assert len(self.mock_db.update_calls) == 1
        update_call = self.mock_db.update_calls[0]
        assert update_call["doc_id"] == "doc-1"
        assert update_call["updates"]["system_metadata"]["content"] == "This is the content of a test text file"

    async def test_prepare_document_content_parse_pdf_file(self):
        """Test parsing a PDF file"""
        # Set up file content in mock storage
        file_content = b"PDF file content bytes"
        self.mock_storage.set_file_content("test-bucket", "document.pdf", file_content)

        # Create unparsed document
        doc = self.create_test_document(
            doc_id="doc-2",
            filename="document.pdf",
            content=None,  # No content at all
            bucket="test-bucket",
            key="document.pdf",
        )
        self.mock_db.set_document("doc-2", doc)

        # Call the method
        result = await self.graph_service._prepare_document_content(doc, self.mock_document_service)

        # Should return parsed content
        assert result == "Parsed content from PDF: document.pdf"

        # Database should be updated
        assert len(self.mock_db.update_calls) == 1
        update_call = self.mock_db.update_calls[0]
        assert update_call["doc_id"] == "doc-2"
        assert update_call["updates"]["system_metadata"]["content"] == "Parsed content from PDF: document.pdf"

    async def test_prepare_document_content_missing_storage_info(self):
        """Test document with no storage info"""
        # Create document without storage info
        doc = Document(
            external_id="doc-3",
            filename="test.txt",
            content_type="text/plain",
            app_id="test-app",
            system_metadata={
                "content": "",
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
                "version": 1,
                "status": "processing",
            },
            storage_info={},  # Empty storage info
        )

        # Call the method
        result = await self.graph_service._prepare_document_content(doc, self.mock_document_service)

        # Should return empty string
        assert result == ""

        # Database should not be updated
        assert len(self.mock_db.update_calls) == 0

    async def test_prepare_document_content_storage_error(self):
        """Test handling of storage download errors"""
        # Create unparsed document (but don't set file content in storage)
        doc = self.create_test_document(
            doc_id="doc-4", filename="missing.txt", content="", bucket="test-bucket", key="missing.txt"
        )
        self.mock_db.set_document("doc-4", doc)

        # Call the method (should handle FileNotFoundError gracefully)
        result = await self.graph_service._prepare_document_content(doc, self.mock_document_service)

        # Should return empty string on error
        assert result == ""

        # Database should not be updated
        assert len(self.mock_db.update_calls) == 0

    async def test_prepare_document_content_text_cleaning(self):
        """Test that text cleaning is applied correctly"""
        # Set up file content with problematic characters
        file_content = b"Clean text\x00with\u0000null\x0Bchars and\x1Fcontrol chars"
        self.mock_storage.set_file_content("test-bucket", "dirty.txt", file_content)

        # Create unparsed document
        doc = self.create_test_document(
            doc_id="doc-5", filename="dirty.txt", content="", bucket="test-bucket", key="dirty.txt"
        )
        self.mock_db.set_document("doc-5", doc)

        # Call the method
        result = await self.graph_service._prepare_document_content(doc, self.mock_document_service)

        # Should have cleaned text (null bytes and non-printable chars removed)
        # The exact cleaning depends on the regex patterns
        assert "\x00" not in result
        assert "\u0000" not in result
        # Should still contain the clean parts
        assert "Clean text" in result
        assert "null" in result
        assert "control chars" in result

    async def test_prepare_document_content_auth_context_creation(self):
        """Test that proper auth context is created for database updates"""
        # Set up file content
        file_content = b"Test content for auth verification"
        self.mock_storage.set_file_content("test-bucket", "auth-test.txt", file_content)

        # Create unparsed document
        doc = self.create_test_document(
            doc_id="doc-6", filename="auth-test.txt", content="", bucket="test-bucket", key="auth-test.txt"
        )
        self.mock_db.set_document("doc-6", doc)

        # Call the method
        result = await self.graph_service._prepare_document_content(doc, self.mock_document_service)

        # Should return parsed content
        assert result == "Test content for auth verification"

        # Check that auth context was created correctly
        assert len(self.mock_db.update_calls) == 1
        update_call = self.mock_db.update_calls[0]
        auth_context = update_call["auth"]

        assert auth_context.entity_type == EntityType.DEVELOPER
        assert auth_context.entity_id == "graph_service"
        assert auth_context.app_id == "test-app"
        assert "write" in auth_context.permissions
        assert auth_context.user_id == "graph_service"


# Async test runner
async def run_async_tests():
    """Run all async tests"""
    test_instance = TestMorphikGraphService()
    test_instance.setUp()

    print("Running async tests for MorphikGraphService...")

    # Test cases
    test_cases = [
        ("test_prepare_document_content_already_parsed", test_instance.test_prepare_document_content_already_parsed),
        ("test_prepare_document_content_parse_text_file", test_instance.test_prepare_document_content_parse_text_file),
        ("test_prepare_document_content_parse_pdf_file", test_instance.test_prepare_document_content_parse_pdf_file),
        (
            "test_prepare_document_content_missing_storage_info",
            test_instance.test_prepare_document_content_missing_storage_info,
        ),
        ("test_prepare_document_content_storage_error", test_instance.test_prepare_document_content_storage_error),
        ("test_prepare_document_content_text_cleaning", test_instance.test_prepare_document_content_text_cleaning),
        (
            "test_prepare_document_content_auth_context_creation",
            test_instance.test_prepare_document_content_auth_context_creation,
        ),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in test_cases:
        try:
            # Reset for each test
            test_instance.setUp()
            await test_func()
            print(f"✅ {test_name} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} - FAILED: {e}")
            failed += 1

    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    # Run the async tests
    success = asyncio.run(run_async_tests())
    if not success:
        exit(1)

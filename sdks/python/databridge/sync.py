from io import BytesIO
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, BinaryIO
from urllib.parse import urlparse

import jwt
import requests

from .models import (
    Document,
    IngestTextRequest,
    ChunkResult,
    DocumentResult,
    CompletionResponse,
)


class DataBridge:
    """
    DataBridge client for document operations.

    Args:
        uri (str): DataBridge URI in the format "databridge://<owner_id>:<token>@<host>"
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        is_local (bool, optional): Whether connecting to local development server. Defaults to False.

    Examples:
        ```python
        with DataBridge("databridge://owner_id:token@api.databridge.ai") as db:
            # Ingest text
            doc = db.ingest_text(
                "Sample content",
                metadata={"category": "sample"}
            )

            # Query documents
            results = db.query("search query")
        ```
    """

    def __init__(self, uri: str, timeout: int = 30, is_local: bool = False):
        self._timeout = timeout
        self._session = requests.Session()
        if is_local:
            self._session.verify = False  # Disable SSL for localhost
        self._is_local = is_local
        self._setup_auth(uri)

    def _setup_auth(self, uri: str) -> None:
        """Setup authentication from URI"""
        parsed = urlparse(uri)
        if not parsed.netloc:
            raise ValueError("Invalid URI format")

        # Split host and auth parts
        auth, host = parsed.netloc.split("@")
        self._owner_id, self._auth_token = auth.split(":")

        # Set base URL
        self._base_url = f"{'http' if self._is_local else 'https'}://{host}"

        # Basic token validation
        jwt.decode(self._auth_token, options={"verify_signature": False})

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated HTTP request"""
        headers = {"Authorization": f"Bearer {self._auth_token}"}

        if not files:
            headers["Content-Type"] = "application/json"

        response = self._session.request(
            method,
            f"{self._base_url}/{endpoint.lstrip('/')}",
            json=data if not files else None,
            files=files,
            data=data if files else None,
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def ingest_text(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Ingest a text document into DataBridge.

        Args:
            content: Text content to ingest
            metadata: Optional metadata dictionary

        Returns:
            Document: Metadata of the ingested document

        Example:
            ```python
            doc = db.ingest_text(
                "Machine learning is fascinating...",
                metadata={
                    "title": "ML Introduction",
                    "category": "tech"
                }
            )
            ```
        """
        request = IngestTextRequest(content=content, metadata=metadata or {})

        response = self._request("POST", "ingest/text", request.model_dump())
        return Document(**response)

    def ingest_file(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Ingest a file document into DataBridge.

        Args:
            file: File to ingest (path string, bytes, file object, or Path)
            filename: Name of the file
            content_type: MIME type (optional, will be guessed if not provided)
            metadata: Optional metadata dictionary

        Returns:
            Document: Metadata of the ingested document

        Example:
            ```python
            # From file path
            doc = db.ingest_file(
                "document.pdf",
                filename="document.pdf",
                content_type="application/pdf",
                metadata={"department": "research"}
            )

            # From file object
            with open("document.pdf", "rb") as f:
                doc = db.ingest_file(f, "document.pdf")
            ```
        """
        # Handle different file input types
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            if not file_path.exists():
                raise ValueError(f"File not found: {file}")
            with open(file_path, "rb") as f:
                content = f.read()
                file_obj = BytesIO(content)
        elif isinstance(file, bytes):
            file_obj = BytesIO(file)
        else:
            file_obj = file

        try:
            # Prepare multipart form data
            files = {
                "file": (filename, file_obj, content_type or "application/octet-stream")
            }

            # Add metadata
            data = {"metadata": json.dumps(metadata or {})}

            response = self._request("POST", "ingest/file", data=data, files=files)
            return Document(**response)
        finally:
            # Close file if we opened it
            if isinstance(file, (str, Path)):
                file_obj.close()

    def retrieve_chunks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
    ) -> List[ChunkResult]:
        """
        Retrieve relevant chunks.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)

        Returns:
            List[ChunkResult]

        Example:
            ```python
            chunks = db.retrieve_chunks(
                "What are the key findings?",
                filters={"department": "research"}
            )
            ```
        """
        request = {"query": query, "filters": filters, "k": k, "min_score": min_score}

        response = self._request("POST", "search/chunks", request)
        return [ChunkResult(**r) for r in response]

    def retrieve_docs(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
    ) -> List[DocumentResult]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)

        Returns:
            List[DocumentResult]

        Example:
            ```python
            docs = db.retrieve_docs(
                "machine learning",
                k=5
            )
            ```
        """
        request = {"query": query, "filters": filters, "k": k, "min_score": min_score}

        response = self._request("POST", "retrieve/docs", request)
        return [DocumentResult(**r) for r in response]

    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> CompletionResponse:
        """
        Generate completion using relevant chunks as context.

        Args:
            query: Query text
            filters: Optional metadata filters
            k: Number of chunks to use as context (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            max_tokens: Maximum tokens in completion
            temperature: Model temperature

        Returns:
            CompletionResponse

        Example:
            ```python
            response = db.query(
                "What are the key findings about customer satisfaction?",
                filters={"department": "research"},
                temperature=0.7
            )
            print(response.completion)
            ```
        """
        request = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self._request("POST", "query", request)
        return CompletionResponse(**response)

    def list_documents(
        self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        List accessible documents.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters

        Returns:
            List[Document]: List of accessible documents

        Example:
            ```python
            # Get first page
            docs = db.list_documents(limit=10)

            # Get next page
            next_page = db.list_documents(skip=10, limit=10, filters={"department": "research"})
            ```
        """
        response = self._request(
            "GET", f"documents?skip={skip}&limit={limit}&filters={filters}"
        )
        return [Document(**doc) for doc in response]

    def get_document(self, document_id: str) -> Document:
        """
        Get document metadata by ID.

        Args:
            document_id: ID of the document

        Returns:
            Document: Document metadata

        Example:
            ```python
            doc = db.get_document("doc_123")
            print(f"Title: {doc.metadata.get('title')}")
            ```
        """
        response = self._request("GET", f"documents/{document_id}")
        return Document(**response)

    def close(self):
        """Close the HTTP session"""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

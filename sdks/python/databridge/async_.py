from io import BytesIO
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, BinaryIO
from urllib.parse import urlparse

import httpx
import jwt

from .models import (
    Document,
    IngestTextRequest,
    ChunkResult,
    DocumentResult,
    CompletionResponse,
)


class AsyncDataBridge:
    """
    DataBridge client for document operations.

    Args:
        uri (str, optional): DataBridge URI in format "databridge://<owner_id>:<token>@<host>".
            If not provided, connects to http://localhost:8000 without authentication.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        is_local (bool, optional): Whether to connect to a local server. Defaults to False.

    Examples:
        ```python
        # Without authentication
        async with AsyncDataBridge() as db:
            doc = await db.ingest_text("Sample content")

        # With authentication
        async with AsyncDataBridge("databridge://owner_id:token@api.databridge.ai") as db:
            doc = await db.ingest_text("Sample content")
        ```
    """

    def __init__(self, uri: Optional[str] = None, timeout: int = 30, is_local: bool = False):
        self._timeout = timeout
        self._client = (
            httpx.AsyncClient(timeout=timeout)
            if not is_local
            else httpx.AsyncClient(
                timeout=timeout,
                verify=False,  # Disable SSL for localhost
                http2=False,  # Force HTTP/1.1
            )
        )
        self._is_local = is_local

        if uri:
            self._setup_auth(uri)
        else:
            self._base_url = "http://localhost:8000"
            self._auth_token = None

    def _setup_auth(self, uri: str) -> None:
        """Setup authentication from URI"""
        parsed = urlparse(uri)
        if not parsed.netloc:
            raise ValueError("Invalid URI format")

        # Split host and auth parts
        auth, host = parsed.netloc.split("@")
        _, self._auth_token = auth.split(":")

        # Set base URL
        self._base_url = f"{'http' if self._is_local else 'https'}://{host}"

        # Basic token validation
        jwt.decode(self._auth_token, options={"verify_signature": False})

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request"""
        headers = {}
        if self._auth_token:  # Only add auth header if we have a token
            headers["Authorization"] = f"Bearer {self._auth_token}"

        if not files:
            headers["Content-Type"] = "application/json"

        response = await self._client.request(
            method,
            f"{self._base_url}/{endpoint.lstrip('/')}",
            json=data if not files else None,
            files=files,
            data=data if files else None,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def ingest_text(
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
            doc = await db.ingest_text(
                "Machine learning is fascinating...",
                metadata={
                    "title": "ML Introduction",
                    "category": "tech"
                }
            )
            ```
        """
        request = IngestTextRequest(content=content, metadata=metadata or {})

        response = await self._request("POST", "ingest/text", request.model_dump())
        return Document(**response)

    async def ingest_file(
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
            doc = await db.ingest_file(
                "document.pdf",
                filename="document.pdf",
                content_type="application/pdf",
                metadata={"department": "research"}
            )

            # From file object
            with open("document.pdf", "rb") as f:
                doc = await db.ingest_file(f, "document.pdf")
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
            files = {"file": (filename, file_obj, content_type or "application/octet-stream")}

            # Add metadata
            data = {"metadata": json.dumps(metadata or {})}

            response = await self._request("POST", "ingest/file", data=data, files=files)
            return Document(**response)
        finally:
            # Close file if we opened it
            if isinstance(file, (str, Path)):
                file_obj.close()

    async def retrieve_chunks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
    ) -> List[ChunkResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)

        Returns:
            List[ChunkResult]

        Example:
            ```python
            chunks = await db.retrieve_chunks(
                "What are the key findings?",
                filters={"department": "research"}
            )
            ```
        """
        request = {"query": query, "filters": filters, "k": k, "min_score": min_score}

        response = await self._request("POST", "retrieve/chunks", request)
        return [ChunkResult(**r) for r in response]

    async def retrieve_docs(
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
            docs = await db.retrieve_docs(
                "machine learning",
                k=5
            )
            ```
        """
        request = {"query": query, "filters": filters, "k": k, "min_score": min_score}

        response = await self._request("POST", "retrieve/docs", request)
        return [DocumentResult(**r) for r in response]

    async def query(
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
            response = await db.query(
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

        response = await self._request("POST", "query", request)
        return CompletionResponse(**response)

    async def list_documents(
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
            docs = await db.list_documents(limit=10)

            # Get next page
            next_page = await db.list_documents(skip=10, limit=10, filters={"department": "research"})
            ```
        """
        response = await self._request(
            "GET", f"documents?skip={skip}&limit={limit}&filters={filters}"
        )
        return [Document(**doc) for doc in response]

    async def get_document(self, document_id: str) -> Document:
        """
        Get document metadata by ID.

        Args:
            document_id: ID of the document

        Returns:
            Document: Document metadata

        Example:
            ```python
            doc = await db.get_document("doc_123")
            print(f"Title: {doc.metadata.get('title')}")
            ```
        """
        response = await self._request("GET", f"documents/{document_id}")
        return Document(**response)

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

from io import BytesIO
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, BinaryIO
from urllib.parse import urlparse

import httpx
import jwt
from PIL.Image import Image as PILImage

from .models import (
    Document,
    ChunkResult,
    DocumentResult,
    CompletionResponse,
    IngestTextRequest,
)
from .rules import Rule

# Type alias for rules
RuleOrDict = Union[Rule, Dict[str, Any]]


class AsyncCache:
    def __init__(self, db: "AsyncDataBridge", name: str):
        self._db = db
        self._name = name

    async def update(self) -> bool:
        response = await self._db._request("POST", f"cache/{self._name}/update")
        return response.get("success", False)

    async def add_docs(self, docs: List[str]) -> bool:
        response = await self._db._request("POST", f"cache/{self._name}/add_docs", {"docs": docs})
        return response.get("success", False)

    async def query(
        self, query: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> CompletionResponse:
        response = await self._db._request(
            "POST",
            f"cache/{self._name}/query",
            params={"query": query, "max_tokens": max_tokens, "temperature": temperature},
            data="",
        )
        return CompletionResponse(**response)


class FinalChunkResult:
    content: str | PILImage
    score: float
    document_id: str
    chunk_number: int
    metadata: Dict[str, Any]
    content_type: str
    filename: Optional[str]
    download_url: Optional[str]


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
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request"""
        headers = {}
        if self._auth_token:  # Only add auth header if we have a token
            headers["Authorization"] = f"Bearer {self._auth_token}"

        # Configure request data based on type
        if files:
            # Multipart form data for files
            request_data = {"files": files, "data": data}
            # Don't set Content-Type, let httpx handle it
        else:
            # JSON for everything else
            headers["Content-Type"] = "application/json"
            request_data = {"json": data}

        response = await self._client.request(
            method,
            f"{self._base_url}/{endpoint.lstrip('/')}",
            headers=headers,
            params=params,
            **request_data,
        )
        response.raise_for_status()
        return response.json()

    def _convert_rule(self, rule: RuleOrDict) -> Dict[str, Any]:
        """Convert a rule to a dictionary format"""
        if hasattr(rule, "to_dict"):
            return rule.to_dict()
        return rule

    async def ingest_text(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
    ) -> Document:
        """
        Ingest a text document into DataBridge.

        Args:
            content: Text content to ingest
            metadata: Optional metadata dictionary
            rules: Optional list of rules to apply during ingestion. Can be:
                  - MetadataExtractionRule: Extract metadata using a schema
                  - NaturalLanguageRule: Transform content using natural language
            use_colpali: Whether to use ColPali-style embedding model to ingest the text (slower, but significantly better retrieval accuracy for text and images)
        Returns:
            Document: Metadata of the ingested document

        Example:
            ```python
            from databridge.rules import MetadataExtractionRule, NaturalLanguageRule
            from pydantic import BaseModel

            class DocumentInfo(BaseModel):
                title: str
                author: str
                date: str

            doc = await db.ingest_text(
                "Machine learning is fascinating...",
                metadata={"category": "tech"},
                rules=[
                    # Extract metadata using schema
                    MetadataExtractionRule(schema=DocumentInfo),
                    # Transform content
                    NaturalLanguageRule(prompt="Shorten the content, use keywords")
                ]
            )
            ```
        """
        request = IngestTextRequest(
            content=content,
            metadata=metadata or {},
            rules=[self._convert_rule(r) for r in (rules or [])],
            use_colpali=use_colpali,
        )
        response = await self._request("POST", "ingest/text", data=request.model_dump())
        return Document(**response)

    async def ingest_file(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
    ) -> Document:
        """
        Ingest a file document into DataBridge.

        Args:
            file: File to ingest (path string, bytes, file object, or Path)
            filename: Name of the file
            metadata: Optional metadata dictionary
            rules: Optional list of rules to apply during ingestion. Can be:
                  - MetadataExtractionRule: Extract metadata using a schema
                  - NaturalLanguageRule: Transform content using natural language
            use_colpali: Whether to use ColPali-style embedding model to ingest the file (slower, but significantly better retrieval accuracy for text and images)
        Returns:
            Document: Metadata of the ingested document

        Example:
            ```python
            from databridge.rules import MetadataExtractionRule, NaturalLanguageRule
            from pydantic import BaseModel

            class DocumentInfo(BaseModel):
                title: str
                author: str
                department: str

            doc = await db.ingest_file(
                "document.pdf",
                filename="document.pdf",
                metadata={"category": "research"},
                rules=[
                    MetadataExtractionRule(schema=DocumentInfo),
                    NaturalLanguageRule(prompt="Extract key points only")
                ]
            )
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
            files = {"file": (filename, file_obj)}

            # Add metadata and rules
            data = {
                "metadata": json.dumps(metadata or {}),
                "rules": json.dumps([self._convert_rule(r) for r in (rules or [])]),
                "use_colpali": json.dumps(use_colpali),
            }

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
        use_colpali: bool = True,
    ) -> List[ChunkResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model to retrieve chunks (only works for documents ingested with `use_colpali=True`)
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
        params = {
            "query": query,
            "k": k,
            "min_score": min_score,
            "use_colpali": use_colpali,
        }
        if filters:
            params["filters"] = json.dumps(filters)

        response = await self._request("POST", "retrieve/chunks", params=params)
        return [ChunkResult(**r) for r in response]

    async def retrieve_docs(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
    ) -> List[DocumentResult]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model to retrieve documents (only works for documents ingested with `use_colpali=True`)
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
        params = {
            "query": query,
            "k": k,
            "min_score": min_score,
            "use_colpali": use_colpali,
        }
        if filters:
            params["filters"] = json.dumps(filters)

        response = await self._request("POST", "retrieve/docs", params=params)
        return [DocumentResult(**r) for r in response]

    async def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_colpali: bool = True,
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
            use_colpali: Whether to use ColPali-style embedding model to generate the completion (only works for documents ingested with `use_colpali=True`)
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
        params = {
            "query": query,
            "k": k,
            "min_score": min_score,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_colpali": use_colpali,
        }
        if filters:
            params["filters"] = json.dumps(filters)

        response = await self._request("POST", "query", params=params)
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

    async def create_cache(
        self,
        name: str,
        model: str,
        gguf_file: str,
        filters: Optional[Dict[str, Any]] = None,
        docs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new cache with specified configuration.

        Args:
            name: Name of the cache to create
            model: Name of the model to use (e.g. "llama2")
            gguf_file: Name of the GGUF file to use for the model
            filters: Optional metadata filters to determine which documents to include. These filters will be applied in addition to any specific docs provided.
            docs: Optional list of specific document IDs to include. These docs will be included in addition to any documents matching the filters.

        Returns:
            Dict[str, Any]: Created cache configuration

        Example:
            ```python
            # This will include both:
            # 1. Any documents with category="programming"
            # 2. The specific documents "doc1" and "doc2" (regardless of their category)
            cache = await db.create_cache(
                name="programming_cache",
                model="llama2",
                gguf_file="llama-2-7b-chat.Q4_K_M.gguf",
                filters={"category": "programming"},
                docs=["doc1", "doc2"]
            )
            ```
        """
        # Build query parameters for name, model and gguf_file
        params = {"name": name, "model": model, "gguf_file": gguf_file}

        # Build request body for filters and docs
        request = {"filters": filters, "docs": docs}

        response = await self._request("POST", "cache/create", request, params=params)
        return response

    async def get_cache(self, name: str) -> AsyncCache:
        """
        Get a cache by name.

        Args:
            name: Name of the cache to retrieve

        Returns:
            cache: A cache object that is used to interact with the cache.

        Example:
            ```python
            cache = await db.get_cache("programming_cache")
            ```
        """
        response = await self._request("GET", f"cache/{name}")
        if response.get("exists", False):
            return AsyncCache(self, name)
        raise ValueError(f"Cache '{name}' not found")

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

from io import BytesIO, IOBase
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, BinaryIO

import httpx
from PIL.Image import Image as PILImage

from .models import (
    Document,
    ChunkResult,
    DocumentResult,
    CompletionResponse,
    IngestTextRequest,
    ChunkSource,
    Graph,
    # Prompt override models
    EntityExtractionExample,
    EntityResolutionExample,
    EntityExtractionPromptOverride,
    EntityResolutionPromptOverride,
    QueryPromptOverride,
    GraphPromptOverrides,
    QueryPromptOverrides,
)
from .rules import Rule
from ._internal import _MorphikClientLogic, FinalChunkResult, RuleOrDict

logger = logging.getLogger(__name__)


class AsyncCache:
    def __init__(self, db: "AsyncMorphik", name: str):
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


class AsyncFolder:
    """
    A folder that allows operations to be scoped to a specific folder.

    Args:
        client: The AsyncMorphik client instance
        name: The name of the folder
    """

    def __init__(self, client: "AsyncMorphik", name: str):
        self._client = client
        self._name = name

    @property
    def name(self) -> str:
        """Returns the folder name."""
        return self._name

    def signin(self, end_user_id: str) -> "AsyncUserScope":
        """
        Returns an AsyncUserScope object scoped to this folder and the end user.

        Args:
            end_user_id: The ID of the end user

        Returns:
            AsyncUserScope: A user scope scoped to this folder and the end user
        """
        return AsyncUserScope(client=self._client, end_user_id=end_user_id, folder_name=self._name)

    async def ingest_text(
        self,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
    ) -> Document:
        """
        Ingest a text document into Morphik within this folder.

        Args:
            content: Text content to ingest
            filename: Optional file name
            metadata: Optional metadata dictionary
            rules: Optional list of rules to apply during ingestion
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            Document: Metadata of the ingested document
        """
        rules_list = [self._client._convert_rule(r) for r in (rules or [])]
        payload = self._client._logic._prepare_ingest_text_request(
            content, filename, metadata, rules_list, use_colpali, self._name, None
        )
        response = await self._client._request("POST", "ingest/text", data=payload)
        doc = self._client._logic._parse_document_response(response)
        doc._client = self._client
        return doc

    async def ingest_file(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
    ) -> Document:
        """
        Ingest a file document into Morphik within this folder.

        Args:
            file: File to ingest (path string, bytes, file object, or Path)
            filename: Name of the file
            metadata: Optional metadata dictionary
            rules: Optional list of rules to apply during ingestion
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            Document: Metadata of the ingested document
        """
        # Process file input
        file_obj, filename = self._client._logic._prepare_file_for_upload(file, filename)

        try:
            # Prepare multipart form data
            files = {"file": (filename, file_obj)}

            # Create form data
            form_data = self._client._logic._prepare_ingest_file_form_data(
                metadata, rules, self._name, None
            )

            response = await self._client._request(
                "POST",
                f"ingest/file?use_colpali={str(use_colpali).lower()}",
                data=form_data,
                files=files,
            )
            doc = self._client._logic._parse_document_response(response)
            doc._client = self._client
            return doc
        finally:
            # Close file if we opened it
            if isinstance(file, (str, Path)):
                file_obj.close()

    async def ingest_files(
        self,
        files: List[Union[str, bytes, BinaryIO, Path]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
        parallel: bool = True,
    ) -> List[Document]:
        """
        Ingest multiple files into Morphik within this folder.

        Args:
            files: List of files to ingest
            metadata: Optional metadata
            rules: Optional list of rules to apply
            use_colpali: Whether to use ColPali-style embedding
            parallel: Whether to process files in parallel

        Returns:
            List[Document]: List of ingested documents
        """
        # Convert files to format expected by API
        file_objects = self._client._logic._prepare_files_for_upload(files)

        try:
            # Prepare form data
            data = self._client._logic._prepare_ingest_files_form_data(
                metadata, rules, use_colpali, parallel, self._name, None
            )

            response = await self._client._request(
                "POST", "ingest/files", data=data, files=file_objects
            )

            if response.get("errors"):
                # Log errors but don't raise exception
                for error in response["errors"]:
                    logger.error(f"Failed to ingest {error['filename']}: {error['error']}")

            docs = [
                self._client._logic._parse_document_response(doc) for doc in response["documents"]
            ]
            for doc in docs:
                doc._client = self._client
            return docs
        finally:
            # Clean up file objects
            for _, (_, file_obj) in file_objects:
                if isinstance(file_obj, (IOBase, BytesIO)) and not file_obj.closed:
                    file_obj.close()

    async def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
        pattern: str = "*",
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
        parallel: bool = True,
    ) -> List[Document]:
        """
        Ingest all files in a directory into Morphik within this folder.

        Args:
            directory: Path to directory containing files to ingest
            recursive: Whether to recursively process subdirectories
            pattern: Optional glob pattern to filter files
            metadata: Optional metadata dictionary to apply to all files
            rules: Optional list of rules to apply
            use_colpali: Whether to use ColPali-style embedding
            parallel: Whether to process files in parallel

        Returns:
            List[Document]: List of ingested documents
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        # Collect all files matching pattern
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        # Filter out directories
        files = [f for f in files if f.is_file()]

        if not files:
            return []

        # Use ingest_files with collected paths
        return await self.ingest_files(
            files=files, metadata=metadata, rules=rules, use_colpali=use_colpali, parallel=parallel
        )

    async def retrieve_chunks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
    ) -> List[FinalChunkResult]:
        """
        Retrieve relevant chunks within this folder.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            List[FinalChunkResult]: List of relevant chunks
        """
        payload = self._client._logic._prepare_retrieve_chunks_request(
            query, filters, k, min_score, use_colpali, self._name, None
        )
        response = await self._client._request("POST", "retrieve/chunks", data=payload)
        return self._client._logic._parse_chunk_result_list_response(response)

    async def retrieve_docs(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
    ) -> List[DocumentResult]:
        """
        Retrieve relevant documents within this folder.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            List[DocumentResult]: List of relevant documents
        """
        payload = self._client._logic._prepare_retrieve_docs_request(
            query, filters, k, min_score, use_colpali, self._name, None
        )
        response = await self._client._request("POST", "retrieve/docs", data=payload)
        return self._client._logic._parse_document_result_list_response(response)

    async def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_colpali: bool = True,
        graph_name: Optional[str] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
        prompt_overrides: Optional[Union[QueryPromptOverrides, Dict[str, Any]]] = None,
    ) -> CompletionResponse:
        """
        Generate completion using relevant chunks as context within this folder.

        Args:
            query: Query text
            filters: Optional metadata filters
            k: Number of chunks to use as context (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            max_tokens: Maximum tokens in completion
            temperature: Model temperature
            use_colpali: Whether to use ColPali-style embedding model
            graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            hop_depth: Number of relationship hops to traverse in the graph (1-3)
            include_paths: Whether to include relationship paths in the response
            prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts

        Returns:
            CompletionResponse: Generated completion
        """
        payload = self._client._logic._prepare_query_request(
            query,
            filters,
            k,
            min_score,
            max_tokens,
            temperature,
            use_colpali,
            graph_name,
            hop_depth,
            include_paths,
            prompt_overrides,
            self._name,
            None,
        )
        response = await self._client._request("POST", "query", data=payload)
        return self._client._logic._parse_completion_response(response)

    async def list_documents(
        self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        List accessible documents within this folder.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters

        Returns:
            List[Document]: List of documents
        """
        params, data = self._client._logic._prepare_list_documents_request(
            skip, limit, filters, self._name, None
        )
        response = await self._client._request("POST", "documents", data=data, params=params)
        docs = self._client._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self._client
        return docs

    async def batch_get_documents(self, document_ids: List[str]) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation within this folder.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List[Document]: List of document metadata for found documents
        """
        request = self._client._logic._prepare_batch_get_documents_request(
            document_ids, self._name, None
        )
        response = await self._client._request("POST", "batch/documents", data=request)
        docs = self._client._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self._client
        return docs

    async def batch_get_chunks(
        self, sources: List[Union[ChunkSource, Dict[str, Any]]]
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation within this folder.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number

        Returns:
            List[FinalChunkResult]: List of chunk results
        """
        request = self._client._logic._prepare_batch_get_chunks_request(sources, self._name, None)
        response = await self._client._request("POST", "batch/chunks", data=request)
        return self._client._logic._parse_chunk_result_list_response(response)

    async def create_graph(
        self,
        name: str,
        filters: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
        prompt_overrides: Optional[Union[GraphPromptOverrides, Dict[str, Any]]] = None,
    ) -> Graph:
        """
        Create a graph from documents within this folder.

        Args:
            name: Name of the graph to create
            filters: Optional metadata filters to determine which documents to include
            documents: Optional list of specific document IDs to include
            prompt_overrides: Optional customizations for entity extraction and resolution prompts

        Returns:
            Graph: The created graph object
        """
        request = self._client._logic._prepare_create_graph_request(
            name, filters, documents, prompt_overrides, self._name, None
        )
        response = await self._client._request("POST", "graph/create", data=request)
        return self._client._logic._parse_graph_response(response)

    async def update_graph(
        self,
        name: str,
        additional_filters: Optional[Dict[str, Any]] = None,
        additional_documents: Optional[List[str]] = None,
        prompt_overrides: Optional[Union[GraphPromptOverrides, Dict[str, Any]]] = None,
    ) -> Graph:
        """
        Update an existing graph with new documents from this folder.

        Args:
            name: Name of the graph to update
            additional_filters: Optional additional metadata filters to determine which new documents to include
            additional_documents: Optional list of additional document IDs to include
            prompt_overrides: Optional customizations for entity extraction and resolution prompts

        Returns:
            Graph: The updated graph
        """
        request = self._client._logic._prepare_update_graph_request(
            name, additional_filters, additional_documents, prompt_overrides, self._name, None
        )
        response = await self._client._request("POST", f"graph/{name}/update", data=request)
        return self._client._logic._parse_graph_response(response)

    async def delete_document_by_filename(self, filename: str) -> Dict[str, str]:
        """
        Delete a document by its filename within this folder.

        Args:
            filename: Filename of the document to delete

        Returns:
            Dict[str, str]: Deletion status
        """
        # Get the document by filename with folder scope
        request = {"filename": filename, "folder_name": self._name}

        # First get the document ID
        response = await self._client._request(
            "GET", f"documents/filename/{filename}", params={"folder_name": self._name}
        )
        doc = self._client._logic._parse_document_response(response)

        # Then delete by ID
        return await self._client.delete_document(doc.external_id)


class AsyncUserScope:
    """
    A user scope that allows operations to be scoped to a specific end user and optionally a folder.

    Args:
        client: The AsyncMorphik client instance
        end_user_id: The ID of the end user
        folder_name: Optional folder name to further scope operations
    """

    def __init__(self, client: "AsyncMorphik", end_user_id: str, folder_name: Optional[str] = None):
        self._client = client
        self._end_user_id = end_user_id
        self._folder_name = folder_name

    @property
    def end_user_id(self) -> str:
        """Returns the end user ID."""
        return self._end_user_id

    @property
    def folder_name(self) -> Optional[str]:
        """Returns the folder name if any."""
        return self._folder_name

    async def ingest_text(
        self,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
    ) -> Document:
        """
        Ingest a text document into Morphik as this end user.

        Args:
            content: Text content to ingest
            filename: Optional file name
            metadata: Optional metadata dictionary
            rules: Optional list of rules to apply during ingestion
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            Document: Metadata of the ingested document
        """
        rules_list = [self._client._convert_rule(r) for r in (rules or [])]
        payload = self._client._logic._prepare_ingest_text_request(
            content,
            filename,
            metadata,
            rules_list,
            use_colpali,
            self._folder_name,
            self._end_user_id,
        )
        response = await self._client._request("POST", "ingest/text", data=payload)
        doc = self._client._logic._parse_document_response(response)
        doc._client = self._client
        return doc

    async def ingest_file(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
    ) -> Document:
        """
        Ingest a file document into Morphik as this end user.

        Args:
            file: File to ingest (path string, bytes, file object, or Path)
            filename: Name of the file
            metadata: Optional metadata dictionary
            rules: Optional list of rules to apply during ingestion
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            Document: Metadata of the ingested document
        """
        # Handle different file input types
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            if not file_path.exists():
                raise ValueError(f"File not found: {file}")
            filename = file_path.name if filename is None else filename
            with open(file_path, "rb") as f:
                content = f.read()
                file_obj = BytesIO(content)
        elif isinstance(file, bytes):
            if filename is None:
                raise ValueError("filename is required when ingesting bytes")
            file_obj = BytesIO(file)
        else:
            if filename is None:
                raise ValueError("filename is required when ingesting file object")
            file_obj = file

        try:
            # Prepare multipart form data
            files = {"file": (filename, file_obj)}

            # Add metadata and rules
            data = {
                "metadata": json.dumps(metadata or {}),
                "rules": json.dumps([self._client._convert_rule(r) for r in (rules or [])]),
                "end_user_id": self._end_user_id,  # Add end user ID here
            }

            # Add folder name if scoped to a folder
            if self._folder_name:
                data["folder_name"] = self._folder_name

            response = await self._client._request("POST", "ingest/file", data=data, files=files)
            doc = self._client._logic._parse_document_response(response)
            doc._client = self._client
            return doc
        finally:
            # Close file if we opened it
            if isinstance(file, (str, Path)):
                file_obj.close()

    async def ingest_files(
        self,
        files: List[Union[str, bytes, BinaryIO, Path]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
        parallel: bool = True,
    ) -> List[Document]:
        """
        Ingest multiple files into Morphik as this end user.

        Args:
            files: List of files to ingest
            metadata: Optional metadata
            rules: Optional list of rules to apply
            use_colpali: Whether to use ColPali-style embedding
            parallel: Whether to process files in parallel

        Returns:
            List[Document]: List of ingested documents
        """
        # Convert files to format expected by API
        file_objects = []
        for file in files:
            if isinstance(file, (str, Path)):
                path = Path(file)
                file_objects.append(("files", (path.name, open(path, "rb"))))
            elif isinstance(file, bytes):
                file_objects.append(("files", ("file.bin", file)))
            else:
                file_objects.append(("files", (getattr(file, "name", "file.bin"), file)))

        try:
            # Prepare request data
            # Convert rules appropriately
            if rules:
                if all(isinstance(r, list) for r in rules):
                    # List of lists - per-file rules
                    converted_rules = [
                        [self._client._convert_rule(r) for r in rule_list] for rule_list in rules
                    ]
                else:
                    # Flat list - shared rules for all files
                    converted_rules = [self._client._convert_rule(r) for r in rules]
            else:
                converted_rules = []

            data = {
                "metadata": json.dumps(metadata or {}),
                "rules": json.dumps(converted_rules),
                "use_colpali": str(use_colpali).lower() if use_colpali is not None else None,
                "parallel": str(parallel).lower(),
                "end_user_id": self._end_user_id,  # Add end user ID here
            }

            # Add folder name if scoped to a folder
            if self._folder_name:
                data["folder_name"] = self._folder_name

            response = await self._client._request(
                "POST", "ingest/files", data=data, files=file_objects
            )

            if response.get("errors"):
                # Log errors but don't raise exception
                for error in response["errors"]:
                    logger.error(f"Failed to ingest {error['filename']}: {error['error']}")

            docs = [
                self._client._logic._parse_document_response(doc) for doc in response["documents"]
            ]
            for doc in docs:
                doc._client = self._client
            return docs
        finally:
            # Clean up file objects
            for _, (_, file_obj) in file_objects:
                if isinstance(file_obj, (IOBase, BytesIO)) and not file_obj.closed:
                    file_obj.close()

    async def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
        pattern: str = "*",
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
        parallel: bool = True,
    ) -> List[Document]:
        """
        Ingest all files in a directory into Morphik as this end user.

        Args:
            directory: Path to directory containing files to ingest
            recursive: Whether to recursively process subdirectories
            pattern: Optional glob pattern to filter files
            metadata: Optional metadata dictionary to apply to all files
            rules: Optional list of rules to apply
            use_colpali: Whether to use ColPali-style embedding
            parallel: Whether to process files in parallel

        Returns:
            List[Document]: List of ingested documents
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        # Collect all files matching pattern
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        # Filter out directories
        files = [f for f in files if f.is_file()]

        if not files:
            return []

        # Use ingest_files with collected paths
        return await self.ingest_files(
            files=files, metadata=metadata, rules=rules, use_colpali=use_colpali, parallel=parallel
        )

    async def retrieve_chunks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
    ) -> List[FinalChunkResult]:
        """
        Retrieve relevant chunks as this end user.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            List[FinalChunkResult]: List of relevant chunks
        """
        payload = self._client._logic._prepare_retrieve_chunks_request(
            query, filters, k, min_score, use_colpali, self._folder_name, self._end_user_id
        )
        response = await self._client._request("POST", "retrieve/chunks", data=payload)
        return self._client._logic._parse_chunk_result_list_response(response)

    async def retrieve_docs(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
    ) -> List[DocumentResult]:
        """
        Retrieve relevant documents as this end user.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            List[DocumentResult]: List of relevant documents
        """
        payload = self._client._logic._prepare_retrieve_docs_request(
            query, filters, k, min_score, use_colpali, self._folder_name, self._end_user_id
        )
        response = await self._client._request("POST", "retrieve/docs", data=payload)
        return self._client._logic._parse_document_result_list_response(response)

    async def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_colpali: bool = True,
        graph_name: Optional[str] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
        prompt_overrides: Optional[Union[QueryPromptOverrides, Dict[str, Any]]] = None,
    ) -> CompletionResponse:
        """
        Generate completion using relevant chunks as context as this end user.

        Args:
            query: Query text
            filters: Optional metadata filters
            k: Number of chunks to use as context (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            max_tokens: Maximum tokens in completion
            temperature: Model temperature
            use_colpali: Whether to use ColPali-style embedding model
            graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            hop_depth: Number of relationship hops to traverse in the graph (1-3)
            include_paths: Whether to include relationship paths in the response
            prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts

        Returns:
            CompletionResponse: Generated completion
        """
        payload = self._client._logic._prepare_query_request(
            query,
            filters,
            k,
            min_score,
            max_tokens,
            temperature,
            use_colpali,
            graph_name,
            hop_depth,
            include_paths,
            prompt_overrides,
            self._folder_name,
            self._end_user_id,
        )
        response = await self._client._request("POST", "query", data=payload)
        return self._client._logic._parse_completion_response(response)

    async def list_documents(
        self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        List accessible documents for this end user.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters

        Returns:
            List[Document]: List of documents
        """
        params, data = self._client._logic._prepare_list_documents_request(
            skip, limit, filters, self._folder_name, self._end_user_id
        )
        response = await self._client._request("POST", "documents", data=data, params=params)
        docs = self._client._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self._client
        return docs

    async def batch_get_documents(self, document_ids: List[str]) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation for this end user.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List[Document]: List of document metadata for found documents
        """
        request = self._client._logic._prepare_batch_get_documents_request(
            document_ids, self._folder_name, self._end_user_id
        )
        response = await self._client._request("POST", "batch/documents", data=request)
        docs = self._client._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self._client
        return docs

    async def batch_get_chunks(
        self, sources: List[Union[ChunkSource, Dict[str, Any]]]
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation for this end user.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number

        Returns:
            List[FinalChunkResult]: List of chunk results
        """
        request = self._client._logic._prepare_batch_get_chunks_request(
            sources, self._folder_name, self._end_user_id
        )
        response = await self._client._request("POST", "batch/chunks", data=request)
        return self._client._logic._parse_chunk_result_list_response(response)

    async def create_graph(
        self,
        name: str,
        filters: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
        prompt_overrides: Optional[Union[GraphPromptOverrides, Dict[str, Any]]] = None,
    ) -> Graph:
        """
        Create a graph from documents for this end user.

        Args:
            name: Name of the graph to create
            filters: Optional metadata filters to determine which documents to include
            documents: Optional list of specific document IDs to include
            prompt_overrides: Optional customizations for entity extraction and resolution prompts

        Returns:
            Graph: The created graph object
        """
        request = self._client._logic._prepare_create_graph_request(
            name, filters, documents, prompt_overrides, self._folder_name, self._end_user_id
        )
        response = await self._client._request("POST", "graph/create", data=request)
        return self._client._logic._parse_graph_response(response)

    async def update_graph(
        self,
        name: str,
        additional_filters: Optional[Dict[str, Any]] = None,
        additional_documents: Optional[List[str]] = None,
        prompt_overrides: Optional[Union[GraphPromptOverrides, Dict[str, Any]]] = None,
    ) -> Graph:
        """
        Update an existing graph with new documents for this end user.

        Args:
            name: Name of the graph to update
            additional_filters: Optional additional metadata filters to determine which new documents to include
            additional_documents: Optional list of additional document IDs to include
            prompt_overrides: Optional customizations for entity extraction and resolution prompts

        Returns:
            Graph: The updated graph
        """
        request = self._client._logic._prepare_update_graph_request(
            name,
            additional_filters,
            additional_documents,
            prompt_overrides,
            self._folder_name,
            self._end_user_id,
        )
        response = await self._client._request("POST", f"graph/{name}/update", data=request)
        return self._client._logic._parse_graph_response(response)

    async def delete_document_by_filename(self, filename: str) -> Dict[str, str]:
        """
        Delete a document by its filename for this end user.

        Args:
            filename: Filename of the document to delete

        Returns:
            Dict[str, str]: Deletion status
        """
        # Build parameters for the filename lookup
        params = {"end_user_id": self._end_user_id}

        # Add folder name if scoped to a folder
        if self._folder_name:
            params["folder_name"] = self._folder_name

        # First get the document ID
        response = await self._client._request(
            "GET", f"documents/filename/{filename}", params=params
        )
        doc = self._client._logic._parse_document_response(response)

        # Then delete by ID
        return await self._client.delete_document(doc.external_id)


class AsyncMorphik:
    """
    Morphik client for document operations.

    Args:
        uri (str, optional): Morphik URI in format "morphik://<owner_id>:<token>@<host>".
            If not provided, connects to http://localhost:8000 without authentication.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        is_local (bool, optional): Whether to connect to a local server. Defaults to False.

    Examples:
        ```python
        # Without authentication
        async with AsyncMorphik() as db:
            doc = await db.ingest_text("Sample content")

        # With authentication
        async with AsyncMorphik("morphik://owner_id:token@api.morphik.ai") as db:
            doc = await db.ingest_text("Sample content")
        ```
    """

    def __init__(self, uri: Optional[str] = None, timeout: int = 30, is_local: bool = False):
        self._logic = _MorphikClientLogic(uri, timeout, is_local)
        self._client = httpx.AsyncClient(
            timeout=self._logic._timeout,
            verify=not self._logic._is_local,
            http2=False if self._logic._is_local else True,
        )

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request"""
        url = self._logic._get_url(endpoint)
        headers = self._logic._get_headers()
        if self._logic._auth_token:  # Only add auth header if we have a token
            headers["Authorization"] = f"Bearer {self._logic._auth_token}"

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
            url,
            headers=headers,
            params=params,
            **request_data,
        )
        response.raise_for_status()
        return response.json()

    def _convert_rule(self, rule: RuleOrDict) -> Dict[str, Any]:
        """Convert a rule to a dictionary format"""
        return self._logic._convert_rule(rule)

    def create_folder(self, name: str) -> AsyncFolder:
        """
        Create a folder to scope operations.

        Args:
            name: The name of the folder

        Returns:
            AsyncFolder: A folder object for scoped operations
        """
        return AsyncFolder(self, name)

    def get_folder(self, name: str) -> AsyncFolder:
        """
        Get a folder by name to scope operations.

        Args:
            name: The name of the folder

        Returns:
            AsyncFolder: A folder object for scoped operations
        """
        return AsyncFolder(self, name)

    def signin(self, end_user_id: str) -> AsyncUserScope:
        """
        Sign in as an end user to scope operations.

        Args:
            end_user_id: The ID of the end user

        Returns:
            AsyncUserScope: A user scope object for scoped operations
        """
        return AsyncUserScope(self, end_user_id)

    async def ingest_text(
        self,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
    ) -> Document:
        """
        Ingest a text document into Morphik.

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
            from morphik.rules import MetadataExtractionRule, NaturalLanguageRule
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
        rules_list = [self._convert_rule(r) for r in (rules or [])]
        payload = self._logic._prepare_ingest_text_request(
            content, filename, metadata, rules_list, use_colpali, None, None
        )
        response = await self._request("POST", "ingest/text", data=payload)
        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    async def ingest_file(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
    ) -> Document:
        """Ingest a file document into Morphik."""
        # Process file input
        file_obj, filename = self._logic._prepare_file_for_upload(file, filename)

        try:
            # Prepare multipart form data
            files = {"file": (filename, file_obj)}

            # Create form data
            form_data = self._logic._prepare_ingest_file_form_data(metadata, rules, None, None)

            response = await self._request(
                "POST",
                f"ingest/file?use_colpali={str(use_colpali).lower()}",
                data=form_data,
                files=files,
            )
            doc = self._logic._parse_document_response(response)
            doc._client = self
            return doc
        finally:
            # Close file if we opened it
            if isinstance(file, (str, Path)):
                file_obj.close()

    async def ingest_files(
        self,
        files: List[Union[str, bytes, BinaryIO, Path]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
        parallel: bool = True,
    ) -> List[Document]:
        """
        Ingest multiple files into Morphik.

        Args:
            files: List of files to ingest (path strings, bytes, file objects, or Paths)
            metadata: Optional metadata (single dict for all files or list of dicts)
            rules: Optional list of rules to apply
            use_colpali: Whether to use ColPali-style embedding
            parallel: Whether to process files in parallel

        Returns:
            List[Document]: List of successfully ingested documents

        Raises:
            ValueError: If metadata list length doesn't match files length
        """
        # Convert files to format expected by API
        file_objects = self._logic._prepare_files_for_upload(files)

        try:
            # Prepare form data
            data = self._logic._prepare_ingest_files_form_data(
                metadata, rules, use_colpali, parallel, None, None
            )

            response = await self._request("POST", "ingest/files", data=data, files=file_objects)

            if response.get("errors"):
                # Log errors but don't raise exception
                for error in response["errors"]:
                    logger.error(f"Failed to ingest {error['filename']}: {error['error']}")

            # Parse the documents from the response
            docs = [self._logic._parse_document_response(doc) for doc in response["documents"]]
            for doc in docs:
                doc._client = self
            return docs
        finally:
            # Clean up file objects
            for _, (_, file_obj) in file_objects:
                if isinstance(file_obj, (IOBase, BytesIO)) and not file_obj.closed:
                    file_obj.close()

    async def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
        pattern: str = "*",
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
        parallel: bool = True,
    ) -> List[Document]:
        """
        Ingest all files in a directory into Morphik.

        Args:
            directory: Path to directory containing files to ingest
            recursive: Whether to recursively process subdirectories
            pattern: Optional glob pattern to filter files (e.g. "*.pdf")
            metadata: Optional metadata dictionary to apply to all files
            rules: Optional list of rules to apply
            use_colpali: Whether to use ColPali-style embedding
            parallel: Whether to process files in parallel

        Returns:
            List[Document]: List of ingested documents

        Raises:
            ValueError: If directory not found
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        # Collect all files matching pattern
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        # Filter out directories
        files = [f for f in files if f.is_file()]

        if not files:
            return []

        # Use ingest_files with collected paths
        return await self.ingest_files(
            files=files, metadata=metadata, rules=rules, use_colpali=use_colpali, parallel=parallel
        )

    async def retrieve_chunks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
    ) -> List[FinalChunkResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model to retrieve chunks (only works for documents ingested with `use_colpali=True`)
        Returns:
            List[FinalChunkResult]

        Example:
            ```python
            chunks = await db.retrieve_chunks(
                "What are the key findings?",
                filters={"department": "research"}
            )
            ```
        """
        payload = self._logic._prepare_retrieve_chunks_request(
            query, filters, k, min_score, use_colpali, None, None
        )
        response = await self._request("POST", "retrieve/chunks", data=payload)
        return self._logic._parse_chunk_result_list_response(response)

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
        payload = self._logic._prepare_retrieve_docs_request(
            query, filters, k, min_score, use_colpali, None, None
        )
        response = await self._request("POST", "retrieve/docs", data=payload)
        return self._logic._parse_document_result_list_response(response)

    async def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_colpali: bool = True,
        graph_name: Optional[str] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
        prompt_overrides: Optional[Union[QueryPromptOverrides, Dict[str, Any]]] = None,
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
            graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            hop_depth: Number of relationship hops to traverse in the graph (1-3)
            include_paths: Whether to include relationship paths in the response
            prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts
                Either a QueryPromptOverrides object or a dictionary with the same structure
        Returns:
            CompletionResponse

        Example:
            ```python
            # Standard query
            response = await db.query(
                "What are the key findings about customer satisfaction?",
                filters={"department": "research"},
                temperature=0.7
            )

            # Knowledge graph enhanced query
            response = await db.query(
                "How does product X relate to customer segment Y?",
                graph_name="market_graph",
                hop_depth=2,
                include_paths=True
            )

            # With prompt customization
            from morphik.models import QueryPromptOverride, QueryPromptOverrides
            response = await db.query(
                "What are the key findings?",
                prompt_overrides=QueryPromptOverrides(
                    query=QueryPromptOverride(
                        prompt_template="Answer the question in a formal, academic tone: {question}"
                    )
                )
            )

            # Or using a dictionary
            response = await db.query(
                "What are the key findings?",
                prompt_overrides={
                    "query": {
                        "prompt_template": "Answer the question in a formal, academic tone: {question}"
                    }
                }
            )

            print(response.completion)

            # If include_paths=True, you can inspect the graph paths
            if response.metadata and "graph" in response.metadata:
                for path in response.metadata["graph"]["paths"]:
                    print(" -> ".join(path))
            ```
        """
        payload = self._logic._prepare_query_request(
            query,
            filters,
            k,
            min_score,
            max_tokens,
            temperature,
            use_colpali,
            graph_name,
            hop_depth,
            include_paths,
            prompt_overrides,
            None,
            None,
        )
        response = await self._request("POST", "query", data=payload)
        return self._logic._parse_completion_response(response)

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
        params, data = self._logic._prepare_list_documents_request(skip, limit, filters, None, None)
        response = await self._request("POST", "documents", data=data, params=params)
        docs = self._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self
        return docs

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
        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    async def get_document_by_filename(self, filename: str) -> Document:
        """
        Get document metadata by filename.
        If multiple documents have the same filename, returns the most recently updated one.

        Args:
            filename: Filename of the document to retrieve

        Returns:
            Document: Document metadata

        Example:
            ```python
            doc = await db.get_document_by_filename("report.pdf")
            print(f"Document ID: {doc.external_id}")
            ```
        """
        response = await self._request("GET", f"documents/filename/{filename}")
        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    async def update_document_with_text(
        self,
        document_id: str,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> Document:
        """
        Update a document with new text content using the specified strategy.

        Args:
            document_id: ID of the document to update
            content: The new content to add
            filename: Optional new filename for the document
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Add new content to an existing document
            updated_doc = await db.update_document_with_text(
                document_id="doc_123",
                content="This is additional content that will be appended to the document.",
                filename="updated_document.txt",
                metadata={"category": "updated"},
                update_strategy="add"
            )
            print(f"Document version: {updated_doc.system_metadata.get('version')}")
            ```
        """
        # Use the dedicated text update endpoint
        request = IngestTextRequest(
            content=content,
            filename=filename,
            metadata=metadata or {},
            rules=[self._convert_rule(r) for r in (rules or [])],
            use_colpali=use_colpali if use_colpali is not None else True,
        )

        params = {}
        if update_strategy != "add":
            params["update_strategy"] = update_strategy

        response = await self._request(
            "POST", f"documents/{document_id}/update_text", data=request.model_dump(), params=params
        )

        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    async def update_document_with_file(
        self,
        document_id: str,
        file: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> Document:
        """
        Update a document with content from a file using the specified strategy.

        Args:
            document_id: ID of the document to update
            file: File to add (path string, bytes, file object, or Path)
            filename: Name of the file
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Add content from a file to an existing document
            updated_doc = await db.update_document_with_file(
                document_id="doc_123",
                file="path/to/update.pdf",
                metadata={"status": "updated"},
                update_strategy="add"
            )
            print(f"Document version: {updated_doc.system_metadata.get('version')}")
            ```
        """
        # Handle different file input types
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            if not file_path.exists():
                raise ValueError(f"File not found: {file}")
            filename = file_path.name if filename is None else filename
            with open(file_path, "rb") as f:
                content = f.read()
                file_obj = BytesIO(content)
        elif isinstance(file, bytes):
            if filename is None:
                raise ValueError("filename is required when updating with bytes")
            file_obj = BytesIO(file)
        else:
            if filename is None:
                raise ValueError("filename is required when updating with file object")
            file_obj = file

        try:
            # Prepare multipart form data
            files = {"file": (filename, file_obj)}

            # Convert metadata and rules to JSON strings
            form_data = {
                "metadata": json.dumps(metadata or {}),
                "rules": json.dumps([self._convert_rule(r) for r in (rules or [])]),
                "update_strategy": update_strategy,
            }

            if use_colpali is not None:
                form_data["use_colpali"] = str(use_colpali).lower()

            # Use the dedicated file update endpoint
            response = await self._request(
                "POST", f"documents/{document_id}/update_file", data=form_data, files=files
            )

            doc = self._logic._parse_document_response(response)
            doc._client = self
            return doc
        finally:
            # Close file if we opened it
            if isinstance(file, (str, Path)):
                file_obj.close()

    async def update_document_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any],
    ) -> Document:
        """
        Update a document's metadata only.

        Args:
            document_id: ID of the document to update
            metadata: Metadata to update

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Update just the metadata of a document
            updated_doc = await db.update_document_metadata(
                document_id="doc_123",
                metadata={"status": "reviewed", "reviewer": "Jane Smith"}
            )
            print(f"Updated metadata: {updated_doc.metadata}")
            ```
        """
        # Use the dedicated metadata update endpoint
        response = await self._request(
            "POST", f"documents/{document_id}/update_metadata", data=metadata
        )
        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    async def update_document_by_filename_with_text(
        self,
        filename: str,
        content: str,
        new_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> Document:
        """
        Update a document identified by filename with new text content using the specified strategy.

        Args:
            filename: Filename of the document to update
            content: The new content to add
            new_filename: Optional new filename for the document
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Add new content to an existing document identified by filename
            updated_doc = await db.update_document_by_filename_with_text(
                filename="report.pdf",
                content="This is additional content that will be appended to the document.",
                new_filename="updated_report.pdf",
                metadata={"category": "updated"},
                update_strategy="add"
            )
            print(f"Document version: {updated_doc.system_metadata.get('version')}")
            ```
        """
        # First get the document by filename to obtain its ID
        doc = await self.get_document_by_filename(filename)

        # Then use the regular update_document_with_text endpoint with the document ID
        return await self.update_document_with_text(
            document_id=doc.external_id,
            content=content,
            filename=new_filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )

    async def update_document_by_filename_with_file(
        self,
        filename: str,
        file: Union[str, bytes, BinaryIO, Path],
        new_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> Document:
        """
        Update a document identified by filename with content from a file using the specified strategy.

        Args:
            filename: Filename of the document to update
            file: File to add (path string, bytes, file object, or Path)
            new_filename: Optional new filename for the document (defaults to the filename of the file)
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Add content from a file to an existing document identified by filename
            updated_doc = await db.update_document_by_filename_with_file(
                filename="report.pdf",
                file="path/to/update.pdf",
                metadata={"status": "updated"},
                update_strategy="add"
            )
            print(f"Document version: {updated_doc.system_metadata.get('version')}")
            ```
        """
        # First get the document by filename to obtain its ID
        doc = await self.get_document_by_filename(filename)

        # Then use the regular update_document_with_file endpoint with the document ID
        return await self.update_document_with_file(
            document_id=doc.external_id,
            file=file,
            filename=new_filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )

    async def update_document_by_filename_metadata(
        self,
        filename: str,
        metadata: Dict[str, Any],
        new_filename: Optional[str] = None,
    ) -> Document:
        """
        Update a document's metadata using filename to identify the document.

        Args:
            filename: Filename of the document to update
            metadata: Metadata to update
            new_filename: Optional new filename to assign to the document

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Update just the metadata of a document identified by filename
            updated_doc = await db.update_document_by_filename_metadata(
                filename="report.pdf",
                metadata={"status": "reviewed", "reviewer": "Jane Smith"},
                new_filename="reviewed_report.pdf"  # Optional: rename the file
            )
            print(f"Updated metadata: {updated_doc.metadata}")
            ```
        """
        # First get the document by filename to obtain its ID
        doc = await self.get_document_by_filename(filename)

        # Update the metadata
        result = await self.update_document_metadata(
            document_id=doc.external_id,
            metadata=metadata,
        )

        # If new_filename is provided, update the filename as well
        if new_filename:
            # Create a request that retains the just-updated metadata but also changes filename
            combined_metadata = result.metadata.copy()

            # Update the document again with filename change and the same metadata
            response = await self._request(
                "POST",
                f"documents/{doc.external_id}/update_text",
                data={
                    "content": "",
                    "filename": new_filename,
                    "metadata": combined_metadata,
                    "rules": [],
                },
            )
            result = self._logic._parse_document_response(response)
            result._client = self

        return result

    async def batch_get_documents(self, document_ids: List[str]) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List[Document]: List of document metadata for found documents

        Example:
            ```python
            docs = await db.batch_get_documents(["doc_123", "doc_456", "doc_789"])
            for doc in docs:
                print(f"Document {doc.external_id}: {doc.metadata.get('title')}")
            ```
        """
        request = self._logic._prepare_batch_get_documents_request(document_ids, None, None)
        response = await self._request("POST", "batch/documents", data=request)
        docs = self._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self
        return docs

    async def batch_get_chunks(
        self, sources: List[Union[ChunkSource, Dict[str, Any]]]
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number

        Returns:
            List[FinalChunkResult]: List of chunk results

        Example:
            ```python
            # Using dictionaries
            sources = [
                {"document_id": "doc_123", "chunk_number": 0},
                {"document_id": "doc_456", "chunk_number": 2}
            ]

            # Or using ChunkSource objects
            from morphik.models import ChunkSource
            sources = [
                ChunkSource(document_id="doc_123", chunk_number=0),
                ChunkSource(document_id="doc_456", chunk_number=2)
            ]

            chunks = await db.batch_get_chunks(sources)
            for chunk in chunks:
                print(f"Chunk from {chunk.document_id}, number {chunk.chunk_number}: {chunk.content[:50]}...")
            ```
        """
        request = self._logic._prepare_batch_get_chunks_request(sources, None, None)
        response = await self._request("POST", "batch/chunks", data=request)
        return self._logic._parse_chunk_result_list_response(response)

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

    async def create_graph(
        self,
        name: str,
        filters: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
        prompt_overrides: Optional[Union[GraphPromptOverrides, Dict[str, Any]]] = None,
    ) -> Graph:
        """
        Create a graph from documents.

        This method extracts entities and relationships from documents
        matching the specified filters or document IDs and creates a graph.

        Args:
            name: Name of the graph to create
            filters: Optional metadata filters to determine which documents to include
            documents: Optional list of specific document IDs to include
            prompt_overrides: Optional customizations for entity extraction and resolution prompts
                Either a GraphPromptOverrides object or a dictionary with the same structure

        Returns:
            Graph: The created graph object

        Example:
            ```python
            # Create a graph from documents with category="research"
            graph = await db.create_graph(
                name="research_graph",
                filters={"category": "research"}
            )

            # Create a graph from specific documents
            graph = await db.create_graph(
                name="custom_graph",
                documents=["doc1", "doc2", "doc3"]
            )

            # With custom entity extraction examples
            from morphik.models import EntityExtractionPromptOverride, EntityExtractionExample, GraphPromptOverrides
            graph = await db.create_graph(
                name="medical_graph",
                filters={"category": "medical"},
                prompt_overrides=GraphPromptOverrides(
                    entity_extraction=EntityExtractionPromptOverride(
                        examples=[
                            EntityExtractionExample(label="Insulin", type="MEDICATION"),
                            EntityExtractionExample(label="Diabetes", type="CONDITION")
                        ]
                    )
                )
            )
            ```
        """
        request = self._logic._prepare_create_graph_request(
            name, filters, documents, prompt_overrides, None, None
        )
        response = await self._request("POST", "graph/create", data=request)
        return self._logic._parse_graph_response(response)

    async def get_graph(self, name: str) -> Graph:
        """
        Get a graph by name.

        Args:
            name: Name of the graph to retrieve

        Returns:
            Graph: The requested graph object

        Example:
            ```python
            # Get a graph by name
            graph = await db.get_graph("finance_graph")
            print(f"Graph has {len(graph.entities)} entities and {len(graph.relationships)} relationships")
            ```
        """
        response = await self._request("GET", f"graph/{name}")
        return self._logic._parse_graph_response(response)

    async def list_graphs(self) -> List[Graph]:
        """
        List all graphs the user has access to.

        Returns:
            List[Graph]: List of graph objects

        Example:
            ```python
            # List all accessible graphs
            graphs = await db.list_graphs()
            for graph in graphs:
                print(f"Graph: {graph.name}, Entities: {len(graph.entities)}")
            ```
        """
        response = await self._request("GET", "graphs")
        return self._logic._parse_graph_list_response(response)

    async def update_graph(
        self,
        name: str,
        additional_filters: Optional[Dict[str, Any]] = None,
        additional_documents: Optional[List[str]] = None,
        prompt_overrides: Optional[Union[GraphPromptOverrides, Dict[str, Any]]] = None,
    ) -> Graph:
        """
        Update an existing graph with new documents.

        This method processes additional documents matching the original or new filters,
        extracts entities and relationships, and updates the graph with new information.

        Args:
            name: Name of the graph to update
            additional_filters: Optional additional metadata filters to determine which new documents to include
            additional_documents: Optional list of additional document IDs to include
            prompt_overrides: Optional customizations for entity extraction and resolution prompts
                Either a GraphPromptOverrides object or a dictionary with the same structure

        Returns:
            Graph: The updated graph

        Example:
            ```python
            # Update a graph with new documents
            updated_graph = await db.update_graph(
                name="research_graph",
                additional_filters={"category": "new_research"},
                additional_documents=["doc4", "doc5"]
            )
            print(f"Graph now has {len(updated_graph.entities)} entities")

            # With entity resolution examples
            from morphik.models import EntityResolutionPromptOverride, EntityResolutionExample, GraphPromptOverrides
            updated_graph = await db.update_graph(
                name="research_graph",
                additional_documents=["doc4"],
                prompt_overrides=GraphPromptOverrides(
                    entity_resolution=EntityResolutionPromptOverride(
                        examples=[
                            EntityResolutionExample(
                                canonical="Machine Learning",
                                variants=["ML", "machine learning", "AI/ML"]
                            )
                        ]
                    )
                )
            )
            ```
        """
        request = self._logic._prepare_update_graph_request(
            name, additional_filters, additional_documents, prompt_overrides, None, None
        )
        response = await self._request("POST", f"graph/{name}/update", data=request)
        return self._logic._parse_graph_response(response)

    async def delete_document(self, document_id: str) -> Dict[str, str]:
        """
        Delete a document and all its associated data.

        This method deletes a document and all its associated data, including:
        - Document metadata
        - Document content in storage
        - Document chunks and embeddings in vector store

        Args:
            document_id: ID of the document to delete

        Returns:
            Dict[str, str]: Deletion status

        Example:
            ```python
            # Delete a document
            result = await db.delete_document("doc_123")
            print(result["message"])  # Document doc_123 deleted successfully
            ```
        """
        response = await self._request("DELETE", f"documents/{document_id}")
        return response

    async def delete_document_by_filename(self, filename: str) -> Dict[str, str]:
        """
        Delete a document by its filename.

        This is a convenience method that first retrieves the document ID by filename
        and then deletes the document by ID.

        Args:
            filename: Filename of the document to delete

        Returns:
            Dict[str, str]: Deletion status

        Example:
            ```python
            # Delete a document by filename
            result = await db.delete_document_by_filename("report.pdf")
            print(result["message"])
            ```
        """
        # First get the document by filename to obtain its ID
        doc = await self.get_document_by_filename(filename)

        # Then delete the document by ID
        return await self.delete_document(doc.external_id)

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

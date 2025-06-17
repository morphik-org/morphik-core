import json
import logging
from datetime import datetime
from io import BytesIO, IOBase
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Type, Union

import httpx
from pydantic import BaseModel

from ._internal import FinalChunkResult, RuleOrDict, _MorphikClientLogic
from .models import CompletionResponse  # Prompt override models
from .models import (
    ChunkSource,
    Document,
    DocumentResult,
    FolderInfo,
    Graph,
    GraphPromptOverrides,
    IngestTextRequest,
    QueryPromptOverrides,
)

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
        folder_id: Optional folder ID (if already known)
    """

    def __init__(self, client: "AsyncMorphik", name: str, folder_id: Optional[str] = None):
        self._client = client
        self._name = name
        self._id = folder_id

    @property
    def name(self) -> str:
        """Returns the folder name."""
        return self._name

    @property
    def id(self) -> Optional[str]:
        """Returns the folder ID if available."""
        return self._id

    async def get_info(self) -> Dict[str, Any]:
        """
        Get detailed information about this folder.

        Returns:
            Dict[str, Any]: Detailed folder information
        """
        if not self._id:
            # If we don't have the ID, find the folder by name first
            folders = await self._client.list_folders()
            for folder in folders:
                if folder.name == self._name:
                    self._id = folder.id
                    break
            if not self._id:
                raise ValueError(f"Folder '{self._name}' not found")

        return await self._client._request("GET", f"folders/{self._id}")

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
                metadata, rules, self._name, None, use_colpali
            )

            response = await self._client._request(
                "POST",
                "ingest/file",
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
                "POST",
                "ingest/files",
                data=data,
                files=file_objects,
            )

            if response.get("errors"):
                # Log errors but don't raise exception
                for error in response["errors"]:
                    logger.error(f"Failed to ingest {error['filename']}: {error['error']}")

            docs = [self._client._logic._parse_document_response(doc) for doc in response["documents"]]
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
        additional_folders: Optional[List[str]] = None,
    ) -> List[FinalChunkResult]:
        """
        Retrieve relevant chunks within this folder.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model
            additional_folders: Optional list of additional folder names to further scope operations

        Returns:
            List[FinalChunkResult]: List of relevant chunks
        """
        effective_folder = self._merge_folders(additional_folders)
        payload = self._client._logic._prepare_retrieve_chunks_request(
            query, filters, k, min_score, use_colpali, effective_folder, None
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
        additional_folders: Optional[List[str]] = None,
    ) -> List[DocumentResult]:
        """
        Retrieve relevant documents within this folder.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model
            additional_folders: Optional list of additional folder names to further scope operations

        Returns:
            List[DocumentResult]: List of relevant documents
        """
        effective_folder = self._merge_folders(additional_folders)
        payload = self._client._logic._prepare_retrieve_docs_request(
            query, filters, k, min_score, use_colpali, effective_folder, None
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
        additional_folders: Optional[List[str]] = None,
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        chat_id: Optional[str] = None,
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
            schema: Optional schema for structured output
            additional_folders: Optional list of additional folder names to further scope operations

        Returns:
            CompletionResponse: Generated completion or structured output
        """
        effective_folder = self._merge_folders(additional_folders)
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
            effective_folder,
            None,
            chat_id,
            schema,
        )

        # Add schema to payload if provided
        if schema:
            # If schema is a Pydantic model class, we need to serialize it to a schema dict
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                payload["schema"] = schema.model_json_schema()
            else:
                payload["schema"] = schema

            # Add a hint to the query to return in JSON format
            payload["query"] = f"{payload['query']}\nReturn the answer in JSON format according to the required schema."

        response = await self._client._request("POST", "query", data=payload)
        return self._client._logic._parse_completion_response(response)

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        additional_folders: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        List accessible documents within this folder.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters
            additional_folders: Optional list of additional folder names to further scope operations

        Returns:
            List[Document]: List of documents
        """
        effective_folder = self._merge_folders(additional_folders)
        params, data = self._client._logic._prepare_list_documents_request(skip, limit, filters, effective_folder, None)
        response = await self._client._request("POST", "documents", data=data, params=params)
        docs = self._client._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self._client
        return docs

    async def batch_get_documents(
        self, document_ids: List[str], additional_folders: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation within this folder.

        Args:
            document_ids: List of document IDs to retrieve
            additional_folders: Optional list of additional folder names to further scope operations

        Returns:
            List[Document]: List of document metadata for found documents
        """
        merged = self._merge_folders(additional_folders)
        request = {"document_ids": document_ids, "folder_name": merged}
        response = await self._client._request("POST", "batch/documents", data=request)
        docs = self._client._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self._client
        return docs

    async def batch_get_chunks(
        self,
        sources: List[Union[ChunkSource, Dict[str, Any]]],
        additional_folders: Optional[List[str]] = None,
        use_colpali: bool = True,
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation within this folder.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number
            additional_folders: Optional list of additional folder names to further scope operations
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            List[FinalChunkResult]: List of chunk results
        """
        merged = self._merge_folders(additional_folders)
        request = self._client._logic._prepare_batch_get_chunks_request(sources, merged, None, use_colpali)
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
        graph = self._logic._parse_graph_response(response)
        graph._client = self  # Attach AsyncMorphik client for polling helpers
        return graph

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
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

    async def delete_document_by_filename(self, filename: str) -> Dict[str, str]:
        """
        Delete a document by its filename within this folder.

        Args:
            filename: Filename of the document to delete

        Returns:
            Dict[str, str]: Deletion status
        """
        # First get the document ID
        response = await self._client._request(
            "GET", f"documents/filename/{filename}", params={"folder_name": self._name}
        )
        doc = self._client._logic._parse_document_response(response)

        # Then delete by ID
        return await self._client.delete_document(doc.external_id)

    # Helper --------------------------------------------------------------
    def _merge_folders(self, additional_folders: Optional[List[str]] = None) -> Union[str, List[str]]:
        """Return the effective folder scope for this folder instance.

        If *additional_folders* is provided it will be combined with the scoped
        folder (*self._name*) and returned as a list.  Otherwise just
        *self._name* is returned so the API keeps backward-compatibility with
        accepting a single string."""
        if not additional_folders:
            return self._name
        return [self._name] + additional_folders


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

            # Add metadata, rules and scoping information
            data = {
                "metadata": json.dumps(metadata or {}),
                "rules": json.dumps([self._client._convert_rule(r) for r in (rules or [])]),
                "end_user_id": self._end_user_id,
                "use_colpali": str(use_colpali).lower(),
            }

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
                    converted_rules = [[self._client._convert_rule(r) for r in rule_list] for rule_list in rules]
                else:
                    # Flat list - shared rules for all files
                    converted_rules = [self._client._convert_rule(r) for r in rules]
            else:
                converted_rules = []

            data = {
                "metadata": json.dumps(metadata or {}),
                "rules": json.dumps(converted_rules),
                "parallel": str(parallel).lower(),
                "end_user_id": self._end_user_id,
                "use_colpali": str(use_colpali).lower(),
            }

            # Add folder name if scoped to a folder
            if self._folder_name:
                data["folder_name"] = self._folder_name

            response = await self._client._request(
                "POST",
                "ingest/files",
                data=data,
                files=file_objects,
            )

            if response.get("errors"):
                # Log errors but don't raise exception
                for error in response["errors"]:
                    logger.error(f"Failed to ingest {error['filename']}: {error['error']}")

            docs = [self._client._logic._parse_document_response(doc) for doc in response["documents"]]
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
        additional_folders: Optional[List[str]] = None,
    ) -> List[FinalChunkResult]:
        """
        Retrieve relevant chunks as this end user.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model
            additional_folders: Optional list of additional folder names to further scope operations

        Returns:
            List[FinalChunkResult]: List of relevant chunks
        """
        effective_folder = self._merge_folders(additional_folders)
        payload = self._client._logic._prepare_retrieve_chunks_request(
            query, filters, k, min_score, use_colpali, effective_folder, self._end_user_id
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
        additional_folders: Optional[List[str]] = None,
    ) -> List[DocumentResult]:
        """
        Retrieve relevant documents as this end user.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model
            additional_folders: Optional list of additional folder names to further scope operations

        Returns:
            List[DocumentResult]: List of relevant documents
        """
        effective_folder = self._merge_folders(additional_folders)
        payload = self._client._logic._prepare_retrieve_docs_request(
            query, filters, k, min_score, use_colpali, effective_folder, self._end_user_id
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
        additional_folders: Optional[List[str]] = None,
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        chat_id: Optional[str] = None,
    ) -> CompletionResponse:
        """
        Generate completion using relevant chunks as context, scoped to the end user.

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
            schema: Optional schema for structured output
            additional_folders: Optional list of additional folder names to further scope operations

        Returns:
            CompletionResponse: Generated completion or structured output
        """
        effective_folder = self._merge_folders(additional_folders)
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
            effective_folder,
            self._end_user_id,
            chat_id,
            schema,
        )

        # Add schema to payload if provided
        if schema:
            # If schema is a Pydantic model class, we need to serialize it to a schema dict
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                payload["schema"] = schema.model_json_schema()
            else:
                payload["schema"] = schema

            # Add a hint to the query to return in JSON format
            payload["query"] = f"{payload['query']}\nReturn the answer in JSON format according to the required schema."

        response = await self._client._request("POST", "query", data=payload)
        return self._client._logic._parse_completion_response(response)

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
    ) -> List[Document]:
        """
        List accessible documents for this end user.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters
            folder_name: Optional folder name (or list of names) to scope the request

        Returns:
            List[Document]: List of documents
        """
        params, data = self._client._logic._prepare_list_documents_request(
            skip, limit, filters, folder_name, self._end_user_id
        )
        response = await self._client._request("POST", "documents", data=data, params=params)
        docs = self._client._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self._client
        return docs

    async def batch_get_documents(
        self, document_ids: List[str], folder_name: Optional[Union[str, List[str]]] = None
    ) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation for this end user.

        Args:
            document_ids: List of document IDs to retrieve
            folder_name: Optional folder name (or list of names) to scope the request

        Returns:
            List[Document]: List of document metadata for found documents
        """
        # API expects a dict with document_ids key
        request = {"document_ids": document_ids}
        if self._end_user_id:
            request["end_user_id"] = self._end_user_id
        if self._folder_name:
            request["folder_name"] = self._folder_name
        response = await self._client._request("POST", "batch/documents", data=request)
        docs = self._client._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self._client
        return docs

    async def batch_get_chunks(
        self,
        sources: List[Union[ChunkSource, Dict[str, Any]]],
        folder_name: Optional[Union[str, List[str]]] = None,
        use_colpali: bool = True,
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation for this end user.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number
            folder_name: Optional folder name (or list of names) to scope the request
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            List[FinalChunkResult]: List of chunk results
        """
        request = self._client._logic._prepare_batch_get_chunks_request(
            sources, self._folder_name, self._end_user_id, use_colpali
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
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

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
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

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
        response = await self._client._request("GET", f"documents/filename/{filename}", params=params)
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
            # When uploading files, we need to make sure not to set Content-Type
            # Remove Content-Type if it exists - httpx will set the correct multipart boundary
            if "Content-Type" in headers:
                del headers["Content-Type"]

            # For file uploads with form data, use form data (not json)
            request_data = {"files": files}
            if data:
                request_data["data"] = data
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

    async def create_folder(self, name: str, description: Optional[str] = None) -> AsyncFolder:
        """
        Create a folder to scope operations.

        Args:
            name: The name of the folder
            description: Optional description for the folder

        Returns:
            AsyncFolder: A folder object ready for scoped operations
        """
        payload = {"name": name}
        if description:
            payload["description"] = description

        response = await self._request("POST", "folders", data=payload)
        folder_info = FolderInfo(**response)

        # Return a usable AsyncFolder object with the ID from the response
        return AsyncFolder(self, name, folder_id=folder_info.id)

    def get_folder_by_name(self, name: str) -> AsyncFolder:
        """
        Get a folder by name to scope operations.

        Args:
            name: The name of the folder

        Returns:
            AsyncFolder: A folder object for scoped operations
        """
        return AsyncFolder(self, name)

    async def get_folder(self, folder_id: str) -> AsyncFolder:
        """
        Get a folder by ID.

        Args:
            folder_id: ID of the folder

        Returns:
            AsyncFolder: A folder object for scoped operations
        """
        response = await self._request("GET", f"folders/{folder_id}")
        return AsyncFolder(self, response["name"], folder_id)

    async def list_folders(self) -> List[AsyncFolder]:
        """
        List all folders the user has access to as AsyncFolder objects.

        Returns:
            List[AsyncFolder]: List of AsyncFolder objects ready for operations
        """
        response = await self._request("GET", "folders")
        return [AsyncFolder(self, folder["name"], folder["id"]) for folder in response]

    async def add_document_to_folder(self, folder_id: str, document_id: str) -> Dict[str, str]:
        """
        Add a document to a folder.

        Args:
            folder_id: ID of the folder
            document_id: ID of the document

        Returns:
            Dict[str, str]: Success status
        """
        response = await self._request("POST", f"folders/{folder_id}/documents/{document_id}")
        return response

    async def remove_document_from_folder(self, folder_id: str, document_id: str) -> Dict[str, str]:
        """
        Remove a document from a folder.

        Args:
            folder_id: ID of the folder
            document_id: ID of the document

        Returns:
            Dict[str, str]: Success status
        """
        response = await self._request("DELETE", f"folders/{folder_id}/documents/{document_id}")
        return response

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
            use_colpali: Whether to use ColPali-style embedding model to ingest the text
                (slower, but significantly better retrieval accuracy for text and images)
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
            form_data = self._logic._prepare_ingest_file_form_data(metadata, rules, None, None, use_colpali)

            response = await self._request(
                "POST",
                "ingest/file",
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
            data = self._logic._prepare_ingest_files_form_data(metadata, rules, use_colpali, parallel, None, None)

            response = await self._request(
                "POST",
                "ingest/files",
                data=data,
                files=file_objects,
            )

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
        folder_name: Optional[Union[str, List[str]]] = None,
    ) -> List[FinalChunkResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model to retrieve chunks
                (only works for documents ingested with `use_colpali=True`)
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
        effective_folder = folder_name if folder_name is not None else None
        payload = self._logic._prepare_retrieve_chunks_request(
            query, filters, k, min_score, use_colpali, effective_folder, None
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
        folder_name: Optional[Union[str, List[str]]] = None,
    ) -> List[DocumentResult]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model to retrieve documents
                (only works for documents ingested with `use_colpali=True`)
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
        effective_folder = folder_name if folder_name is not None else None
        payload = self._logic._prepare_retrieve_docs_request(
            query, filters, k, min_score, use_colpali, effective_folder, None
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
        folder_name: Optional[Union[str, List[str]]] = None,
        chat_id: Optional[str] = None,
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
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
            use_colpali: Whether to use ColPali-style embedding model to generate the completion
                (only works for documents ingested with `use_colpali=True`)
            graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            hop_depth: Number of relationship hops to traverse in the graph (1-3)
            include_paths: Whether to include relationship paths in the response
            prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts
                Either a QueryPromptOverrides object or a dictionary with the same structure
            schema: Optional schema for structured output, can be a Pydantic model or a JSON schema dict
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

            # Using structured output with a Pydantic model
            from pydantic import BaseModel

            class ResearchFindings(BaseModel):
                main_finding: str
                supporting_evidence: List[str]
                limitations: List[str]

            response = await db.query(
                "Summarize the key research findings from these documents",
                schema=ResearchFindings
            )

            # Access structured output
            if response.structured_output:
                findings = response.structured_output
                print(f"Main finding: {findings.main_finding}")
                print("Supporting evidence:")
                for evidence in findings.supporting_evidence:
                    print(f"- {evidence}")
            ```
        """
        effective_folder = folder_name if folder_name is not None else None
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
            effective_folder,
            None,
            chat_id,
            schema,
        )

        # Add schema to payload if provided
        if schema:
            # If schema is a Pydantic model class, we need to serialize it to a schema dict
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                payload["schema"] = schema.model_json_schema()
            else:
                payload["schema"] = schema

            # Add a hint to the query to return in JSON format
            payload["query"] = f"{payload['query']}\nReturn the answer in JSON format according to the required schema."

        response = await self._request("POST", "query", data=payload)
        return self._logic._parse_completion_response(response)

    async def agent_query(self, query: str) -> Dict[str, Any]:
        """
        Execute an agentic query with tool access and conversation handling.

        The agent can autonomously use various tools to answer complex queries including:
        - Searching and retrieving relevant documents
        - Analyzing document content
        - Performing calculations and data processing
        - Creating summaries and reports
        - Managing knowledge graphs

        Args:
            query: Natural language query for the Morphik agent

        Returns:
            Dict[str, Any]: Agent response with potential tool execution results and sources

        Example:
            ```python
            # Simple query
            result = await db.agent_query("What are the main trends in our Q3 sales data?")
            print(result["response"])

            # Complex analysis request
            result = await db.agent_query(
                "Analyze all documents from the marketing department, "
                "identify key performance metrics, and create a summary "
                "with actionable insights"
            )
            print(result["response"])

            # Tool usage is automatic - the agent will decide which tools to use
            # based on the query requirements
            ```
        """
        request = {"query": query}
        response = await self._request("POST", "agent", data=request)
        return response

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
    ) -> List[Document]:
        """
        List accessible documents.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters
            folder_name: Optional folder name (or list of names) to scope the request

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
        params, data = self._logic._prepare_list_documents_request(skip, limit, filters, folder_name, None)
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

    async def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """
        Get the current processing status of a document.

        Args:
            document_id: ID of the document to check

        Returns:
            Dict[str, Any]: Status information including current status, potential errors, and other metadata

        Example:
            ```python
            status = await db.get_document_status("doc_123")
            if status["status"] == "completed":
                print("Document processing complete")
            elif status["status"] == "failed":
                print(f"Processing failed: {status['error']}")
            else:
                print("Document still processing...")
            ```
        """
        response = await self._request("GET", f"documents/{document_id}/status")
        return response

    async def wait_for_document_completion(
        self, document_id: str, timeout_seconds=300, check_interval_seconds=2
    ) -> Document:
        """
        Wait for a document's processing to complete.

        Args:
            document_id: ID of the document to wait for
            timeout_seconds: Maximum time to wait for completion (default: 300 seconds)
            check_interval_seconds: Time between status checks (default: 2 seconds)

        Returns:
            Document: Updated document with the latest status

        Raises:
            TimeoutError: If processing doesn't complete within the timeout period
            ValueError: If processing fails with an error

        Example:
            ```python
            # Upload a file and wait for processing to complete
            doc = await db.ingest_file("large_document.pdf")
            try:
                completed_doc = await db.wait_for_document_completion(doc.external_id)
                print(f"Processing complete! Document has {len(completed_doc.chunk_ids)} chunks")
            except TimeoutError:
                print("Processing is taking too long")
            except ValueError as e:
                print(f"Processing failed: {e}")
            ```
        """
        import asyncio

        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            status = await self.get_document_status(document_id)

            if status["status"] == "completed":
                # Get the full document now that it's complete
                return await self.get_document(document_id)
            elif status["status"] == "failed":
                raise ValueError(f"Document processing failed: {status.get('error', 'Unknown error')}")

            # Wait before checking again
            await asyncio.sleep(check_interval_seconds)

        raise TimeoutError(f"Document processing did not complete within {timeout_seconds} seconds")

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
            response = await self._request("POST", f"documents/{document_id}/update_file", data=form_data, files=files)

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
        response = await self._request("POST", f"documents/{document_id}/update_metadata", data=metadata)
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

    async def batch_get_documents(
        self, document_ids: List[str], folder_name: Optional[Union[str, List[str]]] = None
    ) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation.

        Args:
            document_ids: List of document IDs to retrieve
            folder_name: Optional folder name (or list of names) to scope the request

        Returns:
            List[Document]: List of document metadata for found documents

        Example:
            ```python
            docs = await db.batch_get_documents(["doc_123", "doc_456", "doc_789"])
            for doc in docs:
                print(f"Document {doc.external_id}: {doc.metadata.get('title')}")
            ```
        """
        # API expects a dict with document_ids key, not a direct list
        request = {"document_ids": document_ids}
        if folder_name:
            request["folder_name"] = folder_name
        response = await self._request("POST", "batch/documents", data=request)
        docs = self._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self
        return docs

    async def batch_get_chunks(
        self,
        sources: List[Union[ChunkSource, Dict[str, Any]]],
        folder_name: Optional[Union[str, List[str]]] = None,
        use_colpali: bool = True,
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number
            folder_name: Optional folder name (or list of names) to scope the request
            use_colpali: Whether to use ColPali-style embedding model

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
        request = self._logic._prepare_batch_get_chunks_request(sources, folder_name, None, use_colpali)
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
            filters: Optional metadata filters to determine which documents to include.
                These filters will be applied in addition to any specific docs provided.
            docs: Optional list of specific document IDs to include.
                These docs will be included in addition to any documents matching the filters.

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
        request = self._logic._prepare_create_graph_request(name, filters, documents, prompt_overrides, None, None)
        response = await self._request("POST", "graph/create", data=request)
        graph = self._logic._parse_graph_response(response)
        graph._client = self  # Attach AsyncMorphik client for polling helpers
        return graph

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
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

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
        graphs = self._logic._parse_graph_list_response(response)
        for g in graphs:
            g._client = self
        return graphs

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
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

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

    async def create_app(self, app_id: str, name: str, expiry_days: int = 30) -> Dict[str, str]:
        """Create a new application in Morphik Cloud and obtain its auth URI (async)."""

        payload = {"app_id": app_id, "name": name, "expiry_days": expiry_days}
        return await self._request("POST", "ee/create_app", data=payload)

    async def wait_for_graph_completion(
        self,
        graph_name: str,
        timeout_seconds: int = 300,
        check_interval_seconds: int = 5,
    ) -> Graph:
        """Block until the specified graph finishes processing (async).

        Args:
            graph_name: Name of the graph to monitor.
            timeout_seconds: Maximum seconds to wait.
            check_interval_seconds: Seconds between status checks.

        Returns:
            Graph: The completed graph object.
        """
        import asyncio

        start = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start) < timeout_seconds:
            graph = await self.get_graph(graph_name)
            if graph.is_completed:
                return graph
            if graph.is_failed:
                raise RuntimeError(graph.error or "Graph processing failed")
            await asyncio.sleep(check_interval_seconds)
        raise TimeoutError("Timed out waiting for graph completion")

    async def ping(self) -> Dict[str, Any]:
        """Simple health-check call to ``/ping`` endpoint."""
        return await self._request("GET", "ping")

    # ------------------------------------------------------------------
    # Chat API ----------------------------------------------------------
    # ------------------------------------------------------------------
    async def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Return the full message history for *chat_id*."""
        return await self._request("GET", f"chat/{chat_id}")

    async def list_chat_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent chat conversations for the current user (async)."""
        limit_capped = max(1, min(limit, 500))
        return await self._request("GET", "chats", params={"limit": limit_capped})

    # ------------------------------------------------------------------
    # Usage API ---------------------------------------------------------
    # ------------------------------------------------------------------
    async def get_usage_stats(self) -> Dict[str, int]:
        """Return cumulative token usage statistics (async)."""
        return await self._request("GET", "usage/stats")

    async def get_recent_usage(
        self,
        operation_type: Optional[str] = None,
        since: Optional["datetime"] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent usage entries with optional filtering (async)."""
        from datetime import datetime

        params: Dict[str, Any] = {}
        if operation_type:
            params["operation_type"] = operation_type
        if since:
            params["since"] = since.isoformat() if isinstance(since, datetime) else str(since)
        if status:
            params["status"] = status
        return await self._request("GET", "usage/recent", params=params)

    # ------------------------------------------------------------------
    # Graph helpers -----------------------------------------------------
    # ------------------------------------------------------------------
    async def get_graph_visualization(
        self,
        name: str,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch nodes & links for visualising *name* graph (async)."""
        params: Dict[str, Any] = {}
        if folder_name is not None:
            params["folder_name"] = folder_name
        if end_user_id is not None:
            params["end_user_id"] = end_user_id
        return await self._request("GET", f"graph/{name}/visualization", params=params)

    async def check_workflow_status(self, workflow_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Poll the status of an async graph build/update workflow."""
        params = {"run_id": run_id} if run_id else None
        return await self._request("GET", f"graph/workflow/{workflow_id}/status", params=params)

    # ------------------------------------------------------------------
    # Document download helpers ----------------------------------------
    # ------------------------------------------------------------------
    async def get_document_download_url(self, document_id: str, expires_in: int = 3600) -> Dict[str, Any]:
        """Generate a presigned download URL for a document (async)."""
        return await self._request("GET", f"documents/{document_id}/download_url", params={"expires_in": expires_in})

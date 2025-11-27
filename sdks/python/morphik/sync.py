import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Type, Union

import httpx
from pydantic import BaseModel

from ._internal import FinalChunkResult, RuleOrDict, _MorphikClientLogic
from ._scoped_ops import _ScopedOperationsMixin
from .models import CompletionResponse  # Prompt override models
from .models import (
    ChunkSource,
    Document,
    DocumentQueryResponse,
    DocumentResult,
    FolderInfo,
    Graph,
    GraphPromptOverrides,
    IngestTextRequest,
    ListDocsResponse,
    QueryPromptOverrides,
)

logger = logging.getLogger(__name__)


class Folder:
    """
    A folder that allows operations to be scoped to a specific folder.

    Args:
        client: The Morphik client instance
        name: The name of the folder
        folder_id: Optional folder ID (if already known)
    """

    def __init__(self, client: "Morphik", name: str, folder_id: Optional[str] = None):
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

    def get_info(self) -> Dict[str, Any]:
        """
        Get detailed information about this folder.

        Returns:
            Dict[str, Any]: Detailed folder information
        """
        if not self._id:
            # If we don't have the ID, find the folder by name first
            folders = self._client.list_folders()
            for folder in folders:
                if folder.name == self._name:
                    self._id = folder.id
                    break
            if not self._id:
                raise ValueError(f"Folder '{self._name}' not found")

        return self._client._request("GET", f"folders/{self._id}")

    def signin(self, end_user_id: str) -> "UserScope":
        """
        Returns a UserScope object scoped to this folder and the end user.

        Args:
            end_user_id: The ID of the end user

        Returns:
            UserScope: A user scope scoped to this folder and the end user
        """
        return UserScope(client=self._client, end_user_id=end_user_id, folder_name=self._name)

    def ingest_text(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            Document: Metadata of the ingested document
        """
        return self._client._scoped_ingest_text(
            content=content,
            filename=filename,
            metadata=metadata,
            rules=rules,
            use_colpali=use_colpali,
            folder_name=self._name,
            end_user_id=None,
        )

    def ingest_file(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            Document: Metadata of the ingested document
        """
        return self._client._scoped_ingest_file(
            file=file,
            filename=filename,
            metadata=metadata,
            rules=rules,
            use_colpali=use_colpali,
            folder_name=self._name,
            end_user_id=None,
        )

    def ingest_files(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            use_colpali: Whether to use ColPali-style embedding
            parallel: Whether to process files in parallel

        Returns:
            List[Document]: List of ingested documents
        """
        return self._client._scoped_ingest_files(
            files=files,
            metadata=metadata,
            rules=rules,
            use_colpali=use_colpali,
            parallel=parallel,
            folder_name=self._name,
            end_user_id=None,
        )

    def ingest_directory(
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
            rules: Deprecated; retained for backwards compatibility and ignored
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
        return self.ingest_files(
            files=files, metadata=metadata, rules=rules, use_colpali=use_colpali, parallel=parallel
        )

    def query_document(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        prompt: str,
        schema: Optional[Union[Dict[str, Any], Type[BaseModel], BaseModel, str]] = None,
        ingestion_options: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> DocumentQueryResponse:
        """
        Run a one-off document query scoped to this folder.

        Args:
            file: File-like input analysed inline by Morphik On-the-Fly.
            prompt: Natural-language instruction to execute against the document.
            schema: Optional schema definition (dict, Pydantic model, or JSON string) for structured output.
            ingestion_options: Optional dict controlling ingestion follow-up.
            filename: Override filename when providing bytes or file-like objects.

        Returns:
            DocumentQueryResponse: Structured response containing outputs and ingestion status.
        """
        options = dict(ingestion_options or {})
        options.setdefault("folder_name", self._name)

        return self._client.query_document(
            file=file,
            prompt=prompt,
            schema=schema,
            ingestion_options=options,
            filename=filename,
            folder_name=self._name,
        )

    def retrieve_chunks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
        additional_folders: Optional[List[str]] = None,
        padding: int = 0,
        output_format: Optional[str] = None,
    ) -> List[FinalChunkResult]:
        """
        Retrieve relevant chunks within this folder.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model
            additional_folders: Optional list of extra folders to include in the scope
            padding: Number of additional chunks/pages to retrieve before and after matched chunks (ColPali only, default: 0)
            output_format: Controls how image chunks are returned (e.g., "base64" or "url")

        Returns:
            List[FinalChunkResult]: List of relevant chunks
        """
        effective_folder = self._merge_folders(additional_folders)
        return self._client._scoped_retrieve_chunks(
            query=query,
            filters=filters,
            k=k,
            min_score=min_score,
            use_colpali=use_colpali,
            folder_name=effective_folder,
            end_user_id=None,
            padding=padding,
            output_format=output_format,
        )

    def retrieve_docs(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
        use_reranking: Optional[bool] = None,  # Add missing parameter
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
            use_reranking: Whether to use reranking
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[DocumentResult]: List of relevant documents
        """
        effective_folder = self._merge_folders(additional_folders)
        return self._client._scoped_retrieve_docs(
            query=query,
            filters=filters,
            k=k,
            min_score=min_score,
            use_colpali=use_colpali,
            folder_name=effective_folder,
            end_user_id=None,
            use_reranking=use_reranking,
        )

    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_colpali: bool = True,
        use_reranking: Optional[bool] = None,  # Add missing parameter
        graph_name: Optional[str] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
        prompt_overrides: Optional[Union[QueryPromptOverrides, Dict[str, Any]]] = None,
        additional_folders: Optional[List[str]] = None,
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        chat_id: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        padding: int = 0,
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
            use_reranking: Whether to use reranking
            graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            hop_depth: Number of relationship hops to traverse in the graph (1-3)
            include_paths: Whether to include relationship paths in the response
            prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts
            additional_folders: Optional list of extra folders to include in the scope
            schema: Optional schema for structured output
            padding: Number of additional chunks/pages to retrieve before and after matched chunks (ColPali only, default: 0)

        Returns:
            CompletionResponse: Generated completion
        """
        effective_folder = self._merge_folders(additional_folders)
        return self._client._scoped_query(
            query=query,
            filters=filters,
            k=k,
            min_score=min_score,
            max_tokens=max_tokens,
            temperature=temperature,
            use_colpali=use_colpali,
            graph_name=graph_name,
            hop_depth=hop_depth,
            include_paths=include_paths,
            prompt_overrides=prompt_overrides,
            folder_name=effective_folder,
            end_user_id=None,
            use_reranking=use_reranking,
            chat_id=chat_id,
            schema=schema,
            llm_config=llm_config,
            padding=padding,
        )

    def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        additional_folders: Optional[List[str]] = None,
        include_total_count: bool = False,
        include_status_counts: bool = False,
        include_folder_counts: bool = False,
        completed_only: bool = False,
        sort_by: Optional[str] = "updated_at",
        sort_direction: str = "desc",
    ) -> ListDocsResponse:
        """
        List accessible documents within this folder.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters
            additional_folders: Optional list of extra folders to include in the scope
            include_total_count: Include total count of matching documents
            include_status_counts: Include counts grouped by status
            include_folder_counts: Include counts grouped by folder
            completed_only: Only return completed documents
            sort_by: Field to sort by (created_at, updated_at, filename, external_id)
            sort_direction: Sort direction (asc, desc)

        Returns:
            ListDocsResponse: Response with documents and metadata
        """
        effective_folder = self._merge_folders(additional_folders)
        return self._client._scoped_list_documents(
            skip=skip,
            limit=limit,
            filters=filters,
            folder_name=effective_folder,
            end_user_id=None,
            include_total_count=include_total_count,
            include_status_counts=include_status_counts,
            include_folder_counts=include_folder_counts,
            completed_only=completed_only,
            sort_by=sort_by,
            sort_direction=sort_direction,
        )

    def batch_get_documents(
        self, document_ids: List[str], additional_folders: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation within this folder.

        Args:
            document_ids: List of document IDs to retrieve
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[Document]: List of document metadata for found documents
        """
        merged = self._merge_folders(additional_folders)
        request = {"document_ids": document_ids, "folder_name": merged}

        response = self._client._request("POST", "batch/documents", data=request)
        docs = [self._client._logic._parse_document_response(doc) for doc in response]
        for doc in docs:
            doc._client = self._client
        return docs

    def batch_get_chunks(
        self,
        sources: List[Union[ChunkSource, Dict[str, Any]]],
        additional_folders: Optional[List[str]] = None,
        use_colpali: bool = True,
        output_format: Optional[str] = None,
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in this folder.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number
            additional_folders: Optional list of extra folders to include in the scope
            use_colpali: Whether to request multimodal chunks when available
            output_format: Controls how image chunks are returned (e.g., "base64" or "url")

        Returns:
            List[FinalChunkResult]: List of chunk results
        """
        merged = self._merge_folders(additional_folders)
        request = self._client._logic._prepare_batch_get_chunks_request(
            sources,
            merged,
            None,
            use_colpali,
            output_format,
        )

        response = self._client._request("POST", "batch/chunks", data=request)
        return self._client._logic._parse_chunk_result_list_response(response)

    def create_graph(
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
        # Convert prompt_overrides to dict if it's a model
        if prompt_overrides and isinstance(prompt_overrides, GraphPromptOverrides):
            prompt_overrides = prompt_overrides.model_dump(exclude_none=True)

        request = {
            "name": name,
            "filters": filters,
            "documents": documents,
            "prompt_overrides": prompt_overrides,
            "folder_name": self._name,  # Add folder name here
        }

        response = self._client._request("POST", "graph/create", request)
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

    def update_graph(
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
        # Convert prompt_overrides to dict if it's a model
        if prompt_overrides and isinstance(prompt_overrides, GraphPromptOverrides):
            prompt_overrides = prompt_overrides.model_dump(exclude_none=True)

        request = {
            "additional_filters": additional_filters,
            "additional_documents": additional_documents,
            "prompt_overrides": prompt_overrides,
            "folder_name": self._name,  # Add folder name here
        }

        response = self._client._request("POST", f"graph/{name}/update", request)
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

    def delete_document_by_filename(self, filename: str) -> Dict[str, str]:
        """
        Delete a document by its filename within this folder.

        Args:
            filename: Filename of the document to delete

        Returns:
            Dict[str, str]: Deletion status
        """
        # First get the document ID
        response = self._client._request("GET", f"documents/filename/{filename}", params={"folder_name": self._name})
        doc = self._client._logic._parse_document_response(response)

        # Then delete by ID
        return self._client.delete_document(doc.external_id)

    # Helper --------------------------------------------------------------
    def _merge_folders(self, additional_folders: Optional[List[str]] = None) -> Union[str, List[str]]:
        """Return the effective folder scope.

        If *additional_folders* is provided it will be combined with the folder's
        own *self._name* and returned as a list (to preserve ordering and allow
        duplicates to be removed server-side).  Otherwise just *self._name* is
        returned so we keep backward-compatibility with the original API that
        expected a single string.
        """
        if not additional_folders:
            return self._name
        # Pre-pend the scoped folder to the list provided by the caller.
        return [self._name] + additional_folders


class UserScope:
    """
    A user scope that allows operations to be scoped to a specific end user and optionally a folder.

    Args:
        client: The Morphik client instance
        end_user_id: The ID of the end user
        folder_name: Optional folder name to further scope operations
    """

    def __init__(self, client: "Morphik", end_user_id: str, folder_name: Optional[str] = None):
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

    def ingest_text(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            Document: Metadata of the ingested document
        """
        return self._client._scoped_ingest_text(
            content=content,
            filename=filename,
            metadata=metadata,
            rules=rules,
            use_colpali=use_colpali,
            folder_name=self._folder_name,
            end_user_id=self._end_user_id,
        )

    def ingest_file(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            Document: Metadata of the ingested document
        """
        return self._client._scoped_ingest_file(
            file=file,
            filename=filename,
            metadata=metadata,
            rules=rules,
            use_colpali=use_colpali,
            folder_name=self._folder_name,
            end_user_id=self._end_user_id,
        )

    def ingest_files(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            use_colpali: Whether to use ColPali-style embedding
            parallel: Whether to process files in parallel

        Returns:
            List[Document]: List of ingested documents
        """
        return self._client._scoped_ingest_files(
            files=files,
            metadata=metadata,
            rules=rules,
            use_colpali=use_colpali,
            parallel=parallel,
            folder_name=self._folder_name,
            end_user_id=self._end_user_id,
        )

    def ingest_directory(
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
            rules: Deprecated; retained for backwards compatibility and ignored
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
        return self.ingest_files(
            files=files, metadata=metadata, rules=rules, use_colpali=use_colpali, parallel=parallel
        )

    def query_document(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        prompt: str,
        schema: Optional[Union[Dict[str, Any], Type[BaseModel], BaseModel, str]] = None,
        ingestion_options: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> DocumentQueryResponse:
        """
        Run a one-off document query scoped to this end user (and optional folder).

        Args:
            file: File-like input analysed inline by Morphik On-the-Fly.
            prompt: Natural-language instruction to execute against the document.
            schema: Optional schema definition (dict, Pydantic model, or JSON string) for structured output.
            ingestion_options: Optional dict controlling ingestion follow-up.
            filename: Override filename when providing bytes or file-like objects.

        Returns:
            DocumentQueryResponse: Structured response containing outputs and ingestion status.
        """
        options = dict(ingestion_options or {})
        options.setdefault("end_user_id", self._end_user_id)
        if self._folder_name and "folder_name" not in options:
            options["folder_name"] = self._folder_name

        return self._client.query_document(
            file=file,
            prompt=prompt,
            schema=schema,
            ingestion_options=options,
            filename=filename,
            folder_name=self._folder_name,
            end_user_id=self._end_user_id,
        )

    def retrieve_chunks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
        additional_folders: Optional[List[str]] = None,
        padding: int = 0,
        output_format: Optional[str] = None,
    ) -> List[FinalChunkResult]:
        """
        Retrieve relevant chunks as this end user.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model
            additional_folders: Optional list of extra folders to include in the scope
            padding: Number of additional chunks/pages to retrieve before and after matched chunks (ColPali only, default: 0)
            output_format: Controls how image chunks are returned (e.g., "base64" or "url")

        Returns:
            List[FinalChunkResult]: List of relevant chunks
        """
        effective_folder = self._merge_folders(additional_folders)
        return self._client._scoped_retrieve_chunks(
            query=query,
            filters=filters,
            k=k,
            min_score=min_score,
            use_colpali=use_colpali,
            folder_name=effective_folder,
            end_user_id=self._end_user_id,
            padding=padding,
            output_format=output_format,
        )

    def retrieve_docs(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
        use_reranking: Optional[bool] = None,  # Add missing parameter
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
            use_reranking: Whether to use reranking
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[DocumentResult]: List of relevant documents
        """
        effective_folder = self._merge_folders(additional_folders)
        return self._client._scoped_retrieve_docs(
            query=query,
            filters=filters,
            k=k,
            min_score=min_score,
            use_colpali=use_colpali,
            folder_name=effective_folder,
            end_user_id=self._end_user_id,
            use_reranking=use_reranking,
        )

    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_colpali: bool = True,
        use_reranking: Optional[bool] = None,  # Add missing parameter
        graph_name: Optional[str] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
        prompt_overrides: Optional[Union[QueryPromptOverrides, Dict[str, Any]]] = None,
        additional_folders: Optional[List[str]] = None,
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        chat_id: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        padding: int = 0,
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
            use_reranking: Whether to use reranking
            graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            hop_depth: Number of relationship hops to traverse in the graph (1-3)
            include_paths: Whether to include relationship paths in the response
            prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts
            additional_folders: Optional list of extra folders to include in the scope
            schema: Optional schema for structured output
            padding: Number of additional chunks/pages to retrieve before and after matched chunks (ColPali only, default: 0)

        Returns:
            CompletionResponse: Generated completion
        """
        effective_folder = self._merge_folders(additional_folders)
        return self._client._scoped_query(
            query=query,
            filters=filters,
            k=k,
            min_score=min_score,
            max_tokens=max_tokens,
            temperature=temperature,
            use_colpali=use_colpali,
            graph_name=graph_name,
            hop_depth=hop_depth,
            include_paths=include_paths,
            prompt_overrides=prompt_overrides,
            folder_name=effective_folder,
            end_user_id=self._end_user_id,
            use_reranking=use_reranking,
            chat_id=chat_id,
            schema=schema,
            llm_config=llm_config,
            padding=padding,
        )

    def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        additional_folders: Optional[List[str]] = None,
        include_total_count: bool = False,
        include_status_counts: bool = False,
        include_folder_counts: bool = False,
        completed_only: bool = False,
        sort_by: Optional[str] = "updated_at",
        sort_direction: str = "desc",
    ) -> ListDocsResponse:
        """
        List accessible documents for this end user.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters
            additional_folders: Optional list of extra folders to include in the scope
            include_total_count: Include total count of matching documents
            include_status_counts: Include counts grouped by status
            include_folder_counts: Include counts grouped by folder
            completed_only: Only return completed documents
            sort_by: Field to sort by (created_at, updated_at, filename, external_id)
            sort_direction: Sort direction (asc, desc)

        Returns:
            ListDocsResponse: Response with documents and metadata
        """
        effective_folder = self._merge_folders(additional_folders)
        return self._client._scoped_list_documents(
            skip=skip,
            limit=limit,
            filters=filters,
            folder_name=effective_folder,
            end_user_id=self._end_user_id,
            include_total_count=include_total_count,
            include_status_counts=include_status_counts,
            include_folder_counts=include_folder_counts,
            completed_only=completed_only,
            sort_by=sort_by,
            sort_direction=sort_direction,
        )

    def batch_get_documents(
        self, document_ids: List[str], additional_folders: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation for this end user.

        Args:
            document_ids: List of document IDs to retrieve
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[Document]: List of document metadata for found documents
        """
        merged = self._merge_folders(additional_folders)
        request = {"document_ids": document_ids, "end_user_id": self._end_user_id}

        if merged:
            request["folder_name"] = merged

        response = self._client._request("POST", "batch/documents", data=request)
        docs = [self._client._logic._parse_document_response(doc) for doc in response]
        for doc in docs:
            doc._client = self._client
        return docs

    def batch_get_chunks(
        self,
        sources: List[Union[ChunkSource, Dict[str, Any]]],
        additional_folders: Optional[List[str]] = None,
        use_colpali: bool = True,
        output_format: Optional[str] = None,
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in this folder.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number
            additional_folders: Optional list of extra folders to include in the scope
            use_colpali: Whether to request multimodal chunks when available
            output_format: Controls how image chunks are returned (e.g., "base64" or "url")

        Returns:
            List[FinalChunkResult]: List of chunk results
        """
        merged = self._merge_folders(additional_folders)
        request = self._client._logic._prepare_batch_get_chunks_request(
            sources,
            merged,
            None,
            use_colpali,
            output_format,
        )

        response = self._client._request("POST", "batch/chunks", data=request)
        return self._client._logic._parse_chunk_result_list_response(response)

    def create_graph(
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
        # Convert prompt_overrides to dict if it's a model
        if prompt_overrides and isinstance(prompt_overrides, GraphPromptOverrides):
            prompt_overrides = prompt_overrides.model_dump(exclude_none=True)

        request = {
            "name": name,
            "filters": filters,
            "documents": documents,
            "prompt_overrides": prompt_overrides,
            "end_user_id": self._end_user_id,  # Add end user ID here
        }

        # Add folder name if scoped to a folder
        if self._folder_name:
            request["folder_name"] = self._folder_name

        response = self._client._request("POST", "graph/create", request)
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

    def update_graph(
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
        # Convert prompt_overrides to dict if it's a model
        if prompt_overrides and isinstance(prompt_overrides, GraphPromptOverrides):
            prompt_overrides = prompt_overrides.model_dump(exclude_none=True)

        request = {
            "additional_filters": additional_filters,
            "additional_documents": additional_documents,
            "prompt_overrides": prompt_overrides,
            "end_user_id": self._end_user_id,  # Add end user ID here
        }

        # Add folder name if scoped to a folder
        if self._folder_name:
            request["folder_name"] = self._folder_name

        response = self._client._request("POST", f"graph/{name}/update", request)
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

    def delete_document_by_filename(self, filename: str) -> Dict[str, str]:
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
        response = self._client._request("GET", f"documents/filename/{filename}", params=params)
        doc = self._client._logic._parse_document_response(response)

        # Then delete by ID
        return self._client.delete_document(doc.external_id)

    # Helper --------------------------------------------------------------
    def _merge_folders(self, additional_folders: Optional[List[str]] = None) -> Union[str, List[str], None]:
        """Return combined folder scope for user.

        When this user scope is already tied to *self._folder_name* we combine it
        with any *additional_folders* passed by the caller.  Otherwise just the
        *additional_folders* (or None) is returned so that upstream logic is
        unchanged.
        """
        base = self._folder_name
        if additional_folders:
            if base:
                return [base] + additional_folders
            return additional_folders
        return base


class Morphik(_ScopedOperationsMixin):
    """
    Morphik client for document operations.

    Args:
        uri (str, optional): Morphik URI in format "morphik://<owner_id>:<token>@<host>".
            If not provided, connects to http://localhost:8000 without authentication.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        is_local (bool, optional): Whether connecting to local development server. Defaults to False.

    Examples:
        ```python
        # Without authentication
        db = Morphik()

        # With authentication
        db = Morphik("morphik://owner_id:token@api.morphik.ai")
        ```
    """

    def __init__(self, uri: Optional[str] = None, timeout: int = 30, is_local: bool = False):
        self._logic = _MorphikClientLogic(uri, timeout, is_local)
        self._client = httpx.Client(timeout=self._logic._timeout, verify=not self._logic._is_local)

    def _request(
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

            # Files are now properly handled
        else:
            # JSON for everything else
            headers["Content-Type"] = "application/json"
            request_data = {"json": data}

        response = self._client.request(
            method,
            url,
            headers=headers,
            params=params,
            **request_data,
        )
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # Print error response for debugging
            print(f"Error response: {e.response.status_code} - {e.response.text}")
            raise

    def _convert_rule(self, rule: RuleOrDict) -> Dict[str, Any]:
        """Convert a rule to a dictionary format"""
        return self._logic._convert_rule(rule)

    def create_folder(self, name: str, description: Optional[str] = None) -> Folder:
        """
        Create a folder to scope operations.

        Args:
            name: The name of the folder
            description: Optional description for the folder

        Returns:
            Folder: A folder object ready for scoped operations
        """
        payload = {"name": name}
        if description:
            payload["description"] = description

        response = self._request("POST", "folders", data=payload)
        folder_info = FolderInfo(**response)

        # Return a usable Folder object with the ID from the response
        return Folder(self, name, folder_id=folder_info.id)

    def delete_folder(self, folder_id_or_name: str) -> Dict[str, Any]:
        """
        Delete a folder and all associated documents.

        Args:
            folder_id_or_name: Name or ID of the folder to delete

        Returns:
            Dict containing status and message
        """
        response = self._request("DELETE", f"folders/{folder_id_or_name}")
        return response

    def get_folder_by_name(self, name: str) -> Folder:
        """
        Get a folder by name to scope operations.

        Args:
            name: The name of the folder

        Returns:
            Folder: A folder object for scoped operations
        """
        return Folder(self, name)

    def get_folder(self, folder_id_or_name: str) -> Folder:
        """
        Get a folder by ID or name.

        Args:
            folder_id_or_name: ID or name of the folder

        Returns:
            Folder: A folder object for scoped operations
        """
        response = self._request("GET", f"folders/{folder_id_or_name}")
        folder_id = response.get("id", folder_id_or_name)
        return Folder(self, response["name"], folder_id)

    def list_folders(self) -> List[Folder]:
        """
        List all folders the user has access to as Folder objects.

        Returns:
            List[Folder]: List of Folder objects ready for operations
        """
        folder_infos = self._request("GET", "folders")
        return [Folder(self, info["name"], info["id"]) for info in folder_infos]

    def add_document_to_folder(self, folder_id_or_name: str, document_id: str) -> Dict[str, str]:
        """
        Add a document to a folder.

        Args:
            folder_id_or_name: ID or name of the folder
            document_id: ID of the document

        Returns:
            Dict[str, str]: Success status
        """
        response = self._request("POST", f"folders/{folder_id_or_name}/documents/{document_id}")
        return response

    def remove_document_from_folder(self, folder_id_or_name: str, document_id: str) -> Dict[str, str]:
        """
        Remove a document from a folder.

        Args:
            folder_id_or_name: ID or name of the folder
            document_id: ID of the document

        Returns:
            Dict[str, str]: Success status
        """
        response = self._request("DELETE", f"folders/{folder_id_or_name}/documents/{document_id}")
        return response

    def signin(self, end_user_id: str) -> UserScope:
        """
        Sign in as an end user to scope operations.

        Args:
            end_user_id: The ID of the end user

        Returns:
            UserScope: A user scope object for scoped operations
        """
        return UserScope(self, end_user_id)

    def ingest_text(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            use_colpali: Whether to use ColPali-style embedding model to ingest the text
                (slower, but significantly better retrieval accuracy for text and images)
        Returns:
            Document: Metadata of the ingested document
        """
        return self._scoped_ingest_text(
            content=content,
            filename=filename,
            metadata=metadata,
            rules=rules,
            use_colpali=use_colpali,
            folder_name=None,
            end_user_id=None,
        )

    def ingest_file(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List[RuleOrDict]] = None,
        use_colpali: bool = True,
    ) -> Document:
        """
        Ingest a file document into Morphik.

        Args:
            file: File to ingest (path string, bytes, file object, or Path)
            filename: Name of the file
            metadata: Optional metadata dictionary
            rules: Deprecated; retained for backwards compatibility and ignored
            use_colpali: Whether to use ColPali-style embedding model to ingest the file
                (slower, but significantly better retrieval accuracy for images)

        Returns:
            Document: Metadata of the ingested document

        """
        return self._scoped_ingest_file(
            file=file,
            filename=filename,
            metadata=metadata,
            rules=rules,
            use_colpali=use_colpali,
            folder_name=None,
            end_user_id=None,
        )

    def query_document(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        prompt: str,
        schema: Optional[Union[Dict[str, Any], Type[BaseModel], BaseModel, str]] = None,
        ingestion_options: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
    ) -> DocumentQueryResponse:
        """
        Run a one-off document query using Morphik On-the-Fly.

        Args:
            file: File-like input analysed inline.
            prompt: Natural-language instruction to execute against the document.
            schema: Optional schema definition (dict, Pydantic model, or JSON string) for structured output.
            ingestion_options: Optional dict controlling ingestion follow-up behaviour. Supported keys: `ingest`,
                `metadata`, `use_colpali`, `folder_name`, `end_user_id`. Unknown keys are ignored server-side.
            filename: Override filename when providing bytes or file-like objects.
            folder_name: Optional folder scope (auto-set when using Folder helpers).
            end_user_id: Optional end-user scope (auto-set when using UserScope helpers).

        Returns:
            DocumentQueryResponse: Structured response containing outputs and ingestion status. When `ingest=True`, the
            server queues ingestion after merging any provided metadata with schema-derived fields.
        """
        file_obj, resolved_filename = self._logic._prepare_file_for_upload(file, filename)

        try:
            files = {"file": (resolved_filename, file_obj)}
            form_data = self._logic._prepare_document_query_form_data(
                prompt=prompt,
                schema=schema,
                ingestion_options=ingestion_options,
                folder_name=folder_name,
                end_user_id=end_user_id,
            )

            response = self._request(
                "POST",
                "ingest/document/query",
                data=form_data,
                files=files,
            )
            result = self._logic._parse_document_query_response(response)
            if result.ingestion_document is not None:
                result.ingestion_document._client = self
            return result
        finally:
            if isinstance(file, (str, Path)):
                file_obj.close()

    def ingest_files(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            use_colpali: Whether to use ColPali-style embedding
            parallel: Whether to process files in parallel

        Returns:
            List[Document]: List of successfully ingested documents

        Raises:
            ValueError: If metadata list length doesn't match files length
        """
        return self._scoped_ingest_files(
            files=files,
            metadata=metadata,
            rules=rules,
            use_colpali=use_colpali,
            parallel=parallel,
            folder_name=None,
            end_user_id=None,
        )

    def ingest_directory(
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
            rules: Deprecated; retained for backwards compatibility and ignored
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
        return self.ingest_files(
            files=files, metadata=metadata, rules=rules, use_colpali=use_colpali, parallel=parallel
        )

    def retrieve_chunks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
        folder_name: Optional[Union[str, List[str]]] = None,
        padding: int = 0,
        output_format: Optional[str] = None,
    ) -> List[FinalChunkResult]:
        """
        Retrieve relevant chunks.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model to retrieve the chunks
                (only works for documents ingested with `use_colpali=True`)
            padding: Number of additional chunks/pages to retrieve before and after matched chunks (ColPali only, default: 0)
        Returns:
            List[ChunkResult]

        """
        return self._scoped_retrieve_chunks(
            query=query,
            filters=filters,
            k=k,
            min_score=min_score,
            use_colpali=use_colpali,
            folder_name=folder_name,
            end_user_id=None,
            padding=padding,
            output_format=output_format,
        )

    def retrieve_docs(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        use_colpali: bool = True,
        use_reranking: Optional[bool] = None,  # Add missing parameter
        folder_name: Optional[Union[str, List[str]]] = None,
    ) -> List[DocumentResult]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query text
            filters: Optional metadata filters
            k: Number of results (default: 4)
            min_score: Minimum similarity threshold (default: 0.0)
            use_colpali: Whether to use ColPali-style embedding model to retrieve the documents
                (only works for documents ingested with `use_colpali=True`)
            use_reranking: Whether to use reranking
            folder_name: Optional folder name (or list of names) to scope the request
        Returns:
            List[DocumentResult]

        """
        return self._scoped_retrieve_docs(
            query=query,
            filters=filters,
            k=k,
            min_score=min_score,
            use_colpali=use_colpali,
            folder_name=folder_name,
            end_user_id=None,
            use_reranking=use_reranking,
        )

    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_colpali: bool = True,
        use_reranking: Optional[bool] = None,  # Add missing parameter
        graph_name: Optional[str] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
        prompt_overrides: Optional[Union[QueryPromptOverrides, Dict[str, Any]]] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        chat_id: Optional[str] = None,
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        padding: int = 0,
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
            use_reranking: Whether to use reranking
            graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            hop_depth: Number of relationship hops to traverse in the graph (1-3)
            include_paths: Whether to include relationship paths in the response
            prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts
                Either a QueryPromptOverrides object or a dictionary with the same structure
            folder_name: Optional folder name to further scope operations
            schema: Optional schema for structured output, can be a Pydantic model or a JSON schema dict
            llm_config: Optional LiteLLM-compatible model configuration (e.g., model name, API key, base URL)
            padding: Number of additional chunks/pages to retrieve before and after matched chunks (ColPali only, default: 0)
        Returns:
            CompletionResponse

        """
        return self._scoped_query(
            query=query,
            filters=filters,
            k=k,
            min_score=min_score,
            max_tokens=max_tokens,
            temperature=temperature,
            use_colpali=use_colpali,
            graph_name=graph_name,
            hop_depth=hop_depth,
            include_paths=include_paths,
            prompt_overrides=prompt_overrides,
            folder_name=folder_name,
            end_user_id=None,
            use_reranking=use_reranking,
            chat_id=chat_id,
            schema=schema,
            llm_config=llm_config,
            padding=padding,
        )

    def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        include_total_count: bool = False,
        include_status_counts: bool = False,
        include_folder_counts: bool = False,
        completed_only: bool = False,
        sort_by: Optional[str] = "updated_at",
        sort_direction: str = "desc",
    ) -> ListDocsResponse:
        """
        List accessible documents.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters
            folder_name: Optional folder name (or list of names) to scope the request
            include_total_count: Include total count of matching documents
            include_status_counts: Include counts grouped by status
            include_folder_counts: Include counts grouped by folder
            completed_only: Only return completed documents
            sort_by: Field to sort by (created_at, updated_at, filename, external_id)
            sort_direction: Sort direction (asc, desc)

        Returns:
            ListDocsResponse: Response with documents and metadata

        """
        return self._scoped_list_documents(
            skip=skip,
            limit=limit,
            filters=filters,
            folder_name=folder_name,
            end_user_id=None,
            include_total_count=include_total_count,
            include_status_counts=include_status_counts,
            include_folder_counts=include_folder_counts,
            completed_only=completed_only,
            sort_by=sort_by,
            sort_direction=sort_direction,
        )

    def get_document(self, document_id: str) -> Document:
        """
        Get document metadata by ID.

        Args:
            document_id: ID of the document

        Returns:
            Document: Document metadata

        """
        response = self._request("GET", f"documents/{document_id}")
        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """
        Get the current processing status of a document.

        Args:
            document_id: ID of the document to check

        Returns:
            Dict[str, Any]: Status information including current status, potential errors, and other metadata

        """
        response = self._request("GET", f"documents/{document_id}/status")
        return response

    def wait_for_document_completion(
        self, document_id: str, timeout_seconds=300, check_interval_seconds=2, progress_callback=None
    ) -> Document:
        """
        Wait for a document's processing to complete.

        Args:
            document_id: ID of the document to wait for
            timeout_seconds: Maximum time to wait for completion (default: 300 seconds)
            check_interval_seconds: Time between status checks (default: 2 seconds)
            progress_callback: Optional callback function that receives progress updates.
                               Called with (current_step, total_steps, step_name, percentage)

        Returns:
            Document: Updated document with the latest status

        Raises:
            TimeoutError: If processing doesn't complete within the timeout period
            ValueError: If processing fails with an error

        """
        import time

        start_time = time.time()

        while (time.time() - start_time) < timeout_seconds:
            status = self.get_document_status(document_id)

            if status["status"] == "completed":
                # Get the full document now that it's complete
                return self.get_document(document_id)
            elif status["status"] == "failed":
                raise ValueError(f"Document processing failed: {status.get('error', 'Unknown error')}")
            elif status["status"] == "processing" and "progress" in status and progress_callback:
                # Call the progress callback with progress information
                progress = status["progress"]
                progress_callback(
                    progress.get("current_step", 0),
                    progress.get("total_steps", 1),
                    progress.get("step_name", "Processing"),
                    progress.get("percentage", 0),
                )

            # Wait before checking again
            time.sleep(check_interval_seconds)

        raise TimeoutError(f"Document processing did not complete within {timeout_seconds} seconds")

    def get_document_by_filename(self, filename: str) -> Document:
        """
        Get document metadata by filename.
        If multiple documents have the same filename, returns the most recently updated one.

        Args:
            filename: Filename of the document to retrieve

        Returns:
            Document: Document metadata

        """
        response = self._request("GET", f"documents/filename/{filename}")
        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    def update_document_with_text(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        """
        # Use the dedicated text update endpoint
        self._logic._warn_legacy_rules(rules, "documents/update_text")

        serialized_metadata, metadata_types_map = self._logic._serialize_metadata_map(metadata)
        request = IngestTextRequest(
            content=content,
            filename=filename,
            metadata=serialized_metadata,
            metadata_types=metadata_types_map or None,
            use_colpali=use_colpali if use_colpali is not None else True,
        )

        params = {}
        if update_strategy != "add":
            params["update_strategy"] = update_strategy

        response = self._request(
            "POST", f"documents/{document_id}/update_text", data=request.model_dump(), params=params
        )

        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    def update_document_with_file(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

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

            self._logic._warn_legacy_rules(rules, "documents/update_file")

            serialized_metadata, metadata_types_map = self._logic._serialize_metadata_map(metadata)
            form_data = {
                "metadata": json.dumps(serialized_metadata),
                "update_strategy": update_strategy,
            }

            if use_colpali is not None:
                form_data["use_colpali"] = str(use_colpali).lower()
            if metadata_types_map:
                form_data["metadata_types"] = json.dumps(metadata_types_map)

            # Use the dedicated file update endpoint
            response = self._request("POST", f"documents/{document_id}/update_file", data=form_data, files=files)

            doc = self._logic._parse_document_response(response)
            doc._client = self
            return doc
        finally:
            # Close file if we opened it
            if isinstance(file, (str, Path)):
                file_obj.close()

    def update_document_metadata(
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

        """
        # Use the dedicated metadata update endpoint
        serialized_metadata, metadata_types_map = self._logic._serialize_metadata_map(metadata)
        payload: Dict[str, Any] = {"metadata": serialized_metadata}
        if metadata_types_map:
            payload["metadata_types"] = metadata_types_map

        response = self._request("POST", f"documents/{document_id}/update_metadata", data=payload)
        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    def update_document_by_filename_with_text(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        """
        # First get the document by filename to obtain its ID
        doc = self.get_document_by_filename(filename)

        # Then use the regular update_document_with_text endpoint with the document ID
        return self.update_document_with_text(
            document_id=doc.external_id,
            content=content,
            filename=new_filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )

    def update_document_by_filename_with_file(
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
            rules: Deprecated; retained for backwards compatibility and ignored
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        """
        # First get the document by filename to obtain its ID
        doc = self.get_document_by_filename(filename)

        # Then use the regular update_document_with_file endpoint with the document ID
        return self.update_document_with_file(
            document_id=doc.external_id,
            file=file,
            filename=new_filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )

    def update_document_by_filename_metadata(
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

        """
        # First get the document by filename to obtain its ID
        doc = self.get_document_by_filename(filename)

        # Update the metadata
        result = self.update_document_metadata(
            document_id=doc.external_id,
            metadata=metadata,
        )

        # If new_filename is provided, update the filename as well
        if new_filename:
            # Create a request that retains the just-updated metadata but also changes filename
            combined_metadata = result.metadata.copy()

            # Update the document again with filename change and the same metadata
            response = self._request(
                "POST",
                f"documents/{doc.external_id}/update_text",
                data={
                    "content": "",
                    "filename": new_filename,
                    "metadata": combined_metadata,
                },
            )
            result = self._logic._parse_document_response(response)
            result._client = self

        return result

    def batch_get_documents(
        self, document_ids: List[str], folder_name: Optional[Union[str, List[str]]] = None
    ) -> List[Document]:
        """
        Retrieve multiple documents by their IDs.

        Args:
            document_ids: List of document IDs to retrieve
            folder_name: Optional folder name (or list of names) to scope the request

        Returns:
            List[Document]: List of document metadata for found documents

        """
        # Build request respecting folder scoping if provided
        request = self._logic._prepare_batch_get_documents_request(document_ids, folder_name, None)
        response = self._request("POST", "batch/documents", data=request)
        docs = self._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self
        return docs

    def batch_get_chunks(
        self,
        sources: List[Union[ChunkSource, Dict[str, Any]]],
        folder_name: Optional[Union[str, List[str]]] = None,
        use_colpali: bool = True,
        output_format: Optional[str] = None,
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number
            folder_name: Optional folder name (or list of names) to scope the request
            use_colpali: Whether to request multimodal chunks when available
            output_format: Controls how image chunks are returned (e.g., "base64" or "url")

        Returns:
            List[FinalChunkResult]: List of chunk results

        """
        request = self._logic._prepare_batch_get_chunks_request(
            sources,
            folder_name,
            None,
            use_colpali,
            output_format,
        )
        response = self._request("POST", "batch/chunks", data=request)
        return self._logic._parse_chunk_result_list_response(response)

    def create_graph(
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

        """
        # Convert prompt_overrides to dict if it's a model
        if prompt_overrides and isinstance(prompt_overrides, GraphPromptOverrides):
            prompt_overrides = prompt_overrides.model_dump(exclude_none=True)

        # Initialize request with required fields
        request = {"name": name}

        # Add optional fields only if they are not None
        if filters is not None:
            request["filters"] = filters
        if documents is not None:
            request["documents"] = documents
        if prompt_overrides is not None:
            request["prompt_overrides"] = prompt_overrides

        response = self._request("POST", "graph/create", request)
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

    def get_graph(self, name: str) -> Graph:
        """
        Get a graph by name.

        Args:
            name: Name of the graph to retrieve

        Returns:
            Graph: The requested graph object

        """
        response = self._request("GET", f"graph/{name}")
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

    def list_graphs(self) -> List[Graph]:
        """
        List all graphs the user has access to.

        Returns:
            List[Graph]: List of graph objects

        """
        response = self._request("GET", "graph")
        graphs = self._logic._parse_graph_list_response(response)
        for g in graphs:
            g._client = self
        return graphs

    def update_graph(
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

        """
        # Convert prompt_overrides to dict if it's a model
        if prompt_overrides and isinstance(prompt_overrides, GraphPromptOverrides):
            prompt_overrides = prompt_overrides.model_dump(exclude_none=True)

        request = {
            "additional_filters": additional_filters,
            "additional_documents": additional_documents,
            "prompt_overrides": prompt_overrides,
        }

        response = self._request("POST", f"graph/{name}/update", request)
        graph = self._logic._parse_graph_response(response)
        graph._client = self
        return graph

    def delete_document(self, document_id: str) -> Dict[str, str]:
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

        """
        response = self._request("DELETE", f"documents/{document_id}")
        return response

    def delete_document_by_filename(self, filename: str) -> Dict[str, str]:
        """
        Delete a document by its filename.

        This is a convenience method that first retrieves the document ID by filename
        and then deletes the document by ID.

        Args:
            filename: Filename of the document to delete

        Returns:
            Dict[str, str]: Deletion status

        """
        # First get the document by filename to obtain its ID
        doc = self.get_document_by_filename(filename)

        # Then delete the document by ID
        return self.delete_document(doc.external_id)

    def close(self):
        """Close the HTTP client"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def wait_for_graph_completion(
        self,
        graph_name: str,
        timeout_seconds: int = 300,
        check_interval_seconds: int = 2,
    ) -> Graph:
        """Block until the specified graph finishes processing.

        Args:
            graph_name: Name of the graph to monitor.
            timeout_seconds: Maximum seconds to wait.
            check_interval_seconds: Seconds between status checks.

        Returns:
            Graph: The completed graph object.
        """
        import time

        start = time.time()
        while time.time() - start < timeout_seconds:
            graph = self.get_graph(graph_name)
            if graph.is_completed:
                return graph
            if graph.is_failed:
                raise RuntimeError(graph.error or "Graph processing failed")
            time.sleep(check_interval_seconds)
        raise TimeoutError("Timed out waiting for graph completion")

    def ping(self) -> Dict[str, Any]:
        """Simple health-check call to the server (``/ping``).

        Returns
        -------
        Dict[str, Any]
            The JSON payload returned by the server, typically
            ``{"status": "ok", "message": "Server is running"}``.
        """
        return self._request("GET", "ping")

    # ------------------------------------------------------------------
    # Internal scoped helper execution
    # ------------------------------------------------------------------
    def _execute_scoped_operation(
        self,
        method: str,
        endpoint: str,
        *,
        parser: Callable[[Any], Any],
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        cleanup: Optional[Callable[[], None]] = None,
    ) -> Any:
        try:
            response = self._request(method, endpoint, data=data, files=files, params=params)
            return parser(response)
        finally:
            if cleanup:
                cleanup()

    # ------------------------------------------------------------------
    # Chat API ----------------------------------------------------------
    # ------------------------------------------------------------------
    def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Return the full message history for the given *chat_id*.

        Parameters
        ----------
        chat_id:
            Identifier of the chat conversation returned by previous
            calls that used ``chat_id``.
        """
        return self._request("GET", f"chat/{chat_id}")

    def list_chat_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent chat conversations available to the current user.

        Parameters
        ----------
        limit:
            Maximum number of conversations to return (1-500).
        """
        limit_capped = max(1, min(limit, 500))
        return self._request("GET", "chats", params={"limit": limit_capped})

    # ------------------------------------------------------------------
    # Graph helpers -----------------------------------------------------
    # ------------------------------------------------------------------
    def get_graph_visualization(
        self,
        name: str,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch nodes & links for visualising *name* graph."""
        params: Dict[str, Any] = {}
        if folder_name is not None:
            params["folder_name"] = folder_name
        if end_user_id is not None:
            params["end_user_id"] = end_user_id
        return self._request("GET", f"graph/{name}/visualization", params=params)

    def check_workflow_status(self, workflow_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Poll the status of an async graph build/update workflow."""

        params = {"run_id": run_id} if run_id else None
        return self._request("GET", f"graph/workflow/{workflow_id}/status", params=params)

    def get_graph_status(
        self, graph_name: str, folder_name: Optional[str] = None, end_user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get the current status of a graph with pipeline stage information.

        This is a lightweight endpoint that examines local status metadata and
        augments it with remote pipeline details when available.

        Args:
            graph_name: Name of the graph to check
            folder_name: Optional folder name for scoping
            end_user_id: Optional end user ID for scoping

        Returns:
            Dict containing status, pipeline_stage (if processing), and other metadata
        """
        params = {}
        if folder_name:
            params["folder_name"] = folder_name
        if end_user_id:
            params["end_user_id"] = end_user_id

        return self._request("GET", f"graph/{graph_name}/status", params=params if params else None)

    # ------------------------------------------------------------------
    # Document download helpers ----------------------------------------
    # ------------------------------------------------------------------
    def get_document_download_url(self, document_id: str, expires_in: int = 3600) -> Dict[str, Any]:
        """Generate a presigned download URL for a document stored remotely."""
        return self._request("GET", f"documents/{document_id}/download_url", params={"expires_in": expires_in})

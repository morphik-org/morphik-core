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


class Cache:
    def __init__(self, db: "Morphik", name: str):
        self._db = db
        self._name = name

    def update(self) -> bool:
        response = self._db._request("POST", f"cache/{self._name}/update")
        return response.get("success", False)

    def add_docs(self, docs: List[str]) -> bool:
        response = self._db._request("POST", f"cache/{self._name}/add_docs", {"docs": docs})
        return response.get("success", False)

    def query(
        self, query: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> CompletionResponse:
        response = self._db._request(
            "POST",
            f"cache/{self._name}/query",
            params={"query": query, "max_tokens": max_tokens, "temperature": temperature},
            data="",
        )
        return CompletionResponse(**response)


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
            rules: Optional list of rules to apply during ingestion
            use_colpali: Whether to use ColPali-style embedding model

        Returns:
            Document: Metadata of the ingested document
        """
        rules_list = [self._client._convert_rule(r) for r in (rules or [])]
        payload = self._client._logic._prepare_ingest_text_request(
            content, filename, metadata, rules_list, use_colpali, self._name, None
        )
        response = self._client._request("POST", "ingest/text", data=payload)
        doc = self._client._logic._parse_document_response(response)
        doc._client = self._client
        return doc

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

            # use_colpali flag is included in multipart form data for consistency
            response = self._client._request(
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

            response = self._client._request(
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
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[FinalChunkResult]: List of relevant chunks
        """
        effective_folder = self._merge_folders(additional_folders)
        request = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "use_colpali": use_colpali,
            "folder_name": effective_folder,
        }

        response = self._client._request("POST", "retrieve/chunks", request)
        return self._client._logic._parse_chunk_result_list_response(response)

    def retrieve_docs(
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
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[DocumentResult]: List of relevant documents
        """
        effective_folder = self._merge_folders(additional_folders)
        request = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "use_colpali": use_colpali,
            "folder_name": effective_folder,
        }

        response = self._client._request("POST", "retrieve/docs", request)
        return self._client._logic._parse_document_result_list_response(response)

    def query(
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
            additional_folders: Optional list of extra folders to include in the scope
            schema: Optional schema for structured output

        Returns:
            CompletionResponse: Generated completion
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
            None,  # end_user_id not supported at this level
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

        response = self._client._request("POST", "query", data=payload)
        return self._client._logic._parse_completion_response(response)

    def list_documents(
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
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[Document]: List of documents
        """
        effective_folder = self._merge_folders(additional_folders)
        params, data = self._client._logic._prepare_list_documents_request(skip, limit, filters, effective_folder, None)
        response = self._client._request("POST", "documents", data=data, params=params)
        docs = self._client._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self._client
        return docs

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
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in this folder.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number
            additional_folders: Optional list of extra folders to include in the scope
            use_colpali: Whether to request multimodal chunks when available

        Returns:
            List[FinalChunkResult]: List of chunk results
        """
        merged = self._merge_folders(additional_folders)
        request = self._client._logic._prepare_batch_get_chunks_request(sources, merged, None, use_colpali)

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
        response = self._client._request("POST", "ingest/text", data=payload)
        doc = self._client._logic._parse_document_response(response)
        doc._client = self._client
        return doc

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
            form_data = {
                "metadata": json.dumps(metadata or {}),
                "rules": json.dumps([self._client._convert_rule(r) for r in (rules or [])]),
                "end_user_id": self._end_user_id,
                "use_colpali": str(use_colpali).lower(),
            }

            if self._folder_name:
                form_data["folder_name"] = self._folder_name

            response = self._client._request(
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
                "end_user_id": self._end_user_id,  # Add end user ID here
                "use_colpali": str(use_colpali).lower(),
            }

            # Add folder name if scoped to a folder
            if self._folder_name:
                data["folder_name"] = self._folder_name

            response = self._client._request(
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
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[FinalChunkResult]: List of relevant chunks
        """
        effective_folder = self._merge_folders(additional_folders)
        request = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "use_colpali": use_colpali,
            "end_user_id": self._end_user_id,  # Add end user ID here
            "folder_name": effective_folder,  # Add folder name if provided
        }

        # Add folder name if scoped to a folder
        if self._folder_name:
            request["folder_name"] = self._folder_name

        response = self._client._request("POST", "retrieve/chunks", request)
        return self._client._logic._parse_chunk_result_list_response(response)

    def retrieve_docs(
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
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[DocumentResult]: List of relevant documents
        """
        effective_folder = self._merge_folders(additional_folders)
        request = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "use_colpali": use_colpali,
            "end_user_id": self._end_user_id,  # Add end user ID here
            "folder_name": effective_folder,  # Add folder name if provided
        }

        # Add folder name if scoped to a folder
        if self._folder_name:
            request["folder_name"] = self._folder_name

        response = self._client._request("POST", "retrieve/docs", request)
        return self._client._logic._parse_document_result_list_response(response)

    def query(
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
            additional_folders: Optional list of extra folders to include in the scope
            schema: Optional schema for structured output

        Returns:
            CompletionResponse: Generated completion
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

        response = self._client._request("POST", "query", data=payload)
        return self._client._logic._parse_completion_response(response)

    def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        additional_folders: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        List accessible documents for this end user.

        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Optional filters
            additional_folders: Optional list of extra folders to include in the scope

        Returns:
            List[Document]: List of documents
        """
        # Add end_user_id and folder_name to params
        params = {"skip": skip, "limit": limit, "end_user_id": self._end_user_id}

        # Add folder name if scoped to a folder
        if self._folder_name:
            params["folder_name"] = self._folder_name

        # Merge any additional folders into the request params
        effective_folder = self._merge_folders(additional_folders)
        if effective_folder:
            params["folder_name"] = effective_folder

        response = self._client._request("POST", "documents", data=filters or {}, params=params)

        docs = [self._client._logic._parse_document_response(doc) for doc in response]
        for doc in docs:
            doc._client = self._client
        return docs

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
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in this folder.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number
            additional_folders: Optional list of extra folders to include in the scope
            use_colpali: Whether to request multimodal chunks when available

        Returns:
            List[FinalChunkResult]: List of chunk results
        """
        merged = self._merge_folders(additional_folders)
        request = self._client._logic._prepare_batch_get_chunks_request(sources, merged, None, use_colpali)

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


class Morphik:
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

    def get_folder_by_name(self, name: str) -> Folder:
        """
        Get a folder by name to scope operations.

        Args:
            name: The name of the folder

        Returns:
            Folder: A folder object for scoped operations
        """
        return Folder(self, name)

    def get_folder(self, folder_id: str) -> Folder:
        """
        Get a folder by ID.

        Args:
            folder_id: ID of the folder

        Returns:
            Folder: A folder object for scoped operations
        """
        response = self._request("GET", f"folders/{folder_id}")
        return Folder(self, response["name"], folder_id)

    def list_folders(self) -> List[Folder]:
        """
        List all folders the user has access to as Folder objects.

        Returns:
            List[Folder]: List of Folder objects ready for operations
        """
        folder_infos = self._request("GET", "folders")
        return [Folder(self, info["name"], info["id"]) for info in folder_infos]

    def add_document_to_folder(self, folder_id: str, document_id: str) -> Dict[str, str]:
        """
        Add a document to a folder.

        Args:
            folder_id: ID of the folder
            document_id: ID of the document

        Returns:
            Dict[str, str]: Success status
        """
        response = self._request("POST", f"folders/{folder_id}/documents/{document_id}")
        return response

    def remove_document_from_folder(self, folder_id: str, document_id: str) -> Dict[str, str]:
        """
        Remove a document from a folder.

        Args:
            folder_id: ID of the folder
            document_id: ID of the document

        Returns:
            Dict[str, str]: Success status
        """
        response = self._request("DELETE", f"folders/{folder_id}/documents/{document_id}")
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

            doc = db.ingest_text(
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
        response = self._request("POST", "ingest/text", data=payload)
        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

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
            rules: Optional list of rules to apply during ingestion. Can be:
                  - MetadataExtractionRule: Extract metadata using a schema
                  - NaturalLanguageRule: Transform content using natural language
            use_colpali: Whether to use ColPali-style embedding model to ingest the file
                (slower, but significantly better retrieval accuracy for images)

        Returns:
            Document: Metadata of the ingested document

        Example:
            ```python
            from morphik.rules import MetadataExtractionRule, NaturalLanguageRule
            from pydantic import BaseModel

            class DocumentInfo(BaseModel):
                title: str
                author: str
                department: str

            doc = db.ingest_file(
                "document.pdf",
                filename="document.pdf",
                metadata={"category": "research"},
                rules=[
                    MetadataExtractionRule(schema=DocumentInfo),
                    NaturalLanguageRule(prompt="Extract key points only")
                ], # Optional
                use_colpali=True, # Optional
            )
            ```
        """
        # Process file input
        file_obj, filename = self._logic._prepare_file_for_upload(file, filename)

        try:
            # Prepare multipart form data
            files = {"file": (filename, file_obj)}

            # Create form data
            form_data = self._logic._prepare_ingest_file_form_data(metadata, rules, None, None, use_colpali)

            # use_colpali should be a query parameter as defined in the API
            response = self._request(
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
            # Prepare form data - use_colpali should be a query parameter, not form data
            data = self._logic._prepare_ingest_files_form_data(metadata, rules, use_colpali, parallel, None, None)

            response = self._request(
                "POST",
                "ingest/files",
                data=data,
                files=file_objects,
            )

            if response.get("errors"):
                # Log errors but don't raise exception
                for error in response["errors"]:
                    logger.error(f"Failed to ingest {error['filename']}: {error['error']}")

            docs = [self._logic._parse_document_response(doc) for doc in response["documents"]]
            for doc in docs:
                doc._client = self
            return docs
        finally:
            # Clean up file objects
            for _, (_, file_obj) in file_objects:
                if isinstance(file_obj, (IOBase, BytesIO)) and not file_obj.closed:
                    file_obj.close()

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
        payload = self._logic._prepare_retrieve_chunks_request(
            query, filters, k, min_score, use_colpali, folder_name, None
        )
        response = self._request("POST", "retrieve/chunks", data=payload)
        return self._logic._parse_chunk_result_list_response(response)

    def retrieve_docs(
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
            use_colpali: Whether to use ColPali-style embedding model to retrieve the documents
                (only works for documents ingested with `use_colpali=True`)
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
        payload = self._logic._prepare_retrieve_docs_request(
            query, filters, k, min_score, use_colpali, folder_name, None
        )
        response = self._request("POST", "retrieve/docs", data=payload)
        return self._logic._parse_document_result_list_response(response)

    def query(
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
            folder_name: Optional folder name to further scope operations
            schema: Optional schema for structured output, can be a Pydantic model or a JSON schema dict
        Returns:
            CompletionResponse

        Example:
            ```python
            # Standard query
            response = db.query(
                "What are the key findings about customer satisfaction?",
                filters={"department": "research"},
                temperature=0.7
            )

            # Knowledge graph enhanced query
            response = db.query(
                "How does product X relate to customer segment Y?",
                graph_name="market_graph",
                hop_depth=2,
                include_paths=True
            )

            # With prompt customization
            from morphik.models import QueryPromptOverride, QueryPromptOverrides
            response = db.query(
                "What are the key findings?",
                prompt_overrides=QueryPromptOverrides(
                    query=QueryPromptOverride(
                        prompt_template="Answer the question in a formal, academic tone: {question}"
                    )
                )
            )

            # Or using a dictionary
            response = db.query(
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

            response = db.query(
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
        # Directly forward the supplied folder_name (may be None, str, or List[str])
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
            folder_name,
            None,  # end_user_id not supported at this level
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

        response = self._request("POST", "query", data=payload)
        return self._logic._parse_completion_response(response)

    def agent_query(self, query: str) -> Dict[str, Any]:
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
            result = db.agent_query("What are the main trends in our Q3 sales data?")
            print(result["response"])

            # Complex analysis request
            result = db.agent_query(
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
        response = self._request("POST", "agent", data=request)
        return response

    def list_documents(
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
            docs = db.list_documents(limit=10)

            # Get next page
            next_page = db.list_documents(skip=10, limit=10, filters={"department": "research"})
            ```
        """
        params, data = self._logic._prepare_list_documents_request(skip, limit, filters, folder_name, None)
        response = self._request("POST", "documents", data=data, params=params)
        docs = self._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self
        return docs

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

        Example:
            ```python
            status = db.get_document_status("doc_123")
            if status["status"] == "completed":
                print("Document processing complete")
            elif status["status"] == "failed":
                print(f"Processing failed: {status['error']}")
            else:
                print("Document still processing...")
            ```
        """
        response = self._request("GET", f"documents/{document_id}/status")
        return response

    def wait_for_document_completion(self, document_id: str, timeout_seconds=300, check_interval_seconds=2) -> Document:
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
            doc = db.ingest_file("large_document.pdf")
            try:
                completed_doc = db.wait_for_document_completion(doc.external_id)
                print(f"Processing complete! Document has {len(completed_doc.chunk_ids)} chunks")
            except TimeoutError:
                print("Processing is taking too long")
            except ValueError as e:
                print(f"Processing failed: {e}")
            ```
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

        Example:
            ```python
            doc = db.get_document_by_filename("report.pdf")
            print(f"Document ID: {doc.external_id}")
            ```
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
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Add new content to an existing document
            updated_doc = db.update_document_with_text(
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
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Add content from a file to an existing document
            updated_doc = db.update_document_with_file(
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

        Example:
            ```python
            # Update just the metadata of a document
            updated_doc = db.update_document_metadata(
                document_id="doc_123",
                metadata={"status": "reviewed", "reviewer": "Jane Smith"}
            )
            print(f"Updated metadata: {updated_doc.metadata}")
            ```
        """
        # Use the dedicated metadata update endpoint
        response = self._request("POST", f"documents/{document_id}/update_metadata", data=metadata)
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
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Add new content to an existing document identified by filename
            updated_doc = db.update_document_by_filename_with_text(
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
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata

        Example:
            ```python
            # Add content from a file to an existing document identified by filename
            updated_doc = db.update_document_by_filename_with_file(
                filename="report.pdf",
                file="path/to/update.pdf",
                metadata={"status": "updated"},
                update_strategy="add"
            )
            print(f"Document version: {updated_doc.system_metadata.get('version')}")
            ```
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

        Example:
            ```python
            # Update just the metadata of a document identified by filename
            updated_doc = db.update_document_by_filename_metadata(
                filename="report.pdf",
                metadata={"status": "reviewed", "reviewer": "Jane Smith"},
                new_filename="reviewed_report.pdf"  # Optional: rename the file
            )
            print(f"Updated metadata: {updated_doc.metadata}")
            ```
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
                    "rules": [],
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

        Example:
            ```python
            docs = db.batch_get_documents(["doc_123", "doc_456", "doc_789"])
            for doc in docs:
                print(f"Document {doc.external_id}: {doc.metadata.get('title')}")
            ```
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
    ) -> List[FinalChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number.

        Args:
            sources: List of ChunkSource objects or dictionaries with document_id and chunk_number
            folder_name: Optional folder name (or list of names) to scope the request

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

            chunks = db.batch_get_chunks(sources)
            for chunk in chunks:
                print(f"Chunk from {chunk.document_id}, number {chunk.chunk_number}: {chunk.content[:50]}...")
            ```
        """
        request = self._logic._prepare_batch_get_chunks_request(sources, folder_name, None, use_colpali)
        response = self._request("POST", "batch/chunks", data=request)
        return self._logic._parse_chunk_result_list_response(response)

    def create_cache(
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
            cache = db.create_cache(
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

        response = self._request("POST", "cache/create", request, params=params)
        return response

    def get_cache(self, name: str) -> Cache:
        """
        Get a cache by name.

        Args:
            name: Name of the cache to retrieve

        Returns:
            cache: A cache object that is used to interact with the cache.

        Example:
            ```python
            cache = db.get_cache("programming_cache")
            ```
        """
        response = self._request("GET", f"cache/{name}")
        if response.get("exists", False):
            return Cache(self, name)
        raise ValueError(f"Cache '{name}' not found")

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

        Example:
            ```python
            # Create a graph from documents with category="research"
            graph = db.create_graph(
                name="research_graph",
                filters={"category": "research"}
            )

            # Create a graph from specific documents
            graph = db.create_graph(
                name="custom_graph",
                documents=["doc1", "doc2", "doc3"]
            )

            # With custom entity extraction examples
            from morphik.models import EntityExtractionPromptOverride, EntityExtractionExample, GraphPromptOverrides
            graph = db.create_graph(
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

        Example:
            ```python
            # Get a graph by name
            graph = db.get_graph("finance_graph")
            print(f"Graph has {len(graph.entities)} entities and {len(graph.relationships)} relationships")
            ```
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

        Example:
            ```python
            # List all accessible graphs
            graphs = db.list_graphs()
            for graph in graphs:
                print(f"Graph: {graph.name}, Entities: {len(graph.entities)}")
            ```
        """
        response = self._request("GET", "graphs")
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

        Example:
            ```python
            # Update a graph with new documents
            updated_graph = db.update_graph(
                name="research_graph",
                additional_filters={"category": "new_research"},
                additional_documents=["doc4", "doc5"]
            )
            print(f"Graph now has {len(updated_graph.entities)} entities")

            # With entity resolution examples
            from morphik.models import EntityResolutionPromptOverride, EntityResolutionExample, GraphPromptOverrides
            updated_graph = db.update_graph(
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

        Example:
            ```python
            # Delete a document
            result = db.delete_document("doc_123")
            print(result["message"])  # Document doc_123 deleted successfully
            ```
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

        Example:
            ```python
            # Delete a document by filename
            result = db.delete_document_by_filename("report.pdf")
            print(result["message"])
            ```
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

    def create_app(self, app_id: str, name: str, expiry_days: int = 30) -> Dict[str, str]:
        """Create a new application in Morphik Cloud and obtain its auth URI.

        This wraps the enterprise endpoint ``/ee/create_app`` which
        returns a dictionary ``{\"uri\": ..., \"app_id\": ...}``.

        Parameters
        ----------
        app_id:
            Identifier for the new application.
        name:
            Human-readable application name (will be slugified by the server).
        expiry_days:
            Token validity period.  Defaults to 30 days.
        """

        payload = {"app_id": app_id, "name": name, "expiry_days": expiry_days}
        return self._request("POST", "ee/create_app", data=payload)

    def wait_for_graph_completion(
        self,
        graph_name: str,
        timeout_seconds: int = 300,
        check_interval_seconds: int = 5,
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
    # Usage API ---------------------------------------------------------
    # ------------------------------------------------------------------
    def get_usage_stats(self) -> Dict[str, int]:
        """Return cumulative usage statistics for the authenticated user."""
        return self._request("GET", "usage/stats")

    def get_recent_usage(
        self,
        operation_type: Optional[str] = None,
        since: Optional["datetime"] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent usage records with optional filtering."""
        from datetime import datetime  # Local import ensures small dependency surface

        params: Dict[str, Any] = {}
        if operation_type:
            params["operation_type"] = operation_type
        if since:
            # Accept either ``str`` or ``datetime`` for *since*
            params["since"] = since.isoformat() if isinstance(since, datetime) else str(since)
        if status:
            params["status"] = status
        return self._request("GET", "usage/recent", params=params)

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
        """Poll the status of an asynchronous graph build/update workflow."""
        params = {"run_id": run_id} if run_id else None
        return self._request("GET", f"graph/workflow/{workflow_id}/status", params=params)

    # ------------------------------------------------------------------
    # Document download helpers ----------------------------------------
    # ------------------------------------------------------------------
    def get_document_download_url(self, document_id: str, expires_in: int = 3600) -> Dict[str, Any]:
        """Generate a presigned download URL for a document stored remotely."""
        return self._request("GET", f"documents/{document_id}/download_url", params={"expires_in": expires_in})

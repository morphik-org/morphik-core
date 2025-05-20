import base64
import io
import json
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse

import jwt
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel, Field

from .models import ChunkSource  # Prompt override models
from .models import (
    ChunkResult,
    CompletionResponse,
    Document,
    DocumentResult,
    Graph,
    GraphPromptOverrides,
    IngestTextRequest,
)
from .rules import Rule

# Type alias for rules
RuleOrDict = Union[Rule, Dict[str, Any]]


class FinalChunkResult(BaseModel):
    content: str | PILImage = Field(..., description="Chunk content")
    score: float = Field(..., description="Relevance score")
    document_id: str = Field(..., description="Parent document ID")
    chunk_number: int = Field(..., description="Chunk sequence number")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    content_type: str = Field(..., description="Content type")
    filename: Optional[str] = Field(None, description="Original filename")
    download_url: Optional[str] = Field(None, description="URL to download full document")

    class Config:
        arbitrary_types_allowed = True


class _MorphikClientLogic:
    """
    Internal shared logic for Morphik clients.

    This class contains the shared logic between synchronous and asynchronous clients.
    It handles URL generation, request preparation, and response parsing.
    """

    def __init__(self, uri: Optional[str] = None, timeout: int = 30, is_local: bool = False):
        """Initialize shared client logic"""
        self._timeout = timeout
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

    def _convert_rule(self, rule: RuleOrDict) -> Dict[str, Any]:
        """Convert a rule to a dictionary format"""
        if hasattr(rule, "to_dict"):
            return rule.to_dict()
        return rule

    def _get_url(self, endpoint: str) -> str:
        """Get the full URL for an API endpoint"""
        return f"{self._base_url}/{endpoint.lstrip('/')}"

    def _get_headers(self) -> Dict[str, str]:
        """Get base headers for API requests"""
        headers = {"Content-Type": "application/json"}
        return headers

    # Request preparation methods

    def _prepare_ingest_text_request(
        self,
        content: str,
        filename: Optional[str],
        metadata: Optional[Dict[str, Any]],
        rules: Optional[List[RuleOrDict]],
        use_colpali: bool,
        folder_name: Optional[str],
        end_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare request for ingest_text endpoint"""
        rules_dict = [self._convert_rule(r) for r in (rules or [])]
        payload = {
            "content": content,
            "filename": filename,
            "metadata": metadata or {},
            "rules": rules_dict,
            "use_colpali": use_colpali,
        }
        if folder_name:
            payload["folder_name"] = folder_name
        if end_user_id:
            payload["end_user_id"] = end_user_id
        return payload

    def _prepare_file_for_upload(
        self,
        file: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
    ) -> Tuple[BinaryIO, str]:
        """
        Process file input and return file object and filename.
        Handles different file input types (str, Path, bytes, file-like object).
        """
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

        return file_obj, filename

    def _prepare_files_for_upload(
        self,
        files: List[Union[str, bytes, BinaryIO, Path]],
    ) -> List[Tuple[str, Tuple[str, BinaryIO]]]:
        """
        Process multiple files and return a list of file objects in the format
        expected by the API: [("files", (filename, file_obj)), ...]
        """
        file_objects = []
        for file in files:
            if isinstance(file, (str, Path)):
                path = Path(file)
                file_objects.append(("files", (path.name, open(path, "rb"))))
            elif isinstance(file, bytes):
                file_objects.append(("files", ("file.bin", BytesIO(file))))
            else:
                file_objects.append(("files", (getattr(file, "name", "file.bin"), file)))

        return file_objects

    def _prepare_ingest_file_form_data(
        self,
        metadata: Optional[Dict[str, Any]],
        rules: Optional[List[RuleOrDict]],
        folder_name: Optional[str],
        end_user_id: Optional[str],
        use_colpali: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Prepare form data for ingest_file endpoint.

        All parameters are included in the multipart body so that the server
        never relies on query-string values.  *use_colpali* is therefore always
        embedded here when provided.
        """
        form_data = {
            "metadata": json.dumps(metadata or {}),
            "rules": json.dumps([self._convert_rule(r) for r in (rules or [])]),
        }
        if folder_name:
            form_data["folder_name"] = folder_name
        if end_user_id:
            form_data["end_user_id"] = end_user_id

        # Only include the flag when caller supplied a specific value to avoid
        # overriding server defaults unintentionally.
        if use_colpali is not None:
            form_data["use_colpali"] = str(use_colpali).lower()

        return form_data

    def _prepare_ingest_files_form_data(
        self,
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
        rules: Optional[List[RuleOrDict]],
        use_colpali: bool,
        parallel: bool,
        folder_name: Optional[str],
        end_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare form data for ingest_files endpoint"""
        # Convert rules appropriately based on whether it's a flat list or list of lists
        if rules:
            if all(isinstance(r, list) for r in rules):
                # List of lists - per-file rules
                converted_rules = [[self._convert_rule(r) for r in rule_list] for rule_list in rules]
            else:
                # Flat list - shared rules for all files
                converted_rules = [self._convert_rule(r) for r in rules]
        else:
            converted_rules = []

        data = {
            "metadata": json.dumps(metadata or {}),
            "rules": json.dumps(converted_rules),
            "parallel": str(parallel).lower(),
        }

        # Always carry use_colpali in the body for consistency with single-file
        # ingestion.  The API treats missing values as "true" for backward
        # compatibility, hence we only add it when explicitly provided.
        if use_colpali is not None:
            data["use_colpali"] = str(use_colpali).lower()

        if folder_name:
            data["folder_name"] = folder_name
        if end_user_id:
            data["end_user_id"] = end_user_id

        return data

    def _prepare_query_request(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        k: int,
        min_score: float,
        max_tokens: Optional[int],
        temperature: Optional[float],
        use_colpali: bool,
        graph_name: Optional[str],
        hop_depth: int,
        include_paths: bool,
        prompt_overrides: Optional[Dict],
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
        chat_id: Optional[str] = None,
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Prepare request for query endpoint"""
        payload = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_colpali": use_colpali,
            "graph_name": graph_name,
            "hop_depth": hop_depth,
            "include_paths": include_paths,
            "prompt_overrides": prompt_overrides,
        }
        if folder_name:
            payload["folder_name"] = folder_name
        if end_user_id:
            payload["end_user_id"] = end_user_id
        if chat_id:
            payload["chat_id"] = chat_id

        # Add schema to payload if provided
        if schema:
            # If schema is a Pydantic model class, serialize it to a JSON schema dict
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                payload["schema"] = schema.model_json_schema()
            elif isinstance(schema, dict):
                # Basic check if it looks like a JSON schema (has 'properties' or 'type')
                if "properties" not in schema and "type" not in schema:
                    raise ValueError("Provided schema dictionary does not look like a valid JSON schema")
                payload["schema"] = schema
            else:
                raise TypeError("schema must be a Pydantic model type or a dictionary representing a JSON schema")

        # Filter out None values before sending
        return {k_p: v_p for k_p, v_p in payload.items() if v_p is not None}

    def _prepare_retrieve_chunks_request(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        k: int,
        min_score: float,
        use_colpali: bool,
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare request for retrieve_chunks endpoint"""
        request = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "use_colpali": use_colpali,
        }
        if folder_name:
            request["folder_name"] = folder_name
        if end_user_id:
            request["end_user_id"] = end_user_id
        return request

    def _prepare_retrieve_docs_request(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        k: int,
        min_score: float,
        use_colpali: bool,
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare request for retrieve_docs endpoint"""
        request = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "use_colpali": use_colpali,
        }
        if folder_name:
            request["folder_name"] = folder_name
        if end_user_id:
            request["end_user_id"] = end_user_id
        return request

    def _prepare_list_documents_request(
        self,
        skip: int,
        limit: int,
        filters: Optional[Dict[str, Any]],
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare request for list_documents endpoint"""
        params = {
            "skip": skip,
            "limit": limit,
        }
        if folder_name:
            params["folder_name"] = folder_name
        if end_user_id:
            params["end_user_id"] = end_user_id
        data = filters or {}
        return params, data

    def _prepare_batch_get_documents_request(
        self, document_ids: List[str], folder_name: Optional[Union[str, List[str]]], end_user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare request for batch_get_documents endpoint"""
        if folder_name or end_user_id:
            request = {"document_ids": document_ids}
            if folder_name:
                request["folder_name"] = folder_name
            if end_user_id:
                request["end_user_id"] = end_user_id
            return request
        return document_ids  # Return just IDs list if no scoping is needed

    def _prepare_batch_get_chunks_request(
        self,
        sources: List[Union[ChunkSource, Dict[str, Any]]],
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
        use_colpali: bool = True,
    ) -> Dict[str, Any]:
        """Prepare request for batch_get_chunks endpoint"""
        source_dicts = []
        for source in sources:
            if isinstance(source, dict):
                source_dicts.append(source)
            else:
                source_dicts.append(source.model_dump())

        # Always include use_colpali flag so the server can decide how to
        # enrich chunks.  Keep any additional scoping parameters.
        request: Dict[str, Any] = {"sources": source_dicts, "use_colpali": use_colpali}
        if folder_name:
            request["folder_name"] = folder_name
        if end_user_id:
            request["end_user_id"] = end_user_id
        return request

    def _prepare_create_graph_request(
        self,
        name: str,
        filters: Optional[Dict[str, Any]],
        documents: Optional[List[str]],
        prompt_overrides: Optional[Union[GraphPromptOverrides, Dict[str, Any]]],
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare request for create_graph endpoint"""
        # Convert prompt_overrides to dict if it's a model
        if prompt_overrides and isinstance(prompt_overrides, GraphPromptOverrides):
            prompt_overrides = prompt_overrides.model_dump(exclude_none=True)

        request = {
            "name": name,
            "filters": filters,
            "documents": documents,
            "prompt_overrides": prompt_overrides,
        }
        if folder_name:
            request["folder_name"] = folder_name
        if end_user_id:
            request["end_user_id"] = end_user_id
        return request

    def _prepare_update_graph_request(
        self,
        name: str,
        additional_filters: Optional[Dict[str, Any]],
        additional_documents: Optional[List[str]],
        prompt_overrides: Optional[Union[GraphPromptOverrides, Dict[str, Any]]],
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare request for update_graph endpoint"""
        # Convert prompt_overrides to dict if it's a model
        if prompt_overrides and isinstance(prompt_overrides, GraphPromptOverrides):
            prompt_overrides = prompt_overrides.model_dump(exclude_none=True)

        request = {
            "additional_filters": additional_filters,
            "additional_documents": additional_documents,
            "prompt_overrides": prompt_overrides,
        }
        if folder_name:
            request["folder_name"] = folder_name
        if end_user_id:
            request["end_user_id"] = end_user_id
        return request

    def _prepare_update_document_with_text_request(
        self,
        document_id: str,
        content: str,
        filename: Optional[str],
        metadata: Optional[Dict[str, Any]],
        rules: Optional[List],
        update_strategy: str,
        use_colpali: Optional[bool],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare request for update_document_with_text endpoint"""
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

        return params, request.model_dump()

    # Response parsing methods

    def _parse_document_response(self, response_json: Dict[str, Any]) -> Document:
        """Parse document response"""
        return Document(**response_json)

    def _parse_completion_response(self, response_json: Dict[str, Any]) -> CompletionResponse:
        """Parse completion response"""
        return CompletionResponse(**response_json)

    def _parse_document_list_response(self, response_json: List[Dict[str, Any]]) -> List[Document]:
        """Parse document list response"""
        docs = [Document(**doc) for doc in response_json]
        return docs

    def _parse_document_result_list_response(self, response_json: List[Dict[str, Any]]) -> List[DocumentResult]:
        """Parse document result list response"""
        return [DocumentResult(**r) for r in response_json]

    def _parse_chunk_result_list_response(self, response_json: List[Dict[str, Any]]) -> List[FinalChunkResult]:
        """Parse chunk result list response"""
        chunks = [ChunkResult(**r) for r in response_json]

        final_chunks = []
        for chunk in chunks:
            content = chunk.content
            if chunk.metadata.get("is_image"):
                try:
                    # Handle data URI format "data:image/png;base64,..."
                    if content.startswith("data:"):
                        # Extract the base64 part after the comma
                        content = content.split(",", 1)[1]

                    # Now decode the base64 string
                    image_bytes = base64.b64decode(content)
                    content = Image.open(io.BytesIO(image_bytes))
                except Exception:
                    # Fall back to using the content as text
                    content = chunk.content

            final_chunks.append(
                FinalChunkResult(
                    content=content,
                    score=chunk.score,
                    document_id=chunk.document_id,
                    chunk_number=chunk.chunk_number,
                    metadata=chunk.metadata,
                    content_type=chunk.content_type,
                    filename=chunk.filename,
                    download_url=chunk.download_url,
                )
            )

        return final_chunks

    def _parse_graph_response(self, response_json: Dict[str, Any]) -> Graph:
        """Parse graph response"""
        return Graph(**response_json)

    def _parse_graph_list_response(self, response_json: List[Dict[str, Any]]) -> List[Graph]:
        """Parse graph list response"""
        return [Graph(**graph) for graph in response_json]

import base64
import io
import json
import warnings
from datetime import date, datetime, timezone
from decimal import Decimal
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
    DocumentQueryResponse,
    DocumentResult,
    Graph,
    GraphPromptOverrides,
    IngestTextRequest,
    QueryPromptOverrides,
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

    def _warn_legacy_rules(self, rules: Optional[List[RuleOrDict]], context: str) -> None:
        """Emit a deprecation warning when legacy rules are supplied."""
        if rules:
            warnings.warn(
                f"'rules' support has been removed; payload supplied to {context} is ignored.",
                DeprecationWarning,
                stacklevel=3,
            )

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
        self._warn_legacy_rules(rules, "ingest/text")
        serialized_metadata, metadata_types_map = self._serialize_metadata_map(metadata)
        payload = {
            "content": content,
            "filename": filename,
            "metadata": serialized_metadata,
            "use_colpali": use_colpali,
        }
        if folder_name:
            payload["folder_name"] = folder_name
        if end_user_id:
            payload["end_user_id"] = end_user_id
        if metadata_types_map:
            payload["metadata_types"] = metadata_types_map
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
        self._warn_legacy_rules(rules, "ingest/file")
        serialized_metadata, metadata_types_map = self._serialize_metadata_map(metadata)
        form_data = {
            "metadata": json.dumps(serialized_metadata),
        }
        if folder_name:
            form_data["folder_name"] = folder_name
        if end_user_id:
            form_data["end_user_id"] = end_user_id

        # Only include the flag when caller supplied a specific value to avoid
        # overriding server defaults unintentionally.
        if use_colpali is not None:
            form_data["use_colpali"] = str(use_colpali).lower()

        if metadata_types_map:
            form_data["metadata_types"] = json.dumps(metadata_types_map)

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
        self._warn_legacy_rules(rules, "ingest/files")

        serialized_metadata, metadata_types_payload = self._serialize_metadata_collection(metadata)

        data = {
            "metadata": json.dumps(serialized_metadata),
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
        if metadata_types_payload is not None:
            data["metadata_types"] = json.dumps(metadata_types_payload)

        return data

    def _prepare_document_query_form_data(
        self,
        prompt: str,
        schema: Optional[Union[Dict[str, Any], BaseModel, Type[BaseModel], str]],
        ingestion_options: Optional[Dict[str, Any]],
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare form data for the document query endpoint."""
        form_data: Dict[str, Any] = {"prompt": prompt}

        if schema is not None:
            if isinstance(schema, str):
                form_data["schema"] = schema
            elif isinstance(schema, type) and issubclass(schema, BaseModel):
                form_data["schema"] = json.dumps(schema.model_json_schema())
            elif isinstance(schema, BaseModel):
                form_data["schema"] = json.dumps(schema.model_dump(exclude_none=True))
            else:
                form_data["schema"] = json.dumps(schema)

        options: Dict[str, Any] = dict(ingestion_options or {})
        if folder_name is not None and "folder_name" not in options:
            options["folder_name"] = folder_name
        if end_user_id is not None and "end_user_id" not in options:
            options["end_user_id"] = end_user_id

        form_data["ingestion_options"] = json.dumps(options)
        return form_data

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
        prompt_overrides: Optional[Union[QueryPromptOverrides, Dict[str, Any]]],
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
        use_reranking: Optional[bool] = None,  # Add missing parameter
        chat_id: Optional[str] = None,
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        padding: int = 0,
    ) -> Dict[str, Any]:
        """Prepare request for query endpoint"""
        # Convert prompt_overrides to dict if it's a model
        if prompt_overrides and isinstance(prompt_overrides, QueryPromptOverrides):
            prompt_overrides = prompt_overrides.model_dump(exclude_none=True)

        payload = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_colpali": use_colpali,
            "use_reranking": use_reranking,  # Add to payload
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
        if llm_config:
            payload["llm_config"] = llm_config
        if padding > 0:
            payload["padding"] = padding

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
        padding: int = 0,
        output_format: Optional[str] = None,
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
        if padding > 0:
            request["padding"] = padding
        if output_format:
            request["output_format"] = output_format
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
        use_reranking: Optional[bool] = None,  # Add missing parameter
    ) -> Dict[str, Any]:
        """Prepare request for retrieve_docs endpoint"""
        request = {
            "query": query,
            "filters": filters,
            "k": k,
            "min_score": min_score,
            "use_colpali": use_colpali,
            "use_reranking": use_reranking,  # Add to payload
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
        include_total_count: bool,
        include_status_counts: bool,
        include_folder_counts: bool,
        completed_only: bool,
        sort_by: Optional[str],
        sort_direction: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare request for list_docs endpoint"""
        params = {}
        if folder_name:
            params["folder_name"] = folder_name
        if end_user_id:
            params["end_user_id"] = end_user_id

        data = {
            "skip": skip,
            "limit": limit,
            "document_filters": filters,
            "return_documents": True,
            "include_total_count": include_total_count,
            "include_status_counts": include_status_counts,
            "include_folder_counts": include_folder_counts,
            "completed_only": completed_only,
            "sort_by": sort_by,
            "sort_direction": sort_direction,
        }
        return params, data

    def _prepare_batch_get_documents_request(
        self, document_ids: List[str], folder_name: Optional[Union[str, List[str]]], end_user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare request for batch_get_documents endpoint"""
        request = {"document_ids": document_ids}
        if folder_name:
            request["folder_name"] = folder_name
        if end_user_id:
            request["end_user_id"] = end_user_id
        return request

    def _prepare_batch_get_chunks_request(
        self,
        sources: List[Union[ChunkSource, Dict[str, Any]]],
        folder_name: Optional[Union[str, List[str]]],
        end_user_id: Optional[str],
        use_colpali: bool = True,
        output_format: Optional[str] = None,
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
        if output_format:
            request["output_format"] = output_format
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
        self._warn_legacy_rules(rules, "documents/update_text")

        serialized_metadata, metadata_types_map = self._serialize_metadata_map(metadata)
        request = IngestTextRequest(
            content=content,
            filename=filename,
            metadata=serialized_metadata,
            use_colpali=use_colpali if use_colpali is not None else True,
            metadata_types=metadata_types_map or None,
        )

        params = {}
        if update_strategy != "add":
            params["update_strategy"] = update_strategy

        return params, request.model_dump()

    # ------------------------------------------------------------------
    # Metadata serialization helpers
    # ------------------------------------------------------------------

    def _serialize_metadata_map(self, metadata: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Normalize metadata values and build a type map for top-level fields."""
        serialized: Dict[str, Any] = {}
        type_map: Dict[str, str] = {}
        source = metadata or {}

        for key, value in source.items():
            normalized_value, type_name = self._normalize_metadata_value(value)
            serialized[key] = normalized_value
            if type_name:
                type_map[key] = type_name

        return serialized, type_map

    def _serialize_metadata_collection(
        self, metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
    ) -> Tuple[Union[Dict[str, Any], List[Dict[str, Any]]], Optional[Union[Dict[str, str], List[Dict[str, str]]]]]:
        """Serialize metadata for single or batched ingestion requests."""
        if isinstance(metadata, list):
            serialized_items: List[Dict[str, Any]] = []
            type_maps: List[Dict[str, str]] = []
            has_types = False
            for item in metadata:
                normalized, type_map = self._serialize_metadata_map(item)
                serialized_items.append(normalized)
                type_maps.append(type_map)
                has_types = has_types or bool(type_map)
            return serialized_items, type_maps if has_types else None

        normalized, type_map = self._serialize_metadata_map(metadata)
        return normalized, type_map if type_map else None

    def _normalize_metadata_value(self, value: Any) -> Tuple[Any, Optional[str]]:
        """Coerce a metadata value into a JSON-serializable form with optional type info."""
        if value is None:
            return None, None
        if isinstance(value, bool):
            return value, "boolean"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value, "number"
        if isinstance(value, Decimal):
            return self._format_decimal(value), "decimal"
        if isinstance(value, datetime):
            return self._format_datetime(value), "datetime"
        if isinstance(value, date):
            return value.isoformat(), "date"
        if isinstance(value, list):
            return [self._sanitize_nested_metadata(item) for item in value], None
        if isinstance(value, dict):
            return {k: self._sanitize_nested_metadata(v) for k, v in value.items()}, None
        return value, None

    def _sanitize_nested_metadata(self, value: Any) -> Any:
        """Recursively sanitize nested metadata structures."""
        if isinstance(value, datetime):
            return self._format_datetime(value)
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, Decimal):
            return self._format_decimal(value)
        if isinstance(value, list):
            return [self._sanitize_nested_metadata(item) for item in value]
        if isinstance(value, dict):
            return {k: self._sanitize_nested_metadata(v) for k, v in value.items()}
        return value

    def _format_datetime(self, value: datetime) -> str:
        """Return an ISO 8601 string for datetime values, assuming UTC when naive."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    def _format_decimal(self, value: Decimal) -> str:
        """Serialize Decimal values without introducing binary floating point errors."""
        normalized = value.normalize()
        as_str = format(normalized, "f")
        if "." in as_str:
            as_str = as_str.rstrip("0").rstrip(".")
        return as_str or "0"

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

    def _parse_document_query_response(self, response_json: Dict[str, Any]) -> DocumentQueryResponse:
        """Parse document query response."""
        payload = dict(response_json)
        ingestion_document = payload.get("ingestion_document")
        if isinstance(ingestion_document, dict):
            payload["ingestion_document"] = Document(**ingestion_document)
        return DocumentQueryResponse(**payload)

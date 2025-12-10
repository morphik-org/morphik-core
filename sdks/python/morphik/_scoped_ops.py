import logging
from io import BytesIO, IOBase
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from ._internal import FinalChunkResult, RuleOrDict

T = TypeVar("T")

logger = logging.getLogger(__name__)


class _ScopedOperationsMixin:
    """Shared helpers that keep ingest/retrieve/list payloads consistent across clients."""

    _logic: Any  # Populated by Morphik/AsyncMorphik

    def _execute_scoped_operation(
        self,
        method: str,
        endpoint: str,
        parser: Callable[[Any], T],
        *,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        cleanup: Optional[Callable[[], None]] = None,
    ) -> T:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Document ingestion helpers
    # ------------------------------------------------------------------
    def _scoped_ingest_text(
        self,
        *,
        content: str,
        filename: Optional[str],
        metadata: Optional[Dict[str, Any]],
        rules: Optional[List[RuleOrDict]],
        use_colpali: bool,
        folder_name: Optional[str],
        end_user_id: Optional[str],
    ):
        payload = self._logic._prepare_ingest_text_request(
            content,
            filename,
            metadata,
            rules,
            use_colpali,
            folder_name,
            end_user_id,
        )

        return self._execute_scoped_operation(
            "POST",
            "ingest/text",
            data=payload,
            parser=self._parse_document_response,
        )

    def _scoped_ingest_file(
        self,
        *,
        file: Union[str, bytes, BytesIO, IOBase, Path],
        filename: Optional[str],
        metadata: Optional[Dict[str, Any]],
        rules: Optional[List[RuleOrDict]],
        use_colpali: bool,
        folder_name: Optional[str],
        end_user_id: Optional[str],
    ):
        file_obj, resolved_filename = self._logic._prepare_file_for_upload(file, filename)

        cleanup: Optional[Callable[[], None]] = None
        if isinstance(file, (str, Path)):
            cleanup = file_obj.close

        form_data = self._logic._prepare_ingest_file_form_data(
            metadata,
            rules,
            folder_name,
            end_user_id,
            use_colpali,
        )

        return self._execute_scoped_operation(
            "POST",
            "ingest/file",
            data=form_data,
            files={"file": (resolved_filename, file_obj)},
            parser=self._parse_document_response,
            cleanup=cleanup,
        )

    def _scoped_ingest_files(
        self,
        *,
        files: List[Union[str, bytes, BytesIO, IOBase, Path]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
        rules: Optional[List[RuleOrDict]],
        use_colpali: bool,
        parallel: bool,
        folder_name: Optional[str],
        end_user_id: Optional[str],
    ) -> List[Any]:
        file_objects = self._logic._prepare_files_for_upload(files)

        def cleanup() -> None:
            for _, (_, file_obj) in file_objects:
                if isinstance(file_obj, (IOBase, BytesIO)) and not file_obj.closed:
                    file_obj.close()

        form_data = self._logic._prepare_ingest_files_form_data(
            metadata,
            rules,
            use_colpali,
            parallel,
            folder_name,
            end_user_id,
        )

        def parser(response: Dict[str, Any]):
            if response.get("errors"):
                for error in response["errors"]:
                    logger.error(f"Failed to ingest {error['filename']}: {error['error']}")

            docs = [self._logic._parse_document_response(doc) for doc in response.get("documents", [])]
            for doc in docs:
                doc._client = self
            return docs

        return self._execute_scoped_operation(
            "POST",
            "ingest/files",
            data=form_data,
            files=file_objects,
            parser=parser,
            cleanup=cleanup,
        )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _scoped_retrieve_chunks(
        self,
        *,
        query: Optional[str],
        filters: Optional[Dict[str, Any]],
        k: int,
        min_score: float,
        use_colpali: bool,
        folder_name: Optional[Union[str, List[str]]],
        folder_depth: Optional[int],
        end_user_id: Optional[str],
        padding: int,
        output_format: Optional[str] = None,
        query_image: Optional[str] = None,
    ) -> List[FinalChunkResult]:
        payload = self._logic._prepare_retrieve_chunks_request(
            query,
            filters,
            k,
            min_score,
            use_colpali,
            folder_name,
            folder_depth,
            end_user_id,
            padding,
            output_format,
            query_image,
        )

        return self._execute_scoped_operation(
            "POST",
            "retrieve/chunks",
            data=payload,
            parser=self._logic._parse_chunk_result_list_response,
        )

    def _scoped_retrieve_docs(
        self,
        *,
        query: str,
        filters: Optional[Dict[str, Any]],
        k: int,
        min_score: float,
        use_colpali: bool,
        folder_name: Optional[Union[str, List[str]]],
        folder_depth: Optional[int],
        end_user_id: Optional[str],
        use_reranking: Optional[bool],
    ) -> List[Any]:
        payload = self._logic._prepare_retrieve_docs_request(
            query,
            filters,
            k,
            min_score,
            use_colpali,
            folder_name,
            folder_depth,
            end_user_id,
            use_reranking,
        )

        return self._execute_scoped_operation(
            "POST",
            "retrieve/docs",
            data=payload,
            parser=self._logic._parse_document_result_list_response,
        )

    def _scoped_query(
        self,
        *,
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
        prompt_overrides: Optional[Dict[str, Any]],
        folder_name: Optional[Union[str, List[str]]],
        folder_depth: Optional[int],
        end_user_id: Optional[str],
        use_reranking: Optional[bool],
        chat_id: Optional[str],
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]],
        llm_config: Optional[Dict[str, Any]],
        padding: int,
    ):
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
            folder_depth,
            end_user_id,
            use_reranking,
            chat_id,
            schema,
            llm_config,
            padding,
        )

        if schema:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                payload["schema"] = schema.model_json_schema()
            else:
                payload["schema"] = schema
            payload["query"] = f"{payload['query']}\nReturn the answer in JSON format according to the required schema."

        return self._execute_scoped_operation(
            "POST",
            "query",
            data=payload,
            parser=self._logic._parse_completion_response,
        )

    # ------------------------------------------------------------------
    # Document listing helpers
    # ------------------------------------------------------------------
    def _scoped_list_documents(
        self,
        *,
        skip: int,
        limit: int,
        filters: Optional[Dict[str, Any]],
        folder_name: Optional[Union[str, List[str]]],
        folder_depth: Optional[int],
        end_user_id: Optional[str],
        include_total_count: bool,
        include_status_counts: bool,
        include_folder_counts: bool,
        completed_only: bool,
        sort_by: Optional[str],
        sort_direction: str,
    ):
        params, data = self._logic._prepare_list_documents_request(
            skip,
            limit,
            filters,
            folder_name,
            folder_depth,
            end_user_id,
            include_total_count,
            include_status_counts,
            include_folder_counts,
            completed_only,
            sort_by,
            sort_direction,
        )

        return self._execute_scoped_operation(
            "POST",
            "documents/list_docs",
            data=data,
            params=params,
            parser=self._parse_list_docs_response,
        )

    # ------------------------------------------------------------------
    # Parsers shared across clients
    # ------------------------------------------------------------------
    def _parse_document_response(self, response: Dict[str, Any]):
        doc = self._logic._parse_document_response(response)
        doc._client = self
        return doc

    def _parse_document_list_response(self, response: List[Dict[str, Any]]):
        docs = self._logic._parse_document_list_response(response)
        for doc in docs:
            doc._client = self
        return docs

    def _parse_list_docs_response(self, response: Dict[str, Any]):
        from .models import ListDocsResponse

        result = ListDocsResponse(**response)
        for doc in result.documents:
            doc._client = self
        return result

"""
Ingestion Service - Handles all document ingestion operations.

This service is responsible for:
- File ingestion (ingest_file_content)
- Document updates (update_document)
- ColPali multi-vector chunk creation
- PDF/Image/Office document processing for visual embeddings

The service can operate in different modes based on configuration:
- Standard mode: Text embedding only
- ColPali local mode: Local torch-based visual embeddings (heavy deps)
- ColPali API mode: Remote API for visual embeddings (light deps)
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import arq
import fitz  # PyMuPDF
import pdf2image
from fastapi import HTTPException, UploadFile
from PIL import Image as PILImage

from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.limits_utils import check_and_increment_limits, estimate_pages_by_chars
from core.models.auth import AuthContext
from core.models.chunk import Chunk, DocumentChunk
from core.models.documents import Document
from core.models.folders import Folder
from core.parser.base_parser import BaseParser
from core.storage.base_storage import BaseStorage
from core.storage.utils_file_extensions import detect_content_type, detect_file_type
from core.utils.fast_ops import bytes_to_data_uri, encode_base64
from core.utils.folder_utils import normalize_folder_path, normalize_ingest_folder_inputs
from core.utils.storage_usage import extract_storage_bytes
from core.utils.typed_metadata import MetadataBundle, merge_metadata, normalize_metadata
from core.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)
settings = get_settings()


class PdfConversionError(Exception):
    """Raised when PDF conversion to images fails."""

    pass


class IngestionService:
    """
    Service for handling document ingestion operations.

    This service encapsulates all ingestion-related functionality, including:
    - File ingestion
    - Document updates
    - ColPali multi-vector processing
    - Chunk creation and storage

    The service is designed to be instantiated with only the dependencies needed
    for the current operation mode (standard vs ColPali local vs ColPali API).
    """

    _SYSTEM_METADATA_SCOPE_KEYS = {"folder_name", "folder_id", "end_user_id", "app_id"}
    _USER_IMMUTABLE_FIELDS = {
        "folder_name",
        "folder_id",
        "folder_path",
        "external_id",
        "filename",
        "app_id",
        "owner_id",
        "end_user_id",
    }

    def __init__(
        self,
        database: PostgresDatabase,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
        storage: BaseStorage,
        parser: BaseParser,
        colpali_embedding_model: Optional[BaseEmbeddingModel] = None,
        colpali_vector_store: Optional[BaseVectorStore] = None,
    ):
        """
        Initialize the IngestionService.

        Args:
            database: Database for document storage
            vector_store: Vector store for standard embeddings
            embedding_model: Embedding model for text chunks
            storage: File storage backend
            parser: Document parser for text extraction
            colpali_embedding_model: Optional ColPali embedding model (local or API)
            colpali_vector_store: Optional ColPali vector store for multi-vector embeddings
        """
        self.db = database
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.storage = storage
        self.parser = parser
        self.colpali_embedding_model = colpali_embedding_model
        self.colpali_vector_store = colpali_vector_store

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------

    def _enforce_no_user_mutable_fields(
        self,
        metadata: Optional[Dict[str, Any]],
        extra_fields: Optional[Dict[str, Any]] = None,
        metadata_types: Optional[Dict[str, Any]] = None,
        context: str = "ingest",
    ) -> None:
        """Prevent users from setting reserved system fields directly."""
        invalid_fields = set()

        if isinstance(metadata, dict):
            invalid_fields.update({key for key in metadata.keys() if key in self._USER_IMMUTABLE_FIELDS})

        if isinstance(extra_fields, dict):
            invalid_fields.update({key for key in extra_fields.keys() if key in self._USER_IMMUTABLE_FIELDS})

        if isinstance(metadata_types, dict):
            invalid_fields.update({key for key in metadata_types.keys() if key in self._USER_IMMUTABLE_FIELDS})

        if invalid_fields:
            fields_str = ", ".join(sorted(invalid_fields))
            raise ValueError(
                f"The following fields are managed by Morphik and cannot be set during {context}: {fields_str}. "
                "Remove them from the request."
            )

    @classmethod
    def _clean_system_metadata(cls, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Remove scope fields that are persisted in dedicated columns."""
        if not metadata:
            return {}

        cleaned_metadata = dict(metadata)
        for key in cls._SYSTEM_METADATA_SCOPE_KEYS:
            cleaned_metadata.pop(key, None)
        return cleaned_metadata

    @staticmethod
    def folder_update_fields(folder_obj: Folder) -> Dict[str, Any]:
        """Build a consistent update payload for folder metadata columns."""
        try:
            path_value = folder_obj.full_path or (normalize_folder_path(folder_obj.name) if folder_obj.name else None)
        except ValueError:
            path_value = folder_obj.name

        return {
            "folder_id": folder_obj.id,
            "folder_path": path_value,
            "folder_name": folder_obj.name,
        }

    # -------------------------------------------------------------------------
    # Folder management
    # -------------------------------------------------------------------------

    async def _ensure_folder_exists(
        self, folder_name: Union[str, List[str]], document_id: str, auth: AuthContext
    ) -> Optional[Folder]:
        """
        Check if a folder exists, if not create it. Also adds the document to the folder.

        Args:
            folder_name: Name of the folder
            document_id: ID of the document to add to the folder
            auth: Authentication context

        Returns:
            Folder object if found or created, None on error
        """
        try:
            # If multiple folders provided, ensure each exists and contains the document
            if isinstance(folder_name, list):
                last_folder = None
                for fname in folder_name:
                    last_folder = await self._ensure_folder_exists(fname, document_id, auth)
                return last_folder

            canonical_path = normalize_folder_path(folder_name)
            segments = canonical_path.strip("/").split("/") if canonical_path and canonical_path != "/" else []

            if canonical_path == "/":
                logger.error("Cannot ingest into root folder '/'")
                raise ValueError("Cannot ingest into root folder '/'")

            parent_id: Optional[str] = None
            current_path_parts: List[str] = []
            target_folder: Optional[Folder] = None

            for idx, segment in enumerate(segments):
                current_path_parts.append(segment)
                current_path = "/" + "/".join(current_path_parts)
                existing = await self.db.get_folder_by_full_path(current_path, auth)
                if existing:
                    parent_id = existing.id
                    if idx == len(segments) - 1:
                        target_folder = existing
                    continue

                folder_depth = idx + 1
                folder = Folder(
                    name=segment,
                    full_path=current_path,
                    parent_id=parent_id,
                    depth=folder_depth,
                    document_ids=[document_id] if idx == len(segments) - 1 else [],
                    app_id=auth.app_id,
                )
                await self.db.create_folder(folder, auth)
                parent_id = folder.id
                if idx == len(segments) - 1:
                    target_folder = folder

            if target_folder is None:
                logger.error("Failed to ensure target folder for path %s", canonical_path)
                return None

            # Add document to target folder if not already
            if document_id not in (target_folder.document_ids or []):
                success = await self.db.add_document_to_folder(target_folder.id, document_id, auth)
                if not success:
                    logger.warning(
                        f"Failed to add document {document_id} to folder {target_folder.name}. "
                        "This may be due to a race condition during ingestion."
                    )
                else:
                    logger.info(f"Successfully added document {document_id} to folder {target_folder.name}")

            return target_folder

        except Exception as e:
            logger.error(f"Error ensuring folder exists: {e}")
            return None

    # -------------------------------------------------------------------------
    # Ingestion helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_auth_dict(auth: AuthContext) -> Dict[str, Any]:
        user_id = getattr(auth, "user_id", None)
        return {
            "user_id": user_id,
            "entity_id": user_id,
            "app_id": auth.app_id,
        }

    @staticmethod
    def _build_storage_info(
        bucket: str,
        key: str,
        filename: Optional[str],
        content_type: Optional[str],
    ) -> Dict[str, str]:
        return {
            "bucket": bucket,
            "key": key,
            "filename": filename or "",
            "content_type": content_type or "",
        }

    @staticmethod
    def _resolve_content_type(
        content_bytes: bytes,
        filename: Optional[str],
        content_type_hint: Optional[str],
    ) -> str:
        return detect_content_type(
            content=content_bytes,
            filename=filename,
            content_type_hint=content_type_hint,
        )

    @staticmethod
    def _normalize_text_filename(filename: Optional[str], content: str) -> str:
        def _needs_html_ext(text: str) -> bool:
            head = text.lstrip().lower()
            return head.startswith("<!doctype html") or "<html" in head

        if not filename:
            ext = ".html" if _needs_html_ext(content) else ".txt"
            return f"document_text_{uuid.uuid4().hex}{ext}"

        base, ext = os.path.splitext(filename)
        if ext:
            return filename
        suffix = ".html" if _needs_html_ext(content) else ".txt"
        return f"{base or filename}{suffix}"

    @staticmethod
    def _build_storage_key(filename: Optional[str], content_bytes: bytes) -> Tuple[str, str]:
        safe_filename = Path(filename or "").name or "uploaded_file"
        storage_key = f"ingest_uploads/{uuid.uuid4()}/{safe_filename}"
        if not Path(storage_key).suffix:
            detected_ext = detect_file_type(content_bytes)
            if detected_ext:
                storage_key = f"{storage_key}{detected_ext}"
                if not Path(safe_filename).suffix:
                    safe_filename = f"{safe_filename}{detected_ext}"
        return storage_key, safe_filename

    @classmethod
    def _reset_processing_metadata(cls, system_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cleaned_metadata = dict(system_metadata or {})
        cleaned_metadata.pop("progress", None)
        cleaned_metadata.pop("error", None)
        cleaned_metadata["status"] = "processing"
        cleaned_metadata["updated_at"] = datetime.now(UTC)
        return cls._clean_system_metadata(cleaned_metadata)

    async def _mark_document_failed(self, doc: Document, auth: AuthContext, error: str) -> None:
        failure_metadata = dict(doc.system_metadata or {})
        failure_metadata["status"] = "failed"
        failure_metadata["error"] = error
        failure_metadata["updated_at"] = datetime.now(UTC)
        doc.system_metadata = failure_metadata
        try:
            await self.db.update_document(
                doc.external_id,
                {"system_metadata": self._clean_system_metadata(failure_metadata)},
                auth=auth,
            )
        except Exception as db_update_err:
            logger.error("Additionally failed to mark doc %s as failed in DB: %s", doc.external_id, db_update_err)

    @classmethod
    def _build_ingestion_job_payload(
        cls,
        *,
        document_id: str,
        file_key: str,
        bucket: str,
        original_filename: Optional[str],
        content_type: Optional[str],
        auth: AuthContext,
        use_colpali: bool,
        folder_name: Optional[str] = None,
        folder_path: Optional[str] = None,
        folder_leaf: Optional[str] = None,
        end_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "_job_id": f"ingest:{document_id}",
            "_expires": timedelta(days=7),
            "document_id": document_id,
            "file_key": file_key,
            "bucket": bucket,
            "original_filename": original_filename,
            "content_type": content_type,
            "auth_dict": cls._build_auth_dict(auth),
            "use_colpali": bool(use_colpali),
            "folder_name": folder_name,
            "folder_path": folder_path,
            "folder_leaf": folder_leaf,
            "end_user_id": end_user_id,
        }

    async def _upload_content_bytes(
        self,
        *,
        content_bytes: bytes,
        filename: Optional[str],
        content_type: Optional[str],
    ) -> Tuple[str, str, str]:
        storage_key, safe_filename = self._build_storage_key(filename, content_bytes)
        bucket_name, full_storage_path = await self._upload_to_app_bucket(
            content_bytes=content_bytes,
            key=storage_key,
            content_type=content_type,
        )
        return bucket_name, full_storage_path, safe_filename

    async def _verify_ingest_and_storage_limits(
        self,
        auth: AuthContext,
        content_length: int,
        document_id: str,
    ) -> None:
        if settings.MODE != "cloud" or not auth.user_id:
            return

        num_pages = estimate_pages_by_chars(content_length)
        await check_and_increment_limits(
            auth,
            "ingest",
            num_pages,
            document_id,
            verify_only=True,
        )
        await check_and_increment_limits(auth, "storage_file", 1, verify_only=True)
        await check_and_increment_limits(
            auth,
            "storage_size",
            content_length,
            verify_only=True,
        )
        logger.info(
            "Quota verification passed for user %s – pages=%s, file=%s bytes",
            auth.user_id,
            num_pages,
            content_length,
        )

    async def _record_storage_usage(self, auth: AuthContext, content_length: int, document_id: str) -> None:
        if settings.MODE != "cloud" or not auth.user_id:
            return

        try:
            await check_and_increment_limits(auth, "storage_file", 1)
            await check_and_increment_limits(auth, "storage_size", content_length)
        except Exception as rec_err:  # noqa: BLE001
            logger.error("Failed recording storage usage for doc %s: %s", document_id, rec_err)

    async def _record_raw_storage_bytes(
        self, auth: Optional[AuthContext], document_id: str, content_length: int
    ) -> None:
        if not auth or not document_id:
            return

        try:
            await self.db.set_document_raw_bytes(document_id, auth.app_id, content_length)
        except Exception as rec_err:  # noqa: BLE001
            logger.error("Failed recording raw storage bytes for doc %s: %s", document_id, rec_err)

    async def _get_storage_object_size(self, bucket: str, key: str) -> Optional[int]:
        if not key:
            return None

        if not hasattr(self.storage, "get_object_size"):
            return None

        try:
            return await self.storage.get_object_size(bucket, key)
        except Exception as size_err:  # noqa: BLE001
            logger.warning("Failed reading stored size for %s/%s: %s", bucket, key, size_err)
            return None

    async def _record_vector_storage_bytes(
        self, auth: Optional[AuthContext], document_id: str, store_metrics: Optional[Dict[str, Any]]
    ) -> None:
        if not auth or not document_id:
            return

        chunk_bytes, multivector_bytes = extract_storage_bytes(store_metrics)
        if not chunk_bytes and not multivector_bytes:
            return

        try:
            await self.db.record_document_storage_deltas(
                document_id,
                auth.app_id,
                chunk_bytes_delta=chunk_bytes,
                multivector_bytes_delta=multivector_bytes,
            )
        except Exception as rec_err:  # noqa: BLE001
            logger.error("Failed recording vector storage bytes for doc %s: %s", document_id, rec_err)

    # -------------------------------------------------------------------------
    # File ingestion
    # -------------------------------------------------------------------------

    async def ingest_file_content(
        self,
        file_content_bytes: bytes,
        filename: str,
        content_type: Optional[str],
        metadata: Optional[Dict[str, Any]],
        auth: AuthContext,
        redis: arq.ArqRedis,
        metadata_types: Optional[Dict[str, str]] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
        use_colpali: Optional[bool] = False,
    ) -> Document:
        """
        Ingests file content from bytes. Saves to storage, creates document record,
        and then enqueues a background job for chunking and embedding.
        """
        logger.info(
            f"Starting ingestion for filename: {filename}, content_type: {content_type}, " f"user: {auth.user_id}"
        )

        # Prevent callers from overriding reserved fields
        self._enforce_no_user_mutable_fields(metadata, metadata_types=metadata_types, context="ingest")

        normalized_folder = normalize_ingest_folder_inputs(folder_name=folder_name)
        folder_path, folder_leaf = normalized_folder.path, normalized_folder.leaf

        resolved_content_type = self._resolve_content_type(
            file_content_bytes,
            filename,
            content_type,
        )

        doc = Document(
            filename=filename,
            content_type=resolved_content_type,
            metadata=metadata or {},
            app_id=auth.app_id,
            end_user_id=end_user_id,
            folder_name=folder_leaf,
            folder_path=folder_path,
        )
        doc.system_metadata = self._reset_processing_metadata(doc.system_metadata)

        await self._verify_ingest_and_storage_limits(auth, len(file_content_bytes), doc.external_id)

        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("external_id", doc.external_id)
        folder_metadata_value = normalized_folder.metadata_value
        if folder_metadata_value is not None:
            metadata_payload["folder_name"] = folder_metadata_value
        metadata_bundle = normalize_metadata(metadata_payload, metadata_types)
        doc.metadata = metadata_bundle.values
        doc.metadata_types = metadata_bundle.types

        # 1. Create initial document record in DB
        await self.db.store_document(doc, auth, metadata_bundle=metadata_bundle)
        logger.info(f"Initial document record created for {filename} (doc_id: {doc.external_id})")

        try:
            # 2. Save raw file to Storage
            bucket_name, full_storage_path, safe_filename = await self._upload_content_bytes(
                content_bytes=file_content_bytes,
                filename=filename,
                content_type=resolved_content_type,
            )
            doc.storage_info = self._build_storage_info(
                bucket_name,
                full_storage_path,
                safe_filename,
                resolved_content_type,
            )

            doc.system_metadata = self._clean_system_metadata(doc.system_metadata)
            await self.db.update_document(
                document_id=doc.external_id,
                updates={
                    "storage_info": doc.storage_info,
                    "system_metadata": doc.system_metadata,
                },
                auth=auth,
            )
            logger.info(
                "File %s (doc_id: %s) uploaded to storage at %s/%s and DB updated.",
                filename,
                doc.external_id,
                bucket_name,
                full_storage_path,
            )

            await self._record_storage_usage(auth, len(file_content_bytes), doc.external_id)
            stored_size = await self._get_storage_object_size(bucket_name, full_storage_path)
            if stored_size is not None:
                await self._record_raw_storage_bytes(auth, doc.external_id, stored_size)

        except Exception as e:
            logger.error(f"Failed to upload file {filename} (doc_id: {doc.external_id}) to storage or update DB: {e}")
            await self._mark_document_failed(doc, auth, f"Storage upload/DB update failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file to storage: {str(e)}")

        # 3. Ensure folder exists if folder_name is provided
        if folder_name:
            try:
                folder_obj = await self._ensure_folder_exists(folder_name, doc.external_id, auth)
                if folder_obj and folder_obj.id:
                    doc.folder_id = folder_obj.id
                    folder_updates = self.folder_update_fields(folder_obj)
                    await self.db.update_document(doc.external_id, folder_updates, auth=auth)
                logger.debug(f"Ensured folder '{folder_name}' exists and contains document {doc.external_id}")
            except Exception as e:
                logger.error(f"Error during _ensure_folder_exists for doc {doc.external_id}: {e}. Continuing.")

        try:
            job_payload = self._build_ingestion_job_payload(
                document_id=doc.external_id,
                file_key=full_storage_path,
                bucket=bucket_name,
                original_filename=filename,
                content_type=resolved_content_type,
                auth=auth,
                use_colpali=bool(use_colpali),
                folder_name=folder_leaf,
                folder_path=folder_path,
                folder_leaf=folder_leaf,
                end_user_id=end_user_id,
            )
            job = await redis.enqueue_job("process_ingestion_job", **job_payload)
            if job is None:
                logger.info("Connector file ingestion job already queued (doc_id=%s)", doc.external_id)
            else:
                logger.info(
                    "Connector file ingestion job queued with ID: %s for document: %s", job.job_id, doc.external_id
                )
        except Exception as e:
            logger.error(f"Failed to enqueue ingestion job for doc {doc.external_id} ({filename}): {e}")
            await self._mark_document_failed(doc, auth, f"Failed to enqueue processing job: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to enqueue document processing job: {str(e)}")

        return doc

    # -------------------------------------------------------------------------
    # Document update (queued)
    # -------------------------------------------------------------------------

    async def queue_document_update(
        self,
        document_id: str,
        auth: AuthContext,
        redis: arq.ArqRedis,
        content: Optional[str] = None,
        file: Optional[UploadFile] = None,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_types: Optional[Dict[str, str]] = None,
        use_colpali: Optional[bool] = None,
    ) -> Optional[Document]:
        """
        Update a document by replacing its content and re-queueing ingestion.
        """
        self._enforce_no_user_mutable_fields(metadata, metadata_types=metadata_types, context="update")

        doc = await self._validate_update_access(document_id, auth)
        if not doc:
            return None

        metadata_only_update = content is None and file is None and metadata is not None
        if metadata_only_update:
            metadata_bundle = self._update_metadata(doc, metadata, metadata_types, None)
            return await self._update_document_metadata_only(doc, auth, metadata_bundle)

        if content is None and file is None:
            logger.error("Neither content nor file provided for document update")
            raise ValueError("Either content or file must be provided for document update")

        if content is not None:
            candidate_filename = filename
            if not candidate_filename:
                existing_filename = doc.filename or ""
                existing_ext = os.path.splitext(existing_filename)[1].lower()
                if existing_ext in {".txt", ".html", ".htm"}:
                    candidate_filename = existing_filename
            updated_filename = self._normalize_text_filename(candidate_filename, content)
            content_bytes = content.encode("utf-8")
            detected_content_type = self._resolve_content_type(content_bytes, updated_filename, None)
        else:
            content_bytes = await file.read()
            updated_filename = file.filename or doc.filename or "uploaded_file"
            detected_content_type = self._resolve_content_type(content_bytes, updated_filename, file.content_type)

        doc.filename = updated_filename
        doc.content_type = detected_content_type
        metadata_bundle = self._update_metadata(doc, metadata, metadata_types, None)

        await self._verify_ingest_and_storage_limits(auth, len(content_bytes), doc.external_id)

        old_storage_info = dict(doc.storage_info) if isinstance(doc.storage_info, dict) else None

        try:
            bucket_name, full_storage_path, safe_filename = await self._upload_content_bytes(
                content_bytes=content_bytes,
                filename=updated_filename,
                content_type=detected_content_type,
            )
            new_storage_info = self._build_storage_info(
                bucket_name,
                full_storage_path,
                safe_filename,
                detected_content_type,
            )
            doc.storage_info = new_storage_info
        except Exception as e:
            logger.error("Failed to upload updated file for doc %s: %s", doc.external_id, e)
            await self._mark_document_failed(doc, auth, f"Storage upload/DB update failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to upload updated file to storage: {str(e)}")

        doc.system_metadata = self._reset_processing_metadata(doc.system_metadata)

        updates = {
            "metadata": doc.metadata,
            "metadata_types": doc.metadata_types,
            "storage_info": doc.storage_info,
            "system_metadata": doc.system_metadata,
            "filename": doc.filename,
            "content_type": doc.content_type,
        }

        success = await self.db.update_document(
            doc.external_id,
            updates,
            auth,
            metadata_bundle=metadata_bundle,
        )
        if not success:
            logger.error("Failed to update document %s prior to queueing update", doc.external_id)
            await self._mark_document_failed(doc, auth, "Failed to persist document updates")
            raise HTTPException(status_code=500, detail="Failed to update document before queueing ingestion")

        if old_storage_info and hasattr(self.storage, "delete_file"):
            old_bucket = old_storage_info.get("bucket")
            old_key = old_storage_info.get("key")
            new_bucket = doc.storage_info.get("bucket") if doc.storage_info else None
            new_key = doc.storage_info.get("key") if doc.storage_info else None
            if old_bucket and old_key and (old_bucket != new_bucket or old_key != new_key):
                try:
                    await self.storage.delete_file(old_bucket, old_key)
                    logger.info("Deleted old file from bucket `%s` with key `%s`", old_bucket, old_key)
                except Exception as e:
                    logger.warning("Failed to delete old file %s/%s: %s", old_bucket, old_key, e)

        await self._record_storage_usage(auth, len(content_bytes), doc.external_id)
        stored_size = await self._get_storage_object_size(bucket_name, full_storage_path)
        if stored_size is not None:
            await self._record_raw_storage_bytes(auth, doc.external_id, stored_size)

        try:
            job_payload = self._build_ingestion_job_payload(
                document_id=doc.external_id,
                file_key=full_storage_path,
                bucket=bucket_name,
                original_filename=doc.filename,
                content_type=doc.content_type,
                auth=auth,
                use_colpali=bool(use_colpali),
                folder_name=doc.folder_name,
                folder_path=doc.folder_path,
                folder_leaf=doc.folder_name,
                end_user_id=doc.end_user_id,
            )
            job = await redis.enqueue_job("process_ingestion_job", **job_payload)
            if job is None:
                logger.info("Update ingestion job already queued (doc_id=%s)", doc.external_id)
            else:
                logger.info("Update ingestion job queued (job_id=%s, doc=%s)", job.job_id, doc.external_id)
        except Exception as e:
            logger.error("Failed to enqueue update ingestion job for doc %s: %s", doc.external_id, e)
            await self._mark_document_failed(doc, auth, f"Failed to enqueue processing job: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to enqueue document update job: {str(e)}")

        return doc

    # -------------------------------------------------------------------------
    # Document update
    # -------------------------------------------------------------------------

    async def update_document(
        self,
        document_id: str,
        auth: AuthContext,
        content: Optional[str] = None,
        file: Optional[UploadFile] = None,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_types: Optional[Dict[str, str]] = None,
        use_colpali: Optional[bool] = None,
    ) -> Optional[Document]:
        """
        Update a document by replacing its content and/or metadata.

        Args:
            document_id: ID of the document to update
            auth: Authentication context
            content: New text content (replaces existing)
            file: New file (replaces existing)
            filename: Optional new filename for the document
            metadata: Additional metadata to merge
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Updated document if successful, None if failed
        """
        # Prevent callers from modifying reserved fields
        self._enforce_no_user_mutable_fields(metadata, metadata_types=metadata_types, context="update")

        # Validate permissions and get document
        doc = await self._validate_update_access(document_id, auth)
        if not doc:
            return None

        metadata_only_update = content is None and file is None and metadata is not None

        # Process content based on update type
        updated_content = None
        file_content = None
        file_type = None
        file_content_base64 = None
        if content is not None:
            updated_content = await self._process_text_update(content, doc, filename)
            updated_content = self._normalize_document_content(updated_content)
            logger.info(f"Replacing document content with new text of length {len(updated_content)}")
        elif file is not None:
            updated_content, file_content, file_type, file_content_base64 = await self._process_file_update(file, doc)
            updated_content = self._normalize_document_content(updated_content)
            logger.info(f"Replacing document content with parsed file text of length {len(updated_content)}")

            await self._record_storage_usage(auth, len(file_content), doc.external_id)
            stored_size = await self._get_storage_object_size(
                doc.storage_info.get("bucket") if doc.storage_info else "",
                doc.storage_info.get("key") if doc.storage_info else "",
            )
            if stored_size is not None:
                await self._record_raw_storage_bytes(auth, doc.external_id, stored_size)
        elif not metadata_only_update:
            logger.error("Neither content nor file provided for document update")
            return None

        # Replace content if we have new content
        if updated_content:
            doc.system_metadata["content"] = updated_content
            logger.info(f"Updated system_metadata['content'] with content of length {len(updated_content)}")
        else:
            # Keep existing content for metadata-only updates
            updated_content = doc.system_metadata.get("content", "")
            logger.info(f"No content update - keeping current content of length {len(updated_content)}")

        # Update metadata
        metadata_bundle = self._update_metadata(doc, metadata, metadata_types, file)

        # For metadata-only updates, we don't need to re-process chunks
        if metadata_only_update:
            return await self._update_document_metadata_only(doc, auth, metadata_bundle)

        # Process content into chunks and generate embeddings
        chunks, chunk_objects = await self._process_chunks_and_embeddings(doc.external_id, updated_content)
        if not chunks:
            return None

        # Handle colpali (multi-vector) embeddings if needed
        chunk_objects_multivector = await self._process_colpali_embeddings(
            use_colpali, doc.external_id, chunks, file, file_type, file_content, file_content_base64
        )

        # Store everything
        await self._store_chunks_and_doc(
            chunk_objects,
            doc,
            use_colpali,
            chunk_objects_multivector,
            is_update=True,
            auth=auth,
            metadata_bundle=metadata_bundle,
        )
        logger.info(f"Successfully updated document {doc.external_id}")

        return doc

    async def _validate_update_access(self, document_id: str, auth: AuthContext) -> Optional[Document]:
        """Validate user permissions and document access."""
        doc = await self.db.get_document(document_id, auth)
        if not doc:
            logger.error(f"Document {document_id} not found")
            return None

        if not await self.db.check_access(document_id, auth, "write"):
            logger.error(f"User {auth.user_id} does not have write permission for document {document_id}")
            raise PermissionError(f"User does not have write permission for document {document_id}")

        return doc

    async def _process_text_update(
        self,
        content: str,
        doc: Document,
        filename: Optional[str],
    ) -> str:
        """Process text content updates."""
        update_content = content

        if filename:
            doc.filename = filename

        return update_content

    async def _process_file_update(
        self,
        file: UploadFile,
        doc: Document,
    ) -> Tuple[str, bytes, str, str]:
        """Process file content updates."""
        # Read file content
        file_content = await file.read()

        # Parse the file content
        additional_file_metadata, file_text = await self.parser.parse_file_to_text(file_content, file.filename)
        logger.info(f"Parsed file into text of length {len(file_text)}")

        # Add additional metadata from file if available
        if additional_file_metadata:
            if not doc.additional_metadata:
                doc.additional_metadata = {}
            doc.additional_metadata.update(additional_file_metadata)

        # Store file in storage if needed
        file_content_base64 = encode_base64(file_content)

        # Store file type
        content_type = self._resolve_content_type(file_content, file.filename, file.content_type)

        # Store file in storage and update storage info
        await self._update_storage_info(doc, file, file_content, content_type=content_type)
        doc.content_type = content_type

        doc.filename = file.filename

        return file_text, file_content, content_type, file_content_base64

    async def _update_storage_info(
        self,
        doc: Document,
        file: UploadFile,
        file_content: bytes,
        *,
        content_type: Optional[str] = None,
    ):
        """Update document storage information for file content, deleting the old file if present."""
        # Delete old file from storage if it exists
        if doc.storage_info:
            old_bucket = doc.storage_info.get("bucket")
            old_key = doc.storage_info.get("key")
            if old_bucket and old_key and hasattr(self.storage, "delete_file"):
                try:
                    await self.storage.delete_file(old_bucket, old_key)
                    logger.info(f"Deleted old file from bucket `{old_bucket}` with key `{old_key}`")
                except Exception as e:
                    logger.warning(f"Failed to delete old file {old_bucket}/{old_key}: {e}")

        bucket, key, safe_filename = await self._upload_content_bytes(
            content_bytes=file_content,
            filename=file.filename,
            content_type=content_type or file.content_type,
        )
        resolved_content_type = content_type or file.content_type
        doc.storage_info = self._build_storage_info(bucket, key, safe_filename, resolved_content_type)
        logger.info(f"Stored new file in bucket `{bucket}` with key `{key}`")

    @staticmethod
    def _normalize_document_content(content: Any) -> str:
        """Ensure stored document content is always handled as text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, bytes):
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("Failed to decode bytes document content using UTF-8; returning base64 encoded fallback")
                return encode_base64(content)
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=False)
            except (TypeError, ValueError):
                logger.warning(
                    "Failed to serialize %s content to JSON; falling back to string conversion",
                    type(content).__name__,
                )
                return str(content)
        logger.warning("Coercing unexpected content type %s to string", type(content).__name__)
        return str(content)

    async def _update_document_metadata_only(
        self,
        doc: Document,
        auth: AuthContext,
        metadata_bundle: Optional[MetadataBundle],
    ) -> Optional[Document]:
        """Update document metadata without reprocessing chunks."""
        doc.system_metadata = self._clean_system_metadata(doc.system_metadata)

        updates = {
            "metadata": doc.metadata,
            "metadata_types": doc.metadata_types,
            "system_metadata": doc.system_metadata,
            "filename": doc.filename,
            "storage_info": doc.storage_info if hasattr(doc, "storage_info") else None,
        }
        updates = {k: v for k, v in updates.items() if v is not None}

        if doc.folder_id:
            updates["folder_id"] = doc.folder_id
            updates["folder_name"] = doc.folder_name
            updates["folder_path"] = doc.folder_path

        success = await self.db.update_document(
            doc.external_id,
            updates,
            auth,
            metadata_bundle=metadata_bundle,
        )
        if not success:
            logger.error(f"Failed to update document {doc.external_id} metadata")
            return None

        logger.info(f"Successfully updated document metadata for {doc.external_id}")
        return doc

    def _update_metadata(
        self,
        doc: Document,
        metadata: Optional[Dict[str, Any]],
        metadata_types: Optional[Dict[str, str]],
        file: Optional[UploadFile],
    ) -> MetadataBundle:
        """Update document metadata."""
        if metadata:
            payload = dict(metadata)
            metadata_bundle = merge_metadata(
                doc.metadata,
                doc.metadata_types,
                payload,
                metadata_types,
                external_id=doc.external_id,
            )
            doc.metadata = metadata_bundle.values
            doc.metadata_types = metadata_bundle.types
        else:
            doc.metadata.setdefault("external_id", doc.external_id)
            doc.metadata_types.setdefault("external_id", "string")
            metadata_bundle = MetadataBundle(
                values=dict(doc.metadata),
                types=dict(doc.metadata_types),
                is_normalized=True,
            )

        doc.system_metadata["updated_at"] = datetime.now(UTC)

        if file:
            doc.filename = file.filename

        return metadata_bundle

    # -------------------------------------------------------------------------
    # Chunk processing and storage
    # -------------------------------------------------------------------------

    async def _process_chunks_and_embeddings(
        self, doc_id: str, content: str
    ) -> Tuple[List[Chunk], List[DocumentChunk]]:
        """Process content into chunks and generate embeddings."""
        parsed_chunks = await self.parser.split_text(content)
        if not parsed_chunks:
            logger.error("No content chunks extracted after update")
            return None, None

        logger.info(f"Split updated text into {len(parsed_chunks)} chunks")

        processed_chunks = parsed_chunks

        embeddings = await self.embedding_model.embed_for_ingestion(processed_chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        chunk_objects = self._create_chunk_objects(doc_id, processed_chunks, embeddings)
        logger.info(f"Created {len(chunk_objects)} chunk objects")

        return processed_chunks, chunk_objects

    async def _process_colpali_embeddings(
        self,
        use_colpali: bool,
        doc_id: str,
        chunks: List[Chunk],
        file: Optional[UploadFile],
        file_type: Any,
        file_content: Optional[bytes],
        file_content_base64: Optional[str],
    ) -> List[DocumentChunk]:
        """Process colpali multi-vector embeddings if enabled."""
        chunk_objects_multivector = []

        if not (use_colpali and settings.ENABLE_COLPALI and self.colpali_embedding_model and self.colpali_vector_store):
            return chunk_objects_multivector

        mime_type = file_type if isinstance(file_type, str) else (file_type.mime if file_type is not None else None)
        if (
            file
            and mime_type
            and (mime_type.startswith("image/") or mime_type == "application/pdf" or mime_type == "application/dicom")
        ):
            if hasattr(file, "seek") and callable(file.seek) and not file_content:
                await file.seek(0)
                file_content = await file.read()
                file_content_base64 = encode_base64(file_content)

            chunks_multivector = self._create_chunks_multivector(mime_type, file_content_base64, file_content, chunks)
            logger.info(f"Created {len(chunks_multivector)} chunks for multivector embedding")
            colpali_embeddings = await self.colpali_embedding_model.embed_for_ingestion(chunks_multivector)
            logger.info(f"Generated {len(colpali_embeddings)} embeddings for multivector embedding")
            chunk_objects_multivector = self._create_chunk_objects(doc_id, chunks_multivector, colpali_embeddings)
        else:
            embeddings_multivector = await self.colpali_embedding_model.embed_for_ingestion(chunks)
            logger.info(f"Generated {len(embeddings_multivector)} embeddings for multivector embedding")
            chunk_objects_multivector = self._create_chunk_objects(doc_id, chunks, embeddings_multivector)

        logger.info(f"Created {len(chunk_objects_multivector)} chunk objects for multivector embedding")
        return chunk_objects_multivector

    def _create_chunk_objects(
        self,
        doc_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        start_index: int = 0,
    ) -> List[DocumentChunk]:
        """Helper to create chunk objects."""
        chunk_objects: List[DocumentChunk] = []
        for index, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            original_metadata = chunk.metadata or {}
            sanitized_metadata: Dict[str, Any] = {}
            for key, value in original_metadata.items():
                if key == "_image_bytes":
                    continue
                if isinstance(value, (bytes, bytearray, memoryview)):
                    sanitized_metadata[key] = encode_base64(bytes(value))
                else:
                    sanitized_metadata[key] = value
            sanitized_chunk = Chunk(content=chunk.content, metadata=sanitized_metadata)
            chunk_objects.append(
                sanitized_chunk.to_document_chunk(
                    chunk_number=start_index + index,
                    embedding=embedding,
                    document_id=doc_id,
                )
            )
        return chunk_objects

    async def _store_chunks_and_doc(
        self,
        chunk_objects: List[DocumentChunk],
        doc: Document,
        use_colpali: bool = False,
        chunk_objects_multivector: Optional[List[DocumentChunk]] = None,
        is_update: bool = False,
        auth: Optional[AuthContext] = None,
        metadata_bundle: Optional[MetadataBundle] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Helper to store chunks and document."""
        max_retries = 3
        retry_delay = 1.0

        async def store_with_retry(store, objects, store_name="regular"):
            attempt = 0
            success = False
            current_retry_delay = retry_delay

            while attempt < max_retries and not success:
                try:
                    success, result, metrics = await store.store_embeddings(objects, auth.app_id if auth else None)
                    if not success:
                        raise Exception(f"Failed to store {store_name} chunk embeddings")
                    return result, metrics
                except Exception as e:
                    attempt += 1
                    error_msg = str(e)
                    if "connection was closed" in error_msg or "ConnectionDoesNotExistError" in error_msg:
                        if attempt < max_retries:
                            logger.warning(
                                f"Database connection error during {store_name} embeddings storage "
                                f"(attempt {attempt}/{max_retries}): {error_msg}. "
                                f"Retrying in {current_retry_delay}s..."
                            )
                            await asyncio.sleep(current_retry_delay)
                            current_retry_delay *= 2
                        else:
                            logger.error(
                                f"All {store_name} database connection attempts failed "
                                f"after {max_retries} retries: {error_msg}"
                            )
                            raise Exception(f"Failed to store {store_name} chunk embeddings after multiple retries")
                    else:
                        logger.error(f"Error storing {store_name} embeddings: {error_msg}")
                        raise

        async def store_document_with_retry():
            attempt = 0
            success = False
            current_retry_delay = retry_delay

            while attempt < max_retries and not success:
                try:
                    doc.system_metadata = self._clean_system_metadata(doc.system_metadata)

                    if is_update and auth:
                        updates = {
                            "chunk_ids": doc.chunk_ids,
                            "metadata": doc.metadata,
                            "metadata_types": doc.metadata_types,
                            "system_metadata": doc.system_metadata,
                            "filename": doc.filename,
                            "content_type": doc.content_type,
                            "storage_info": doc.storage_info,
                        }
                        success = await self.db.update_document(
                            doc.external_id,
                            updates,
                            auth,
                            metadata_bundle=metadata_bundle,
                        )
                        if not success:
                            raise Exception("Failed to update document metadata")
                    else:
                        success = await self.db.store_document(
                            doc,
                            auth,
                            metadata_bundle=metadata_bundle,
                        )
                        if not success:
                            raise Exception("Failed to store document metadata")
                    return success
                except Exception as e:
                    attempt += 1
                    error_msg = str(e)
                    if "connection was closed" in error_msg or "ConnectionDoesNotExistError" in error_msg:
                        if attempt < max_retries:
                            logger.warning(
                                f"Database connection error during document metadata storage "
                                f"(attempt {attempt}/{max_retries}): {error_msg}. "
                                f"Retrying in {current_retry_delay}s..."
                            )
                            await asyncio.sleep(current_retry_delay)
                            current_retry_delay *= 2
                        else:
                            logger.error(
                                f"All database connection attempts failed after {max_retries} retries: {error_msg}"
                            )
                            raise Exception("Failed to store document metadata after multiple retries")
                    else:
                        logger.error(f"Error storing document metadata: {error_msg}")
                        raise

        # Store in the appropriate vector store based on use_colpali
        if use_colpali and self.colpali_vector_store and chunk_objects_multivector:
            chunk_ids, store_metrics = await store_with_retry(
                self.colpali_vector_store, chunk_objects_multivector, "colpali"
            )
        else:
            chunk_ids, store_metrics = await store_with_retry(self.vector_store, chunk_objects, "regular")

        doc.chunk_ids = chunk_ids

        logger.debug(f"Stored chunk embeddings in vector stores: {len(doc.chunk_ids)} chunks total")

        await store_document_with_retry()

        logger.debug("Stored document metadata in database")
        logger.debug(f"Chunk IDs stored: {doc.chunk_ids}")
        await self._record_vector_storage_bytes(auth, doc.external_id, store_metrics)
        return doc.chunk_ids, store_metrics

    # -------------------------------------------------------------------------
    # ColPali multi-vector chunk creation
    # -------------------------------------------------------------------------

    def _image_bytes_to_chunk(
        self,
        image_bytes: bytes,
        mime_type: str,
        base64_override: Optional[str] = None,
    ) -> Chunk:
        """Build a Chunk that preserves raw image bytes alongside the data URI."""
        content = base64_override
        if content is None:
            content = bytes_to_data_uri(image_bytes, mime_type)
        return Chunk(
            content=content,
            metadata={"is_image": True, "_image_bytes": image_bytes, "mime_type": mime_type},
        )

    def img_to_base64_with_bytes(
        self,
        img: PILImage.Image,
        format: str = "PNG",
        mime_type: Optional[str] = None,
    ) -> Tuple[str, bytes]:
        """Convert PIL Image to base64 string and raw bytes."""
        buffered = BytesIO()
        img.save(buffered, format=format)
        buffered.seek(0)
        img_bytes = buffered.getvalue()
        mime = mime_type or f"image/{format.lower()}"
        img_str = bytes_to_data_uri(img_bytes, mime)
        return img_str, img_bytes

    def img_to_base64_str(self, img: PILImage.Image) -> str:
        """Convert PIL Image to base64 string."""
        img_str, _ = self.img_to_base64_with_bytes(img)
        return img_str

    def _render_pdf_with_pymupdf(
        self, file_content: bytes, dpi: int, include_bytes: bool = False
    ) -> List[Union[str, Tuple[str, bytes]]]:
        """Render a PDF into base64-encoded PNG images using PyMuPDF."""
        pdf_document = fitz.open("pdf", file_content)
        try:
            images: List[Union[str, Tuple[str, bytes]]] = []
            for page in pdf_document:
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                png_bytes = pix.tobytes("png")
                b64 = bytes_to_data_uri(png_bytes, "image/png")
                if include_bytes:
                    images.append((b64, png_bytes))
                else:
                    images.append(b64)
            return images
        finally:
            pdf_document.close()

    def _create_chunks_multivector(
        self,
        mime_type: Optional[str],
        file_content_base64: Optional[str],
        file_content: bytes,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        """
        Create image-based chunks for ColPali multi-vector embedding.

        Handles:
        - Direct images (PNG, JPEG, etc.)
        - PDFs (renders each page as image)
        - Word documents (converts to PDF, then to images)
        - PowerPoint presentations (converts to PDF, then to images)
        - Other formats: falls back to text chunks
        """
        normalized_mime = mime_type
        if normalized_mime in {"application/octet-stream", "binary/octet-stream", "application/x-octet-stream"}:
            normalized_mime = None
        logger.info("Creating chunks for multivector embedding for file type %s", normalized_mime or "unknown")

        # If we don't have a reliable MIME, attempt a light-weight heuristic to detect images.
        if not normalized_mime:
            try:
                PILImage.open(BytesIO(file_content)).verify()
                logger.info("Heuristic image detection succeeded (Pillow). Treating as image.")
                if file_content_base64 is None:
                    file_content_base64 = encode_base64(file_content)
                return [
                    self._image_bytes_to_chunk(
                        file_content,
                        mime_type="image/unknown",
                        base64_override=file_content_base64,
                    )
                ]
            except Exception:
                logger.info("File type is None and not an image – treating as text")
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks
                ]

        # Treat any direct image MIME as an image
        if normalized_mime.startswith("image/"):
            try:
                img = PILImage.open(BytesIO(file_content))
                max_width = 256
                if img.width > max_width:
                    ratio = max_width / float(img.width)
                    new_height = int(float(img.height) * ratio)
                    img = img.resize((max_width, new_height))

                buffered = BytesIO()
                img.convert("RGB").save(buffered, format="JPEG", quality=70, optimize=True)
                jpeg_bytes = buffered.getvalue()
                img_b64 = bytes_to_data_uri(jpeg_bytes, "image/jpeg")
                return [
                    self._image_bytes_to_chunk(
                        jpeg_bytes,
                        mime_type="image/jpeg",
                        base64_override=img_b64,
                    )
                ]
            except Exception as e:
                logger.error(f"Error resizing image for base64 encoding: {e}. Falling back to original size.")
                if file_content_base64 is None:
                    file_content_base64 = encode_base64(file_content)
                return [
                    self._image_bytes_to_chunk(
                        file_content,
                        mime_type=normalized_mime,
                        base64_override=file_content_base64,
                    )
                ]

        match normalized_mime:
            case "application/pdf":
                return self._process_pdf_for_colpali(file_content)

            case "application/dicom":
                if file_content_base64 is None:
                    file_content_base64 = encode_base64(file_content)
                return [
                    self._image_bytes_to_chunk(
                        file_content,
                        mime_type=normalized_mime,
                        base64_override=file_content_base64,
                    )
                ]

            case "application/vnd.openxmlformats-officedocument.wordprocessingml.document" | "application/msword":
                return self._process_word_for_colpali(file_content, chunks)

            case (
                "application/vnd.ms-powerpoint"
                | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                | "application/vnd.openxmlformats-officedocument.presentationml.slideshow"
            ):
                return self._process_powerpoint_for_colpali(file_content, normalized_mime, chunks)

            case (
                "application/vnd.ms-excel"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/vnd.ms-excel.sheet.macroEnabled.12"
            ):
                # Excel files use text chunks embedded via ColPali's text path
                # (no image conversion — text retrieval is much better for tabular data)
                logger.info("Excel file detected — using text chunks for ColPali embedding")
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks
                ]

            case _:
                logger.warning("Colpali is not supported for file type %s - skipping", normalized_mime)
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks
                ]

    def _process_pdf_for_colpali(self, file_content: bytes) -> List[Chunk]:
        """Process PDF file for ColPali embedding."""
        logger.info("Working with PDF file - using PyMuPDF for faster processing!")

        if not file_content:
            logger.error("PDF file content is empty")
            raise PdfConversionError("PDF file content is empty")

        dpi = settings.COLPALI_PDF_DPI

        try:
            # Check document density to decide processing strategy
            # High-density docs (large images per page) need batched processing to avoid OOM
            pdf_document = fitz.open("pdf", file_content)
            try:
                page_count = len(pdf_document)
            finally:
                pdf_document.close()

            file_size_mb = len(file_content) / (1024 * 1024)
            density_mb_per_page = file_size_mb / page_count if page_count > 0 else 0

            # High-density threshold: 1.0 MB per page indicates large images/drawings
            # These can cause OOM when all pages are rendered simultaneously
            HIGH_DENSITY_THRESHOLD_MB = 1.0
            HIGH_DENSITY_BATCH_SIZE = 2  # Process 2 pages at a time for high-density docs

            if density_mb_per_page > HIGH_DENSITY_THRESHOLD_MB:
                logger.info(
                    f"High-density PDF detected: {file_size_mb:.2f}MB / {page_count} pages = "
                    f"{density_mb_per_page:.2f}MB/page (threshold: {HIGH_DENSITY_THRESHOLD_MB}). "
                    f"Using batched rendering with batch_size={HIGH_DENSITY_BATCH_SIZE}."
                )
                return self._render_pdf_with_pymupdf_batched(file_content, dpi, batch_size=HIGH_DENSITY_BATCH_SIZE)

            # Normal processing for low-density PDFs
            images_with_bytes = self._render_pdf_with_pymupdf(file_content, dpi, include_bytes=True)
            logger.info(f"PyMuPDF processed {len(images_with_bytes)} pages")
            return [
                self._image_bytes_to_chunk(raw_bytes, mime_type="image/png", base64_override=image_b64)
                for image_b64, raw_bytes in images_with_bytes
            ]
        except Exception as e:
            logger.warning(f"PyMuPDF failed ({e}), falling back to pdf2image")

            try:
                images = pdf2image.convert_from_bytes(file_content, dpi=dpi)
                image_payloads = [self.img_to_base64_with_bytes(image) for image in images]
                logger.info(f"pdf2image fallback processed {len(image_payloads)} pages")
                return [
                    self._image_bytes_to_chunk(raw_bytes, mime_type="image/png", base64_override=image_b64)
                    for image_b64, raw_bytes in image_payloads
                ]
            except Exception as fallback_error:
                logger.error(f"pdf2image fallback failed: {fallback_error}")
                raise PdfConversionError(f"Unable to convert PDF to images: {fallback_error}") from fallback_error

    def _render_pdf_with_pymupdf_batched(self, file_content: bytes, dpi: int, batch_size: int = 2) -> List[Chunk]:
        """
        Render a high-density PDF in batches to limit memory usage.

        For PDFs with large images per page (e.g., architectural drawings, high-res scans),
        rendering all pages simultaneously can cause OOM. This method processes pages in
        small batches, releasing pixmap memory between batches.

        Args:
            file_content: Raw PDF bytes
            dpi: Resolution for rendering
            batch_size: Number of pages to render simultaneously (default: 2)

        Returns:
            List of Chunk objects for each page
        """
        all_chunks: List[Chunk] = []
        pdf_document = fitz.open("pdf", file_content)

        try:
            total_pages = len(pdf_document)
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                logger.debug(f"Rendering pages {batch_start + 1}-{batch_end} of {total_pages} (batched mode)")

                # Render and convert this batch
                for page_num in range(batch_start, batch_end):
                    page = pdf_document[page_num]
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    png_bytes = pix.tobytes("png")
                    b64 = bytes_to_data_uri(png_bytes, "image/png")

                    # Create chunk immediately
                    chunk = self._image_bytes_to_chunk(png_bytes, mime_type="image/png", base64_override=b64)
                    all_chunks.append(chunk)

                    # Explicitly release pixmap memory - this is the key memory optimization
                    del pix
                    del png_bytes

            logger.info(f"PyMuPDF processed {total_pages} pages in batched mode")
            return all_chunks
        finally:
            pdf_document.close()

    def _process_word_for_colpali(self, file_content: bytes, chunks: List[Chunk]) -> List[Chunk]:
        """Process Word document for ColPali embedding."""
        logger.info("Working with Word document!")

        if not file_content or len(file_content) == 0:
            logger.error("Word document content is empty")
            return [Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks]

        return self._convert_office_to_images(file_content, ".docx", "Word document", chunks)

    def _process_powerpoint_for_colpali(self, file_content: bytes, mime_type: str, chunks: List[Chunk]) -> List[Chunk]:
        """Process PowerPoint presentation for ColPali embedding."""
        logger.info("Working with PowerPoint presentation!")

        if not file_content or len(file_content) == 0:
            logger.error("PowerPoint presentation content is empty")
            return [Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks]

        suffix = ".ppt" if mime_type == "application/vnd.ms-powerpoint" else ".pptx"
        return self._convert_office_to_images(file_content, suffix, "PowerPoint presentation", chunks)

    def _convert_office_to_images(
        self, file_content: bytes, suffix: str, doc_type: str, chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        Convert Office document to images via LibreOffice PDF conversion.

        Args:
            file_content: Raw bytes of the Office document
            suffix: File extension (e.g., ".docx", ".pptx", ".xlsx")
            doc_type: Human-readable document type for logging
            chunks: Fallback text chunks if conversion fails

        Returns:
            List of image chunks for ColPali processing
        """
        import shutil
        import subprocess

        # Check if LibreOffice is available
        if not shutil.which("soffice"):
            logger.warning(f"LibreOffice (soffice) not found in PATH. Falling back to text extraction for {doc_type}.")
            logger.info("To enable visual processing, install LibreOffice: apt-get install libreoffice")
            return [Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks]

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_input:
            temp_input.write(file_content)
            temp_input_path = temp_input.name

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name

        expected_pdf_path = None

        try:
            base_filename = os.path.splitext(os.path.basename(temp_input_path))[0]
            output_dir = os.path.dirname(temp_pdf_path)
            expected_pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")

            # Convert to PDF with timeout
            result = subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    output_dir,
                    temp_input_path,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.warning(f"LibreOffice conversion failed for {doc_type}: {result.stderr}")
                logger.info(f"Falling back to text extraction for {doc_type}")
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks
                ]

            if not os.path.exists(expected_pdf_path) or os.path.getsize(expected_pdf_path) == 0:
                logger.warning(f"Generated PDF is empty or doesn't exist at: {expected_pdf_path}")
                logger.info(f"Falling back to text extraction for {doc_type}")
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks
                ]

            # Process the PDF
            with open(expected_pdf_path, "rb") as pdf_file:
                pdf_content = pdf_file.read()

            try:
                pdf_document = fitz.open("pdf", pdf_content)
                images_payload: List[Tuple[str, bytes]] = []

                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    dpi = settings.COLPALI_PDF_DPI
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")

                    img = PILImage.open(BytesIO(img_data))
                    img_str, img_bytes = self.img_to_base64_with_bytes(img)
                    images_payload.append((img_str, img_bytes))

                pdf_document.close()

                logger.info(f"{doc_type} successfully processed {len(images_payload)} pages as images")
                return [
                    self._image_bytes_to_chunk(raw_bytes, mime_type="image/png", base64_override=image_b64)
                    for image_b64, raw_bytes in images_payload
                ]

            except Exception as pymupdf_error:
                logger.warning(f"PyMuPDF failed for {doc_type} ({pymupdf_error}), trying pdf2image")
                try:
                    images = pdf2image.convert_from_bytes(pdf_content)
                    images_payload = [self.img_to_base64_with_bytes(image) for image in images]

                    logger.info(f"{doc_type} processed {len(images_payload)} pages with pdf2image")
                    return [
                        self._image_bytes_to_chunk(raw_bytes, mime_type="image/png", base64_override=image_b64)
                        for image_b64, raw_bytes in images_payload
                    ]
                except Exception as pdf2image_error:
                    logger.warning(f"pdf2image also failed: {pdf2image_error}")
                    logger.info(f"Falling back to text extraction for {doc_type}")
                    return [
                        Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                        for chunk in chunks
                    ]

        except subprocess.TimeoutExpired:
            logger.warning(f"LibreOffice conversion timed out for {doc_type}")
            logger.info("Falling back to text extraction")
            return [Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks]
        except Exception as e:
            logger.warning(f"Unexpected error processing {doc_type}: {str(e)}")
            logger.info(f"Falling back to text extraction for {doc_type}")
            return [Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks]
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
                if expected_pdf_path and os.path.exists(expected_pdf_path) and expected_pdf_path != temp_pdf_path:
                    os.unlink(expected_pdf_path)
            except Exception as cleanup_error:
                logger.debug(f"Error cleaning up temporary files: {cleanup_error}")

    # -------------------------------------------------------------------------
    # Storage helpers
    # -------------------------------------------------------------------------

    async def _upload_to_app_bucket(
        self,
        content_bytes: bytes,
        key: str,
        content_type: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Upload file to app-specific bucket."""
        return await self.storage.upload_file(
            content_bytes,
            key,
            content_type,
            bucket="",
        )

    def close(self):
        """Close all resources."""
        pass

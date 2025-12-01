"""
Ingestion Service - Handles all document ingestion operations.

This service is responsible for:
- Text ingestion (ingest_text)
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
import base64
import json
import logging
import os
import tempfile
import uuid
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import arq
import filetype
import fitz  # PyMuPDF
import pdf2image
from fastapi import HTTPException, UploadFile
from filetype.types import IMAGE
from PIL import Image as PILImage

from core.config import get_settings
from core.database.base_database import BaseDatabase
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.limits_utils import check_and_increment_limits, estimate_pages_by_chars
from core.models.auth import AuthContext
from core.models.chunk import Chunk, DocumentChunk
from core.models.documents import Document, StorageFileInfo
from core.models.folders import Folder
from core.parser.base_parser import BaseParser
from core.storage.base_storage import BaseStorage
from core.storage.utils_file_extensions import detect_file_type
from core.utils.typed_metadata import merge_metadata, normalize_metadata
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
    - Text and file ingestion
    - Document updates
    - ColPali multi-vector processing
    - Chunk creation and storage

    The service is designed to be instantiated with only the dependencies needed
    for the current operation mode (standard vs ColPali local vs ColPali API).
    """

    _SYSTEM_METADATA_SCOPE_KEYS = {"folder_name", "end_user_id", "app_id"}
    _USER_IMMUTABLE_FIELDS = {"folder_name", "external_id"}

    def __init__(
        self,
        database: BaseDatabase,
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
        folder_name: Optional[Union[str, List[str]]],
        extra_fields: Optional[Dict[str, Any]] = None,
        context: str = "ingest",
    ) -> None:
        """Prevent users from setting reserved system fields directly."""
        invalid_fields = set()

        if isinstance(metadata, dict):
            invalid_fields.update({key for key in metadata.keys() if key in self._USER_IMMUTABLE_FIELDS})

        if isinstance(extra_fields, dict):
            invalid_fields.update({key for key in extra_fields.keys() if key in self._USER_IMMUTABLE_FIELDS})

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

            # Validate folder name - no slashes allowed (nested folders not supported)
            if "/" in folder_name:
                error_msg = (
                    f"Invalid folder name '{folder_name}'. Folder names cannot contain '/'. "
                    f"Nested folders are not supported. Use '_' instead to denote subfolders "
                    f"(e.g., 'folder_subfolder_subsubfolder')."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # First check if the folder already exists
            folder = await self.db.get_folder_by_name(folder_name, auth)
            if folder:
                # Add document to existing folder
                if document_id not in folder.document_ids:
                    success = await self.db.add_document_to_folder(folder.id, document_id, auth)
                    if not success:
                        logger.warning(
                            f"Failed to add document {document_id} to existing folder {folder.name}. "
                            "This may be due to a race condition during ingestion."
                        )
                    else:
                        logger.info(f"Successfully added document {document_id} to existing folder {folder.name}")
                else:
                    logger.info(f"Document {document_id} is already in folder {folder.name}")
                return folder

            # Create a new folder
            folder = Folder(
                name=folder_name,
                document_ids=[document_id],
                app_id=auth.app_id,
            )

            await self.db.create_folder(folder, auth)
            return folder

        except Exception as e:
            logger.error(f"Error ensuring folder exists: {e}")
            return None

    # -------------------------------------------------------------------------
    # Text ingestion
    # -------------------------------------------------------------------------

    async def ingest_text(
        self,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_types: Optional[Dict[str, str]] = None,
        auth: AuthContext = None,
        use_colpali: Optional[bool] = None,
        folder_name: Optional[str] = None,
        end_user_id: Optional[str] = None,
    ) -> Document:
        """Ingest a text document."""
        if "write" not in auth.permissions:
            logger.error(f"User {auth.entity_id} does not have write permission")
            raise PermissionError("User does not have write permission")

        # Prevent callers from overriding reserved fields
        self._enforce_no_user_mutable_fields(metadata, folder_name, context="ingest")

        doc = Document(
            content_type="text/plain",
            filename=filename,
            metadata=metadata or {},
            folder_name=folder_name,
            end_user_id=end_user_id,
            app_id=auth.app_id,
        )

        logger.debug(f"Created text document record with ID {doc.external_id}")

        combined_metadata = dict(metadata or {})
        combined_metadata.setdefault("external_id", doc.external_id)
        if folder_name is not None:
            combined_metadata["folder_name"] = folder_name
        normalized_metadata, normalized_types = normalize_metadata(combined_metadata, metadata_types)
        doc.metadata = normalized_metadata
        doc.metadata_types = normalized_types

        if settings.MODE == "cloud" and auth.user_id:
            # Verify limits before heavy processing
            num_pages = estimate_pages_by_chars(len(content))
            await check_and_increment_limits(
                auth,
                "ingest",
                num_pages,
                doc.external_id,
                verify_only=True,
            )

        doc.system_metadata["content"] = content

        # Split text into chunks
        parsed_chunks = await self.parser.split_text(content)
        if not parsed_chunks:
            raise ValueError("No content chunks extracted from document text")
        logger.debug(f"Split processed text into {len(parsed_chunks)} chunks")

        processed_chunks = parsed_chunks

        # Generate embeddings for processed chunks
        embeddings = await self.embedding_model.embed_for_ingestion(processed_chunks)
        logger.debug(f"Generated {len(embeddings)} embeddings")

        # Create chunk objects with processed chunk content
        chunk_objects = self._create_chunk_objects(doc.external_id, processed_chunks, embeddings)
        logger.debug(f"Created {len(chunk_objects)} chunk objects")

        chunk_objects_multivector = []

        # Check both use_colpali parameter AND global enable_colpali setting
        if use_colpali and settings.ENABLE_COLPALI and self.colpali_embedding_model:
            embeddings_multivector = await self.colpali_embedding_model.embed_for_ingestion(processed_chunks)
            logger.info(f"Generated {len(embeddings_multivector)} embeddings for multivector embedding")
            chunk_objects_multivector = self._create_chunk_objects(
                doc.external_id, processed_chunks, embeddings_multivector
            )
            logger.info(f"Created {len(chunk_objects_multivector)} chunk objects for multivector embedding")

        # Store everything
        await self._store_chunks_and_doc(
            chunk_objects,
            doc,
            use_colpali and settings.ENABLE_COLPALI,
            chunk_objects_multivector,
            auth=auth,
        )
        logger.debug(f"Successfully stored text document {doc.external_id}")

        # Ensure folder membership now that the document is persisted
        if folder_name:
            try:
                await self._ensure_folder_exists(folder_name, doc.external_id, auth)
            except Exception as folder_exc:
                logger.warning(
                    "Failed to ensure folder %s contains text document %s: %s",
                    folder_name,
                    doc.external_id,
                    folder_exc,
                )

        colpali_count_for_limit_fn = (
            len(chunk_objects_multivector)
            if use_colpali and settings.ENABLE_COLPALI and chunk_objects_multivector
            else None
        )
        final_page_count = estimate_pages_by_chars(len(content))
        if use_colpali and settings.ENABLE_COLPALI and colpali_count_for_limit_fn is not None:
            final_page_count = colpali_count_for_limit_fn
        final_page_count = max(1, final_page_count)
        doc.system_metadata["page_count"] = final_page_count
        logger.info(f"Determined final page count for ingest_text usage: {final_page_count}")

        # Update the document status to completed after successful storage
        doc.system_metadata["status"] = "completed"
        doc.system_metadata["updated_at"] = datetime.now(UTC)
        doc.system_metadata = self._clean_system_metadata(doc.system_metadata)
        await self.db.update_document(
            document_id=doc.external_id, updates={"system_metadata": doc.system_metadata}, auth=auth
        )
        logger.debug(f"Updated document status to 'completed' for {doc.external_id}")

        # Record ingest usage after successful completion
        if settings.MODE == "cloud" and auth.user_id:
            try:
                await check_and_increment_limits(
                    auth,
                    "ingest",
                    final_page_count,
                    doc.external_id,
                    use_colpali=use_colpali and settings.ENABLE_COLPALI,
                    colpali_chunks_count=colpali_count_for_limit_fn,
                )
            except Exception as rec_exc:
                logger.error("Failed to record ingest usage in ingest_text: %s", rec_exc)

        return doc

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
            f"Starting ingestion for filename: {filename}, content_type: {content_type}, "
            f"user: {auth.user_id or auth.entity_id}"
        )

        # Ensure user has write permission
        if "write" not in auth.permissions:
            logger.error(f"User {auth.entity_id} does not have write permission for ingest_file_content")
            raise PermissionError("User does not have write permission for ingest_file_content")

        doc = Document(
            filename=filename,
            content_type=content_type,
            metadata=metadata or {},
            system_metadata={"status": "processing"},
            content_info={"type": "file", "mime_type": content_type},
            app_id=auth.app_id,
            end_user_id=end_user_id,
            folder_name=folder_name,
        )

        # Verify quotas before incurring heavy compute or storage
        if settings.MODE == "cloud" and auth.user_id:
            num_pages = estimate_pages_by_chars(len(file_content_bytes))

            await check_and_increment_limits(
                auth,
                "ingest",
                num_pages,
                doc.external_id,
                verify_only=True,
            )
            await check_and_increment_limits(auth, "storage_file", 1, verify_only=True)
            await check_and_increment_limits(
                auth,
                "storage_size",
                len(file_content_bytes),
                verify_only=True,
            )
            logger.info(
                "Quota verification passed for user %s – pages=%s, file=%s bytes",
                auth.user_id,
                num_pages,
                len(file_content_bytes),
            )

        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("external_id", doc.external_id)
        normalized_metadata, normalized_types = normalize_metadata(metadata_payload, metadata_types)
        doc.metadata = normalized_metadata
        doc.metadata_types = normalized_types

        # 1. Create initial document record in DB
        await self.db.store_document(doc, auth)
        logger.info(f"Initial document record created for {filename} (doc_id: {doc.external_id})")

        # 2. Save raw file to Storage
        file_key_suffix = str(uuid.uuid4())
        safe_filename = Path(filename or "").name or "uploaded_file"
        storage_key = f"ingest_uploads/{file_key_suffix}/{safe_filename}"
        if not Path(storage_key).suffix:
            detected_ext = detect_file_type(file_content_bytes)
            if detected_ext:
                storage_key = f"{storage_key}{detected_ext}"

        try:
            bucket_name, full_storage_path = await self._upload_to_app_bucket(
                auth=auth, content_bytes=file_content_bytes, key=storage_key, content_type=content_type
            )
            sfi = StorageFileInfo(
                bucket=bucket_name,
                key=full_storage_path,
                content_type=content_type,
                size=len(file_content_bytes),
                last_modified=datetime.now(UTC),
                version=1,
                filename=safe_filename,
            )
            doc.storage_info = {k: str(v) if v is not None else "" for k, v in sfi.model_dump().items()}
            doc.storage_files = [sfi]

            doc.system_metadata = self._clean_system_metadata(doc.system_metadata)
            await self.db.update_document(
                document_id=doc.external_id,
                updates={
                    "storage_info": doc.storage_info,
                    "storage_files": [sf.model_dump() for sf in doc.storage_files],
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

            # Record usage now that upload passed
            if settings.MODE == "cloud" and auth.user_id:
                try:
                    await check_and_increment_limits(auth, "storage_file", 1)
                    await check_and_increment_limits(auth, "storage_size", len(file_content_bytes))
                except Exception as rec_err:
                    logger.error("Failed recording usage for doc %s: %s", doc.external_id, rec_err)

        except Exception as e:
            logger.error(f"Failed to upload file {filename} (doc_id: {doc.external_id}) to storage or update DB: {e}")
            doc.system_metadata["status"] = "failed"
            doc.system_metadata["error"] = f"Storage upload/DB update failed: {str(e)}"
            try:
                await self.db.update_document(
                    doc.external_id,
                    {"system_metadata": self._clean_system_metadata(doc.system_metadata)},
                    auth=auth,
                )
            except Exception as db_update_err:
                logger.error(f"Additionally failed to mark doc {doc.external_id} as failed in DB: {db_update_err}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file to storage: {str(e)}")

        # 3. Ensure folder exists if folder_name is provided
        if folder_name:
            try:
                await self._ensure_folder_exists(folder_name, doc.external_id, auth)
                logger.debug(f"Ensured folder '{folder_name}' exists and contains document {doc.external_id}")
            except Exception as e:
                logger.error(f"Error during _ensure_folder_exists for doc {doc.external_id}: {e}. Continuing.")

        # 4. Enqueue background job for processing
        auth_dict = {
            "entity_type": auth.entity_type.value,
            "entity_id": auth.entity_id,
            "app_id": auth.app_id,
            "permissions": list(auth.permissions),
            "user_id": auth.user_id,
        }

        metadata_json_str = json.dumps(doc.metadata or {})
        metadata_types_json = json.dumps(doc.metadata_types or {})

        try:
            job = await redis.enqueue_job(
                "process_ingestion_job",
                _job_id=f"ingest:{doc.external_id}",
                document_id=doc.external_id,
                file_key=full_storage_path,
                bucket=bucket_name,
                original_filename=filename,
                content_type=content_type,
                metadata_json=metadata_json_str,
                metadata_types_json=metadata_types_json,
                auth_dict=auth_dict,
                use_colpali=use_colpali,
                folder_name=str(folder_name) if folder_name else None,
                end_user_id=end_user_id,
            )
            if job is None:
                logger.info("Connector file ingestion job already queued (doc_id=%s)", doc.external_id)
            else:
                logger.info(
                    "Connector file ingestion job queued with ID: %s for document: %s", job.job_id, doc.external_id
                )
        except Exception as e:
            logger.error(f"Failed to enqueue ingestion job for doc {doc.external_id} ({filename}): {e}")
            doc.system_metadata["status"] = "failed"
            doc.system_metadata["error"] = f"Failed to enqueue processing job: {str(e)}"
            try:
                await self.db.update_document(
                    doc.external_id,
                    {"system_metadata": self._clean_system_metadata(doc.system_metadata)},
                    auth=auth,
                )
            except Exception as db_update_err:
                logger.error(f"Additionally failed to mark doc {doc.external_id} as failed in DB: {db_update_err}")
            raise HTTPException(status_code=500, detail=f"Failed to enqueue document processing job: {str(e)}")

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
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> Optional[Document]:
        """
        Update a document with new content and/or metadata using the specified strategy.

        Args:
            document_id: ID of the document to update
            auth: Authentication context
            content: The new text content to add
            file: File to add
            filename: Optional new filename for the document
            metadata: Additional metadata to update
            update_strategy: Strategy for updating the document ('add' to append content)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Updated document if successful, None if failed
        """
        # Prevent callers from modifying reserved fields
        self._enforce_no_user_mutable_fields(metadata, folder_name=None, context="update")

        # Validate permissions and get document
        doc = await self._validate_update_access(document_id, auth)
        if not doc:
            return None

        # Get current content and determine update type
        raw_current_content = doc.system_metadata.get("content", "")
        current_content = self._normalize_document_content(raw_current_content)
        if current_content != raw_current_content:
            logger.warning(
                "Normalized non-textual stored content for document %s before applying update strategy",
                doc.external_id,
            )
            doc.system_metadata["content"] = current_content
        metadata_only_update = content is None and file is None and metadata is not None

        # Process content based on update type
        update_content = None
        file_content = None
        file_type = None
        file_content_base64 = None
        if content is not None:
            update_content = await self._process_text_update(content, doc, filename, metadata)
            update_content = self._normalize_document_content(update_content)
        elif file is not None:
            update_content, file_content, file_type, file_content_base64 = await self._process_file_update(
                file, doc, metadata
            )
            update_content = self._normalize_document_content(update_content)

            # Record storage usage for the newly uploaded file (cloud mode)
            if settings.MODE == "cloud" and auth.user_id:
                try:
                    await check_and_increment_limits(auth, "storage_file", 1)
                    await check_and_increment_limits(auth, "storage_size", len(file_content))
                except Exception as rec_err:
                    logger.error("Failed to record storage usage in update_document: %s", rec_err)
        elif not metadata_only_update:
            logger.error("Neither content nor file provided for document update")
            return None

        # Apply content update strategy if we have new content
        if update_content:
            if not current_content:
                logger.info(f"No current content found, using only new content of length {len(update_content)}")
                updated_content = update_content
            else:
                updated_content = self._apply_update_strategy(current_content, update_content, update_strategy)
                logger.info(
                    f"Applied update strategy '{update_strategy}': original length={len(current_content)}, "
                    f"new length={len(updated_content)}"
                )

            doc.system_metadata["content"] = updated_content
            logger.info(f"Updated system_metadata['content'] with content of length {len(updated_content)}")
        else:
            updated_content = current_content
            logger.info(f"No content update - keeping current content of length {len(current_content)}")

        # Update metadata and version information
        self._update_metadata_and_version(doc, metadata, metadata_types, update_strategy, file)

        # For metadata-only updates, we don't need to re-process chunks
        if metadata_only_update:
            return await self._update_document_metadata_only(doc, auth)

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
            logger.error(f"User {auth.entity_id} does not have write permission for document {document_id}")
            raise PermissionError(f"User does not have write permission for document {document_id}")

        return doc

    async def _process_text_update(
        self,
        content: str,
        doc: Document,
        filename: Optional[str],
        metadata: Optional[Dict[str, Any]],
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
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[str, bytes, Any, str]:
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
        file_content_base64 = base64.b64encode(file_content).decode()

        # Store file in storage and update storage info
        await self._update_storage_info(doc, file, file_content_base64)

        # Store file type
        file_type = filetype.guess(file_content)
        if file_type:
            doc.content_type = file_type.mime
        else:
            import mimetypes

            guessed_type = mimetypes.guess_type(file.filename)[0]
            if guessed_type:
                doc.content_type = guessed_type
            else:
                doc.content_type = "text/plain" if file.filename.endswith(".txt") else "application/octet-stream"

        doc.filename = file.filename

        return file_text, file_content, file_type, file_content_base64

    async def _update_storage_info(self, doc: Document, file: UploadFile, file_content_base64: str):
        """Update document storage information for file content."""
        if not hasattr(doc, "storage_files") or not doc.storage_files:
            doc.storage_files = []

            if doc.storage_info and doc.storage_info.get("bucket") and doc.storage_info.get("key"):
                legacy_file_info = StorageFileInfo(
                    bucket=doc.storage_info.get("bucket", ""),
                    key=doc.storage_info.get("key", ""),
                    version=1,
                    filename=doc.filename,
                    content_type=doc.content_type,
                    timestamp=doc.system_metadata.get("updated_at", datetime.now(UTC)),
                )
                doc.storage_files.append(legacy_file_info)
                logger.info(f"Migrated legacy storage_info to storage_files: {doc.storage_files}")

        version = len(doc.storage_files) + 1
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ""

        storage_info_tuple = await self.storage.upload_from_base64(
            file_content_base64,
            f"{doc.external_id}_{version}{file_extension}",
            file.content_type,
            bucket="",
        )

        new_sfi = StorageFileInfo(
            bucket=storage_info_tuple[0],
            key=storage_info_tuple[1],
            version=version,
            filename=file.filename,
            content_type=file.content_type,
            timestamp=datetime.now(UTC),
        )
        doc.storage_files.append(new_sfi)

        doc.storage_info = {k: str(v) if v is not None else "" for k, v in new_sfi.model_dump().items()}
        logger.info(f"Stored file in bucket `{storage_info_tuple[0]}` with key `{storage_info_tuple[1]}`")

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
                return base64.b64encode(content).decode("utf-8")
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

    def _apply_update_strategy(self, current_content: str, update_content: str, update_strategy: str) -> str:
        """Apply the update strategy to combine current and new content."""
        if update_strategy == "add":
            return current_content + "\n\n" + update_content
        else:
            logger.warning(f"Unknown update strategy '{update_strategy}', defaulting to 'add'")
            return current_content + "\n\n" + update_content

    async def _update_document_metadata_only(self, doc: Document, auth: AuthContext) -> Optional[Document]:
        """Update document metadata without reprocessing chunks."""
        doc.system_metadata = self._clean_system_metadata(doc.system_metadata)

        updates = {
            "metadata": doc.metadata,
            "metadata_types": doc.metadata_types,
            "system_metadata": doc.system_metadata,
            "filename": doc.filename,
            "storage_files": doc.storage_files if hasattr(doc, "storage_files") else None,
            "storage_info": doc.storage_info if hasattr(doc, "storage_info") else None,
        }
        updates = {k: v for k, v in updates.items() if v is not None}

        success = await self.db.update_document(doc.external_id, updates, auth)
        if not success:
            logger.error(f"Failed to update document {doc.external_id} metadata")
            return None

        logger.info(f"Successfully updated document metadata for {doc.external_id}")
        return doc

    def _update_metadata_and_version(
        self,
        doc: Document,
        metadata: Optional[Dict[str, Any]],
        metadata_types: Optional[Dict[str, str]],
        update_strategy: str,
        file: Optional[UploadFile],
    ):
        """Update document metadata and version tracking."""
        if metadata:
            payload = dict(metadata)
            doc.metadata, doc.metadata_types = merge_metadata(
                doc.metadata,
                doc.metadata_types,
                payload,
                metadata_types,
                external_id=doc.external_id,
            )
        else:
            doc.metadata.setdefault("external_id", doc.external_id)
            doc.metadata_types.setdefault("external_id", "string")

        current_version = doc.system_metadata.get("version", 1)
        doc.system_metadata["version"] = current_version + 1
        doc.system_metadata["updated_at"] = datetime.now(UTC)

        history = doc.system_metadata.setdefault("update_history", [])
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": current_version + 1,
            "strategy": update_strategy,
        }
        if file:
            entry["filename"] = file.filename
        if metadata:
            entry["metadata_updated"] = True

        history.append(entry)

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

        file_type_mime = (
            file_type if isinstance(file_type, str) else (file_type.mime if file_type is not None else None)
        )
        if file and file_type_mime and (file_type_mime in IMAGE or file_type_mime == "application/pdf"):
            if hasattr(file, "seek") and callable(file.seek) and not file_content:
                await file.seek(0)
                file_content = await file.read()
                file_content_base64 = base64.b64encode(file_content).decode()

            chunks_multivector = self._create_chunks_multivector(file_type, file_content_base64, file_content, chunks)
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
                    sanitized_metadata[key] = base64.b64encode(bytes(value)).decode()
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
    ) -> List[str]:
        """Helper to store chunks and document."""
        max_retries = 3
        retry_delay = 1.0

        async def store_with_retry(store, objects, store_name="regular"):
            attempt = 0
            success = False
            current_retry_delay = retry_delay

            while attempt < max_retries and not success:
                try:
                    success, result = await store.store_embeddings(objects, auth.app_id if auth else None)
                    if not success:
                        raise Exception(f"Failed to store {store_name} chunk embeddings")
                    return result
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
                            "storage_files": (
                                [
                                    (
                                        file.model_dump()
                                        if hasattr(file, "model_dump")
                                        else (file.dict() if hasattr(file, "dict") else file)
                                    )
                                    for file in doc.storage_files
                                ]
                                if doc.storage_files
                                else []
                            ),
                        }
                        success = await self.db.update_document(doc.external_id, updates, auth)
                        if not success:
                            raise Exception("Failed to update document metadata")
                    else:
                        success = await self.db.store_document(doc, auth)
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
            chunk_ids = await store_with_retry(self.colpali_vector_store, chunk_objects_multivector, "colpali")
        else:
            chunk_ids = await store_with_retry(self.vector_store, chunk_objects, "regular")

        doc.chunk_ids = chunk_ids

        logger.debug(f"Stored chunk embeddings in vector stores: {len(doc.chunk_ids)} chunks total")

        await store_document_with_retry()

        logger.debug("Stored document metadata in database")
        logger.debug(f"Chunk IDs stored: {doc.chunk_ids}")
        return doc.chunk_ids

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
            content = f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode()
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
        img_str = f"data:{mime};base64," + base64.b64encode(img_bytes).decode()
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
                b64 = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")
                if include_bytes:
                    images.append((b64, png_bytes))
                else:
                    images.append(b64)
            return images
        finally:
            pdf_document.close()

    def _create_chunks_multivector(
        self,
        file_type,
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
        - Excel spreadsheets (converts to PDF, then to images)
        """
        # Derive a safe MIME type string regardless of input shape
        if isinstance(file_type, str):
            mime_type = file_type
        elif file_type is not None and hasattr(file_type, "mime"):
            mime_type = file_type.mime
        else:
            mime_type = "text/plain"
        logger.info(f"Creating chunks for multivector embedding for file type {mime_type}")

        # If file_type is None, attempt a light-weight heuristic to detect images
        if file_type is None:
            try:
                PILImage.open(BytesIO(file_content)).verify()
                logger.info("Heuristic image detection succeeded (Pillow). Treating as image.")
                if file_content_base64 is None:
                    file_content_base64 = base64.b64encode(file_content).decode()
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
        if mime_type.startswith("image/"):
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
                img_b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()
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
                    file_content_base64 = base64.b64encode(file_content).decode()
                return [
                    self._image_bytes_to_chunk(
                        file_content,
                        mime_type=mime_type,
                        base64_override=file_content_base64,
                    )
                ]

        match mime_type:
            case file_type if file_type in IMAGE:
                if file_content_base64 is None:
                    file_content_base64 = base64.b64encode(file_content).decode()
                detected_mime = mime_type if isinstance(mime_type, str) else "image/unknown"
                return [
                    self._image_bytes_to_chunk(
                        file_content,
                        mime_type=detected_mime,
                        base64_override=file_content_base64,
                    )
                ]

            case "application/pdf":
                return self._process_pdf_for_colpali(file_content, chunks)

            case "application/vnd.openxmlformats-officedocument.wordprocessingml.document" | "application/msword":
                return self._process_word_for_colpali(file_content, mime_type, chunks)

            case (
                "application/vnd.ms-powerpoint"
                | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                | "application/vnd.openxmlformats-officedocument.presentationml.slideshow"
            ):
                return self._process_powerpoint_for_colpali(file_content, mime_type, chunks)

            case (
                "application/vnd.ms-excel"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/vnd.ms-excel.sheet.macroEnabled.12"
            ):
                return self._process_excel_for_colpali(file_content, mime_type, chunks)

            case _:
                logger.warning(f"Colpali is not supported for file type {mime_type} - skipping")
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks
                ]

    def _process_pdf_for_colpali(self, file_content: bytes, chunks: List[Chunk]) -> List[Chunk]:
        """Process PDF file for ColPali embedding."""
        logger.info("Working with PDF file - using PyMuPDF for faster processing!")

        if not file_content:
            logger.error("PDF file content is empty")
            raise PdfConversionError("PDF file content is empty")

        dpi = settings.COLPALI_PDF_DPI

        try:
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

    def _process_word_for_colpali(self, file_content: bytes, mime_type: str, chunks: List[Chunk]) -> List[Chunk]:
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

    def _process_excel_for_colpali(self, file_content: bytes, mime_type: str, chunks: List[Chunk]) -> List[Chunk]:
        """Process Excel spreadsheet for ColPali embedding."""
        logger.info("Working with Excel spreadsheet!")

        if not file_content or len(file_content) == 0:
            logger.error("Excel spreadsheet content is empty")
            return [Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks]

        suffix = ".xls" if mime_type == "application/vnd.ms-excel" else ".xlsx"
        return self._convert_office_to_images(file_content, suffix, "Excel spreadsheet", chunks)

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
                timeout=30,
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
        auth: AuthContext,
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

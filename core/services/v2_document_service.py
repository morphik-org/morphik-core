import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from html import escape as html_escape
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import arq
from fastapi import HTTPException

from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.limits_utils import check_and_increment_limits, estimate_pages_by_chars
from core.models.auth import AuthContext
from core.models.chunk import Chunk
from core.models.documents import Document
from core.models.folders import Folder
from core.parser.docling_v2 import DoclingV2Parser
from core.parser.morphik_parser import MorphikParser
from core.storage.base_storage import BaseStorage
from core.storage.utils_file_extensions import detect_content_type
from core.utils.folder_utils import normalize_folder_path, normalize_ingest_folder_inputs
from core.utils.typed_metadata import MetadataBundle, normalize_metadata
from core.vector_store.chunk_v2_store import ChunkV2Store

logger = logging.getLogger(__name__)
settings = get_settings()


class V2DocumentService:
    """Service for v2 ingestion + retrieval (chunk_v2 store)."""

    _TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".json",
        ".csv",
        ".tsv",
        ".log",
        ".rst",
        ".yaml",
        ".yml",
    }

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
        storage: BaseStorage,
        parser: MorphikParser,
        embedding_model: BaseEmbeddingModel,
        chunk_store: ChunkV2Store,
    ):
        self.db = database
        self.storage = storage
        self.parser = parser
        self.embedding_model = embedding_model
        self.chunk_store = chunk_store
        self.docling_parser = DoclingV2Parser(settings)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enforce_no_user_mutable_fields(
        self,
        metadata: Optional[Dict[str, Any]],
        metadata_types: Optional[Dict[str, Any]] = None,
        context: str = "ingest",
    ) -> None:
        invalid_fields = set()
        if isinstance(metadata, dict):
            invalid_fields.update({key for key in metadata.keys() if key in self._USER_IMMUTABLE_FIELDS})
        if isinstance(metadata_types, dict):
            invalid_fields.update({key for key in metadata_types.keys() if key in self._USER_IMMUTABLE_FIELDS})
        if invalid_fields:
            fields_str = ", ".join(sorted(invalid_fields))
            raise ValueError(
                f"The following fields are managed by Morphik and cannot be set during {context}: {fields_str}. "
                "Remove them from the request."
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
        from core.storage.utils_file_extensions import detect_file_type

        safe_filename = Path(filename or "").name or "uploaded_file"
        storage_key = f"ingest_uploads/{uuid.uuid4()}/{safe_filename}"
        if not Path(storage_key).suffix:
            detected_ext = detect_file_type(content_bytes)
            if detected_ext:
                storage_key = f"{storage_key}{detected_ext}"
                if not Path(safe_filename).suffix:
                    safe_filename = f"{safe_filename}{detected_ext}"
        return storage_key, safe_filename

    async def _upload_content_bytes(
        self,
        *,
        content_bytes: bytes,
        filename: Optional[str],
        content_type: Optional[str],
    ) -> Tuple[str, str, str]:
        storage_key, safe_filename = self._build_storage_key(filename, content_bytes)
        bucket_name, full_storage_path = await self.storage.upload_file(
            file=content_bytes,
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
        await check_and_increment_limits(auth, "ingest", num_pages, document_id, verify_only=True)
        await check_and_increment_limits(auth, "storage_file", 1, verify_only=True)
        await check_and_increment_limits(auth, "storage_size", content_length, verify_only=True)

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
        if not key or not hasattr(self.storage, "get_object_size"):
            return None
        try:
            return await self.storage.get_object_size(bucket, key)
        except Exception as size_err:  # noqa: BLE001
            logger.warning("Failed reading stored size for %s/%s: %s", bucket, key, size_err)
            return None

    @staticmethod
    def _strip_xml_tags(text: str) -> str:
        import html

        without_tags = re.sub(r"<[^>]+>", " ", text)
        collapsed = re.sub(r"\s+", " ", without_tags).strip()
        return html.unescape(collapsed)

    @staticmethod
    def _build_auth_dict(auth: AuthContext) -> Dict[str, Any]:
        user_id = getattr(auth, "user_id", None)
        return {
            "user_id": user_id,
            "entity_id": user_id,
            "app_id": auth.app_id,
        }

    @staticmethod
    def _resolve_content_type(content_bytes: bytes, filename: Optional[str], content_type_hint: Optional[str]) -> str:
        return detect_content_type(content=content_bytes, filename=filename, content_type_hint=content_type_hint)

    @staticmethod
    def _build_storage_info(
        bucket: str, key: str, filename: Optional[str], content_type: Optional[str]
    ) -> Dict[str, str]:
        return {
            "bucket": bucket,
            "key": key,
            "filename": filename or "",
            "content_type": content_type or "",
        }

    @staticmethod
    def _reset_processing_metadata(system_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cleaned_metadata = dict(system_metadata or {})
        cleaned_metadata.pop("progress", None)
        cleaned_metadata.pop("error", None)
        cleaned_metadata["status"] = "processing"
        cleaned_metadata["updated_at"] = datetime.now(UTC)
        return cleaned_metadata

    @staticmethod
    def _folder_update_fields(folder_obj: Folder) -> Dict[str, Any]:
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

    async def _ensure_folder_exists(self, folder_path: str, document_id: str, auth: AuthContext) -> Optional[Folder]:
        """
        Ensure a folder path exists (creating ancestors as needed) and add the document to the leaf.
        """
        try:
            canonical_path = normalize_folder_path(folder_path)
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

            if document_id not in (target_folder.document_ids or []):
                success = await self.db.add_document_to_folder(target_folder.id, document_id, auth)
                if not success:
                    logger.warning(
                        "Failed to add document %s to folder %s. This may be due to a race condition.",
                        document_id,
                        target_folder.name,
                    )
                else:
                    logger.info("Successfully added document %s to folder %s", document_id, target_folder.name)

            return target_folder

        except Exception as exc:  # noqa: BLE001
            logger.error("Error ensuring folder exists: %s", exc)
            return None

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
        folder_path: Optional[str] = None,
        end_user_id: Optional[str] = None,
        force_plain_text: bool = False,
    ) -> Dict[str, Any]:
        return {
            "_job_id": f"v2_ingest:{document_id}",
            "_expires": timedelta(days=7),
            "document_id": document_id,
            "file_key": file_key,
            "bucket": bucket,
            "original_filename": original_filename,
            "content_type": content_type,
            "auth_dict": cls._build_auth_dict(auth),
            "folder_path": folder_path,
            "end_user_id": end_user_id,
            "force_plain_text": bool(force_plain_text),
        }

    async def _mark_document_failed(self, doc: Document, auth: AuthContext, error: str) -> None:
        failure_metadata = dict(doc.system_metadata or {})
        failure_metadata["status"] = "failed"
        failure_metadata["error"] = error
        failure_metadata["updated_at"] = datetime.now(UTC)
        doc.system_metadata = failure_metadata
        try:
            await self.db.update_document(doc.external_id, {"system_metadata": failure_metadata}, auth=auth)
        except Exception as db_update_err:  # noqa: BLE001
            logger.error("Failed to mark doc %s as failed: %s", doc.external_id, db_update_err)

    @staticmethod
    def _is_rich_doc(filename: str) -> bool:
        """Check if filename indicates a rich document (has bbox/pages)."""
        ext = Path(filename).suffix.lower()
        return ext in {".pdf", ".pptx", ".docx"}

    @staticmethod
    def _strip_metadata_scope(metadata: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(metadata)
        for key in {"folder_name", "folder_id", "end_user_id", "app_id"}:
            cleaned.pop(key, None)
        return cleaned

    @staticmethod
    def _convert_office_to_pdf_bytes(file_bytes: bytes, suffix: str, doc_type: str) -> bytes:
        if not shutil.which("soffice"):
            raise HTTPException(
                status_code=500,
                detail=f"LibreOffice is required to convert {doc_type} to PDF. Install 'soffice' and retry.",
            )

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_input:
            temp_input.write(file_bytes)
            temp_input_path = temp_input.name

        with tempfile.TemporaryDirectory() as output_dir:
            try:
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
                    raise HTTPException(
                        status_code=500,
                        detail=f"LibreOffice conversion failed: {result.stderr.strip() or result.stdout.strip()}",
                    )

                base_filename = Path(temp_input_path).stem
                pdf_path = Path(output_dir) / f"{base_filename}.pdf"
                if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                    raise HTTPException(status_code=500, detail="LibreOffice produced an empty PDF.")

                return pdf_path.read_bytes()
            finally:
                try:
                    os.unlink(temp_input_path)
                except OSError:
                    pass

    async def _build_xml_chunks(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        force_plain_text: bool = False,
    ) -> Tuple[List[Tuple[str, int]], bool]:
        xml_chunks: List[Tuple[str, int]] = []
        ext = Path(filename).suffix.lower()
        is_rich = self._is_rich_doc(filename)

        if force_plain_text or ext in self._TEXT_EXTENSIONS:
            is_rich = False
            text_content = file_bytes.decode("utf-8", errors="ignore")
            chunks = await self.parser.split_text(text_content)
            if not chunks:
                raise ValueError("No text chunks produced for plain text document")

            file_attr = html_escape(filename, quote=True)
            doc_attr = html_escape(document_id, quote=True)
            for idx, chunk in enumerate(chunks, start=1):
                escaped_text = html_escape(chunk.content, quote=False)
                xml = f'<doc id="{doc_attr}" file="{file_attr}" chunk="{idx}">' f"{escaped_text}" "</doc>"
                xml_chunks.append((xml, idx))
            return xml_chunks, is_rich

        if ext in {".doc"}:
            raise HTTPException(status_code=400, detail=".doc files are not supported in v2")
        if ext in {".ppt"}:
            raise HTTPException(status_code=400, detail=".ppt files are not supported in v2")
        if ext in {".xls", ".xlsx"}:
            raise HTTPException(status_code=400, detail="Excel files are not supported in v2")
        if ext in {".html", ".htm"}:
            raise HTTPException(status_code=400, detail="HTML files are not supported in v2")
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}:
            raise HTTPException(status_code=400, detail="Images are not supported in v2")

        parse_bytes = file_bytes
        parse_filename = filename
        if ext == ".docx":
            parse_bytes = self._convert_office_to_pdf_bytes(file_bytes, ".docx", "Word document")
            parse_filename = f"{Path(filename).stem}.pdf"

        if ext == ".pptx":
            try:
                xml_chunks = await self.docling_parser.parse(
                    parse_bytes,
                    parse_filename,
                    document_id,
                    display_filename=filename,
                )
            except Exception:  # noqa: BLE001
                parse_bytes = self._convert_office_to_pdf_bytes(
                    file_bytes,
                    ".pptx",
                    "PowerPoint presentation",
                )
                parse_filename = f"{Path(filename).stem}.pdf"
                xml_chunks = await self.docling_parser.parse(
                    parse_bytes,
                    parse_filename,
                    document_id,
                    display_filename=filename,
                )
        else:
            xml_chunks = await self.docling_parser.parse(
                parse_bytes,
                parse_filename,
                document_id,
                display_filename=filename,
            )

        if not xml_chunks:
            raise ValueError("No page chunks extracted from document")

        return xml_chunks, is_rich

    async def process_document_bytes(
        self,
        *,
        doc: Document,
        file_bytes: bytes,
        auth: AuthContext,
        force_plain_text: bool = False,
        progress_cb: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        filename = doc.filename or "uploaded_file"

        try:
            if progress_cb:
                await progress_cb("Parsing file")
            xml_chunks, is_rich = await self._build_xml_chunks(
                file_bytes=file_bytes,
                filename=filename,
                document_id=doc.external_id,
                force_plain_text=force_plain_text,
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._mark_document_failed(doc, auth, f"Parsing failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Failed to parse document: {exc}") from exc

        stripped_texts = [self._strip_xml_tags(chunk_xml) for chunk_xml, _ in xml_chunks]
        embedding_chunks = [Chunk(content=text, metadata={}) for text in stripped_texts]

        try:
            if progress_cb:
                await progress_cb("Generating embeddings")
            embeddings = await self.embedding_model.embed_for_ingestion(embedding_chunks)
            if len(embeddings) != len(xml_chunks):
                raise ValueError("Embedding count mismatch")
        except Exception as exc:  # noqa: BLE001
            await self._mark_document_failed(doc, auth, f"Embedding failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {exc}") from exc

        metadata_values = doc.metadata or {}
        metadata_types = doc.metadata_types or {}

        chunk_records = []
        for idx, (xml, number) in enumerate(xml_chunks):
            chunk_records.append(
                {
                    "id": uuid.uuid4(),
                    "document_id": doc.external_id,
                    "content": xml,
                    "embedding": embeddings[idx],
                    "page_number": number if is_rich else None,
                    "chunk_number": number if not is_rich else None,
                    "app_id": auth.app_id,
                    "end_user_id": doc.end_user_id,
                    "folder_path": doc.folder_path,
                    "doc_metadata": metadata_values,
                    "metadata_types": metadata_types,
                    "filename": filename,
                }
            )

        stored_ids: List[str] = []
        store_metrics: Dict[str, Any] = {}
        try:
            if progress_cb:
                await progress_cb("Storing chunks")
            _success, stored_ids, store_metrics = await self.chunk_store.store_chunks(chunk_records)
        except Exception as exc:  # noqa: BLE001
            await self._mark_document_failed(doc, auth, f"Chunk storage failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Failed to store chunks: {exc}") from exc

        page_count = len(xml_chunks)
        doc.system_metadata = dict(doc.system_metadata or {})
        doc.system_metadata["page_count"] = max(1, page_count)
        doc.system_metadata["status"] = "completed"
        doc.system_metadata["updated_at"] = datetime.now(UTC)

        try:
            await self.db.update_document(doc.external_id, {"system_metadata": doc.system_metadata}, auth=auth)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to update v2 document status: %s", exc)

        if settings.MODE == "cloud" and auth.user_id:
            try:
                await check_and_increment_limits(auth, "ingest", doc.system_metadata["page_count"], doc.external_id)
            except Exception as rec_exc:  # noqa: BLE001
                logger.error("Failed to record v2 ingest usage: %s", rec_exc)

        if store_metrics:
            try:
                await self.db.record_document_storage_deltas(
                    doc.external_id,
                    auth.app_id,
                    chunk_bytes_delta=int(store_metrics.get("chunk_payload_bytes") or 0),
                    multivector_bytes_delta=0,
                )
            except Exception as rec_exc:  # noqa: BLE001
                logger.error("Failed to record v2 chunk storage bytes: %s", rec_exc)

        return {
            "document_id": doc.external_id,
            "filename": filename,
            "chunk_count": len(xml_chunks),
            "chunk_ids": stored_ids,
        }

    async def ingest_document(
        self,
        *,
        file_bytes: Optional[bytes],
        filename: Optional[str],
        content: Optional[str],
        content_type: Optional[str],
        metadata: Optional[Dict[str, Any]],
        metadata_types: Optional[Dict[str, str]],
        folder_path: Optional[str],
        end_user_id: Optional[str],
        auth: AuthContext,
        redis: Optional[arq.ArqRedis] = None,
        queue: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not auth.app_id:
            raise HTTPException(status_code=403, detail="app_id is required for v2 ingestion")
        if bool(file_bytes) == bool(content):
            raise ValueError("Provide either file or content, not both.")

        self._enforce_no_user_mutable_fields(metadata, metadata_types=metadata_types, context="ingest")

        force_plain_text = content is not None
        if content is not None:
            if not filename:
                raise ValueError("filename is required when using content")
            filename = self._normalize_text_filename(filename, content)
            file_bytes = content.encode("utf-8")
            resolved_content_type = self._resolve_content_type(file_bytes, filename, content_type)
        else:
            if not file_bytes:
                raise ValueError("file bytes are required when ingesting a file")
            filename = filename or "uploaded_file"
            resolved_content_type = self._resolve_content_type(file_bytes, filename, content_type)

        normalized_folder = normalize_ingest_folder_inputs(folder_path=folder_path)
        folder_path_value, folder_leaf = normalized_folder.path, normalized_folder.leaf

        doc = Document(
            filename=filename,
            content_type=resolved_content_type,
            metadata=metadata or {},
            app_id=auth.app_id,
            end_user_id=end_user_id,
            folder_name=folder_leaf,
            folder_path=folder_path_value,
        )
        doc.system_metadata = self._reset_processing_metadata(doc.system_metadata)

        await self._verify_ingest_and_storage_limits(auth, len(file_bytes), doc.external_id)

        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("external_id", doc.external_id)
        if normalized_folder.metadata_value is not None:
            metadata_payload["folder_name"] = normalized_folder.metadata_value

        metadata_bundle: MetadataBundle = normalize_metadata(metadata_payload, metadata_types)
        doc.metadata = metadata_bundle.values
        doc.metadata_types = metadata_bundle.types

        await self.db.store_document(doc, auth, metadata_bundle=metadata_bundle)
        logger.info("v2 document record created for %s (doc_id=%s)", filename, doc.external_id)

        try:
            bucket_name, storage_key, safe_filename = await self._upload_content_bytes(
                content_bytes=file_bytes,
                filename=filename,
                content_type=resolved_content_type,
            )
            doc.storage_info = self._build_storage_info(bucket_name, storage_key, safe_filename, resolved_content_type)
            doc.system_metadata = self._strip_metadata_scope(doc.system_metadata)
            await self.db.update_document(
                document_id=doc.external_id,
                updates={"storage_info": doc.storage_info, "system_metadata": doc.system_metadata},
                auth=auth,
            )
            await self._record_storage_usage(auth, len(file_bytes), doc.external_id)
            stored_size = await self._get_storage_object_size(bucket_name, storage_key)
            if stored_size is not None:
                await self._record_raw_storage_bytes(auth, doc.external_id, stored_size)
        except Exception as exc:  # noqa: BLE001
            await self._mark_document_failed(doc, auth, f"Storage upload failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file to storage: {exc}") from exc

        if folder_path_value:
            try:
                folder_obj = await self._ensure_folder_exists(folder_path_value, doc.external_id, auth)
                if folder_obj and folder_obj.id:
                    doc.folder_id = folder_obj.id
                    folder_updates = self._folder_update_fields(folder_obj)
                    await self.db.update_document(doc.external_id, folder_updates, auth=auth)
                logger.debug("Ensured folder '%s' exists and contains document %s", folder_path_value, doc.external_id)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Error ensuring folder exists for doc %s (path=%s): %s",
                    doc.external_id,
                    folder_path_value,
                    exc,
                )

        if queue is None:
            queue = redis is not None

        if queue:
            if redis is None:
                raise ValueError("redis pool is required to queue v2 ingestion")
            try:
                job_payload = self._build_ingestion_job_payload(
                    document_id=doc.external_id,
                    file_key=storage_key,
                    bucket=bucket_name,
                    original_filename=filename,
                    content_type=resolved_content_type,
                    auth=auth,
                    folder_path=folder_path_value,
                    end_user_id=end_user_id,
                    force_plain_text=force_plain_text,
                )
                job = await redis.enqueue_job("process_v2_ingestion_job", **job_payload)
                if job is None:
                    logger.info("V2 ingestion job already queued (doc_id=%s)", doc.external_id)
                else:
                    logger.info("V2 ingestion job queued (job_id=%s, doc=%s)", job.job_id, doc.external_id)
            except Exception as exc:  # noqa: BLE001
                await self._mark_document_failed(doc, auth, f"Failed to enqueue processing job: {exc}")
                raise HTTPException(status_code=500, detail=f"Failed to enqueue v2 processing job: {exc}") from exc

            return {
                "document_id": doc.external_id,
                "filename": filename,
                "chunk_count": 0,
                "status": "queued",
            }

        result = await self.process_document_bytes(
            doc=doc,
            file_bytes=file_bytes,
            auth=auth,
            force_plain_text=force_plain_text,
        )
        result["status"] = "completed"
        return result

    async def retrieve_chunks(
        self,
        *,
        query: str,
        top_k: int,
        auth: AuthContext,
        document_ids: Optional[List[str]] = None,
        folder_paths: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        end_user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not auth.app_id:
            raise HTTPException(status_code=403, detail="app_id is required for v2 retrieval")
        embedding = await self.embedding_model.embed_query(query)
        return await self.chunk_store.query_similar(
            embedding,
            k=top_k,
            auth=auth,
            document_ids=document_ids,
            folder_paths=folder_paths,
            metadata_filters=metadata_filters,
            end_user_id=end_user_id,
        )

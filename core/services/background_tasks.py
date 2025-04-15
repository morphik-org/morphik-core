import logging
import traceback
from typing import Dict, Any, List, Optional
from fastapi import UploadFile
import base64
import filetype
import mimetypes

from core.models.documents import Document, DocumentStatus
from core.models.auth import AuthContext
from core.services.document_service import DocumentService

logger = logging.getLogger(__name__)

async def process_text_document(
    document_service: DocumentService,
    document_id: str,
    content: str,
    auth: AuthContext,
    rules: Optional[List[str]] = None,
    use_colpali: Optional[bool] = None,
    folder_name: Optional[str] = None,
    end_user_id: Optional[str] = None,
):
    """Process text document in the background"""
    try:
        # Get the document
        document = await document_service.db.get_document(document_id, auth)
        if not document:
            logger.error(f"Document {document_id} not found")
            return
        
        # Update the document status to processing
        await document_service.db.update_document(
            document_id,
            {"status": DocumentStatus.PROCESSING},
            auth
        )
        
        # Check permissions
        if "write" not in auth.permissions:
            error_msg = "User does not have write permission"
            await document_service.db.update_document(
                document_id,
                {"status": DocumentStatus.FAILED, "error_message": error_msg},
                auth
            )
            return
        
        # Apply rules if provided
        if rules:
            rule_metadata, modified_text = await document_service.rules_processor.process_rules(content, rules)
            # Update document metadata with extracted metadata from rules
            document.metadata.update(rule_metadata)
            
            if modified_text:
                content = modified_text
                logger.info("Updated content with modified text from rules")
                
        # Store full content before chunking
        document.system_metadata["content"] = content
        
        # Split into chunks after all processing is done
        chunks = await document_service.parser.split_text(content)
        if not chunks:
            error_msg = "No content chunks extracted"
            await document_service.db.update_document(
                document_id,
                {"status": DocumentStatus.FAILED, "error_message": error_msg},
                auth
            )
            return
            
        logger.debug(f"Split processed text into {len(chunks)} chunks")
        
        # Generate embeddings for chunks
        embeddings = await document_service.embedding_model.embed_for_ingestion(chunks)
        logger.debug(f"Generated {len(embeddings)} embeddings")
        chunk_objects = document_service._create_chunk_objects(document_id, chunks, embeddings)
        logger.debug(f"Created {len(chunk_objects)} chunk objects")
        
        chunk_objects_multivector = []
        
        if use_colpali and document_service.colpali_embedding_model:
            embeddings_multivector = await document_service.colpali_embedding_model.embed_for_ingestion(chunks)
            logger.info(
                f"Generated {len(embeddings_multivector)} embeddings for multivector embedding"
            )
            chunk_objects_multivector = document_service._create_chunk_objects(
                document_id, chunks, embeddings_multivector
            )
            logger.info(
                f"Created {len(chunk_objects_multivector)} chunk objects for multivector embedding"
            )
        
        # Store everything - use is_update=True to update the existing document
        document.chunk_ids = await document_service._store_chunks_and_doc(
            chunk_objects, document, use_colpali, chunk_objects_multivector, is_update=True, auth=auth
        )
        
        # Update document status to completed
        # Make sure to include system_metadata which contains the content
        await document_service.db.update_document(
            document_id,
            {
                "status": DocumentStatus.COMPLETED,
                "chunk_ids": document.chunk_ids,
                "system_metadata": document.system_metadata
            },
            auth
        )
        logger.info(f"Successfully processed text document {document_id}")
        
    except Exception as e:
        logger.error(f"Error processing text document {document_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update document status to failed with error message
        await document_service.db.update_document(
            document_id,
            {
                "status": DocumentStatus.FAILED,
                "error_message": str(e)
            },
            auth
        )

async def process_file_document(
    document_service: DocumentService,
    document_id: str,
    file_content: bytes,
    filename: str,
    content_type: str,
    metadata: Dict[str, Any],
    auth: AuthContext,
    rules: Optional[List[str]] = None,
    use_colpali: Optional[bool] = None,
    folder_name: Optional[str] = None,
    end_user_id: Optional[str] = None,
):
    """Process file document in the background"""
    try:
        # Get the document
        document = await document_service.db.get_document(document_id, auth)
        if not document:
            logger.error(f"Document {document_id} not found")
            return
            
        # Update the document status to processing
        await document_service.db.update_document(
            document_id,
            {"status": DocumentStatus.PROCESSING},
            auth
        )
        
        # Check permissions
        if "write" not in auth.permissions:
            error_msg = "User does not have write permission"
            await document_service.db.update_document(
                document_id,
                {"status": DocumentStatus.FAILED, "error_message": error_msg},
                auth
            )
            return
        
        # Determine file type
        file_type = filetype.guess(file_content)
        
        # Set default mime type for cases where filetype.guess returns None
        mime_type = ""
        if file_type is not None:
            mime_type = file_type.mime
        elif filename:
            # Try to determine by file extension as fallback
            guessed_type = mimetypes.guess_type(filename)[0]
            if guessed_type:
                mime_type = guessed_type
            else:
                # Default for text files
                mime_type = "text/plain"
        else:
            mime_type = "application/octet-stream"  # Generic binary data
        
        logger.info(f"Determined MIME type: {mime_type} for file {filename}")
        
        # Parse file to text
        additional_metadata, text = await document_service.parser.parse_file_to_text(
            file_content, filename
        )
        logger.debug(f"Parsed file into text of length {len(text)}")
        
        # Apply rules if provided
        if rules:
            rule_metadata, modified_text = await document_service.rules_processor.process_rules(text, rules)
            # Update document metadata with extracted metadata from rules
            metadata.update(rule_metadata)
            if modified_text:
                text = modified_text
                logger.info("Updated text with modified content from rules")
        
        document.content_type = mime_type or content_type
        document.metadata.update(metadata)
        
        if additional_metadata:
            document.additional_metadata.update(additional_metadata)
        
        # Store full content
        document.system_metadata["content"] = text
        
        # Encode file content for storage
        file_content_base64 = base64.b64encode(file_content).decode()
        
        # Store the original file
        storage_info = await document_service.storage.upload_from_base64(
            file_content_base64, document_id, document.content_type
        )
        document.storage_info = {"bucket": storage_info[0], "key": storage_info[1]}
        logger.debug(f"Stored file in bucket `{storage_info[0]}` with key `{storage_info[1]}`")
        
        # Split into chunks after all processing is done
        chunks = await document_service.parser.split_text(text)
        if not chunks:
            error_msg = "No content chunks extracted"
            await document_service.db.update_document(
                document_id,
                {"status": DocumentStatus.FAILED, "error_message": error_msg},
                auth
            )
            return
            
        logger.debug(f"Split processed text into {len(chunks)} chunks")
        
        # Generate embeddings for chunks
        embeddings = await document_service.embedding_model.embed_for_ingestion(chunks)
        logger.debug(f"Generated {len(embeddings)} embeddings")
        
        # Create and store chunk objects
        chunk_objects = document_service._create_chunk_objects(document_id, chunks, embeddings)
        logger.debug(f"Created {len(chunk_objects)} chunk objects")
        
        chunk_objects_multivector = []
        logger.debug(f"use_colpali: {use_colpali}")
        if use_colpali and document_service.colpali_embedding_model:
            chunks_multivector = document_service._create_chunks_multivector(
                file_type, file_content_base64, file_content, chunks
            )
            logger.debug(f"Created {len(chunks_multivector)} chunks for multivector embedding")
            colpali_embeddings = await document_service.colpali_embedding_model.embed_for_ingestion(
                chunks_multivector
            )
            logger.debug(f"Generated {len(colpali_embeddings)} embeddings for multivector embedding")
            chunk_objects_multivector = document_service._create_chunk_objects(
                document_id, chunks_multivector, colpali_embeddings
            )
        
        # Store everything - use is_update=True to update the existing document
        document.chunk_ids = await document_service._store_chunks_and_doc(
            chunk_objects, document, use_colpali, chunk_objects_multivector, is_update=True, auth=auth
        )
        
        # Update document status to completed
        # Make sure to include system_metadata which contains the content
        await document_service.db.update_document(
            document_id,
            {
                "status": DocumentStatus.COMPLETED,
                "chunk_ids": document.chunk_ids,
                "storage_info": document.storage_info,
                "content_type": document.content_type,
                "system_metadata": document.system_metadata
            },
            auth
        )
        logger.info(f"Successfully processed file document {document_id}")
        
    except Exception as e:
        logger.error(f"Error processing file document {document_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update document status to failed with error message
        await document_service.db.update_document(
            document_id,
            {
                "status": DocumentStatus.FAILED,
                "error_message": str(e)
            },
            auth
        )
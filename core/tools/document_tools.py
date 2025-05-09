"""Document retrieval and management tools."""

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, Optional

from core.models.auth import AuthContext
from core.models.documents import ChunkResult
from core.services.document_service import DocumentService

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Exception raised when a tool execution fails."""

    pass


async def retrieve_chunks(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    min_relevance: float = 0.7,
    use_colpali: bool = True,
    folder_name: Optional[str] = None,
    end_user_id: Optional[str] = None,
    document_service: DocumentService = None,
    auth: AuthContext = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant text and image chunks from the knowledge base.
    Returns a list of dictionaries, each representing a chunk with its ID, type, and content/description.
    """
    if document_service is None:
        raise ToolError("Document service not provided")

    try:
        # Directly await the document service method
        raw_chunks: List[ChunkResult] = await document_service.retrieve_chunks(
            query=query,
            auth=auth,
            filters=filters,
            k=k,
            min_score=min_relevance,
            use_colpali=use_colpali,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )

        # Format the results for LiteLLM tool response
        # The 'content' for the LLM will be a list of structured chunk information
        llm_chunk_info_list = []

        # Add a header text element (optional, but can be useful for the LLM's context)
        # content.append({"type": "text", "text": f"Found {len(raw_chunks)} relevant chunks:"})

        for chunk in raw_chunks:
            chunk_type = "image" if chunk.metadata.get("is_image", False) else "text"

            # Generate the unique ID
            # Ensure document_id and chunk_number are strings if they might not be.
            # Forcing string type for document_id just in case, chunk_number is int.
            doc_id_str = str(chunk.document_id)
            chunk_id_str = f"doc:{doc_id_str}::chunk:{chunk.chunk_number}::type:{chunk_type}"

            if chunk_type == "image":
                # For images, provide a description or placeholder.
                # The actual image data will be fetched by the grounding service using the ID.
                image_description = (
                    chunk.metadata.get("alt_text")
                    or chunk.metadata.get("description")
                    or f"Image from document {chunk.filename or doc_id_str}, chunk {chunk.chunk_number}"
                )

                llm_chunk_info_list.append(
                    {
                        "id": chunk_id_str,
                        "type": "image",
                        "description": image_description,
                        "source_document_id": doc_id_str,
                        "source_filename": chunk.filename,
                        "relevance_score": round(chunk.score, 3) if chunk.score is not None else None,
                    }
                )
            else:  # text chunk
                # Provide a snippet of the text content for the LLM
                # The full content can be retrieved by the grounding service if necessary,
                # but typically the snippet is enough for the LLM to decide if it's relevant.
                text_snippet = chunk.content
                if len(text_snippet) > 300:  # Keep snippet length reasonable for the prompt
                    text_snippet = text_snippet[:297] + "..."

                llm_chunk_info_list.append(
                    {
                        "id": chunk_id_str,
                        "type": "text",
                        "content_snippet": text_snippet,
                        "source_document_id": doc_id_str,
                        "source_filename": chunk.filename,
                        "relevance_score": round(chunk.score, 3) if chunk.score is not None else None,
                    }
                )

        # The tool will return this list of structured chunk information.
        # The LLM should be prompted to use the 'id' from these items in its citations.
        # Prepending a summary message for the LLM.
        if not llm_chunk_info_list:
            return [{"type": "text", "text": "No relevant chunks found."}]  # Return a text message if no chunks

        # Return structure for LLM: a list where first item is summary, rest are chunk infos.
        # Or, just the list of chunks. For function calling, often a direct list of the data items is better.
        # Let's return the list of chunk dicts directly.
        # The system prompt for the agent will need to explain how to use this.
        return llm_chunk_info_list
    except Exception as e:
        logger.error(f"Error retrieving or formatting chunks: {str(e)}", exc_info=True)  # Log with traceback
        raise ToolError(f"Error retrieving chunks: {str(e)}")


async def retrieve_document(
    document_id: str,
    format: Optional[Literal["text", "metadata"]] = "text",
    document_service: DocumentService = None,
    auth: AuthContext = None,
    end_user_id: Optional[str] = None,
) -> str:
    """
    Retrieve full content of a specific document.

    Args:
        document_id: ID of the document to retrieve
        format: Desired format of the returned document
        document_service: DocumentService instance
        auth: Authentication context
        end_user_id: Optional end-user ID to retrieve as

    Returns:
        Document content or metadata as a string
    """
    if document_service is None:
        raise ToolError("Document service not provided")

    try:
        # Directly await the document service method
        doc = await document_service.batch_retrieve_documents(
            document_ids=[document_id], auth=auth, end_user_id=end_user_id
        )

        if not doc or len(doc) == 0:
            return f"Document {document_id} not found or not accessible"

        doc = doc[0]  # Get the first document from the list

        if format == "text":
            return doc.system_metadata.get("content", "No content available")
        else:
            # Return both user-defined metadata and system metadata separately
            result: Dict[str, Any] = {}
            # User metadata
            if hasattr(doc, "metadata") and doc.metadata:
                result["metadata"] = doc.metadata
            # System metadata without content field
            if hasattr(doc, "system_metadata") and doc.system_metadata:
                system_metadata = doc.system_metadata.copy()
                if "content" in system_metadata:
                    del system_metadata["content"]
                result["system_metadata"] = system_metadata
            return json.dumps(result, indent=2, default=str)

    except Exception as e:
        raise ToolError(f"Error retrieving document: {str(e)}")


async def save_to_memory(
    content: str,
    memory_type: Literal["session", "long_term", "research_thread"],
    tags: Optional[List[str]] = None,
    document_service: DocumentService = None,
    auth: AuthContext = None,
    end_user_id: Optional[str] = None,
) -> str:
    """
    Save important information to persistent memory.

    Args:
        content: Content to save
        memory_type: Type of memory to save to
        tags: Tags for categorizing the memory
        document_service: DocumentService instance
        auth: Authentication context
        end_user_id: Optional end-user ID to save as

    Returns:
        Save operation result as a string
    """
    if document_service is None:
        raise ToolError("Document service not provided")

    try:
        # Create metadata for the saved memory
        metadata = {"memory_type": memory_type, "source": "agent"}

        if tags:
            metadata["tags"] = tags

        # Use document service to ingest the content as a document
        timestamp = await get_timestamp()
        result = await document_service.ingest_text(
            content=content,
            filename=f"memory_{memory_type}_{timestamp}",
            metadata=metadata,
            auth=auth,
            end_user_id=end_user_id,
        )

        return json.dumps({"success": True, "memory_id": result.external_id, "memory_type": memory_type})
    except Exception as e:
        raise ToolError(f"Error saving to memory: {str(e)}")


async def list_documents(
    filters: Optional[Dict[str, Any]] = None,
    skip: int = 0,
    limit: int = 100,
    folder_name: Optional[str] = None,
    end_user_id: Optional[str] = None,
    document_service: DocumentService = None,
    auth: AuthContext = None,
) -> str:
    """
    List accessible documents, showing their IDs and filenames.

    Args:
        filters: Optional metadata filters
        skip: Number of documents to skip (default: 0)
        limit: Maximum number of documents to return (default: 100)
        folder_name: Optional folder to scope the listing to
        end_user_id: Optional end-user ID to scope the listing to
        document_service: DocumentService instance
        auth: Authentication context

    Returns:
        JSON string list of documents with id and filename
    """
    if document_service is None:
        raise ToolError("Document service not provided")

    try:
        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name:
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Retrieve documents from the database
        docs = await document_service.db.get_documents(
            auth=auth, skip=skip, limit=limit, filters=filters, system_filters=system_filters
        )

        # Format the results to only include ID and filename
        formatted_docs = [{"id": doc.external_id, "filename": doc.filename} for doc in docs]

        return json.dumps({"count": len(formatted_docs), "documents": formatted_docs}, indent=2)

    except PermissionError as e:
        # Re-raise PermissionError as ToolError for consistent handling
        raise ToolError(str(e))
    except Exception as e:
        raise ToolError(f"Error listing documents: {str(e)}")


async def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(UTC).isoformat().replace(":", "-").replace(".", "-")

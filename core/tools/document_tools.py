"""Document retrieval and management tools."""

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, Optional

from core.models.auth import AuthContext
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

    Args:
        query: The search query or question
        k: Number of chunks to retrieve (default: 5)
        filters: Metadata filters to narrow results
        min_relevance: Minimum relevance score threshold (0-1)
        use_colpali: Whether to use multimodal features
        folder_name: Optional folder to scope the search to
        end_user_id: Optional end-user ID to scope the search to
        document_service: DocumentService instance
        auth: Authentication context

    Returns:
        List of content items with text and images
    """
    if document_service is None:
        raise ToolError("Document service not provided")

    try:
        # Directly await the document service method
        chunks = await document_service.retrieve_chunks(
            query=query,
            auth=auth,
            filters=filters,
            k=k,
            min_score=min_relevance,
            use_colpali=use_colpali,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )
        sources = {}

        # Format the results for LiteLLM tool response
        content = []

        # Add a header text element
        content.append({"type": "text", "text": f"Found {len(chunks)} relevant chunks:"})

        for chunk in chunks:
            # Create a unique source ID for this chunk
            source_id = f"doc{chunk.document_id}-chunk{chunk.chunk_number}"

            # Store source information
            sources[source_id] = {
                "document_id": chunk.document_id,
                "document_name": chunk.filename or "Unnamed Document",
                "chunk_number": chunk.chunk_number,
                "score": chunk.score,
                "content": chunk.content,
            }

            chunk_content = [{"type": "text", "text": f"Source ID: {source_id}"}]

            # Check if this is an image chunk
            if chunk.metadata.get("is_image", False):
                # Add image to content
                if chunk.content.startswith("data:"):
                    # Already in data URL format
                    chunk_content.append({"type": "image_url", "image_url": {"url": chunk.content}})
                else:
                    # Assuming it's base64, convert to data URL format
                    chunk_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chunk.content}"}}
                    )

                # Tell the agent this is a reference to an image
                chunk_content.append(
                    {
                        "type": "text",
                        "text": f"This is an image from {chunk.filename or 'Unnamed'} (Score: {chunk.score:.2f}). "
                        + f"When referencing this image, cite source: {source_id}",
                    }
                )
            else:
                # Add text content with metadata
                text = f"Document: {chunk.filename or 'Unnamed'} (Score: {chunk.score:.2f})\n"
                text += f"When referencing this content, cite source: {source_id}\n\n"
                text += chunk.content

                chunk_content.append(
                    {
                        "type": "text",
                        "text": text,
                    }
                )
            content.extend(chunk_content)
        return content, sources
    except Exception as e:
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

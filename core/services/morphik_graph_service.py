import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

import httpx

from core.completion.base_completion import BaseCompletionModel
from core.database.base_database import BaseDatabase
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.auth import AuthContext
from core.models.completion import ChunkSource, CompletionRequest, CompletionResponse
from core.models.documents import ChunkResult, Document
from core.models.graph import Graph
from core.models.prompts import GraphPromptOverrides, QueryPromptOverrides

logger = logging.getLogger(__name__)


class MorphikGraphService:
    """Service for managing knowledge graphs and graph-based operations"""

    def __init__(
        self,
        db: BaseDatabase,
        embedding_model: BaseEmbeddingModel,
        completion_model: BaseCompletionModel,
        base_url: str,
        graph_api_key: str,
    ):
        self.db = db
        self.embedding_model = embedding_model
        self.completion_model = completion_model
        self.base_url = base_url
        self.graph_api_key = graph_api_key

    async def _prepare_document_content(self, doc: Document, document_service) -> str:
        """
        Prepare document content for graph processing.
        If content is empty but storage info exists, parse the document internally.

        Args:
            doc: Document object from morphik database
            document_service: DocumentService instance with storage access

        Returns:
            str: The document's text content (parsed if necessary)
        """
        doc_content = doc.system_metadata.get("content", "") if doc.system_metadata else ""

        # If content is empty but we have storage info, parse the document internally
        if not doc_content.strip() and doc.storage_info:
            try:
                logger.info(f"Document {doc.external_id} content is empty, parsing document internally...")

                # Download the file from storage
                bucket = doc.storage_info.get("bucket")
                key = doc.storage_info.get("key")
                if not bucket or not key:
                    logger.warning(f"Missing storage info for document {doc.external_id}: bucket={bucket}, key={key}")
                    return ""

                file_content = await document_service.storage.download_file(bucket, key)

                # Ensure file_content is bytes
                if hasattr(file_content, "read"):
                    file_content = file_content.read()

                # Parse the file using the document service parser
                additional_metadata, text = await document_service.parser.parse_file_to_text(file_content, doc.filename)

                # Clean the extracted text to remove problematic escape characters
                import re

                text = re.sub(r"[\x00\u0000]", "", text)
                text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)

                if text.strip():
                    # Update the document with the parsed content
                    logger.info(f"Successfully parsed document {doc.external_id}, content length: {len(text)}")

                    # Update system_metadata with the parsed content
                    updated_system_metadata = doc.system_metadata.copy() if doc.system_metadata else {}
                    updated_system_metadata["content"] = text

                    # Create auth context for the update (using minimal permissions needed)
                    from core.models.auth import AuthContext, EntityType

                    auth_context = AuthContext(
                        entity_type=EntityType.DEVELOPER,
                        entity_id="graph_service",
                        app_id=doc.app_id,
                        permissions={"write"},
                        user_id="graph_service",
                    )

                    # Update the document in the database
                    updates = {"system_metadata": updated_system_metadata}
                    await document_service.db.update_document(doc.external_id, updates, auth_context)

                    doc_content = text
                else:
                    logger.warning(f"Failed to extract text content from document {doc.external_id}")

            except Exception as e:
                logger.error(f"Failed to parse document {doc.external_id}: {e}")
                # Return empty content on parsing failure rather than raising
                return ""

        return doc_content

    async def _make_api_request(
        self,
        method: str,
        endpoint: str,
        auth: AuthContext,  # auth is passed for context, actual token extraction TBD
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.graph_api_key}"}

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Set default timeout based on endpoint type
        if timeout is None:
            if "visualization" in endpoint:
                timeout = 1200.0  # 20 minutes for visualization requests
            else:
                timeout = 300.0  # 5 minutes for other requests

        timeout_config = httpx.Timeout(timeout)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.debug(
                        f"Making API request: {method} {url} Data: {json_data} Params: {params} Timeout: {timeout}s (attempt {attempt + 1}/{max_retries})"
                    )
                    response = await client.request(method, url, json=json_data, headers=headers, params=params)
                    response.raise_for_status()  # Raise an exception for HTTP error codes (4xx or 5xx)

                    if response.status_code == 204:  # No Content
                        return None

                    if not response.content:  # Empty body for 200 OK etc.
                        logger.info(f"API request to {url} returned {response.status_code} with empty body.")
                        return {}

                    return response.json()
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    logger.error(f"HTTP error for {method} {url}: {status_code} - {e.response.text}")
                    if status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    raise Exception(f"API request failed: {status_code}, {e.response.text}") from e
                except httpx.RequestError as e:  # Covers connection errors, timeouts, etc.
                    logger.error(f"Request error for {method} {url}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    raise Exception(f"API request failed for {url}") from e
                except ValueError as e:  # JSONDecodeError inherits from ValueError
                    logger.error(
                        f"JSON decoding error for {method} {url}: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}"
                    )
                    raise Exception(f"API response JSON decoding failed for {url}") from e

    async def _find_graph(
        self, graph_name: str, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None
    ) -> Graph:
        # Initialize system_filters if None
        if system_filters is None:
            system_filters = {}

        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        # Get the existing graph with system filters for proper user_id scoping
        existing_graph = await self.db.get_graph(graph_name, auth, system_filters=system_filters)
        if not existing_graph:
            raise ValueError(f"Graph '{graph_name}' not found")

        return existing_graph

    async def get_graph(
        self, graph_name: str, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Graph]:
        """
        Get graph information from the remote service and sync with local database.

        Args:
            graph_name: Name of the graph to fetch
            auth: Authentication context
            system_filters: Optional system filters for graph retrieval

        Returns:
            Graph object with updated metadata from remote service, or None if not found
        """
        try:
            # First get the local graph to get its ID and workflow info
            local_graph = await self.db.get_graph(graph_name, auth, system_filters=system_filters)
            if not local_graph:
                return None

            # Check graph status using the more reliable graph status endpoint
            if local_graph.system_metadata.get("status") == "processing":
                # Check processing timeout first
                processing_started = local_graph.system_metadata.get("processing_started")
                if processing_started:
                    from datetime import datetime

                    try:
                        if isinstance(processing_started, str):
                            processing_started = datetime.fromisoformat(processing_started.replace("Z", "+00:00"))
                        elapsed = datetime.now(processing_started.tzinfo) - processing_started
                        if elapsed.total_seconds() > 3600:  # 1 hour timeout
                            local_graph.system_metadata["status"] = "failed"
                            local_graph.system_metadata["error"] = "Processing timeout exceeded"
                            await self.db.update_graph(local_graph, auth)
                            return local_graph
                    except Exception as e:
                        logger.warning(f"Failed to check processing timeout: {e}")

                # Use the superior graph status endpoint instead of workflow status
                try:
                    graph_status_response = await self._make_api_request(
                        method="GET", endpoint=f"/graph/{local_graph.id}/status", auth=auth
                    )

                    if graph_status_response:
                        remote_status = graph_status_response.get("status", "processing")
                        local_graph.system_metadata["status"] = remote_status

                        # Copy additional fields from graph status response
                        if "pipeline_stage" in graph_status_response:
                            local_graph.system_metadata["pipeline_stage"] = graph_status_response["pipeline_stage"]
                        if "node_count" in graph_status_response:
                            local_graph.system_metadata["node_count"] = graph_status_response["node_count"]
                        if "edge_count" in graph_status_response:
                            local_graph.system_metadata["edge_count"] = graph_status_response["edge_count"]
                        if "error" in graph_status_response:
                            local_graph.system_metadata["error"] = graph_status_response["error"]
                        if "message" in graph_status_response:
                            local_graph.system_metadata["message"] = graph_status_response["message"]

                        # Update the database with new status
                        if remote_status in ["completed", "failed"]:
                            await self.db.update_graph(local_graph, auth)

                except Exception as e:
                    logger.warning(f"Failed to check graph status via graph endpoint: {e}")
                    # Fallback to workflow status check if graph endpoint fails
                    workflow_id = local_graph.system_metadata.get("workflow_id")
                    run_id = local_graph.system_metadata.get("run_id")

                    if workflow_id:
                        status_response = await self.check_workflow_status(workflow_id, run_id, auth)
                        if status_response:
                            remote_status = status_response.get("status", "processing")
                            local_graph.system_metadata["status"] = remote_status

                            if "error" in status_response:
                                local_graph.system_metadata["error"] = status_response["error"]
                            if "pipeline_stage" in status_response:
                                local_graph.system_metadata["pipeline_stage"] = status_response["pipeline_stage"]

            return local_graph

        except Exception as e:
            logger.error(f"Error in get_graph: {e}")
            return None

    async def _make_graph_object(
        self,
        name: str,
        auth: AuthContext,
        document_service,  # Passed in to avoid circular import
        filters: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        # Initialize system_filters if None
        if system_filters is None:
            system_filters = {}

        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        # Find documents to process based on filters and/or specific document IDs
        document_ids = set(documents or [])

        # If filters were provided, get matching documents
        if filters or system_filters:
            filtered_docs = await self.db.get_documents(auth, filters=filters, system_filters=system_filters)
            document_ids.update(doc.external_id for doc in filtered_docs)

        if not document_ids:
            raise ValueError("No documents found matching criteria")

        # Convert system_filters for document retrieval
        folder_name = system_filters.get("folder_name") if system_filters else None
        end_user_id = system_filters.get("end_user_id") if system_filters else None

        # Batch retrieve documents for authorization check
        document_objects = await document_service.batch_retrieve_documents(
            list(document_ids),
            auth,
            folder_name=folder_name,
            folder_depth=None,
            end_user_id=end_user_id,
        )

        # Log for debugging
        logger.info(f"Graph creation with folder_name={folder_name}, end_user_id={end_user_id}")
        logger.info(f"Documents retrieved: {len(document_objects)} out of {len(document_ids)} requested")
        if not document_objects:
            raise ValueError("No authorized documents found matching criteria")

        # Validation is now handled by type annotations

        # Create a new graph
        graph = Graph(name=name, document_ids=[doc.external_id for doc in document_objects], filters=filters)

        return graph

    async def _get_new_document_ids(
        self,
        auth: AuthContext,
        existing_graph: Graph,
        additional_filters: Optional[Dict[str, Any]] = None,
        additional_documents: Optional[List[str]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> Set[str]:
        """Get IDs of new documents to add to the graph."""
        # Initialize system_filters if None
        if system_filters is None:
            system_filters = {}
        # Initialize with explicitly specified documents, ensuring it's a set
        document_ids = set(additional_documents or [])

        # Process documents matching additional filters
        if additional_filters or system_filters:
            filtered_docs = await self.db.get_documents(auth, filters=additional_filters, system_filters=system_filters)
            filter_doc_ids = {doc.external_id for doc in filtered_docs}
            logger.info(f"Found {len(filter_doc_ids)} documents matching additional filters and system filters")
            document_ids.update(filter_doc_ids)

        # Process documents matching the original filters
        if existing_graph.filters:
            # Original filters shouldn't include system filters, as we're applying them separately
            filtered_docs = await self.db.get_documents(
                auth, filters=existing_graph.filters, system_filters=system_filters
            )
            orig_filter_doc_ids = {doc.external_id for doc in filtered_docs}
            logger.info(f"Found {len(orig_filter_doc_ids)} documents matching original filters and system filters")
            document_ids.update(orig_filter_doc_ids)

        # Get only the document IDs that are not already in the graph
        new_doc_ids = document_ids - set(existing_graph.document_ids)
        logger.info(f"Found {len(new_doc_ids)} new documents to add to graph '{existing_graph.name}'")
        return new_doc_ids

    async def create_graph(
        self,
        name: str,
        auth: AuthContext,
        document_service,  # Passed in to avoid circular import
        filters: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
        prompt_overrides: Optional[GraphPromptOverrides] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        """Create a graph from documents.

        This function processes documents matching filters or specific document IDs,
        extracts entities and relationships from document chunks, and saves them as a graph.
        It now also calls an external service to build the graph representation.

        Args:
            name: Name of the graph to create
            auth: Authentication context
            document_service: DocumentService instance for retrieving documents and chunks
            filters: Optional metadata filters to determine which documents to include
            documents: Optional list of specific document IDs to include
            prompt_overrides: Optional GraphPromptOverrides with customizations for prompts
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)
            to determine which documents to include

        Returns:
            Graph: The created graph
        """
        graph = await self._make_graph_object(name, auth, document_service, filters, documents, system_filters)
        docs = await self.db.get_documents_by_id(graph.document_ids, auth, system_filters)

        # Process documents individually instead of concatenating
        if not docs:
            logger.warning(f"No documents found for graph {graph.id}")
            graph.system_metadata["status"] = "completed"
        else:
            # Create empty graph first, then add documents one by one
            try:
                # Initialize empty graph
                initial_request_data = {"graph_id": graph.id, "text": ""}
                api_response = await self._make_api_request(
                    method="POST",
                    endpoint="/build",
                    auth=auth,
                    json_data=initial_request_data,
                )
                logger.info(f"Initial graph build API call for graph_id {graph.id} successful")

                # Process each document individually using /update endpoint
                successful_docs = 0
                failed_docs = 0

                for doc in docs:
                    doc_content = await self._prepare_document_content(doc, document_service)

                    if not doc_content.strip():
                        continue

                    try:
                        request_data = {"graph_id": graph.id, "text": doc_content}
                        doc_response = await self._make_api_request(
                            method="POST",
                            endpoint="/update",
                            auth=auth,
                            json_data=request_data,
                        )
                        logger.debug(f"Document {doc.external_id} response: {doc_response}")
                        successful_docs += 1
                        logger.debug(f"Successfully processed document {doc.external_id} for graph {graph.id}")
                    except Exception as doc_e:
                        failed_docs += 1
                        logger.error(f"Failed to process document {doc.external_id} for graph {graph.id}: {doc_e}")

                logger.info(
                    f"Graph {graph.id}: processed {successful_docs} documents successfully, {failed_docs} failed"
                )

                # Check if the initial response contains workflow_id and run_id (async API)
                if isinstance(api_response, dict) and "workflow_id" in api_response and "run_id" in api_response:
                    logger.info(
                        f"Graph build is async. workflow_id: {api_response['workflow_id']}, run_id: {api_response['run_id']}"
                    )
                    from datetime import datetime, timezone

                    graph.system_metadata["status"] = "processing"
                    graph.system_metadata["workflow_id"] = api_response["workflow_id"]
                    graph.system_metadata["run_id"] = api_response["run_id"]
                    graph.system_metadata["processing_started"] = datetime.now(timezone.utc).isoformat()
                else:
                    # Legacy synchronous response - mark as completed
                    graph.system_metadata["status"] = "completed"

            except Exception as e:
                logger.error(f"Failed to call graph build API for graph_id {graph.id}: {e}")
                graph.system_metadata["status"] = "build_api_failed"
                # Attempt to store graph with failed status before re-raising
                try:
                    await self.db.store_graph(graph, auth)
                except Exception as db_exc:
                    logger.error(f"Failed to store graph {graph.id} with build_api_failed status: {db_exc}")
                raise

        if not await self.db.store_graph(graph, auth):
            # This case might be redundant if the above block handles storing on failure/success appropriately
            # For now, ensure it's stored after successful API call.
            raise Exception("Failed to store graph in the database after API build call")
        return graph

    async def update_graph(
        self,
        name: str,
        auth: AuthContext,
        document_service,  # Passed in to avoid circular import
        additional_filters: Optional[Dict[str, Any]] = None,
        additional_documents: Optional[List[str]] = None,
        prompt_overrides: Optional[GraphPromptOverrides] = None,
        system_filters: Optional[Dict[str, Any]] = None,
        is_initial_build: bool = False,
    ) -> Graph:
        """Update an existing graph with new documents.

        This function processes additional documents, calls an external service to update the graph,
        and updates the graph metadata.

        Args:
            name: Name of the graph to update
            auth: Authentication context
            document_service: DocumentService instance for retrieving documents and chunks
            additional_filters: Optional additional metadata filters
            additional_documents: Optional list of specific additional document IDs
            prompt_overrides: Optional GraphPromptOverrides
            system_filters: Optional system metadata filters
            is_initial_build: Whether this is the initial build of the graph

        Returns:
            Graph: The updated graph
        """
        graph = await self._find_graph(name, auth, system_filters)
        new_doc_ids_set = await self._get_new_document_ids(
            auth, graph, additional_filters, additional_documents, system_filters
        )

        if not new_doc_ids_set:
            logger.info(f"No new documents to add to graph '{name}'. Marking as completed.")
            graph.system_metadata["status"] = "completed"  # Or perhaps "unchanged"
        else:
            new_docs = await self.db.get_documents_by_id(list(new_doc_ids_set), auth, system_filters)

            # Process new documents individually instead of concatenating
            successful_docs = 0
            failed_docs = 0

            try:
                for doc in new_docs:
                    doc_content = await self._prepare_document_content(doc, document_service)

                    if not doc_content.strip():
                        continue

                    try:
                        request_data = {"graph_id": graph.id, "text": doc_content}
                        await self._make_api_request(
                            method="POST",
                            endpoint="/update",
                            auth=auth,
                            json_data=request_data,
                        )
                        successful_docs += 1
                        logger.debug(f"Successfully updated graph {graph.id} with document {doc.external_id}")
                    except Exception as doc_e:
                        failed_docs += 1
                        logger.error(f"Failed to update graph {graph.id} with document {doc.external_id}: {doc_e}")

                logger.info(
                    f"Graph update {graph.id}: processed {successful_docs} documents successfully, {failed_docs} failed"
                )

                # Keep as processing; polling in get_graph will mark completed when nodes/links exist
                from datetime import datetime, timezone

                graph.system_metadata["status"] = "processing"
                # Update processing timestamp for timeout tracking
                graph.system_metadata["processing_started"] = datetime.now(timezone.utc).isoformat()

                # Update local graph object with new document IDs
                current_doc_ids = set(graph.document_ids)
                current_doc_ids.update(new_doc_ids_set)
                graph.document_ids = list(current_doc_ids)

            except Exception as e:
                logger.error(f"Failed to call graph update API for graph_id {graph.id}: {e}")
                graph.system_metadata["status"] = "update_api_failed"
                # Attempt to update graph with failed status before re-raising
                try:
                    await self.db.update_graph(graph)
                except Exception as db_exc:
                    logger.error(f"Failed to update graph {graph.id} with update_api_failed status: {db_exc}")
                raise

        if not await self.db.update_graph(graph):
            raise Exception("Failed to update graph in the database")
        return graph

    async def retrieve(
        self,
        query: str,
        graph_name: str,
        auth: AuthContext,
        document_service,  # Passed to avoid circular import
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        graph = await self._find_graph(graph_name, auth, system_filters)
        # _find_graph raises ValueError if not found, so graph object is guaranteed here.

        graph_id = graph.id

        request_data = {"graph_id": graph_id, "question": query}
        try:
            api_response = await self._make_api_request(
                method="POST",
                endpoint="/retrieval",
                auth=auth,
                json_data=request_data,
            )
            logger.info(f"Retrieval API call for graph_id {graph_id} with query '{query}' successful.")

            if isinstance(api_response, dict):
                if "result" in api_response and isinstance(api_response["result"], str):
                    return api_response["result"]
                if "data" in api_response and isinstance(api_response["data"], str):  # Check common alternative
                    return api_response["data"]
                if not api_response:  # Empty dict {} as per spec for 200 OK
                    logger.warning(
                        f"Retrieval API for graph_id {graph_id} returned an empty JSON object. Returning empty string."
                    )
                    return ""
                # If dict is not empty but doesn't have known fields
                logger.warning(
                    f"Retrieval API for graph_id {graph_id} returned a dictionary with unexpected structure: {api_response}. Returning string representation."
                )
                return str(api_response)
            elif api_response is None:  # From 204 No Content or empty body handled by helper
                logger.warning(f"Retrieval API for graph_id {graph_id} returned no content. Returning empty string.")
                return ""
            else:  # Fallback for other non-dict, non-None types (e.g. if API returns a raw string unexpectedly)
                logger.warning(
                    f"Retrieval API for graph_id {graph_id} returned unexpected type: {type(api_response)}. Value: {api_response}. Returning string representation."
                )
                return str(api_response)

        except Exception as e:
            # Log the original exception, which now includes more details from _make_api_request
            logger.error(f"Failed to call retrieval API for graph_id {graph_id} with query '{query}'. Error: {e}")
            # Depending on requirements, either re-raise or return an error message / empty string
            raise  # Re-raise the exception to be handled by the caller

    async def get_graph_visualization_data(
        self,
        graph_name: str,
        auth: AuthContext,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get graph visualization data from the external graph API.

        Args:
            graph_name: Name of the graph to visualize
            auth: Authentication context
            system_filters: Optional system filters for graph retrieval

        Returns:
            Dict containing nodes and links for visualization
        """
        graph = await self._find_graph(graph_name, auth, system_filters)
        graph_id = graph.id

        request_data = {"graph_id": graph_id}
        try:
            logger.info(
                f"Requesting visualization data for graph_id {graph_id} (may take up to 2 minutes for large graphs)"
            )
            api_response = await self._make_api_request(
                method="POST",
                endpoint="/visualization",
                auth=auth,
                json_data=request_data,
            )
            logger.info(f"Visualization API call for graph_id {graph_id} successful.")

            # The API should return a structure like:
            # {
            #   "nodes": [{"id": "...", "label": "...", "type": "...", "properties": {...}}, ...],
            #   "links": [{"source": "...", "target": "...", "type": "..."}, ...]
            # }

            if isinstance(api_response, dict):
                # Ensure we have the expected structure
                nodes = api_response.get("nodes", [])
                links = api_response.get("links", [])

                # Soft retry once if empty
                if not nodes and not links:
                    await asyncio.sleep(1.5)
                    api_response = await self._make_api_request(
                        method="POST",
                        endpoint="/visualization",
                        auth=auth,
                        json_data=request_data,
                    )
                    nodes = (api_response or {}).get("nodes", [])
                    links = (api_response or {}).get("links", [])

                # Transform to match the expected format for the UI
                formatted_nodes = []
                for node in nodes:
                    formatted_nodes.append(
                        {
                            "id": node.get("id", ""),
                            "label": node.get("label", ""),
                            "type": node.get("type", "unknown"),
                            "properties": node.get("properties", {}),
                            "color": self._get_node_color(node.get("type", "unknown")),
                        }
                    )

                formatted_links = []
                for link in links:
                    formatted_links.append(
                        {
                            "source": link.get("source", ""),
                            "target": link.get("target", ""),
                            "type": link.get("type", ""),
                        }
                    )

                return {"nodes": formatted_nodes, "links": formatted_links}
            else:
                logger.warning(f"Unexpected response format from visualization API: {type(api_response)}")
                return {"nodes": [], "links": []}

        except Exception as e:
            logger.error(f"Failed to call visualization API for graph_id {graph_id}: {e}")
            # Check if this is a timeout error specifically
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.warning(
                    f"Visualization API timed out for graph_id {graph_id}. The graph may be large and still processing."
                )
                # For timeout errors, we could either:
                # 1. Return empty data with a warning (current behavior)
                # 2. Raise the exception to let the UI handle it
                # For now, keeping the existing behavior but adding better logging
            # Return empty visualization data on error
            return {"nodes": [], "links": []}

    def _get_node_color(self, node_type: str) -> str:
        """Get color for a node type to match the UI color scheme."""
        color_map = {
            "person": "#4f46e5",  # Indigo
            "organization": "#06b6d4",  # Cyan
            "location": "#10b981",  # Emerald
            "date": "#f59e0b",  # Amber
            "concept": "#8b5cf6",  # Violet
            "event": "#ec4899",  # Pink
            "product": "#ef4444",  # Red
            "entity": "#4f46e5",  # Indigo (for generic entities)
            "attribute": "#f59e0b",  # Amber
            "relationship": "#ec4899",  # Pink
            "high_level_element": "#10b981",  # Emerald
            "semantic_unit": "#8b5cf6",  # Violet
        }
        return color_map.get(node_type.lower(), "#6b7280")  # Gray as default

    async def query_with_graph(
        self,
        query: str,
        graph_name: str,
        auth: AuthContext,
        document_service,  # core.services.document_service.DocumentService
        filters: Optional[Dict[str, Any]] = None,
        k: int = 20,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_reranking: Optional[bool] = None,  # For document_service.retrieve_chunks
        use_colpali: Optional[bool] = None,  # For document_service.retrieve_chunks
        prompt_overrides: Optional[QueryPromptOverrides] = None,
        system_filters: Optional[Dict[str, Any]] = None,  # For graph retrieval in self.retrieve
        folder_name: Optional[Union[str, List[str]]] = None,  # For document_service and CompletionRequest
        end_user_id: Optional[str] = None,  # For document_service and CompletionRequest
        hop_depth: Optional[int] = None,  # maintain signature
        include_paths: Optional[bool] = None,  # maintain signature
        stream_response: Optional[bool] = False,  # Add stream_response parameter
    ) -> Union[CompletionResponse, tuple[AsyncGenerator[str, None], List[ChunkSource]]]:
        """Generate completion using combined context from an external graph API and standard document retrieval.

        1. Retrieves a context string from the external graph API via /retrieval.
        2. Retrieves standard document chunks via document_service.
        3. Combines these contexts.
        4. Generates a completion using the combined context.
        Args:
            query: The query text.
            graph_name: Name of the graph for external API retrieval.
            auth: Authentication context.
            document_service: Service for standard document/chunk retrieval.
            filters: Metadata filters for standard chunk retrieval.
            k: Number of standard chunks to retrieve.
            min_score: Minimum similarity score for standard chunks.
            max_tokens: Maximum tokens for the completion.
            temperature: Temperature for the completion.
            use_reranking: Whether to use reranking for standard chunks.
            use_colpali: Whether to use colpali embedding for standard chunks.
            prompt_overrides: Customizations for prompts.
            system_filters: System filters for retrieving the graph for external API.
            folder_name: Folder name for scoping standard retrieval and completion.
            end_user_id: End user ID for scoping standard retrieval and completion.
            stream_response: Whether to return a streaming response.

        Returns:
            CompletionResponse or tuple of (AsyncGenerator, List[ChunkSource]) for streaming.
        """
        graph_api_context_str = ""
        try:
            # This call uses the /retrieval endpoint of the external graph API
            graph_api_context_str = await self.retrieve(
                query=query,
                graph_name=graph_name,
                auth=auth,
                document_service=document_service,  # Passed through as per existing pattern
                system_filters=system_filters,
            )
            logger.info(f"Retrieved context from graph API for '{graph_name}': '{graph_api_context_str[:100]}...'")
        except ValueError as e:  # From _find_graph if graph not found
            logger.warning(
                f"Graph '{graph_name}' not found for API retrieval: {e}. Proceeding with standard retrieval only."
            )
        except Exception as e:
            logger.error(
                f"Error retrieving context from graph API for '{graph_name}': {e}. Proceeding with standard retrieval only."
            )

        # Retrieve standard chunks from document_service
        standard_chunks_results: List[ChunkResult] = []
        chunk_contents_list: List[str] = []

        try:
            standard_chunks_results = await document_service.retrieve_chunks(
                query=query,
                auth=auth,
                filters=filters,
                k=k,
                min_score=min_score,
                use_reranking=use_reranking,
                use_colpali=use_colpali,
                folder_name=folder_name,
                folder_depth=None,
                end_user_id=end_user_id,
            )
            logger.info(f"Document service retrieved {len(standard_chunks_results)} standard chunks.")

            if standard_chunks_results:
                # Attempt to get augmented content, similar to GraphService
                try:
                    docs_for_augmentation = await document_service._create_document_results(
                        auth, standard_chunks_results
                    )
                    chunk_contents_list = [
                        chunk.augmented_content(docs_for_augmentation[chunk.document_id])
                        for chunk in standard_chunks_results
                        if chunk.document_id in docs_for_augmentation and hasattr(chunk, "augmented_content")
                    ]
                    if (
                        not chunk_contents_list and standard_chunks_results
                    ):  # Fallback if augmented_content wasn't available/successful
                        logger.info(
                            "Falling back to raw chunk content as augmentation was not fully successful or 'augmented_content' is missing."
                        )
                        chunk_contents_list = [
                            chunk.content for chunk in standard_chunks_results if hasattr(chunk, "content")
                        ]

                except AttributeError as ae:
                    logger.warning(
                        f"DocumentService might be missing _create_document_results or ChunkResult missing augmented_content. Falling back to raw content. Error: {ae}"
                    )
                    chunk_contents_list = [
                        chunk.content for chunk in standard_chunks_results if hasattr(chunk, "content")
                    ]
        except Exception as e:
            logger.error(f"Error during standard chunk retrieval or processing: {e}")

        # Combine contexts
        final_context_list: List[str] = []
        if graph_api_context_str and graph_api_context_str.strip():  # Ensure non-empty context
            final_context_list.append(graph_api_context_str)
        final_context_list.extend(chunk_contents_list)

        if not final_context_list:
            logger.warning("No context available from graph API or document service. Completion may be inadequate.")
            # Return a response indicating no context was found
            return CompletionResponse(
                text="Unable to find relevant information to answer the query.",
                sources=[],
                error="No context available for query processing.",
            )

        # Generate completion
        custom_prompt_template = None
        custom_system_prompt = None
        if prompt_overrides and prompt_overrides.query:
            if hasattr(prompt_overrides.query, "prompt_template"):
                custom_prompt_template = prompt_overrides.query.prompt_template
            if hasattr(prompt_overrides.query, "system_prompt"):
                custom_system_prompt = prompt_overrides.query.system_prompt

        completion_req = CompletionRequest(
            query=query,
            context_chunks=final_context_list,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_template=custom_prompt_template,
            system_prompt=custom_system_prompt,
            folder_name=folder_name,
            end_user_id=end_user_id,
            stream_response=stream_response,
        )

        try:
            response = await self.completion_model.complete(completion_req)
        except Exception as e:
            logger.error(f"Error during completion generation: {e}")
            if stream_response:
                # Return empty stream and sources for error case
                async def empty_stream():
                    yield ""

                return (empty_stream(), [])
            else:
                return CompletionResponse(text="", error=f"Failed to generate completion: {e}")

        # Prepare sources from standard chunks
        response_sources = [
            ChunkSource(
                document_id=chunk.document_id,
                chunk_number=chunk.chunk_number,
                score=getattr(chunk, "score", 0.0),  # Default score to 0.0 if not present
            )
            for chunk in standard_chunks_results
        ]

        # Handle streaming vs non-streaming responses
        if stream_response:
            # For streaming, response should be an async generator
            return (response, response_sources)
        else:
            # Add sources information from the standard_chunks_results
            if hasattr(response, "sources") and response.sources is None:
                response.sources = []  # Ensure sources is a list if None

            # If response already has sources, this will overwrite. If it should append, logic needs change.
            response.sources = response_sources

            # Add metadata about retrieval
            if not hasattr(response, "metadata") or response.metadata is None:
                response.metadata = {}

            response.metadata["retrieval_info"] = {
                "graph_api_context_used": bool(graph_api_context_str and graph_api_context_str.strip()),
                "standard_chunks_retrieved": len(standard_chunks_results),
            }

            return response

    async def check_workflow_status(
        self,
        workflow_id: str,
        run_id: Optional[str] = None,
        auth: Optional[AuthContext] = None,
    ) -> Dict[str, Any]:
        """Check the status of a workflow from the graph API.

        Args:
            workflow_id: The workflow ID to check
            run_id: Optional run ID for the specific workflow run
            auth: Authentication context

        Returns:
            Dict containing status and optional result
        """
        try:
            # Build query params
            params = {}
            if run_id:
                params["run_id"] = run_id

            api_response = await self._make_api_request(
                method="GET",
                endpoint=f"/status/{workflow_id}",
                auth=auth,
                json_data=None,  # GET request, no body
                params=params,
            )

            logger.info(f"Workflow status check for {workflow_id} successful. Response: {api_response}")
            return api_response

        except Exception as e:
            logger.error(f"Failed to check workflow status for {workflow_id}: {e}")
            # Return failed status instead of raising
            return {"status": "failed", "error": str(e)}

    async def delete_graph(
        self,
        graph_name: str,
        auth: AuthContext,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Delete a graph and its associated data from the external graph API.

        Args:
            graph_name: Name of the graph to delete
            auth: Authentication context
            system_filters: Optional system metadata filters

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Find the graph to get its ID
            graph = await self._find_graph(graph_name, auth, system_filters)
            graph_id = graph.id

            # Call the external graph delete service
            api_response = await self._make_api_request(
                method="DELETE",
                endpoint=f"/delete/{graph_id}",
                auth=auth,
            )
            logger.info(f"Graph delete API call for graph_id {graph_id} successful. Response: {api_response}")

            # Delete the graph from our database
            success = await self.db.delete_graph(graph_name, auth)
            if not success:
                logger.error(f"Failed to delete graph '{graph_name}' from database after successful API deletion")
                return False

            return True

        except ValueError as e:
            logger.error(f"Graph '{graph_name}' not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to delete graph '{graph_name}': {e}")
            raise

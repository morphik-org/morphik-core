import logging
from typing import Any, Dict, List, Optional, Set, Union

import httpx

from core.completion.base_completion import BaseCompletionModel
from core.database.base_database import BaseDatabase
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.auth import AuthContext
from core.models.completion import ChunkSource, CompletionRequest, CompletionResponse
from core.models.documents import ChunkResult
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

    async def _make_api_request(
        self,
        method: str,
        endpoint: str,
        auth: AuthContext,  # auth is passed for context, actual token extraction TBD
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.graph_api_key}"}

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient() as client:
            try:
                logger.debug(f"Making API request: {method} {url} Data: {json_data}")
                response = await client.request(method, url, json=json_data, headers=headers)
                response.raise_for_status()  # Raise an exception for HTTP error codes (4xx or 5xx)

                if response.status_code == 204:  # No Content
                    return None

                if not response.content:  # Empty body for 200 OK etc.
                    logger.info(f"API request to {url} returned {response.status_code} with empty body.")
                    return {}

                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error for {method} {url}: {e.response.status_code} - {e.response.text}")
                raise Exception(f"API request failed: {e.response.status_code}, {e.response.text}") from e
            except httpx.RequestError as e:  # Covers connection errors, timeouts, etc.
                logger.error(f"Request error for {method} {url}: {e}")
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
            list(document_ids), auth, folder_name, end_user_id
        )

        # Log for debugging
        logger.info(f"Graph creation with folder_name={folder_name}, end_user_id={end_user_id}")
        logger.info(f"Documents retrieved: {len(document_objects)} out of {len(document_ids)} requested")
        if not document_objects:
            raise ValueError("No authorized documents found matching criteria")

        # Validation is now handled by type annotations

        # Create a new graph with authorization info
        access_control = {
            "readers": [auth.entity_id],
            "writers": [auth.entity_id],
            "admins": [auth.entity_id],
        }

        # Add user_id to access_control if present (for proper user_id scoping)
        if auth.user_id:
            # User ID must be provided as a list to match the Graph model's type constraints
            access_control["user_id"] = [auth.user_id]

        # Ensure entity_type is a string value for storage
        entity_type = auth.entity_type.value if hasattr(auth.entity_type, "value") else auth.entity_type

        graph = Graph(
            name=name,
            document_ids=[doc.external_id for doc in document_objects],
            filters=filters,
            owner={"type": entity_type, "id": auth.entity_id},
            access_control=access_control,
        )

        return graph

    async def get_graph_visualization(self, graph_name: str, auth: AuthContext) -> Dict[str, Any]:
        """
        Retrieves the visualization data (nodes and links) for a given graph.

        Args:
            graph_name: The name of the graph.
            auth: The authentication context.

        Returns:
            A dictionary containing 'nodes' and 'links' for the graph visualization,
            or raises an exception if the API call fails.
        """
        logger.debug(f"Attempting to find graph '{graph_name}' for visualization.")
        graph = await self._find_graph(graph_name, auth, system_filters=None)
        # _find_graph raises ValueError if graph not found or PermissionError if no access

        logger.debug(f"Found graph with ID '{graph.id}'. Preparing API request for visualization.")
        request_data = {"graph_id": graph.id}

        try:
            visualization_data = await self._make_api_request(
                method="POST",
                endpoint="/graph",  # Target endpoint for graph visualization
                auth=auth,
                json_data=request_data,
            )
            logger.info(f"Successfully retrieved visualization data for graph ID '{graph.id}'.")
            # Basic validation of the response structure
            if not isinstance(visualization_data, dict) or "nodes" not in visualization_data or "links" not in visualization_data:
                logger.warning(f"Visualization data for graph ID '{graph.id}' has unexpected format: {visualization_data}")
                # Depending on requirements, one might raise an error here or return the data as is.
                # For now, returning as is, as the external API's contract is what defines the output.
            return visualization_data
        except Exception as e:
            # _make_api_request already logs details of HTTPStatusError, RequestError, JSONDecodeError
            logger.error(f"API call for graph visualization failed for graph ID '{graph.id}'. Error: {e}")
            # Re-raise the exception to be handled by the calling layer (e.g., API endpoint)
            raise

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
        content = "\n".join([doc.system_metadata.get("content", "") for doc in docs if doc.system_metadata])

        # Call the external graph building service
        request_data = {"graph_id": graph.id, "text": content}
        try:
            api_response = await self._make_api_request(
                method="POST",
                endpoint="/build",
                auth=auth,
                json_data=request_data,
            )
            logger.info(f"Graph build API call for graph_id {graph.id} successful. Response: {api_response}")
            graph.system_metadata["status"] = "completed"
        except Exception as e:
            logger.error(f"Failed to call graph build API for graph_id {graph.id}: {e}")
            graph.system_metadata["status"] = "build_api_failed"
            # Attempt to store graph with failed status before re-raising
            try:
                await self.db.store_graph(graph)
            except Exception as db_exc:
                logger.error(f"Failed to store graph {graph.id} with build_api_failed status: {db_exc}")
            raise

        if not await self.db.store_graph(graph):
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
            new_content = "\n".join([doc.system_metadata.get("content", "") for doc in new_docs if doc.system_metadata])

            request_data = {"graph_id": graph.id, "text": new_content}
            try:
                api_response = await self._make_api_request(
                    method="POST",
                    endpoint="/update",
                    auth=auth,
                    json_data=request_data,
                )
                logger.info(f"Graph update API call for graph_id {graph.id} successful. Response: {api_response}")
                # Update local graph object with new document IDs
                current_doc_ids = set(graph.document_ids)
                current_doc_ids.update(new_doc_ids_set)
                graph.document_ids = list(current_doc_ids)
                graph.system_metadata["status"] = "completed"
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
    ) -> CompletionResponse:
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

        Returns:
            CompletionResponse: The generated completion response.
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
                query, auth, filters, k, min_score, use_reranking, use_colpali, folder_name, end_user_id
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
        if prompt_overrides and prompt_overrides.query and hasattr(prompt_overrides.query, "prompt_template"):
            custom_prompt_template = prompt_overrides.query.prompt_template

        completion_req = CompletionRequest(
            query=query,
            context_chunks=final_context_list,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_template=custom_prompt_template,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )

        try:
            response = await self.completion_model.complete(completion_req)
        except Exception as e:
            logger.error(f"Error during completion generation: {e}")
            return CompletionResponse(text="", error=f"Failed to generate completion: {e}")

        # Add sources information from the standard_chunks_results
        if hasattr(response, "sources") and response.sources is None:
            response.sources = []  # Ensure sources is a list if None

        response_sources = [
            ChunkSource(
                document_id=chunk.document_id,
                chunk_number=chunk.chunk_number,
                score=getattr(chunk, "score", 0.0),  # Default score to 0.0 if not present
            )
            for chunk in standard_chunks_results
        ]
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

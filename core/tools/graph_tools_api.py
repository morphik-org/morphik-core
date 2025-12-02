import json
import logging
from typing import Dict, Optional

from core.models.auth import AuthContext
from core.services.document_service import DocumentService
from core.tools.document_tools import ToolError

logger = logging.getLogger(__name__)


async def graph_api_retrieve(
    query: str,
    graph_name: Optional[str] = None,
    document_service: Optional[DocumentService] = None,
    auth: Optional[AuthContext] = None,
    end_user_id: Optional[str] = None,
) -> str:
    """Retrieve information from a remote (API-backed) Morphik knowledge graph.

    Args:
        query: Natural-language query or question.
        graph_name: Optional name of the graph to query. If omitted, we try to
            infer a sensible default ("graph_main" or the user's only graph).
        document_service: The active DocumentService instance.
        auth: Authentication context forwarded from the agent.
        end_user_id: Optional end-user ID for scoping.

    Returns:
        A string response returned by the Morphik graph API.
    """

    # Sanity-check prerequisites
    if document_service is None:
        raise ToolError("Document service not provided")

    if auth is None:
        raise ToolError("Authentication context not provided")

    graph_service = getattr(document_service, "graph_service", None)
    if graph_service is None:
        raise ToolError("Graph service not initialised on document service")

    # Ensure we are talking to the API-backed implementation.
    from core.services.morphik_graph_service import MorphikGraphService

    if not isinstance(graph_service, MorphikGraphService):
        raise ToolError("graph_api_retrieve can only be used when GRAPH_MODE is 'api' (MorphikGraphService)")

    # Infer graph_name if not provided
    if not graph_name:
        system_filters: Dict[str, str] = {}
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Ask the DB for available graphs visible to the user
        available_graphs = await graph_service.db.list_graphs(auth, system_filters=system_filters)
        if not available_graphs:
            raise ToolError("No graphs found. Please create a graph first.")

        if len(available_graphs) == 1:
            graph_name = available_graphs[0].name
            logger.info(f"Using the only available graph: {graph_name}")
        else:
            main_graph = next((g for g in available_graphs if g.name == "graph_main"), None)
            graph_name = main_graph.name if main_graph else available_graphs[0].name
            logger.info(f"Multiple graphs available, using {graph_name}")

    # Build system_filters for retrieve call
    system_filters = {"end_user_id": end_user_id} if end_user_id else None

    try:
        result = await graph_service.retrieve(
            query=query,
            graph_name=graph_name,
            auth=auth,
            document_service=document_service,
            system_filters=system_filters,
        )
        # If the API returned a dict, serialise to JSON for consistency
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return result
    except Exception as e:
        logger.error(f"Error retrieving from graph '{graph_name}': {e}")
        raise ToolError(f"Error retrieving from graph '{graph_name}': {e}")

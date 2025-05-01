"""Knowledge graph query and management tools."""

import json
import logging
from typing import List, Literal, Optional

from core.models.auth import AuthContext
from core.models.graph import Entity
from core.services.document_service import DocumentService
from core.tools.document_tools import ToolError

logger = logging.getLogger(__name__)


async def knowledge_graph_query(
    query_type: Literal["list_entities", "entity", "path", "subgraph"],
    start_nodes: List[str],
    max_depth: int = 3,
    graph_name: Optional[str] = None,
    document_service: DocumentService = None,
    auth: AuthContext = None,
    end_user_id: Optional[str] = None,
) -> str:
    """
    Query the knowledge graph for entities, relationships, and connections.

    Args:
        query_type: Type of knowledge graph query
        start_nodes: Starting entity/entities for the query
        max_depth: Maximum path length/depth to explore
        graph_name: Name of the graph to query (optional)
        document_service: DocumentService instance
        auth: Authentication context
        end_user_id: Optional end-user ID

    Returns:
        Knowledge graph query results as a string
    """
    if document_service is None:
        raise ToolError("Document service not provided")

    try:
        # Get graph service from document service
        graph_service = document_service.graph_service

        # Set up system filters for proper scoping
        system_filters = {}
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # If no graph name provided, try to find an appropriate graph
        if not graph_name:
            # Get all available graphs for the user
            available_graphs = await graph_service.db.list_graphs(auth, system_filters=system_filters)

            if not available_graphs:
                raise ToolError("No graphs found. Please create a graph first.")

            # If user has only one graph, use that
            if len(available_graphs) == 1:
                graph_name = available_graphs[0].name
                logger.info(f"Using the only available graph: {graph_name}")
            else:
                # Use "graph_main" as default if available
                main_graph = next((g for g in available_graphs if g.name == "graph_main"), None)
                if main_graph:
                    graph_name = "graph_main"
                else:
                    # Otherwise use the first available graph
                    graph_name = available_graphs[0].name

                logger.info(f"Multiple graphs available, using {graph_name}")

        # Get the graph for the authorized user
        graph = await graph_service.db.get_graph(graph_name, auth, system_filters=system_filters)
        if not graph:
            raise ToolError(f"Graph '{graph_name}' not found or not accessible")

        # Create entity map for faster lookups
        entity_map = {entity.id: entity for entity in graph.entities}
        entity_by_label = {entity.label.lower(): entity for entity in graph.entities}

        match query_type:
            case "list_entities":
                entities = await graph_service._find_similar_entities(start_nodes[0], graph.entities, 10)
                results = [
                    {
                        "id": entity.id,
                        "label": entity.label,
                        "type": entity.type,
                        "properties": entity.properties,
                        "similarity_score": score,
                    }
                    for entity, score in entities
                ]
                return json.dumps(results, indent=2)

            case "entity":
                if not start_nodes or len(start_nodes) == 0:
                    raise ToolError("Entity ID or label is required for entity query")

                # Try to find entity by ID first, then by label
                entity = entity_map.get(start_nodes[0])

                # If not found by ID, try looking up by label (case-insensitive)
                if not entity:
                    entity = entity_by_label.get(start_nodes[0].lower())

                if not entity:
                    raise ToolError(f"Entity '{start_nodes[0]}' not found in the knowledge graph")

                # Return entity details
                return json.dumps(
                    {
                        "id": entity.id,
                        "label": entity.label,
                        "type": entity.type,
                        "properties": entity.properties,
                        "document_ids": entity.document_ids,
                    },
                    indent=2,
                )

            case "path":
                if not start_nodes or len(start_nodes) < 2:
                    raise ToolError("Exactly two entity IDs/labels are required for path query")

                # Find source and target entities
                source = entity_map.get(start_nodes[0])
                if not source:
                    source = entity_by_label.get(start_nodes[0].lower())

                target = entity_map.get(start_nodes[1])
                if not target:
                    target = entity_by_label.get(start_nodes[1].lower())

                if not source or not target:
                    missing = []
                    if not source:
                        missing.append(start_nodes[0])
                    if not target:
                        missing.append(start_nodes[1])
                    raise ToolError(f"Entities not found: {', '.join(missing)}")

                # Find all paths between the two entities
                paths = graph_service._find_relationship_paths(graph, [source], max_depth)

                # Filter paths that end with the target entity
                target_paths = []
                for path in paths:
                    if path and path[-1] == target.label:
                        target_paths.append(path)

                if not target_paths:
                    return f"No path found between '{source.label}' and '{target.label}' within {max_depth} hops"

                # Format results as human-readable paths
                formatted_paths = []
                for path in target_paths:
                    formatted_paths.append(" -> ".join(path))

                return json.dumps({"source": source.label, "target": target.label, "paths": formatted_paths}, indent=2)

            case "subgraph":
                if not start_nodes or len(start_nodes) == 0:
                    raise ToolError("Entity ID or label is required for subgraph query")

                # Find central entity
                central = entity_map.get(start_nodes[0])
                if not central:
                    central = entity_by_label.get(start_nodes[0].lower())

                if not central:
                    raise ToolError(f"Entity '{start_nodes[0]}' not found in the knowledge graph")

                # Get related entities through graph traversal
                related_entities = graph_service._expand_entities(graph, [central], max_depth)

                # Create a set of entity IDs in the subgraph
                subgraph_entity_ids = {entity.id for entity in related_entities}

                # Find relationships between these entities
                subgraph_relationships = [
                    rel
                    for rel in graph.relationships
                    if rel.source_id in subgraph_entity_ids and rel.target_id in subgraph_entity_ids
                ]

                # Format results
                nodes = []
                for entity in related_entities:
                    nodes.append(
                        {"id": entity.id, "label": entity.label, "type": entity.type, "properties": entity.properties}
                    )

                relationships = []
                for rel in subgraph_relationships:
                    source_label = entity_map.get(rel.source_id, Entity(label="Unknown")).label
                    target_label = entity_map.get(rel.target_id, Entity(label="Unknown")).label

                    relationships.append(
                        {
                            "id": rel.id,
                            "source": rel.source_id,
                            "source_label": source_label,
                            "target": rel.target_id,
                            "target_label": target_label,
                            "type": rel.type,
                        }
                    )

                return json.dumps({"nodes": nodes, "relationships": relationships}, indent=2)

    except Exception as e:
        raise ToolError(f"Error querying knowledge graph: {str(e)}")


async def list_graphs(
    document_service: DocumentService = None, auth: AuthContext = None, end_user_id: Optional[str] = None
) -> str:
    """
    List all available knowledge graphs for the authorized user.

    Args:
        document_service: DocumentService instance
        auth: Authentication context
        end_user_id: Optional end-user ID for scoping

    Returns:
        List of available graphs as JSON string
    """
    if document_service is None:
        raise ToolError("Document service not provided")

    try:
        # Get graph service from document service
        graph_service = document_service.graph_service

        # Set up system filters for proper scoping
        system_filters = {}
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Get all available graphs for the user
        available_graphs = await graph_service.db.list_graphs(auth, system_filters=system_filters)

        if not available_graphs:
            return json.dumps({"message": "No graphs found", "graphs": []})

        # Format the results
        graphs_info = []
        for graph in available_graphs:
            graphs_info.append(
                {
                    "name": graph.name,
                    "document_count": len(graph.document_ids),
                    "entity_count": len(graph.entities),
                    "relationship_count": len(graph.relationships),
                    "created_at": (
                        graph.created_at.isoformat() if hasattr(graph, "created_at") and graph.created_at else None
                    ),
                    "updated_at": (
                        graph.updated_at.isoformat() if hasattr(graph, "updated_at") and graph.updated_at else None
                    ),
                }
            )

        return json.dumps({"message": f"Found {len(graphs_info)} graph(s)", "graphs": graphs_info}, indent=2)

    except Exception as e:
        raise ToolError(f"Error listing graphs: {str(e)}")

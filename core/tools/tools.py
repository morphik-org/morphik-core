"""Tool implementations for document operations."""

import logging

from core.tools.analysis_tools import document_analyzer, execute_code

# Import tools from specialized modules
from core.tools.document_tools import list_documents, retrieve_chunks, retrieve_document, save_to_memory
from core.tools.graph_tools import knowledge_graph_query, list_graphs
from core.tools.graph_tools_api import graph_api_retrieve

# Set up logging
logger = logging.getLogger(__name__)

# Export all tool functions
__all__ = [
    "retrieve_chunks",
    "retrieve_document",
    "document_analyzer",
    "execute_code",
    "knowledge_graph_query",
    "list_graphs",
    "save_to_memory",
    "list_documents",
    "graph_api_retrieve",
]

# ToolError is a common exception that might be useful here too
__all__.append("ToolError")

# get_timestamp might be useful for consumers, though not directly a tool
__all__.append("get_timestamp")

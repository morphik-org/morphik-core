from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..models.auth import AuthContext
from ..models.documents import Document
from ..models.folders import Folder
from ..models.graph import Graph

if TYPE_CHECKING:
    from core.models.workflows import Workflow, WorkflowRun


class BaseDatabase(ABC):
    """Base interface for document metadata storage."""

    @abstractmethod
    async def store_document(self, document: Document, auth: AuthContext) -> bool:
        """
        Store document metadata.
        Returns: Success status
        """
        pass

    @abstractmethod
    async def get_document(self, document_id: str, auth: AuthContext) -> Optional[Document]:
        """
        Retrieve document metadata by ID if user has access.
        Returns: Document if found and accessible, None otherwise
        """
        pass

    @abstractmethod
    async def get_document_by_filename(
        self, filename: str, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Document]:
        """
        Retrieve document metadata by filename if user has access.
        If multiple documents have the same filename, returns the most recently updated one.

        Args:
            filename: The filename to search for
            auth: Authentication context
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)

        Returns:
            Document if found and accessible, None otherwise
        """
        pass

    @abstractmethod
    async def get_documents_by_id(
        self,
        document_ids: List[str],
        auth: AuthContext,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation.
        Only returns documents the user has access to.
        Can filter by system metadata fields like folder_name and end_user_id.

        Args:
            document_ids: List of document IDs to retrieve
            auth: Authentication context
            system_filters: Optional filters for system metadata fields

        Returns:
            List of Document objects that were found and user has access to
        """
        pass

    @abstractmethod
    async def get_documents(
        self,
        auth: AuthContext,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        List documents the user has access to.
        Supports pagination and filtering.

        Args:
            auth: Authentication context
            skip: Number of documents to skip (for pagination)
            limit: Maximum number of documents to return
            filters: Optional metadata filters
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)

        Returns:
            List of documents matching the criteria
        """
        pass

    @abstractmethod
    async def list_documents_flexible(
        self,
        auth: AuthContext,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
        include_total_count: bool = False,
        include_status_counts: bool = False,
        include_folder_counts: bool = False,
        return_documents: bool = True,
        sort_by: Optional[str] = None,
        sort_direction: str = "desc",
    ) -> Dict[str, Any]:
        """
        List documents with optional aggregates and flexible pagination metadata.

        Returns:
            Dictionary containing documents and requested aggregate information.
        """
        pass

    @abstractmethod
    async def update_document(self, document_id: str, updates: Dict[str, Any], auth: AuthContext) -> bool:
        """
        Update document metadata if user has access.
        Returns: Success status
        """
        pass

    @abstractmethod
    async def delete_document(self, document_id: str, auth: AuthContext) -> bool:
        """
        Delete document metadata if user has admin access.
        Returns: Success status
        """
        pass

    @abstractmethod
    async def find_authorized_and_filtered_documents(
        self,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Find document IDs matching filters that user has access to.

        Args:
            auth: Authentication context
            filters: Optional metadata filters
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)

        Returns:
            List of document IDs matching the criteria
        """
        pass

    @abstractmethod
    async def check_access(self, document_id: str, auth: AuthContext, required_permission: str = "read") -> bool:
        """
        Check if user has required permission for document.
        Returns: True if user has required access, False otherwise
        """
        pass

    @abstractmethod
    async def store_cache_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Store metadata for a cache.

        Args:
            name: Name of the cache
            metadata: Cache metadata including model info and storage location

        Returns:
            bool: Whether the operation was successful
        """
        pass

    @abstractmethod
    async def get_cache_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a cache.

        Args:
            name: Name of the cache

        Returns:
            Optional[Dict[str, Any]]: Cache metadata if found, None otherwise
        """
        pass

    @abstractmethod
    async def store_graph(self, graph: Graph) -> bool:
        """Store a graph.

        Args:
            graph: Graph to store

        Returns:
            bool: Whether the operation was successful
        """
        pass

    @abstractmethod
    async def get_graph(
        self, name: str, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Graph]:
        """Get a graph by name.

        Args:
            name: Name of the graph
            auth: Authentication context
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)

        Returns:
            Optional[Graph]: Graph if found and accessible, None otherwise
        """
        pass

    @abstractmethod
    async def list_graphs(self, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None) -> List[Graph]:
        """List all graphs the user has access to.

        Args:
            auth: Authentication context
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)

        Returns:
            List[Graph]: List of graphs
        """
        pass

    @abstractmethod
    async def update_graph(self, graph: Graph) -> bool:
        """Update an existing graph.

        Args:
            graph: Graph to update

        Returns:
            bool: Whether the operation was successful
        """
        pass

    @abstractmethod
    async def delete_graph(self, name: str, auth: AuthContext) -> bool:
        """Delete a graph by name.

        Args:
            name: Name of the graph to delete
            auth: Authentication context

        Returns:
            bool: Whether the operation was successful
        """
        pass

    @abstractmethod
    async def create_folder(self, folder: Folder) -> bool:
        """Create a new folder.

        Args:
            folder: Folder to create

        Returns:
            bool: Whether the operation was successful
        """
        pass

    @abstractmethod
    async def get_folder(self, folder_id: str, auth: AuthContext) -> Optional[Folder]:
        """Get a folder by ID.

        Args:
            folder_id: ID of the folder
            auth: Authentication context

        Returns:
            Optional[Folder]: Folder if found and accessible, None otherwise
        """
        pass

    @abstractmethod
    async def get_folder_by_name(self, name: str, auth: AuthContext) -> Optional[Folder]:
        """Get a folder by name.

        Args:
            name: Name of the folder
            auth: Authentication context

        Returns:
            Optional[Folder]: Folder if found and accessible, None otherwise
        """
        pass

    @abstractmethod
    async def list_folders(self, auth: AuthContext) -> List[Folder]:
        """List all folders the user has access to.

        Args:
            auth: Authentication context

        Returns:
            List[Folder]: List of folders
        """
        pass

    @abstractmethod
    async def add_document_to_folder(self, folder_id: str, document_id: str, auth: AuthContext) -> bool:
        """Add a document to a folder.

        Args:
            folder_id: ID of the folder
            document_id: ID of the document
            auth: Authentication context

        Returns:
            bool: Whether the operation was successful
        """
        pass

    @abstractmethod
    async def remove_document_from_folder(self, folder_id: str, document_id: str, auth: AuthContext) -> bool:
        """Remove a document from a folder.

        Args:
            folder_id: ID of the folder
            document_id: ID of the document
            auth: Authentication context

        Returns:
            bool: Whether the operation was successful
        """
        pass

    # ------------------------------------------------------------------
    # Workflows API (Step-2)
    # ------------------------------------------------------------------

    @abstractmethod
    async def store_workflow(self, workflow: "Workflow", auth: AuthContext) -> bool:  # noqa: D401 – forward reference
        """Persist a Workflow definition."""

    @abstractmethod
    async def list_workflows(self, auth: AuthContext) -> List["Workflow"]:
        """List workflows visible to *auth*."""

    @abstractmethod
    async def get_workflow(self, workflow_id: str, auth: AuthContext) -> Optional["Workflow"]:
        """Fetch a single workflow if accessible."""

    @abstractmethod
    async def update_workflow(
        self, workflow_id: str, updates: Dict[str, Any], auth: AuthContext
    ) -> Optional["Workflow"]:
        """Apply partial updates and return updated record."""

    @abstractmethod
    async def delete_workflow(self, workflow_id: str, auth: AuthContext) -> bool:
        """Delete a workflow definition."""

    # Workflow runs

    @abstractmethod
    async def store_workflow_run(self, run: "WorkflowRun") -> bool:  # noqa: D401
        """Persist a WorkflowRun record."""

    @abstractmethod
    async def get_workflow_run(self, run_id: str, auth: AuthContext) -> Optional["WorkflowRun"]:
        """Fetch a workflow run if visible."""

    @abstractmethod
    async def list_workflow_runs(self, workflow_id: str, auth: AuthContext) -> List["WorkflowRun"]:
        """List all runs for a specific workflow."""

    @abstractmethod
    async def delete_workflow_run(self, run_id: str, auth: AuthContext) -> bool:
        """Delete a workflow run."""

    @abstractmethod
    async def search_documents_by_name(
        self,
        query: str,
        auth: AuthContext,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search documents by filename using full-text search.

        Args:
            query: Search query for document names/filenames
            auth: Authentication context
            limit: Maximum number of documents to return
            filters: Optional metadata filters
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)

        Returns:
            List of documents matching the search query, ordered by relevance
        """

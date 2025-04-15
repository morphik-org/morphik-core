import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

from core.services.graph_service import GraphService
from core.models.auth import AuthContext, EntityType
from core.models.graph import Graph


class TestUKGHelpers:
    """Tests for UKG helper functions in GraphService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = MagicMock()
        db.get_graph = AsyncMock()
        db.store_graph = AsyncMock()
        db.update_graph = AsyncMock()
        return db

    @pytest.fixture
    def graph_service(self, mock_db):
        """Create a GraphService with mock components."""
        mock_embedding_model = MagicMock()
        mock_completion_model = MagicMock()
        return GraphService(
            db=mock_db,
            embedding_model=mock_embedding_model,
            completion_model=mock_completion_model
        )

    @pytest.fixture
    def developer_auth(self):
        """Create a developer authentication context."""
        return AuthContext(
            entity_type=EntityType.DEVELOPER,
            entity_id="dev_123",
            permissions=["read", "write"],
            user_id="dev_123"
        )

    def test_hash_id(self, graph_service):
        """Test _hash_id generates consistent hashes."""
        # Test with the same input
        id_str = "user_123"
        hash1 = graph_service._hash_id(id_str)
        hash2 = graph_service._hash_id(id_str)
        
        # Same input should produce same hash
        assert hash1 == hash2
        # Hash should be 16 chars long
        assert len(hash1) == 16
        
        # Different inputs should produce different hashes
        hash3 = graph_service._hash_id("user_456")
        assert hash1 != hash3

    def test_generate_ukg_name(self, graph_service):
        """Test _generate_ukg_name creates correct format."""
        dev_id = "dev_123"
        user_id = "user_456"
        
        # Get hashed versions manually to compare
        hashed_dev_id = graph_service._hash_id(dev_id)
        hashed_user_id = graph_service._hash_id(user_id)
        
        # Generate UKG name
        ukg_name = graph_service._generate_ukg_name(dev_id, user_id)
        
        # Verify format
        assert ukg_name == f"ukg_{hashed_dev_id}_{hashed_user_id}"
        
        # Verify consistency
        assert ukg_name == graph_service._generate_ukg_name(dev_id, user_id)

    @pytest.mark.asyncio
    async def test_get_ukg(self, graph_service, developer_auth, mock_db):
        """Test _get_ukg retrieves the correct graph."""
        end_user_id = "user_789"
        expected_graph_name = graph_service._generate_ukg_name(developer_auth.entity_id, end_user_id)
        
        # Set up mock return value
        mock_graph = Graph(name=expected_graph_name, entities=[], relationships=[])
        mock_db.get_graph.return_value = mock_graph
        
        # Call the function
        result = await graph_service._get_ukg(developer_auth, end_user_id)
        
        # Verify correct graph was retrieved
        mock_db.get_graph.assert_called_once_with(expected_graph_name, developer_auth)
        assert result == mock_graph

    @pytest.mark.asyncio
    async def test_store_or_update_ukg_new_graph(self, graph_service, developer_auth, mock_db):
        """Test _store_or_update_ukg stores a new graph correctly."""
        end_user_id = "user_789"
        hashed_user_id = graph_service._hash_id(end_user_id)
        expected_graph_name = graph_service._generate_ukg_name(developer_auth.entity_id, end_user_id)
        
        # Mock that the graph doesn't exist yet
        mock_db.get_graph.return_value = None
        mock_db.store_graph.return_value = True
        
        # Create a graph to store
        graph = Graph(name="temp_name", entities=[], relationships=[])
        
        # Call the function
        result = await graph_service._store_or_update_ukg(developer_auth, end_user_id, graph)
        
        # Verify graph was stored with correct attributes
        assert graph.name == expected_graph_name
        assert graph.owner["id"] == developer_auth.entity_id
        assert graph.owner["type"] == developer_auth.entity_type
        assert graph.metadata["ukg_for_user"] == hashed_user_id
        
        # Verify correct methods were called
        mock_db.get_graph.assert_called_once()
        mock_db.store_graph.assert_called_once_with(graph)
        mock_db.update_graph.assert_not_called()
        assert result is True

    @pytest.mark.asyncio
    async def test_store_or_update_ukg_existing_graph(self, graph_service, developer_auth, mock_db):
        """Test _store_or_update_ukg updates an existing graph correctly."""
        end_user_id = "user_789"
        hashed_user_id = graph_service._hash_id(end_user_id)
        expected_graph_name = graph_service._generate_ukg_name(developer_auth.entity_id, end_user_id)
        
        # Mock that the graph already exists
        existing_graph = Graph(name=expected_graph_name, entities=[], relationships=[])
        mock_db.get_graph.return_value = existing_graph
        mock_db.update_graph.return_value = True
        
        # Create a graph to update
        graph = Graph(name="temp_name", entities=[], relationships=[])
        
        # Call the function
        result = await graph_service._store_or_update_ukg(developer_auth, end_user_id, graph)
        
        # Verify graph was updated with correct attributes
        assert graph.name == expected_graph_name
        assert graph.owner["id"] == developer_auth.entity_id
        assert graph.owner["type"] == developer_auth.entity_type
        assert graph.metadata["ukg_for_user"] == hashed_user_id
        
        # Verify correct methods were called
        mock_db.get_graph.assert_called_once()
        mock_db.update_graph.assert_called_once_with(graph)
        mock_db.store_graph.assert_not_called()
        assert result is True

    @pytest.mark.asyncio
    async def test_store_or_update_ukg_with_existing_metadata(self, graph_service, developer_auth, mock_db):
        """Test _store_or_update_ukg preserves existing metadata."""
        end_user_id = "user_789"
        hashed_user_id = graph_service._hash_id(end_user_id)
        
        # Mock that the graph doesn't exist yet
        mock_db.get_graph.return_value = None
        mock_db.store_graph.return_value = True
        
        # Create a graph with existing metadata
        graph = Graph(
            name="temp_name", 
            entities=[], 
            relationships=[],
            metadata={"existing_key": "existing_value"}
        )
        
        # Call the function
        await graph_service._store_or_update_ukg(developer_auth, end_user_id, graph)
        
        # Verify existing metadata was preserved and new metadata was added
        assert "existing_key" in graph.metadata
        assert graph.metadata["existing_key"] == "existing_value"
        assert "ukg_for_user" in graph.metadata
        assert graph.metadata["ukg_for_user"] == hashed_user_id
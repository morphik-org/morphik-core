import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from core.services.graph_service import GraphService
from core.models.auth import AuthContext, EntityType
from core.models.graph import Graph, Entity, Relationship


class TestProcessMemoryUpdate:
    """Tests for process_memory_update function in GraphService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = MagicMock()
        db.get_graph = AsyncMock()
        db.store_graph = AsyncMock()
        db.update_graph = AsyncMock()
        return db

    @pytest.fixture
    def mock_entity_extractor(self):
        """Create a mock entity extractor."""
        return AsyncMock(return_value=(
            # Sample entities
            [
                Entity(
                    label="Paris",
                    type="LOCATION",
                    properties={"country": "France"},
                    chunk_sources={"conv_123": [0]},
                    document_ids=["conv_123"]
                ),
                Entity(
                    label="Eiffel Tower",
                    type="LANDMARK",
                    properties={"height": "330m"},
                    chunk_sources={"conv_123": [0]},
                    document_ids=["conv_123"]
                )
            ],
            # Sample relationships
            [
                Relationship(
                    source_id="some-id-1",  # Will be fixed in the implementation
                    target_id="some-id-2",  # Will be fixed in the implementation
                    type="located_in",
                    chunk_sources={"conv_123": [0]},
                    document_ids=["conv_123"]
                )
            ]
        ))

    @pytest.fixture
    def graph_service(self, mock_db, mock_entity_extractor):
        """Create a GraphService with mock components."""
        service = GraphService.__new__(GraphService)
        service.db = mock_db
        # Mock helper methods
        service._get_ukg = AsyncMock()
        service._store_or_update_ukg = AsyncMock(return_value=True)
        service.extract_entities_from_text = mock_entity_extractor
        service._merge_entities = MagicMock()
        service._merge_relationships = MagicMock()
        # Add required class methods that are used in the implementation
        service._hash_id = lambda id_str: id_str[:8]  # Simple mock
        return service

    @pytest.fixture
    def developer_auth(self):
        """Create a developer authentication context."""
        return AuthContext(
            entity_type=EntityType.DEVELOPER,
            entity_id="dev_123",
            permissions=["read", "write"],
            user_id="dev_123"
        )

    @pytest.fixture
    def sample_conversation(self):
        """Create a sample conversation segment."""
        return [
            {"role": "user", "content": "Tell me about Paris"},
            {"role": "assistant", "content": "Paris is the capital of France and home to the Eiffel Tower."},
            {"role": "user", "content": "How tall is the Eiffel Tower?"}
        ]

    @pytest.mark.asyncio
    async def test_process_memory_update_new_ukg(
        self, graph_service, developer_auth, sample_conversation
    ):
        """Test process_memory_update when UKG doesn't exist yet."""
        # Set up mock to return None for _get_ukg (new UKG case)
        graph_service._get_ukg.return_value = None
        
        # Call function
        result = await graph_service.process_memory_update(
            developer_auth=developer_auth,
            end_user_id="user_456",
            conversation_segment=sample_conversation,
            conversation_id="123"
        )
        
        # Verify result
        assert result is True
        
        # Verify _get_ukg was called correctly
        graph_service._get_ukg.assert_called_once_with(developer_auth, "user_456")
        
        # Verify extract_entities_from_text was called
        graph_service.extract_entities_from_text.assert_called_once()
        # Check content passed to extraction contains conversation text
        call_args = graph_service.extract_entities_from_text.call_args
        assert "Paris" in call_args[1]["content"]
        assert "Eiffel Tower" in call_args[1]["content"]
        
        # Verify _store_or_update_ukg was called with a new graph
        graph_service._store_or_update_ukg.assert_called_once()
        new_graph = graph_service._store_or_update_ukg.call_args[0][2]
        assert isinstance(new_graph, Graph)
        assert new_graph.name == "temp_name"  # Will be replaced by UKG naming convention
        assert "conv_123" in new_graph.document_ids
        # Verify entities and relationships were added to the graph
        assert len(new_graph.entities) == 2
        assert len(new_graph.relationships) == 1

    @pytest.mark.asyncio
    async def test_process_memory_update_existing_ukg(
        self, graph_service, developer_auth, sample_conversation
    ):
        """Test process_memory_update when UKG already exists."""
        # Create an existing graph
        existing_graph = Graph(
            id="existing-id",
            name="ukg_dev_123_user_456",
            entities=[
                Entity(
                    label="France",
                    type="COUNTRY",
                    properties={"population": "67 million"},
                    chunk_sources={"conv_previous": [0]},
                    document_ids=["conv_previous"]
                )
            ],
            relationships=[],
            document_ids=["conv_previous"],
            owner={"type": "developer", "id": "dev_123"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={"ukg_for_user": "user_45"}
        )
        
        # Set up mock to return existing graph
        graph_service._get_ukg.return_value = existing_graph
        
        # Set up merged entities mock return value
        merged_entities = {
            "Paris": Entity(
                label="Paris",
                type="LOCATION",
                chunk_sources={"conv_123": [0]},
                document_ids=["conv_123"]
            ),
            "France": Entity(
                label="France",
                type="COUNTRY",
                chunk_sources={"conv_previous": [0]},
                document_ids=["conv_previous"]
            ),
            "Eiffel Tower": Entity(
                label="Eiffel Tower",
                type="LANDMARK",
                chunk_sources={"conv_123": [0]},
                document_ids=["conv_123"]
            )
        }
        graph_service._merge_entities.return_value = merged_entities
        
        # Set up merged relationships return value
        merged_relationships = [
            Relationship(
                source_id="paris-id",
                target_id="france-id",
                type="located_in",
                chunk_sources={"conv_123": [0]},
                document_ids=["conv_123"]
            )
        ]
        graph_service._merge_relationships.return_value = merged_relationships
        
        # Call function
        result = await graph_service.process_memory_update(
            developer_auth=developer_auth,
            end_user_id="user_456",
            conversation_segment=sample_conversation,
            conversation_id="123"
        )
        
        # Verify result
        assert result is True
        
        # Verify _get_ukg was called correctly
        graph_service._get_ukg.assert_called_once_with(developer_auth, "user_456")
        
        # Verify extract_entities_from_text was called
        graph_service.extract_entities_from_text.assert_called_once()
        # Check content passed to extraction contains conversation text
        call_args = graph_service.extract_entities_from_text.call_args
        assert "Paris" in call_args[1]["content"]
        assert "Eiffel Tower" in call_args[1]["content"]
        
        # Verify _merge_entities was called correctly
        graph_service._merge_entities.assert_called_once()
        existing_entities_arg = graph_service._merge_entities.call_args[0][0]
        new_entities_arg = graph_service._merge_entities.call_args[0][1]
        assert "France" in existing_entities_arg
        assert "Paris" in new_entities_arg
        assert "Eiffel Tower" in new_entities_arg
        
        # Verify _merge_relationships was called
        graph_service._merge_relationships.assert_called_once()
        
        # Verify _store_or_update_ukg was called with the updated graph
        graph_service._store_or_update_ukg.assert_called_once()
        updated_graph = graph_service._store_or_update_ukg.call_args[0][2]
        assert updated_graph is existing_graph  # Should be the same object
        # Verify entities and relationships were updated
        assert updated_graph.entities == list(merged_entities.values())
        assert updated_graph.relationships == merged_relationships

    @pytest.mark.asyncio
    async def test_process_memory_update_empty_conversation(
        self, graph_service, developer_auth
    ):
        """Test process_memory_update with empty conversation."""
        # Call function with empty conversation
        result = await graph_service.process_memory_update(
            developer_auth=developer_auth,
            end_user_id="user_456",
            conversation_segment=[],
            conversation_id="123"
        )
        
        # Verify result
        assert result is False
        
        # Verify _store_or_update_ukg was not called
        graph_service._store_or_update_ukg.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_memory_update_extraction_error(
        self, graph_service, developer_auth, sample_conversation
    ):
        """Test process_memory_update when entity extraction fails."""
        # Set up mock to raise an exception
        graph_service.extract_entities_from_text.side_effect = Exception("Extraction failed")
        
        # Call function
        result = await graph_service.process_memory_update(
            developer_auth=developer_auth,
            end_user_id="user_456",
            conversation_segment=sample_conversation,
            conversation_id="123"
        )
        
        # Verify result
        assert result is False
        
        # Verify _store_or_update_ukg was not called
        graph_service._store_or_update_ukg.assert_not_called()
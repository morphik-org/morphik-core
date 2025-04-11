#!/usr/bin/env python
"""
Memory Update Integration Test

This script tests the conversation memory update functionality by:
1. Creating and processing a sample conversation
2. Verifying that entities and relationships are extracted and stored in the UKG
3. Updating the UKG with a new conversation segment
4. Verifying that the UKG is correctly updated
"""

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models.auth import AuthContext, EntityType
from core.models.graph import Graph, Entity
from core.models.prompts import GraphPromptOverrides, EntityExtractionPromptOverride
from core.services.graph_service import GraphService
from core.database.postgres_database import PostgresDatabase
from core.embedding.litellm_embedding import LiteLLMEmbeddingModel
from core.completion.litellm_completion import LiteLLMCompletionModel
from core.config import get_settings

# Setup test data
DEV_ID = "dev_integration_test"
USER_ID = "user_integration_test"

def create_auth_context(entity_id):
    """Create an authentication context for a developer."""
    return AuthContext(
        entity_type=EntityType.DEVELOPER,
        entity_id=entity_id,
        permissions=["read", "write", "admin"],
        user_id=entity_id
    )

def hash_id(id_str):
    """Create a consistent hash of an ID string."""
    return hashlib.sha256(id_str.encode()).hexdigest()[:16]

async def test_memory_update():
    """Test memory update functionality with real conversations."""
    settings = get_settings()
    
    # Initialize database
    db = PostgresDatabase(uri=settings.POSTGRES_URI)
    await db.initialize()
    
    # Initialize required components for GraphService
    embedding_model = LiteLLMEmbeddingModel(model_key=settings.EMBEDDING_MODEL)
    completion_model = LiteLLMCompletionModel(model_key=settings.COMPLETION_MODEL)
    
    # Initialize GraphService
    graph_service = GraphService(
        db=db,
        embedding_model=embedding_model,
        completion_model=completion_model
    )
    
    # Create auth context
    dev_auth = create_auth_context(DEV_ID)
    
    # Generate UKG name for cleanup
    ukg_name = graph_service._generate_ukg_name(DEV_ID, USER_ID)
    
    try:
        print("\n=== Memory Update Integration Test ===\n")
        
        # Clean up any existing UKG from previous test runs
        try:
            print(f"Cleaning up any existing UKG ({ukg_name})...")
            existing_graph = await graph_service._get_ukg(dev_auth, USER_ID)
            if existing_graph:
                # Delete the graph (by using a direct SQL command)
                async with db.async_session() as session:
                    from sqlalchemy import text
                    
                    await session.execute(text(f"DELETE FROM graphs WHERE name = '{ukg_name}'"))
                    await session.commit()
                    print("Existing UKG deleted.")
            else:
                print("No existing UKG found.")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        
        # Test 1: Process first conversation segment
        print("\n--- Test 1: Process First Conversation Segment ---")
        
        # Create a sample conversation
        conversation1 = [
            {"role": "user", "content": "Tell me about Paris, France."},
            {"role": "assistant", "content": "Paris is the capital of France and known for its iconic Eiffel Tower. It's often called the City of Light and is famous for its art, fashion, and cuisine."},
            {"role": "user", "content": "How tall is the Eiffel Tower?"}
        ]
        
        # Create custom entity extraction prompt override for better results
        prompt_overrides = GraphPromptOverrides(
            entity_extraction=EntityExtractionPromptOverride(
                prompt_template="Extract named entities and relationships from this conversation: {content}\n\nLook for people, places, organizations, concepts, and other important entities.\n{examples}",
                examples=[
                    {"label": "Paris", "type": "LOCATION", "properties": {"country": "France"}},
                    {"label": "Eiffel Tower", "type": "LANDMARK", "properties": {"location": "Paris"}},
                ]
            )
        )
        
        # Process the conversation
        success = await graph_service.process_memory_update(
            developer_auth=dev_auth,
            end_user_id=USER_ID,
            conversation_segment=conversation1,
            conversation_id="conv1",
            prompt_overrides=prompt_overrides
        )
        
        if success:
            print("✓ Successfully processed first conversation segment")
            
            # Retrieve the UKG
            ukg = await graph_service._get_ukg(dev_auth, USER_ID)
            
            if ukg:
                print(f"✓ UKG created successfully")
                print(f"  Graph name: {ukg.name}")
                print(f"  Entity count: {len(ukg.entities)}")
                print(f"  Relationship count: {len(ukg.relationships)}")
                
                # Display entity details
                print("\n  Entities found:")
                for entity in ukg.entities:
                    print(f"    - {entity.label} ({entity.type})")
                    if entity.properties:
                        print(f"      Properties: {json.dumps(entity.properties)}")
                
                # Store entity count for later comparison
                initial_entity_count = len(ukg.entities)
                initial_relationship_count = len(ukg.relationships)
            else:
                print("✗ Failed to create UKG")
        else:
            print("✗ Failed to process first conversation segment")
            return
        
        # Test 2: Process second conversation segment
        print("\n--- Test 2: Process Second Conversation Segment ---")
        
        # Create a second sample conversation with new information
        conversation2 = [
            {"role": "user", "content": "Tell me about the Louvre Museum."},
            {"role": "assistant", "content": "The Louvre Museum is one of the world's largest and most visited art museums, located in Paris, France. It houses famous works including the Mona Lisa painted by Leonardo da Vinci."},
            {"role": "user", "content": "Who painted the Mona Lisa?"}
        ]
        
        # Process the second conversation
        success = await graph_service.process_memory_update(
            developer_auth=dev_auth,
            end_user_id=USER_ID,
            conversation_segment=conversation2,
            conversation_id="conv2",
            prompt_overrides=prompt_overrides
        )
        
        if success:
            print("✓ Successfully processed second conversation segment")
            
            # Retrieve the updated UKG
            updated_ukg = await graph_service._get_ukg(dev_auth, USER_ID)
            
            if updated_ukg:
                print(f"✓ UKG updated successfully")
                print(f"  Graph name: {updated_ukg.name}")
                print(f"  Entity count: {len(updated_ukg.entities)}")
                print(f"  Relationship count: {len(updated_ukg.relationships)}")
                
                # Display new entities
                print("\n  Entities found:")
                for entity in updated_ukg.entities:
                    print(f"    - {entity.label} ({entity.type})")
                    if entity.properties:
                        print(f"      Properties: {json.dumps(entity.properties)}")
                
                # Verify that UKG was updated with new entities
                if len(updated_ukg.entities) > initial_entity_count:
                    print(f"\n✓ New entities were added (from {initial_entity_count} to {len(updated_ukg.entities)})")
                else:
                    print(f"\n! No new entities were added (still {len(updated_ukg.entities)})")
                
                # Check if there are new relationships
                if len(updated_ukg.relationships) > initial_relationship_count:
                    print(f"✓ New relationships were added (from {initial_relationship_count} to {len(updated_ukg.relationships)})")
                else:
                    print(f"! No new relationships were added (still {len(updated_ukg.relationships)})")
            else:
                print("✗ Failed to retrieve updated UKG")
        else:
            print("✗ Failed to process second conversation segment")
        
        print("\n=== Test Complete ===")
        
    except Exception as e:
        print(f"\n✗ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_memory_update())
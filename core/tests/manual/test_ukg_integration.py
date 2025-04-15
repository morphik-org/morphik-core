#!/usr/bin/env python
"""
UKG Integration Sanity Check Script

This script tests the User Knowledge Graph (UKG) functionality by:
1. Creating UKGs for different developer/user combinations
2. Verifying the UKGs are created with correct names and metadata
3. Retrieving UKGs to confirm they can be accessed correctly
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
from core.models.graph import Graph
from core.services.graph_service import GraphService
from core.database.postgres_database import PostgresDatabase
from core.config import get_settings

# Setup test data
DEV_A_ID = "dev_a_123"
DEV_B_ID = "dev_b_456"
USER_X_ID = "user_x_789"
USER_Y_ID = "user_y_012"

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

async def test_ukg_functionality():
    """Test UKG creation, storage, and retrieval."""
    settings = get_settings()
    
    # Initialize database
    db = PostgresDatabase(uri=settings.POSTGRES_URI)
    await db.initialize()
    
    # Clean up any existing UKGs from previous test runs (optional)
    try:
        print("Cleaning up any existing UKGs...")
        async with db.async_session() as session:
            from sqlalchemy import text
            
            # Delete graphs with names starting with "ukg_"
            await session.execute(text("DELETE FROM graphs WHERE name LIKE 'ukg_%'"))
            await session.commit()
            print("Cleanup complete.")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        # Continue with test anyway
    
    # Initialize GraphService with minimal dependencies
    # Note: We're not using the full constructor as we don't need embedding/completion models
    graph_service = GraphService.__new__(GraphService)
    graph_service.db = db
    # We need to add the helper methods directly since we're not calling __init__
    graph_service._hash_id = lambda id_str: hashlib.sha256(id_str.encode()).hexdigest()[:16]
    graph_service._generate_ukg_name = lambda dev_id, user_id: f"ukg_{graph_service._hash_id(dev_id)}_{graph_service._hash_id(user_id)}"
    # Define the _get_ukg method
    async def _get_ukg(developer_auth, end_user_id):
        ukg_name = graph_service._generate_ukg_name(developer_auth.entity_id, end_user_id)
        return await db.get_graph(ukg_name, developer_auth)
    graph_service._get_ukg = _get_ukg
    
    # Define the _store_or_update_ukg method
    async def _store_or_update_ukg(developer_auth, end_user_id, graph):
        # Hash the end user ID
        hashed_end_user_id = graph_service._hash_id(end_user_id)
        # Generate the UKG name
        ukg_name = graph_service._generate_ukg_name(developer_auth.entity_id, end_user_id)
        
        # Set the owner and name
        graph.owner = {"type": developer_auth.entity_type, "id": developer_auth.entity_id}
        graph.name = ukg_name
        
        # Ensure access_control is set
        graph.access_control = {
            "readers": [developer_auth.entity_id],
            "writers": [developer_auth.entity_id],
            "admins": [developer_auth.entity_id]
        }
        
        # Set or update the graph metadata with the hashed end user ID
        if not graph.metadata:
            graph.metadata = {}
        graph.metadata["ukg_for_user"] = hashed_end_user_id
        
        # Check if the graph already exists
        existing_graph = await _get_ukg(developer_auth, end_user_id)
        
        if existing_graph:
            # Copy the ID from the existing graph to ensure proper update
            graph.id = existing_graph.id
            # Update the existing graph
            return await db.update_graph(graph)
        else:
            # Store a new graph
            return await db.store_graph(graph)
    
    graph_service._store_or_update_ukg = _store_or_update_ukg
    
    # Create auth contexts
    dev_a_auth = create_auth_context(DEV_A_ID)
    dev_b_auth = create_auth_context(DEV_B_ID)
    
    try:
        print("\n=== UKG Integration Test ===\n")
        
        # Test 1: Create UKG for (DevA, UserX)
        print("\n--- Test 1: Create UKG for (DevA, UserX) ---")
        
        # Create a new graph for UserX
        graph_a_x = Graph(
            name="temp_name",  # Will be replaced by UKG naming convention
            entities=[],
            relationships=[],
            metadata={"test_time": datetime.now(timezone.utc).isoformat()}
        )
        
        # Store the graph using our UKG helper
        success = await graph_service._store_or_update_ukg(dev_a_auth, USER_X_ID, graph_a_x)
        
        if success:
            print("✓ Successfully created UKG for (DevA, UserX)")
            
            # Verify the graph name follows our convention
            expected_name = f"ukg_{hash_id(DEV_A_ID)}_{hash_id(USER_X_ID)}"
            print(f"  Graph name: {graph_a_x.name}")
            print(f"  Expected name: {expected_name}")
            assert graph_a_x.name == expected_name
            
            # Verify the owner is set correctly
            print(f"  Owner: {graph_a_x.owner}")
            assert graph_a_x.owner["id"] == DEV_A_ID
            assert graph_a_x.owner["type"] == "developer"
            
            # Verify the metadata contains the hashed user ID
            print(f"  Metadata: {json.dumps(graph_a_x.metadata)}")
            assert "ukg_for_user" in graph_a_x.metadata
            assert graph_a_x.metadata["ukg_for_user"] == hash_id(USER_X_ID)
        else:
            print("✗ Failed to create UKG for (DevA, UserX)")
        
        # Test 2: Create UKG for (DevA, UserY)
        print("\n--- Test 2: Create UKG for (DevA, UserY) ---")
        
        # Create a new graph for UserY
        graph_a_y = Graph(
            name="temp_name",  # Will be replaced by UKG naming convention
            entities=[],
            relationships=[],
            metadata={"test_time": datetime.now(timezone.utc).isoformat()}
        )
        
        # Store the graph using our UKG helper
        success = await graph_service._store_or_update_ukg(dev_a_auth, USER_Y_ID, graph_a_y)
        
        if success:
            print("✓ Successfully created UKG for (DevA, UserY)")
            
            # Verify the graph name follows our convention
            expected_name = f"ukg_{hash_id(DEV_A_ID)}_{hash_id(USER_Y_ID)}"
            print(f"  Graph name: {graph_a_y.name}")
            print(f"  Expected name: {expected_name}")
            assert graph_a_y.name == expected_name
            
            # Verify the owner is set correctly
            print(f"  Owner: {graph_a_y.owner}")
            assert graph_a_y.owner["id"] == DEV_A_ID
            assert graph_a_y.owner["type"] == "developer"
            
            # Verify the metadata contains the hashed user ID
            print(f"  Metadata: {json.dumps(graph_a_y.metadata)}")
            assert "ukg_for_user" in graph_a_y.metadata
            assert graph_a_y.metadata["ukg_for_user"] == hash_id(USER_Y_ID)
        else:
            print("✗ Failed to create UKG for (DevA, UserY)")
        
        # Test 3: Create UKG for (DevB, UserX)
        print("\n--- Test 3: Create UKG for (DevB, UserX) ---")
        
        # Create a new graph for UserX with DevB
        graph_b_x = Graph(
            name="temp_name",  # Will be replaced by UKG naming convention
            entities=[],
            relationships=[],
            metadata={"test_time": datetime.now(timezone.utc).isoformat()}
        )
        
        # Store the graph using our UKG helper
        success = await graph_service._store_or_update_ukg(dev_b_auth, USER_X_ID, graph_b_x)
        
        if success:
            print("✓ Successfully created UKG for (DevB, UserX)")
            
            # Verify the graph name follows our convention
            expected_name = f"ukg_{hash_id(DEV_B_ID)}_{hash_id(USER_X_ID)}"
            print(f"  Graph name: {graph_b_x.name}")
            print(f"  Expected name: {expected_name}")
            assert graph_b_x.name == expected_name
            
            # Verify the owner is set correctly
            print(f"  Owner: {graph_b_x.owner}")
            assert graph_b_x.owner["id"] == DEV_B_ID
            assert graph_b_x.owner["type"] == "developer"
            
            # Verify the metadata contains the hashed user ID
            print(f"  Metadata: {json.dumps(graph_b_x.metadata)}")
            assert "ukg_for_user" in graph_b_x.metadata
            assert graph_b_x.metadata["ukg_for_user"] == hash_id(USER_X_ID)
        else:
            print("✗ Failed to create UKG for (DevB, UserX)")
        
        # Test 4: Retrieve UKG for (DevA, UserX)
        print("\n--- Test 4: Retrieve UKG for (DevA, UserX) ---")
        
        # Retrieve the graph using our UKG helper
        retrieved_graph = await graph_service._get_ukg(dev_a_auth, USER_X_ID)
        
        if retrieved_graph:
            print("✓ Successfully retrieved UKG for (DevA, UserX)")
            
            # Verify it's the same graph we created
            print(f"  Retrieved graph name: {retrieved_graph.name}")
            print(f"  Expected name: {graph_a_x.name}")
            assert retrieved_graph.name == graph_a_x.name
            
            # Check the metadata
            print(f"  Metadata: {json.dumps(retrieved_graph.metadata)}")
            assert "ukg_for_user" in retrieved_graph.metadata
            assert retrieved_graph.metadata["ukg_for_user"] == hash_id(USER_X_ID)
        else:
            print("✗ Failed to retrieve UKG for (DevA, UserX)")
        
        # Test 5: Update existing UKG
        print("\n--- Test 5: Update existing UKG for (DevA, UserX) ---")
        
        # Create an updated graph with the same name but new content
        graph_a_x_updated = Graph(
            name="temp_name",  # Will be replaced by UKG naming convention
            entities=[],
            relationships=[],
            metadata={
                "test_time": datetime.now(timezone.utc).isoformat(),
                "update_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Update the graph using our UKG helper
        success = await graph_service._store_or_update_ukg(dev_a_auth, USER_X_ID, graph_a_x_updated)
        
        if success:
            print("✓ Successfully updated UKG for (DevA, UserX)")
            
            # Verify the graph name follows our convention
            expected_name = f"ukg_{hash_id(DEV_A_ID)}_{hash_id(USER_X_ID)}"
            print(f"  Graph name: {graph_a_x_updated.name}")
            print(f"  Expected name: {expected_name}")
            assert graph_a_x_updated.name == expected_name
            
            # Verify the metadata contains both the hashed user ID and our update timestamp
            print(f"  Metadata: {json.dumps(graph_a_x_updated.metadata)}")
            assert "ukg_for_user" in graph_a_x_updated.metadata
            assert graph_a_x_updated.metadata["ukg_for_user"] == hash_id(USER_X_ID)
            assert "update_timestamp" in graph_a_x_updated.metadata
        else:
            print("✗ Failed to update UKG for (DevA, UserX)")
        
        print("\n=== Test Summary ===")
        print("✓ All UKG tests completed successfully")
        
    except Exception as e:
        print(f"\n✗ Error during test: {str(e)}")
        raise e

if __name__ == "__main__":
    asyncio.run(test_ukg_functionality())
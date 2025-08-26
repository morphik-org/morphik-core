#!/usr/bin/env python3
"""
Script to sync graph statuses with the remote image-graph-rag service.

This script helps fix graphs that are stuck in 'processing' state when they
are actually completed on the remote service.

Usage:
    python scripts/sync_graph_status.py [--all] [--graph-name GRAPH_NAME]
"""

import argparse
import asyncio
import logging
import sys
from typing import List

# Add the project root to the path
sys.path.insert(0, ".")

from core.models.auth import AuthContext, EntityType
from core.services_init import document_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def sync_single_graph(graph_name: str, auth: AuthContext) -> bool:
    """Sync a single graph's status."""
    try:
        from core.services.morphik_graph_service import MorphikGraphService

        graph_service = document_service.graph_service
        if not isinstance(graph_service, MorphikGraphService):
            logger.error("Graph service is not MorphikGraphService - status sync not available")
            return False

        logger.info(f"Syncing status for graph: {graph_name}")
        success = await graph_service.sync_graph_status(graph_name, auth)

        if success:
            logger.info(f"✅ Successfully synced status for graph: {graph_name}")
        else:
            logger.error(f"❌ Failed to sync status for graph: {graph_name}")

        return success
    except Exception as e:
        logger.error(f"❌ Error syncing graph {graph_name}: {e}")
        return False


async def sync_all_processing_graphs(auth: AuthContext) -> List[str]:
    """Find and sync all graphs that are in 'processing' state."""
    try:
        # Get all graphs
        graphs = await document_service.db.list_graphs(auth)
        processing_graphs = [g for g in graphs if g.system_metadata.get("status") == "processing"]

        if not processing_graphs:
            logger.info("No graphs found in 'processing' state")
            return []

        logger.info(f"Found {len(processing_graphs)} graphs in 'processing' state:")
        for graph in processing_graphs:
            logger.info(f"  - {graph.name} (ID: {graph.id})")

        synced_graphs = []
        for graph in processing_graphs:
            success = await sync_single_graph(graph.name, auth)
            if success:
                synced_graphs.append(graph.name)

        return synced_graphs

    except Exception as e:
        logger.error(f"Error syncing processing graphs: {e}")
        return []


async def main():
    parser = argparse.ArgumentParser(description="Sync graph statuses with remote service")
    parser.add_argument("--all", action="store_true", help="Sync all graphs that are in 'processing' state")
    parser.add_argument("--graph-name", type=str, help="Sync a specific graph by name")
    parser.add_argument("--app-id", type=str, default="default", help="App ID for authentication (default: 'default')")

    args = parser.parse_args()

    if not args.all and not args.graph_name:
        parser.error("Must specify either --all or --graph-name")

    # Create a developer auth context for the sync operation
    auth = AuthContext(
        entity_type=EntityType.DEVELOPER,
        entity_id="graph_sync_script",
        app_id=args.app_id,
        permissions={"read", "write"},
        user_id="graph_sync_script",
    )

    try:
        if args.graph_name:
            # Sync specific graph
            success = await sync_single_graph(args.graph_name, auth)
            if not success:
                sys.exit(1)
        else:
            # Sync all processing graphs
            synced_graphs = await sync_all_processing_graphs(auth)
            logger.info(f"Sync complete. {len(synced_graphs)} graphs synchronized.")

            if synced_graphs:
                logger.info("Synchronized graphs:")
                for graph_name in synced_graphs:
                    logger.info(f"  ✅ {graph_name}")

    except KeyboardInterrupt:
        logger.info("Sync operation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

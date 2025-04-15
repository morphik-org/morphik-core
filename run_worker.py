#!/usr/bin/env python3
"""
Run the ARQ worker for processing background jobs.

This script starts an ARQ worker that processes background jobs
from the Redis queue, such as file ingestion tasks.

Usage:
    python run_worker.py

Environment variables:
    REDIS_HOST: Redis host (default: from morphik.toml)
    REDIS_PORT: Redis port (default: from morphik.toml)
"""

import os
import sys
import logging
import arq
from core.workers.ingestion_worker import WorkerSettings
from core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

async def main():
    """Run the ARQ worker."""
    settings = get_settings()
    
    # Get Redis settings from environment variables or config
    redis_host = os.environ.get("REDIS_HOST", settings.REDIS_HOST)
    redis_port = int(os.environ.get("REDIS_PORT", settings.REDIS_PORT))
    
    logger.info(f"Starting worker with Redis at {redis_host}:{redis_port}")
    
    redis_settings = arq.connections.RedisSettings(
        host=redis_host,
        port=redis_port,
    )
    
    worker = arq.worker.Worker(
        WorkerSettings,
        redis_settings=redis_settings,
    )
    
    await worker.async_run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
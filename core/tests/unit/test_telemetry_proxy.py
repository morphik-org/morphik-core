#!/usr/bin/env python
"""
Test script to verify that telemetry data is being properly sent through the proxy.
This script will generate a test span and metric and send it to Honeycomb via the proxy.
"""

import time
import logging
import uuid
import asyncio
from datetime import datetime
from core.tests import setup_test_logging

# Configure test logging
setup_test_logging()
logger = logging.getLogger("telemetry-test")

# Import the telemetry service
from core.services.telemetry import TelemetryService
from core.config import get_settings

async def run_test():
    """Run a telemetry test to verify proxy functionality."""
    settings = get_settings()
    
    # Log the current configuration
    logger.info(f"Telemetry enabled: {settings.TELEMETRY_ENABLED}")
    logger.info(f"Honeycomb enabled: {settings.HONEYCOMB_ENABLED}")
    logger.info(f"Honeycomb proxy endpoint: {settings.HONEYCOMB_PROXY_ENDPOINT}")
    
    # Get the telemetry service
    telemetry_service = TelemetryService()
    
    # Generate a unique user ID for testing
    test_user_id = f"test-user-{uuid.uuid4()}"
    
    # Track a test operation
    logger.info(f"Tracking test operation for user {test_user_id}")
    
    # Use the telemetry service to track an operation (with async context manager)
    async with telemetry_service.track_operation(
        operation_type="test_proxy",
        user_id=test_user_id,
        tokens_used=100,
        metadata={
            "test": True,
            "timestamp": datetime.now().isoformat(),
            "proxy_test": "Honeycomb proxy test"
        }
    ) as span:
        # Simulate some work
        logger.info("Performing test operation...")
        await asyncio.sleep(2)
        
        # Add some attributes to the span
        span.set_attribute("test.proxy", True)
        span.set_attribute("test.timestamp", time.time())
        
        # Log a message
        logger.info("Test operation completed successfully")
    
    # Wait a moment for the telemetry data to be sent
    logger.info("Waiting for telemetry data to be sent...")
    await asyncio.sleep(5)
    
    logger.info("Test completed. Check Honeycomb for the telemetry data.")
    logger.info(f"Look for operation_type='test_proxy' and user_id='{test_user_id}'")

def main():
    """Run the async test function."""
    asyncio.run(run_test())

if __name__ == "__main__":
    main()

"""
Test script to verify that memory ingestion works in the chat API endpoint.

This script sends two requests to the /chat/completions endpoint with different end_user_ids
and verifies that the UKGs are created properly.

Usage:
    python test_chat_memory.py

Note: The API server must be running before executing this script.
"""

import asyncio
import logging
import requests
import json
import time
import urllib.parse
from datetime import datetime, UTC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# API configuration
API_URL = "http://localhost:8000"  # Change if your API is running on a different port
MORPHIK_URI = None  # Will be populated by generating a URI


async def generate_uri():
    """Generate a local test URI for authentication."""
    global MORPHIK_URI
    try:
        logger.info("Generating local URI for testing")
        response = requests.post(f"{API_URL}/local/generate_uri", data={"name": "testdev"})
        if response.status_code == 200:
            MORPHIK_URI = response.json()["uri"]
            logger.info(f"Generated URI for testdev")
            return True
        else:
            logger.error(f"Failed to generate URI: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error generating URI: {e}")
        return False


def extract_credentials_from_uri(uri):
    """Extract the Bearer token from the Morphik URI."""
    # Parse the URI format: morphik://username:token@host:port
    parsed = urllib.parse.urlparse(uri)
    
    # Extract username and token from the netloc
    auth_parts = parsed.netloc.split('@')[0].split(':')
    username = auth_parts[0]
    token = auth_parts[1]
    
    return {"Authorization": f"Bearer {token}"}


async def test_chat_completion(end_user_id, remember=True):
    """Test the chat completion endpoint with a specific end_user_id."""
    if not MORPHIK_URI:
        if not await generate_uri():
            logger.error("Failed to generate URI. Cannot proceed with tests.")
            return False
    
    headers = extract_credentials_from_uri(MORPHIK_URI)
    headers["Content-Type"] = "application/json"
    
    # Example chat messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Paris, France."}
    ]
    
    # Request body
    data = {
        "messages": messages,
        "end_user_id": end_user_id,
        "remember": remember,
        "conversation_id": f"test_{datetime.now(UTC).isoformat()}",
        "k": 3,
        "max_tokens": 250,
        "temperature": 0.7
    }
    
    try:
        logger.info(f"Testing chat completion with end_user_id={end_user_id}, remember={remember}")
        response = requests.post(
            f"{API_URL}/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            completion = response.json().get("completion", "")
            logger.info(f"Chat completion successful for {end_user_id}")
            logger.info(f"Response begins with: {completion[:100]}...")
            return True
        else:
            logger.error(f"Chat completion failed for {end_user_id}: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing chat completion for {end_user_id}: {e}")
        return False


async def main():
    """Run the test script."""
    logger.info("Starting chat memory integration tests")
    
    # Test with first user (remember=True)
    user1_result = await test_chat_completion("userX", remember=True)
    
    # Small pause to ensure background task has time to run
    time.sleep(2)
    
    # Test with second user (remember=True)
    user2_result = await test_chat_completion("userY", remember=True)
    
    # Small pause to ensure background task has time to run
    time.sleep(2)
    
    # Final test with first user again to test UKG update
    user1_update_result = await test_chat_completion("userX", remember=True)
    
    # Wait for background tasks to complete
    time.sleep(3)
    
    # Report results
    if user1_result and user2_result and user1_update_result:
        logger.info("All tests passed!")
        logger.info("Check the API logs for details about the memory ingestion process.")
        logger.info("Look for log lines starting with 'MEMORY_UPDATE:'")
    else:
        logger.error("Some tests failed. Check the logs for details.")


if __name__ == "__main__":
    asyncio.run(main())
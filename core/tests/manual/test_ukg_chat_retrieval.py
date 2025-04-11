"""
Test script to verify that end-user aware UKG context retrieval works correctly.

This script:
1. Ingests two documents: one about Paris (Topic A) and one about Tokyo (Topic B)
2. Has chat 1 with User X about Paris, with remember=True
3. Has chat 2 with User Y about Tokyo, with remember=True
4. Tests retrieval for User X asking about Paris
5. Tests retrieval for User Y asking about Paris
6. Tests retrieval for User Y asking about Tokyo

Usage:
    python test_ukg_chat_retrieval.py

Note: The API server must be running before executing this script.
"""

import asyncio
import logging
import requests
import json
import time
import urllib.parse
from datetime import datetime, UTC
import random

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


async def ingest_document(content, filename, metadata=None):
    """Ingest a text document into the system."""
    if not MORPHIK_URI:
        if not await generate_uri():
            logger.error("Failed to generate URI. Cannot proceed with tests.")
            return None
    
    headers = extract_credentials_from_uri(MORPHIK_URI)
    headers["Content-Type"] = "application/json"
    
    # Default metadata
    if metadata is None:
        metadata = {}
    
    # Request body
    data = {
        "content": content,
        "filename": filename,
        "metadata": metadata
    }
    
    try:
        logger.info(f"Ingesting document: {filename}")
        response = requests.post(
            f"{API_URL}/ingest/text",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            result = response.json()
            doc_id = result.get("id") or result.get("external_id")
            logger.info(f"Successfully ingested document {filename} with ID {doc_id}")
            return doc_id or True  # Return True as a fallback if no ID is found
        else:
            logger.error(f"Failed to ingest document: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        return None


async def chat_completion(end_user_id, message, remember=True, conversation_id=None):
    """Send a chat completion request with the specified end_user_id."""
    if not MORPHIK_URI:
        if not await generate_uri():
            logger.error("Failed to generate URI. Cannot proceed with tests.")
            return None
    
    headers = extract_credentials_from_uri(MORPHIK_URI)
    headers["Content-Type"] = "application/json"
    
    # Example chat messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
    ]
    
    # Generate a conversation ID if not provided
    if not conversation_id:
        conversation_id = f"test_{datetime.now(UTC).isoformat()}_{random.randint(1000, 9999)}"
    
    # Request body
    data = {
        "messages": messages,
        "end_user_id": end_user_id,
        "remember": remember,
        "conversation_id": conversation_id,
        "k": 3,
        "max_tokens": 250,
        "temperature": 0.7
    }
    
    try:
        logger.info(f"Testing chat completion with end_user_id={end_user_id}, message='{message[:50]}...'")
        response = requests.post(
            f"{API_URL}/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            result = response.json()
            completion = result.get("completion", "")
            logger.info(f"Chat completion successful for {end_user_id}")
            logger.info(f"Response begins with: {completion[:100]}...")
            return result
        else:
            logger.error(f"Chat completion failed for {end_user_id}: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error in chat completion for {end_user_id}: {e}")
        return None


async def main():
    """Run the test script."""
    logger.info("Starting UKG-aware chat retrieval tests")
    
    # Use timestamp to generate unique user IDs for this test run
    timestamp = int(time.time())
    user_x = f"userX_{timestamp}"
    user_y = f"userY_{timestamp}"
    logger.info(f"Using unique test users: {user_x} and {user_y}")
    
    # Step 1: Ingest test documents
    paris_content = """
    Paris is the capital and most populous city of France. Situated on the Seine River, 
    in the north of the country, it is at the heart of the Île-de-France region. 
    The city is known for its museums and architectural landmarks: the Louvre was the 
    most visited art museum in the world in 2019, with 9.6 million visitors. 
    The Eiffel Tower is one of the world's most recognizable landmarks, 
    completed in 1889 as the entrance arch to the 1889 World's Fair.
    
    Paris is often referred to as the "City of Light" (La Ville Lumière), 
    both because of its leading role during the Age of Enlightenment and 
    because Paris was one of the first large European cities to use gas street lighting.
    
    The Seine River runs through the city and divides it into the Right Bank and Left Bank.
    """
    
    tokyo_content = """
    Tokyo is the capital and most populous prefecture of Japan. Located at the head of 
    Tokyo Bay, the prefecture forms part of the Kantō region on the central Pacific coast 
    of Japan's main island of Honshu. Tokyo is the political, economic, and cultural 
    center of Japan, and houses the seat of the Emperor and the national government.
    
    As of 2021, the prefecture has an estimated population of 14.04 million. 
    The Greater Tokyo Area is the most populous metropolitan area in the world, 
    with an estimated 37.468 million residents in 2018.
    
    Tokyo was formerly known as Edo when Shōgun Tokugawa Ieyasu made the city his 
    headquarters in 1603. It became the capital after Emperor Meiji moved his seat 
    to the city from Kyoto in 1868; at that time Edo was renamed Tokyo.
    
    The Tokyo Tower is a communications and observation tower in the Shiba-koen district 
    of Tokyo, Japan, built in 1958. At 333 meters, it is the second-tallest structure in Japan.
    """
    
    paris_doc_id = await ingest_document(
        content=paris_content,
        filename="paris_info.txt",
        metadata={"topic": "geography", "city": "Paris"}
    )
    
    tokyo_doc_id = await ingest_document(
        content=tokyo_content,
        filename="tokyo_info.txt",
        metadata={"topic": "geography", "city": "Tokyo"}
    )
    
    if paris_doc_id is None or tokyo_doc_id is None:
        logger.error("Failed to ingest test documents. Aborting tests.")
        return
    
    # Wait for the documents to be processed
    logger.info("Waiting for documents to be processed...")
    time.sleep(10)
    
    # Step 2: Chat with User X about Paris with remember=True
    paris_conversation_id = f"test_paris_{datetime.now(UTC).isoformat()}"
    paris_response_x = await chat_completion(
        end_user_id=user_x,
        message="Tell me about Paris and its famous landmarks.",
        remember=True,
        conversation_id=paris_conversation_id
    )
    
    # Wait for the background task to complete
    logger.info(f"Waiting for UKG creation for {user_x}...")
    time.sleep(10)
    
    # Step 3: Chat with User Y about Tokyo with remember=True
    tokyo_conversation_id = f"test_tokyo_{datetime.now(UTC).isoformat()}"
    tokyo_response_y = await chat_completion(
        end_user_id=user_y,
        message="What can you tell me about Tokyo and its history?",
        remember=True,
        conversation_id=tokyo_conversation_id
    )
    
    # Wait for the background task to complete
    logger.info(f"Waiting for UKG creation for {user_y}...")
    time.sleep(10)
    
    # Step 4: Test retrieval for User X asking about Paris
    logger.info(f"\n\n===== TEST 1: {user_x} asking about Paris (should use UKG) =====")
    test1_response = await chat_completion(
        end_user_id=user_x,
        message="What is the Eiffel Tower?",
        remember=True
    )
    
    if test1_response:
        logger.info("TEST 1 RESPONSE ANALYSIS:")
        completion = test1_response.get("completion", "")
        # Check if the completion contains indicators of memory usage
        memory_indicators = ["memory", "previously discussed", "you mentioned", "we talked about"]
        memory_found = any(indicator in completion.lower() for indicator in memory_indicators)
        logger.info(f"Memory indicators found in response: {memory_found}")
        logger.info(f"Full response: {completion}")
    
    # Step 5: Test retrieval for User Y asking about Paris
    logger.info(f"\n\n===== TEST 2: {user_y} asking about Paris (should NOT use UKG) =====")
    test2_response = await chat_completion(
        end_user_id=user_y,
        message="What is the Eiffel Tower?",
        remember=True
    )
    
    if test2_response:
        logger.info("TEST 2 RESPONSE ANALYSIS:")
        completion = test2_response.get("completion", "")
        # Check if the completion contains indicators of memory usage
        memory_indicators = ["memory", "previously discussed", "you mentioned", "we talked about"]
        memory_found = any(indicator in completion.lower() for indicator in memory_indicators)
        logger.info(f"Memory indicators found in response: {memory_found}")
        logger.info(f"Full response: {completion}")
    
    # Step 6: Test retrieval for User Y asking about Tokyo
    logger.info(f"\n\n===== TEST 3: {user_y} asking about Tokyo (should use UKG) =====")
    test3_response = await chat_completion(
        end_user_id=user_y,
        message="Tell me about Tokyo Tower.",
        remember=True
    )
    
    if test3_response:
        logger.info("TEST 3 RESPONSE ANALYSIS:")
        completion = test3_response.get("completion", "")
        # Check if the completion contains indicators of memory usage
        memory_indicators = ["memory", "previously discussed", "you mentioned", "we talked about"]
        memory_found = any(indicator in completion.lower() for indicator in memory_indicators)
        logger.info(f"Memory indicators found in response: {memory_found}")
        logger.info(f"Full response: {completion}")
    
    logger.info("\n\nAll tests completed! Check the API logs for details about UKG retrieval.")
    logger.info("Look for log lines containing 'Retrieved UKG for end_user_id=' and 'Combined UKG chunks and RAG chunks'")


if __name__ == "__main__":
    asyncio.run(main())
import requests
import json
import sys

# Server URL
BASE_URL = "http://localhost:8000"

# Default auth token (using dev mode)
AUTH_TOKEN = "Bearer dev_token"

def test_chat_completion():
    """Test the /chat/completions endpoint with various end_user_id scenarios"""
    
    # Test cases with valid end_user_id
    test_cases = [
        {
            "name": "Basic request with end_user_id",
            "payload": {
                "messages": [{"role": "user", "content": "What is Morphik?"}],
                "end_user_id": "test_user_123",
                "remember": False,
                "k": 3
            },
            "expected_code": 200
        },
        {
            "name": "Different end_user_id",
            "payload": {
                "messages": [{"role": "user", "content": "Tell me about RAG systems"}],
                "end_user_id": "another_user_456",
                "remember": False,
                "k": 3
            },
            "expected_code": 200
        },
        {
            "name": "Multi-message conversation with end_user_id",
            "payload": {
                "messages": [
                    {"role": "user", "content": "What is Morphik?"},
                    {"role": "assistant", "content": "Morphik is a document intelligence platform."},
                    {"role": "user", "content": "How does it work?"}
                ],
                "end_user_id": "test_user_789",
                "remember": True,
                "k": 5
            },
            "expected_code": 200
        },
        {
            "name": "Empty end_user_id",
            "payload": {
                "messages": [{"role": "user", "content": "Hello"}],
                "end_user_id": "",
                "remember": False
            },
            "expected_code": 422  # Should fail with empty string
        },
        {
            "name": "Missing end_user_id",
            "payload": {
                "messages": [{"role": "user", "content": "Hello"}],
                "remember": False
            },
            "expected_code": 422  # Should fail when missing
        }
    ]
    
    print("\n=== Testing Chat Completions API with end_user_id Variations ===")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                json=test_case['payload'],
                headers={"Authorization": AUTH_TOKEN}
            )
            print(f"Status code: {response.status_code} (Expected: {test_case['expected_code']})")
            
            if response.status_code == test_case['expected_code']:
                print("✓ Success! Status code matches expected code")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Completion (first 100 chars): {result['completion'][:100]}...")
                    print(f"Sources: {len(result.get('sources', []))} sources returned")
                else:
                    print(f"Error details: {response.text[:200]}...")
            else:
                print("✗ Error! Status code does not match expected code")
                print(f"Response: {response.text[:200]}...")
        except Exception as e:
            print(f"Exception: {str(e)}")

if __name__ == "__main__":
    test_chat_completion()
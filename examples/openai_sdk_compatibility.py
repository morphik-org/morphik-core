"""
Example demonstrating OpenAI SDK compatibility with Morphik.

This example shows how to use the OpenAI SDK with Morphik as the backend,
providing seamless migration from OpenAI to Morphik while retaining
RAG capabilities and LiteLLM model support.
"""

import asyncio
import os
from openai import AsyncOpenAI

# Example usage with OpenAI SDK pointing to Morphik
async def main():
    # Initialize OpenAI client with Morphik base URL
    client = AsyncOpenAI(
        api_key=os.getenv("JWT_TOKEN", "your-morphik-jwt-token"),
        base_url=os.getenv("MORPHIK_BASE_URL", "http://localhost:8000/v1")
    )
    
    # List available models
    print("Available models:")
    models = await client.models.list()
    for model in models.data:
        print(f"- {model.id}")
    
    # Basic chat completion
    print("\n=== Basic Chat Completion ===")
    response = await client.chat.completions.create(
        model="gpt-4",  # Use your configured model from morphik.toml
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")
    
    # Chat completion with RAG (Morphik extension)
    print("\n=== Chat Completion with RAG ===")
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Tell me about the documents I've uploaded"}
        ],
        max_tokens=200,
        temperature=0.7,
        # Morphik-specific parameters
        extra_body={
            "use_rag": True,
            "folder_name": "my_documents",
            "top_k": 5
        }
    )
    
    print(f"RAG Response: {response.choices[0].message.content}")
    
    # Streaming completion
    print("\n=== Streaming Completion ===")
    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Write a short poem about AI"}
        ],
        max_tokens=100,
        temperature=0.8,
        stream=True
    )
    
    print("Streaming response:")
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")
    
    # Persistent chat session (Morphik extension)
    print("\n=== Persistent Chat Session ===")
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Remember this: my favorite color is blue"}
        ],
        extra_body={
            "chat_id": "my_persistent_chat_session"
        }
    )
    
    print(f"First message: {response.choices[0].message.content}")
    
    # Continue the conversation
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What's my favorite color?"}
        ],
        extra_body={
            "chat_id": "my_persistent_chat_session"
        }
    )
    
    print(f"Follow-up: {response.choices[0].message.content}")
    
    # Structured output (JSON mode)
    print("\n=== Structured Output ===")
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Extract key information about the following person: John Doe, 30 years old, Software Engineer at Tech Corp"}
        ],
        response_format={"type": "json_object"},
        max_tokens=150
    )
    
    print(f"Structured response: {response.choices[0].message.content}")


if __name__ == "__main__":
    asyncio.run(main())
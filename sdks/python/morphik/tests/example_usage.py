#!/usr/bin/env python
"""
Example script demonstrating basic usage of the Morphik SDK.
This can be run to verify that the SDK is working correctly.

Usage:
    python example_usage.py [--async]

Options:
    --async    Run the example using the async client
"""

import argparse
import time
from pathlib import Path


def run_sync_example():
    """Run synchronous SDK examples"""
    from morphik import Morphik

    # Get the test files directory - this script is in the tests directory
    test_docs_dir = Path(__file__).parent / "test_docs"

    print("Running Morphik SDK Sync Example")
    print("===============================")

    # Initialize the client - using default localhost:8000
    print("\n1. Initializing Morphik client...")
    db = Morphik()  # Connects to localhost:8000 by default
    print(f"   Connected to {db._logic._base_url}")

    try:
        # Ingest a text document
        print("\n2. Ingesting a text document...")
        text_doc = db.ingest_text(
            content="This is a sample document created using the Morphik SDK. "
            "It demonstrates the text ingestion capabilities.",
            filename="sdk_example.txt",
            metadata={"source": "sdk_example", "type": "text"},
        )
        print(f"   Document created with ID: {text_doc.external_id}")
        print(f"   Filename: {text_doc.filename}")
        print(f"   Metadata: {text_doc.metadata}")

        # Ingest a file
        print("\n3. Ingesting a file from disk...")
        file_path = test_docs_dir / "sample1.txt"
        file_doc = db.ingest_file(file=file_path, metadata={"source": "sdk_example", "type": "file"})
        print(f"   Document created with ID: {file_doc.external_id}")
        print(f"   Filename: {file_doc.filename}")

        # Create a folder
        print("\n4. Creating a folder...")
        folder = db.create_folder(name="sdk_example_folder", description="Example folder created by SDK")
        print(f"   Folder created with name: {folder.name}")
        print(f"   Folder ID: {folder.id}")

        # Ingest document into folder
        print("\n5. Ingesting a document into the folder...")
        folder_doc = folder.ingest_text(
            content="This document is stored in a specific folder.",
            filename="folder_example.txt",
            metadata={"source": "sdk_example", "type": "folder_doc"},
        )
        print(f"   Document created with ID: {folder_doc.external_id}")

        # Create a user scope
        print("\n6. Creating a user scope...")
        user = db.signin("sdk_example_user")
        print(f"   User scope created for: {user.end_user_id}")

        # Ingest document as user
        print("\n7. Ingesting a document as this user...")
        user_doc = user.ingest_text(
            content="This document is associated with a specific user.",
            filename="user_example.txt",
            metadata={"source": "sdk_example", "type": "user_doc"},
        )
        print(f"   Document created with ID: {user_doc.external_id}")

        # Wait for processing to complete
        print("\n8. Waiting for documents to be processed...")
        for _ in range(10):
            status = db.get_document_status(text_doc.external_id)
            if status.get("status") == "completed":
                print(f"   Document {text_doc.external_id} is now processed")
                break
            print(f"   Document status: {status.get('status')}. Waiting...")
            time.sleep(3)

        # Search using retrieve_chunks
        print("\n9. Retrieving relevant chunks...")
        chunks = db.retrieve_chunks(query="What is this document about?", filters={"source": "sdk_example"}, k=2)
        print(f"   Found {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: Score {chunk.score}")
            print(f"      Content: {chunk.content[:50]}...")

        # Query using RAG
        print("\n10. Generating a completion using RAG...")
        completion = db.query(
            query="Summarize what these documents contain",
            filters={"source": "sdk_example"},
            k=3,
            temperature=0.7,
        )
        print(f"   Completion: {completion.completion}")
        print(f"   Using {len(completion.sources)} sources")
        for i, source in enumerate(completion.sources):
            print(f"      Source {i+1}: Document {source.document_id}, Chunk {source.chunk_number}")

        # List documents
        print("\n11. Listing documents...")
        docs = db.list_documents(filters={"source": "sdk_example"})
        print(f"   Found {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"   Document {i+1}: {doc.filename} (ID: {doc.external_id})")

        # Cleanup
        print("\n12. Cleaning up test documents...")
        # Delete the documents in reverse order (won't delete folder)
        doc_ids = [
            user_doc.external_id,
            folder_doc.external_id,
            file_doc.external_id,
            text_doc.external_id,
        ]
        for doc_id in doc_ids:
            result = db.delete_document(doc_id)
            print(f"   Deleted document {doc_id}: {result.get('message', 'No message')}")

        print("\nExample completed successfully!")

    finally:
        db.close()


async def run_async_example():
    """Run asynchronous SDK examples"""
    import asyncio

    from morphik.async_ import AsyncMorphik

    # Get the test files directory - this script is in the tests directory
    test_docs_dir = Path(__file__).parent / "test_docs"

    print("Running Morphik SDK Async Example")
    print("================================")

    # Initialize the client - using default localhost:8000
    print("\n1. Initializing AsyncMorphik client...")
    async with AsyncMorphik() as db:  # Connects to localhost:8000 by default
        print(f"   Connected to {db._logic._base_url}")

        try:
            # Ingest a text document
            print("\n2. Ingesting a text document...")
            text_doc = await db.ingest_text(
                content="This is a sample document created using the Morphik SDK async client. "
                "It demonstrates the text ingestion capabilities.",
                filename="async_sdk_example.txt",
                metadata={"source": "async_sdk_example", "type": "text"},
            )
            print(f"   Document created with ID: {text_doc.external_id}")
            print(f"   Filename: {text_doc.filename}")
            print(f"   Metadata: {text_doc.metadata}")

            # Ingest a file
            print("\n3. Ingesting a file from disk...")
            file_path = test_docs_dir / "sample2.txt"
            file_doc = await db.ingest_file(file=file_path, metadata={"source": "async_sdk_example", "type": "file"})
            print(f"   Document created with ID: {file_doc.external_id}")
            print(f"   Filename: {file_doc.filename}")

            # Create a folder
            print("\n4. Creating a folder...")
            folder = await db.create_folder(
                name="async_sdk_example_folder", description="Example folder created by SDK"
            )
            print(f"   Folder created with name: {folder.name}")
            print(f"   Folder ID: {folder.id}")

            # Ingest document into folder
            print("\n5. Ingesting a document into the folder...")
            folder_doc = await folder.ingest_text(
                content="This document is stored in a specific folder using the async client.",
                filename="async_folder_example.txt",
                metadata={"source": "async_sdk_example", "type": "folder_doc"},
            )
            print(f"   Document created with ID: {folder_doc.external_id}")

            # Create a user scope
            print("\n6. Creating a user scope...")
            user = db.signin("async_sdk_example_user")
            print(f"   User scope created for: {user.end_user_id}")

            # Ingest document as user
            print("\n7. Ingesting a document as this user...")
            user_doc = await user.ingest_text(
                content="This document is associated with a specific user using the async client.",
                filename="async_user_example.txt",
                metadata={"source": "async_sdk_example", "type": "user_doc"},
            )
            print(f"   Document created with ID: {user_doc.external_id}")

            # Wait for processing to complete
            print("\n8. Waiting for documents to be processed...")
            for _ in range(10):
                status = await db.get_document_status(text_doc.external_id)
                if status.get("status") == "completed":
                    print(f"   Document {text_doc.external_id} is now processed")
                    break
                print(f"   Document status: {status.get('status')}. Waiting...")
                await asyncio.sleep(3)

            # Search using retrieve_chunks
            print("\n9. Retrieving relevant chunks...")
            chunks = await db.retrieve_chunks(
                query="What is this document about?", filters={"source": "async_sdk_example"}, k=2
            )
            print(f"   Found {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"   Chunk {i+1}: Score {chunk.score}")
                print(f"      Content: {chunk.content[:50]}...")

            # Query using RAG
            print("\n10. Generating a completion using RAG...")
            completion = await db.query(
                query="Summarize what these documents contain",
                filters={"source": "async_sdk_example"},
                k=3,
                temperature=0.7,
            )
            print(f"   Completion: {completion.completion}")
            print(f"   Using {len(completion.sources)} sources")
            for i, source in enumerate(completion.sources):
                print(f"      Source {i+1}: Document {source.document_id}, Chunk {source.chunk_number}")

            # List documents
            print("\n11. Listing documents...")
            docs = await db.list_documents(filters={"source": "async_sdk_example"})
            print(f"   Found {len(docs)} documents")
            for i, doc in enumerate(docs):
                print(f"   Document {i+1}: {doc.filename} (ID: {doc.external_id})")

            # Cleanup
            print("\n12. Cleaning up test documents...")
            # Delete the documents in reverse order (won't delete folder)
            doc_ids = [
                user_doc.external_id,
                folder_doc.external_id,
                file_doc.external_id,
                text_doc.external_id,
            ]
            for doc_id in doc_ids:
                result = await db.delete_document(doc_id)
                print(f"   Deleted document {doc_id}: {result.get('message', 'No message')}")

            print("\nAsync example completed successfully!")

        except Exception as e:
            print(f"Error in async example: {e}")
            raise


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Morphik SDK example script")
    parser.add_argument("--run-async", action="store_true", help="Run the async example")
    args = parser.parse_args()

    if args.run_async:
        # Run the async example
        import asyncio

        asyncio.run(run_async_example())
    else:
        # Run the sync example
        run_sync_example()

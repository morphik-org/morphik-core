# Morphik

A Python client for Morphik API that enables document ingestion, semantic search, and retrieval augmented generation capabilities.

## Installation

```bash
pip install morphik
```

## Usage

The SDK provides both synchronous and asynchronous clients:

### Synchronous Usage

```python
from morphik import Morphik

# Initialize client - connects to localhost:8000 by default
db = Morphik()

# Or with authentication URI (for production)
# db = Morphik("morphik://owner_id:token@api.morphik.ai")

# Ingest a text document
doc = db.ingest_text(
    content="Your document content",
    metadata={"title": "Example Document"}
)

# Ingest a file
doc = db.ingest_file(
    file="path/to/document.pdf",
    metadata={"category": "reports"}
)

# Retrieve relevant chunks
chunks = db.retrieve_chunks(
    query="Your search query",
    filters={"category": "reports"}
)

# Query with RAG
response = db.query(
    query="Summarize the key points in the document",
    filters={"category": "reports"}
)

print(response.completion)
```

### Asynchronous Usage

```python
import asyncio
from morphik.async_ import AsyncMorphik

async def main():
    # Initialize async client - connects to localhost:8000 by default
    async with AsyncMorphik() as db:

    # Or with authentication URI (for production)
    # async with AsyncMorphik("morphik://owner_id:token@api.morphik.ai") as db:
        # Ingest a text document
        doc = await db.ingest_text(
            content="Your document content",
            metadata={"title": "Example Document"}
        )

        # Query with RAG
        response = await db.query(
            query="Summarize the key points in the document",
        )

        print(response.completion)

# Run the async function
asyncio.run(main())
```

## Features

- Document ingestion (text, files, directories)
- Semantic search and retrieval
- Retrieval-augmented generation (RAG)
- Knowledge graph creation and querying
- Multi-user and multi-folder scoping
- Metadata filtering
- Document management

## Development

### Running Tests

To run the tests, first install the development dependencies:

```bash
pip install -r test_requirements.txt
```

Then run the tests:

```bash
# Run all tests (requires a running Morphik server)
pytest morphik/tests/ -v

# Run specific test modules
pytest morphik/tests/test_sync.py -v
pytest morphik/tests/test_async.py -v

# Skip tests if you don't have a running server
SKIP_LIVE_TESTS=1 pytest morphik/tests/ -v

# Specify a custom server URL for tests
MORPHIK_TEST_URL=http://custom-server:8000 pytest morphik/tests/ -v
```

### Example Usage Script

The SDK comes with an example script that demonstrates basic usage:

```bash
# Run synchronous example
python -m morphik.tests.example_usage

# Run asynchronous example
python -m morphik.tests.example_usage --async
```

The example script demonstrates:
- Text and file ingestion
- Creating folders and user scopes
- Retrieving chunks and documents
- Generating completions using RAG
- Batch operations and cleanup

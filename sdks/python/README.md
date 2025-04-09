# Morphik

A Python client for Morphik API that enables document ingestion and semantic search capabilities.

## Installation

```bash
pip install morphik
```

```python
from morphik import Morphik

# Initialize client
db = Morphik("your-api-key")

# Ingest a document
doc_id = await db.ingest_document(
    content="Your document content",
    metadata={"title": "Example Document"}
)

# Query documents
results = await db.query(
    query="Your search query",
    filters={"title": "Example Document"}
)

# Process results
for result in results:
    print(f"Content: {result.content}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```
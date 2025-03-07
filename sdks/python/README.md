# DataBridge

A Python client for DataBridge API that enables document ingestion and semantic search capabilities.

## Installation

```bash
pip install databridge-client
```

```python
from databridge import DataBridge

# Initialize client
db = DataBridge("your-api-key")

# Ingest a document
doc_id = await db.ingest_document(
    content="Your document content",
    metadata={"title": "Example Document"}
)

# Query documents
response = await db.query(
    query="Your search query",
    filters={"title": "Example Document"}
)

# Access the answer
print(f"Completion: {response.completion}")

# Access the context used for generating the answer
if response.context:
    print("Context used:")
    for i, chunk in enumerate(response.context):
        print(f"Chunk {i+1}: {chunk[:100]}...")
```
import os
from dotenv import load_dotenv
from databridge import DataBridge

# Load environment variables
load_dotenv()

# Connect to DataBridge
db = DataBridge(os.getenv("DATABRIDGE_URI"), timeout=10000, is_local=True)

# Basic text ingestion
text_doc = db.ingest_text(
    "DataBridge is an open-source database designed for AI applications that simplifies working with unstructured data.",
    metadata={"category": "tech", "author": "DataBridge"}
)
print(f"Ingested text document with ID: {text_doc.external_id}")

# Basic file ingestion
file_doc = db.ingest_file(
    "examples/assets/colpali_example.pdf",
    metadata={"category": "research", "topic": "technology"}
)
print(f"Ingested file with ID: {file_doc.external_id}")

# Basic retrieval
chunks = db.retrieve_chunks(
    query="What is DataBridge?",
    k=3
)

print("Retrieved chunks:")
for chunk in chunks:
    print(f"Content: {chunk.content[:100]}...")
    print(f"Score: {chunk.score}\n")

# Basic query with RAG
response = db.query("What is DataBridge and what is it used for?")
print("Query response:")
print(response.completion)
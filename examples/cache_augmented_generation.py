import os
from dotenv import load_dotenv
from morphik import Morphik

# Load environment variables
load_dotenv()

# Connect to Morphik
db = Morphik(os.getenv("MORPHIK_URI"), timeout=10000, is_local=True)

# Sample document for demonstration
long_document = """
# Artificial Intelligence and Machine Learning: A Comprehensive Overview

## Introduction
Artificial Intelligence (AI) represents the simulation of human intelligence in machines programmed to think and learn like humans. Machine Learning (ML) is a subset of AI focused on building systems that learn from data.

## Historical Development
The concept of AI dates back to ancient history with myths and stories about artificial beings. However, the formal field began in 1956 at the Dartmouth Conference where the term "Artificial Intelligence" was coined.

## Types of Artificial Intelligence
1. **Narrow AI**: Designed for specific tasks (e.g., voice assistants)
2. **General AI**: Hypothetical AI with human-level intelligence across various domains
3. **Superintelligent AI**: AI surpassing human intelligence, mostly theoretical

## Machine Learning Approaches
1. **Supervised Learning**: Models trained on labeled data
2. **Unsupervised Learning**: Finding patterns in unlabeled data
3. **Reinforcement Learning**: Learning through trial and error with rewards
"""

# Ingest the document
print("Ingesting document...")
doc = db.ingest_text(
    long_document,
    metadata={"category": "technology", "topic": "AI"}
)
print(f"Ingested document with ID: {doc.external_id}")

# Create a cache
print("\nCreating cache...")
cache_result = db.create_cache(
    name="ai_overview_cache",
    model="llama2",
    gguf_file="llama-2-7b-chat.Q4_K_M.gguf",
    docs=[doc.external_id]  # Include our document in the cache
)
print("Cache created successfully")

# Get a reference to the cache
cache = db.get_cache("ai_overview_cache")

# Update the cache to process documents
print("Updating cache...")
cache.update()
print("Cache updated")

# Query the cache directly
print("\nQuerying the cache...")
response = cache.query(
    "What are the different types of artificial intelligence?",
    max_tokens=200,
    temperature=0.7
)
print("Response from cache:")
print(response.completion)

# Add more documents to the cache (if needed)
additional_doc = db.ingest_text(
    "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize rewards.",
    metadata={"category": "technology", "topic": "AI", "subtopic": "reinforcement learning"}
)

cache.add_docs([additional_doc.external_id])
print("\nAdded additional document to cache")

# Update cache again to process the new document
cache.update()
print("Cache updated with new document")
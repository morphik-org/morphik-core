import os
from dotenv import load_dotenv
from databridge import DataBridge
from databridge.rules import MetadataExtractionRule, NaturalLanguageRule
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Connect to DataBridge
db = DataBridge(os.getenv("DATABRIDGE_URI"), timeout=10000, is_local=True)

# Define sample text with information we want to extract
sample_text = """
Report: Q2 Financial Analysis
Date: June 30, 2023
Author: John Smith, Chief Financial Analyst
Department: Finance

CONFIDENTIAL - INTERNAL USE ONLY

The second quarter showed a 15% increase in revenue compared to Q1.
Key metrics:
- Total Revenue: $5.2M
- Operating Expenses: $3.1M
- Net Profit: $2.1M

Contact john.smith@example.com for questions.
"""

# Define schema for metadata extraction using Pydantic
class DocumentInfo(BaseModel):
    title: str
    date: str
    author: str
    department: str

# Define rules using the rules builder
# 1. Metadata extraction rule
metadata_rule = MetadataExtractionRule(schema=DocumentInfo)

# 2. Natural language rule to remove PII
pii_removal_rule = NaturalLanguageRule(
    prompt="Remove all personally identifiable information including names, email addresses, and phone numbers."
)

# 3. Natural language rule to summarize
summary_rule = NaturalLanguageRule(
    prompt="Summarize this document in one paragraph focusing on the financial results."
)

# Combine rules
rules = [metadata_rule, pii_removal_rule, summary_rule]

# Ingest document with rules
print("Ingesting document with rules...")
doc = db.ingest_text(
    sample_text,
    rules=rules,
    metadata={"category": "financial"}
)

print(f"Ingested document with ID: {doc.external_id}")

# Check the extracted metadata
print("\nExtracted metadata:")
for key, value in doc.metadata.items():
    print(f"  {key}: {value}")

# Retrieve the transformed document
chunks = db.retrieve_chunks(
    query="Financial results",
    filters={"document_id": doc.external_id},
    k=1
)

print("\nTransformed document (with PII removed):")
if chunks:
    print(chunks[0].content)

# Use rules with file ingestion
print("\nDefining rules for file ingestion...")

# Rules can also be defined using dictionaries
file_rules = [
    {
        "type": "metadata_extraction",
        "schema": {
            "title": "string",
            "author": "string", 
            "company": "string",
            "year": "number"
        }
    },
    {
        "type": "natural_language",
        "prompt": "Classify this document as either 'technical', 'financial', or 'legal'."
    }
]

# Try to ingest a file with these rules
file_doc = db.ingest_file(
    "examples/assets/colpali_example.pdf",
    rules=file_rules
)

print(f"Ingested file with rules, ID: {file_doc.external_id}")
print(f"Classification: {file_doc.metadata.get('classification', 'Not classified')}")
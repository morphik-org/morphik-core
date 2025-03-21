import os
from dotenv import load_dotenv
from databridge import DataBridge

# Load environment variables
load_dotenv()

# Connect to DataBridge
db = DataBridge(os.getenv("DATABRIDGE_URI"), timeout=10000, is_local=True)

# First, ensure we have some documents to work with
sample_texts = [
    {
        "text": "AI technology is advancing rapidly with applications in healthcare. Machine learning models are being used to predict patient outcomes.",
        "metadata": {"category": "tech", "domain": "healthcare"}
    },
    {
        "text": "Cloud computing enables AI systems to scale. AWS, Azure, and Google Cloud provide infrastructure for machine learning.",
        "metadata": {"category": "tech", "domain": "cloud"}
    },
    {
        "text": "Electronic health records are being analyzed with natural language processing to improve diagnoses.",
        "metadata": {"category": "tech", "domain": "nlp"}
    }
]

# Ingest the documents
doc_ids = []
for item in sample_texts:
    doc = db.ingest_text(item["text"], metadata=item["metadata"])
    doc_ids.append(doc.external_id)
    print(f"Ingested document with ID: {doc.external_id}")

# Create a knowledge graph from the documents
print("Creating knowledge graph...")
graph = db.create_graph(
    name="tech_healthcare_graph",
    filters={"category": "tech"}
)
print(f"Created graph with name: {graph.name}")

# Query using the knowledge graph
response = db.query(
    "How is AI technology being used in healthcare?",
    graph_name="tech_healthcare_graph",
    hop_depth=2  # Consider connections up to 2 hops away
)

print("Graph-enhanced query response:")
print(response.completion)

# Example of using a graph with path information
response_with_paths = db.query(
    "What technologies are used for analyzing electronic health records?",
    graph_name="tech_healthcare_graph",
    hop_depth=2,
    include_paths=True
)

# If path information is included, it will be in the response metadata
if response_with_paths.metadata and "graph" in response_with_paths.metadata:
    print("\nGraph paths found:")
    for path in response_with_paths.metadata["graph"]["paths"]:
        print(" -> ".join(path))
import os

from dotenv import load_dotenv
from morphik import Morphik
from morphik.models import (EntityExtractionExample,
                            EntityExtractionPromptOverride,
                            EntityResolutionExample,
                            EntityResolutionPromptOverride,
                            GraphPromptOverrides, QueryPromptOverride,
                            QueryPromptOverrides)

# Load environment variables
load_dotenv()

# Connect to Morphik
db = Morphik(os.getenv("MORPHIK_URI"), timeout=10000, is_local=True)

# Ingest some sample medical documents
medical_texts = [
    {
        "text": "Patients with Type 2 Diabetes often show increased insulin resistance. Treatment options include metformin and lifestyle changes.",
        "metadata": {"category": "medical", "specialty": "endocrinology"},
    },
    {
        "text": "Hypertension (high blood pressure) is a common comorbidity of diabetes. ACE inhibitors are frequently prescribed to manage blood pressure.",
        "metadata": {"category": "medical", "specialty": "cardiology"},
    },
    {
        "text": "Studies show that regular exercise can improve glucose control in diabetic patients and reduce the risk of cardiovascular disease.",
        "metadata": {"category": "medical", "specialty": "research"},
    },
]

# Ingest documents
doc_ids = []
for item in medical_texts:
    doc = db.ingest_text(item["text"], metadata=item["metadata"])
    doc_ids.append(doc.external_id)
    print(f"Ingested document with ID: {doc.external_id}")

# Example 1: Basic Query with Prompt Override
print("\nExample 1: Basic Query with Prompt Override")
basic_query_override = QueryPromptOverrides(
    query=QueryPromptOverride(
        prompt_template="Respond as if you are a medical professional. Answer the following question based on the provided context: {question}"
    )
)

response = db.query("What treatments are available for diabetes?", prompt_overrides=basic_query_override)

print("Response with medical professional prompt:")
print(response.completion)

# Example 2: Query without override for comparison
print("\nExample 2: Same Query without Override")
standard_response = db.query("What treatments are available for diabetes?")
print("Standard response:")
print(standard_response.completion)

# Example 3: Create a knowledge graph with customized entity extraction
print("\nExample 3: Knowledge Graph with Customized Entity Extraction")
graph_overrides = GraphPromptOverrides(
    entity_extraction=EntityExtractionPromptOverride(
        examples=[
            EntityExtractionExample(label="Diabetes", type="CONDITION"),
            EntityExtractionExample(label="Metformin", type="MEDICATION"),
            EntityExtractionExample(label="Hypertension", type="CONDITION"),
            EntityExtractionExample(label="ACE inhibitors", type="MEDICATION"),
        ]
    ),
    entity_resolution=EntityResolutionPromptOverride(
        examples=[
            EntityResolutionExample(canonical="Diabetes Mellitus", variants=["Diabetes", "Type 2 Diabetes", "T2DM"]),
            EntityResolutionExample(canonical="Hypertension", variants=["High Blood Pressure", "HTN", "Elevated BP"]),
        ]
    ),
)

graph = db.create_graph(name="medical_conditions_graph", documents=doc_ids, prompt_overrides=graph_overrides)

print(f"Created graph with {len(graph.entities)} entities and {len(graph.relationships)} relationships")

# Example 4: Query using the customized graph
print("\nExample 4: Query Using the Customized Graph")
graph_response = db.query(
    "How are diabetes and hypertension related?",
    graph_name="medical_conditions_graph",
    hop_depth=2,
    include_paths=True,
)

print("Response using knowledge graph:")
print(graph_response.completion)

# Print the relationship paths if available
if graph_response.metadata and "graph" in graph_response.metadata:
    print("\nRelationship paths:")
    for path in graph_response.metadata["graph"]["paths"]:
        print(" -> ".join(path))

# Example 5: Using dictionary for prompt overrides
print("\nExample 5: Using Dictionary for Prompt Overrides")
dict_override = {
    "query": {"prompt_template": "Summarize the information in a bulleted list format, focusing on: {question}"}
}

dict_response = db.query(
    "What are the connections between diabetes, medications, and exercise?",
    prompt_overrides=dict_override,
)

print("Response with dictionary-based prompt override:")
print(dict_response.completion)

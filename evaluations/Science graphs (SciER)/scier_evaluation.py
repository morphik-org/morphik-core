#!/usr/bin/env python3
"""
SciER Graph Creation Script for Morphik

This script creates a knowledge graph from the SciER dataset.
It ingests the documents and creates a graph using custom prompt overrides.
"""

import os
import uuid
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

from morphik import Morphik
from morphik.models import (
    EntityExtractionPromptOverride,
    EntityExtractionExample,
    GraphPromptOverrides,
)

# Import SciER data loader
from data_loader import load_jsonl

# Load environment variables
load_dotenv()


def setup_morphik_client() -> Morphik:
    """Initialize and return a Morphik client."""
    # Connect to Morphik (adjust parameters as needed)
    return Morphik(timeout=300000, is_local=True)


def load_scier_data(dataset_path: str = "test.jsonl", limit: int = None) -> List[Dict]:
    """
    Load SciER dataset from the specified JSONL file.

    Args:
        dataset_path: Path to the JSONL file
        limit: Maximum number of records to load (None for all)

    Returns:
        List of dataset records
    """
    data = load_jsonl(dataset_path)
    if limit:
        data = data[:limit]
    return data


def prepare_text_for_ingestion(records: List[Dict]) -> List[Dict]:
    """
    Prepare SciER records for ingestion into Morphik.

    Args:
        records: List of SciER records

    Returns:
        List of dictionaries ready for ingestion
    """
    documents = []

    # Group records by doc_id to create complete documents
    doc_groups = defaultdict(list)
    for record in records:
        doc_groups[record["doc_id"]].append(record)

    # Convert grouped records to documents
    for doc_id, records in doc_groups.items():
        text = "\n".join([record["sentence"] for record in records])

        # Collect all entities and relations for ground truth
        all_entities = []
        all_relations = []
        for record in records:
            all_entities.extend(record["ner"])
            all_relations.extend(record["rel"])

        documents.append(
            {
                "text": text,
                "metadata": {
                    "doc_id": doc_id,
                    "ground_truth_entities": all_entities,
                    "ground_truth_relations": all_relations,
                },
            }
        )

    return documents


def create_graph_extraction_override(entity_types: List[str]) -> EntityExtractionPromptOverride:
    """
    Create graph extraction prompt override with examples for both entities and relations.

    Args:
        entity_types: List of entity types (Dataset, Method, Task)

    Returns:
        EntityExtractionPromptOverride object
    """
    examples = []

    if "Dataset" in entity_types:
        examples.extend(
            [
                EntityExtractionExample(label="ImageNet", type="Dataset"),
                EntityExtractionExample(label="CIFAR-10", type="Dataset"),
                EntityExtractionExample(label="MNIST", type="Dataset"),
                EntityExtractionExample(label="Penn TreeBank", type="Dataset"),
                EntityExtractionExample(label="SQuAD", type="Dataset"),
                EntityExtractionExample(label="MultiNLI", type="Dataset"),
            ]
        )

    if "Method" in entity_types:
        examples.extend(
            [
                # General models
                EntityExtractionExample(label="Convolutional Neural Network", type="Method"),
                EntityExtractionExample(label="Random Forest", type="Method"),
                # Architecture-specific models from SciER
                EntityExtractionExample(label="BERT", type="Method"),
                EntityExtractionExample(label="Transformer", type="Method"),
                EntityExtractionExample(label="LSTM", type="Method"),
                EntityExtractionExample(label="Bidirectional LSTM", type="Method"),
                EntityExtractionExample(label="self-attentive models", type="Method"),
                EntityExtractionExample(label="seq2seq", type="Method"),
                # Components
                EntityExtractionExample(label="attention mechanism", type="Method"),
                EntityExtractionExample(label="feature extraction mechanisms", type="Method"),
            ]
        )

    if "Task" in entity_types:
        examples.extend(
            [
                # General tasks
                EntityExtractionExample(label="Image Classification", type="Task"),
                EntityExtractionExample(label="Named Entity Recognition", type="Task"),
                # NLP tasks from SciER
                EntityExtractionExample(label="Machine Translation", type="Task"),
                EntityExtractionExample(label="neural machine translation", type="Task"),
                EntityExtractionExample(label="sentiment analysis", type="Task"),
                EntityExtractionExample(label="entailment", type="Task"),
                EntityExtractionExample(label="text classification", type="Task"),
                EntityExtractionExample(label="natural language processing", type="Task"),
                EntityExtractionExample(label="sequence-to-sequence problems", type="Task"),
                EntityExtractionExample(label="NLP", type="Task"),
            ]
        )

    # Simplest version - bare standard placeholders
    prompt_template = """
Your task is to carefully read the following scientific text and extract specific information.
You need to extract:
1.  **Entities:** Identify any mentions of Datasets, Methods, and Tasks. Use the entity examples provided below to understand what to look for, that is a very small list, there are many many entities.
2.  **Relationships:** Identify relationships *between the extracted entities* based on the information stated in the text. Use only the relationship types defined below.

**Entity Examples (this is a very brief list, there are many many entities):**
{examples}

**Relationship Information:**
Desired Relationship Types (only extract these relationships, nothing else, there are a lot of relationships, be nuanced and careful, think hard about how entities relate to each other):
- Used-For: [Method/Dataset] is used for [Task]
- Feature-Of: [Feature] is a feature of [Method/Task]
- Hyponym-Of: [Specific] is a type of [General]
- Part-Of: [Component] is part of [System]
- Compare: [Entity A] is compared to [Entity B]
- Evaluate-For: [Method] is evaluated for [Metric/Task]
- Conjunction: [Entity A] is mentioned together with [Entity B] without a specific relation
- Evaluate-On: [Method] is evaluated on [Dataset]
- Synonym-Of: [Entity A] is the same as [Entity B]

**Instructions:**
- Extract entities first, identifying their label (the text mention) and type (Dataset, Method, or Task).
- Then, extract relationships between the entities you found. The 'source' and 'target' of the relationship MUST be the exact entity labels you extracted.
- Only extract information explicitly mentioned in the text. Do not infer or add outside knowledge.
- Format your entire output as a single JSON object containing two keys: "entities" (a list of entity objects) and "relationships" (a list of relationship objects).

**Text to analyze:**
{content}
"""

    return EntityExtractionPromptOverride(prompt_template=prompt_template, examples=examples)


def create_graph(
    db: Morphik, documents: List[Dict], model_name: str, run_id: str
) -> Tuple[List[str], Dict]:
    """
    Create a knowledge graph from the documents.

    Args:
        db: Morphik client
        documents: List of documents to ingest
        model_name: Name of the model being used (for tracking)
        run_id: Unique identifier for this run

    Returns:
        Tuple of (list of document IDs, graphs dict)
    """
    print(f"\n=== Creating graph with {model_name} model ===")

    # Ingest documents
    doc_ids = []
    for doc in tqdm(documents, desc="Ingesting documents"):
        # Add metadata for tracking
        doc["metadata"]["evaluation_run_id"] = run_id
        doc["metadata"]["model"] = model_name

        # Ingest the document
        result = db.ingest_text(doc["text"], metadata=doc["metadata"])
        doc_ids.append(result.external_id)

    # Create graph extraction override (which includes both entity and relationship instructions)
    entity_extraction_override = create_graph_extraction_override(["Dataset", "Method", "Task"])

    # Wrap the combined override correctly for the API
    graph_overrides = GraphPromptOverrides(entity_extraction=entity_extraction_override)

    # Create a knowledge graph with overrides
    print("Creating knowledge graph with prompt overrides...")
    graph = db.create_graph(
        name=f"scier_{model_name}_{run_id}", documents=doc_ids, prompt_overrides=graph_overrides
    )

    print(
        f"Created graph with {len(graph.entities)} entities and {len(graph.relationships)} relationships"
    )

    return doc_ids, {"graph": graph}


def main():
    """Main function to create a graph from the SciER dataset."""
    parser = argparse.ArgumentParser(description="SciER Graph Creation Script for Morphik")
    parser.add_argument(
        "--limit", type=int, default=57, help="Maximum number of documents to process (default: 57)"
    )
    parser.add_argument(
        "--run-id", type=str, default=None, help="Unique run identifier (default: auto-generated)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the currently configured model (default: auto-detected)",
    )

    args = parser.parse_args()

    # Generate run ID if not provided
    run_id = args.run_id or str(uuid.uuid4())[:8]

    # Auto-detect or use provided model name
    model_name = args.model_name or "default_model"

    print(f"Running graph creation for model: {model_name}")
    print(f"Run ID: {run_id}")

    # Initialize Morphik client
    db = setup_morphik_client()

    # Load SciER dataset
    scier_data = load_scier_data("test.jsonl", limit=args.limit)
    print(f"Loaded {len(scier_data)} records")

    # Prepare documents for ingestion
    documents = prepare_text_for_ingestion(scier_data)
    print(f"Prepared {len(documents)} documents for ingestion")

    # Create the graph
    doc_ids, graphs = create_graph(db, documents, model_name, run_id)

    # Print graph name for evaluation
    graph_name = f"scier_{model_name}_{run_id}"
    print(f"\nGraph creation complete! Created graph: {graph_name}")
    print(
        f"To evaluate this graph, run: python evaluate_result.py --graph-name {graph_name}"
    )


if __name__ == "__main__":
    main()

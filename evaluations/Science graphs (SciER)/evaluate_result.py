#!/usr/bin/env python3
"""
SciER Evaluation Script for Morphik - OpenAI Embeddings Version

This script evaluates an existing Morphik graph against the SciER dataset
using OpenAI embeddings for semantic similarity calculations.
"""

import os
import json
import uuid
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
import requests
from scipy.spatial.distance import cosine
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time

from morphik import Morphik

# Import SciER data loader
from data_loader import load_jsonl

# Load environment variables
load_dotenv()


class OpenAIEmbedding:
    """
    OpenAI embedding similarity calculator.
    A faster alternative to SciBERT for computing semantic similarity.
    Uses text-embedding-3-small model.
    """

    def __init__(
        self,
        model_name="text-embedding-3-small",
        threshold=0.70,
        api_base="https://api.openai.com/v1",
        cache_size=10000,
        batch_size=20,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.api_base = api_base
        self.embedding_cache = {}  # Cache to store embeddings
        self.cache_size = cache_size
        self.batch_size = batch_size

        # Get OpenAI API key from environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Rate limiting parameters
        self.requests_per_minute = 3500  # Adjust based on your OpenAI rate limits
        self.min_time_between_requests = 60.0 / self.requests_per_minute
        self.last_request_time = 0

    def get_embedding(self, text):
        """Get embeddings for a text string using OpenAI API."""
        if not text.strip():
            # Return a zero vector for empty text
            return np.zeros(1536)  # text-embedding-3-small dimension

        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_time_between_requests:
            wait_time = self.min_time_between_requests - time_since_last_request
            time.sleep(wait_time)

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            response = requests.post(
                f"{self.api_base}/embeddings",
                headers=headers,
                json={"model": self.model_name, "input": text, "encoding_format": "float"},
            )

            self.last_request_time = time.time()

            if response.status_code != 200:
                print(f"Error from OpenAI API: {response.text}")
                return np.zeros(1536)

            data = response.json()
            embedding = np.array(data["data"][0]["embedding"])

            # Cache the embedding
            if len(self.embedding_cache) < self.cache_size:
                self.embedding_cache[text] = embedding

            return embedding

        except Exception as e:
            print(f"Exception when calling OpenAI API: {e}")
            return np.zeros(1536)

    def get_embeddings_batch(self, texts):
        """Get embeddings for multiple texts in batch."""
        embeddings = []
        texts_to_process = []
        indices_to_process = []

        # First check cache
        for i, text in enumerate(texts):
            if not text.strip():
                embeddings.append(np.zeros(1536))
            elif text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)
                embeddings.append(None)  # Placeholder

        # Process uncached texts in smaller batches
        if texts_to_process:
            for i in range(0, len(texts_to_process), self.batch_size):
                batch = texts_to_process[i : i + self.batch_size]
                batch_indices = indices_to_process[i : i + self.batch_size]

                # OpenAI supports batching directly in the API
                try:
                    # Rate limiting
                    current_time = time.time()
                    time_since_last_request = current_time - self.last_request_time
                    if time_since_last_request < self.min_time_between_requests:
                        wait_time = self.min_time_between_requests - time_since_last_request
                        time.sleep(wait_time)

                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = requests.post(
                        f"{self.api_base}/embeddings",
                        headers=headers,
                        json={"model": self.model_name, "input": batch, "encoding_format": "float"},
                    )

                    self.last_request_time = time.time()

                    if response.status_code != 200:
                        print(f"Error from OpenAI API: {response.text}")
                        batch_embeddings = [np.zeros(1536) for _ in batch]
                    else:
                        data = response.json()
                        # Sort by index as OpenAI returns in the same order as input
                        batch_embeddings = [np.array(item["embedding"]) for item in data["data"]]

                    # Update embeddings and cache
                    for j, embedding in enumerate(batch_embeddings):
                        idx = batch_indices[j]
                        text = batch[j]
                        embeddings[idx] = embedding
                        if len(self.embedding_cache) < self.cache_size:
                            self.embedding_cache[text] = embedding

                except Exception as e:
                    print(f"Exception when batch calling OpenAI API: {e}")
                    # Fall back to individual API calls
                    with ThreadPoolExecutor(max_workers=min(len(batch), 5)) as executor:
                        batch_embeddings = list(executor.map(self.get_embedding, batch))

                    # Update embeddings and cache
                    for j, embedding in enumerate(batch_embeddings):
                        idx = batch_indices[j]
                        text = batch[j]
                        embeddings[idx] = embedding
                        if len(self.embedding_cache) < self.cache_size:
                            self.embedding_cache[text] = embedding

        return embeddings

    def compute_similarity(self, text1, text2):
        """Compute cosine similarity between two texts."""
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)

        # Compute cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(embedding1, embedding2)
        return similarity

    def compute_similarity_with_embeddings(self, embedding1, embedding2):
        """Compute cosine similarity between two pre-computed embeddings."""
        # Compute cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(embedding1, embedding2)
        return similarity

    def are_semantically_similar(self, text1, text2):
        """Check if two texts are semantically similar based on threshold."""
        similarity = self.compute_similarity(text1, text2)
        return similarity >= self.threshold, similarity

    def compute_similarities_batch(self, text_pairs):
        """
        Compute similarities for multiple text pairs in parallel.

        Args:
            text_pairs: List of (text1, text2) tuples

        Returns:
            List of (is_similar, similarity) tuples
        """
        # Extract all unique texts
        all_texts = []
        text_indices = {}

        for text1, text2 in text_pairs:
            if text1 not in text_indices:
                text_indices[text1] = len(all_texts)
                all_texts.append(text1)
            if text2 not in text_indices:
                text_indices[text2] = len(all_texts)
                all_texts.append(text2)

        # Get embeddings for all texts in batch
        all_embeddings = self.get_embeddings_batch(all_texts)

        # Compute similarities
        results = []
        for text1, text2 in text_pairs:
            embedding1 = all_embeddings[text_indices[text1]]
            embedding2 = all_embeddings[text_indices[text2]]
            similarity = self.compute_similarity_with_embeddings(embedding1, embedding2)
            is_similar = similarity >= self.threshold
            results.append((is_similar, similarity))

        return results


def setup_morphik_client() -> Morphik:
    """Initialize and return a Morphik client."""
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


def prepare_text_for_evaluation(records: List[Dict]) -> List[Dict]:
    """
    Prepare SciER records for evaluation.

    Args:
        records: List of SciER records

    Returns:
        List of dictionaries with text and ground truth
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


def load_existing_graph(db: Morphik, graph_name: str) -> Dict:
    """
    Load an existing graph by name.

    Args:
        db: Morphik client
        graph_name: Name of the graph to load

    Returns:
        Dictionary containing the graph
    """
    print(f"Loading existing graph: {graph_name}")

    # List all graphs and find the one with the specified name
    graphs = db.list_graphs()

    target_graph = None
    for graph in graphs:
        if graph.name == graph_name:
            target_graph = graph
            break

    if target_graph is None:
        raise ValueError(
            f"Graph '{graph_name}' not found. Available graphs: {[g.name for g in graphs]}"
        )

    print(
        f"Found graph with {len(target_graph.entities)} entities and {len(target_graph.relationships)} relationships"
    )

    return {"graph": target_graph}


def evaluate_entity_extraction(
    ground_truth: List[List],
    extracted_entities: List[Dict],
    similarity_calculator,
    entity_type_match_required: bool = True,
    batch_size: int = 100,
    max_workers: int = None,
) -> Dict[str, float]:
    """
    Evaluate entity extraction performance using semantic similarity with parallelization.

    Args:
        ground_truth: List of ground truth entities from SciER
        extracted_entities: List of entities extracted by Morphik
        similarity_calculator: Similarity calculator instance
        entity_type_match_required: Whether entity types must match
        batch_size: Size of batches for parallel processing
        max_workers: Maximum number of worker processes (defaults to CPU count)

    Returns:
        Dict with precision, recall, and F1 metrics
    """
    # Set default max_workers if not provided
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    # Process ground truth
    gt_entities = []
    for entity in ground_truth:
        entity_text = entity[0].lower()
        entity_type = entity[1]
        gt_entities.append((entity_text, entity_type))

    # Process extracted entities
    extracted = []
    for entity in extracted_entities:
        entity_text = entity.label.lower()
        entity_type = entity.type
        extracted.append((entity_text, entity_type))

    print(
        f"Processing {len(gt_entities)} ground truth entities against {len(extracted)} extracted entities"
    )

    # Prepare all valid entity comparisons based on type matching
    comparisons = []
    for i, gt_entity in enumerate(gt_entities):
        gt_text, gt_type = gt_entity
        for j, ext_entity in enumerate(extracted):
            ext_text, ext_type = ext_entity
            # Skip if entity type doesn't match (if required)
            if entity_type_match_required and gt_type != ext_type:
                continue
            comparisons.append((i, j, gt_text, ext_text))

    print(f"Generated {len(comparisons)} comparisons to process")

    # Process in batches
    gt_best_matches = {}  # gt_index -> (ext_index, score)

    for i in tqdm(range(0, len(comparisons), batch_size), desc="Processing entity batches"):
        batch = comparisons[i : i + batch_size]

        # Extract text pairs for batch processing
        text_pairs = [(comp[2], comp[3]) for comp in batch]

        # Compute similarities in batch
        similarity_results = similarity_calculator.compute_similarities_batch(text_pairs)

        # Process results
        for idx, ((gt_idx, ext_idx, _, _), (is_similar, similarity)) in enumerate(
            zip(batch, similarity_results)
        ):
            if is_similar:
                # Update best match if it's better than the current one
                if gt_idx not in gt_best_matches or similarity > gt_best_matches[gt_idx][1]:
                    gt_best_matches[gt_idx] = (ext_idx, similarity)

    # Extract matched entities
    ext_matched = set(match[0] for match in gt_best_matches.values())

    # Count matches
    true_positives = len(gt_best_matches)
    false_negatives = len(gt_entities) - true_positives
    false_positives = len(extracted) - len(ext_matched)

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Create a mapping of ground truth texts to extracted entity IDs for relation evaluation
    entity_match_map = {}
    for i, (ext_idx, _) in gt_best_matches.items():
        gt_text, _ = gt_entities[i]
        ext_id = (
            extracted_entities[ext_idx].id if hasattr(extracted_entities[ext_idx], "id") else None
        )
        if ext_id:
            entity_match_map[gt_text] = ext_id

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ground_truth_count": len(gt_entities),
        "extracted_count": len(extracted),
        "entity_match_map": entity_match_map,
    }


def evaluate_relation_extraction(
    ground_truth: List[List],
    extracted_relations: List,
    entity_match_map: Dict[str, str],
    similarity_calculator,
    relation_type_match_required: bool = True,
    batch_size: int = 100,
    max_workers: int = None,
) -> Dict[str, float]:
    """
    Evaluate relation extraction performance using semantic similarity with parallelization.

    Args:
        ground_truth: List of ground truth relations from SciER
        extracted_relations: List of relations extracted by Morphik
        entity_match_map: Mapping from ground truth entity text to extracted entity ID
        similarity_calculator: Similarity calculator instance
        relation_type_match_required: Whether relation types must match
        batch_size: Size of batches for parallel processing
        max_workers: Maximum number of worker processes (defaults to CPU count)

    Returns:
        Dict with precision, recall, and F1 metrics
    """
    # Set default max_workers if not provided
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    # Process ground truth
    gt_relations = []
    for relation in ground_truth:
        source = relation[0].lower()
        relation_type = relation[1]
        target = relation[2].lower()
        gt_relations.append((source, relation_type, target))

    # Reverse map from entity ID to ground truth text
    id_to_gt_text = {entity_id: gt_text for gt_text, entity_id in entity_match_map.items()}

    # Debug output to understand entity mappings
    print(f"Entity mapping size: {len(entity_match_map)}")
    print(f"Entity reverse mapping size: {len(id_to_gt_text)}")

    # Process extracted relations
    extracted_rel_tuples = []
    skipped_relations = 0

    # Get unique entity IDs from relationships to see what might be missing
    all_source_ids = set()
    all_target_ids = set()

    for relation in extracted_relations:
        try:
            if hasattr(relation, "source_id") and hasattr(relation, "target_id"):
                source_id = relation.source_id
                target_id = relation.target_id
                relation_type = relation.type

                # Track all source and target IDs for debugging
                all_source_ids.add(source_id)
                all_target_ids.add(target_id)

                # Don't skip missing entities, we'll use direct text comparison instead
                source_text = id_to_gt_text.get(source_id, None)
                target_text = id_to_gt_text.get(target_id, None)

                # If we have direct mappings, use them
                if source_text is not None and target_text is not None:
                    extracted_rel_tuples.append((source_text, relation_type, target_text))
                # Otherwise, try to get the source and target directly from the entity object
                else:
                    try:
                        # Try to get the source and target texts from their labels
                        source_entity = next((e for e in relation.source_entity if e), None)
                        target_entity = next((e for e in relation.target_entity if e), None)

                        if (
                            source_entity
                            and target_entity
                            and hasattr(source_entity, "label")
                            and hasattr(target_entity, "label")
                        ):
                            source_label = source_entity.label.lower()
                            target_label = target_entity.label.lower()
                            extracted_rel_tuples.append((source_label, relation_type, target_label))
                        else:
                            skipped_relations += 1
                    except Exception as inner_e:
                        skipped_relations += 1
                        print(f"Error extracting entity labels: {inner_e}")

        except (AttributeError, KeyError, TypeError) as e:
            print(f"Error processing relation: {relation}. Error: {e}")

    print(
        f"Processing {len(gt_relations)} ground truth relations against {len(extracted_rel_tuples)} extracted relations"
    )
    print(f"Skipped {skipped_relations} relations due to missing entity mappings")
    print(f"Total unique source IDs: {len(all_source_ids)}, target IDs: {len(all_target_ids)}")

    # Debug: Print some sample ground truth relations
    if gt_relations:
        print("Sample ground truth relations:")
        for i, rel in enumerate(gt_relations[:5]):
            print(f"  {i}: {rel}")

    # Debug: Print some sample extracted relations
    if extracted_rel_tuples:
        print("Sample extracted relations:")
        for i, rel in enumerate(extracted_rel_tuples[:5]):
            print(f"  {i}: {rel}")

    # Prepare relation comparisons, but don't filter by entity map membership
    comparisons = []
    for i, gt_relation in enumerate(gt_relations):
        gt_source, gt_rel_type, gt_target = gt_relation

        for j, ext_relation in enumerate(extracted_rel_tuples):
            ext_source, ext_rel_type, ext_target = ext_relation

            # Skip if relation type doesn't match (if required)
            if relation_type_match_required and gt_rel_type != ext_rel_type:
                continue

            comparisons.append((i, j, gt_source, ext_source, gt_target, ext_target))

    print(f"Generated {len(comparisons)} relation comparisons to process")

    # Process in batches
    matched_gt_indices = set()
    matched_ext_indices = set()
    best_scores = {}  # gt_idx -> (ext_idx, score)

    for i in tqdm(range(0, len(comparisons), batch_size), desc="Processing relation batches"):
        batch = comparisons[i : i + batch_size]

        # Create batches for source and target comparisons
        source_pairs = [(comp[2], comp[3]) for comp in batch]
        target_pairs = [(comp[4], comp[5]) for comp in batch]

        # Compute source similarities in batch
        source_results = similarity_calculator.compute_similarities_batch(source_pairs)

        # Compute target similarities in batch
        target_results = similarity_calculator.compute_similarities_batch(target_pairs)

        # Process results
        for idx, (
            (gt_idx, ext_idx, _, _, _, _),
            (source_is_similar, source_sim),
            (target_is_similar, target_sim),
        ) in enumerate(zip(batch, source_results, target_results)):

            if source_is_similar and target_is_similar:
                match_score = (source_sim + target_sim) / 2

                # Update best match if it's better than current one
                if gt_idx not in best_scores or match_score > best_scores[gt_idx][1]:
                    best_scores[gt_idx] = (ext_idx, match_score)

    # Extract matched indices
    for gt_idx, (ext_idx, _) in best_scores.items():
        matched_gt_indices.add(gt_idx)
        matched_ext_indices.add(ext_idx)

    # Count matches
    true_positives = len(matched_gt_indices)
    false_negatives = len(gt_relations) - true_positives
    false_positives = len(extracted_rel_tuples) - len(matched_ext_indices)

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ground_truth_count": len(gt_relations),
        "extracted_count": len(extracted_rel_tuples),
    }


def evaluate_graph(
    db: Morphik,
    documents: List[Dict],
    graph: Any,
    model: str,
    similarity_calculator,
    batch_size: int = 100,
    max_workers: int = None,
) -> Dict[str, Dict]:
    """
    Evaluate graph extraction against ground truth using semantic similarity.

    Args:
        db: Morphik client
        documents: Original documents with ground truth
        graph: Graph to evaluate
        model: Model name used (for tracking)
        similarity_calculator: The initialized similarity calculator instance
        batch_size: Size of batches for parallel processing
        max_workers: Maximum number of worker processes

    Returns:
        Dictionary of evaluation results
    """
    print(f"\n=== Evaluating graph for {model} ===")

    # Get all entities and relationships from the graph
    entities = graph.entities
    relationships = graph.relationships

    # Debug information about the graph
    print(f"Total entities in graph: {len(entities)}")
    print(f"Total relationships in graph: {len(relationships)}")

    # Debug Entity structure if available
    if len(entities) > 0:
        entity = entities[0]
        print(f"DEBUG - Entity structure: {entity}")
        print(f"DEBUG - Entity type: {type(entity)}")
        print(f"DEBUG - Entity attributes: {dir(entity)[:20]}")

    # Debug Relationship structure if available
    if len(relationships) > 0:
        relationship = relationships[0]
        print(f"DEBUG - Relationship structure: {relationship}")
        print(f"DEBUG - Relationship type: {type(relationship)}")
        print(f"DEBUG - Relationship attributes: {dir(relationship)[:20]}")

    # Aggregate ground truth from all documents
    all_gt_entities = []
    all_gt_relations = []

    for doc in documents:
        all_gt_entities.extend(doc["metadata"]["ground_truth_entities"])
        all_gt_relations.extend(doc["metadata"]["ground_truth_relations"])

    print(f"Total ground truth entities: {len(all_gt_entities)}")
    print(f"Total ground truth relations: {len(all_gt_relations)}")

    # Print some sample ground truth relations
    print("Sample ground truth relations:")
    for i, rel in enumerate(all_gt_relations[:5]):
        print(f"  {i}: {rel}")

    # Evaluate entity extraction with parallelization parameters
    entity_metrics = evaluate_entity_extraction(
        all_gt_entities,
        entities,
        similarity_calculator=similarity_calculator,
        batch_size=batch_size,
        max_workers=max_workers,
    )

    # Get entity mapping for semantic relation evaluation
    entity_match_map = entity_metrics.pop("entity_match_map", {})

    # Evaluate relation extraction with parallelization parameters
    relation_metrics = evaluate_relation_extraction(
        all_gt_relations,
        relationships,
        entity_match_map=entity_match_map,
        similarity_calculator=similarity_calculator,
        batch_size=batch_size,
        max_workers=max_workers,
    )

    # Store results
    results = {
        "model": model,
        "test_type": "evaluation",
        "entity_metrics": entity_metrics,
        "relation_metrics": relation_metrics,
        "evaluation_method": "openai_embeddings",
    }

    # Print summary
    print(
        f"Entity Extraction (openai_embeddings) - Precision: {entity_metrics['precision']:.4f}, "
        f"Recall: {entity_metrics['recall']:.4f}, "
        f"F1: {entity_metrics['f1']:.4f}"
    )
    print(
        f"Relation Extraction (openai_embeddings) - Precision: {relation_metrics['precision']:.4f}, "
        f"Recall: {relation_metrics['recall']:.4f}, "
        f"F1: {relation_metrics['f1']:.4f}"
    )

    return results


def save_results(results: Dict[str, Dict], model_name: str, run_id: str) -> str:
    """
    Save evaluation results to CSV and generate visualizations.

    Args:
        results: Evaluation results dictionary
        model_name: Name of the model used
        run_id: Unique run identifier

    Returns:
        Path to the saved results directory
    """
    # Create results directory
    results_dir = f"scier_results_{model_name}_{run_id}"
    os.makedirs(results_dir, exist_ok=True)

    # Prepare data for DataFrame
    rows = []
    model = results["model"]
    test_type = results["test_type"]
    evaluation_method = results.get("evaluation_method", "openai_embeddings")

    # Entity metrics
    entity_metrics = results["entity_metrics"]
    rows.append(
        {
            "model": model,
            "test_type": test_type,
            "extraction_type": "entity",
            "evaluation_method": evaluation_method,
            "precision": entity_metrics["precision"],
            "recall": entity_metrics["recall"],
            "f1": entity_metrics["f1"],
            "true_positives": entity_metrics["true_positives"],
            "false_positives": entity_metrics["false_positives"],
            "false_negatives": entity_metrics["false_negatives"],
            "ground_truth_count": entity_metrics["ground_truth_count"],
            "extracted_count": entity_metrics["extracted_count"],
        }
    )

    # Relation metrics
    relation_metrics = results["relation_metrics"]
    rows.append(
        {
            "model": model,
            "test_type": test_type,
            "extraction_type": "relation",
            "evaluation_method": evaluation_method,
            "precision": relation_metrics["precision"],
            "recall": relation_metrics["recall"],
            "f1": relation_metrics["f1"],
            "true_positives": relation_metrics["true_positives"],
            "false_positives": relation_metrics["false_positives"],
            "false_negatives": relation_metrics["false_negatives"],
            "ground_truth_count": relation_metrics["ground_truth_count"],
            "extracted_count": relation_metrics["extracted_count"],
        }
    )

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, f"{model_name}_evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Generate visualizations
    generate_visualizations(df, results_dir, model_name)

    return results_dir


def generate_visualizations(df: pd.DataFrame, output_dir: str, model_name: str) -> None:
    """
    Generate visualization charts for the results.

    Args:
        df: DataFrame with evaluation results
        output_dir: Directory to save visualizations
        model_name: Name of the model used
    """
    # Precision, recall, and F1
    plt.figure(figsize=(15, 10))

    # Set up the data
    entity_df = df[df["extraction_type"] == "entity"]
    relation_df = df[df["extraction_type"] == "relation"]

    # Get evaluation method if available
    evaluation_method = (
        df["evaluation_method"].iloc[0]
        if "evaluation_method" in df.columns
        else "openai_embeddings"
    )

    # Set up the plots
    metrics = ["precision", "recall", "f1"]
    positions = [0, 1, 2, 4, 5, 6]
    width = 0.35

    # Plot entity data
    plt.bar(
        positions[:3],
        entity_df[metrics].values[0],
        width,
        label="Entity",
    )

    # Plot relation data
    plt.bar(
        positions[3:],
        relation_df[metrics].values[0],
        width,
        label="Relation",
    )

    # Add labels and formatting
    plt.xticks(
        positions,
        ["P", "R", "F1", "P", "R", "F1"],
    )
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(f"{model_name} Performance ({evaluation_method})")
    plt.legend()
    plt.tight_layout()

    # Add text annotations for entity metrics
    for i, p in enumerate(positions[:3]):
        plt.text(
            p,
            entity_df[metrics[i]].values[0] + 0.02,
            f"{entity_df[metrics[i]].values[0]:.3f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    # Add text annotations for relation metrics
    for i, p in enumerate(positions[3:]):
        plt.text(
            p,
            relation_df[metrics[i]].values[0] + 0.02,
            f"{relation_df[metrics[i]].values[0]:.3f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    # Add vertical separator between entity and relation metrics
    plt.axvline(x=3.5, color="gray", linestyle="--", alpha=0.5)
    plt.text(1.5, 0.95, "Entity Extraction", ha="center", va="top", fontsize=12)
    plt.text(5.5, 0.95, "Relation Extraction", ha="center", va="top", fontsize=12)

    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{model_name}_metrics_comparison.png"))
    plt.close()

    # Save ground truth vs extracted counts comparison
    plt.figure(figsize=(10, 6))

    # Entity counts
    plt.subplot(1, 2, 1)

    gt_count = entity_df["ground_truth_count"].values[0]
    extracted_count = entity_df["extracted_count"].values[0]

    counts = [gt_count, extracted_count]
    labels = ["Ground Truth", "Extracted"]

    x_positions = list(range(len(counts)))
    plt.bar(x_positions, counts, width=0.6)
    plt.xticks(x_positions, labels)
    plt.ylabel("Count")
    plt.title("Entity Counts")

    # Add count labels
    for i, count in enumerate(counts):
        plt.text(i, count + 2, str(count), ha="center", va="bottom")

    # Relation counts
    plt.subplot(1, 2, 2)

    gt_count_rel = relation_df["ground_truth_count"].values[0]
    extracted_count_rel = relation_df["extracted_count"].values[0]

    counts_rel = [gt_count_rel, extracted_count_rel]
    labels_rel = ["Ground Truth", "Extracted"]

    x_positions_rel = list(range(len(counts_rel)))
    plt.bar(x_positions_rel, counts_rel, width=0.6)
    plt.xticks(x_positions_rel, labels_rel)
    plt.ylabel("Count")
    plt.title("Relation Counts")

    # Add count labels
    for i, count in enumerate(counts_rel):
        plt.text(i, count + 2, str(count), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_count_comparison.png"))
    plt.close()


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description="SciER Evaluation Script for Morphik using OpenAI embeddings"
    )
    parser.add_argument(
        "--limit", type=int, default=57, help="Maximum number of documents to process (default: 57)"
    )
    parser.add_argument(
        "--run-id", type=str, default=None, help="Unique run identifier (default: auto-generated)"
    )
    parser.add_argument(
        "--graph-name",
        type=str,
        required=True,
        help="Name of the existing graph to evaluate (e.g., 'scier_gpt4o_12345678')",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="existing_model_openai",
        help="Name for this evaluation run (default: existing_model_openai)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.70,
        help="Threshold for semantic similarity matching (default: 0.70)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model to use (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI API base URL (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for parallel processing (default: 100)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10000,
        help="Size of embedding cache (default: 10000)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=20,
        help="Batch size for embedding API calls (default: 20)",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=3500,
        help="Rate limit for OpenAI API requests per minute (default: 3500)",
    )

    args = parser.parse_args()

    # Generate run ID if not provided
    run_id = args.run_id or str(uuid.uuid4())[:8]

    print(f"Running evaluation for model: {args.model_name}")
    print(f"Run ID: {run_id}")
    print(f"Using OpenAI embeddings with model: {args.embedding_model}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"API base URL: {args.api_base}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max workers: {args.max_workers or 'CPU count - 1'}")
    print(f"Embedding cache size: {args.cache_size}")
    print(f"Embedding batch size: {args.embedding_batch_size}")
    print(f"Requests per minute: {args.requests_per_minute}")

    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it using: export OPENAI_API_KEY=your_api_key")
        return

    # Initialize Morphik client
    db = setup_morphik_client()

    # Initialize OpenAI embedding with optimized parameters
    similarity_calculator = OpenAIEmbedding(
        model_name=args.embedding_model,
        threshold=args.similarity_threshold,
        api_base=args.api_base,
        cache_size=args.cache_size,
        batch_size=args.embedding_batch_size,
    )
    # Update rate limit if specified
    similarity_calculator.requests_per_minute = args.requests_per_minute
    similarity_calculator.min_time_between_requests = 60.0 / args.requests_per_minute

    # Load SciER dataset
    scier_data = load_scier_data("test.jsonl", limit=args.limit)
    print(f"Loaded {len(scier_data)} records")

    # Prepare documents
    documents = prepare_text_for_evaluation(scier_data)
    print(f"Prepared {len(documents)} documents for evaluation")

    # Load existing graph
    graph_data = load_existing_graph(db, args.graph_name)

    # Evaluate graph
    results = evaluate_graph(
        db=db,
        documents=documents,
        graph=graph_data["graph"],
        model=args.model_name,
        similarity_calculator=similarity_calculator,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )

    # Save results using the input graph name as requested
    results_dir = save_results(results, f"{args.graph_name}_result", run_id)
    print(f"\nEvaluation complete! Results saved to {results_dir}")


if __name__ == "__main__":
    main()

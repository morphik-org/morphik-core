#!/usr/bin/env python3
"""Morphik Evaluator

Morphik-specific implementation of the RAG evaluation framework.
Inherits from BaseRAGEvaluator and implements Morphik-specific
ingest and query methods.

Usage:
    python morphik_eval.py
    python morphik_eval.py --output morphik_answers_v2.csv
    python morphik_eval.py --skip-ingestion
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from base_eval import BaseRAGEvaluator
from dotenv import load_dotenv
from morphik import Morphik

# Load environment variables
load_dotenv(override=True)


class MorphikEvaluator(BaseRAGEvaluator):
    """Morphik-specific RAG evaluator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optional: Set query delay to avoid overwhelming Morphik
        self.query_delay = 0.0  # No delay needed for Morphik

    def setup_client(self, **kwargs) -> Morphik:
        """Initialize Morphik client."""
        morphik_uri = os.getenv("MORPHIK_URI")
        if not morphik_uri:
            raise ValueError(
                "MORPHIK_URI environment variable not set. " "Please set it with: export MORPHIK_URI=your_morphik_uri"
            )

        print(f"Connecting to Morphik at: {morphik_uri}")

        try:
            db = Morphik(morphik_uri, timeout=30000)
            print("✓ Connected to Morphik successfully")
            return db
        except Exception as e:
            raise ConnectionError(f"Error connecting to Morphik: {e}. Make sure Morphik server is running.")

    def ingest(self, client: Morphik, docs_dir: Path, **kwargs) -> List[str]:
        """Ingest documents into Morphik."""
        # List available documents
        doc_files = list(docs_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")

        print(f"Found {len(doc_files)} documents to ingest:")
        for doc_file in doc_files:
            print(f"  - {doc_file.name}")

        # Ingest documents using ingest_directory
        try:
            ingested_docs = client.ingest_directory(
                directory=docs_dir,
                metadata={"source": "financial_eval", "type": "financial_document"},
                use_colpali=True,
            )

            print(f"✓ Successfully ingested {len(ingested_docs)} documents")

            # Wait for processing to complete
            print("Waiting for document processing to complete...")
            for doc in ingested_docs:
                client.wait_for_document_completion(doc.external_id, timeout_seconds=300)

            print("✓ All documents processed successfully")

            return [doc.external_id for doc in ingested_docs]

        except Exception as e:
            raise RuntimeError(f"Error ingesting documents: {e}")

    def query(self, client: Morphik, question: str, **kwargs) -> str:
        """Query Morphik with a question."""
        # Default query parameters optimized for financial documents
        query_params = {
            "k": 7,  # Retrieve more chunks for complex questions
            "padding": 1,  # Add context padding around chunks
            "min_score": 0.1,  # Lower threshold for financial data
            "llm_config": {"model": "o4-mini", "api_key": os.getenv("OPENAI_API_KEY")},
        }

        # Override with any provided kwargs
        query_params.update(kwargs)

        try:
            response = client.query(query=question, **query_params)
            return response.completion
        except Exception as e:
            raise RuntimeError(f"Error querying Morphik: {e}")


def main():
    """Main entry point for Morphik evaluation."""
    # Create CLI parser using base class helper
    parser = MorphikEvaluator.create_cli_parser("morphik")
    args = parser.parse_args()

    # Create evaluator instance
    evaluator = MorphikEvaluator(
        system_name="morphik", docs_dir=args.docs_dir, questions_file=args.questions, output_file=args.output
    )

    # Run evaluation
    try:
        output_file = evaluator.run_evaluation(skip_ingestion=args.skip_ingestion)
        print("\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""Base Evaluation Class

This module provides a base class for implementing RAG system evaluations.
Each system (Morphik, OpenAI, etc.) can inherit from this class and implement
the `ingest` and `query` methods specific to their system.

The base class handles:
- Loading questions from CSV
- Managing the evaluation process
- Saving results to CSV
- Progress tracking and error handling
"""

from __future__ import annotations

import abc
import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseRAGEvaluator(abc.ABC):
    """Base class for RAG system evaluators.

    Subclasses must implement `ingest` and `query` methods.
    """

    def __init__(self, system_name: str, docs_dir: Path, questions_file: Path, output_file: str = None):
        """Initialize the evaluator.

        Args:
            system_name: Name of the RAG system (e.g., "morphik", "openai")
            docs_dir: Directory containing documents to ingest
            questions_file: CSV file with evaluation questions
            output_file: Output CSV file for answers (defaults to {system_name}_answers.csv)
        """
        self.system_name = system_name
        self.docs_dir = Path(docs_dir)
        self.questions_file = Path(questions_file)
        self.output_file = output_file or f"{system_name}_answers.csv"

        # Validate inputs
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

        if not self.questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")

    @abc.abstractmethod
    def setup_client(self, **kwargs) -> Any:
        """Initialize the RAG system client.

        Returns:
            Client object for the RAG system
        """
        pass

    @abc.abstractmethod
    def ingest(self, client: Any, docs_dir: Path, **kwargs) -> List[str]:
        """Ingest documents into the RAG system.

        Args:
            client: The RAG system client
            docs_dir: Directory containing documents to ingest
            **kwargs: Additional system-specific parameters

        Returns:
            List of document IDs or identifiers
        """
        pass

    @abc.abstractmethod
    def query(self, client: Any, question: str, **kwargs) -> str:
        """Query the RAG system with a question.

        Args:
            client: The RAG system client
            question: Question to ask
            **kwargs: Additional system-specific parameters

        Returns:
            Answer string from the RAG system
        """
        pass

    def load_questions(self) -> List[Dict[str, str]]:
        """Load questions from CSV file."""
        questions = []

        with open(self.questions_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append({"question": row["question"].strip(), "golden_answer": row.get("answer", "").strip()})

        return questions

    def generate_answers(
        self, client: Any, questions: List[Dict[str, str]], skip_ingestion: bool = False, **kwargs
    ) -> List[Dict[str, str]]:
        """Generate answers for all questions.

        Args:
            client: The RAG system client
            questions: List of question dictionaries
            skip_ingestion: Skip document ingestion step
            **kwargs: Additional parameters for ingest/query methods

        Returns:
            List of result dictionaries with question and answer
        """
        # Ingest documents if not skipped
        if not skip_ingestion:
            print(f"Ingesting documents from {self.docs_dir}...")
            doc_ids = self.ingest(client, self.docs_dir, **kwargs)
            print(f"✓ Ingested {len(doc_ids)} documents")
        else:
            print("Skipping document ingestion")

        print(f"\nGenerating answers for {len(questions)} questions...")

        results = []

        for i, q_data in enumerate(questions, 1):
            question = q_data["question"]

            print(f"Processing question {i}/{len(questions)}: {question[:80]}...")

            try:
                answer = self.query(client, question, **kwargs)

                # Handle empty or error responses
                if not answer or answer.strip().lower() in ["", "none", "n/a"]:
                    answer = "[No answer generated]"

                results.append({"question": question, "answer": answer.strip()})

                print(f"  ✓ Generated answer ({len(answer)} chars)")

                # Optional delay to avoid overwhelming systems
                if hasattr(self, "query_delay") and self.query_delay > 0:
                    time.sleep(self.query_delay)

            except Exception as e:
                print(f"  ✗ Error generating answer: {e}")
                results.append({"question": question, "answer": f"[Error: {str(e)}]"})

        return results

    def save_results(self, results: List[Dict[str, str]]) -> None:
        """Save results to CSV file."""
        print(f"\nSaving results to {self.output_file}...")

        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(results)

        print(f"✓ Saved {len(results)} answers to {self.output_file}")

    def run_evaluation(
        self,
        skip_ingestion: bool = False,
        client_kwargs: Optional[Dict] = None,
        ingest_kwargs: Optional[Dict] = None,
        query_kwargs: Optional[Dict] = None,
    ) -> str:
        """Run the complete evaluation process.

        Args:
            skip_ingestion: Skip document ingestion step
            client_kwargs: Parameters for setup_client()
            ingest_kwargs: Parameters for ingest()
            query_kwargs: Parameters for query()

        Returns:
            Path to the output CSV file
        """
        client_kwargs = client_kwargs or {}
        ingest_kwargs = ingest_kwargs or {}
        query_kwargs = query_kwargs or {}

        print("=" * 60)
        print(f"{self.system_name.upper()} EVALUATION")
        print("=" * 60)
        print(f"System: {self.system_name}")
        print(f"Documents: {self.docs_dir}")
        print(f"Questions: {self.questions_file}")
        print(f"Output: {self.output_file}")
        print(f"Skip ingestion: {skip_ingestion}")
        print("=" * 60)

        # Setup client
        print(f"Setting up {self.system_name} client...")
        client = self.setup_client(**client_kwargs)
        print("✓ Client setup complete")

        # Store client for potential cleanup
        self._client = client

        # Load questions
        print(f"Loading questions from {self.questions_file}...")
        questions = self.load_questions()
        print(f"✓ Loaded {len(questions)} questions")

        # Generate answers
        results = self.generate_answers(
            client, questions, skip_ingestion=skip_ingestion, **{**ingest_kwargs, **query_kwargs}
        )

        # Save results
        self.save_results(results)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Generated answers saved to: {self.output_file}")
        print("Next steps:")
        print(f"1. Run evaluation: python evaluate.py {self.output_file}")
        print("2. Check results in eval_results.csv")
        print("=" * 60)

        return self.output_file

    @classmethod
    def create_cli_parser(cls, system_name: str) -> argparse.ArgumentParser:
        """Create a standard CLI parser for evaluation scripts."""
        default_docs_dir = Path(__file__).parent / "docs"
        default_questions_file = Path(__file__).parent / "questions_and_answers.csv"
        default_output_file = f"{system_name}_answers.csv"

        parser = argparse.ArgumentParser(
            description=f"Generate {system_name} answers for financial document evaluation"
        )
        parser.add_argument(
            "--docs-dir",
            type=Path,
            default=default_docs_dir,
            help=(f"Directory containing financial documents " f"(default: {default_docs_dir})"),
        )
        parser.add_argument(
            "--questions",
            type=Path,
            default=default_questions_file,
            help=f"CSV file with questions (default: {default_questions_file})",
        )
        parser.add_argument(
            "--output",
            default=default_output_file,
            help=f"Output CSV file for answers (default: {default_output_file})",
        )
        parser.add_argument(
            "--skip-ingestion", action="store_true", help="Skip document ingestion (use existing documents)"
        )

        return parser


# Example usage template for implementing a new evaluator:
"""
class MySystemEvaluator(BaseRAGEvaluator):
    def setup_client(self, **kwargs):
        # Initialize your system's client
        return MySystemClient(**kwargs)

    def ingest(self, client, docs_dir, **kwargs):
        # Ingest documents into your system
        doc_files = list(docs_dir.glob("*.pdf"))
        doc_ids = []
        for doc_file in doc_files:
            doc_id = client.ingest_document(doc_file)
            doc_ids.append(doc_id)
        return doc_ids

    def query(self, client, question, **kwargs):
        # Query your system
        response = client.query(question, **kwargs)
        return response.answer

# Usage:
if __name__ == "__main__":
    parser = MySystemEvaluator.create_cli_parser("mysystem")
    args = parser.parse_args()

    evaluator = MySystemEvaluator(
        system_name="mysystem",
        docs_dir=args.docs_dir,
        questions_file=args.questions,
        output_file=args.output
    )

    evaluator.run_evaluation(skip_ingestion=args.skip_ingestion)
"""

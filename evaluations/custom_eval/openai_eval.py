#!/usr/bin/env python3
"""OpenAI Evaluator

OpenAI-specific implementation of the RAG evaluation framework.
Inherits from BaseRAGEvaluator and implements OpenAI-specific
methods using direct file uploads.

Usage:
    python openai_eval.py
    python openai_eval.py --output openai_answers.csv
    python openai_eval.py --skip-ingestion
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import openai
from base_eval import BaseRAGEvaluator
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


class OpenAIEvaluator(BaseRAGEvaluator):
    """OpenAI-specific RAG evaluator using direct file uploads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_delay = 1.0  # Add delay to respect rate limits
        self.uploaded_files = []  # Track uploaded files for cleanup

    def setup_client(self, **kwargs) -> openai.OpenAI:
        """Initialize OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. " "Please set it with: export OPENAI_API_KEY=your_api_key"
            )

        print("Connecting to OpenAI API...")

        try:
            client = openai.OpenAI(api_key=api_key)
            # Test the connection
            client.models.list()
            print("✓ Connected to OpenAI successfully")
            return client
        except Exception as e:
            raise ConnectionError(f"Error connecting to OpenAI: {e}")

    def ingest(self, client: openai.OpenAI, docs_dir: Path, **kwargs) -> List[str]:
        """Upload PDF files to OpenAI."""
        # List available documents
        doc_files = list(docs_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")

        print(f"Found {len(doc_files)} documents to upload:")
        for doc_file in doc_files:
            print(f"  - {doc_file.name}")

        try:
            uploaded_file_ids = []

            for doc_file in doc_files:
                print(f"Uploading {doc_file.name}...")

                # Upload file to OpenAI
                with open(doc_file, "rb") as f:
                    uploaded_file = client.files.create(
                        file=f, purpose="assistants"  # Use 'assistants' purpose for file processing
                    )

                uploaded_file_ids.append(uploaded_file.id)
                self.uploaded_files.append(uploaded_file.id)
                print(f"  ✓ Uploaded as {uploaded_file.id}")

            print(f"✓ Successfully uploaded {len(uploaded_file_ids)} documents")
            return uploaded_file_ids

        except Exception as e:
            raise RuntimeError(f"Error uploading documents: {e}")

    def query(self, client: openai.OpenAI, question: str, **kwargs) -> str:
        """Query OpenAI with the uploaded files as context."""
        if not self.uploaded_files:
            raise RuntimeError("No files have been uploaded. Please run ingestion first.")

        try:
            # Create file content objects for the API
            file_attachments = [
                {"file_id": file_id, "tools": [{"type": "file_search"}]} for file_id in self.uploaded_files
            ]

            # Create a thread with the files attached
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""Please answer the following question based on the uploaded financial documents.

Question: {question}

Instructions:
- Provide a precise, factual answer based only on the information in the documents
- Include specific numbers, percentages, and figures when relevant
- If the answer requires calculations, show your work
- Be concise but complete
- If the information is not available in the documents, state that clearly

Answer:""",
                        "attachments": file_attachments,
                    }
                ]
            )

            # Create and run an assistant
            assistant = client.beta.assistants.create(
                name="Financial Document Analyzer",
                instructions="""You are a financial document analysis expert. Your job is to provide accurate, precise answers to questions about financial documents.

Key guidelines:
- Only use information from the provided documents
- Be precise with numbers and calculations
- Show your work for any calculations
- If information is not available, state that clearly
- Focus on factual accuracy over completeness""",
                model="gpt-4.1",  # Use latest model with file processing capabilities
                tools=[{"type": "file_search"}],
            )

            # Run the assistant
            run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)

            # Wait for completion
            import time

            while run.status in ["queued", "in_progress"]:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            if run.status == "completed":
                # Get the assistant's response
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                assistant_message = messages.data[0]

                # Extract text content
                response_text = ""
                for content_block in assistant_message.content:
                    if content_block.type == "text":
                        response_text += content_block.text.value

                # Cleanup
                client.beta.assistants.delete(assistant.id)
                client.beta.threads.delete(thread.id)

                return response_text.strip()
            else:
                raise RuntimeError(f"Assistant run failed with status: {run.status}")

        except Exception as e:
            raise RuntimeError(f"Error querying OpenAI: {e}")

    def cleanup(self, client: openai.OpenAI):
        """Clean up uploaded files."""
        print("Cleaning up uploaded files...")
        for file_id in self.uploaded_files:
            try:
                client.files.delete(file_id)
                print(f"  ✓ Deleted {file_id}")
            except Exception as e:
                print(f"  ✗ Error deleting {file_id}: {e}")
        self.uploaded_files.clear()


def main():
    """Main entry point for OpenAI evaluation."""
    # Create CLI parser using base class helper
    parser = OpenAIEvaluator.create_cli_parser("openai")
    args = parser.parse_args()

    # Create evaluator instance
    evaluator = OpenAIEvaluator(
        system_name="openai", docs_dir=args.docs_dir, questions_file=args.questions, output_file=args.output
    )

    # Run evaluation
    try:
        output_file = evaluator.run_evaluation(skip_ingestion=args.skip_ingestion)
        print("\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return 1
    finally:
        # Always cleanup uploaded files
        if hasattr(evaluator, "_client") and evaluator._client:
            evaluator.cleanup(evaluator._client)

    return 0


if __name__ == "__main__":
    exit(main())

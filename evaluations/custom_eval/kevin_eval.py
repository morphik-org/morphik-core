#!/usr/bin/env python3
"""Kevin's Evaluator

This is a custom evaluator created by Kevin.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from base_eval import BaseRAGEvaluator
import psycopg2
from unstructured.partition.pdf import partition_pdf
from litellm import embedding, completion
from flag_reranker import FlagReranker

# Load environment variables
load_dotenv(override=True)


class KevinEvaluator(BaseRAGEvaluator):
    """Kevin's custom RAG evaluator."""

    def setup_client(self, **kwargs) -> dict:
        """Initialize the RAG system client."""
        print("Setting up client...")
        
        # Initialize database connection
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI environment variable not set.")
        
        try:
            conn = psycopg2.connect(postgres_uri)
            print("✓ Connected to PostgreSQL successfully")
        except Exception as e:
            raise ConnectionError(f"Error connecting to PostgreSQL: {e}")

        # Create table for chunks and enable pgvector
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(1536)
                );
            """)
            # Clear the table before ingestion
            cur.execute("TRUNCATE TABLE chunks;")
        conn.commit()
        print("✓ 'chunks' table created successfully")

        # Initialize re-ranker
        reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        print("✓ Re-ranker initialized successfully")

        return {"db_connection": conn, "reranker": reranker}
        
        # Initialize database connection
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI environment variable not set.")
        
        try:
            conn = psycopg2.connect(postgres_uri)
            print("✓ Connected to PostgreSQL successfully")
        except Exception as e:
            raise ConnectionError(f"Error connecting to PostgreSQL: {e}")

        # Create table for chunks and enable pgvector
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(1536)
                );
            """)
        conn.commit()
        print("✓ 'chunks' table created successfully")

        # Initialize re-ranker
        reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        print("✓ Re-ranker initialized successfully")

        return {"db_connection": conn, "reranker": reranker}

    def ingest(self, client: dict, docs_dir: Path, **kwargs) -> List[str]:
        """Ingest documents into the RAG system."""
        print(f"Ingesting documents from: {docs_dir}")
        doc_files = list(docs_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")

        conn = client["db_connection"]

        for doc_file in doc_files:
            print(f"  - Ingesting {doc_file.name}")
            # Parse PDF using unstructured
            elements = partition_pdf(filename=doc_file, strategy="hi_res")
            chunks = [str(el) for el in elements]

            # Generate embeddings using litellm
            response = embedding(model="text-embedding-3-small", input=chunks)
            embeddings = [item['embedding'] for item in response.data]

            # Store chunks and embeddings in the database
            with conn.cursor() as cur:
                for i, chunk in enumerate(chunks):
                    cur.execute(
                        "INSERT INTO chunks (doc_id, chunk_text, embedding) VALUES (%s, %s, %s)",
                        (doc_file.name, chunk, embeddings[i])
                    )
            conn.commit()

        print(f"✓ Successfully ingested {len(doc_files)} documents")
        return [doc.name for doc in doc_files]

    def query(self, client: dict, question: str, **kwargs) -> str:
        """Query the RAG system with a question."""
        print(f"Querying with question: {question}")

        conn = client["db_connection"]
        reranker = client["reranker"]

        # Generate query embedding
        response = embedding(model="text-embedding-3-small", input=[question])
        query_embedding = response.data[0]['embedding']

        # Retrieve chunks from the database
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_text FROM chunks ORDER BY embedding <=> %s LIMIT 20",
                (str(query_embedding),)
            )
            retrieved_chunks = [row[0] for row in cur.fetchall()]

        # Re-rank chunks
        reranked_chunks = reranker.rerank(question, retrieved_chunks, top_k=5)
        context = "\n".join([chunk['text'] for chunk in reranked_chunks])

        # Construct prompt
        prompt = f"""Answer the following question based on the provided context.

        Context:
        {context}

        Question: {question}

        Answer:"""

        # Generate answer
        response = completion(
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
            api_key=os.getenv("OPENAI_API_KEY")
        )

        return response.choices[0].message.content


def main():
    """Main entry point for Kevin's evaluation."""
    parser = KevinEvaluator.create_cli_parser("kevin")
    args = parser.parse_args()

    evaluator = KevinEvaluator(
        system_name="kevin",
        docs_dir=args.docs_dir,
        questions_file=args.questions,
        output_file=args.output,
    )

    try:
        output_file = evaluator.run_evaluation(skip_ingestion=args.skip_ingestion)
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

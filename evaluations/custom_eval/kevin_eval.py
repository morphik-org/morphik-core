#!/usr/bin/env python3
"""Kevin's Evaluator

This is a custom evaluator created by Kevin.
"""

from __future__ import annotations

import sys
import os
import psycopg2
import psycopg2.extras
import asyncio
import re
from unstructured.documents.elements import Table
from pgvector.psycopg import register_vector

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.models.chunk import DocumentChunk
from core.reranker.flag_reranker import FlagReranker

# Remove the project root from Python path after imports
sys.path.pop(0)
from base_eval import BaseRAGEvaluator
from dotenv import load_dotenv
# Load environment variables
load_dotenv("../../.env.example",override=True)

# Global debug flag
DEBUG_MODE = False

class KevinEvaluator(BaseRAGEvaluator):
    """Kevin's custom RAG evaluator."""

    def setup_client(self, **kwargs) -> dict:
        """Initialize the RAG system client."""
        print("Setting up client...")
        
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI environment variable not set.")
        
        clean_uri = postgres_uri.replace("postgresql+asyncpg://", "postgresql://")
        
        try:
            conn = psycopg2.connect(clean_uri)
            print("✓ Connected to PostgreSQL successfully")
            
            self._setup_database(conn)
            print("✓ 'chunks' table created successfully")

            reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
            print("✓ Re-ranker initialized successfully")
            
            # Create an event loop for the async reranker
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        except Exception as e:
            raise ConnectionError(f"Error connecting to PostgreSQL: {e}")

        return {"db_conn": conn, "reranker": reranker, "loop": loop}
        
    def _setup_database(self, conn: psycopg2.extensions.connection):
        """Setup database tables and extensions."""
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(1536),
                    tsv tsvector
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin(tsv);")
            cur.execute("TRUNCATE TABLE chunks;")
            conn.commit()

    def ingest(self, client: dict, docs_dir: Path, **kwargs) -> List[str]:
        """Ingest documents into the RAG system."""
        print(f"Ingesting documents from: {docs_dir}")
        doc_files = list(docs_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")

        conn = client["db_conn"]
        
        self._ingest_sync(conn, doc_files)
        
        print(f"✓ Successfully ingested {len(doc_files)} documents")
        return [doc.name for doc in doc_files]

    def _ingest_sync(self, conn: psycopg2.extensions.connection, doc_files: List[Path]):
        """Sync ingestion helper."""
        for doc_file in doc_files:
            print(f"  - Ingesting {doc_file.name}")
            chunks = []
            
            try:
                chunks = self._process_with_unstructured(doc_file)
                print(f"    ✓ Successfully processed with unstructured")
            except Exception as e:
                print(f"    ⚠️ unstructured failed: {e}")
                continue
            
            if not chunks:
                print(f"    ❌ All PDF processing methods failed for {doc_file.name}")
                continue

            try:
                batch_size = 20
                all_embeddings = []
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    response = embedding(model="text-embedding-3-small", input=batch)
                    batch_embeddings = [item['embedding'] for item in response.data]
                    all_embeddings.extend(batch_embeddings)

                with conn.cursor() as cur:
                    data_to_insert = [
                        (doc_file.name, chunk, all_embeddings[i], chunk)
                        for i, chunk in enumerate(chunks)
                    ]
                    psycopg2.extras.execute_batch(
                        cur,
                        "INSERT INTO chunks (doc_id, chunk_text, embedding, tsv) VALUES (%s, %s, %s, to_tsvector('english', %s))",
                        data_to_insert
                    )
                conn.commit()
                
                print(f"    ✓ Stored {len(chunks)} chunks in database")
                
            except Exception as e:
                conn.rollback()
                print(f"    ❌ Failed to process embeddings for {doc_file.name}: {e}")

    def _process_with_unstructured(self, doc_file: Path) -> List[str]:
        """Process PDF using unstructured, preserving table structures as HTML."""
        elements = partition_pdf(filename=str(doc_file), strategy="hi_res", infer_table_structure=True)
        
        chunks = []
        for el in elements:
            if isinstance(el, Table):
                # For tables, get the HTML representation
                table_html = el.metadata.text_as_html
                if table_html:
                    chunks.append(table_html.strip())
            else:
                # For other elements, get the text representation
                text = str(el).strip()
                if text:
                    chunks.append(text)
        return chunks

    def query(self, client: dict, question: str, **kwargs) -> str:
        """Query the RAG system with a question."""
        print(f"Querying with question: {question}")

        conn = client["db_conn"]
        reranker = client["reranker"]
        loop = client["loop"]

        return self._query_sync(conn, reranker, loop, question)

    def _query_sync(self, conn: psycopg2.extensions.connection, reranker: FlagReranker, loop: asyncio.AbstractEventLoop, question: str) -> str:
        """Sync query helper."""
        try:
            # Get query embedding
            response = embedding(model="text-embedding-3-small", input=[question])
            query_embedding = response.data[0]['embedding']

            # Prepare for hybrid search
            with conn.cursor() as cur:
                # 1. Vector Search
                cur.execute(
                    "SELECT doc_id, chunk_text, embedding, id FROM chunks ORDER BY embedding <=> %s LIMIT 50",
                    (str(query_embedding),)
                )
                vector_rows = cur.fetchall()

                # 2. Full-Text Search
                sanitized_question = re.sub(r'[^\w\s]', '', question)
                query_text = " & ".join(sanitized_question.split())
                
                fts_rows = []
                if query_text:
                    try:
                        cur.execute(
                            "SELECT doc_id, chunk_text, embedding, id FROM chunks WHERE tsv @@ to_tsquery('english', %s) LIMIT 50",
                            (query_text,)
                        )
                        fts_rows = cur.fetchall()
                    except psycopg2.Error as e:
                        if DEBUG_MODE:
                            print(f"[DEBUG] FTS query failed: {e}")

                # Combine and deduplicate results
                combined_rows = {row[3]: row for row in vector_rows}
                for row in fts_rows:
                    combined_rows[row[3]] = row
                
                document_chunks = [
                    DocumentChunk(document_id=row[0], content=row[1], embedding=row[2], chunk_number=row[3])
                    for row in combined_rows.values()
                ]
                
                if DEBUG_MODE:
                    print("\n---\n[DEBUG] Top 5 retrieved chunks (before reranking):")
                    for chunk in document_chunks[:5]:
                        print(f"  - ID: {chunk.chunk_number}, Doc: {chunk.document_id}, Content: {chunk.content[:100]}...")

                # Rerank combined results to find the single best chunk
                reranked_chunks = loop.run_until_complete(reranker.rerank(question, document_chunks))

                if not reranked_chunks:
                    return "Could not find a relevant chunk of text."

                if DEBUG_MODE:
                    print("\n[DEBUG] Top 5 reranked chunks:")
                    for chunk in reranked_chunks[:5]:
                        print(f"  - ID: {chunk.chunk_number}, Doc: {chunk.document_id}, Content: {chunk.content[:100]}...")

                # --- Context Window Retrieval ---
                best_chunk = reranked_chunks[0]
                best_chunk_id = best_chunk.chunk_number
                doc_id = best_chunk.document_id
                window_size = 2 # chunks before and after

                context_chunk_ids = list(range(max(1, best_chunk_id - window_size), best_chunk_id + window_size + 1))

                cur.execute(
                    "SELECT chunk_text FROM chunks WHERE doc_id = %s AND id = ANY(%s) ORDER BY id",
                    (doc_id, context_chunk_ids)
                )
                context_rows = cur.fetchall()
                context = "\n".join([row[0] for row in context_rows])

            if DEBUG_MODE:
                print(f"\n[DEBUG] Final context passed to LLM (from chunk {best_chunk_id} in {doc_id}):\n{context}\n---")

            prompt = f'''You are a financial document analysis expert. Your job is to provide accurate, precise answers to questions about financial documents based *only* on the provided context.

            Key Instructions:
            - Provide a precise, factual answer.
            - Be concise but complete.
            - Focus on factual accuracy over completeness.
            - Include specific numbers, percentages, and figures when relevant.
            - If the answer requires calculations, perform them, but do not show your calculations in the final answer.
            - If the information is not available in the provided context, state that clearly and do not provide an answer.

            Context:
            {context}

            Question: {question}

            Answer:'''

            response = completion(
                model="o4-mini",
                messages=[{"role": "user", "content": prompt}],
                api_key=os.getenv("OPENAI_API_KEY")
            )

            final_answer = response.choices[0].message.content
            
            if DEBUG_MODE:
                print(f"\n[DEBUG] Final Generated Answer:\n{final_answer}\n---")

            return final_answer
        except Exception as e:
            conn.rollback()
            raise e


def main():
    """Main entry point for Kevin's evaluation."""
    parser = KevinEvaluator.create_cli_parser("kevin")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print retrieval info.")
    args = parser.parse_args()

    global DEBUG_MODE
    if args.debug:
        DEBUG_MODE = True
        print("\n[--- DEBUG MODE ENABLED ---]\n")

    evaluator = KevinEvaluator(
        system_name="kevin",
        docs_dir=args.docs_dir,
        questions_file=args.questions,
        output_file=args.output,
    )
    
    client = None
    try:
        # Manually get client to manage connection closing
        client = evaluator.setup_client()
        evaluator._client = client # so run_evaluation can use it
        output_file = evaluator.run_evaluation(skip_ingestion=args.skip_ingestion)
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return 1
    
    finally:
        # Ensure the database connection and event loop are closed
        if client:
            if "db_conn" in client:
                client["db_conn"].close()
            if "loop" in client:
                client["loop"].close()

    return 0


if __name__ == "__main__":
    exit(main())

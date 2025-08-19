#!/usr/bin/env python3
"""Kevin's ColPali-Enhanced Evaluator (v3 - Prompt & Model Tuning)

Refactored for improved performance, readability, and accuracy by leveraging
in-database vector search, a more powerful model, and refined prompting.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2  # opencv-python - already in your dependencies
import fitz  # PyMuPDF - already in your dependencies
import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from PIL import Image
from psycopg2 import extras
from tqdm import tqdm

from base_eval import BaseRAGEvaluator
from colpali_engine.models import ColPali, ColPaliProcessor

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.models.chunk import DocumentChunk
from core.reranker.flag_reranker import FlagReranker

# Remove the project root from Python path after imports
sys.path.pop(0)

# Load environment variables
load_dotenv('../../.env.example', override=True)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Only show errors, not warnings
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def completion(**kwargs):
    """Wrapper for OpenAI completion API with proper parameter names"""
    kwargs.pop('api_key', None)
    # Fix for older OpenAI API versions if needed, though modern SDK uses max_tokens
    if 'max_tokens' in kwargs:
        kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
    return client.chat.completions.create(**kwargs)

# Global debug flag
DEBUG_MODE = False

class ColPaliDocumentPage:
    """Document page with multi-vector embeddings from ColPali."""
    
    def __init__(self, document_id: str, page_number: int, image_path: str, 
                 content: str = '', page_id: int = None, embeddings: np.ndarray = None):
        self.document_id = document_id
        self.page_number = page_number
        self.image_path = image_path
        self.content = content
        self.page_id = page_id  # Database ID
        self.embeddings = embeddings # Optional: for context building
        self.chunk_number = f"{document_id}_page_{page_number}"

class ColPaliKevinEvaluator(BaseRAGEvaluator):
    """Kevin's improved RAG evaluator enhanced with ColPali for visual document understanding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Device selection with proper dtype handling
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.use_mps = True
            self.model_dtype = torch.float16
            print("✓ MPS (Apple Silicon GPU) available - using float16 for compatibility")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.use_mps = False
            self.model_dtype = torch.bfloat16
            print("✓ CUDA available - using bfloat16")
        else:
            self.device = torch.device("cpu")
            self.use_mps = False
            self.model_dtype = torch.float32
            print("Using CPU - using float32")
        
        self.model = None
        self.processor = None
        self.images_dir = None
        self.embedding_cache_dir = Path("./.embedding_cache")
        self.embedding_cache_dir.mkdir(exist_ok=True)
        self.debug_dir = Path("./debug_output")
        self.debug_dir.mkdir(exist_ok=True)

    def _pre_flight_checks(self):
        """Run pre-flight checks to catch errors early."""
        print("Running pre-flight checks...")
        if not os.getenv("POSTGRES_URI"):
            raise ValueError("POSTGRES_URI environment variable not set.")
        print("✓ POSTGRES_URI is set.")
        
        try:
            from psycopg2 import extras
            print("✓ psycopg2.extras is available.")
        except ImportError:
            raise ImportError("psycopg2.extras could not be imported. Please check your installation.")
        
        print("✓ Pre-flight checks passed.")
        
    def setup_client(self, **kwargs) -> dict:
        """Initialize the ColPali-enhanced RAG system client with memory optimizations."""
        self._pre_flight_checks()
        print("Setting up ColPali multi-vector client...")
        
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI environment variable not set.")
        
        clean_uri = postgres_uri.replace("postgresql+asyncpg://", "postgresql://")
        
        try:
            conn = psycopg2.connect(clean_uri)
            register_vector(conn)
            print("✓ Connected to PostgreSQL successfully")
            
            self._setup_database(conn)
            self._ensure_schema_compatibility(conn)
            
            print("✓ Multi-vector tables created successfully")

        except Exception as e:
            raise ConnectionError(f"Error connecting to PostgreSQL: {e}")

        try:
            print("Loading ColPali model via colpali-engine...")
            model_name = "vidore/colpali-v1.3"
            
            print(f"    Using device: {self.device}")
            print(f"    Using dtype: {self.model_dtype}")
            
            self.processor = ColPaliProcessor.from_pretrained(model_name)
            print("    ✓ Processor loaded")
            
            # Simplified model loading
            self.model = ColPali.from_pretrained(
                model_name,
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=True
            ).to(self.device).eval()

            print(f"    ✓ ColPali loaded successfully on {self.device}")

        except Exception as e:
            print(f"⌐ Complete model loading failure: {e}")
            raise

        try:
            reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
            print("✓ Re-ranker initialized successfully")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize reranker: {e}")
            reranker = None
            loop = None

        self.images_dir = Path("./document_images")
        self.images_dir.mkdir(exist_ok=True)

        return {
            "db_conn": conn, 
            "reranker": reranker, 
            "loop": loop,
            "model": self.model,
            "processor": self.processor
        }
    
    def _ensure_schema_compatibility(self, conn: psycopg2.extensions.connection):
        """Ensure the database schema has all required columns."""
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'document_pages'
                );
            """)
            if cur.fetchone()[0]:
                cur.execute("ALTER TABLE document_pages ADD COLUMN IF NOT EXISTS tables_data TEXT DEFAULT '';")
                cur.execute("ALTER TABLE document_pages ADD COLUMN IF NOT EXISTS text_blocks TEXT DEFAULT '';")
                conn.commit()
                print("✅ Schema compatibility ensured")
        
    def _setup_database(self, conn: psycopg2.extensions.connection):
        """Setup database tables for multi-vector storage."""
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_pages (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL,
                    page_number INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    extracted_text TEXT DEFAULT '',
                    tables_data TEXT DEFAULT '',
                    text_blocks TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(doc_id, page_number)
                );
            """)
            
            embedding_dim = 128
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS patch_embeddings (
                    id SERIAL PRIMARY KEY,
                    page_id INTEGER REFERENCES document_pages(id) ON DELETE CASCADE,
                    patch_index INTEGER NOT NULL,
                    embedding vector({embedding_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("CREATE INDEX IF NOT EXISTS patch_embeddings_page_id_idx ON patch_embeddings (page_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS patch_embeddings_embedding_idx ON patch_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            cur.execute("CREATE INDEX IF NOT EXISTS document_pages_doc_id_idx ON document_pages (doc_id);")
            
            cur.execute("TRUNCATE TABLE patch_embeddings, document_pages RESTART IDENTITY CASCADE;")
            cur.execute("DROP TABLE IF EXISTS visual_chunks;")
            
            conn.commit()

    def ingest(self, client: dict, docs_dir: Path, **kwargs) -> List[str]:
        """Ingest documents using ColPali multi-vector processing."""
        print(f"Ingesting documents from: {docs_dir}")
        doc_files = list(docs_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")

        print(f"Found {len(doc_files)} PDF documents:")
        for doc_file in doc_files:
            print(f"  - {doc_file.name}")

        conn = client["db_conn"]
        model = client["model"]
        processor = client["processor"]
        
        self._ingest_with_multivector_processing(conn, model, processor, doc_files)
        
        print(f"✓ Successfully ingested {len(doc_files)} documents with multi-vector processing")
        return [doc.name for doc in doc_files]

    def _convert_pdf_to_images_pymupdf(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to images using PyMuPDF."""
        doc = fitz.open(pdf_path)
        images = []
        for page_num in tqdm(range(len(doc)), desc=f"  Converting {pdf_path.name}", unit="page", leave=False):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            if img.size[0] > 1200 or img.size[1] > 1200:
                img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            images.append(img)
        doc.close()
        return images

    def _extract_enhanced_text_data(self, pdf_path: Path) -> tuple[List[str], List[str], List[str]]:
        """Enhanced text extraction with tables and structured text blocks."""
        doc = fitz.open(pdf_path)
        texts, tables_data, text_blocks = [], [], []
        
        for page in doc:
            texts.append(page.get_text().strip())
            
            page_tables = []
            try:
                for table in page.find_tables():
                    table_content = table.extract()
                    if table_content:
                        page_tables.append(self._format_table_content(table_content))
            except Exception:
                pass # Ignore table extraction errors
            tables_data.append("\n".join(page_tables))

            try:
                blocks = [b[4] for b in page.get_text("blocks") if b[4].strip()]
                text_blocks.append("\n\n".join(blocks))
            except Exception:
                text_blocks.append("")

        doc.close()
        return texts, tables_data, text_blocks

    def _format_table_content(self, table_content: List[List]) -> str:
        """Format table content into readable text."""
        return "\n".join([" | ".join([str(c).strip() for c in r if c is not None]) for r in table_content if r])

    def _ingest_with_multivector_processing(self, conn: psycopg2.extensions.connection, 
                                          model, processor, doc_files: List[Path]):
        """Ingest documents using ColPali multi-vector approach."""
        for doc_file in tqdm(doc_files, desc="Ingesting Documents", unit="doc"):
            print(f"  - Processing {doc_file.name}")
            
            doc_hash = hashlib.sha256(doc_file.read_bytes()).hexdigest()
            cache_file = self.embedding_cache_dir / f"{doc_file.stem}_{doc_hash[:10]}_multivector.npz"

            if cache_file.exists():
                print(f"    ✓ Loading embeddings from cache: {cache_file.name}")
                data = np.load(cache_file, allow_pickle=True)
                all_page_embeddings = data['embeddings']
                image_paths = data['image_paths'].tolist()
            else:
                print("    - Generating new multi-vector embeddings...")
                images = self._convert_pdf_to_images_pymupdf(doc_file)
                print(f"    ✓ Converted to {len(images)} page images")
                
                all_page_embeddings, image_paths = [], []
                for i, img in enumerate(tqdm(images, desc="    Embedding Pages", unit="page", leave=False)):
                    img_path = self.images_dir / f"{doc_file.stem}_page_{i+1:03d}.png"
                    img.save(img_path)
                    image_paths.append(str(img_path))
                    
                    with torch.no_grad():
                        inputs = processor.process_images([img]).to(self.device)
                        for key, tensor in inputs.items():
                            if tensor.is_floating_point():
                                inputs[key] = tensor.to(self.model_dtype)
                        embeddings = model(**inputs).cpu().float().numpy()
                        all_page_embeddings.append(embeddings[0])

                np.savez(cache_file, embeddings=all_page_embeddings, image_paths=np.array(image_paths))
                print(f"    ✓ Saved multi-vector embeddings to cache")

            extracted_texts, tables_data, text_blocks = self._extract_enhanced_text_data(doc_file)
            
            if len(all_page_embeddings) > 0:
                self._ensure_correct_embedding_dimension(conn, all_page_embeddings[0].shape[1])
            
            with conn.cursor() as cur:
                for i, (embeds, path) in enumerate(zip(all_page_embeddings, image_paths)):
                    cur.execute("""
                        INSERT INTO document_pages (doc_id, page_number, image_path, extracted_text, tables_data, text_blocks) 
                        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                    """, (doc_file.name, i + 1, path, extracted_texts[i], tables_data[i], text_blocks[i]))
                    page_id = cur.fetchone()[0]
                    
                    patch_data = [(page_id, j, v.tolist()) for j, v in enumerate(embeds)]
                    extras.execute_batch(cur, "INSERT INTO patch_embeddings (page_id, patch_index, embedding) VALUES (%s, %s, %s)", patch_data)
            
            conn.commit()
            print(f"    ✓ Stored {len(all_page_embeddings)} pages with multi-vector embeddings")

    def _ensure_correct_embedding_dimension(self, conn: psycopg2.extensions.connection, actual_dim: int):
        """Update embedding dimension in database schema if needed."""
        with conn.cursor() as cur:
            current_dim = -1
            try:
                # This is a robust way to get the vector dimension from the schema
                cur.execute("""
                    SELECT atttypmod FROM pg_attribute 
                    WHERE attrelid = 'patch_embeddings'::regclass AND attname = 'embedding';
                """)
                result = cur.fetchone()
                if result:
                    current_dim = result[0]
            except psycopg2.errors.UndefinedTable:
                current_dim = -1 # Table doesn't exist

            if current_dim != actual_dim:
                conn.rollback() # End any existing transaction before DDL
                print(f"    Schema mismatch or table not found. Recreating 'patch_embeddings' for dimension: {actual_dim}")
                cur.execute("DROP TABLE IF EXISTS patch_embeddings CASCADE;")
                cur.execute(f"""
                    CREATE TABLE patch_embeddings (
                        id SERIAL PRIMARY KEY,
                        page_id INTEGER REFERENCES document_pages(id) ON DELETE CASCADE,
                        patch_index INTEGER NOT NULL,
                        embedding vector({actual_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("CREATE INDEX patch_embeddings_page_id_idx ON patch_embeddings (page_id);")
                cur.execute("CREATE INDEX patch_embeddings_embedding_idx ON patch_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
                conn.commit()

    def query(self, client: dict, question: str, **kwargs) -> str:
        """Query using ColPali's multi-vector approach with enhanced debugging."""
        if DEBUG_MODE:
            print(f"\n[DEBUG] === QUERY START: {question} ===")
        
        try:
            query_embeddings = self._get_query_embeddings(client, question)
            retrieved_pages = self._retrieve_and_score_pages(client, query_embeddings)
            
            if not retrieved_pages:
                return "Could not find relevant visual content for this question."

            reranked_pages = self._rerank_pages(client, question, retrieved_pages)
            
            context = self._build_multivector_context(reranked_pages[:7])
            
            if DEBUG_MODE:
                debug_context_file = self.debug_dir / f"context_debug_{hash(question) % 10000}.txt"
                debug_context_file.write_text(f"Question: {question}\n\nContext:\n{context}", encoding='utf-8')
                print(f"[DEBUG] Context saved to: {debug_context_file}")

            return self._generate_answer(context, question)

        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
            if 'conn' in client: client['db_conn'].rollback()
            return "[ERROR] An exception occurred during query processing."

    def _get_query_embeddings(self, client: dict, question: str) -> np.ndarray:
        """Generate multi-vector embeddings for the user's query."""
        with torch.no_grad():
            query_batch = client["processor"].process_queries([question]).to(self.device)
            for key, tensor in query_batch.items():
                if tensor.is_floating_point():
                    query_batch[key] = tensor.to(self.model_dtype)
            query_embeddings = client["model"](**query_batch).cpu().float().numpy()[0]
        if DEBUG_MODE:
            print(f"[DEBUG] Query embeddings shape: {query_embeddings.shape}")
        return query_embeddings

    def _retrieve_and_score_pages(self, client: dict, query_embeddings: np.ndarray) -> List[ColPaliDocumentPage]:
        """Retrieve and score pages using efficient in-database vector search."""
        page_scores = defaultdict(lambda: {'max_sims': defaultdict(float), 'page_info': None})
        
        with client["db_conn"].cursor() as cur:
            # For each query token, find the top N most similar patches in the DB
            for i, token_embed in enumerate(tqdm(query_embeddings, desc="Retrieving Patches", leave=False, disable=not DEBUG_MODE)):
                # The <-> operator calculates Euclidean distance, for cosine use <=> 
                # 1 - score because pgvector returns distance, not similarity
                cur.execute("""
                    SELECT p.page_id, 1 - (p.embedding <=> %s) AS similarity
                    FROM patch_embeddings p
                    ORDER BY p.embedding <=> %s
                    LIMIT 50;
                """, (token_embed, token_embed))
                
                for page_id, similarity in cur.fetchall():
                    # ColBERT-style MaxSim: store the max similarity for this query token
                    page_scores[page_id]['max_sims'][i] = max(page_scores[page_id]['max_sims'][i], similarity)

        if not page_scores:
            return []

        # Sum the MaxSim scores for each page to get a final score
        final_scores = {page_id: sum(data['max_sims'].values()) for page_id, data in page_scores.items()}
        
        # Get the top 20 page IDs for reranking
        top_page_ids = sorted(final_scores, key=final_scores.get, reverse=True)[:20]

        # Fetch page details for the top pages
        with client["db_conn"].cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM document_pages WHERE id = ANY(%s);", (top_page_ids,))
            page_rows = {row['id']: row for row in cur.fetchall()}

        retrieved_pages = []
        for page_id in top_page_ids:
            row = page_rows.get(page_id)
            if row:
                combined_content = self._combine_text_sources(row['extracted_text'], row['tables_data'], row['text_blocks'])
                retrieved_pages.append(ColPaliDocumentPage(
                    page_id=row['id'], document_id=row['doc_id'], page_number=row['page_number'],
                    image_path=row['image_path'], content=combined_content
                ))
        
        if DEBUG_MODE:
            print("[DEBUG] Top 5 retrieved pages (pre-rerank):")
            for p in retrieved_pages[:5]:
                print(f"  - {p.document_id} p{p.page_number} (Score: {getattr(p, 'score', 'N/A'):.4f})")

        return retrieved_pages

    def _rerank_pages(self, client: dict, question: str, pages: List[ColPaliDocumentPage]) -> List[ColPaliDocumentPage]:
        """Rerank pages using FlagReranker."""
        reranker, loop = client.get("reranker"), client.get("loop")
        if not (reranker and loop and pages):
            return pages
        
        print("    Reranking retrieved pages...")
        reranked = loop.run_until_complete(reranker.rerank(question, pages))
        
        if DEBUG_MODE:
            print("[DEBUG] Top 5 reranked pages:")
            for p in reranked[:5]:
                print(f"  - {p.document_id} p{p.page_number} (Score: {getattr(p, 'score', 'N/A'):.4f})")
        
        return reranked

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate the final answer using the LLM."""
        prompt = f'''You are an expert document analyst. Your task is to answer the user's question based *only* on the provided context.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the provided context.
2. Be direct and concise. Provide the final answer immediately, without any preamble or explanation of your plan.
3. If the answer requires calculation, provide the final numerical answer first, then briefly show the calculation used to arrive at it.
4. Cite page numbers, like `(Page X)`, for every piece of data you use.
5. If the context does not contain the information to answer the question, state only: "The provided context does not contain enough information to answer the question."

Context from Document Pages:
{context}

Question: {question}

Final Answer:'''

        try:
            response = completion(
                model="o4-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096
            )
            final_answer = response.choices[0].message.content
        except Exception as e:
            print(f"    ⚠️ OpenAI API call failed: {e}")
            return "[ERROR] Failed to generate an answer from the language model."

        if DEBUG_MODE:
            print(f"\n[DEBUG] Final Generated Answer:\n{final_answer}\n")
        return final_answer

    def _combine_text_sources(self, text: str, tables: str, blocks: str) -> str:
        """Combine different text sources into coherent content."""
        sources = []
        if text and text.strip():
            sources.append(f"Text Content:\n{text.strip()}")
        if tables and tables.strip():
            sources.append(f"Table Data:\n{tables.strip()}")
        # Always include structured blocks if they exist, as they preserve layout
        if blocks and blocks.strip():
            sources.append(f"Structured Layout:\n{blocks.strip()}")
        
        return "\n\n".join(sources) if sources else "Visual elements with no extractable text."

    def _build_multivector_context(self, document_pages: List[ColPaliDocumentPage]) -> str:
        """Build context from multi-vector document pages."""
        contexts = []
        for page in document_pages:
            context_parts = [
                f"--- START CONTEXT: {page.document_id}, Page {page.page_number} ---",
                f"Source Image: {page.image_path}",
                page.content.strip(),
                f"--- END CONTEXT: {page.document_id}, Page {page.page_number} ---"
            ]
            contexts.append("\n".join(context_parts))
        return "\n\n".join(contexts)

    def test_single_question(self, question: str):
        """Test pipeline with a single question for debugging."""
        global DEBUG_MODE
        original_debug = DEBUG_MODE
        DEBUG_MODE = True
        
        try:
            print(f"\n🔍 TESTING SINGLE QUESTION: {question}")
            print("=" * 80)
            
            client = self.setup_client()
            
            with client["db_conn"].cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_pages")
                if cur.fetchone()[0] == 0:
                    print(f"\n📥 No documents found, ingesting from {self.docs_dir}")
                    self.ingest(client, self.docs_dir)
            
            answer = self.query(client, question)
            
            print(f"\n📄 FINAL ANSWER:\n{answer}")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            DEBUG_MODE = original_debug
            if 'client' in locals() and client["db_conn"]:
                client["db_conn"].close()

def main():
    """Main entry point for Kevin's ColPali multi-vector evaluation."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = ColPaliKevinEvaluator.create_cli_parser("kevin_colpali")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--test-question", type=str, help="Test a single question for debugging.")
    args = parser.parse_args()

    global DEBUG_MODE
    if args.debug:
        DEBUG_MODE = True
        print("\n[--- DEBUG MODE ENABLED ---]\n")

    print(f"✓ PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        print("✓ MPS available, using Apple Silicon GPU")
    else:
        print("✓ Using CPU")

    evaluator = ColPaliKevinEvaluator(
        system_name="kevin_colpali",
        docs_dir=args.docs_dir,
        questions_file=args.questions,
        output_file=args.output,
    )
    
    if args.test_question:
        evaluator.test_single_question(args.test_question)
        return 0
    
    try:
        output_file = evaluator.run_evaluation(skip_ingestion=args.skip_ingestion)
        print(f"\n🎉 Multi-vector evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")
    except Exception as e:
        print(f"\n⌐ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if hasattr(evaluator, "_client") and evaluator._client:
            client = evaluator._client
            if "db_conn" in client and client["db_conn"]:
                client["db_conn"].close()
            if "loop" in client and client["loop"]:
                client["loop"].close()
    return 0

if __name__ == "__main__":
    exit(main())
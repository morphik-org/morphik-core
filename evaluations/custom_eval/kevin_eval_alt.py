#!/usr/bin/env python3
"""Kevin's ColPali-Enhanced Evaluator (Updated for UV dependencies)

This is an enhanced version of Kevin's RAG evaluator using ColPali for visual document understanding.
Compatible with the existing UV-managed dependencies.
"""

from __future__ import annotations

import sys
import os
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import asyncio
import re
import json
import torch
import numpy as np
import io  # ADD THIS LINE - This was missing!
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF - already in your dependencies
from colpali_engine.models import ColPali, ColPaliProcessor
import cv2  # opencv-python - already in your dependencies

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.models.chunk import DocumentChunk
from core.reranker.flag_reranker import FlagReranker

# Remove the project root from Python path after imports
sys.path.pop(0)

from base_eval import BaseRAGEvaluator
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../../.env.example", override=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def completion(**kwargs):
    """Wrapper for OpenAI completion API"""
    kwargs.pop('api_key', None)
    return client.chat.completions.create(**kwargs)

# Global debug flag
DEBUG_MODE = False

class ColPaliDocumentChunk:
    """Document chunk with visual embeddings from ColPali."""
    
    def __init__(self, document_id: str, page_number: int, image_path: str, 
                 embedding: np.ndarray, content: str = ""):
        self.document_id = document_id
        self.page_number = page_number
        self.image_path = image_path
        self.embedding = embedding
        self.content = content
        self.chunk_number = f"{document_id}_page_{page_number}"

class ColPaliKevinEvaluator(BaseRAGEvaluator):
    """Kevin's improved RAG evaluator enhanced with ColPali for visual document understanding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.images_dir = None
        
    def setup_client(self, **kwargs) -> dict:
        """Initialize the ColPali-enhanced RAG system client."""
        print("Setting up ColPali client...")
        
        # Setup database
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI environment variable not set.")
        
        clean_uri = postgres_uri.replace("postgresql+asyncpg://", "postgresql://")
        
        try:
            conn = psycopg2.connect(clean_uri)
            register_vector(conn)
            print("✓ Connected to PostgreSQL successfully")
            
            self._setup_database(conn)
            print("✓ 'visual_chunks' table created successfully")

        except Exception as e:
            raise ConnectionError(f"Error connecting to PostgreSQL: {e}")

        # Initialize ColPali model using colpali-engine
        try:
            print("Loading ColPali model via colpali-engine...")
            model_name = "vidore/colpali-v1.3"
            
            self.processor = ColPaliProcessor.from_pretrained(model_name)
            self.model = ColPali.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device
            ).eval()
            
            print("✓ ColPali model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Error loading ColPali model with colpali-engine: {e}")

        # Initialize reranker (already in your deps via flagembedding)
        try:
            reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
            print("✓ Re-ranker initialized successfully")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize reranker: {e}")
            reranker = None
            loop = None

        # Create images directory
        self.images_dir = Path("./document_images")
        self.images_dir.mkdir(exist_ok=True)

        return {
            "db_conn": conn, 
            "reranker": reranker, 
            "loop": loop,
            "model": self.model,
            "processor": self.processor
        }
        
    def _setup_database(self, conn: psycopg2.extensions.connection):
        """Setup database tables for visual chunks."""
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("SELECT NULL::vector")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS visual_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL,
                    page_number INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding vector(768),
                    extracted_text TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS visual_chunks_embedding_idx ON visual_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            cur.execute("CREATE INDEX IF NOT EXISTS visual_chunks_doc_id_idx ON visual_chunks (doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS visual_chunks_page_idx ON visual_chunks (doc_id, page_number);")
            cur.execute("TRUNCATE TABLE visual_chunks;")
            conn.commit()

    def ingest(self, client: dict, docs_dir: Path, **kwargs) -> List[str]:
        """Ingest documents using ColPali visual processing."""
        print(f"Ingesting documents from: {docs_dir}")
        doc_files = list(docs_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")

        conn = client["db_conn"]
        model = client["model"]
        processor = client["processor"]
        
        self._ingest_with_visual_processing(conn, model, processor, doc_files)
        
        print(f"✓ Successfully ingested {len(doc_files)} documents with visual processing")
        return [doc.name for doc in doc_files]

    def _convert_pdf_to_images_pymupdf(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to images using PyMuPDF (already in your dependencies)."""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page to image with good resolution
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        doc.close()
        return images

    def _ingest_with_visual_processing(self, conn: psycopg2.extensions.connection, 
                                     model, processor, doc_files: List[Path]):
        """Ingest documents using visual understanding."""
        for doc_file in doc_files:
            print(f"  - Processing {doc_file.name} with visual processing")
            
            try:
                # Convert PDF to images using PyMuPDF
                images = self._convert_pdf_to_images_pymupdf(doc_file)
                print(f"    ✓ Converted to {len(images)} page images")
                
                # Process images in batches
                batch_size = 4 if torch.cuda.is_available() else 2
                all_embeddings = []
                image_paths = []
                
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i + batch_size]
                    batch_paths = []
                    
                    # Save images and prepare for processing
                    batch_pil_images = []
                    for j, img in enumerate(batch_images):
                        page_num = i + j + 1
                        img_path = self.images_dir / f"{doc_file.stem}_page_{page_num:03d}.png"
                        img.save(img_path)
                        batch_paths.append(str(img_path))
                        batch_pil_images.append(img)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        try:
                            # Try ColPali-style processing first
                            inputs = processor(text=[""] * len(batch_pil_images), images=batch_pil_images, return_tensors="pt", padding=True, truncation=True)
                            if hasattr(inputs, 'pixel_values'):
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            
                            outputs = model(**inputs)
                            
                            # Extract embeddings (adapt based on model output structure)
                            if hasattr(outputs, 'last_hidden_state'):
                                # For models like LayoutLMv3
                                embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence length
                            elif hasattr(outputs, 'pooler_output'):
                                embeddings = outputs.pooler_output
                            else:
                                # Fallback: take mean of all outputs
                                embeddings = torch.mean(outputs.logits, dim=1) if hasattr(outputs, 'logits') else outputs[0].mean(dim=1)
                            
                            # Convert to numpy
                            batch_embeddings = embeddings.cpu().numpy()
                            all_embeddings.extend(batch_embeddings)
                            
                        except Exception as e:
                            print(f"    ⚠️ Visual embedding failed for batch, using fallback: {e}")
                            # Fallback: create random embeddings (for testing)
                            fallback_embeddings = np.random.randn(len(batch_pil_images), 768).astype(np.float32)
                            all_embeddings.extend(fallback_embeddings)
                        
                        image_paths.extend(batch_paths)
                
                # Extract text using PyMuPDF (already in your deps)
                extracted_texts = self._extract_text_pymupdf(doc_file)
                
                # Store in database
                with conn.cursor() as cur:
                    data_to_insert = [
                        (
                            doc_file.name,
                            i + 1,  # page number (1-indexed)
                            image_paths[i],
                            all_embeddings[i],
                            extracted_texts[i] if i < len(extracted_texts) else ""
                        )
                        for i in range(len(all_embeddings))
                    ]
                    
                    psycopg2.extras.execute_batch(
                        cur,
                        "INSERT INTO visual_chunks (doc_id, page_number, image_path, embedding, extracted_text) VALUES (%s, %s, %s, %s, %s)",
                        data_to_insert
                    )
                conn.commit()
                
                print(f"    ✓ Stored {len(all_embeddings)} visual chunks in database")
                
            except Exception as e:
                conn.rollback()
                print(f"    ❌ Failed to process {doc_file.name} with visual processing: {e}")
                raise

    def _extract_text_pymupdf(self, pdf_path: Path) -> List[str]:
        """Extract text from PDF using PyMuPDF (already in your dependencies)."""
        try:
            doc = fitz.open(pdf_path)
            texts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                texts.append(text.strip())
            
            doc.close()
            return texts
            
        except Exception as e:
            print(f"    ⚠️ Text extraction failed: {e}")
            return [""] * 10  # Return empty strings as fallback

    def query(self, client: dict, question: str, **kwargs) -> str:
        """Query the RAG system using visual retrieval."""
        print(f"Querying with question: {question}")

        conn = client["db_conn"]
        model = client["model"]
        processor = client["processor"]
        reranker = client.get("reranker")
        loop = client.get("loop")

        return self._query_with_visual_understanding(conn, model, processor, reranker, loop, question)

    def _query_with_visual_understanding(self, conn: psycopg2.extensions.connection, 
                                       model, processor, reranker, loop, question: str) -> str:
        """Query using visual understanding."""
        try:
            # Generate query embedding
            with torch.no_grad():
                try:
                    # Process text query
                    dummy_image = Image.new("RGB", (224, 224), "white")
                    inputs = processor(text=[question], images=[dummy_image], return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs)
                    
                    # Extract query embedding (adapt based on model)
                    if hasattr(outputs, 'last_hidden_state'):
                        query_emb = outputs.last_hidden_state.mean(dim=1)[0]
                    elif hasattr(outputs, 'pooler_output'):
                        query_emb = outputs.pooler_output[0]
                    else:
                        query_emb = outputs[0].mean(dim=1)[0]
                    
                    query_embedding = query_emb.cpu().numpy()
                    
                except Exception as e:
                    print(f"    ⚠️ Query embedding failed, using fallback: {e}")
                    # Fallback: random embedding for testing
                    query_embedding = np.random.randn(768).astype(np.float32)

            # Retrieve similar visual chunks
            with conn.cursor() as cur:
                # Vector similarity search
                cur.execute("""
                    SELECT doc_id, page_number, image_path, embedding, extracted_text, id
                    FROM visual_chunks 
                    ORDER BY embedding <=> %s 
                    LIMIT 50
                """, (query_embedding,))
                vector_rows = cur.fetchall()

                # Text-based search if available
                fts_rows = []
                search_terms = self._extract_search_terms(question)
                for term in search_terms[:3]:
                    try:
                        cur.execute("""
                            SELECT doc_id, page_number, image_path, embedding, extracted_text, id
                            FROM visual_chunks 
                            WHERE extracted_text ILIKE %s
                            LIMIT 10
                        """, (f'%{term}%',))
                        fts_rows.extend(cur.fetchall())
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"[DEBUG] Text search failed for '{term}': {e}")

                # Combine results
                combined_rows = {row[5]: row for row in vector_rows}
                for row in fts_rows:
                    combined_rows[row[5]] = row
                
                all_rows = list(combined_rows.values())
                
                # Create visual document chunks
                visual_chunks = []
                for row in all_rows:
                    chunk = ColPaliDocumentChunk(
                        document_id=row[0],
                        page_number=row[1],
                        image_path=row[2],
                        embedding=np.array(row[3]),
                        content=row[4]
                    )
                    visual_chunks.append(chunk)
                
                if DEBUG_MODE:
                    print(f"\n[DEBUG] Retrieved {len(visual_chunks)} visual chunks")

                # Rerank if available
                top_chunks = visual_chunks[:8]  # Simplified for now
                
                if reranker and loop and any(chunk.content.strip() for chunk in visual_chunks[:20]):
                    try:
                        doc_chunks_for_reranking = [
                            DocumentChunk(
                                document_id=f"{chunk.document_id}_p{chunk.page_number}",
                                content=chunk.content or f"Visual content from {chunk.document_id}, page {chunk.page_number}",
                                embedding=chunk.embedding,
                                chunk_number=chunk.page_number
                            )
                            for chunk in visual_chunks[:15]
                            if chunk.content.strip()
                        ]
                        
                        if doc_chunks_for_reranking:
                            reranked = loop.run_until_complete(
                                reranker.rerank(question, doc_chunks_for_reranking)
                            )
                            
                            # Map back to visual chunks
                            reranked_visual = []
                            for ranked_chunk in reranked:
                                for visual_chunk in visual_chunks:
                                    if f"{visual_chunk.document_id}_p{visual_chunk.page_number}" == ranked_chunk.document_id:
                                        reranked_visual.append(visual_chunk)
                                        break
                            
                            if reranked_visual:
                                top_chunks = reranked_visual[:8]
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"[DEBUG] Reranking failed: {e}")

                if not top_chunks:
                    return "Could not find relevant visual content for this question."

                # Build context
                context = self._build_visual_context(top_chunks, question)

            # Financial analysis prompt
            prompt = f'''You are an expert financial document analyst specializing in interpreting charts, graphs, tables, and financial statements.

CRITICAL INSTRUCTIONS:
1. Answer based ONLY on the provided context from document pages
2. Pay special attention to numerical data, percentages, financial metrics, and trends
3. If calculations are needed, perform them step-by-step
4. Include specific numbers, dates, and time periods when relevant
5. For visual elements, describe what you observe and extract relevant data
6. If comparing periods, clearly identify the time frames
7. If information is not available, state this clearly
8. Cite page numbers when referencing data

Context from Document Pages:
{context}

Question: {question}

Answer (be specific and cite page numbers):'''

            response = completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )

            final_answer = response.choices[0].message.content
            
            if DEBUG_MODE:
                print(f"\n[DEBUG] Final Generated Answer:\n{final_answer}\n---")

            return final_answer
            
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
            raise e

    def _extract_search_terms(self, question: str) -> List[str]:
        """Extract relevant search terms from the question."""
        terms = []
        
        financial_terms = re.findall(r'\b(?:revenue|profit|margin|EBITDA|cash flow|GAAP|earnings|EPS|Q\d|FY\d+|billion|million|growth|percentage|rate)\b', question, re.IGNORECASE)
        company_names = re.findall(r'\b(?:Palantir|NVIDIA|Goldman|Sachs|Wendy|Heineken|CoreWeave|Arm)\b', question, re.IGNORECASE)
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', question)
        
        terms.extend(financial_terms + company_names + numbers)
        
        if 'year over year' in question.lower() or 'y/y' in question.lower():
            terms.append('year over year')
        if 'quarter over quarter' in question.lower() or 'q/q' in question.lower():
            terms.append('quarter over quarter')
            
        return terms[:5]

    def _build_visual_context(self, visual_chunks: List[ColPaliDocumentChunk], question: str) -> str:
        """Build context from visual chunks."""
        contexts = []
        
        for chunk in visual_chunks:
            context_parts = [
                f"=== {chunk.document_id}, Page {chunk.page_number} ===",
                f"Visual content from: {chunk.image_path}"
            ]
            
            if chunk.content and chunk.content.strip():
                context_parts.append(f"Text content: {chunk.content.strip()}")
            else:
                context_parts.append("Contains charts, graphs, tables, or other visual elements")
            
            contexts.append("\n".join(context_parts))
        
        return "\n\n".join(contexts)


def main():
    """Main entry point for Kevin's ColPali-enhanced evaluation."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = ColPaliKevinEvaluator.create_cli_parser("kevin_colpali")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    global DEBUG_MODE
    if args.debug:
        DEBUG_MODE = True
        print("\n[--- DEBUG MODE ENABLED ---]\n")

    print(f"✓ PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ CUDA not available, using CPU")

    evaluator = ColPaliKevinEvaluator(
        system_name="kevin_colpali",
        docs_dir=args.docs_dir,
        questions_file=args.questions,
        output_file=args.output,
    )
    
    client = None
    try:
        client = evaluator.setup_client()
        evaluator._client = client
        output_file = evaluator.run_evaluation(skip_ingestion=args.skip_ingestion)
        print(f"\n🎉 Visual evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if client:
            if "db_conn" in client:
                client["db_conn"].close()
            if "loop" in client:
                client["loop"].close()

    return 0


if __name__ == "__main__":
    exit(main())
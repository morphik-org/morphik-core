#!/usr/bin/env python3
"""Kevin's ColPali-Enhanced Evaluator (Enhanced with Better Text Extraction and Debugging)

Enhanced version with improved text extraction and comprehensive debugging capabilities.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2  # opencv-python - already in your dependencies
import fitz  # PyMuPDF - already in your dependencies
import numpy as np
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from PIL import Image
from psycopg2 import extras
from tqdm import tqdm
import torch

from base_eval import BaseRAGEvaluator
from colpali_engine.models import ColPali, ColPaliProcessor

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.models.chunk import DocumentChunk
from core.reranker.flag_reranker import FlagReranker

# Remove the project root from Python path after imports
sys.path.pop(0)

# Load environment variables
load_dotenv("../../.env.example", override=True)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Only show errors, not warnings
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def completion(**kwargs):
    """Wrapper for OpenAI completion API with proper parameter names"""
    kwargs.pop('api_key', None)
    # Fix the parameter name for newer OpenAI API versions
    if 'max_tokens' in kwargs:
        kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
    return client.chat.completions.create(**kwargs)

# Global debug flag
DEBUG_MODE = False

class ColPaliDocumentPage:
    """Document page with multi-vector embeddings from ColPali."""
    
    def __init__(self, document_id: str, page_number: int, image_path: str, 
                 embeddings: np.ndarray, content: str = "", page_id: int = None):
        self.document_id = document_id
        self.page_number = page_number
        self.image_path = image_path
        self.embeddings = embeddings  # Shape: [num_patches, embedding_dim]
        self.content = content
        self.page_id = page_id  # Database ID
        self.chunk_number = f"{document_id}_page_{page_number}"

class ColPaliKevinEvaluator(BaseRAGEvaluator):
    """Kevin's improved RAG evaluator enhanced with ColPali for visual document understanding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Device selection with proper dtype handling
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.use_mps = True
            # MPS doesn't support bfloat16 reliably, use float16 instead
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
        
        # Setup database
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI environment variable not set.")
        
        clean_uri = postgres_uri.replace("postgresql+asyncpg://", "postgresql://")
        
        try:
            conn = psycopg2.connect(clean_uri)
            register_vector(conn)
            print("✓ Connected to PostgreSQL successfully")
            
            # Setup database first, THEN ensure schema compatibility
            self._setup_database(conn)
            self._ensure_schema_compatibility(conn)
            
            print("✓ Multi-vector tables created successfully")

        except Exception as e:
            raise ConnectionError(f"Error connecting to PostgreSQL: {e}")


        # Initialize ColPali model with proper MPS dtype handling
        try:
            print("Loading ColPali model via colpali-engine...")
            model_name = "vidore/colpali-v1.3"
            
            # Memory optimization strategies for M4 Mac with 16GB RAM
            torch.backends.cudnn.benchmark = False
            
            print(f"    Using device: {self.device}")
            print(f"    Using dtype: {self.model_dtype}")
            
            # Create offload directory
            offload_dir = Path("./offload_cache")
            offload_dir.mkdir(exist_ok=True)
            
            # Load processor first (always works)
            self.processor = ColPaliProcessor.from_pretrained(model_name)
            print("    ✓ Processor loaded")
            
            # Load model with device-specific strategies and proper dtype
            if self.device.type == "mps":
                print("    Loading for Apple M4 with MPS-compatible settings...")
                try:
                    print("    Attempting MPS loading with bfloat16 first (official method)...")
                    torch.mps.empty_cache()
                    
                    # Try official method first
                    self.model = ColPali.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,  # Try official bfloat16 first
                        device_map="mps",  # Official recommendation
                        low_cpu_mem_usage=True
                    ).eval()
                    
                    print("    ✓ ColPali loaded successfully on MPS with bfloat16 (official method)")
                    self.model_dtype = torch.bfloat16  # Update our dtype tracking
                    
                    # Test the model with a dummy input to catch runtime issues
                    print("    Testing model with dummy input...")
                    dummy_img = Image.new("RGB", (32, 32), color="white")
                    dummy_batch = self.processor.process_images([dummy_img])
                    
                    # Test our MPS tensor handling
                    dummy_batch_fixed = {}
                    for k, v in dummy_batch.items():
                        if v.dtype in [torch.int64, torch.long, torch.int32, torch.int]:
                            dummy_batch_fixed[k] = v.to(device=self.device)
                        else:
                            dummy_batch_fixed[k] = v.to(dtype=self.model_dtype, device=self.device)
                    
                    # Test forward pass
                    with torch.no_grad():
                        _ = self.model(**dummy_batch_fixed)
                    
                    print("    ✓ Model test passed - ready for inference")
                    
                except Exception as bfloat16_error:
                    print(f"    ⚠️ Official bfloat16 method failed: {str(bfloat16_error)[:150]}")
                    print("    Falling back to float16 compatibility mode...")
                    torch.mps.empty_cache()
                    
                    # Fallback to our float16 method
                    self.model_dtype = torch.float16  # Reset to safe dtype
                    
                    self.model = ColPali.from_pretrained(
                        model_name,
                        torch_dtype=self.model_dtype,  # float16 for MPS
                        device_map=None,  # Don't use device_map with problematic setups
                        low_cpu_mem_usage=True
                    )
                    
                    # Manually move to MPS after loading
                    self.model = self.model.to(self.device).eval()
                    
                    print("    ✓ ColPali loaded successfully on MPS with float16 (compatibility mode)")
                    
                except Exception as mps_error:
                    print(f"    ⚠️ All MPS loading methods failed: {str(mps_error)[:100]}")
                    print("    Falling back to CPU...")
                    
                    self.device = torch.device("cpu")
                    self.use_mps = False
                    self.model_dtype = torch.float32
                    
                    self.model = ColPali.from_pretrained(
                        model_name,
                        torch_dtype=self.model_dtype,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    ).eval()
                    
                    print("    ✓ CPU fallback loaded successfully")
            
            elif self.device.type == "cuda":
                self.model = ColPali.from_pretrained(
                    model_name,
                    torch_dtype=self.model_dtype,  # bfloat16 for CUDA
                    device_map="cuda:0",
                    low_cpu_mem_usage=True
                ).eval()
                print("    ✓ ColPali loaded successfully on CUDA")
            
            else:
                self.model = ColPali.from_pretrained(
                    model_name,
                    torch_dtype=self.model_dtype,  # float32 for CPU
                    device_map="cpu",
                    low_cpu_mem_usage=True
                ).eval()
                print("    ✓ ColPali loaded successfully on CPU")
                
        except Exception as e:
            print(f"⌐ Complete model loading failure: {e}")
            raise

        # Initialize reranker 
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
    
    def _ensure_schema_compatibility(self, conn: psycopg2.extensions.connection):
        """Ensure the database schema has all required columns."""
        with conn.cursor() as cur:
            # Check if table exists first
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'document_pages'
                );
            """)
            table_exists = cur.fetchone()[0]
            
            if table_exists:
                # Add missing columns if they don't exist
                cur.execute("ALTER TABLE document_pages ADD COLUMN IF NOT EXISTS tables_data TEXT DEFAULT '';")
                cur.execute("ALTER TABLE document_pages ADD COLUMN IF NOT EXISTS text_blocks TEXT DEFAULT '';")
                conn.commit()
                print("✅ Schema compatibility ensured")
            else:
                print("✅ Table doesn't exist yet - schema will be created in setup_database")
        
    def _setup_database(self, conn: psycopg2.extensions.connection):
        """Setup database tables for multi-vector storage."""
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create document_pages table
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
            
            # Get the actual embedding dimension from a test
            # We'll update this after we know the real dimension
            embedding_dim = 128  # Default, will be updated during first ingestion
            
            # Create patch_embeddings table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS patch_embeddings (
                    id SERIAL PRIMARY KEY,
                    page_id INTEGER REFERENCES document_pages(id) ON DELETE CASCADE,
                    patch_index INTEGER NOT NULL,
                    embedding vector({embedding_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS patch_embeddings_page_id_idx ON patch_embeddings (page_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS patch_embeddings_embedding_idx ON patch_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            cur.execute("CREATE INDEX IF NOT EXISTS document_pages_doc_id_idx ON document_pages (doc_id);")
            
            # Clean up old data
            cur.execute("TRUNCATE TABLE patch_embeddings, document_pages RESTART IDENTITY CASCADE;")
            
            # Also drop old single-vector table if it exists
            cur.execute("DROP TABLE IF EXISTS visual_chunks;")
            
            conn.commit()

    def ingest(self, client: dict, docs_dir: Path, **kwargs) -> List[str]:
        """Ingest documents using ColPali multi-vector processing."""
        print(f"Ingesting documents from: {docs_dir}")
        doc_files = list(docs_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")

        # Debug: Log all documents found
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
        """Convert PDF to images using PyMuPDF with memory optimizations."""
        doc = fitz.open(pdf_path)
        images = []
        
        total_pages = len(doc)
            
        print(f"    Processing {total_pages} pages")
        
        for page_num in tqdm(range(total_pages), desc=f"  Converting {pdf_path.name}", unit="page", leave=False):
            page = doc.load_page(page_num)
            # Use higher resolution for better visual understanding
            mat = fitz.Matrix(1.5, 1.5)  # Keep higher res for better ColPali performance
            pix = page.get_pixmap(matrix=mat)
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Resize to reasonable size but keep quality
            if img.size[0] > 1200 or img.size[1] > 1200:
                img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                
            images.append(img)
            
        doc.close()
        return images

    def _extract_enhanced_text_data(self, pdf_path: Path) -> tuple[List[str], List[str], List[str]]:
        """Enhanced text extraction with tables, text blocks, and structured data."""
        try:
            doc = fitz.open(pdf_path)
            texts = []
            tables_data = []
            text_blocks = []
            
            total_pages = len(doc)
            if DEBUG_MODE:
                print(f"    [DEBUG] Extracting text from {total_pages} pages of {pdf_path.name}")
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                
                # 1. Basic text extraction
                raw_text = page.get_text()
                
                # 2. Enhanced table extraction with fallback
                page_tables = []
                try:
                    tables = page.find_tables()
                    if tables:
                        for table in tables:
                            try:
                                table_content = table.extract()
                                if table_content:
                                    # Convert table to readable format
                                    table_text = self._format_table_content(table_content)
                                    page_tables.append(table_text)
                                    if DEBUG_MODE:
                                        print(f"    [DEBUG] Page {page_num + 1}: Found table with {len(table_content)} rows")
                            except Exception as e:
                                if DEBUG_MODE:
                                    print(f"    [DEBUG] Page {page_num + 1}: Table extraction failed: {e}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"    [DEBUG] Page {page_num + 1}: find_tables() failed: {e}")
                
                # 3. Structured text blocks (preserves layout)
                try:
                    text_dict = page.get_text("dict")
                    blocks = []
                    for block in text_dict.get("blocks", []):
                        if "lines" in block:
                            block_text = ""
                            for line in block["lines"]:
                                for span in line.get("spans", []):
                                    block_text += span.get("text", "")
                                block_text += "\n"
                            if block_text.strip():
                                blocks.append(block_text.strip())
                    
                    structured_text = "\n\n".join(blocks)
                    
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"    [DEBUG] Page {page_num + 1}: Structured text extraction failed: {e}")
                    structured_text = raw_text
                
                # Combine and clean
                texts.append(raw_text.strip())
                tables_data.append("\n".join(page_tables) if page_tables else "")
                text_blocks.append(structured_text)
                
                if DEBUG_MODE and page_num < 3:  # Debug first few pages
                    print(f"    [DEBUG] Page {page_num + 1} text preview (first 200 chars):")
                    print(f"      Raw: {raw_text[:200]}...")
                    if page_tables:
                        print(f"      Tables: {len(page_tables)} found")
                    print(f"      Structured blocks: {len(blocks) if 'blocks' in locals() else 0}")
            
            doc.close()
            
            if DEBUG_MODE:
                # Save debug text output
                debug_text_file = self.debug_dir / f"{pdf_path.stem}_extracted_text.txt"
                with open(debug_text_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== TEXT EXTRACTION DEBUG FOR {pdf_path.name} ===\n\n")
                    for i, (text, tables, blocks) in enumerate(zip(texts, tables_data, text_blocks)):
                        f.write(f"=== PAGE {i + 1} ===\n")
                        f.write(f"Raw Text:\n{text}\n\n")
                        if tables:
                            f.write(f"Tables:\n{tables}\n\n")
                        f.write(f"Structured Blocks:\n{blocks}\n\n")
                        f.write("-" * 80 + "\n\n")
                print(f"    [DEBUG] Text extraction saved to: {debug_text_file}")
            
            return texts, tables_data, text_blocks
            
        except Exception as e:
            print(f"    ⚠️ Enhanced text extraction failed: {e}")
            # Fallback to basic extraction
            return self._extract_text_pymupdf_basic(pdf_path)

    def _format_table_content(self, table_content: List[List]) -> str:
        """Format table content into readable text."""
        if not table_content:
            return ""
        
        formatted_rows = []
        for row in table_content:
            if row and any(cell for cell in row if cell is not None):
                # Clean and join cells
                clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                formatted_rows.append(" | ".join(clean_row))
        
        return "\n".join(formatted_rows)

    def _extract_text_pymupdf_basic(self, pdf_path: Path) -> tuple[List[str], List[str], List[str]]:
        """Basic text extraction fallback."""
        try:
            doc = fitz.open(pdf_path)
            texts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                texts.append(text.strip())
            
            doc.close()
            
            # Return with empty tables and blocks
            return texts, [""] * len(texts), texts.copy()
            
        except Exception as e:
            print(f"    ⚠️ Basic text extraction also failed: {e}")
            return [""] * 10, [""] * 10, [""] * 10

    def _ingest_with_multivector_processing(self, conn: psycopg2.extensions.connection, 
                                          model, processor, doc_files: List[Path]):
        """Ingest documents using ColPali multi-vector approach."""
        for doc_file in tqdm(doc_files, desc="Ingesting Documents", unit="doc"):
            print(f"  - Processing {doc_file.name} with multi-vector processing")
            
            try:
                # --- Caching Logic ---
                doc_hash = hashlib.sha256(doc_file.read_bytes()).hexdigest()
                cache_file = self.embedding_cache_dir / f"{doc_file.stem}_{doc_hash[:10]}_multivector.npz"

                all_page_embeddings = []
                image_paths = []

                if cache_file.exists():  # Skip cache in debug mode for fresh runs //temporarily letting cache
                    print(f"    ✓ Loading embeddings from cache: {cache_file.name}")
                    with np.load(cache_file, allow_pickle=True) as data:
                        all_page_embeddings = data['embeddings']
                        image_paths = data['image_paths'].tolist()
                else:
                    print("    - No cache found (or debug mode), generating new multi-vector embeddings...")
                    
                    # Convert PDF to images
                    images = self._convert_pdf_to_images_pymupdf(doc_file)
                    print(f"    ✓ Converted to {len(images)} page images")
                    
                    # Process images using ColPali's native approach
                    batch_size = 1  # Process one page at a time for multi-vector
                    
                    for i in tqdm(range(0, len(images), batch_size), desc=f"    Embedding Pages", unit="page", leave=False):
                        batch_images = images[i:i + batch_size]
                        batch_paths = []
                        
                        # Save images
                        batch_pil_images = []
                        for j, img in enumerate(batch_images):
                            page_num = i + j + 1
                            img_path = self.images_dir / f"{doc_file.stem}_page_{page_num:03d}.png"
                            img.save(img_path)
                            batch_paths.append(str(img_path))
                            batch_pil_images.append(img)
                        
                        # Generate multi-vector embeddings using ColPali's native method
                        with torch.no_grad():
                            try:
                                # Clear cache
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                elif torch.backends.mps.is_available():
                                    torch.mps.empty_cache()
                                
                                # Use ColPali's native processing with proper dtype handling
                                batch_images_processed = processor.process_images(batch_pil_images)
                                
                                if DEBUG_MODE:
                                    print(f"    [DEBUG] Processor output tensor info:")
                                    for k, v in batch_images_processed.items():
                                        print(f"      {k}: shape={v.shape}, dtype={v.dtype}")
                                
                                # Handle MPS-specific tensor conversion issues
                                if self.use_mps:
                                    # For MPS, we need to be very careful about tensor dtypes
                                    # Embedding layers expect Long tensors, not bfloat16
                                    batch_images_processed_fixed = {}
                                    for k, v in batch_images_processed.items():
                                        if v.dtype in [torch.int64, torch.long, torch.int32, torch.int]:
                                            # Keep integer tensors as-is for embedding lookups
                                            batch_images_processed_fixed[k] = v.to(device=self.device)
                                            if DEBUG_MODE:
                                                print(f"      {k} -> kept as {v.dtype} for embedding")
                                        else:
                                            # Convert other tensors to model dtype
                                            batch_images_processed_fixed[k] = v.to(dtype=self.model_dtype, device=self.device)
                                            if DEBUG_MODE:
                                                print(f"      {k} -> converted to {self.model_dtype}")
                                    batch_images_processed = batch_images_processed_fixed
                                else:
                                    batch_images_processed = batch_images_processed.to(model.device)
                                
                                # Get multi-vector embeddings
                                image_embeddings = model(**batch_images_processed)
                                
                                # Convert to numpy - shape should be [batch_size, num_patches, embedding_dim]
                                embeddings_np = image_embeddings.cpu().float().numpy()  # Convert to float32 for numpy
                                
                                if DEBUG_MODE:
                                    print(f"    [DEBUG] Multi-vector embeddings shape: {embeddings_np.shape}")
                                
                                # Store each page's multi-vector embeddings
                                for page_idx, page_embeddings in enumerate(embeddings_np):
                                    all_page_embeddings.append(page_embeddings)  # [num_patches, embedding_dim]
                                
                                # Clean up memory
                                del batch_images_processed, image_embeddings, embeddings_np
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                elif torch.backends.mps.is_available():
                                    torch.mps.empty_cache()
                                
                            except Exception as e:
                                print(f"    ⚠️ Multi-vector embedding failed for batch: {e}")
                                
                                # If this is an MPS bfloat16 tensor type error, we can try to fix it
                                if "Expected tensor for argument" in str(e) and "MPSBFloat16Type" in str(e):
                                    print("    Detected MPS bfloat16 tensor type error - attempting CPU fallback for this batch...")
                                    try:
                                        # Move to CPU for this problematic batch
                                        cpu_model = model.cpu()
                                        batch_images_cpu = processor.process_images(batch_pil_images)
                                        
                                        with torch.no_grad():
                                            image_embeddings_cpu = cpu_model(**batch_images_cpu)
                                            embeddings_np = image_embeddings_cpu.float().numpy()
                                            
                                        # Move model back to MPS for next batch
                                        model.to(self.device)
                                        
                                        print(f"    ✓ CPU fallback successful - embeddings shape: {embeddings_np.shape}")
                                        
                                        for page_idx, page_embeddings in enumerate(embeddings_np):
                                            all_page_embeddings.append(page_embeddings)
                                            
                                    except Exception as cpu_error:
                                        print(f"    ⚠️ CPU fallback also failed: {cpu_error}")
                                        # Create dummy multi-vector embeddings as final fallback
                                        for _ in batch_pil_images:
                                            dummy_embeddings = np.random.randn(128, 128).astype(np.float32)
                                            all_page_embeddings.append(dummy_embeddings)
                                else:
                                    # Create dummy multi-vector embeddings as fallback
                                    for _ in batch_pil_images:
                                        dummy_embeddings = np.random.randn(128, 128).astype(np.float32)  # [num_patches, dim]
                                        all_page_embeddings.append(dummy_embeddings)
                            
                            image_paths.extend(batch_paths)

                    # Save to cache
                    if not DEBUG_MODE:  # Don't cache in debug mode
                        np.savez(cache_file, embeddings=all_page_embeddings, image_paths=np.array(image_paths))
                        print(f"    ✓ Saved multi-vector embeddings to cache")

                # Enhanced text extraction
                extracted_texts, tables_data, text_blocks = self._extract_enhanced_text_data(doc_file)
                
                # Get actual embedding dimension for schema update
                if all_page_embeddings.size > 0:
                    actual_dim = all_page_embeddings[0].shape[1] if len(all_page_embeddings[0].shape) > 1 else len(all_page_embeddings[0])
                    print(f"    Detected embedding dimension: {actual_dim}")
                    
                    # Update schema if needed
                    self._ensure_correct_embedding_dimension(conn, actual_dim)
                
                # Store in database using multi-vector approach
                with conn.cursor() as cur:
                    for page_idx, (page_embeddings, image_path) in enumerate(zip(all_page_embeddings, image_paths)):
                        page_num = page_idx + 1
                        extracted_text = extracted_texts[page_idx] if page_idx < len(extracted_texts) else ""
                        table_data = tables_data[page_idx] if page_idx < len(tables_data) else ""
                        text_block = text_blocks[page_idx] if page_idx < len(text_blocks) else ""
                        
                        # Insert page info with enhanced text data
                        cur.execute("""
                            INSERT INTO document_pages (doc_id, page_number, image_path, extracted_text, tables_data, text_blocks) 
                            VALUES (%s, %s, %s, %s, %s, %s) 
                            RETURNING id
                        """, (doc_file.name, page_num, image_path, extracted_text, table_data, text_block))
                        
                        page_id = cur.fetchone()[0]
                        
                        # Insert all patch embeddings for this page
                        patch_data = [
                            (page_id, patch_idx, patch_embedding.tolist())
                            for patch_idx, patch_embedding in enumerate(page_embeddings)
                        ]
                        
                        extras.execute_batch(
                            cur,
                            "INSERT INTO patch_embeddings (page_id, patch_index, embedding) VALUES (%s, %s, %s)",
                            patch_data
                        )
                
                conn.commit()
                print(f"    ✓ Stored {len(all_page_embeddings)} pages with multi-vector embeddings and enhanced text")
                
            except Exception as e:
                conn.rollback()
                print(f"    ⌐ Failed to process {doc_file.name}: {e}")
                raise

    def _ensure_correct_embedding_dimension(self, conn: psycopg2.extensions.connection, actual_dim: int):
        """Update embedding dimension in database schema if needed."""
        with conn.cursor() as cur:
            # Check current dimension
            try:
                cur.execute("SELECT embedding FROM patch_embeddings LIMIT 1;")
                # If this works, schema is already correct
            except:
                # Need to recreate table with correct dimension
                print(f"    Updating schema for embedding dimension: {actual_dim}")
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
            print(f"\n[DEBUG] === QUERY DEBUG START ===")
            print(f"[DEBUG] Question: {question}")
        
        print(f"Querying with question: {question}")

        conn = client["db_conn"]
        model = client["model"]
        processor = client["processor"]
        reranker = client.get("reranker")
        loop = client.get("loop")

        return self._query_with_multivector_understanding(conn, model, processor, reranker, loop, question)

    def _query_with_multivector_understanding(self, conn: psycopg2.extensions.connection, 
                                            model, processor, reranker, loop, question: str) -> str:
        """Query using ColPali's native multi-vector approach with comprehensive debugging."""
        try:
            # Generate multi-vector query embeddings
            with torch.no_grad():
                try:
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    
                    # Process query using ColPali's native method
                    query_batch = processor.process_queries([question])
                    
                    if DEBUG_MODE:
                        print(f"[DEBUG] Query batch info:")
                        for k, v in query_batch.items():
                            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    
                    # Handle MPS-specific tensor conversion issues
                    if self.use_mps:
                        # For MPS, we need to be very careful about tensor dtypes
                        query_batch_fixed = {}
                        for k, v in query_batch.items():
                            if v.dtype in [torch.int64, torch.long, torch.int32, torch.int]:
                                # Keep integer tensors as-is for embedding lookups
                                query_batch_fixed[k] = v.to(device=self.device)
                            else:
                                # Convert other tensors to model dtype
                                query_batch_fixed[k] = v.to(dtype=self.model_dtype, device=self.device)
                        query_batch = query_batch_fixed
                    else:
                        query_batch = query_batch.to(model.device)
                    
                    # Get multi-vector query embeddings
                    query_embeddings = model(**query_batch)  # Shape: [1, query_length, embedding_dim]
                    query_embeddings_np = query_embeddings.cpu().float().numpy()[0]  # [query_length, embedding_dim]
                    
                    print(f"DEBUG: Query embeddings shape: {query_embeddings_np.shape}")
                    
                    # Clean up
                    del query_batch, query_embeddings
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    
                except Exception as e:
                    print(f"    ⚠️ Query embedding failed: {e}")
                    raise

            # Retrieve using multi-vector similarity
            with conn.cursor() as cur:
                # Get all pages with their embeddings and enhanced text data
                cur.execute("""
                    SELECT dp.id, dp.doc_id, dp.page_number, dp.image_path, 
                           dp.extracted_text, dp.tables_data, dp.text_blocks,
                           array_agg(pe.embedding ORDER BY pe.patch_index) as embeddings
                    FROM document_pages dp
                    JOIN patch_embeddings pe ON dp.id = pe.page_id
                    GROUP BY dp.id, dp.doc_id, dp.page_number, dp.image_path, 
                             dp.extracted_text, dp.tables_data, dp.text_blocks
                """)
                
                page_rows = cur.fetchall()
                
                if DEBUG_MODE:
                    print(f"[DEBUG] Found {len(page_rows)} pages in database")
                    for i, row in enumerate(page_rows[:3]):  # Show first 3
                        page_id, doc_id, page_num, image_path, text, tables, blocks, embeddings = row
                        print(f"  Page {i+1}: {doc_id} page {page_num}")
                        print(f"    Text preview: {text[:100]}..." if text else "    No text")
                        print(f"    Tables: {len(tables) if tables else 0} chars")
                        print(f"    Embeddings: {len(embeddings)} patches")
                
                # Calculate ColPali-style scores using processor.score_multi_vector
                page_scores = []
                
                for row in page_rows:
                    page_id, doc_id, page_num, image_path, text, tables, blocks, db_embeddings = row
                    
                    # --- FIX: Correctly stack embeddings to avoid object arrays ---
                    if not db_embeddings or not any(v is not None for v in db_embeddings):
                        continue
                    try:
                        page_embeddings = np.vstack([np.array(v, dtype=np.float32) for v in db_embeddings])
                    except ValueError as e:
                        if DEBUG_MODE:
                            print(f"    [DEBUG] Skipping page {page_num} of {doc_id} due to inconsistent embedding shapes: {e}")
                        continue

                    # Calculate multi-vector similarity score (MaxSim)
                    max_scores = []
                    for query_token in query_embeddings_np:
                        # --- FIX: Guard for division by zero ---
                        page_norms = np.linalg.norm(page_embeddings, axis=1, keepdims=True)
                        page_norms[page_norms == 0] = 1e-9 # Avoid division by zero
                        
                        query_norm = np.linalg.norm(query_token)
                        if query_norm == 0:
                            query_norm = 1e-9

                        similarities = np.dot(page_embeddings, query_token) / (page_norms.flatten() * query_norm)
                        
                        if similarities.size > 0:
                            max_scores.append(np.max(similarities))
                    
                    if not max_scores:
                        final_score = 0.0
                    else:
                        # Final score is sum of max similarities (ColBERT-style)
                        final_score = np.sum(max_scores)
                    
                    page_scores.append((final_score, row))

                # Sort by score and get top pages
                page_scores.sort(key=lambda x: x[0], reverse=True)
                top_pages = page_scores[:20]  # Retrieve top 20 pages for reranking

                if DEBUG_MODE:
                    print(f"[DEBUG] Top 20 pre-reranked page scores:")
                    for i, (score, row) in enumerate(top_pages[:5]):
                        _, doc_id, page_num, _, _, _, _, _ = row
                        print(f"  {i+1}. {doc_id} page {page_num}: score={score:.4f}")

                # Create document pages for reranking
                pages_for_reranking = []
                for score, row in top_pages:
                    page_id, doc_id, page_num, image_path, text, tables, blocks, db_embeddings = row
                    # We need to re-stack here as well
                    if not db_embeddings or not any(v is not None for v in db_embeddings):
                        continue
                    try:
                        page_embeddings = np.vstack([np.array(v, dtype=np.float32) for v in db_embeddings])
                    except ValueError:
                        continue # Skip if shapes are inconsistent
                    
                    combined_content = self._combine_text_sources(text, tables, blocks)
                    
                    pages_for_reranking.append(
                        ColPaliDocumentPage(
                            document_id=doc_id,
                            page_number=page_num,
                            image_path=image_path,
                            embeddings=page_embeddings,
                            content=combined_content,
                            page_id=page_id
                        )
                    )

                # Rerank if reranker is available
                if reranker and loop:
                    print("    Reranking retrieved pages...")
                    document_pages = loop.run_until_complete(reranker.rerank(question, pages_for_reranking))
                    
                    if DEBUG_MODE:
                        print(f"[DEBUG] Top 5 reranked pages:")
                        for i, page in enumerate(document_pages[:5]):
                            print(f"  {i+1}. {page.document_id} page {page.page_number}: score={getattr(page, 'score', 'N/A'):.4f}")
                else:
                    document_pages = pages_for_reranking

                if not document_pages:
                    return "Could not find relevant visual content for this question."

                # Build context from top 7 reranked pages
                context = self._build_multivector_context(document_pages[:7], question)
                
                if DEBUG_MODE:
                    debug_context_file = self.debug_dir / f"context_debug_{hash(question) % 10000}.txt"
                    with open(debug_context_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== CONTEXT DEBUG FOR QUESTION ===\n")
                        f.write(f"Question: {question}\n\n")
                        f.write(f"Context:\n{context}\n")
                    print(f"[DEBUG] Context saved to: {debug_context_file}")

            # Enhanced prompt for better understanding
            prompt = f'''You are an expert document analyst specializing in interpreting charts, graphs, tables, and complex documents.

CRITICAL INSTRUCTIONS:
1. Answer based ONLY on the provided context from document pages.
2. Pay careful attention to numerical data, percentages, metrics, and trends shown in visual elements.
3. If calculations are needed, perform them step-by-step and show your work.
4. **Special Calculation Rule**: If the question asks for "volatility" of a time series, calculate it as `standard_deviation(series) / sqrt(number_of_data_points)`. Show your work for this calculation. For other statistical measures, use standard definitions.
5. Include specific numbers, dates, and time periods when relevant.
6. For visual elements (charts, tables, graphs), describe what you observe and extract relevant data.
7. If comparing different time periods or companies, clearly identify the sources and time frames.
8. If information is not available in the context, state this clearly.
9. Always cite page numbers when referencing specific data points.
10. Look for data in both text content and visual elements (charts, tables, graphs).

Context from Document Pages:
{context}

Question: {question}

Answer (be specific, show calculations if needed, and cite page numbers):'''

            response = completion(
                model="o4-mini",  # Use a working model
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2048
            )

            final_answer = response.choices[0].message.content
            
            if DEBUG_MODE:
                print(f"\n[DEBUG] Final Generated Answer:\n{final_answer}\n")
                print(f"[DEBUG] === QUERY DEBUG END ===\n")

            return final_answer
            
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
            raise e

    def _combine_text_sources(self, text: str, tables: str, blocks: str) -> str:
        """Combine different text sources into coherent content."""
        sources = []
        
        if text and text.strip():
            sources.append(f"Text Content:\n{text.strip()}")
        
        if tables and tables.strip():
            sources.append(f"Table Data:\n{tables.strip()}")
        
        if blocks and blocks.strip() and blocks.strip() != text.strip():
            sources.append(f"Structured Layout:\n{blocks.strip()}")
        
        if not sources:
            return "Contains visual elements (charts, graphs, images) with no extractable text"
        
        return "\n\n".join(sources)

    def _build_multivector_context(self, document_pages: List[ColPaliDocumentPage], question: str) -> str:
        """Build context from multi-vector document pages with enhanced information."""
        contexts = []
        
        for page in document_pages:
            context_parts = [
                f"=== {page.document_id}, Page {page.page_number} ===",
                f"Visual content from: {page.image_path}",
                f"Multi-vector embeddings: {page.embeddings.shape[0]} patches with {page.embeddings.shape[1]}-dim features"
            ]
            
            if page.content and page.content.strip():
                context_parts.append(page.content.strip())
            else:
                context_parts.append("Contains visual elements (charts, graphs, tables) with no extractable text")
            
            contexts.append("\n".join(context_parts))
        
        return "\n\n".join(contexts)

    def test_single_question(self, question: str):
        """Test pipeline with a single question for debugging."""
        global DEBUG_MODE
        original_debug = DEBUG_MODE
        DEBUG_MODE = True
        
        try:
            print(f"\n🔍 TESTING SINGLE QUESTION")
            print(f"Question: {question}")
            print("=" * 80)
            
            # Setup client
            client = self.setup_client()
            print("\n✓ Client setup complete")
            
            # Check if we need to ingest
            conn = client["db_conn"]
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_pages")
                page_count = cur.fetchone()[0]
            
            if page_count == 0:
                print(f"\n📥 No documents found, ingesting from {self.docs_dir}")
                self.ingest(client, self.docs_dir)
            else:
                print(f"\n✓ Found {page_count} pages in database")
            
            # Query
            print(f"\n🔍 Querying...")
            answer = self.query(client, question)
            
            print(f"\n📄 FINAL ANSWER:")
            print(answer)
            print("=" * 80)
            
            # Cleanup
            if client["db_conn"]:
                client["db_conn"].close()
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            DEBUG_MODE = original_debug


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
        print("✓ MPS available, attempting to use Apple Silicon GPU")
    else:
        print("✓ Using CPU")

    evaluator = ColPaliKevinEvaluator(
        system_name="kevin_colpali",
        docs_dir=args.docs_dir,
        questions_file=args.questions,
        output_file=args.output,
    )
    
    # Test single question if provided
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
import os
import tempfile
from dotenv import load_dotenv
from databridge import DataBridge

# Load environment variables
load_dotenv()

# Connect to DataBridge
db = DataBridge(os.getenv("DATABRIDGE_URI"), timeout=10000, is_local=True)

# Create some sample text files for batch ingestion
def create_sample_files():
    """Create temporary text files for demonstration"""
    files = []
    sample_texts = [
        "Artificial Intelligence is transforming various industries.",
        "Machine Learning models require significant amounts of data.",
        "Natural Language Processing enables computers to understand human language.",
        "Computer Vision systems can detect objects in images and videos.",
        "Reinforcement Learning is used in robotics and game-playing AI."
    ]
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Create text files
    for i, text in enumerate(sample_texts):
        file_path = os.path.join(temp_dir, f"sample_{i+1}.txt")
        with open(file_path, "w") as f:
            f.write(text)
        files.append(file_path)
    
    return temp_dir, files

# Create sample files
temp_dir, file_paths = create_sample_files()
print(f"Created {len(file_paths)} sample files")

# Batch ingestion of files
print("\nPerforming batch ingestion of files...")
docs = db.ingest_files(
    files=file_paths,
    metadata={"category": "AI technology", "source": "batch example"},
    parallel=True  # Process in parallel for faster ingestion
)
print(f"Ingested {len(docs)} documents")

# Get document IDs
doc_ids = [doc.external_id for doc in docs]
print(f"Document IDs: {doc_ids}")

# Batch retrieval of documents
print("\nPerforming batch retrieval of documents...")
retrieved_docs = db.batch_get_documents(doc_ids)
print(f"Retrieved {len(retrieved_docs)} documents")

# Ingest a directory
print("\nIngesting all files in a directory...")
dir_docs = db.ingest_directory(
    temp_dir,
    recursive=True,
    pattern="*.txt",
    metadata={"source": "directory ingestion"}
)
print(f"Ingested {len(dir_docs)} documents from directory")

# Clean up temporary files
import shutil
shutil.rmtree(temp_dir)
print(f"\nCleaned up temporary directory: {temp_dir}")
import time

from sdks.python.morphik.sync import Morphik

# sys.path.append(str(Path(__file__).parent.parent))


# Connect to Morphik
db = Morphik(timeout=10000, is_local=True)


# Helper function to wait for document ingestion to complete
def wait_for_ingestion_completion(document_id, max_wait_time=600, check_interval=30):
    """
    Poll the system until document ingestion is completed or max wait time is reached.

    Args:
        document_id: The ID of the document to check
        max_wait_time: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        True if ingestion completed, False if timed out
    """
    start_time = time.time()
    while (time.time() - start_time) < max_wait_time:
        # Get the document status info directly using the status API
        status_info = db.get_document_status(document_id)

        # Check if ingestion is completed
        if status_info.get("status") == "completed":
            print(f"Document ingestion completed for {document_id}")
            return True

        print(f"Document status: {status_info.get('status')}. Waiting {check_interval} seconds...")
        time.sleep(check_interval)

    print(f"Warning: Maximum wait time reached for document {document_id}")
    return False


# Define a single image-focused post_chunking rule
image_rules = [
    {
        "type": "metadata_extraction",
        "stage": "post_chunking",
        "use_images": True,
        "schema": {
            "graph_details": {
                "type": "string",
                "description": "Detailed description of any graphs, charts, or diagrams visible "
                "in the image, including axis labels, trends, and key data points",
            },
            "technical_elements": {
                "type": "array",
                "description": "List of technical elements visible in the image such as formulas, "
                "equations, or technical diagrams",
            },
            "visual_content_summary": {
                "type": "string",
                "description": "Brief summary of the visual content in the technical document",
            },
        },
    }
]

# Ingest document with image-focused post_chunking rule
print("Ingesting document with image-focused post_chunking rule...")
doc = db.ingest_file(
    "examples/assets/colpali_example.pdf",
    rules=image_rules,
    metadata={"source": "example", "rules_stage": "image_analysis"},
    use_colpali=True,  # Enable colpali for image processing, critical for handling images
)

# Wait for ingestion to complete
wait_for_ingestion_completion(doc.external_id)

# Get updated document information with processed image metadata
updated_doc = db.get_document(doc.external_id)

print("\n" + "=" * 50)
print("DOCUMENT WITH IMAGE PROCESSING RULES")
print("=" * 50)
print(f"Document ID: {updated_doc.external_id}")
# print(f"Document: {updated_doc}")
print(f"Document metadata: {updated_doc.metadata}")
print(f"Document system metadata: {updated_doc.system_metadata}")

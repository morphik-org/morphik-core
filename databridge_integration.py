from databridge import DataBridge
import os


class DatabridgeIntegrator:
    def __init__(self):
        """Initialize DataBridge client"""
        self.client = DataBridge()

    def ingest_document_content(
        self, content: str, image_descriptions: list, full_text: str, metadata: dict = None
    ):
        """
        Ingest document content with image descriptions into DataBridge.

        Args:
            content: Combined content of document and image descriptions
            image_descriptions: List of image descriptions
            full_text: Original document text
            metadata: Optional metadata about the document
        """
        # Combine full text with image descriptions
        combined_content = f"Document Text:\n{full_text}\n\nImage Descriptions:\n"
        for idx, desc in enumerate(image_descriptions, 1):
            combined_content += f"\nImage {idx}:\n{desc}\n"
            combined_content += "-" * 80 + "\n"

        # Add metadata if provided, or create default
        doc_metadata = metadata or {}
        doc_metadata.update(
            {
                "has_images": len(image_descriptions) > 0,
                "num_images": len(image_descriptions),
                "content_type": "document_with_images",
            }
        )

        # Ingest the combined content
        try:
            response = self.client.ingest_text(content=combined_content, metadata=doc_metadata)
            return response
        except Exception as e:
            print(f"Error ingesting document: {str(e)}")
            return None

    def search_documents(self, query: str, filters: dict = None, k: int = 4):
        """
        Search through ingested documents.

        Args:
            query: Search query string
            filters: Optional metadata filters
            k: Number of results to return
        """
        try:
            # Search through document chunks
            results = self.client.retrieve_chunks(
                query=query,
                filters=filters,
                k=k,
                min_score=0.7,  # Only return relatively confident matches
            )
            return results
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []

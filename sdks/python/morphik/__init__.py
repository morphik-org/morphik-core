"""
Morphik Python SDK for document ingestion and querying.
"""

from .async_ import AsyncMorphik
from .models import Document, DocumentQueryResponse, MigrationDocumentResult, MigrationResult, Summary
from .sync import Morphik

__all__ = [
    "Morphik",
    "AsyncMorphik",
    "Document",
    "Summary",
    "DocumentQueryResponse",
    "MigrationDocumentResult",
    "MigrationResult",
]

__version__ = "1.2.3"

"""
Morphik Python SDK for document ingestion and querying.
"""

from .async_ import AsyncMorphik
from .models import Document, DocumentQueryResponse
from .sync import Morphik

__all__ = [
    "Morphik",
    "AsyncMorphik",
    "Document",
    "DocumentQueryResponse",
]

__version__ = "1.0.1"

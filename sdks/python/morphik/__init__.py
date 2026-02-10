"""
Morphik Python SDK for document ingestion and querying.
"""

from .async_ import AsyncMorphik
from .models import Document, DocumentQueryResponse, Summary
from .sync import Morphik

__all__ = [
    "Morphik",
    "AsyncMorphik",
    "Document",
    "Summary",
    "DocumentQueryResponse",
]

__version__ = "1.2.2"

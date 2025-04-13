"""
Morphik Python SDK for document ingestion and querying.
"""

from .sync import Morphik
from .async_ import AsyncMorphik
from .models import Document

__all__ = [
    "Morphik",
    "AsyncMorphik",
    "Document",
]

__version__ = "0.1.2"

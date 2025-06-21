"""
Morphik Python SDK for document ingestion and querying.
"""

from .async_ import AsyncMorphik
from .models import Document
from .sync import Morphik

__all__ = [
    "Morphik",
    "AsyncMorphik",
    "Document",
]

__version__ = "0.2.2"

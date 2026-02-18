"""
Pure diagnostic helper functions for ingestion worker.

To facilitate testability, these are deterministic functions with no side effects.
"""

from typing import Optional


def should_warn_empty_parsing(text: str, skip_text_parsing: bool, xml_processing: bool) -> bool:
    """
    Return True if we should emit a warning about empty parsing results.

    This happens when:
    - We did not skip text parsing
    - We are not processing XML, which has its own chunk flow
    - The extracted text is empty or only contains whitespace
    """
    return not skip_text_parsing and not xml_processing and not text.strip()


def format_no_content_chunks_error(
    document_id: str,
    mime_type: Optional[str],
    using_colpali: bool,
    skip_text_parsing: bool,
    xml_processing: bool,
    text: Optional[str],
) -> str:
    """
    Format the error message for when no content chunks could be extracted.

    Returns the complete error message string with all diagnostic context.
    """
    text_length = len(text) if text else 0
    return (
        f"No content chunks (text or image) could be extracted from the document. "
        f"Context: document_id={document_id}, mime_type={mime_type}, "
        f"using_colpali={using_colpali}, skip_text_parsing={skip_text_parsing}, "
        f"xml_processing={xml_processing}, text_length={text_length}"
    )


__all__ = (
    "should_warn_empty_parsing",
    "format_no_content_chunks_error",
)

"""
Unit tests for ingestion worker diagnostic helper functions.

These tests validate the pure helper functions extracted from ingestion_worker.py
for testability. The helpers are deterministic with no side effects.
"""

from core.workers.ingestion_diagnostics import format_no_content_chunks_error, should_warn_empty_parsing


class TestShouldWarnEmptyParsing:
    """Tests for the should_warn_empty_parsing predicate."""

    def test_empty_string_warns_when_text_expected(self):
        """Empty string should trigger warning when text parsing was expected."""
        assert (
            should_warn_empty_parsing(
                text="",
                skip_text_parsing=False,
                xml_processing=False,
            )
            is True
        )

    def test_whitespace_only_warns_when_text_expected(self):
        """Whitespace only text should trigger warning when text parsing was expected."""
        assert (
            should_warn_empty_parsing(
                text="   \n\t  ",
                skip_text_parsing=False,
                xml_processing=False,
            )
            is True
        )

    def test_non_empty_text_does_not_warn(self):
        """Non-empty text should not trigger warning."""
        assert (
            should_warn_empty_parsing(
                text="Some actual content",
                skip_text_parsing=False,
                xml_processing=False,
            )
            is False
        )

    def test_skip_text_parsing_suppresses_warning(self):
        """When skip_text_parsing is True, no warning even for empty text."""
        assert (
            should_warn_empty_parsing(
                text="",
                skip_text_parsing=True,
                xml_processing=False,
            )
            is False
        )

    def test_xml_processing_suppresses_warning(self):
        """When xml_processing is True, no warning even for empty text."""
        assert (
            should_warn_empty_parsing(
                text="",
                skip_text_parsing=False,
                xml_processing=True,
            )
            is False
        )

    def test_both_flags_suppress_warning(self):
        """When both flags are True, no warning."""
        assert (
            should_warn_empty_parsing(
                text="",
                skip_text_parsing=True,
                xml_processing=True,
            )
            is False
        )

    def test_text_with_only_newlines_warns(self):
        """Text with only newlines should trigger warning."""
        assert (
            should_warn_empty_parsing(
                text="\n\n\n",
                skip_text_parsing=False,
                xml_processing=False,
            )
            is True
        )

    def test_text_with_content_after_whitespace_does_not_warn(self):
        """Text with content after whitespace should not warn."""
        assert (
            should_warn_empty_parsing(
                text="   content",
                skip_text_parsing=False,
                xml_processing=False,
            )
            is False
        )


class TestFormatNoContentChunksError:
    """Tests for the format_no_content_chunks_error formatter."""

    def test_message_contains_required_prefix(self):
        """Message should start with the expected error description."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="application/pdf",
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=False,
            text="",
        )
        assert msg.startswith("No content chunks (text or image) could be extracted from the document.")

    def test_message_contains_document_id(self):
        """Message should contain the document_id."""
        msg = format_no_content_chunks_error(
            document_id="my-unique-doc-id",
            mime_type="application/pdf",
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=False,
            text="",
        )
        assert "document_id=my-unique-doc-id" in msg

    def test_message_contains_mime_type(self):
        """Message should contain the mime_type."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="image/png",
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=False,
            text="",
        )
        assert "mime_type=image/png" in msg

    def test_message_contains_none_mime_type(self):
        """Message should show None when mime_type is None."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type=None,
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=False,
            text="",
        )
        assert "mime_type=None" in msg

    def test_message_contains_using_colpali_true(self):
        """Message should contain using_colpali=True when set."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="application/pdf",
            using_colpali=True,
            skip_text_parsing=False,
            xml_processing=False,
            text="",
        )
        assert "using_colpali=True" in msg

    def test_message_contains_using_colpali_false(self):
        """Message should contain using_colpali=False when not set."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="application/pdf",
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=False,
            text="",
        )
        assert "using_colpali=False" in msg

    def test_message_contains_skip_text_parsing(self):
        """Message should contain skip_text_parsing value."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="application/pdf",
            using_colpali=False,
            skip_text_parsing=True,
            xml_processing=False,
            text="",
        )
        assert "skip_text_parsing=True" in msg

    def test_message_contains_xml_processing(self):
        """Message should contain xml_processing value."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="application/pdf",
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=True,
            text="",
        )
        assert "xml_processing=True" in msg

    def test_text_none_gives_text_length_zero(self):
        """When text is None, text_length should be 0."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="application/pdf",
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=False,
            text=None,
        )
        assert "text_length=0" in msg

    def test_text_empty_string_gives_text_length_zero(self):
        """When text is empty string, text_length should be 0."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="application/pdf",
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=False,
            text="",
        )
        assert "text_length=0" in msg

    def test_text_with_content_gives_correct_length(self):
        """When text has content, text_length should be len(text)."""
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="application/pdf",
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=False,
            text="abc",
        )
        assert "text_length=3" in msg

    def test_text_with_longer_content(self):
        """Text length should reflect actual content length."""
        long_text = "x" * 12345
        msg = format_no_content_chunks_error(
            document_id="doc-123",
            mime_type="application/pdf",
            using_colpali=False,
            skip_text_parsing=False,
            xml_processing=False,
            text=long_text,
        )
        assert "text_length=12345" in msg

    def test_all_fields_present_in_context(self):
        """All required fields should be present in a single message."""
        msg = format_no_content_chunks_error(
            document_id="test-doc",
            mime_type="text/plain",
            using_colpali=True,
            skip_text_parsing=True,
            xml_processing=True,
            text="hello",
        )
        # Verify all fields are present
        assert "document_id=test-doc" in msg
        assert "mime_type=text/plain" in msg
        assert "using_colpali=True" in msg
        assert "skip_text_parsing=True" in msg
        assert "xml_processing=True" in msg
        assert "text_length=5" in msg
        # Verify it contains the Context prefix
        assert "Context:" in msg

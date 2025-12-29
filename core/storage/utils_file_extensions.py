import base64
import binascii
import mimetypes
from typing import Optional, Union

import filetype

_COLPALI_NATIVE_MIME_TYPES = {
    "application/pdf",
    "application/dicom",
    # Word documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    # PowerPoint presentations
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    # Excel spreadsheets
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.macroEnabled.12",
}

_GENERIC_CONTENT_TYPES = {
    "application/octet-stream",
    "binary/octet-stream",
    "application/x-octet-stream",
}


def _looks_like_text(content: bytes, sample_size: int = 8192) -> bool:
    if not content:
        return True
    sample = content[:sample_size]
    try:
        text_sample = sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text_sample:
        return True
    printable_chars = sum(1 for ch in text_sample if ch.isprintable() or ch.isspace())
    printable_ratio = printable_chars / len(text_sample)
    return printable_ratio >= 0.9


def detect_content_type(
    content: Optional[bytes] = None,
    filename: Optional[str] = None,
    content_type_hint: Optional[str] = None,
) -> str:
    """
    Detect the most likely MIME type using content bytes, filename, and an optional hint.
    """
    hint = None
    if content_type_hint:
        hint = content_type_hint.split(";", 1)[0].strip().lower()
        if hint in _GENERIC_CONTENT_TYPES:
            hint = None

    if content:
        kind = filetype.guess(content)
        if kind and kind.mime:
            return kind.mime

    if hint:
        return hint

    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed

    if content and _looks_like_text(content):
        return "text/plain"

    return "application/octet-stream"


def is_colpali_native_format(mime_type: Optional[str]) -> bool:
    if not mime_type:
        return False
    if mime_type.startswith("image/"):
        return True
    return mime_type in _COLPALI_NATIVE_MIME_TYPES


def detect_file_type(content: Union[str, bytes]) -> str:
    """
    Detect file type from content string and return appropriate extension.
    Content can be either base64 encoded or plain text.
    """
    # Special-case data URIs (e.g. "data:image/png;base64,...")
    if isinstance(content, str) and content.startswith("data:"):
        try:
            header, base64_part = content.split(",", 1)
            mime = header.split(":", 1)[1].split(";", 1)[0]
            extension_map = {
                "application/pdf": ".pdf",
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "image/tiff": ".tiff",
                "image/bmp": ".bmp",
                "image/svg+xml": ".svg",
                "video/mp4": ".mp4",
                "video/mpeg": ".mpeg",
                "video/quicktime": ".mov",
                "video/x-msvideo": ".avi",
                "video/webm": ".webm",
                "video/x-matroska": ".mkv",
                "video/3gpp": ".3gp",
                "text/plain": ".txt",
            }
            # Prefer mapping by MIME
            ext = extension_map.get(mime)
            if ext:
                return ext
            # Fallback to sniffing decoded bytes when MIME not recognized
            decoded_content = base64.b64decode(base64_part)
        except Exception:
            decoded_content = content.encode("utf-8")
    else:
        # Decode base64 if possible, otherwise treat as plain text
        if isinstance(content, bytes):
            decoded_content = content
        else:
            try:
                decoded_content = base64.b64decode(content)
            except binascii.Error:
                decoded_content = content.encode("utf-8")

    # Use filetype to detect mime type from content
    kind = filetype.guess(decoded_content)
    if kind is None:
        if isinstance(content, str):
            return ".txt"

        try:
            text_sample = decoded_content.decode("utf-8")
        except UnicodeDecodeError:
            return ".bin"

        if not text_sample:
            return ".txt"

        printable_chars = sum(1 for ch in text_sample if ch.isprintable() or ch.isspace())
        printable_ratio = printable_chars / len(text_sample)
        return ".txt" if printable_ratio >= 0.9 else ".bin"

    # Map mime type to extension
    extension_map = {
        "application/pdf": ".pdf",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
        "image/svg+xml": ".svg",
        "video/mp4": ".mp4",
        "video/mpeg": ".mpeg",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/webm": ".webm",
        "video/x-matroska": ".mkv",
        "video/3gpp": ".3gp",
        "text/plain": ".txt",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    }
    return extension_map.get(kind.mime, ".bin")

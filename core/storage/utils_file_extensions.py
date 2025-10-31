import base64
import binascii
from typing import Union

import filetype


def detect_file_type(content: Union[str, bytes]) -> str:
    """
    Detect file type from content string and return appropriate extension.
    Content can be either base64 encoded or plain text.
    """
    # Decode base64 content
    if isinstance(content, bytes):
        decoded_content = content
    else:
        try:
            decoded_content = base64.b64decode(content)
        except binascii.Error:
            # If not base64, treat as plain text
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

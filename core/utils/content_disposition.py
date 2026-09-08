"""Content-Disposition headers for file responses."""

from typing import Literal, Optional
from urllib.parse import quote


def build_content_disposition(filename: Optional[str], disposition: Literal["inline", "attachment"] = "inline") -> str:
    """Build an ASCII header with a fallback and RFC 6266/8187 UTF-8 filename*.

    Replace control characters in the suggested name without changing stored
    metadata. Keep Unicode and punctuation in filename*, but replace non-ASCII,
    quotes, path separators and percent signs in the legacy fallback to avoid
    quoted-string escaping and inconsistent browser percent-decoding.
    """
    if disposition not in ("inline", "attachment"):
        raise ValueError("Unsupported content disposition")

    filename = "".join("_" if ord(char) < 32 or ord(char) == 127 else char for char in (filename or "document"))
    fallback = "".join(char if 32 <= ord(char) < 127 and char not in '"\\/%' else "_" for char in filename)
    return f"{disposition}; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"

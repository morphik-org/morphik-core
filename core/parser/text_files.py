"""Plain-text attachments: which extensions are read as text, and how their bytes are decoded."""

import codecs
from pathlib import Path

# Read directly instead of going through Docling. Source code and config files belong here: Docling rejects them,
# and their text is all there is to index.
PLAIN_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".csv",
    ".tsv",
    ".log",
    ".rst",
    ".yaml",
    ".yml",
    ".sql",
    ".py",
    ".ps1",
    ".sh",
    ".bat",
    ".cmd",
    ".js",
    ".ts",
    ".css",
    ".ini",
    ".cfg",
    ".conf",
    ".toml",
    ".properties",
}

_BOMS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    # UTF-32 LE starts with the UTF-16 LE BOM, so it has to be checked first.
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def is_plain_text_filename(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in PLAIN_TEXT_EXTENSIONS


def decode_text(data: bytes) -> str:
    """Decode a text file. SQL Server Management Studio and PowerShell save UTF-16 with a BOM, which utf-8 rejects
    and latin-1 turns into text with a NUL after every letter."""
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                break
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        # cp1252 leaves five bytes undefined; latin-1 maps every byte.
        return data.decode("latin-1")

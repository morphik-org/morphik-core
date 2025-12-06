"""Utility functions for folder operations."""

from typing import List, Optional, Union


def normalize_folder_name(folder_name: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    """Convert string 'null' to None for folder_name parameter."""
    if folder_name is None:
        return None
    if isinstance(folder_name, str):
        return None if folder_name.lower() == "null" else folder_name
    if isinstance(folder_name, list):
        return [None if f.lower() == "null" else f for f in folder_name]
    return folder_name


def normalize_folder_path(path: Optional[str]) -> Optional[str]:
    """
    Normalize a folder path into canonical form (leading slash, no trailing slash).

    Rules:
    - Trim outer whitespace.
    - Strip leading/trailing slashes.
    - Reject empty segments and ".." segments.
    - Collapse duplicate slashes by ignoring empty segments between separators.
    - Preserve case and internal spaces.
    """
    if path is None:
        return None

    if not isinstance(path, str):
        raise ValueError("Folder path must be a string")

    cleaned = path.strip()
    if cleaned == "":
        raise ValueError("Folder path cannot be empty")

    # Remove surrounding slashes so we can inspect individual segments.
    cleaned = cleaned.strip("/")

    # Split into segments, dropping empty parts produced by duplicate slashes.
    segments = [segment for segment in cleaned.split("/") if segment != ""]

    if not segments:
        # Input was only slashes (e.g., "/"), treat as root path.
        return "/"

    # Disallow path traversal or zero-length segments.
    for segment in segments:
        if segment == "..":
            raise ValueError("Folder path cannot contain '..' segments")
        if segment.strip() == "":
            raise ValueError("Folder path cannot contain empty segments")

    canonical = "/" + "/".join(segments)
    return canonical


def normalize_folder_selector(folder: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    """
    Normalize folder selectors coming from request parameters (string or list).

    - Converts the string "null" (case-insensitive) to None.
    - Applies `normalize_folder_path` to non-empty strings.
    - Preserves explicit None values inside lists.
    """
    if folder is None:
        return None

    def _normalize_single(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Folder selector values must be strings or None")
        if value.lower() == "null":
            return None
        return normalize_folder_path(value)

    if isinstance(folder, list):
        return [_normalize_single(item) for item in folder]

    if isinstance(folder, str):
        return _normalize_single(folder)

    raise ValueError("Folder selector must be a string, list of strings, or None")

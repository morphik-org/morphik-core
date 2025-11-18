from __future__ import annotations

import hashlib

TELEMETRY_SIGNATURE_SALT = "morphik-telemetry-upload-v1"


def compute_telemetry_signature(installation_id: str) -> str:
    """Match the usage backend signature requirements."""
    value = f"{installation_id}{TELEMETRY_SIGNATURE_SALT}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()

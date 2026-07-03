from typing import Optional

from fastapi import HTTPException


LOCAL_URI_PASSWORD_DISABLED_DETAIL = "LOCAL_URI_PASSWORD is not configured; /local/generate_uri is disabled"


def require_local_uri_password_configured(local_uri_password: Optional[str]) -> None:
    if not local_uri_password:
        raise HTTPException(status_code=503, detail=LOCAL_URI_PASSWORD_DISABLED_DETAIL)

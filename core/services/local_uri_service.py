import logging
import uuid
from typing import Any, Optional

from core.models.apps import AppModel

logger = logging.getLogger(__name__)


def _safe_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    """Convert a string to UUID when possible; otherwise return None."""
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        logger.debug("Value %s is not a valid UUID; storing NULL in apps.user_id", value)
        return None


async def persist_local_app_record(
    *,
    database: Any,
    app_id: str,
    user_name: str,
    uri: str,
    token_version: int,
) -> None:
    """Create or update the apps-table row required for local JWT auth."""
    user_uuid = _safe_uuid(user_name)

    async with database.async_session() as session:
        app_record = await session.get(AppModel, app_id)
        if app_record is None:
            app_record = AppModel(
                app_id=app_id,
                user_id=user_uuid,
                created_by_user_id=user_name,
                name=user_name,
                uri=uri,
                token_version=token_version,
            )
            session.add(app_record)
        else:
            if user_uuid:
                app_record.user_id = user_uuid
            app_record.created_by_user_id = user_name
            app_record.name = user_name
            app_record.uri = uri
            app_record.token_version = token_version

        await session.commit()

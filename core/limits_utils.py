import logging
from math import ceil
from typing import Optional

# Imports needed for check_and_increment_limits
from fastapi import HTTPException

from core.config import get_settings
from core.models.auth import AuthContext  # Assuming AuthContext is needed
from core.models.tiers import AccountTier
from core.services.user_service import UserService

# Initialize logger
logger = logging.getLogger(__name__)

# --- Shared UserService instance ---
_user_service_instance: Optional[UserService] = None
_user_service_initialized: bool = False


async def get_initialized_user_service() -> UserService:
    """Provides a shared, initialized instance of UserService."""
    global _user_service_instance, _user_service_initialized

    if _user_service_instance is None:
        _user_service_instance = UserService()

    if not _user_service_initialized:
        logger.info("Initializing shared UserService instance for limits_utils...")
        if await _user_service_instance.initialize():
            _user_service_initialized = True
            logger.info("Shared UserService instance initialized successfully.")
        else:
            # If initialization fails, log the error.
            # Subsequent calls to check_limit/record_usage might fail or operate unexpectedly
            # if the database isn't correctly set up.
            logger.error(
                "Failed to initialize shared UserService instance in limits_utils. Limits checking may be impaired."
            )
            # We still return the instance; the UserService.initialize() itself logs errors from UserLimitsDatabase.

    return _user_service_instance


# ---------------------------------------------------------------------------
# Helper constants & functions shared by ingestion and quota enforcement
# ---------------------------------------------------------------------------

# Average characters per token in typical English text.  This is only
# a heuristic but matches the value used historically inside
# DocumentService.
CHARS_PER_TOKEN = 4
# Number of tokens we treat as one "page" when counting ingest usage.
TOKENS_PER_PAGE = 630


def estimate_pages_by_chars(char_len: int) -> int:
    """Return the number of pages represented by *char_len* characters.

    The function uses the same heuristic (4 chars per token, 630 tokens per
    page) that the application has adopted historically.  The value is always
    at least **one** so even very small ingests are billed fairly.
    """
    if char_len <= 0:
        return 1
    pages = ceil(char_len / (CHARS_PER_TOKEN * TOKENS_PER_PAGE))
    return max(1, pages)


async def get_org_from_app(app_id: str) -> Optional[str]:
    """Get the organization ID that owns an app."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from core.models.apps import AppModel

    settings = get_settings()
    engine = create_async_engine(settings.POSTGRES_URI)

    async with AsyncSession(engine) as session:
        try:
            result = await session.execute(select(AppModel.org_id).where(AppModel.app_id == app_id))
            row = result.first()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Error getting org_id for app {app_id}: {e}")
            return None
        finally:
            await engine.dispose()


async def check_and_increment_limits(
    auth: AuthContext,  # Explicitly type hint auth
    limit_type: str,
    value: int = 1,
    document_id: str | None = None,
    verify_only: bool = False,
    use_colpali: bool | None = None,
    colpali_chunks_count: int | None = None,
) -> None:
    """Check or record usage against organization limits.

    When *verify_only* is **True** the function **only** validates that the
    operation **would** fit within the org-tier limits and raises an
    ``HTTPException`` (429) if it does not.  **No** usage is recorded in this
    mode which allows callers to perform a dry-run check before executing a
    potentially expensive operation.

    With the default ``verify_only=False`` the helper behaves exactly as
    before – it first verifies the limits and, on success, immediately records
    the usage so that subsequent calls observe the updated counters.

    Args:
        auth: Authentication context carrying ``user_id`` / tier info
        limit_type: Category to check (query, ingest, storage_file …)
        value: The amount to charge towards the limit (e.g. bytes, pages)
        document_id: Optional document identifier used when metering pages
        verify_only: If **True** perform the check without updating counters
        use_colpali: If **True** use ColPali information for ingest limit
        colpali_chunks_count: Number of chunks in ColPali for ingest limit

    Raises:
        HTTPException: 429 when the requested usage exceeds the tier limits
    """
    settings = get_settings()

    # Skip limit checking in self-hosted mode
    if settings.MODE == "self_hosted":
        return

    # Determine which ID to use for limits lookup
    # If we have an app_id, get the org_id and use that
    # Otherwise fall back to user_id for backward compatibility
    limits_id = auth.user_id  # Default to user_id

    if auth.app_id:
        # Get org_id from app_id
        org_id = await get_org_from_app(auth.app_id)
        if org_id:
            limits_id = org_id  # Use org_id as the "user_id" in user_limits table
            logger.debug(f"Using org_id {org_id} for limits check (from app {auth.app_id})")
        else:
            logger.warning(f"No org found for app {auth.app_id}, using user_id {auth.user_id}")

    if not limits_id:
        logger.warning("No ID available for limits check, skipping")
        return

    # Get the shared, initialized UserService instance
    user_service = await get_initialized_user_service()

    # Get limits data (the "user_id" field in user_limits table may actually contain org_id)
    user_data = await user_service.get_user_limits(limits_id)
    if not user_data:
        # Create limits if they don't exist (defaults to free tier)
        await user_service.create_user(limits_id)
        user_data = await user_service.get_user_limits(limits_id)
        if not user_data:
            logger.error(f"Failed to create limits for {limits_id}")
            return

    tier = user_data.get("tier", AccountTier.FREE)

    # Determine the actual value to check/record based on limit type and ColPali status
    value_to_use = value
    if limit_type == "ingest":
        if use_colpali is True and colpali_chunks_count is not None:
            # Use the actual count from ColPali chunks, ensuring at least 1 page.
            value_to_use = max(1, colpali_chunks_count)
            logger.debug(f"Using ColPali chunk count for ingest limit check/record: {value_to_use} pages.")
        else:
            # Use the estimated character count, ensuring at least 1 page.
            value_to_use = max(1, value)
            logger.debug(f"Using estimated character count for ingest limit check/record: {value_to_use} pages.")

    # Only apply limits to free tier users
    if tier != AccountTier.FREE:
        # For paid tiers just record usage (metering) unless caller only
        # requested a dry-run check.
        if not verify_only:
            try:
                # Use value_to_use for recording
                await user_service.record_usage(limits_id, limit_type, value_to_use, document_id)
            except Exception as e:
                logger.error("Failed to record usage: %s", e)
        return

    # For free tier, check if within limits using value_to_use
    within_limits = await user_service.check_limit(limits_id, limit_type, value_to_use)

    if not within_limits:
        # Map limit types to appropriate error messages
        storage_message = (
            "Storage file count limit exceeded for your free tier. "
            "Please delete some files or upgrade to remove limits."
        )
        limit_type_messages = {
            "query": "Query limit exceeded for your free tier. Please upgrade to remove limits.",
            "ingest": "Ingest limit exceeded for your free tier. Please upgrade to remove limits.",
            "storage_file": storage_message,
            "storage_size": (
                "Storage size limit exceeded for your free tier. "
                "Please delete some files or upgrade to remove limits."
            ),
            "graph": "Graph creation limit exceeded for your free tier. Please upgrade to remove limits.",
            "cache": "Cache creation limit exceeded for your free tier. Please upgrade to remove limits.",
            "cache_query": "Cache query limit exceeded for your free tier. Please upgrade to remove limits.",
            "agent": "Agent call limit exceeded for your free tier. Please upgrade to remove limits.",
        }

        # Get message for the limit type or use default message
        default_message = "Limit exceeded for your free tier. Please upgrade to remove limits."
        detail = limit_type_messages.get(limit_type, default_message)

        # Raise the exception with appropriate message
        raise HTTPException(status_code=429, detail=detail)

    # Record usage unless this was only a verification pass.
    if not verify_only:
        try:
            # Use value_to_use for recording
            await user_service.record_usage(limits_id, limit_type, value_to_use, document_id)
        except Exception as e:
            # Just log if recording usage fails, don't fail the operation
            logger.error("Failed to record usage: %s", e)

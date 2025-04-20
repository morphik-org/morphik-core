import logging

# Initialize logger
logger = logging.getLogger(__name__)


async def check_and_increment_limits(auth, limit_type: str, value: int = 1, document_id: str = None) -> None:
    """
    Check if the user is within limits for an operation and increment usage.
    Limits are only applied to the free tier; other tiers have no limits.
    For non-free tiers, usage is tracked and metered through Stripe for billing purposes.

    Args:
        auth: Authentication context with user_id
        limit_type: Type of limit to check (query, ingest, storage_file, storage_size, graph, cache)
        value: Value to check against limit (e.g., file size for storage_size)
        document_id: Optional document ID for tracking in Stripe (used for ingest operations)

    Raises:
        HTTPException: If the user exceeds limits (free tier only)
    """
    from fastapi import HTTPException

    from core.config import get_settings
    from core.models.tiers import AccountTier

    settings = get_settings()

    # Skip limit checking in self-hosted mode
    if settings.MODE == "self_hosted":
        return

    # Check if user_id is available
    if not auth.user_id:
        logger.warning("User ID not available in auth context, skipping limit check")
        return

    # Initialize user service
    from core.services.user_service import UserService

    user_service = UserService()
    await user_service.initialize()

    # Get user data to check tier
    user_data = await user_service.get_user_limits(auth.user_id)
    if not user_data:
        # Create user limits if they don't exist (defaults to free tier)
        await user_service.create_user(auth.user_id)
        user_data = await user_service.get_user_limits(auth.user_id)
        if not user_data:
            logger.error(f"Failed to create user limits for user {auth.user_id}")
            return

    tier = user_data.get("tier", AccountTier.FREE)

    # Only apply limits to free tier users
    if tier != AccountTier.FREE:
        # For non-free tiers, just record usage without checking limits
        try:
            await user_service.record_usage(auth.user_id, limit_type, value, document_id)
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
        return

    # For free tier, check if user is within limits
    within_limits = await user_service.check_limit(auth.user_id, limit_type, value)

    if not within_limits:
        # Map limit types to appropriate error messages
        limit_type_messages = {
            "query": "Query limit exceeded for your free tier. Please upgrade to remove limits.",
            "ingest": "Ingest limit exceeded for your free tier. Please upgrade to remove limits.",
            "storage_file": "Storage file count limit exceeded for your free tier. Please delete some files or upgrade to remove limits.",
            "storage_size": "Storage size limit exceeded for your free tier. Please delete some files or upgrade to remove limits.",
            "graph": "Graph creation limit exceeded for your free tier. Please upgrade to remove limits.",
            "cache": "Cache creation limit exceeded for your free tier. Please upgrade to remove limits.",
            "cache_query": "Cache query limit exceeded for your free tier. Please upgrade to remove limits.",
        }

        # Get message for the limit type or use default message
        detail = limit_type_messages.get(
            limit_type, "Limit exceeded for your free tier. Please upgrade to remove limits."
        )

        # Raise the exception with appropriate message
        raise HTTPException(status_code=429, detail=detail)

    # Record usage asynchronously
    try:
        await user_service.record_usage(auth.user_id, limit_type, value, document_id)
    except Exception as e:
        # Just log if recording usage fails, don't fail the operation
        logger.error(f"Failed to record usage: {e}")

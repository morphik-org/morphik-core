import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional

import jwt

from ..config import get_settings
from ..database.user_limits_db import UserLimitsDatabase
from ..models.tiers import AccountTier, get_tier_limits

logger = logging.getLogger(__name__)


class UserService:
    """Service for managing user limits and usage."""

    def __init__(self):
        """Initialize the UserService."""
        self.settings = get_settings()
        self.db = UserLimitsDatabase(uri=self.settings.POSTGRES_URI)

    async def initialize(self) -> bool:
        """Initialize database tables."""
        return await self.db.initialize()

    async def get_user_limits(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user limits information."""
        return await self.db.get_user_limits(user_id)

    async def create_user(self, user_id: str) -> bool:
        """Create a new user with FREE tier."""
        return await self.db.create_user_limits(user_id, tier=AccountTier.FREE)

    async def update_user_tier(self, user_id: str, tier: str, custom_limits: Optional[Dict[str, Any]] = None) -> bool:
        """Update user tier and custom limits."""
        return await self.db.update_user_tier(user_id, tier, custom_limits)

    async def register_app(self, user_id: str, app_id: str) -> bool:
        """
        Register an app for a user.

        Creates user limits record if it doesn't exist.
        """
        # First check if user limits exist
        user_limits = await self.db.get_user_limits(user_id)

        # If user limits don't exist, create them first
        if not user_limits:
            logger.info(f"Creating user limits for user {user_id}")
            success = await self.db.create_user_limits(user_id, tier=AccountTier.FREE)
            if not success:
                logger.error(f"Failed to create user limits for user {user_id}")
                return False

        # Now register the app
        return await self.db.register_app(user_id, app_id)

    async def check_limit(self, user_id: str, limit_type: str, value: int = 1) -> bool:
        """
        Check if a user's operation is within limits when given value is considered.
        Limits are only applied to the free tier; other tiers have no limits.

        Args:
            user_id: The user ID to check
            limit_type: Type of limit (query, ingest, graph, cache, etc.)
            value: Value to check (e.g., file size for storage)

        Returns:
            True if within limits, False if exceeded
        """
        # Skip limit checking for self-hosted mode
        if self.settings.MODE == "self_hosted":
            return True

        # Get user limits
        user_data = await self.db.get_user_limits(user_id)
        if not user_data:
            # Create user limits if they don't exist
            logger.info(f"User {user_id} not found when checking limits - creating limits record")
            success = await self.db.create_user_limits(user_id, tier=AccountTier.FREE)
            if not success:
                logger.error(f"Failed to create user limits for user {user_id}")
                return False
            # Fetch the newly created limits
            user_data = await self.db.get_user_limits(user_id)
            if not user_data:
                logger.error(f"Failed to retrieve newly created user limits for user {user_id}")
                return False

        # Get tier information
        tier = user_data.get("tier", AccountTier.FREE)

        # Only apply limits to free tier users
        # For non-free tiers, automatically return True (no limits)
        if tier != AccountTier.FREE:
            return True

        # For free tier, check against limits
        tier_limits = get_tier_limits(tier, user_data.get("custom_limits"))

        # Get current usage
        usage = user_data.get("usage", {})

        # Check specific limit type
        if limit_type == "query":
            hourly_limit = tier_limits.get("hourly_query_limit", 0)
            monthly_limit = tier_limits.get("monthly_query_limit", 0)

            hourly_usage = usage.get("hourly_query_count", 0)
            monthly_usage = usage.get("monthly_query_count", 0)

            return hourly_usage + value <= hourly_limit and monthly_usage + value <= monthly_limit

        elif limit_type == "ingest":
            hourly_limit = tier_limits.get("hourly_ingest_limit", 0)
            monthly_limit = tier_limits.get("monthly_ingest_limit", 0)

            hourly_usage = usage.get("hourly_ingest_count", 0)
            monthly_usage = usage.get("monthly_ingest_count", 0)

            return hourly_usage + value <= hourly_limit and monthly_usage + value <= monthly_limit

        elif limit_type == "storage_file":
            file_limit = tier_limits.get("storage_file_limit", 0)
            file_count = usage.get("storage_file_count", 0)

            return file_count + value <= file_limit

        elif limit_type == "storage_size":
            size_limit_bytes = tier_limits.get("storage_size_limit_gb", 0) * 1024 * 1024 * 1024
            size_usage = usage.get("storage_size_bytes", 0)

            return size_usage + value <= size_limit_bytes

        elif limit_type == "graph":
            graph_limit = tier_limits.get("graph_creation_limit", 0)
            graph_count = usage.get("graph_count", 0)

            return graph_count + value <= graph_limit

        elif limit_type == "cache":
            cache_limit = tier_limits.get("cache_creation_limit", 0)
            cache_count = usage.get("cache_count", 0)

            return cache_count + value <= cache_limit

        return True

    async def record_usage(self, user_id: str, usage_type: str, increment: int = 1, document_id: str = None) -> bool:
        """
        Record usage for a user. For non-free tier users in cloud mode, also sends metering data to Stripe.

        Args:
            user_id: The user ID
            usage_type: Type of usage (query, ingest, storage_file, storage_size, etc.)
            increment: Value to increment by
            document_id: Optional document ID for tracking in Stripe (used for ingest operations)

        Returns:
            True if successful, False otherwise
        """
        # Skip usage recording for self-hosted mode
        if self.settings.MODE == "self_hosted":
            return True

        # Check if user limits exist, create if they don't
        user_data = await self.db.get_user_limits(user_id)
        if not user_data:
            logger.info(f"Creating user limits for user {user_id} during usage recording")
            success = await self.db.create_user_limits(user_id, tier=AccountTier.FREE)
            if not success:
                logger.error(f"Failed to create user limits for user {user_id}")
                return False
            # Get the newly created user data
            user_data = await self.db.get_user_limits(user_id)
            if not user_data:
                logger.error(f"Failed to retrieve newly created user limits for user {user_id}")
                return False

        # Get user tier and Stripe customer ID
        tier = user_data.get("tier", AccountTier.FREE)
        stripe_customer_id = user_data.get("stripe_customer_id")

        # For non-free tier users in cloud mode, send metering data to Stripe for ingest operations
        if tier != AccountTier.FREE and self.settings.MODE == "cloud" and usage_type == "ingest" and stripe_customer_id:
            try:
                # Only import stripe if we're in cloud mode and need it
                import os

                import stripe
                from dotenv import load_dotenv

                load_dotenv(override=True)

                # Get Stripe API key from environment variable
                stripe_api_key = os.environ.get("STRIPE_API_KEY")
                if not stripe_api_key:
                    logger.warning("STRIPE_API_KEY not found in environment variables")
                else:
                    stripe.api_key = stripe_api_key

                    # For ingest operations, convert to pages if needed
                    # For PDFs and documents, increment is already in pages
                    # For other data types, convert using 1 page per 630 tokens
                    num_pages = increment

                    # Send metering event to Stripe
                    stripe.billing.MeterEvent.create(
                        event_name="pages-ingested",
                        payload={"value": str(num_pages), "stripe_customer_id": stripe_customer_id},
                        identifier=document_id or f"doc_{user_id}_{int(datetime.now(UTC).timestamp())}",
                    )
                    logger.info(f"Sent Stripe metering event for user {user_id}: {num_pages} pages ingested")
            except Exception as e:
                # Log error but continue with normal usage recording
                logger.error(f"Failed to send Stripe metering event: {e}")

        # Record usage in database
        return await self.db.update_usage(user_id, usage_type, increment)

    async def generate_cloud_uri(self, user_id: str, app_id: str, name: str, expiry_days: int = 30) -> Optional[str]:
        """
        Generate a cloud URI for an app.

        Args:
            user_id: The user ID
            app_id: The app ID
            name: App name for display purposes
            expiry_days: Number of days until token expires

        Returns:
            URI string with embedded token, or None if failed
        """
        # Get user limits to check app limit
        user_limits = await self.db.get_user_limits(user_id)

        # If user doesn't exist yet, create them
        if not user_limits:
            await self.create_user(user_id)
            user_limits = await self.db.get_user_limits(user_id)
            if not user_limits:
                logger.error(f"Failed to create user limits for user {user_id}")
                return None

        # Get tier information
        tier = user_limits.get("tier", AccountTier.FREE)
        current_apps = user_limits.get("app_ids", [])

        # Skip the limit check if app is already registered
        if app_id not in current_apps:
            # Only apply app limits to free tier users
            if tier == AccountTier.FREE:
                # Get tier limits to enforce app limit for free tier
                tier_limits = get_tier_limits(tier, user_limits.get("custom_limits"))
                app_limit = tier_limits.get("app_limit", 1)  # Default to 1 if not specified

                # Check if user has reached app limit
                if len(current_apps) >= app_limit:
                    logger.info(f"User {user_id} has reached app limit ({app_limit}) for free tier")
                    return None

        # Register the app
        success = await self.register_app(user_id, app_id)
        if not success:
            logger.info(f"Failed to register app {app_id} for user {user_id}")
            return None

        # Create token payload
        payload = {
            "user_id": user_id,
            "app_id": app_id,
            "name": name,
            "permissions": ["read", "write"],
            "exp": int((datetime.now(UTC) + timedelta(days=expiry_days)).timestamp()),
            "type": "developer",
            "entity_id": user_id,
        }

        # Generate token
        token = jwt.encode(payload, self.settings.JWT_SECRET_KEY, algorithm=self.settings.JWT_ALGORITHM)

        # Generate URI with API domain
        api_domain = getattr(self.settings, "API_DOMAIN", "api.morphik.ai")
        uri = f"morphik://{name}:{token}@{api_domain}"

        return uri

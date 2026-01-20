import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import Column, Index, String, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()


class UserLimitsModel(Base):
    """SQLAlchemy model for user limits data."""

    __tablename__ = "user_limits"

    org_id = Column(String, primary_key=True)  # Primary key is org_id in actual database
    user_id = Column(String, nullable=False)  # User who owns this org
    tier = Column(String, nullable=False)  # free, developer, startup, custom
    custom_limits = Column(JSONB, nullable=True)
    usage = Column(JSONB, default=dict)  # Holds all usage counters
    app_ids = Column(JSONB, default=list)  # List of app IDs registered by this user
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    stripe_product_id = Column(String, nullable=True)
    subscription_status = Column(String, nullable=True)
    created_at = Column(String)  # ISO format string
    updated_at = Column(String)  # ISO format string

    # Create indexes
    __table_args__ = (Index("idx_user_tier", "tier"),)


class UserLimitsDatabase:
    """Database operations for user limits."""

    def __init__(self, uri: str):
        """Initialize database connection."""
        self.engine = create_async_engine(uri)
        self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize database tables and indexes."""
        if self._initialized:
            return True

        try:
            logger.info("Initializing user limits database tables...")
            # Create tables if they don't exist
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

                # Check if we need to add the new Stripe columns
                # This safely adds columns if they don't exist without affecting existing data
                try:
                    # Check if the columns exist first to avoid errors
                    for column_name in [
                        "stripe_customer_id",
                        "stripe_subscription_id",
                        "stripe_product_id",
                        "subscription_status",
                    ]:
                        await conn.execute(
                            text(
                                f"DO $$\n"
                                f"BEGIN\n"
                                f"    IF NOT EXISTS (SELECT 1 FROM information_schema.columns "
                                f"                 WHERE table_name='user_limits' "
                                f"                 AND column_name='{column_name}') THEN\n"
                                f"        ALTER TABLE user_limits ADD COLUMN {column_name} VARCHAR;\n"
                                f"    END IF;\n"
                                f"END$$;"
                            )
                        )
                    logger.info("Successfully migrated user_limits table schema if needed")
                except Exception as migration_error:
                    logger.warning(f"Migration step failed, but continuing: {migration_error}")
                    # We continue even if migration fails as the app can still function

            self._initialized = True
            logger.info("User limits database tables initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize user limits database: {e}")
            return False

    async def get_user_limits(self, limit_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user limits for a limit_id (can be org_id or user_id for backward compat).

        Args:
            limit_id: The org_id or user_id to get limits for

        Returns:
            Dict with user limits if found, None otherwise
        """
        async with self.async_session() as session:
            # Try to find by org_id first (primary key)
            result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.org_id == limit_id))
            user_limits = result.scalars().first()

            # Fallback to user_id for backward compatibility
            if not user_limits:
                result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.user_id == limit_id))
                user_limits = result.scalars().first()

            if not user_limits:
                return None

            return {
                "org_id": user_limits.org_id,
                "user_id": user_limits.user_id,
                "tier": user_limits.tier,
                "custom_limits": user_limits.custom_limits,
                "usage": user_limits.usage,
                "app_ids": user_limits.app_ids,
                "stripe_customer_id": user_limits.stripe_customer_id,
                "stripe_subscription_id": user_limits.stripe_subscription_id,
                "stripe_product_id": user_limits.stripe_product_id,
                "subscription_status": user_limits.subscription_status,
                "created_at": user_limits.created_at,
                "updated_at": user_limits.updated_at,
            }

    async def create_user_limits(self, limit_id: str, tier: str = "free", user_id: Optional[str] = None) -> bool:
        """
        Create user limits record.

        Args:
            limit_id: The org_id (or user_id for backward compat)
            tier: Initial tier (defaults to "free")
            user_id: The actual user_id (optional, defaults to limit_id for backward compat)

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now(UTC).isoformat()

            # For backward compatibility, if user_id not provided, use limit_id
            if user_id is None:
                user_id = limit_id

            async with self.async_session() as session:
                # Check if already exists by org_id
                result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.org_id == limit_id))
                if result.scalars().first():
                    return True  # Already exists

                # Create new record with properly initialized JSONB columns
                # Create JSON strings and parse them for consistency
                usage_json = json.dumps(
                    {
                        "storage_file_count": 0,
                        "storage_size_bytes": 0,
                        "hourly_query_count": 0,
                        "hourly_query_reset": now,
                        "monthly_query_count": 0,
                        "monthly_query_reset": now,
                        "ingest_count": 0,
                    }
                )
                app_ids_json = json.dumps([])  # Empty array but as JSON string

                # Create the model with the JSON parsed
                user_limits = UserLimitsModel(
                    org_id=limit_id,  # org_id is the primary key
                    user_id=user_id,
                    tier=tier,
                    usage=json.loads(usage_json),
                    app_ids=json.loads(app_ids_json),
                    stripe_customer_id=None,
                    stripe_subscription_id=None,
                    stripe_product_id=None,
                    subscription_status=None,
                    created_at=now,
                    updated_at=now,
                )

                session.add(user_limits)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to create user limits: {e}")
            return False

    async def update_user_tier(self, limit_id: str, tier: str, custom_limits: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update user tier and custom limits.

        Args:
            limit_id: The org_id or user_id
            tier: New tier
            custom_limits: Optional custom limits for CUSTOM tier

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now(UTC).isoformat()

            async with self.async_session() as session:
                # Try org_id first, then user_id
                result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.org_id == limit_id))
                user_limits = result.scalars().first()

                if not user_limits:
                    result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.user_id == limit_id))
                    user_limits = result.scalars().first()

                if not user_limits:
                    return False

                user_limits.tier = tier
                user_limits.custom_limits = custom_limits
                user_limits.updated_at = now

                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update user tier: {e}")
            return False

    async def update_subscription_info(self, limit_id: str, subscription_data: Dict[str, Any]) -> bool:
        """
        Update user subscription information.

        Args:
            limit_id: The org_id or user_id
            subscription_data: Dictionary containing subscription information with keys:
                - stripeCustomerId
                - stripeSubscriptionId
                - stripeProductId
                - subscriptionStatus

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now(UTC).isoformat()

            async with self.async_session() as session:
                # Try org_id first, then user_id
                result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.org_id == limit_id))
                user_limits = result.scalars().first()

                if not user_limits:
                    result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.user_id == limit_id))
                    user_limits = result.scalars().first()

                if not user_limits:
                    return False

                user_limits.stripe_customer_id = subscription_data.get("stripeCustomerId")
                user_limits.stripe_subscription_id = subscription_data.get("stripeSubscriptionId")
                user_limits.stripe_product_id = subscription_data.get("stripeProductId")
                user_limits.subscription_status = subscription_data.get("subscriptionStatus")
                user_limits.updated_at = now

                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update subscription info: {e}")
            return False

    async def register_app(self, limit_id: str, app_id: str) -> bool:
        """
        Register an app for a limit_id.

        Args:
            limit_id: The org_id or user_id
            app_id: The app ID to register

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now(UTC).isoformat()

            async with self.async_session() as session:
                # Try org_id first, then user_id
                result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.org_id == limit_id))
                user_limits = result.scalars().first()

                if not user_limits:
                    result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.user_id == limit_id))
                    user_limits = result.scalars().first()

                if not user_limits:
                    logger.error(f"Limit ID {limit_id} not found in register_app")
                    return False

                # Get the actual primary key value (org_id)
                primary_key = user_limits.org_id

                # Use raw SQL with jsonb_array_append to update the app_ids array
                # This is the most reliable way to append to a JSONB array in PostgreSQL
                query = text(
                    """
                    UPDATE user_limits
                    SET
                        app_ids = CASE
                            WHEN NOT (app_ids ? :app_id)  -- Check if app_id is not in the array
                            THEN app_ids || :app_id_json  -- Append it if not present
                            ELSE app_ids                  -- Keep it unchanged if already present
                        END,
                        updated_at = :now
                    WHERE org_id = :org_id
                    RETURNING app_ids;
                    """
                )

                # Execute the query
                result = await session.execute(
                    query,
                    {
                        "app_id": app_id,  # For the check
                        "app_id_json": f'["{app_id}"]',  # JSON array format for appending
                        "now": now,
                        "org_id": primary_key,
                    },
                )

                # Log the result for debugging
                updated_app_ids = result.scalar()
                logger.info(f"Updated app_ids for limit_id {limit_id} (org_id: {primary_key}): {updated_app_ids}")

                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to register app: {e}")
            return False

    async def unregister_app(self, limit_id: str, app_id: str) -> bool:
        """Remove *app_id* from the limit_id's *app_ids* list.

        Returns ``True`` on success (or if the app was already absent).
        """
        try:
            now = datetime.now(UTC).isoformat()

            async with self.async_session() as session:
                # First find the record to get the org_id
                result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.org_id == limit_id))
                user_limits = result.scalars().first()

                if not user_limits:
                    result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.user_id == limit_id))
                    user_limits = result.scalars().first()

                if not user_limits:
                    logger.warning(f"Limit ID {limit_id} not found in unregister_app")
                    return True  # Consider it success if doesn't exist

                primary_key = user_limits.org_id
                # Use the jsonb "-" operator which natively removes the first
                # occurrence of a matching string element from a JSONB array.
                # This avoids complex casts and the "polymorphic type unknown"
                # errors we observed with to_jsonb().
                query = text(
                    """
                    UPDATE user_limits
                    SET app_ids = app_ids - :app_id,
                        updated_at = :now
                    WHERE org_id = :org_id
                    RETURNING app_ids;
                    """
                )

                result = await session.execute(query, {"app_id": app_id, "now": now, "org_id": primary_key})

                updated_app_ids = result.scalar()
                logger.info(
                    "Unregistered app_id %s for limit_id %s (org_id: %s). Remaining apps: %s",
                    app_id,
                    limit_id,
                    primary_key,
                    updated_app_ids,
                )

                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to unregister app: {e}")
            return False

    async def update_usage(self, limit_id: str, usage_type: str, increment: int = 1) -> bool:
        """
        Update usage counter for a limit_id.

        Args:
            limit_id: The org_id or user_id
            usage_type: Type of usage to update
            increment: Value to increment by

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now(UTC)
            now_iso = now.isoformat()

            async with self.async_session() as session:
                # Try org_id first, then user_id
                result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.org_id == limit_id))
                user_limits = result.scalars().first()

                if not user_limits:
                    result = await session.execute(select(UserLimitsModel).where(UserLimitsModel.user_id == limit_id))
                    user_limits = result.scalars().first()

                if not user_limits:
                    return False

                # Create a new dictionary to force SQLAlchemy to detect the change
                usage = dict(user_limits.usage) if user_limits.usage else {}

                # Handle different usage types
                if usage_type == "query":
                    # Check hourly reset
                    hourly_reset_str = usage.get("hourly_query_reset", "")
                    if hourly_reset_str:
                        hourly_reset = datetime.fromisoformat(hourly_reset_str)
                        if now > hourly_reset + timedelta(hours=1):
                            usage["hourly_query_count"] = increment
                            usage["hourly_query_reset"] = now_iso
                        else:
                            usage["hourly_query_count"] = usage.get("hourly_query_count", 0) + increment
                    else:
                        usage["hourly_query_count"] = increment
                        usage["hourly_query_reset"] = now_iso

                    # Check monthly reset
                    monthly_reset_str = usage.get("monthly_query_reset", "")
                    if monthly_reset_str:
                        monthly_reset = datetime.fromisoformat(monthly_reset_str)
                        if now > monthly_reset + timedelta(days=30):
                            usage["monthly_query_count"] = increment
                            usage["monthly_query_reset"] = now_iso
                        else:
                            usage["monthly_query_count"] = usage.get("monthly_query_count", 0) + increment
                    else:
                        usage["monthly_query_count"] = increment
                        usage["monthly_query_reset"] = now_iso

                elif usage_type == "ingest":
                    # Lifetime counter (no reset)
                    usage["ingest_count"] = usage.get("ingest_count", 0) + increment

                elif usage_type == "storage_file":
                    usage["storage_file_count"] = usage.get("storage_file_count", 0) + increment

                elif usage_type == "storage_size":
                    usage["storage_size_bytes"] = usage.get("storage_size_bytes", 0) + increment

                # Force SQLAlchemy to recognize the change by assigning a new dict
                user_limits.usage = usage
                user_limits.updated_at = now_iso

                # Explicitly mark as modified
                session.add(user_limits)

                # Log the updated usage for debugging
                logger.info(f"Updated usage for limit_id {limit_id}, type: {usage_type}, value: {increment}")
                logger.info(f"New usage values: {usage}")
                logger.info(f"About to commit: limit_id={limit_id}, usage={user_limits.usage}")

                # Commit and flush to ensure changes are written
                await session.commit()

                return True

        except Exception as e:
            logger.error(f"Failed to update usage: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

import logging
import uuid as _uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

import jwt
from sqlalchemy import func, or_, select, text

from utils.env_loader import load_local_env

from ..config import get_settings
from ..database.metadata_filters import InvalidMetadataFilterError, TextColumnFilterBuilder
from ..database.user_limits_db import UserLimitsDatabase
from ..models.tiers import AccountTier, get_tier_limits

logger = logging.getLogger(__name__)


class UserService:
    """Service for managing user limits and usage."""

    def __init__(self):
        """Initialize the UserService."""
        self.settings = get_settings()
        self.db = UserLimitsDatabase(uri=self.settings.POSTGRES_URI)
        self._db_initialized = False  # Flag to track if self.db has been initialized

    async def initialize(self) -> bool:
        """Initialize database tables for the user limits system if not already done."""
        if self._db_initialized:
            return True  # Already initialized for this UserService instance

        # Attempt to initialize the database component (UserLimitsDatabase)
        # UserLimitsDatabase.initialize() has its own internal _initialized flag
        # to prevent its DDL from running multiple times for the same instance.
        if await self.db.initialize():
            self._db_initialized = True  # Mark as initialized for this UserService instance
            logger.info("UserService: Database component initialized successfully.")
            return True

        logger.error("UserService: Failed to initialize database component.")
        return False

    async def get_user_limits(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user limits information."""
        return await self.db.get_user_limits(user_id)

    async def create_user(self, user_id: str, *, tier: AccountTier = AccountTier.FREE) -> bool:
        """Create a new user with the specified tier (defaults to FREE)."""
        return await self.db.create_user_limits(user_id, tier=tier)

    async def update_user_tier(self, user_id: str, tier: str, custom_limits: Optional[Dict[str, Any]] = None) -> bool:
        """Update user tier and custom limits."""
        return await self.db.update_user_tier(user_id, tier, custom_limits)

    async def check_limit(self, user_id: str, limit_type: str, value: int = 1) -> bool:
        """
        Check if a user's operation is within limits when given value is considered.
        Limits are only applied to the free tier; other tiers have no limits.

        Args:
            user_id: The user ID to check
            limit_type: Type of limit (query, ingest, storage, etc.)
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
            total_limit = tier_limits.get("ingest_limit", 0)
            total_usage = usage.get("ingest_count", 0)

            return total_usage + value <= total_limit

        elif limit_type == "storage_file":
            file_limit = tier_limits.get("storage_file_limit", 0)
            file_count = usage.get("storage_file_count", 0)

            return file_count + value <= file_limit

        elif limit_type == "storage_size":
            size_limit_bytes = tier_limits.get("storage_size_limit_gb", 0) * 1024 * 1024 * 1024
            size_usage = usage.get("storage_size_bytes", 0)

            return size_usage + value <= size_limit_bytes

        return True

    async def record_usage(
        self, user_id: str, usage_type: str, increment: int = 1, document_id: Optional[str] = None
    ) -> bool:
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

                load_local_env(override=True)

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

    async def generate_cloud_uri(
        self,
        user_id: str,
        app_id: str,
        name: str,
        expiry_days: int = 5475,  # 15 years
        *,
        org_id: Optional[str] = None,
        created_by_user_id: Optional[str] = None,
        is_admin_call: bool = False,
    ) -> Optional[str]:
        """
        Generate a cloud URI for an app.

        Args:
            user_id: The user ID
            app_id: The app ID
            name: App name for display purposes
            expiry_days: Number of days until token expires
            org_id: Optional organization identifier
            created_by_user_id: Service/admin user that initiated the request
            is_admin_call: When True, bypass user-tier limits and auto-upgrade to SELF_HOSTED tier

        Returns:
            URI string with embedded token, or None if failed
        """
        target_tier = AccountTier.SELF_HOSTED if is_admin_call else AccountTier.FREE

        # Get user limits to check app limit
        user_limits = await self.db.get_user_limits(user_id)

        # If user doesn't exist yet, create them with the appropriate tier
        if not user_limits:
            await self.create_user(user_id, tier=target_tier)
            user_limits = await self.db.get_user_limits(user_id)
            if not user_limits:
                logger.error("Failed to create user limits for user %s", user_id)
                return None
        elif is_admin_call and user_limits.get("tier") != AccountTier.SELF_HOSTED:
            updated = await self.update_user_tier(user_id, AccountTier.SELF_HOSTED.value)
            if updated:
                user_limits = await self.db.get_user_limits(user_id)
            else:
                logger.warning("Unable to promote user %s to SELF_HOSTED tier for admin provisioning", user_id)

        user_uuid = self._safe_uuid(user_id)
        existing_app = await self.get_app_by_id(app_id)
        existing_in_scope = False
        if existing_app:
            if org_id and existing_app.get("org_id") == org_id:
                existing_in_scope = True
            elif not org_id:
                existing_user_id = existing_app.get("user_id")
                if user_uuid and existing_user_id == str(user_uuid):
                    existing_in_scope = True
                elif user_id and existing_app.get("created_by_user_id") == user_id:
                    existing_in_scope = True

        # Get tier information
        if not is_admin_call:
            tier = user_limits.get("tier", AccountTier.FREE)

            if tier == AccountTier.FREE and not existing_in_scope:
                tier_limits = get_tier_limits(tier, user_limits.get("custom_limits"))
                app_limit = tier_limits.get("app_limit", 1)
                app_count = await self._count_apps_in_scope(org_id=org_id, user_uuid=user_uuid, user_id=user_id)

                if app_count >= app_limit:
                    logger.info("User %s has reached app limit (%s) for tier %s", user_id, app_limit, tier)
                    return None

        # Enforce name uniqueness within the same owner/org scope
        if await self._app_name_exists(name=name, user_uuid=user_uuid, org_id=org_id):
            logger.warning("App with name '%s' already exists for scope user=%s org=%s", name, user_id, org_id)
            raise ValueError(f"App with name '{name}' already exists")

        token_version = 0
        if existing_app and "token_version" in existing_app:
            token_version = existing_app.get("token_version") or 0

        # Create token payload (keep entity_id for backward compatibility)
        payload = {
            "user_id": user_id,
            "entity_id": user_id,  # backward compat
            "app_id": app_id,
            "name": name,
            "token_version": token_version,
            "exp": int((datetime.now(UTC) + timedelta(days=expiry_days)).timestamp()),
        }

        token = jwt.encode(payload, self.settings.JWT_SECRET_KEY, algorithm=self.settings.JWT_ALGORITHM)

        # Generate URI with API domain
        api_domain = getattr(self.settings, "API_DOMAIN", "api.morphik.ai")
        uri = f"morphik://{name}:{token}@{api_domain}"

        await self._upsert_app_record(
            app_id=app_id,
            user_uuid=user_uuid,
            org_id=org_id,
            created_by_user_id=created_by_user_id,
            name=name,
            uri=uri,
        )

        return uri

    async def _count_apps_in_scope(
        self,
        *,
        org_id: Optional[str],
        user_uuid: Optional[_uuid.UUID],
        user_id: Optional[str],
    ) -> int:
        """Count apps for the org or legacy user scope."""
        from core.models.apps import AppModel  # Local import to avoid cycles

        async with self.db.async_session() as session:
            stmt = select(func.count(AppModel.app_id))
            if org_id:
                stmt = stmt.where(AppModel.org_id == org_id)
            elif user_uuid:
                stmt = stmt.where(AppModel.user_id == user_uuid)
            elif user_id:
                stmt = stmt.where(AppModel.created_by_user_id == user_id)

            result = await session.execute(stmt)
            return int(result.scalar() or 0)

    async def _app_name_exists(
        self,
        *,
        name: str,
        user_uuid: Optional[_uuid.UUID],
        org_id: Optional[str],
        exclude_app_id: Optional[str] = None,
    ) -> bool:
        """Check whether an app name already exists within the same owner/org scope."""
        from core.models.apps import AppModel  # Local import to avoid cycles

        async with self.db.async_session() as session:
            stmt = select(AppModel).where(AppModel.name == name)
            if org_id:
                stmt = stmt.where(AppModel.org_id == org_id)
            elif user_uuid:
                stmt = stmt.where(AppModel.user_id == user_uuid)

            result = await session.execute(stmt)
            existing = result.scalars().all()
            if not existing:
                return False
            if exclude_app_id:
                return any(app.app_id != exclude_app_id for app in existing)
            return True

    async def _upsert_app_record(
        self,
        *,
        app_id: str,
        user_uuid: Optional[_uuid.UUID],
        org_id: Optional[str],
        created_by_user_id: Optional[str],
        name: str,
        uri: str,
    ) -> None:
        """Create or update the lightweight dashboard app record."""
        from core.models.apps import AppModel  # Local import to avoid cycles

        async with self.db.async_session() as session:
            app_record = await session.get(AppModel, app_id)
            if app_record is None:
                app_record = AppModel(
                    app_id=app_id,
                    user_id=user_uuid,
                    org_id=org_id,
                    created_by_user_id=created_by_user_id,
                    name=name,
                    uri=uri,
                )
                session.add(app_record)
            else:
                if user_uuid:
                    app_record.user_id = user_uuid
                if org_id:
                    app_record.org_id = org_id
                if created_by_user_id:
                    app_record.created_by_user_id = created_by_user_id
                app_record.name = name
                app_record.uri = uri
                if getattr(app_record, "token_version", None) is None:
                    app_record.token_version = 0
            await session.commit()

    async def get_app_by_id(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get an app record by its app_id."""
        from core.models.apps import AppModel  # Local import to avoid cycles

        async with self.db.async_session() as session:
            app_record = await session.get(AppModel, app_id)
            if app_record is None:
                return None
            return {
                "app_id": app_record.app_id,
                "org_id": app_record.org_id,
                "user_id": str(app_record.user_id) if app_record.user_id else None,
                "created_by_user_id": app_record.created_by_user_id,
                "name": app_record.name,
                "uri": app_record.uri,
                "token_version": getattr(app_record, "token_version", 0) or 0,
            }

    async def list_apps(
        self,
        *,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        app_id_filter: Optional[Any] = None,
        name_filter: Optional[Any] = None,
        limit: int = 100,
        offset: int = 0,
        strict_org_scope: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return dashboard app rows filtered by org or user."""

        from core.models.apps import AppModel  # Local import to avoid cycles

        normalized_limit = max(1, min(limit, 500))
        normalized_offset = max(0, offset)

        async with self.db.async_session() as session:
            stmt = select(AppModel).order_by(AppModel.created_at.desc())

            # Org-scoped: show all apps in the org (not filtered by user_id)
            # This allows all org members to see all org apps
            if org_id:
                if strict_org_scope:
                    stmt = stmt.where(AppModel.org_id == org_id)
                else:
                    stmt = stmt.where(or_(AppModel.org_id == org_id, AppModel.org_id.is_(None)))
            elif user_id:
                # Only filter by user_id when no org_id is provided (personal apps)
                user_uuid = self._safe_uuid(user_id)
                if user_uuid:
                    stmt = stmt.where(AppModel.user_id == user_uuid)
                else:
                    stmt = stmt.where(AppModel.created_by_user_id == user_id)

            if app_id_filter:
                app_id_clause = self._build_text_filter_clause(
                    app_id_filter,
                    field_name="app_id",
                    column="apps.app_id",
                    label="App ID",
                )
                stmt = stmt.where(text(app_id_clause))

            if name_filter:
                name_clause = self._build_text_filter_clause(
                    name_filter,
                    field_name="name",
                    column="apps.name",
                    label="App name",
                )
                stmt = stmt.where(text(name_clause))

            stmt = stmt.offset(normalized_offset).limit(normalized_limit)
            result = await session.execute(stmt)
            apps = result.scalars().all()

        return [
            {
                "app_id": app.app_id,
                "org_id": app.org_id,
                "user_id": str(app.user_id) if app.user_id else None,
                "created_by_user_id": app.created_by_user_id,
                "name": app.name,
                "uri": app.uri,
                "created_at": app.created_at.isoformat() if app.created_at else None,
            }
            for app in apps
        ]

    def _build_text_filter_clause(self, filter_expression: Any, *, field_name: str, column: str, label: str) -> str:
        normalized = self._normalize_text_filter_expression(filter_expression, field_name, label)
        builder = TextColumnFilterBuilder(column)
        try:
            return builder.build(normalized)
        except InvalidMetadataFilterError as exc:
            message = str(exc).replace("Filename", label).replace("filename", label.lower())
            raise InvalidMetadataFilterError(message) from exc

    @staticmethod
    def _normalize_text_filter_expression(expression: Any, field_name: str, label: str) -> Dict[str, Any]:
        if isinstance(expression, list):
            if not expression:
                raise InvalidMetadataFilterError(f"{label} filter list cannot be empty.")
            return {
                "$or": [UserService._normalize_text_filter_expression(item, field_name, label) for item in expression]
            }

        if not isinstance(expression, dict):
            raise InvalidMetadataFilterError(f"{label} filter must be provided as a JSON object.")
        if not expression:
            raise InvalidMetadataFilterError(f"{label} filter cannot be empty.")

        has_operator = any(key.startswith("$") for key in expression)
        if has_operator:
            normalized: Dict[str, Any] = {}
            for key, value in expression.items():
                if not key.startswith("$"):
                    raise InvalidMetadataFilterError(f"{label} filters only support the '{field_name}' field.")
                if key in {"$and", "$or", "$nor"}:
                    if not isinstance(value, list) or not value:
                        raise InvalidMetadataFilterError(f"{key} operator expects a non-empty list of conditions.")
                    normalized[key] = [
                        UserService._normalize_text_filter_expression(item, field_name, label) for item in value
                    ]
                elif key == "$not":
                    normalized[key] = UserService._normalize_text_filter_expression(value, field_name, label)
                else:
                    normalized[key] = value
            return normalized

        if field_name not in expression or len(expression) != 1:
            raise InvalidMetadataFilterError(f"{label} filters only support the '{field_name}' field.")
        return UserService._normalize_text_filter_value(expression[field_name], field_name, label)

    @staticmethod
    def _normalize_text_filter_value(value: Any, field_name: str, label: str) -> Dict[str, Any]:
        if isinstance(value, dict):
            if not value:
                raise InvalidMetadataFilterError(f"{label} filter cannot be empty.")
            if any(key.startswith("$") for key in value):
                return UserService._normalize_text_filter_expression(value, field_name, label)
            raise InvalidMetadataFilterError(f"{label} filter objects must use operator keys.")

        if isinstance(value, list):
            if not value:
                raise InvalidMetadataFilterError(f"{label} filter list must contain at least one value.")
            return {"$in": value}

        return {"$eq": value}

    @staticmethod
    def _safe_uuid(value: Optional[str]) -> Optional[_uuid.UUID]:
        """Convert string to UUID if possible; otherwise return None."""
        if not value:
            return None
        try:
            return _uuid.UUID(str(value))
        except (ValueError, TypeError):
            logger.debug("Value %s is not a valid UUID – storing NULL in apps.user_id", value)
            return None

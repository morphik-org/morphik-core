from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Dict
from urllib.parse import urlparse, urlunparse

import boto3  # local import to avoid mandatory dependency when S3 unused
import botocore
import jwt
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from core.config import get_settings
from core.models.app_metadata import AppMetadataModel
from core.models.apps import AppModel
from core.services.neon_client import NeonClient

# =============================================================================
#  AppProvisioningService – orchestrates programmatic provisioning of Neon
#  projects for Morphik "apps" (one project per app) and stores metadata in the
#  control-plane Postgres database so that the instance can keep track of what
#  it created.  The implementation has intentionally *no* dependencies on the
#  public FastAPI router layer – it can be invoked directly from unit tests
#  and background jobs.
# =============================================================================


logger = logging.getLogger(__name__)


load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Public dataclass returned by the service
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProvisionResult:  # noqa: D101 – simple dataclass
    app_id: str
    app_name: str
    morphik_uri: str
    neon_project_id: str
    status: str = "ready"

    def as_dict(self) -> Dict[str, str]:  # Convenience helper for JSON response
        return asdict(self)


# ---------------------------------------------------------------------------
# Helper: slugify app names (very naive – only lower-cases and replaces spaces)
# ---------------------------------------------------------------------------

DEFAULT_REGION = "us-east-2"


def _slugify(value: str, max_length: int = 40) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_\-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value[:max_length]


def bucket_exists(s3_client, bucket_name):
    """Check if an S3 bucket exists."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code in [404, 403]:
            return False
        # raise e


def create_s3_bucket(bucket_name, region=DEFAULT_REGION):
    """Set up S3 bucket."""
    # Clear any existing AWS credentials from environment
    boto3.Session().resource("s3").meta.client.close()

    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION") or region

    if not aws_access_key or not aws_secret_key:
        logger.error("AWS credentials not found in environment variables.")
        return

    logger.debug("Successfully retrieved AWS credentials and region.")
    # Create new session with explicit credentials
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region,
    )

    s3_client = session.client("s3")
    logger.debug("Successfully created S3 client.")

    if bucket_exists(s3_client, bucket_name):
        logger.info(f"Bucket with name {bucket_name} already exists")
        return

    if region == "us-east-1":
        s3_client.create_bucket(Bucket=bucket_name)
    else:
        s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region})

    logger.debug(f"Bucket {bucket_name} created successfully in {region} region.")


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class AppProvisioningService:  # noqa: D101 – obvious from name
    def __init__(self):
        settings = get_settings()

        if not settings.POSTGRES_URI:
            raise RuntimeError("POSTGRES_URI must be set in the environment/config to use AppProvisioningService")
        if not os.getenv("NEON_API_KEY"):
            # We *warn* instead of error-ing because the caller may want to mock NeonClient.
            logger.warning("NEON_API_KEY environment variable not set – real Neon provisioning will fail")

        self._neon_client = NeonClient(api_key=os.getenv("NEON_API_KEY", ""))

        # SQLAlchemy async engine for the control-plane DB (same as used elsewhere)
        self._engine = create_async_engine(
            settings.POSTGRES_URI, pool_size=settings.DB_POOL_SIZE, max_overflow=settings.DB_MAX_OVERFLOW
        )
        self._async_session: sessionmaker[AsyncSession] = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        """Create the *app_metadata* table if it does not exist yet."""
        if self._initialized:
            return True
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(AppMetadataModel.metadata.create_all)  # type: ignore[arg-type]
            self._initialized = True
            logger.info("AppProvisioningService database schema initialised")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialise AppProvisioningService schema: %s", exc)
            return False

    # The main entry point --------------------------------------------------

    async def provision_new_app(
        self, user_id: str, app_name: str, region: str | None = None, morphik_host: str | None = None
    ) -> ProvisionResult:  # noqa: D401
        """Provision a *new* Neon project (one per app) and return the connection URI.

        This method **does not** apply any schema to the database – it solely
        provisions the project/branch and stores a record in *app_metadata* so
        that the control plane knows about it.  Callers that require an initial
        schema must run Alembic with `-x db_url=<uri>` *after* obtaining the
        result.
        """
        if not region:
            region = "aws-us-east-1"

        # ------------------------------------------------------------------
        # 1) Provision project + main branch via the Neon API
        # ------------------------------------------------------------------
        # NOTE: This code intentionally surfaces *NeonAPIError* so that callers
        #       can handle failures – we do NOT swallow errors silently.

        safe_app_name = _slugify(app_name)
        unique_project_name = f"{safe_app_name}-{uuid.uuid4().hex[:8]}"

        project_resp = await self._neon_client.create_project(project_name=unique_project_name, region=region)
        project_id = project_resp["project"]["id"]  # Neon returns nested JSON structure

        # Wait until project is fully provisioned to avoid 423 errors
        await self._neon_client.wait_until_project_ready(project_id)

        # Neon already creates a *main* branch and a default compute endpoint.
        # In most cases the `create_project` response already includes a
        # `connection_uris` array which we can use directly.  If not, fall back
        # to the explicit *connection_uri* helper.

        connection_uri_from_neon: str | None = None

        connection_uris_list = project_resp.get("connection_uris") or []
        if connection_uris_list:
            first_item = connection_uris_list[0]
            if isinstance(first_item, str):
                connection_uri_from_neon = first_item
            elif isinstance(first_item, dict):
                connection_uri_from_neon = first_item.get("connection_uri")

        if not connection_uri_from_neon:
            connection_uri_from_neon = await self._neon_client.get_connection_uri(
                project_id=project_id,
                database_name="neondb",  # Default DB name in Neon
                role_name="neon",  # Default role name in Neon
            )

        assert isinstance(connection_uri_from_neon, str), "connection_uri_from_neon should be a string at this point"

        # 1a) Clean the Neon connection URI (remove query parameters like sslmode)
        parsed_neon_uri = urlparse(connection_uri_from_neon)
        # Remove query parameters
        parsed_no_query = parsed_neon_uri._replace(query="")
        # Ensure asyncpg driver in the scheme
        scheme = parsed_no_query.scheme
        if scheme in ("postgres", "postgresql"):
            new_scheme = "postgresql+asyncpg"
        elif not scheme.endswith("+asyncpg"):
            new_scheme = scheme + "+asyncpg"
        else:
            new_scheme = scheme
        parsed_with_driver = parsed_no_query._replace(scheme=new_scheme)
        stored_connection_uri = urlunparse(parsed_with_driver)

        # 1b) Generate a real JWT that will be embedded in the Morphik URI. This token is
        #     validated by :func:`core.auth_utils.verify_token` on every request from the
        #     provisioned application.  We embed *app_id* so that downstream logic can look
        #     up the dedicated connection URI and S3 bucket for that specific app.

        settings = get_settings()

        # App ID needs to be generated *before* we create the JWT so the same value is used
        # everywhere.  Move it here.
        app_id = str(uuid.uuid4())

        # Use a long-lived expiry (1 year) – callers can always rotate tokens by creating a
        # new one if required.
        expiry = datetime.now(UTC) + timedelta(days=365)

        payload = {
            "type": "developer",  # Matches the check in verify_token()
            "entity_id": user_id,  # The developer / account that owns the app
            "app_id": app_id,  # Newly generated application identifier
            "permissions": ["read", "write"],
            "exp": int(expiry.timestamp()),
        }

        morphik_access_token: str = jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

        # (Optional) we *could* create an additional read-write endpoint by
        # invoking create_branch(), but that is usually unnecessary for the
        # simple app-per-project model.

        # 2a) Create dedicated S3 bucket for raw files --------------------
        # Naming scheme: morphik-<project_id>.  Adjust as needed.
        bucket_name = f"morphik-{project_id}"
        logger.info(f"Creating S3 bucket {bucket_name}")
        create_s3_bucket(bucket_name)
        logger.info(f"S3 bucket {bucket_name} created successfully")

        # ------------------------------------------------------------------
        # 2) Build the Morphik-specific URI
        # ------------------------------------------------------------------
        if not morphik_host:
            # This should be caught by the EE router, but as a safeguard:
            logger.error("morphik_host not provided to AppProvisioningService for new URI format.")
            raise ValueError("morphik_host is required for the new Morphik URI format.")

        # Use safe_app_name (slug) for the username part of the URI
        morphik_uri = f"morphik://{safe_app_name}:{morphik_access_token}@{morphik_host}"

        # ------------------------------------------------------------------
        # 3) Persist *app_metadata* record so that the instance can track it
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # NOTE: *app_id* has already been created above.  We simply reuse it here so the
        #       database row matches the token payload exactly.
        # ------------------------------------------------------------------

        now = datetime.now(UTC)
        async with self._async_session() as session:
            model = AppMetadataModel(
                id=app_id,
                user_id=user_id,
                app_name=app_name,  # Store original app_name
                neon_project_id=project_id,
                connection_uri=stored_connection_uri,  # Store the cleaned Neon URI
                extra={"s3_bucket": bucket_name, "morphik_token": morphik_access_token},
                created_at=now,
                updated_at=now,
            )
            session.add(model)

            # Also create lightweight *apps* record for dashboard listing
            dashboard_app = AppModel(
                app_id=app_id,
                user_id=uuid.UUID(user_id),  # Cast to standard Python UUID object
                name=app_name,
                uri=morphik_uri,
            )
            session.add(dashboard_app)
            await session.commit()

        # ------------------------------------------------------------------
        # 4) Done – return dataclass (easier for unit tests) --------------
        # ------------------------------------------------------------------
        return ProvisionResult(
            app_id=app_id,
            app_name=app_name,
            morphik_uri=morphik_uri,  # Return the new Morphik URI format
            neon_project_id=project_id,
        )

    # ------------------------------------------------------------------
    # Graceful shutdown helper (e.g. when called from background worker)
    # ------------------------------------------------------------------

    async def close(self) -> None:  # pragma: no cover
        try:
            await self._neon_client.close()
        finally:
            await self._engine.dispose()

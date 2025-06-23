import logging
from typing import List, Literal

from fastapi import APIRouter, Depends, HTTPException
from httpx import AsyncClient, Timeout
from pydantic import BaseModel

from core.auth_utils import verify_token
from core.config import get_settings
from core.models.auth import AuthContext
from core.services.telemetry import TelemetryService
from core.services_init import document_service

# ---------------------------------------------------------------------------
# Router initialization
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Embeddings"])
logger = logging.getLogger(__name__)
settings = get_settings()
telemetry = TelemetryService()

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class ColPaliEmbeddingRequest(BaseModel):
    input_type: Literal["text", "image"]
    inputs: List[str]


class ColPaliEmbeddingResponse(BaseModel):
    embeddings: List[List[List[float]]]
    usage: dict = {"embeddings_count": 0}


# ---------------------------------------------------------------------------
# /embeddings/colpali endpoint
# ---------------------------------------------------------------------------


@router.post("/embeddings", response_model=ColPaliEmbeddingResponse)
@telemetry.track(operation_type="colpali_embeddings", metadata_resolver=telemetry.base_metadata)
async def create_colpali_embeddings(
    request: ColPaliEmbeddingRequest,
    auth: AuthContext = Depends(verify_token),
) -> ColPaliEmbeddingResponse:
    """Generate ColPali embeddings for text or images.

    This endpoint requires the user to have a valid payment method on file.
    Usage is tracked and billed per embedding through Stripe metering.

    Args:
        request: Embedding request containing input_type and inputs
        auth: Authentication context

    Returns:
        ColPaliEmbeddingResponse with embeddings and usage info
    """
    # Check if user has active ColPali subscription
    user_data = await document_service.db.get_user_limits(auth.user_id)
    if not user_data:
        raise HTTPException(status_code=403, detail="User account not found. Please contact support.")

    stripe_customer_id = user_data.get("stripe_customer_id")
    stripe_product_id = user_data.get("stripe_product_id")
    subscription_status = user_data.get("subscription_status")
    tier = user_data.get("tier", "free")
    custom_limits = user_data.get("custom_limits", {})

    # Check if user has ColPali subscription in custom_limits
    colpali_sub = custom_limits.get("colpali_subscription", {}) if custom_limits else {}
    has_colpali_subscription = colpali_sub.get("status") in ["active", "trialing"]

    # Also check if they have ColPali as their main subscription (legacy)
    has_colpali_main_subscription = (
        stripe_customer_id
        and stripe_product_id == "prod_SMQlpO3RBXOyEs"  # ColPali product ID
        and subscription_status in ["active", "trialing"]
    )

    # Allow access if user has ColPali subscription OR any paid tier (pro, teams, etc)
    has_api_access = (
        has_colpali_subscription
        or has_colpali_main_subscription
        or (tier in ["pro", "teams", "enterprise", "self_hosted"] and subscription_status in ["active", "trialing"])
    )

    if settings.MODE == "cloud" and not has_api_access:
        raise HTTPException(
            status_code=402,
            detail="ColPali API subscription required. Please subscribe from your billing page to use the ColPali API.",
        )

    # Calculate number of embeddings requested
    num_embeddings = len(request.inputs)

    # Make request to ColPali API
    try:
        # Get ColPali API configuration
        colpali_api_key = settings.COLPALI_API_KEY  # Internal API key for colqwen service
        if not colpali_api_key:
            logger.error("COLPALI_API_KEY not configured")
            raise HTTPException(
                status_code=500, detail="ColPali service not properly configured. Please contact support."
            )

        # Point to the actual colqwen hosted service
        endpoint = "https://embedding-api.morphik.ai/embeddings"

        headers = {"Authorization": f"Bearer {colpali_api_key}"}
        payload = {"input_type": request.input_type, "inputs": request.inputs}

        timeout = Timeout(read=60.0, connect=10.0, write=30.0, pool=10.0)
        async with AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("embeddings", [])

        # Track usage in database
        await document_service.user_service.record_usage(
            user_id=auth.user_id, usage_type="colpali_embeddings", increment=num_embeddings, document_id=None
        )

        # Send metering event to Stripe
        if settings.MODE == "cloud" and stripe_customer_id:
            try:
                import stripe

                stripe.api_key = settings.STRIPE_API_KEY

                # Create a unique identifier for this request
                import uuid

                event_id = f"colpali_{auth.user_id}_{uuid.uuid4().hex[:8]}"

                stripe.billing.MeterEvent.create(
                    event_name="colpali-embeddings",
                    payload={"value": str(num_embeddings), "stripe_customer_id": stripe_customer_id},
                    identifier=event_id,
                )
                logger.info(f"Sent Stripe metering event for user {auth.user_id}: {num_embeddings} ColPali embeddings")
            except Exception as e:
                # Log error but don't fail the request
                logger.error(f"Failed to send Stripe metering event: {e}")

        return ColPaliEmbeddingResponse(embeddings=embeddings, usage={"embeddings_count": num_embeddings})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calling ColPali API: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embeddings. Please try again later.")

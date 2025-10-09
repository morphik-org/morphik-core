"""SimpleAPI demo routes showcasing quotas, resource tracking, and admin helpers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from simpleapi import (
    SimpleAPIClient,
    SimpleAPIContext,
    SimpleAPIError,
    consume_resource,
    current_simpleapi_context,
    get_simpleapi_context,
)

router = APIRouter(prefix="/simple-api", tags=["simple-api"])

RESOURCE_LABELS = {
    "pages_ingested": "pages ingested",
    "graph_queries": "graph queries",
    "agent_calls": "agent calls",
    "cache_entries": "cache entries",
    "cache_queries": "cache queries",
}


def _quota_message(resource_id: str, context: SimpleAPIContext) -> str:
    label = RESOURCE_LABELS.get(resource_id, resource_id.replace("_", " "))
    return f"{label.capitalize()} quota exceeded for plan '{context.plan.id}'."


class CreateKeyRequest(BaseModel):
    plan_id: str | None = Field(default=None, alias="planId")
    label: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class DemoIngestRequest(BaseModel):
    documents: int = Field(default=1, ge=1, le=50)
    estimated_pages: int = Field(default=1, ge=1, le=2000, alias="estimatedPages")
    actual_pages: int | None = Field(default=None, ge=1, le=10000, alias="actualPages")
    dry_run: bool = Field(default=False, alias="dryRun")

    model_config = ConfigDict(populate_by_name=True)


class DemoResourceUsageRequest(BaseModel):
    graph_queries: int = Field(default=0, ge=0, le=100, alias="graphQueries")
    agent_calls: int = Field(default=0, ge=0, le=100, alias="agentCalls")
    cache_entries: int = Field(default=0, ge=0, le=25, alias="cacheEntries")
    cache_queries: int = Field(default=0, ge=0, le=1000, alias="cacheQueries")
    dry_run: bool = Field(default=False, alias="dryRun")

    model_config = ConfigDict(populate_by_name=True)


class CacheProvisionRequest(BaseModel):
    caches: int = Field(default=1, ge=1, le=25)
    cache_queries: int = Field(default=0, ge=0, le=1000, alias="cacheQueries")
    preview_only: bool = Field(default=False, alias="previewOnly")

    model_config = ConfigDict(populate_by_name=True)


def _serialize_plan(context: SimpleAPIContext) -> dict:
    """Return plan information with Pydantic aliases preserved."""
    plan_payload = context.plan.model_dump(by_alias=True)
    plan_payload["limits"] = [limit.model_dump(by_alias=True) for limit in context.plan.limits]
    return plan_payload


@router.get("/hello")
async def simple_api_hello(
    context: SimpleAPIContext = Depends(get_simpleapi_context),
) -> dict:
    """Expose the authenticated SimpleAPI context for debugging."""
    return {
        "message": "Hello from Morphik via SimpleAPI!",
        "project": context.project.model_dump(by_alias=True),
        "plan": _serialize_plan(context),
        "key": context.key.model_dump(by_alias=True),
    }


@router.post("/ingest-demo", status_code=200)
async def simple_api_ingest_demo(
    payload: DemoIngestRequest,
    context: SimpleAPIContext = Depends(get_simpleapi_context),
) -> dict:
    """
    Demonstrate ingest metering and settlement against SimpleAPI quotas.

    The endpoint uses a preflight check to ensure pages are available, optionally performs
    a dry-run preview, and settles the final page count for the request.
    """
    target_pages = payload.actual_pages or payload.estimated_pages
    await context.ensure(
        "pages_ingested",
        target_pages,
        require=True,
        error_detail=_quota_message("pages_ingested", context),
    )

    if payload.dry_run:
        return {
            "allowed": True,
            "message": "Quota check passed – no usage recorded (dry run).",
            "requestedPages": target_pages,
            "planId": context.plan.id,
        }

    if payload.documents > 1:
        # Bill additional units when multiple documents are processed in a single call.
        context.add_units(payload.documents - 1)

    context.consume("pages_ingested", target_pages)
    return {
        "message": "Ingest usage recorded with settlement.",
        "documents": payload.documents,
        "pagesIngested": target_pages,
        "planId": context.plan.id,
    }


@router.post("/usage-demo", status_code=200)
async def simple_api_usage_demo(
    payload: DemoResourceUsageRequest,
    context: SimpleAPIContext = Depends(get_simpleapi_context),
) -> dict:
    """
    Demonstrate recording additional resource consumption alongside the endpoint units.
    """
    resource_counts = {
        "graph_queries": payload.graph_queries,
        "agent_calls": payload.agent_calls,
        "cache_entries": payload.cache_entries,
        "cache_queries": payload.cache_queries,
    }

    if payload.dry_run:
        preview: dict[str, bool] = {}
        for resource_id, amount in resource_counts.items():
            if amount <= 0:
                continue
            preview[resource_id] = await context.ensure(
                resource_id,
                amount,
                include_units=False,
                error_detail=_quota_message(resource_id, context),
            )
        return {
            "message": "Dry-run quota preview completed.",
            "planId": context.plan.id,
            "preview": preview,
        }

    for resource_id, amount in resource_counts.items():
        if amount <= 0:
            continue
        await context.ensure(
            resource_id,
            amount,
            include_units=False,
            require=True,
            error_detail=_quota_message(resource_id, context),
        )

    for resource_id, amount in resource_counts.items():
        if amount > 0:
            consume_resource(resource_id, amount)

    return {
        "message": "Resource usage recorded.",
        "applied": {key: value for key, value in resource_counts.items() if value > 0},
        "planId": context.plan.id,
    }


@router.post("/cache-provision", status_code=200)
async def simple_api_cache_provision(
    payload: CacheProvisionRequest,
    context: SimpleAPIContext = Depends(get_simpleapi_context),
) -> dict:
    """
    Demonstrate resource preflight checks with manual settlement for cache provisioning.
    """
    requested = {
        "cache_entries": payload.caches,
        "cache_queries": payload.cache_queries,
    }

    preview: dict[str, bool] = {}
    for resource_id, amount in requested.items():
        if amount <= 0:
            continue
        allowed = await context.ensure(
            resource_id,
            amount,
            include_units=False,
            require=not payload.preview_only,
            error_detail=_quota_message(resource_id, context),
        )
        preview[resource_id] = allowed if payload.preview_only else True

    if payload.preview_only:
        return {
            "message": "Cache provisioning preview only – no usage recorded.",
            "planId": context.plan.id,
            "preview": preview,
        }

    if payload.caches > 0:
        context.consume("cache_entries", payload.caches)
    if payload.cache_queries > 0:
        consume_resource("cache_queries", payload.cache_queries)

    tracker_context = current_simpleapi_context()
    return {
        "message": "Cache provisioning recorded.",
        "caches": payload.caches,
        "cacheQueries": payload.cache_queries,
        "planId": context.plan.id,
        "trackerPlanId": tracker_context.plan.id,
        "preview": preview,
    }


@router.post("/keys", status_code=201)
async def create_simple_api_key(payload: CreateKeyRequest) -> dict:
    """
    Mint a new SimpleAPI key for the requested plan.

    Requires the `SIMPLE_API_ADMIN_TOKEN` environment variable to be set so the Python
    client can authenticate with the SimpleAPI control plane.
    """
    client = SimpleAPIClient()
    try:
        key = await client.create_key(plan_id=payload.plan_id, label=payload.label)
    except SimpleAPIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return key

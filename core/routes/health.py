"""Health check routes for monitoring service status."""

import time
from datetime import UTC, datetime

import arq
from fastapi import APIRouter, Depends
from sqlalchemy import text

from core.config import get_settings
from core.dependencies import get_document_service, get_redis_pool
from core.models.responses import DetailedHealthCheckResponse, ServiceStatus

router = APIRouter(prefix="", tags=["health"])
settings = get_settings()


@router.get("/health", response_model=DetailedHealthCheckResponse)
async def health_check(redis: arq.ArqRedis = Depends(get_redis_pool), document_service=Depends(get_document_service)):
    """
    Comprehensive health check endpoint that queries all underlying services.

    Checks the following services:
    - PostgreSQL database
    - Redis
    - PGVector store
    - Storage service (Local/S3)
    - ColPali vector store (if enabled)
    """
    services = []
    overall_status = "healthy"

    # Check PostgreSQL Database
    try:
        start_time = time.time()
        # Simple query to check database connectivity
        async with document_service.db.async_session() as session:
            result = await session.execute(text("SELECT 1"))
            _ = result.scalar()
        response_time = (time.time() - start_time) * 1000
        services.append(
            ServiceStatus(
                name="postgresql",
                status="healthy",
                message="Database connection successful",
                response_time_ms=response_time,
            )
        )
    except Exception as e:
        overall_status = "unhealthy"
        services.append(
            ServiceStatus(name="postgresql", status="unhealthy", message=f"Database connection failed: {str(e)}")
        )

    # Check Redis
    try:
        start_time = time.time()
        # Simple ping to check Redis connectivity
        await redis.ping()
        response_time = (time.time() - start_time) * 1000
        services.append(
            ServiceStatus(
                name="redis", status="healthy", message="Redis connection successful", response_time_ms=response_time
            )
        )
    except Exception as e:
        overall_status = "unhealthy"
        services.append(ServiceStatus(name="redis", status="unhealthy", message=f"Redis connection failed: {str(e)}"))

    # Check PGVector Store
    try:
        start_time = time.time()
        # Check if vector store is initialized
        if hasattr(document_service.vector_store, "initialized") and document_service.vector_store.initialized:
            response_time = (time.time() - start_time) * 1000
            services.append(
                ServiceStatus(
                    name="pgvector",
                    status="healthy",
                    message="Vector store initialized",
                    response_time_ms=response_time,
                )
            )
        else:
            # Try a simple query to check connectivity
            async with document_service.vector_store.async_session() as session:
                result = await session.execute(text("SELECT 1"))
                _ = result.scalar()
            response_time = (time.time() - start_time) * 1000
            services.append(
                ServiceStatus(
                    name="pgvector",
                    status="healthy",
                    message="Vector store connection successful",
                    response_time_ms=response_time,
                )
            )
    except Exception as e:
        if overall_status == "healthy":
            overall_status = "degraded"
        services.append(
            ServiceStatus(name="pgvector", status="unhealthy", message=f"Vector store check failed: {str(e)}")
        )

    # Check Storage Service
    try:
        start_time = time.time()
        storage_type = settings.STORAGE_PROVIDER

        if storage_type == "local":
            # For local storage, check if the directory exists and is writable
            import os

            storage_path = settings.STORAGE_PATH
            if os.path.exists(storage_path) and os.access(storage_path, os.W_OK):
                response_time = (time.time() - start_time) * 1000
                services.append(
                    ServiceStatus(
                        name="storage",
                        status="healthy",
                        message=f"Local storage at {storage_path} is accessible",
                        response_time_ms=response_time,
                    )
                )
            else:
                raise Exception(f"Storage path {storage_path} not accessible")
        elif storage_type == "aws-s3":
            # For S3, we could do a lightweight HEAD request on the bucket
            # For now, just check if storage object is initialized
            if document_service.storage:
                response_time = (time.time() - start_time) * 1000
                services.append(
                    ServiceStatus(
                        name="storage",
                        status="healthy",
                        message="S3 storage initialized",
                        response_time_ms=response_time,
                    )
                )
            else:
                raise Exception("S3 storage not initialized")
    except Exception as e:
        if overall_status == "healthy":
            overall_status = "degraded"
        services.append(ServiceStatus(name="storage", status="unhealthy", message=f"Storage check failed: {str(e)}"))

    # Check ColPali Vector Store (if enabled)
    if settings.ENABLE_COLPALI and document_service.colpali_vector_store:
        try:
            start_time = time.time()
            # Check if ColPali is initialized
            if hasattr(document_service.colpali_vector_store, "initialized"):
                is_initialized = document_service.colpali_vector_store.initialized
            else:
                is_initialized = True  # Assume initialized if no attribute

            if is_initialized:
                response_time = (time.time() - start_time) * 1000
                services.append(
                    ServiceStatus(
                        name="colpali",
                        status="healthy",
                        message="ColPali vector store initialized",
                        response_time_ms=response_time,
                    )
                )
            else:
                raise Exception("ColPali not initialized")
        except Exception as e:
            if overall_status == "healthy":
                overall_status = "degraded"
            services.append(
                ServiceStatus(name="colpali", status="unhealthy", message=f"ColPali check failed: {str(e)}")
            )

    return DetailedHealthCheckResponse(
        status=overall_status, services=services, timestamp=datetime.now(UTC).isoformat()
    )

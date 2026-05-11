import logging
from typing import Any, Dict


def arq_result_key(job_id: str) -> str:
    return f"arq:result:{job_id}"


async def enqueue_job_clearing_stale_result(
    redis: Any,
    function_name: str,
    job_payload: Dict[str, Any],
    *,
    logger: logging.Logger,
    context: str,
) -> Any:
    """Enqueue an ARQ job after removing a completed result for the same job id.

    ARQ treats existing result keys as duplicate job ids. Ingestion uses stable
    per-document job ids for dedupe, so a completed result can otherwise block a
    legitimate later update/requeue and leave the document marked processing.
    """

    job_id = job_payload.get("_job_id")
    if isinstance(job_id, str) and job_id:
        result_key = arq_result_key(job_id)
        try:
            deleted = await redis.delete(result_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to clear stale ARQ result key %s before enqueue (%s): %s", result_key, context, exc)
        else:
            if deleted:
                logger.info("Cleared stale ARQ result key %s before enqueue (%s)", result_key, context)

    return await redis.enqueue_job(function_name, **job_payload)

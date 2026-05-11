import logging

import pytest

from core.utils.arq_jobs import arq_result_key, enqueue_job_clearing_stale_result


class FakeRedis:
    def __init__(self, *, delete_result=0, enqueue_result=None, delete_raises=None):
        self.delete_result = delete_result
        self.enqueue_result = enqueue_result
        self.delete_raises = delete_raises
        self.deleted_keys = []
        self.enqueued = []

    async def delete(self, key):
        self.deleted_keys.append(key)
        if self.delete_raises:
            raise self.delete_raises
        return self.delete_result

    async def enqueue_job(self, function_name, **job_payload):
        self.enqueued.append((function_name, job_payload))
        return self.enqueue_result


def test_arq_result_key():
    assert arq_result_key("ingest:doc-1") == "arq:result:ingest:doc-1"


@pytest.mark.asyncio
async def test_enqueue_clears_stale_result_key_before_enqueue():
    redis = FakeRedis(delete_result=1, enqueue_result=object())
    payload = {"_job_id": "ingest:doc-1", "document_id": "doc-1"}

    result = await enqueue_job_clearing_stale_result(
        redis,
        "process_ingestion_job",
        payload,
        logger=logging.getLogger(__name__),
        context="test",
    )

    assert result is redis.enqueue_result
    assert redis.deleted_keys == ["arq:result:ingest:doc-1"]
    assert redis.enqueued == [("process_ingestion_job", payload)]


@pytest.mark.asyncio
async def test_enqueue_returns_none_for_real_duplicate_job():
    redis = FakeRedis(delete_result=0, enqueue_result=None)
    payload = {"_job_id": "ingest:doc-1", "document_id": "doc-1"}

    result = await enqueue_job_clearing_stale_result(
        redis,
        "process_ingestion_job",
        payload,
        logger=logging.getLogger(__name__),
        context="test",
    )

    assert result is None
    assert redis.deleted_keys == ["arq:result:ingest:doc-1"]
    assert redis.enqueued == [("process_ingestion_job", payload)]


@pytest.mark.asyncio
async def test_enqueue_continues_if_stale_result_delete_fails():
    redis = FakeRedis(delete_raises=RuntimeError("redis delete failed"), enqueue_result=object())
    payload = {"_job_id": "ingest:doc-1", "document_id": "doc-1"}

    result = await enqueue_job_clearing_stale_result(
        redis,
        "process_ingestion_job",
        payload,
        logger=logging.getLogger(__name__),
        context="test",
    )

    assert result is redis.enqueue_result
    assert redis.deleted_keys == ["arq:result:ingest:doc-1"]
    assert redis.enqueued == [("process_ingestion_job", payload)]


@pytest.mark.asyncio
async def test_enqueue_without_job_id_does_not_delete():
    redis = FakeRedis(enqueue_result=object())
    payload = {"document_id": "doc-1"}

    result = await enqueue_job_clearing_stale_result(
        redis,
        "process_ingestion_job",
        payload,
        logger=logging.getLogger(__name__),
        context="test",
    )

    assert result is redis.enqueue_result
    assert redis.deleted_keys == []
    assert redis.enqueued == [("process_ingestion_job", payload)]

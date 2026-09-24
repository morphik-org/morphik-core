"""Real Redis/ARQ and PostgreSQL/pgvector regression tests for content updates.

Set CORE_UPDATE_TEST_POSTGRES_URI and CORE_UPDATE_TEST_REDIS_URL to disposable
services. Documents and queue keys are isolated per test; Redis is never flushed.
The parser and worker are real. Embeddings are deterministic and local here;
provider-backed API/download/retrieval verification is a separate runtime proof.
"""

import asyncio
import os
import uuid
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus
from arq.worker import Retry, Worker
from fastapi import HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

pytestmark = pytest.mark.integration
POSTGRES_URI = os.environ.get("CORE_UPDATE_TEST_POSTGRES_URI")
REDIS_URL = os.environ.get("CORE_UPDATE_TEST_REDIS_URL")


@pytest.fixture
async def runtime(tmp_path, monkeypatch):
    if not POSTGRES_URI or not REDIS_URL:
        pytest.skip("Set CORE_UPDATE_TEST_POSTGRES_URI and CORE_UPDATE_TEST_REDIS_URL")

    from core.config import get_settings
    from core.database.postgres_database import PostgresDatabase
    from core.models.auth import AuthContext
    from core.parser.morphik_parser import MorphikParser
    from core.services.ingestion_service import IngestionService
    from core.storage.local_storage import LocalStorage
    from core.vector_store.pgvector_store import PGVectorStore
    from core.workers.ingestion_worker import process_ingestion_job

    settings = get_settings()
    monkeypatch.setattr(settings, "ENABLE_COLPALI", False)
    monkeypatch.setattr(settings, "MODE", "self_hosted")
    queue_name = f"update-test:{uuid.uuid4()}"
    redis = await create_pool(RedisSettings.from_dsn(REDIS_URL), default_queue_name=queue_name)
    db = PostgresDatabase(POSTGRES_URI)
    store = PGVectorStore(POSTGRES_URI)
    assert await db.initialize()
    assert await store.initialize()
    storage = LocalStorage(str(tmp_path))
    embedding = [1.0] + [0.0] * (settings.VECTOR_DIMENSIONS - 1)
    model = SimpleNamespace(embed_for_ingestion=AsyncMock(side_effect=lambda chunks: [embedding for _ in chunks]))
    parser = MorphikParser(chunk_size=80, chunk_overlap=0)
    service = IngestionService(db, store, model, storage, parser)
    auth = AuthContext(user_id="update-regression")
    ctx = dict(database=db, vector_store=store, embedding_model=model, storage=storage, parser=parser)
    ids = []

    async def ingest(content="ORIGINAL-CONTENT " * 40):
        doc = await service.ingest_file_content(
            content.encode(),
            "policy.txt",
            "text/plain",
            {"proof": queue_name},
            auth,
            redis,
        )
        ids.append(doc.external_id)
        return doc

    def payload(doc):
        return dict(
            document_id=doc.external_id,
            file_key=doc.storage_info["key"],
            bucket=doc.storage_info["bucket"],
            original_filename=doc.filename,
            content_type=doc.content_type,
            auth_dict={"user_id": auth.user_id},
            use_colpali=False,
            ingestion_revision=doc.system_metadata.get("ingestion_revision", 0),
        )

    async def drain():
        worker = Worker(
            [process_ingestion_job],
            redis_pool=redis,
            queue_name=queue_name,
            ctx=ctx,
            burst=True,
            poll_delay=0.01,
            handle_signals=False,
            keep_result=3600,
        )
        await asyncio.wait_for(worker.async_run(), timeout=30)
        assert worker.jobs_failed == 0

    async def snapshot(doc):
        current = await db.get_document(doc.external_id, auth)
        async with db.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT chunk_number, content FROM vector_embeddings WHERE document_id = :id ORDER BY chunk_number"
                    ),
                    {"id": doc.external_id},
                )
            ).all()
        return current.model_dump(mode="json"), [tuple(row) for row in rows]

    r = SimpleNamespace(
        db=db,
        store=store,
        redis=redis,
        service=service,
        auth=auth,
        ctx=ctx,
        ingest=ingest,
        payload=payload,
        drain=drain,
        snapshot=snapshot,
        model=model,
        process=process_ingestion_job,
        embedding=embedding,
        queue_name=queue_name,
    )
    try:
        yield r
    finally:
        for document_id in ids:
            await store.delete_chunks_by_document_id(document_id)
            await db.delete_document(document_id, auth)
            for pattern in (f"arq:*:ingest:{document_id}*",):
                keys = [key async for key in redis.scan_iter(match=pattern)]
                if keys:
                    await redis.delete(*keys)
        await redis.delete(queue_name, queue_name + ":health-check")
        await redis.aclose()
        await store.engine.dispose()
        await db.engine.dispose()
        await db._ingestion_lock_engine.dispose()


async def test_update_with_retained_result_replaces_all_chunks_and_preserves_identity(runtime):
    r = runtime
    doc = await r.ingest()
    # Reproduce the exact legacy ID collision with a completed real ARQ job.
    job_id = f"ingest:{doc.external_id}:0"
    await r.redis.zrem(r.queue_name, job_id)
    await r.redis.delete("arq:job:" + job_id)
    legacy_payload = r.payload(doc)
    legacy_payload.pop("ingestion_revision")
    legacy = await r.redis.enqueue_job("process_ingestion_job", _job_id=f"ingest:{doc.external_id}", **legacy_payload)
    await r.drain()
    assert await legacy.status() == JobStatus.complete
    before, old_rows = await r.snapshot(doc)
    assert len(old_rows) > 1
    assert await r.redis.ttl("arq:result:" + legacy.job_id) > 0

    corrected = "CORRECTED-CONTENT retention is twenty-one days."
    for revision in (1, 2):
        updated = await r.service.queue_document_update(doc.external_id, r.auth, r.redis, content=corrected)
        assert updated.external_id == doc.external_id
        assert updated.metadata == before["metadata"]
        assert updated.system_metadata["ingestion_revision"] == revision
        new_job = Job(f"ingest:{doc.external_id}:{revision}", r.redis, _queue_name=r.queue_name)
        assert await new_job.status() == JobStatus.queued
        # Same revision retries remain deduplicated without losing result retention.
        assert await r.redis.enqueue_job("process_ingestion_job", _job_id=new_job.job_id, **r.payload(updated)) is None
        await r.drain()
        current, rows = await r.snapshot(doc)
        assert current["system_metadata"]["status"] == "completed"
        assert current["system_metadata"]["indexed_revision"] == revision
        assert rows == [(0, corrected)]
        assert (
            await r.service.storage.download_file(**{k: current["storage_info"][k] for k in ("bucket", "key")})
            == corrected.encode()
        )
        retrieved = await r.store.query_similar(r.embedding, 100, doc_ids=[doc.external_id])
        assert [chunk.content for chunk in retrieved] == [corrected]
        assert await r.redis.ttl("arq:result:" + legacy.job_id) > 0
        assert await new_job.status() == JobStatus.complete


async def test_superseded_and_duplicate_workers_make_no_writes(runtime):
    r = runtime
    initial = await r.ingest()
    old = await r.service.queue_document_update(initial.external_id, r.auth, r.redis, content="OLDER-UPDATE")
    latest = await r.service.queue_document_update(initial.external_id, r.auth, r.redis, content="LATEST-UPDATE")
    before = await r.snapshot(latest)
    for doc in (initial, old):
        result = await r.process(r.ctx, **r.payload(doc))
        assert result["status"] == "superseded"
        assert await r.snapshot(latest) == before
    assert r.model.embed_for_ingestion.await_count == 0
    await r.drain()
    finished = await r.snapshot(latest)
    assert finished[1] == [(0, "LATEST-UPDATE")]
    calls = r.model.embed_for_ingestion.await_count
    assert (await r.process(r.ctx, **r.payload(latest)))["status"] == "completed"
    assert await r.snapshot(latest) == finished
    assert r.model.embed_for_ingestion.await_count == calls


@pytest.mark.parametrize("fail_worker", [False, True])
async def test_active_worker_blocks_updates_until_all_writes_finish(runtime, fail_worker):
    r = runtime
    doc = await r.ingest()
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_embedding(chunks):
        started.set()
        await release.wait()
        if fail_worker:
            raise ValueError("synthetic parse/embedding failure")
        return [r.embedding for _ in chunks]

    r.model.embed_for_ingestion.side_effect = delayed_embedding
    task = asyncio.create_task(r.process(r.ctx, **r.payload(doc)))
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        before = await r.snapshot(doc)
        with pytest.raises(HTTPException) as error:
            await r.service.queue_document_update(doc.external_id, r.auth, r.redis, content="REJECTED")
        assert error.value.status_code == 409
        assert await r.snapshot(doc) == before
    finally:
        release.set()
        result = await task
    assert result["status"] == ("failed" if fail_worker else "completed")
    r.model.embed_for_ingestion.side_effect = lambda chunks: [r.embedding for _ in chunks]
    updated = await r.service.queue_document_update(doc.external_id, r.auth, r.redis, content="AFTER-WORKER")
    await r.drain()
    assert (await r.snapshot(updated))[1] == [(0, "AFTER-WORKER")]


async def test_partial_store_retry_cleans_untracked_chunks(runtime, monkeypatch):
    r = runtime
    doc = await r.ingest("RETRY-CONTENT")
    store_embeddings = r.store.store_embeddings

    async def partially_store(*args, **kwargs):
        await store_embeddings(*args, **kwargs)
        raise ConnectionError("synthetic connection failure after commit")

    monkeypatch.setattr(r.store, "store_embeddings", partially_store)
    with pytest.raises(Retry):
        await r.process({**r.ctx, "job_try": 1}, **r.payload(doc))
    before, rows = await r.snapshot(doc)
    assert rows and not before["chunk_ids"]
    monkeypatch.setattr(r.store, "store_embeddings", store_embeddings)
    await r.process({**r.ctx, "job_try": 2}, **r.payload(doc))
    current, rows = await r.snapshot(doc)
    assert current["system_metadata"]["status"] == "completed"
    assert rows == [(0, "RETRY-CONTENT")]


@pytest.mark.parametrize("outcome", [None, ConnectionError("synthetic Redis outage")])
async def test_enqueue_failure_is_reported_and_retry_gets_a_new_revision(runtime, monkeypatch, outcome):
    r = runtime
    doc = await r.ingest()
    await r.drain()
    enqueue = r.redis.enqueue_job
    monkeypatch.setattr(r.redis, "enqueue_job", AsyncMock(return_value=None, side_effect=outcome))
    with pytest.raises(HTTPException) as error:
        await r.service.queue_document_update(doc.external_id, r.auth, r.redis, content="CORRECTION")
    assert error.value.status_code == 500
    failed = await r.db.get_document(doc.external_id, r.auth)
    assert failed.system_metadata["status"] == "failed"
    monkeypatch.setattr(r.redis, "enqueue_job", enqueue)
    updated = await r.service.queue_document_update(doc.external_id, r.auth, r.redis, content="CORRECTION")
    assert updated.system_metadata["ingestion_revision"] == 2
    await r.drain()
    assert (await r.snapshot(updated))[1] == [(0, "CORRECTION")]


async def test_file_update_and_cleanup_failure_cannot_publish_stale_chunks(runtime, monkeypatch):
    r = runtime
    doc = await r.ingest()
    await r.drain()
    updated = await r.service.queue_document_update(
        doc.external_id,
        r.auth,
        r.redis,
        file=UploadFile(BytesIO(b"FILE-CORRECTION"), filename="replacement.txt"),
    )
    delete = r.store.delete_chunks_by_document_id
    monkeypatch.setattr(r.store, "delete_chunks_by_document_id", AsyncMock(return_value=False))
    result = await r.process(r.ctx, **r.payload(updated))
    assert result["status"] == "failed"
    assert (await r.db.get_document(doc.external_id, r.auth)).system_metadata["status"] == "failed"
    monkeypatch.setattr(r.store, "delete_chunks_by_document_id", delete)
    await r.process({**r.ctx, "job_try": 2}, **r.payload(updated))
    assert (await r.snapshot(updated))[1] == [(0, "FILE-CORRECTION")]


async def test_manual_requeue_recovers_retained_result_and_deduplicates_pending_job(runtime, monkeypatch):
    from core.models.request import RequeueIngestionRequest
    from core.routes import ingest as routes

    r = runtime
    monkeypatch.setattr(routes, "ingestion_service", r.service)
    doc = await r.ingest("REQUEUE-CONTENT")
    await r.drain()
    await r.db.update_document(doc.external_id, {"system_metadata": {"status": "failed"}}, r.auth)
    request = RequeueIngestionRequest(jobs=[{"external_id": doc.external_id}])
    response = await routes.requeue_ingest_jobs(request, r.auth, r.redis)
    assert response.results[0].status == "requeued"
    pending = await r.db.get_document(doc.external_id, r.auth)
    assert pending.system_metadata["ingestion_revision"] == 1
    before = await r.snapshot(pending)
    repeated = await routes.requeue_ingest_jobs(request, r.auth, r.redis)
    assert repeated.results[0].status == "already_queued"
    assert await r.snapshot(pending) == before
    await r.drain()
    assert (await r.snapshot(doc))[1] == [(0, "REQUEUE-CONTENT")]


async def test_cancelled_worker_releases_lock_and_cannot_resume_over_new_revision(runtime):
    r = runtime
    doc = await r.ingest("CANCELLED-CONTENT")
    started = asyncio.Event()

    async def paused_embedding(chunks):
        started.set()
        await asyncio.Event().wait()

    r.model.embed_for_ingestion.side_effect = paused_embedding
    task = asyncio.create_task(r.process(r.ctx, **r.payload(doc)))
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    r.model.embed_for_ingestion.side_effect = lambda chunks: [r.embedding for _ in chunks]
    updated = await r.service.queue_document_update(doc.external_id, r.auth, r.redis, content="AFTER-CANCEL")
    await r.drain()
    before = await r.snapshot(updated)
    assert (await r.process({**r.ctx, "job_try": 2}, **r.payload(doc)))["status"] == "superseded"
    assert await r.snapshot(updated) == before
    assert before[1] == [(0, "AFTER-CANCEL")]


async def test_revision_read_error_retries_without_writes(runtime, monkeypatch):
    r = runtime
    doc = await r.ingest("READ-RETRY")
    before = await r.snapshot(doc)
    session = r.db.async_session
    monkeypatch.setattr(r.db, "async_session", Mock(side_effect=OperationalError("SELECT", {}, Exception("offline"))))
    with pytest.raises(Retry):
        await r.process(r.ctx, **r.payload(doc))
    assert r.model.embed_for_ingestion.await_count == 0
    monkeypatch.setattr(r.db, "async_session", session)
    assert await r.snapshot(doc) == before
    await r.drain()
    assert (await r.snapshot(doc))[1] == [(0, "READ-RETRY")]

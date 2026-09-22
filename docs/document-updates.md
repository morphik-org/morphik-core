# Document update scheduling

Content updates preserve the document ID, user metadata, and folder association. Each accepted update increments
`system_metadata.ingestion_revision` and queues `ingest:<document_id>:<revision>`. Redis can retain the result of
an earlier ingestion without blocking the update. Completion records `system_metadata.indexed_revision`.

The API and worker take the same PostgreSQL advisory lock for a document. The lock covers initial upload, update
scheduling, manual requeue, and the worker's entire processing attempt, including progress and failure writes.
Workers check the stored revision and source location after taking the lock. Superseded jobs return without
changing files, chunks, or status; duplicate delivery after completion also makes no changes.

An update attempted during active processing returns HTTP 409 and leaves the document unchanged. The caller can
retry after processing finishes. Updates accepted while earlier jobs are still queued supersede those jobs;
the latest accepted revision is indexed. Repeating an HTTP content update creates another revision, even when
the bytes are identical. There is no client `If-Match` precondition or HTTP idempotency key in this change.

`completed` is set only after chunk replacement and its document update succeed. Cleanup includes untracked
chunks from interrupted attempts and rejects failed deletions. While processing or failed, the document remains
excluded from normal retrieval, as before. An enqueue exception or unexpected `None` result returns an error
and marks the persisted revision failed instead of reporting successful scheduling.

`POST /ingest/requeue` reads the current document under the same lock. A queued or active job returns
`already_queued` without resetting status. A finished or missing job gets a new revision, so a retained failed
result cannot prevent recovery. Requeue also recovers a revision left processing if the API exits between the
PostgreSQL write and Redis enqueue; these operations are not a distributed transaction.

## Deployment

Deploy the API and ingestion workers together. Drain or stop workers running the old code before accepting
updates through the new API. Older binaries do not take the lock or accept revision arguments. The new worker
accepts legacy queued messages without a revision, treating them as revision zero and checking their source
location. Existing documents need no migration; an absent revision is zero.

The lock uses one additional PostgreSQL connection per active ingestion or scheduling operation. These
connections use a separate unpooled engine so long-running jobs cannot exhaust the ordinary query pool.

## Verification

`core/tests/integration/test_document_update_revisions.py` runs the real ARQ worker, Redis, PostgreSQL/pgvector,
LocalStorage, and text parser with deterministic local embeddings. Point the following variables at disposable
services and run it with the project's installed dependencies and test configuration:

```bash
export CORE_UPDATE_TEST_POSTGRES_URI='postgresql+asyncpg://morphik:morphik@127.0.0.1:55438/morphik'
export CORE_UPDATE_TEST_REDIS_URL='redis://127.0.0.1:56388/0'
python -m pytest core/tests/integration/test_document_update_revisions.py
```

The tests retain an actual completed legacy ARQ result, replace many chunks with one, repeat updates, replay
superseded and completed jobs, reject updates during active workers, retry partial writes, cancel workers,
exercise enqueue/cleanup/read failures, and verify manual requeue. Each test removes only its own data and keys.

On September 8, 2026, the supplied full runtime proof also passed against the working tree based on `7f72d712`.
It used the real authenticated Core API and worker, real `text-embedding-3-small` embeddings, and isolated
PostgreSQL/pgvector and Redis containers. Corrected text was downloaded and retrieved under the same document
ID and metadata, with no original text in retrieval. The correction survived container recreation and API/worker
restart. This verifies a synthetic standard-text case; it does not cover production images, PDF/OCR, or ColPali.

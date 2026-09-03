# iQor on-prem deployment audit

Audit baseline: `origin/main` at `8c51b8d` on 2026-09-02.

This document separates Morphik Core behavior from iQor's MCP wrapper and UI. It also records which findings have an
executable test. A code path alone is not counted as a passing deployment check.

## Status

| Finding | Status | Evidence | Owner |
| --- | --- | --- | --- |
| Routine stop deletes PostgreSQL data | Verified in Docker | The recreation test calls `stop-morphik.sh` twice, restarts PostgreSQL, and reads the original document from the preserved volume. | Morphik Core |
| Stop leaves optional containers behind | Implemented; runtime proof pending | `stop-morphik.sh` runs `down` with `--profile "*" --remove-orphans`. Static lifecycle tests pass. Runtime Docker verification is still pending. | Morphik Core |
| Start rewrites Compose state and fixed container names collide | Implemented; static verification passed | The API port now uses `MORPHIK_API_PORT`; production services use Compose project-scoped names; static lifecycle tests and `docker compose config` pass. | Morphik Core |
| Documents survive PostgreSQL container recreation | Verified in Docker | `scripts/test_postgres_persistence.sh` inserts a document row, recreates PostgreSQL, and checks the original ID and metadata. | Morphik Core / iQor infrastructure |
| Text update preserves document identity and existing metadata | Verified in a unit test | `test_queued_text_update_preserves_identity_metadata_and_queues_reindex` passes. | Morphik Core |
| Text update queues changed content for re-indexing and exposes processing status | Partially verified | The unit test proves the replacement object and `process_ingestion_job` payload use the same document ID and that the returned status is `processing`. Existing SDK status tests pass. No end-to-end test in this audit proves the changed text is retrievable after the worker finishes. | Morphik Core |
| Text update prevents lost updates | Fails | There is no content revision precondition. Every update uses ARQ job ID `ingest:{document_id}`. A second update can receive a successful API response while `enqueue_job` returns `None`, and the first queued job may refer to an object the second update deleted. | Morphik Core, then iQor caller adoption |
| `min_score` affects retrieval | Fixed and unit verified in this branch | `test_min_score_zero_keeps_zero_and_positive_scores` and `test_min_score_filters_on_the_final_score` pass. | Morphik Core |
| Work item 47490 returns five distinct QA backlog items | Not verified | The repository has no iQor corpus, query text, auth token, or captured response. `scripts/verify_iqor_retrieval.sh` captures and validates the response on iQor's deployment. | iQor MCP wrapper / iQor acceptance test |
| Default Docker config keeps document and query data on premises | Fails by default | `morphik.docker.toml` selects OpenAI for completion and standard embeddings. Telemetry is enabled unless `TELEMETRY=false`. See the data boundary below. | Joint configuration decision |
| Core, MCP wrapper, and UI responsibilities are separated | Documented | See the ownership table below. The iQor MCP wrapper and existing iQor UI are not in this repository. | Joint |

## Lifecycle and persistence contract

The production Compose file stores PostgreSQL data in the `postgres_data` named volume. Uploaded source files use the
host bind mount `./storage`. Neither location is part of the PostgreSQL or Morphik container writable layer.

iQor should set a stable `COMPOSE_PROJECT_NAME`, such as `iqor-morphik`, before the first production start. On an
existing deployment, keep the current project name. Changing it later selects a different named volume and makes the
old database appear missing even though Docker still has the original volume.

Normal lifecycle commands are safe to repeat:

```bash
./start-morphik.sh
./start-morphik.sh
./stop-morphik.sh
./stop-morphik.sh
```

`stop-morphik.sh` removes containers and the Compose network. It does not remove named volumes. The explicit reset
command below deletes PostgreSQL, Redis, downloaded model state, and UI build volumes:

```bash
docker compose -f docker-compose.run.yml --profile "*" down --volumes --remove-orphans
```

Treat that command as destructive. Back up PostgreSQL and `./storage` first.

### Automated recreation test

Run this on a disposable Docker host or CI runner:

```bash
./scripts/test_postgres_persistence.sh
```

The test uses a unique project name beginning with `morphik-persistence-test-`. It starts only PostgreSQL, creates a
`documents` table with the Morphik identity and metadata fields, inserts the 47490 probe row, and checks it after direct
container recreation. It then calls `stop-morphik.sh` twice, starts PostgreSQL again against the same volume, and checks
for:

```text
iqor-persistence-probe|QA|47490
```

It deletes only its isolated test project and volume on exit. The pytest entry point is:

```bash
.venv/bin/pytest -q core/tests/integration/test_docker_postgres_persistence.py
```

Captured on 2026-09-02 with Docker Engine 27.4.0:

```text
PASS: document identity and metadata survived container recreation and repeated stop/start.
1 passed
```

## Document update contract

iQor's `documents.updateText` call maps to:

```http
POST /documents/{document_id}/update_text
Content-Type: application/json
Authorization: Bearer <token>
```

```json
{
  "content": "corrected text",
  "filename": "optional-name.txt",
  "metadata": {},
  "metadata_types": {},
  "use_colpali": false
}
```

Observed behavior:

- The path `document_id` remains the document `external_id`.
- Omitted metadata is preserved. Supplied metadata is merged. Reserved identity, folder, app, owner, and end-user fields
  cannot be changed through this endpoint.
- Morphik uploads the replacement content, changes status to `processing`, and queues `process_ingestion_job`.
- The worker deletes old vector chunks before storing replacement chunks. It marks the document `completed` after the
  write, or `failed` after terminal errors.
- `GET /documents/{document_id}/status` returns `document_id`, `status`, `filename`, `created_at`, `updated_at`, optional
  `progress`, and optional `error`.

The lost-update criterion does not pass. The update request has no `expected_version` or `If-Match` value, and the
document does not have a general content revision that the database checks atomically. The queue key is also shared by
all updates for a document. A proper fix needs all of the following in one change:

1. Return a content revision on document reads and update responses.
2. Require or accept that revision on updates and return HTTP 409 when it is stale.
3. Include the accepted revision in the ARQ job ID and payload.
4. Make workers skip stale revisions before progress, chunk deletion, status, or failure writes.
5. Retain old source objects until the accepted revision completes, then clean them up.

Adding only a request version check would still leave older workers able to overwrite newer chunks or status.

## Retrieval contract for work item 47490

Morphik Core's evidence endpoint is `POST /retrieve/chunks`. For iQor's acceptance test, use an over-fetched candidate
set and let the MCP wrapper deduplicate by work item ID. Core's `k` counts chunks, not distinct work items.

The exact metadata key names and the ID type must match ingestion. This example assumes `project`, `work_item_type`, and
numeric `work_item_id` metadata:

```json
{
  "query": "<the retrieval text for work item 47490>",
  "k": 50,
  "min_score": 0.0,
  "use_colpali": false,
  "output_format": "text",
  "filters": {
    "$and": [
      {"project": {"$eq": "QA"}},
      {"work_item_type": {"$eq": "Product Backlog Item"}},
      {"work_item_id": {"$ne": 47490}}
    ]
  }
}
```

If ingestion stored `work_item_id` as a string, use `"47490"` in the filter. A numeric predicate does not match a string
field.

The response is a JSON array sorted by descending relevance before any wrapper-side grouping:

```json
[
  {
    "content": "matching chunk text",
    "score": 0.82,
    "document_id": "persistent-morphik-document-id",
    "chunk_number": 3,
    "metadata": {
      "project": "QA",
      "work_item_type": "Product Backlog Item",
      "work_item_id": 47501
    },
    "content_type": "text/plain",
    "filename": "47501.txt",
    "download_url": null,
    "is_padding": false
  }
]
```

Use `document_id` plus `chunk_number` as the stable evidence source ID, for example
`persistent-morphik-document-id:3`. Keep the numeric `score` alongside the source. Higher is better. Standard pgvector
cosine scores and the configured normalized reranker are in the 0 to 1 range. Other multivector implementations may
not be calibrated identically, so iQor should validate any nonzero threshold against its chosen backend. This branch
applies `min_score` after vector scoring and optional reranking, before zero-score padding is added.

Core can return several chunks from one work item. The MCP wrapper should keep the highest-scoring chunk for each
`metadata.work_item_id`, sort those representatives by score, and return the first five. It must reject a result if the
project or type is wrong, the current work item is present, the work-item ID is missing or has the wrong type, a source
ID is missing, or fewer than five distinct IDs are available.

### Error behavior

| Condition | HTTP/result behavior |
| --- | --- |
| Missing query and image, both supplied, invalid field shape, or `k <= 0` | FastAPI validation response, HTTP 422 |
| Invalid metadata operator or operand | `{"detail":"..."}`, HTTP 400 |
| Invalid image or image retrieval without ColPali | `{"detail":"..."}`, HTTP 400 |
| Invalid or unauthorized scope | `{"detail":"..."}`, normally HTTP 403 |
| Valid request with no authorized matching documents | `[]`, HTTP 200 |
| Unhandled API or provider error | HTTP 500 |
| Some vector-store query failures | The current stores log the backend error and may return `[]` with HTTP 200 |

The last behavior means the MCP wrapper must not treat an empty 200 as proof that no matching backlog items exist. It
should log the Morphik request ID and check service logs or health when an expected corpus returns no results.

Capture and validate the real response with:

```bash
IQOR_QUERY='<query used by the PO Agent for work item 47490>' \
MORPHIK_BASE_URL='https://customer-hosted-morphik.example' \
MORPHIK_AUTH_TOKEN='<token>' \
./scripts/verify_iqor_retrieval.sh
```

The script saves the raw response in the operating system's temporary directory unless `IQOR_RESPONSE_FILE` is set.
It prints the chosen path to stderr, validates the filters, exclusion, scores, and source fields, then prints five
deduplicated items. Use `IQOR_WORK_ITEM_ID_JSON='"47490"'` if IDs are strings. Set `IQOR_RESPONSE_FILE` only to an
approved location outside the repository when iQor needs to retain the raw customer response.

## On-prem data boundary

The stock production configuration is not a no-egress configuration.

| Path | Default or trigger | Data sent outside the Morphik containers |
| --- | --- | --- |
| Standard ingestion embeddings | `morphik.docker.toml` selects `openai_embedding` | Parsed document chunks go to OpenAI. |
| Retrieval embeddings | Same embedding selection | The user's retrieval query goes to OpenAI. |
| Query completion | `openai_gpt4-1-mini` is the default completion model | Query, retrieved context, prompt, and optional chat history go to OpenAI. |
| Contextual chunking | Off by default; model defaults to OpenAI if enabled | The document text and each chunk go to the selected completion provider. |
| Video frame descriptions | Frame sampling is off by default | Sampled images and prompt text go to the selected vision provider when enabled. |
| Video transcription | Only when `ASSEMBLYAI_API_KEY` is set | Audio goes to AssemblyAI. |
| ColPali API or parser API mode | Local by default | Document images, text, or file bytes go to every configured Morphik embedding API endpoint when API mode is selected. |
| S3 or Morphik multivector storage | Local PostgreSQL and filesystem by default | Source objects go to S3; vectors and chunk content can go to Turbopuffer when those providers are selected. |
| Morphik telemetry | Enabled by default unless `TELEMETRY=false` | Compressed operation events go to `https://logs.morphik.ai`. Query and folder strings are redacted, but events include installation, user/app IDs, operation fields, document IDs or filenames for some operations, timings, and error strings. |
| Historical log lookup | `GET /logs?hours=<value greater than 4>` while telemetry is enabled | The app ID, start time, limit, operation type, and status filter go to `https://logs.morphik.ai`. With `TELEMETRY=false`, this branch returns an empty list without contacting the proxy. |
| Sentry | Only when `SENTRY_DSN` is set | Errors, traces, profiles, request context, and default PII go to the configured Sentry project. |
| LiteLLM model-price map | Disabled by default in this branch's production Compose environment | Without `LITELLM_LOCAL_MODEL_COST_MAP=True`, process import downloads a JSON map from GitHub. This request contains no customer document data. |
| Model and image downloads | First install or uncached local model start | Container images and model weights come from GHCR, Docker Hub, Hugging Face, Ollama, or the configured registry. Customer documents are not part of these downloads. |
| Public-IP lookup | Only when the local URI generation path requests automatic host discovery | The server calls `checkip.amazonaws.com`; no document body is sent. |

For a no-egress iQor deployment:

- Set `TELEMETRY=false`, leave `SENTRY_DSN` empty, and keep
  `LITELLM_LOCAL_MODEL_COST_MAP=True` in `.env`.
- Point the completion and embedding model entries at customer-hosted Ollama or another internal OpenAI-compatible
  service. Do not leave `openai_embedding` or `openai_gpt4-1-mini` selected.
- Keep `colpali_mode="local"` or `"off"` and `parser_mode="local"`.
- Keep storage `local`, vector store `pgvector`, and multivector store `postgres` unless iQor approves the alternative
  destination.
- Do not set OpenAI, Anthropic, Gemini, AssemblyAI, Turbopuffer, AWS, or Sentry credentials unless that provider is
  inside the approved boundary.
- Mirror container images and model weights into iQor-approved registries if the runtime network has no internet
  access.
- Enforce the policy with an egress-deny rule. Configuration review alone cannot prove that a host has no outbound
  path.

## Ownership split

| Morphik Core owns | iQor MCP wrapper owns | iQor UI owns |
| --- | --- | --- |
| Durable PostgreSQL and source-file mounts | Translating PO Agent inputs into the documented request | Pointing at the customer-hosted MCP or Morphik endpoint |
| Safe start/stop scripts and opt-in destructive reset | Applying the exact QA, backlog type, and current-ID filters | Sending corrected text through the existing update action |
| Authentication, metadata filtering, vector scoring, minimum score, chunk source fields | Over-fetching chunks, deduplicating by work item ID, returning exactly five, and preserving score/source ID | Showing processing, completed, failed, conflict, and retrieval error states |
| Stable document identity, update processing state, re-index worker, and a future revision conflict contract | Mapping Morphik errors into MCP tool errors without turning empty results into success | Polling document status after an update and preventing duplicate submissions while processing |
| Provider selection points and documented outbound integrations | No model or embedding calls outside Morphik Core | No direct model or storage-provider calls unless iQor explicitly designs them |

The iQor MCP wrapper and existing UI are not present in this repository, so this branch does not change or test them.

## Files changed by this audit

- `docker-compose.run.yml`: project-scoped service names, parameterized API port, no host-published PostgreSQL port,
  persistent-volume comments, Compose V2-compatible `.env` loading, and local LiteLLM cost map.
- `start-morphik.sh`: repeatable startup without temporary Compose files, profile support, argument validation, and orphan
  cleanup.
- `stop-morphik.sh`: checked-in, repeatable stop that activates every profile and preserves volumes.
- `install_docker.sh` and `install_docker.ps1`: generate the safe start/stop behavior and local LiteLLM cost map setting.
- `.env.example`: project-name migration warning and local LiteLLM cost map setting.
- `.gitignore`: defense-in-depth exclusion for explicitly named iQor retrieval captures.
- `DOCKER.md`: safe stop and explicit destructive reset documentation.
- `core/services/document_service.py`: enforce `min_score` after final scoring.
- `core/routes/logs.py`: avoid the historical Morphik log proxy when telemetry is disabled.
- `core/tests/unit/test_docker_lifecycle.py`: lifecycle and installer guards.
- `core/tests/unit/test_logs_no_egress.py`: historical-log no-egress test.
- `core/tests/integration/test_docker_postgres_persistence.py` and `scripts/test_postgres_persistence.sh`: container
  recreation test.
- `core/tests/unit/test_ingestion_service_metadata_update.py`: identity, metadata, status, and re-index queue test.
- `core/tests/unit/test_retrieval_contract.py`: minimum-score tests.
- `scripts/verify_iqor_retrieval.sh`: captured-response acceptance check for work item 47490.

## Reproduction and verification commands

Confirm the audit base:

```bash
git fetch origin --prune
git rev-list --left-right --count HEAD...origin/main
git log -1 --oneline origin/main
```

The audit began with `0 0` and `8c51b8d` before these branch changes.

Validate the lifecycle scripts and rendered production Compose model without starting containers:

```bash
bash -n start-morphik.sh stop-morphik.sh install_docker.sh \
  scripts/test_postgres_persistence.sh scripts/verify_iqor_retrieval.sh
MORPHIK_API_PORT=8123 MORPHIK_VERSION=test MORPHIK_ENV_FILE=/dev/null \
  docker compose -f docker-compose.run.yml --profile "*" config
```

Run the tests that passed during this audit:

```bash
.venv/bin/pytest -q core/tests/unit/test_docker_lifecycle.py
.venv/bin/pytest -q core/tests/unit/test_ingestion_service_metadata_update.py
LITELLM_LOCAL_MODEL_COST_MAP=True \
  .venv/bin/pytest -q core/tests/unit/test_retrieval_contract.py
.venv/bin/pytest -q core/tests/unit/test_iqor_retrieval_script.py
.venv/bin/pytest -q core/tests/unit/test_logs_no_egress.py
.venv/bin/pytest -q sdks/python/morphik/tests/test_document_status.py
```

Run the Docker proof on a host with its daemon running:

```bash
.venv/bin/pytest -q core/tests/integration/test_docker_postgres_persistence.py
# or
./scripts/test_postgres_persistence.sh
```

Reproduce the original destructive behavior on an old installation only after taking a backup. Inspect the generated
script rather than running it against customer data:

```bash
rg -n 'down.*--volumes' stop-morphik.sh install_docker.sh install_docker.ps1
```

On the audited base, both installers generated a normal stop containing `down --volumes --remove-orphans`. On this
branch, the command returns no matches for the production stop paths.

## Remaining questions for iQor

1. What exact metadata keys and types are stored for project, work-item type, and work-item ID? Is the backlog value
   exactly `Product Backlog Item`?
2. What exact query text does the PO Agent derive from work item 47490, and which embedding, reranker, and ColPali
   settings must the acceptance environment use?
3. Does "five distinct" mean five work item IDs even when one item has several matching chunks? The proposed wrapper
   contract assumes yes.
4. Should the MCP response return one best chunk per work item or all supporting chunks under each of the five items?
5. Can iQor change `documents.updateText` to send and handle a content revision, including HTTP 409, once Morphik adds
   the lost-update fix?
6. Must the deployment have zero outbound network access, or are approved internal Azure/OpenAI, S3, telemetry, or
   registry endpoints allowed?
7. Who owns PostgreSQL backups, restore drills, retention, encryption keys, and the stable `COMPOSE_PROJECT_NAME` in
   customer infrastructure?
8. Will iQor use the bundled Redis container? Redis persistence is retained now, but queued jobs are not a substitute
   for a database backup or an update revision contract.

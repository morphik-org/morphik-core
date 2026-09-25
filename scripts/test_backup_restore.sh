#!/usr/bin/env bash
# Docker test for morphik-backup.sh against the production Compose file.
#
# It seeds a Core-shaped database and storage directory, takes a backup, verifies it, deletes the
# PostgreSQL volume and the storage files, restores, and checks that documents, metadata,
# embeddings, and files are identical. It then checks the refusal paths, restore --force over
# existing data, scheduled backups with retention, and the off-host copy against an S3 mock.
#
# Everything runs under a unique Compose project in a temporary install directory. Cleanup
# removes only that project, its volumes, and that directory.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TEST_PROJECT="morphik-backup-test-$$-${RANDOM}"
TEST_VOLUME="${TEST_PROJECT}_postgres_data"
TEST_NETWORK="${TEST_PROJECT}_morphik-network"
TOOL_IMAGE="pgvector/pgvector:pg16"
S3_IMAGE="${MORPHIK_BACKUP_TEST_S3_IMAGE:-adobe/s3mock:latest}"
S3_CONTAINER="${TEST_PROJECT}-s3"
RUN_S3="${MORPHIK_BACKUP_TEST_S3:-1}"
SECRET_SENTINEL="sentinel-secret-${RANDOM}${RANDOM}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "Docker is installed, but the daemon is not running." >&2
    exit 1
fi
if docker volume inspect "$TEST_VOLUME" >/dev/null 2>&1 ||
    docker network inspect "$TEST_NETWORK" >/dev/null 2>&1 ||
    docker ps -aq --filter "label=com.docker.compose.project=$TEST_PROJECT" | grep -q .; then
    echo "Refusing to reuse existing Docker resources for test project $TEST_PROJECT." >&2
    exit 2
fi

INSTALL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/morphik-backup-test.XXXXXX")
INSTALL_DIR=$(cd "$INSTALL_DIR" && pwd -P)
cp "$REPO_DIR/docker-compose.run.yml" "$REPO_DIR/morphik-backup.sh" "$REPO_DIR/morphik-compose-project.sh" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/morphik-backup.sh"

export COMPOSE_PROJECT_NAME="$TEST_PROJECT"
export MORPHIK_ENV_FILE="$INSTALL_DIR/.env"
export MORPHIK_BACKUP_TOOL_IMAGE="$TOOL_IMAGE"
compose=(docker compose --project-name "$TEST_PROJECT" --project-directory "$INSTALL_DIR" -f "$INSTALL_DIR/docker-compose.run.yml")

cleanup() {
    # This project name is unique to the test. Removing its volumes cannot touch a deployment.
    "${compose[@]}" --profile "*" down --volumes --remove-orphans >/dev/null 2>&1 || true
    docker rm -f "$S3_CONTAINER" >/dev/null 2>&1 || true
    # Restored files are owned by root on Linux, so delete them from a container.
    docker run --rm -v "$INSTALL_DIR:/install" --entrypoint sh "$TOOL_IMAGE" -c 'rm -rf /install/* /install/.[!.]*' >/dev/null 2>&1 || true
    rm -rf "$INSTALL_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

pass() {
    echo "ok: $*"
}

backup_tool() {
    (cd "$INSTALL_DIR" && ./morphik-backup.sh "$@")
}

psql_q() {
    "${compose[@]}" exec -T postgres psql -X -At -v ON_ERROR_STOP=1 -U morphik -d morphik "$@"
}

wait_for_postgres() {
    local attempts=30
    while ((attempts > 0)); do
        if "${compose[@]}" exec -T postgres pg_isready -U morphik -d morphik >/dev/null 2>&1; then
            return 0
        fi
        attempts=$((attempts - 1))
        sleep 2
    done
    "${compose[@]}" logs postgres >&2 || true
    fail "PostgreSQL did not become ready."
}

write_config() {
    local dimensions=$1 backup_enabled=${2:-false} s3_uri=${3:-}
    cat >"$INSTALL_DIR/morphik.toml" <<EOF
[registered_models]
test_embedding = { model_name = "test/fake-embedding-8", api_base = "http://embeddings:8000" }

[embedding]
model = "test_embedding"  # Reference to registered model
dimensions = $dimensions
similarity_metric = "cosine"

[database]
provider = "postgres"

[vector_store]
provider = "pgvector"

[multivector_store]
provider = "postgres"

[storage]
provider = "local"
storage_path = "./storage"

[morphik]
enable_colpali = false
colpali_mode = "off"

[backup]
enabled = $backup_enabled
interval_hours = 0.0001
directory = "./backups"
keep = 2
verify = true
s3_uri = "$s3_uri"
EOF
}

# The documents and vector tables match Core's column layout. The storage files use the
# two bucket forms LocalStorage writes: bucket "storage" with a relative key, and an empty bucket.
seed_database() {
    psql_q <<'SQL' >/dev/null
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
    external_id VARCHAR PRIMARY KEY,
    content_type VARCHAR,
    filename VARCHAR,
    doc_metadata JSONB DEFAULT '{}'::jsonb,
    metadata_types JSONB DEFAULT '{}'::jsonb,
    storage_info JSONB DEFAULT '{}'::jsonb,
    system_metadata JSONB DEFAULT '{}'::jsonb,
    additional_metadata JSONB DEFAULT '{}'::jsonb,
    chunk_ids JSONB DEFAULT '[]'::jsonb,
    owner_id VARCHAR,
    app_id VARCHAR,
    folder_name VARCHAR,
    folder_path VARCHAR,
    folder_id VARCHAR,
    end_user_id VARCHAR
);
CREATE TABLE folders (id VARCHAR PRIMARY KEY, name VARCHAR, app_id VARCHAR, document_ids JSONB DEFAULT '[]'::jsonb);
CREATE TABLE vector_embeddings (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    chunk_metadata TEXT,
    embedding vector(8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX vector_idx ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1);
CREATE TABLE multi_vector_embeddings (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    chunk_metadata TEXT,
    embeddings BIT(128)[]
);
INSERT INTO folders VALUES ('folder-qa', 'QA', 'iqor-test', '["doc-completed","doc-pdf"]');
INSERT INTO documents (external_id, content_type, filename, doc_metadata, metadata_types, storage_info, system_metadata, chunk_ids, owner_id, app_id, folder_name, folder_id)
VALUES
    ('doc-completed', 'text/plain', '47490.txt',
     '{"project":"QA","work_item_type":"Product Backlog Item","work_item_id":47490,"tags":["a","b"]}',
     '{"project":"string","work_item_type":"string","work_item_id":"number"}',
     '{"bucket":"storage","key":"ingest_uploads/doc-completed/47490.txt"}',
     '{"status":"completed","ingestion_revision":1,"indexed_revision":1}',
     '["doc-completed-0","doc-completed-1"]', 'iqor-test', 'iqor-test', 'QA', 'folder-qa'),
    ('doc-pdf', 'application/pdf', 'spec ü.pdf',
     '{"project":"QA","work_item_id":47501}', '{"work_item_id":"number"}',
     '{"bucket":"","key":"doc-pdf/spec ü.pdf"}',
     '{"status":"completed"}', '["doc-pdf-0"]', 'iqor-test', 'iqor-test', 'QA', 'folder-qa'),
    ('doc-processing', 'text/plain', 'in-flight.txt',
     '{"project":"QA","work_item_id":47502}', '{"work_item_id":"number"}',
     '{"bucket":"storage","key":"ingest_uploads/doc-processing/in-flight.txt"}',
     '{"status":"processing","ingestion_revision":2,"progress":{"step":"embedding"}}', '[]',
     'iqor-test', 'iqor-test', NULL, NULL);
INSERT INTO vector_embeddings (document_id, chunk_number, content, chunk_metadata, embedding) VALUES
    ('doc-completed', 0, 'first chunk', '{"page":1}', '[0.123456789,-1.5e-07,3.4028235e+38,0.1,0.2,0.3,0.4,0.5]'),
    ('doc-completed', 1, 'second chunk', '{"page":2}', '[1,2,3,4,5,6,7,8.000001]'),
    ('doc-pdf', 0, 'pdf chunk', '{}', '[-0.333333343,0.666666687,0,0,0,0,0,1e-38]');
INSERT INTO multi_vector_embeddings (document_id, chunk_number, content, embeddings) VALUES
    ('doc-pdf', 0, 'page image', ARRAY[B'10101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010']::bit(128)[]);
SQL
}

seed_storage() {
    mkdir -p "$INSTALL_DIR/storage/ingest_uploads/doc-completed" "$INSTALL_DIR/storage/doc-pdf" \
        "$INSTALL_DIR/storage/ingest_uploads/doc-processing"
    printf 'QA backlog item 47490\n' >"$INSTALL_DIR/storage/ingest_uploads/doc-completed/47490.txt"
    head -c 200000 /dev/urandom >"$INSTALL_DIR/storage/doc-pdf/spec ü.pdf"
    printf 'still ingesting\n' >"$INSTALL_DIR/storage/ingest_uploads/doc-processing/in-flight.txt"
}

# Everything the restore must bring back byte-for-byte. The processing document's status is
# excluded because restore deliberately marks it failed; it is checked separately.
snapshot_state() {
    psql_q <<'SQL'
SELECT 'doc|' || external_id || '|' || COALESCE(filename, '') || '|' || doc_metadata::text || '|' || metadata_types::text || '|' || storage_info::text || '|' || chunk_ids::text || '|' || COALESCE(folder_id, '') || '|' ||
       CASE WHEN external_id = 'doc-processing' THEN '' ELSE system_metadata::text END
FROM documents ORDER BY external_id;
SELECT 'vec|' || document_id || '|' || chunk_number || '|' || content || '|' || embedding::text FROM vector_embeddings ORDER BY document_id, chunk_number;
SELECT 'mv|' || document_id || '|' || chunk_number || '|' || embeddings::text FROM multi_vector_embeddings ORDER BY document_id, chunk_number;
SELECT 'folder|' || id || '|' || document_ids::text FROM folders ORDER BY id;
SQL
    docker run --rm -v "$INSTALL_DIR/storage:/storage:ro" --entrypoint bash "$TOOL_IMAGE" \
        -c 'cd /storage && find . -type f -print0 | sort -z | xargs -0 sha256sum | sed "s/^/file|/"'
}

file_mode() {
    if stat -c %a "$1" >/dev/null 2>&1; then
        stat -c %a "$1"
    else
        stat -f %Lp "$1"
    fi
}

latest_file() {
    ls -1t "$1" 2>/dev/null | grep -E "$2" | head -n 1
}

echo "Test project: $TEST_PROJECT"
printf 'OPENAI_API_KEY=%s\n' "$SECRET_SENTINEL" >"$INSTALL_DIR/.env"
write_config 8
seed_storage
"${compose[@]}" up -d postgres >/dev/null
wait_for_postgres
seed_database
expected=$(snapshot_state)
[ -n "$expected" ] || fail "seed produced no state"

# 1. Backup
output=$(backup_tool backup 2>&1) || {
    echo "$output" >&2
    fail "backup failed"
}
if printf '%s' "$output" | grep -q "$SECRET_SENTINEL"; then
    fail "backup output printed a secret"
fi
backup_file=$(printf '%s\n' "$output" | tail -n 1)
[ -f "$backup_file" ] || fail "backup did not print its file path: $output"
case "$(basename "$backup_file")" in
    morphik-????????T??????Z.backup) ;;
    *) fail "unexpected backup name $backup_file" ;;
esac
[ "$(file_mode "$backup_file")" = "600" ] || fail "backup file mode is $(file_mode "$backup_file"), expected 600"
[ "$(file_mode "$INSTALL_DIR/backups")" = "700" ] || fail "backup directory mode is not 700"
members=$(tar -tf "$backup_file" | tr '\n' ' ')
[ "$members" = "manifest.json database.dump storage.tar config/morphik.toml " ] || fail "unexpected members: $members"
tar -xOf "$backup_file" manifest.json >"$INSTALL_DIR/manifest.json"
grep -q '"documents" : 3' "$INSTALL_DIR/manifest.json" || fail "manifest document count"
grep -q '"processing" : 1' "$INSTALL_DIR/manifest.json" || fail "manifest status counts"
grep -q '"vector_dimensions" : 8' "$INSTALL_DIR/manifest.json" || fail "manifest vector dimensions"
grep -q '"model_name" : "test/fake-embedding-8"' "$INSTALL_DIR/manifest.json" || fail "manifest embedding model"
grep -q '"multivector_chunks" : 1' "$INSTALL_DIR/manifest.json" || fail "manifest multivector count"
grep -q '"storage_files" : 3' "$INSTALL_DIR/manifest.json" || fail "manifest storage file count"
grep -q '"includes_env" : false' "$INSTALL_DIR/manifest.json" || fail "manifest includes_env"
rm -f "$INSTALL_DIR/manifest.json"
ls "$INSTALL_DIR/backups" | grep -q '^\.work-' && fail "backup left a work directory behind"
pass "backup wrote $(basename "$backup_file") with mode 600 and a complete manifest"

env_output=$(backup_tool backup --include-env --no-upload 2>&1) || fail "backup --include-env failed"
env_file=$(printf '%s\n' "$env_output" | tail -n 1)
grep -qx 'config/.env' <<<"$(tar -tf "$env_file")" || fail "--include-env did not add config/.env"
printf '%s' "$env_output" | grep -q "$SECRET_SENTINEL" && fail "backup --include-env printed a secret"
rm -f "$env_file"
pass "--include-env stores .env without printing it"

# 2. Verify, and detect damage
backup_tool verify "$backup_file" >/dev/null 2>&1 || fail "verify failed on a good backup"
pass "verify restored the backup into a temporary server and matched the manifest"

damaged="$INSTALL_DIR/damaged.backup"
cp "$backup_file" "$damaged"
size=$(wc -c <"$damaged" | tr -d ' ')
printf 'X' | dd of="$damaged" bs=1 seek=$((size / 2)) conv=notrunc 2>/dev/null
if backup_tool verify "$damaged" >"$INSTALL_DIR/damaged.log" 2>&1; then
    fail "verify accepted a damaged backup"
fi
grep -q 'Checksum mismatch' "$INSTALL_DIR/damaged.log" || fail "damaged backup did not report a checksum mismatch"
if backup_tool restore "$damaged" --force --no-start >/dev/null 2>&1; then
    fail "restore accepted a damaged backup"
fi
rm -f "$damaged" "$INSTALL_DIR/damaged.log"
[ "$(snapshot_state)" = "$expected" ] || fail "a refused restore changed data"
pass "verify and restore reject a damaged backup"

# 3. Delete the volume completely, then restore into the empty deployment
"${compose[@]}" --profile "*" down --volumes --remove-orphans >/dev/null
docker volume inspect "$TEST_VOLUME" >/dev/null 2>&1 && fail "PostgreSQL volume still exists"
docker run --rm -v "$INSTALL_DIR/storage:/storage" --entrypoint sh "$TOOL_IMAGE" -c 'rm -rf /storage/* /storage/.[!.]*'

write_config 16
if backup_tool restore "$backup_file" --no-start >"$INSTALL_DIR/compat.log" 2>&1; then
    fail "restore ignored an embedding dimension mismatch"
fi
grep -q 'embedding dimensions differ' "$INSTALL_DIR/compat.log" || fail "dimension mismatch message"
rm -f "$INSTALL_DIR/compat.log"
write_config 8
pass "restore refuses a backup whose embedding dimensions differ from morphik.toml"

restore_log="$INSTALL_DIR/restore.log"
backup_tool restore "$backup_file" --no-start >"$restore_log" 2>&1 || {
    cat "$restore_log" >&2
    fail "restore into an empty deployment failed"
}
actual=$(snapshot_state)
if [ "$actual" != "$expected" ]; then
    diff <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
    fail "restored state differs from the original"
fi
processing=$(psql_q -c "SELECT system_metadata->>'status', system_metadata->>'ingestion_revision', system_metadata ? 'progress' FROM documents WHERE external_id = 'doc-processing'")
[ "$processing" = "failed|3|f" ] || fail "mid-ingestion document after restore: $processing"
requeue_file=$(ls "$INSTALL_DIR"/backups/restore-*-requeue.json 2>/dev/null | head -n 1)
[ -n "$requeue_file" ] && grep -q '"external_id" : "doc-processing"' "$requeue_file" || fail "requeue file"
grep -q 'ingest/requeue' "$restore_log" || fail "restore did not explain how to requeue"
pass "restore after volume deletion brought back documents, metadata, embeddings, and files"
pass "the mid-ingestion document is failed with revision 3 and listed for requeue"

# 4. Existing data: refuse without --force, replace with --force
psql_q -c "INSERT INTO documents (external_id, filename, system_metadata) VALUES ('doc-new', 'new.txt', '{\"status\":\"completed\"}')" >/dev/null
printf 'new file\n' >"$INSTALL_DIR/storage/new.txt"
before_force=$(snapshot_state)
if backup_tool restore "$backup_file" --no-start >"$INSTALL_DIR/force.log" 2>&1; then
    fail "restore replaced existing data without --force"
fi
grep -q 'already has data' "$INSTALL_DIR/force.log" || fail "missing existing-data message"
[ "$(snapshot_state)" = "$before_force" ] || fail "refused restore changed data"
pass "restore refuses to overwrite existing data without --force"

backup_tool restore "$backup_file" --force --no-start >"$INSTALL_DIR/force.log" 2>&1 || {
    cat "$INSTALL_DIR/force.log" >&2
    fail "restore --force failed"
}
safety=$(latest_file "$INSTALL_DIR/backups" '^morphik-.*-pre-restore\.backup$')
[ -n "$safety" ] || fail "restore --force did not write a safety backup"
grep -q '"documents" : 4' <<<"$(tar -xOf "$INSTALL_DIR/backups/$safety" manifest.json)" || fail "safety backup is missing the new document"
[ "$(snapshot_state)" = "$expected" ] || fail "restore --force state differs from the backup"
pass "restore --force wrote $safety and replaced the data with the backup"

# 5. Scheduled backups through the Compose service, with retention and verification
write_config 8 true
mkdir -p "$INSTALL_DIR/backups"
export MORPHIK_BACKUP_DIR="$INSTALL_DIR/backups"
export MORPHIK_BACKUP_MIN_INTERVAL_SECONDS=5
"${compose[@]}" --profile backup up -d backup >/dev/null
deadline=$((SECONDS + 240))
while :; do
    auto_count=$(ls "$INSTALL_DIR/backups" | grep -cE '^morphik-[0-9]{8}T[0-9]{6}Z-auto\.backup$' || true)
    retained=$("${compose[@]}" logs backup 2>/dev/null | grep -c 'Retention removed' || true)
    if [ "$retained" -ge 1 ] && [ "$auto_count" -eq 2 ]; then
        break
    fi
    restarts=$(docker inspect -f '{{.RestartCount}}' "$("${compose[@]}" ps -aq backup)" 2>/dev/null || echo 0)
    if [ "$SECONDS" -gt "$deadline" ] || [ "${restarts:-0}" -gt 0 ]; then
        "${compose[@]}" logs backup >&2 || true
        fail "scheduled backups did not reach retention (count $auto_count, restarts $restarts)"
    fi
    sleep 3
done
backup_logs=$("${compose[@]}" logs backup 2>&1)
grep -q 'verified morphik-' <<<"$backup_logs" || fail "scheduled backup was not verified"
newest_auto=$(latest_file "$INSTALL_DIR/backups" '^morphik-.*-auto\.backup$')
backup_tool verify "$INSTALL_DIR/backups/$newest_auto" >/dev/null 2>&1 || fail "scheduled backup does not verify"
"${compose[@]}" --profile backup stop backup >/dev/null
pass "the backup service wrote verified backups and kept the newest 2"

# 6. Off-host copy to an S3-compatible endpoint
if [ "$RUN_S3" = "1" ]; then
    docker run -d --name "$S3_CONTAINER" --network "$TEST_NETWORK" --network-alias s3mock \
        -e COM_ADOBE_TESTING_S3MOCK_STORE_INITIAL_BUCKETS=morphik-backups -e initialBuckets=morphik-backups "$S3_IMAGE" >/dev/null
    export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1
    export AWS_ENDPOINT_URL=http://s3mock:9090
    export MORPHIK_BACKUP_S3_NETWORK="$TEST_NETWORK"
    export MORPHIK_BACKUP_SYNC_SECONDS=3
    write_config 8 true "s3://morphik-backups/iqor"
    s3_ls() {
        docker run --rm --network "$TEST_NETWORK" -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
            -e AWS_DEFAULT_REGION -e AWS_ENDPOINT_URL amazon/aws-cli:latest s3 ls s3://morphik-backups/iqor/ 2>/dev/null || true
    }
    for _ in $(seq 1 30); do
        docker run --rm --network "$TEST_NETWORK" -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
            -e AWS_DEFAULT_REGION -e AWS_ENDPOINT_URL amazon/aws-cli:latest s3 ls s3://morphik-backups >/dev/null 2>&1 && break
        sleep 2
    done
    upload_output=$(backup_tool backup 2>&1) || {
        echo "$upload_output" >&2
        fail "backup with s3_uri failed"
    }
    manual_name=$(basename "$(printf '%s\n' "$upload_output" | tail -n 1)")
    grep -q "$manual_name" <<<"$(s3_ls)" || fail "manual backup was not uploaded"
    pass "backup uploaded $manual_name to the S3 endpoint"

    "${compose[@]}" --profile backup-s3 up -d backup-s3 >/dev/null
    deadline=$((SECONDS + 120))
    while ! grep -q "$newest_auto" <<<"$(s3_ls)"; do
        if [ "$SECONDS" -gt "$deadline" ]; then
            "${compose[@]}" logs backup-s3 >&2 || true
            fail "backup-s3 did not copy $newest_auto"
        fi
        sleep 3
    done
    pass "the backup-s3 service copied scheduled backups off the host"
fi

echo "PASS: backup, verify, restore after volume deletion, restore --force, scheduled backups, and off-host copies work."

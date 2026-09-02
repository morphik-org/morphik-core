#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
COMPOSE_FILE="$REPO_DIR/docker-compose.run.yml"
TEST_PROJECT="${MORPHIK_PERSISTENCE_TEST_PROJECT:-morphik-persistence-test-$$}"

case "$TEST_PROJECT" in
    morphik-persistence-test-*) ;;
    *)
        echo "Test project name must start with morphik-persistence-test-." >&2
        exit 2
        ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "Docker is installed, but the daemon is not running." >&2
    exit 1
fi

compose=(docker compose --project-name "$TEST_PROJECT" --project-directory "$REPO_DIR" -f "$COMPOSE_FILE")

cleanup() {
    # This project name is unique to the test. Removing its volumes cannot touch a deployment.
    "${compose[@]}" --profile "*" down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wait_for_postgres() {
    local attempts=30
    while (( attempts > 0 )); do
        if "${compose[@]}" exec -T postgres pg_isready -U morphik -d morphik >/dev/null 2>&1; then
            return 0
        fi
        attempts=$((attempts - 1))
        sleep 2
    done
    echo "PostgreSQL did not become ready." >&2
    "${compose[@]}" logs postgres >&2 || true
    return 1
}

cleanup
"${compose[@]}" up -d postgres
wait_for_postgres

"${compose[@]}" exec -T postgres psql -v ON_ERROR_STOP=1 -U morphik -d morphik <<'SQL'
CREATE TABLE documents (
    external_id TEXT PRIMARY KEY,
    content_type TEXT,
    filename TEXT,
    doc_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_types JSONB NOT NULL DEFAULT '{}'::jsonb,
    storage_info JSONB NOT NULL DEFAULT '{}'::jsonb,
    system_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    additional_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    owner_id TEXT,
    app_id TEXT,
    folder_name TEXT,
    folder_path TEXT,
    folder_id TEXT,
    end_user_id TEXT
);
INSERT INTO documents (
    external_id,
    content_type,
    filename,
    doc_metadata,
    metadata_types,
    system_metadata,
    owner_id,
    app_id
) VALUES (
    'iqor-persistence-probe',
    'text/plain',
    '47490.txt',
    '{"project":"QA","work_item_type":"Product Backlog Item","work_item_id":"47490"}'::jsonb,
    '{"project":"string","work_item_type":"string","work_item_id":"string"}'::jsonb,
    '{"status":"completed"}'::jsonb,
    'iqor-test',
    'iqor-test'
);
SQL

volume_before=$("${compose[@]}" config --volumes | awk '$0 == "postgres_data" { print; exit }')
test "$volume_before" = "postgres_data"

# Recreate only the PostgreSQL container. Docker Compose must reattach the same named volume.
"${compose[@]}" rm --stop --force postgres
"${compose[@]}" up -d postgres
wait_for_postgres

persisted=$("${compose[@]}" exec -T postgres psql -At -v ON_ERROR_STOP=1 -U morphik -d morphik -c \
    "SELECT external_id || '|' || doc_metadata->>'project' || '|' || doc_metadata->>'work_item_id' FROM documents WHERE external_id = 'iqor-persistence-probe';")

if [ "$persisted" != "iqor-persistence-probe|QA|47490" ]; then
    echo "Document record did not survive PostgreSQL container recreation: $persisted" >&2
    exit 1
fi

echo "PASS: document identity and metadata survived PostgreSQL container recreation."

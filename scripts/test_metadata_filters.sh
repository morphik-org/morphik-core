#!/usr/bin/env bash

set -eo pipefail

BASE_URL="${MORPHIK_API_BASE:-http://localhost:8001}"
TOKEN="${MORPHIK_TOKEN:-}"
if [[ -n "${TOKEN}" ]]; then
  AUTH_HEADER=("-H" "Authorization: Bearer ${TOKEN}")
else
  AUTH_HEADER=()
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd python3

random_string() {
  python3 - <<'PY'
import secrets, string
alphabet = string.ascii_letters + string.digits
print(''.join(secrets.choice(alphabet) for _ in range(8)))
PY
}

DOC_NAME="typed-metadata-test-$(random_string)"
echo "Creating test document '${DOC_NAME}' via /ingest/text ..."

if [[ ${#AUTH_HEADER[@]} -gt 0 ]]; then
  INGEST_RESPONSE=$(curl -s -S -f \
    -X POST "${BASE_URL}/ingest/text" \
    "${AUTH_HEADER[@]}" \
    -H "Content-Type: application/json" \
    -d @- <<JSON
{
  "content": "Metadata filter smoke test ${DOC_NAME}",
  "filename": "${DOC_NAME}.txt",
  "metadata": {
    "start_date": "2024-01-15T12:30:00Z",
    "end_date": "2024-12-31",
    "priority": 42,
    "cost": "1234.56"
  },
  "metadata_types": {
    "start_date": "datetime",
    "end_date": "date",
    "priority": "number",
    "cost": "decimal"
  }
}
JSON
)
else
  INGEST_RESPONSE=$(curl -s -S -f \
    -X POST "${BASE_URL}/ingest/text" \
    -H "Content-Type: application/json" \
    -d @- <<JSON
{
  "content": "Metadata filter smoke test ${DOC_NAME}",
  "filename": "${DOC_NAME}.txt",
  "metadata": {
    "start_date": "2024-01-15T12:30:00Z",
    "end_date": "2024-12-31",
    "priority": 42,
    "cost": "1234.56"
  },
  "metadata_types": {
    "start_date": "datetime",
    "end_date": "date",
    "priority": "number",
    "cost": "decimal"
  }
}
JSON
)
fi

DOCUMENT_ID=$(echo "${INGEST_RESPONSE}" | python3 -c 'import json, sys; data = json.load(sys.stdin); print(data["external_id"])')

if [[ -z "${DOCUMENT_ID}" ]]; then
  echo "Failed to capture document ID from ingest response" >&2
  exit 1
fi

echo "Inserted document id=${DOCUMENT_ID}"

echo "Querying /documents/list_docs to verify typed comparison operators ..."

LIST_RESPONSE=$(curl -s -S -f \
  -X POST "${BASE_URL}/documents/list_docs" \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "document_filters": {
    "\$and": [
      {"priority": {"\$gt": 40}},
      {"cost": {"\$lte": "1234.56"}},
      {"start_date": {"\$gte": "2024-01-01T00:00:00Z"}},
      {"end_date": {"\$type": "date"}}
    ]
  },
  "limit": 5,
  "include_total_count": true
}
JSON
)

MATCHES=$(echo "${LIST_RESPONSE}" | python3 -c 'import json, sys; payload = json.load(sys.stdin); docs = payload.get("documents") or []; print(next((doc["external_id"] for doc in docs if doc["external_id"]), ""))')

if [[ "${MATCHES}" != "${DOCUMENT_ID}" ]]; then
  echo "Typed metadata filters failed; expected ${DOCUMENT_ID}, got '${MATCHES}'" >&2
  echo "Full response: ${LIST_RESPONSE}" >&2
  curl -s -S -X DELETE "${BASE_URL}/documents/${DOCUMENT_ID}" \
    "${AUTH_HEADER[@]}" >/dev/null || true
  exit 1
fi

echo "Typed metadata filters succeeded."

echo "Cleaning up test document ..."
curl -s -S -f \
  -X DELETE "${BASE_URL}/documents/${DOCUMENT_ID}" \
  "${AUTH_HEADER[@]}" >/dev/null

echo "All typed metadata curl tests passed."

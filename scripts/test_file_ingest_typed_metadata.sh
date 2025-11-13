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

RANDOM_ID=$(random_string)
TEST_FILENAME="typed-metadata-file-test-${RANDOM_ID}.txt"
TEST_FILE="/tmp/${TEST_FILENAME}"

echo "Creating temporary test file: ${TEST_FILE}"
cat > "${TEST_FILE}" <<'TESTCONTENT'
This is a test document for typed metadata file ingestion.
It contains information about a project with specific dates, costs, and priorities.

Project: Alpha Initiative
Start Date: January 15, 2024
End Date: December 31, 2024
Priority: 42
Budget: $1234.56
TESTCONTENT

echo "Uploading file via /ingest/file with typed metadata..."

if [[ ${#AUTH_HEADER[@]} -gt 0 ]]; then
  INGEST_RESPONSE=$(curl -s -S -f \
    -X POST "${BASE_URL}/ingest/file" \
    "${AUTH_HEADER[@]}" \
    -F "file=@${TEST_FILE}" \
    -F "metadata={\"start_date\":\"2024-01-15T12:30:00Z\",\"end_date\":\"2024-12-31\",\"priority\":42,\"cost\":\"1234.56\",\"project_name\":\"Alpha Initiative\"}" \
    -F "metadata_types={\"start_date\":\"datetime\",\"end_date\":\"date\",\"priority\":\"number\",\"cost\":\"decimal\",\"project_name\":\"string\"}")
else
  INGEST_RESPONSE=$(curl -s -S -f \
    -X POST "${BASE_URL}/ingest/file" \
    -F "file=@${TEST_FILE}" \
    -F "metadata={\"start_date\":\"2024-01-15T12:30:00Z\",\"end_date\":\"2024-12-31\",\"priority\":42,\"cost\":\"1234.56\",\"project_name\":\"Alpha Initiative\"}" \
    -F "metadata_types={\"start_date\":\"datetime\",\"end_date\":\"date\",\"priority\":\"number\",\"cost\":\"decimal\",\"project_name\":\"string\"}")
fi

echo "Ingest response: ${INGEST_RESPONSE}"

DOCUMENT_ID=$(echo "${INGEST_RESPONSE}" | python3 -c 'import json, sys; data = json.load(sys.stdin); print(data.get("external_id") or "")')
JOB_ID=$(echo "${INGEST_RESPONSE}" | python3 -c 'import json, sys; data = json.load(sys.stdin); print(data.get("job_id") or "")')

if [[ -z "${DOCUMENT_ID}" ]]; then
  echo "Failed to capture document ID from ingest response" >&2
  rm -f "${TEST_FILE}"
  exit 1
fi

echo "Document ID: ${DOCUMENT_ID}"

if [[ -n "${JOB_ID}" ]]; then
  echo "Job ID: ${JOB_ID}"
  echo "Waiting for ingestion to complete..."

  MAX_ATTEMPTS=60
  ATTEMPT=0
  JOB_STATUS=""

  while [[ ${ATTEMPT} -lt ${MAX_ATTEMPTS} ]]; do
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))

    if [[ ${#AUTH_HEADER[@]} -gt 0 ]]; then
      JOB_RESPONSE=$(curl -s -S -f \
        -X GET "${BASE_URL}/ingest/status/${JOB_ID}" \
        "${AUTH_HEADER[@]}" 2>/dev/null || echo '{}')
    else
      JOB_RESPONSE=$(curl -s -S -f \
        -X GET "${BASE_URL}/ingest/status/${JOB_ID}" 2>/dev/null || echo '{}')
    fi

    JOB_STATUS=$(echo "${JOB_RESPONSE}" | python3 -c 'import json, sys; data = json.load(sys.stdin); print(data.get("status") or "")')

    echo "  Attempt ${ATTEMPT}/${MAX_ATTEMPTS}: Job status = ${JOB_STATUS}"

    if [[ "${JOB_STATUS}" == "completed" ]]; then
      echo "Ingestion completed successfully!"
      break
    elif [[ "${JOB_STATUS}" == "failed" ]]; then
      echo "Ingestion job failed!" >&2
      echo "Job response: ${JOB_RESPONSE}" >&2
      rm -f "${TEST_FILE}"
      exit 1
    fi
  done

  if [[ "${JOB_STATUS}" != "completed" ]]; then
    echo "Timeout waiting for ingestion to complete (status: ${JOB_STATUS})" >&2
    rm -f "${TEST_FILE}"
    exit 1
  fi
else
  echo "No job_id returned, assuming synchronous ingestion"
  sleep 3
fi

echo ""
echo "Querying /documents/list_docs with typed metadata filters..."

LIST_RESPONSE=$(curl -s -S -f \
  -X POST "${BASE_URL}/documents/list_docs" \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "document_filters": {
    "\$and": [
      {"priority": {"\$eq": 42}},
      {"cost": {"\$lte": "1500.00"}},
      {"start_date": {"\$gte": "2024-01-01T00:00:00Z"}},
      {"end_date": {"\$lt": "2025-01-01"}},
      {"project_name": {"\$eq": "Alpha Initiative"}}
    ]
  },
  "limit": 10,
  "include_total_count": true
}
JSON
)

echo "List response: ${LIST_RESPONSE}"

MATCHED_DOC=$(echo "${LIST_RESPONSE}" | python3 -c 'import json, sys; payload = json.load(sys.stdin); docs = payload.get("documents") or []; print(next((doc["external_id"] for doc in docs if doc["external_id"] == "'${DOCUMENT_ID}'"), ""))')

if [[ "${MATCHED_DOC}" != "${DOCUMENT_ID}" ]]; then
  echo "Typed metadata filters failed for file upload!" >&2
  echo "Expected document ${DOCUMENT_ID}, got '${MATCHED_DOC}'" >&2
  echo "Full response: ${LIST_RESPONSE}" >&2

  echo "Attempting cleanup..."
  curl -s -S -X DELETE "${BASE_URL}/documents/${DOCUMENT_ID}" \
    "${AUTH_HEADER[@]}" >/dev/null || true
  rm -f "${TEST_FILE}"
  exit 1
fi

echo ""
echo "✓ Typed metadata filters succeeded for file upload!"

echo ""
echo "Testing individual operator queries..."

# Test $gt operator
echo "  Testing priority \$gt 40..."
GT_RESPONSE=$(curl -s -S -f \
  -X POST "${BASE_URL}/documents/list_docs" \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"document_filters\":{\"priority\":{\"\$gt\":40}},\"limit\":10}")

GT_MATCH=$(echo "${GT_RESPONSE}" | python3 -c 'import json, sys; docs = json.load(sys.stdin).get("documents", []); print(any(doc["external_id"] == "'${DOCUMENT_ID}'" for doc in docs))')
if [[ "${GT_MATCH}" != "True" ]]; then
  echo "  ✗ \$gt test failed" >&2
else
  echo "  ✓ \$gt test passed"
fi

# Test $type operator
echo "  Testing end_date \$type date..."
TYPE_RESPONSE=$(curl -s -S -f \
  -X POST "${BASE_URL}/documents/list_docs" \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"document_filters\":{\"end_date\":{\"\$type\":\"date\"}},\"limit\":10}")

TYPE_MATCH=$(echo "${TYPE_RESPONSE}" | python3 -c 'import json, sys; docs = json.load(sys.stdin).get("documents", []); print(any(doc["external_id"] == "'${DOCUMENT_ID}'" for doc in docs))')
if [[ "${TYPE_MATCH}" != "True" ]]; then
  echo "  ✗ \$type test failed" >&2
else
  echo "  ✓ \$type test passed"
fi

echo ""
echo "Cleaning up test document and file..."
curl -s -S -f \
  -X DELETE "${BASE_URL}/documents/${DOCUMENT_ID}" \
  "${AUTH_HEADER[@]}" >/dev/null

rm -f "${TEST_FILE}"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✓ All typed metadata file ingestion tests passed!"
echo "═══════════════════════════════════════════════════════"

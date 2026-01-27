#!/bin/bash
# V2 API Sanity Tests
# Tests v2 ingestion/retrieval for all supported formats + tenant isolation
#
# Usage: JWT_SECRET_KEY=... ./scripts/v2_api_sanity.sh
#        Or with bypass_auth_mode=true, no env var needed

set -euo pipefail

BASE_URL="${MORPHIK_URL:-http://localhost:8000}"
JWT_SECRET_KEY="${JWT_SECRET_KEY:-}"
TEST_RUN_ID="v2test_$(date +%s)"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; TESTS_PASSED=$((TESTS_PASSED + 1)); }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; TESTS_FAILED=$((TESTS_FAILED + 1)); }
log_section() { echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"; echo -e "${YELLOW}  $1${NC}"; echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"; }

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Auth state for two apps
APP_A_TOKEN=""
APP_A_ID=""
APP_A_NAME=""
APP_B_TOKEN=""
APP_B_ID=""
APP_B_NAME=""
BYPASS_MODE=false

# ============================================================================
# Auth Setup
# ============================================================================

check_bypass_mode() {
    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -d '{"skip":0,"limit":1}' 2>/dev/null || echo "000")

    if [[ "$status_code" != "401" ]]; then
        BYPASS_MODE=true
        return 0
    fi
    return 1
}

create_app_token() {
    local app_name="$1"
    local user_id="$2"
    local org_id="$3"

    # Generate bootstrap token with matching user_id
    local bootstrap_token
    bootstrap_token=$(python3 - <<PYEOF 2>/dev/null
import jwt
import time
payload = {
    "sub": "$user_id",
    "user_id": "$user_id",
    "entity_id": "$user_id",
    "type": "developer",
    "exp": int(time.time()) + 3600,
}
print(jwt.encode(payload, "$JWT_SECRET_KEY", algorithm="HS256"))
PYEOF
    )

    if [[ -z "$bootstrap_token" ]]; then
        log_error "Failed to generate bootstrap token for $app_name"
        return 1
    fi

    # Create app via /cloud/generate_uri - requires org_id for fresh tokens
    local response
    response=$(curl -sf -X POST "$BASE_URL/cloud/generate_uri" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $bootstrap_token" \
        -d "{\"name\":\"$app_name\",\"org_id\":\"$org_id\"}" 2>&1) || {
        log_error "Failed to create app $app_name: $response"
        return 1
    }

    local app_id uri token
    app_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('app_id',''))" 2>/dev/null)
    uri=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uri',''))" 2>/dev/null)
    token=$(echo "$uri" | sed -n 's/morphik:\/\/[^:]*:\([^@]*\)@.*/\1/p')

    if [[ -z "$app_id" || -z "$token" ]]; then
        log_error "Failed to extract app_id/token for $app_name"
        return 1
    fi

    echo "$app_id|$token"
}

setup_auth() {
    log_section "Setting Up Authentication"

    if check_bypass_mode; then
        log_info "Auth bypass mode detected - no tokens needed"
        log_info "Tenant isolation tests will be skipped in bypass mode"
        return 0
    fi

    if [[ -z "$JWT_SECRET_KEY" ]]; then
        log_error "Auth required but JWT_SECRET_KEY not set"
        log_info "Set JWT_SECRET_KEY environment variable or enable bypass_auth_mode in morphik.toml"
        exit 1
    fi

    # Create App A (with its own org)
    APP_A_NAME="${TEST_RUN_ID}_app_a"
    local user_a_id org_a_id
    user_a_id=$(python3 -c "import uuid; print(uuid.uuid4())")
    org_a_id=$(python3 -c "import uuid; print(uuid.uuid4())")
    local result_a
    result_a=$(create_app_token "$APP_A_NAME" "$user_a_id" "$org_a_id") || exit 1
    APP_A_ID=$(echo "$result_a" | cut -d'|' -f1)
    APP_A_TOKEN=$(echo "$result_a" | cut -d'|' -f2)
    log_success "Created App A: $APP_A_ID"

    # Create App B (with its own org - ensures complete isolation)
    APP_B_NAME="${TEST_RUN_ID}_app_b"
    local user_b_id org_b_id
    user_b_id=$(python3 -c "import uuid; print(uuid.uuid4())")
    org_b_id=$(python3 -c "import uuid; print(uuid.uuid4())")
    local result_b
    result_b=$(create_app_token "$APP_B_NAME" "$user_b_id" "$org_b_id") || exit 1
    APP_B_ID=$(echo "$result_b" | cut -d'|' -f1)
    APP_B_TOKEN=$(echo "$result_b" | cut -d'|' -f2)
    log_success "Created App B: $APP_B_ID"
}

cleanup_apps() {
    if [[ -n "$APP_A_TOKEN" && -n "$APP_A_NAME" ]]; then
        curl -sf -X DELETE "$BASE_URL/apps?app_name=${APP_A_NAME}" \
            -H "Authorization: Bearer $APP_A_TOKEN" > /dev/null 2>&1 || true
    fi
    if [[ -n "$APP_B_TOKEN" && -n "$APP_B_NAME" ]]; then
        curl -sf -X DELETE "$BASE_URL/apps?app_name=${APP_B_NAME}" \
            -H "Authorization: Bearer $APP_B_TOKEN" > /dev/null 2>&1 || true
    fi
}

get_auth_opts() {
    local token="$1"
    if [[ -n "$token" ]]; then
        echo "-H" "Authorization: Bearer $token"
    fi
}

# ============================================================================
# Test File Creation
# ============================================================================

create_test_files() {
    log_section "Creating Test Files"

    # TXT file
    cat > "$TMP_DIR/test.txt" << 'EOF'
V2 Sanity Test - Plain Text Document

This is a plain text file for testing v2 ingestion.
Keywords: v2_txt_test alpha bravo charlie
EOF
    log_info "Created test.txt"

    # MD file
    cat > "$TMP_DIR/test.md" << 'EOF'
# V2 Sanity Test - Markdown Document

This is a **markdown** file for testing v2 ingestion.

## Section One
Keywords: v2_md_test delta echo foxtrot

## Section Two
- List item one
- List item two
EOF
    log_info "Created test.md"

    # Simple DOCX (minimal valid docx structure)
    # We'll use a real test file if available, otherwise skip
    if [[ -f "core/tests/integration/test_data/test.docx" ]]; then
        cp "core/tests/integration/test_data/test.docx" "$TMP_DIR/test.docx"
        log_info "Copied test.docx"
    else
        log_info "No test.docx available - will skip DOCX test"
    fi

    # PPTX
    if [[ -f "core/tests/integration/test_data/test.pptx" ]]; then
        cp "core/tests/integration/test_data/test.pptx" "$TMP_DIR/test.pptx"
        log_info "Copied test.pptx"
    else
        log_info "No test.pptx available - will skip PPTX test"
    fi

    # PDF
    if [[ -f "core/tests/integration/test_data/test.pdf" ]]; then
        cp "core/tests/integration/test_data/test.pdf" "$TMP_DIR/test.pdf"
        log_info "Copied test.pdf"
    else
        log_info "No test.pdf available - will skip PDF test"
    fi
}

# ============================================================================
# Ingestion Helper
# ============================================================================

ingest_file() {
    local token="$1"
    local file_path="$2"
    local folder_path="$3"
    local metadata="$4"

    local response
    if [[ -n "$token" ]]; then
        response=$(curl -sS -X POST "$BASE_URL/v2/documents" \
            -H "Authorization: Bearer $token" \
            -F "file=@${file_path}" \
            -F "folder_path=${folder_path}" \
            -F "metadata=${metadata}")
    else
        response=$(curl -sS -X POST "$BASE_URL/v2/documents" \
            -F "file=@${file_path}" \
            -F "folder_path=${folder_path}" \
            -F "metadata=${metadata}")
    fi

    local doc_id
    doc_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('document_id',''))" 2>/dev/null || echo "")

    if [[ -z "$doc_id" ]]; then
        echo "ERROR:$response"
    else
        echo "$doc_id"
    fi
}

ingest_content() {
    local token="$1"
    local content="$2"
    local filename="$3"
    local folder_path="$4"
    local metadata="$5"

    local response
    if [[ -n "$token" ]]; then
        response=$(curl -sS -X POST "$BASE_URL/v2/documents" \
            -H "Authorization: Bearer $token" \
            -F "content=${content}" \
            -F "filename=${filename}" \
            -F "folder_path=${folder_path}" \
            -F "metadata=${metadata}")
    else
        response=$(curl -sS -X POST "$BASE_URL/v2/documents" \
            -F "content=${content}" \
            -F "filename=${filename}" \
            -F "folder_path=${folder_path}" \
            -F "metadata=${metadata}")
    fi

    local doc_id
    doc_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('document_id',''))" 2>/dev/null || echo "")

    if [[ -z "$doc_id" ]]; then
        echo "ERROR:$response"
    else
        echo "$doc_id"
    fi
}

# ============================================================================
# Status Helpers
# ============================================================================

wait_for_completion() {
    local token="$1"
    local doc_id="$2"
    local elapsed=0
    local status=""

    while [[ "$elapsed" -lt "$WAIT_TIMEOUT" ]]; do
        local response
        if [[ -n "$token" ]]; then
            response=$(curl -sS -X GET "$BASE_URL/documents/${doc_id}/status" \
                -H "Authorization: Bearer $token" || true)
        else
            response=$(curl -sS -X GET "$BASE_URL/documents/${doc_id}/status" || true)
        fi

        local status
        status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")

        if [[ "$status" == "completed" ]]; then
            return 0
        fi
        if [[ "$status" == "failed" ]]; then
            local err
            err=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error',''))" 2>/dev/null || echo "")
            log_error "Ingestion failed for $doc_id: ${err:-$response}"
            return 1
        fi

        sleep "$POLL_INTERVAL"
        elapsed=$((elapsed + POLL_INTERVAL))
    done

    log_error "Timed out waiting for document $doc_id to complete (last status: $status)"
    return 1
}

# ============================================================================
# Retrieval Helper
# ============================================================================

retrieve_chunks() {
    local token="$1"
    local query="$2"
    local doc_id="$3"
    local top_k="${4:-5}"

    local payload
    payload=$(python3 - <<PYEOF
import json
payload = {
    "query": "$query",
    "filters": {"document_ids": ["$doc_id"]},
    "top_k": $top_k
}
print(json.dumps(payload))
PYEOF
    )

    local response
    if [[ -n "$token" ]]; then
        response=$(curl -sS -X POST "$BASE_URL/v2/retrieve/chunks" \
            -H "Authorization: Bearer $token" \
            -H "Content-Type: application/json" \
            -d "$payload")
    else
        response=$(curl -sS -X POST "$BASE_URL/v2/retrieve/chunks" \
            -H "Content-Type: application/json" \
            -d "$payload")
    fi

    echo "$response"
}

count_chunks() {
    local response="$1"
    echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('chunks',[])))" 2>/dev/null || echo "0"
}

# ============================================================================
# Format Tests
# ============================================================================

test_txt_format() {
    log_section "Testing TXT Format"

    local doc_id
    doc_id=$(ingest_content "$APP_A_TOKEN" \
        "V2 TXT test content. Keywords: txt_unique_keyword_xyz" \
        "v2_test.txt" \
        "/v2/txt" \
        '{"format":"txt","test":"v2_sanity"}')

    if [[ "$doc_id" == ERROR:* ]]; then
        log_error "TXT ingest failed: ${doc_id#ERROR:}"
        return 1
    fi
    log_success "TXT ingest ok (doc_id=$doc_id)"
    if ! wait_for_completion "$APP_A_TOKEN" "$doc_id"; then
        return 1
    fi

    local response chunk_count
    response=$(retrieve_chunks "$APP_A_TOKEN" "txt_unique_keyword_xyz" "$doc_id")
    chunk_count=$(count_chunks "$response")

    if [[ "$chunk_count" -ge 1 ]]; then
        log_success "TXT retrieve ok (chunks=$chunk_count)"
    else
        log_error "TXT retrieve failed: $response"
        return 1
    fi

    echo "$doc_id"
}

test_md_format() {
    log_section "Testing Markdown Format"

    local md_content
    md_content=$(cat "$TMP_DIR/test.md")

    local doc_id
    doc_id=$(ingest_content "$APP_A_TOKEN" \
        "$md_content" \
        "v2_test.md" \
        "/v2/md" \
        '{"format":"md","test":"v2_sanity"}')

    if [[ "$doc_id" == ERROR:* ]]; then
        log_error "MD ingest failed: ${doc_id#ERROR:}"
        return 1
    fi
    log_success "MD ingest ok (doc_id=$doc_id)"
    if ! wait_for_completion "$APP_A_TOKEN" "$doc_id"; then
        return 1
    fi

    local response chunk_count
    response=$(retrieve_chunks "$APP_A_TOKEN" "v2_md_test delta" "$doc_id")
    chunk_count=$(count_chunks "$response")

    if [[ "$chunk_count" -ge 1 ]]; then
        log_success "MD retrieve ok (chunks=$chunk_count)"
    else
        log_error "MD retrieve failed: $response"
        return 1
    fi

    echo "$doc_id"
}

test_pdf_format() {
    log_section "Testing PDF Format"

    if [[ ! -f "$TMP_DIR/test.pdf" ]]; then
        log_info "Skipping PDF test - no test file available"
        return 0
    fi

    local doc_id
    doc_id=$(ingest_file "$APP_A_TOKEN" "$TMP_DIR/test.pdf" "/v2/pdf" '{"format":"pdf","test":"v2_sanity"}')

    if [[ "$doc_id" == ERROR:* ]]; then
        log_error "PDF ingest failed: ${doc_id#ERROR:}"
        return 1
    fi
    log_success "PDF ingest ok (doc_id=$doc_id)"
    if ! wait_for_completion "$APP_A_TOKEN" "$doc_id"; then
        return 1
    fi

    local response chunk_count
    response=$(retrieve_chunks "$APP_A_TOKEN" "test document" "$doc_id")
    chunk_count=$(count_chunks "$response")

    if [[ "$chunk_count" -ge 1 ]]; then
        log_success "PDF retrieve ok (chunks=$chunk_count)"
        # Verify XML structure with loc attributes
        local has_loc
        has_loc=$(echo "$response" | python3 -c "
import sys,json
data = json.load(sys.stdin)
chunks = data.get('chunks', [])
for c in chunks:
    if 'loc=' in c.get('content',''):
        print('yes')
        break
else:
    print('no')
" 2>/dev/null || echo "no")
        if [[ "$has_loc" == "yes" ]]; then
            log_success "PDF chunks contain bbox (loc=) attributes"
        else
            log_info "PDF chunks may not have bbox - check content"
        fi
    else
        log_error "PDF retrieve failed: $response"
        return 1
    fi

    echo "$doc_id"
}

test_docx_format() {
    log_section "Testing DOCX Format"

    if [[ ! -f "$TMP_DIR/test.docx" ]]; then
        log_info "Skipping DOCX test - no test file available"
        return 0
    fi

    local doc_id
    doc_id=$(ingest_file "$APP_A_TOKEN" "$TMP_DIR/test.docx" "/v2/docx" '{"format":"docx","test":"v2_sanity"}')

    if [[ "$doc_id" == ERROR:* ]]; then
        log_error "DOCX ingest failed: ${doc_id#ERROR:}"
        return 1
    fi
    log_success "DOCX ingest ok (doc_id=$doc_id)"
    if ! wait_for_completion "$APP_A_TOKEN" "$doc_id"; then
        return 1
    fi

    local response chunk_count
    response=$(retrieve_chunks "$APP_A_TOKEN" "document" "$doc_id")
    chunk_count=$(count_chunks "$response")

    if [[ "$chunk_count" -ge 1 ]]; then
        log_success "DOCX retrieve ok (chunks=$chunk_count)"
    else
        log_error "DOCX retrieve failed: $response"
        return 1
    fi

    echo "$doc_id"
}

test_pptx_format() {
    log_section "Testing PPTX Format"

    if [[ ! -f "$TMP_DIR/test.pptx" ]]; then
        log_info "Skipping PPTX test - no test file available"
        return 0
    fi

    local doc_id
    doc_id=$(ingest_file "$APP_A_TOKEN" "$TMP_DIR/test.pptx" "/v2/pptx" '{"format":"pptx","test":"v2_sanity"}')

    if [[ "$doc_id" == ERROR:* ]]; then
        log_error "PPTX ingest failed: ${doc_id#ERROR:}"
        return 1
    fi
    log_success "PPTX ingest ok (doc_id=$doc_id)"
    if ! wait_for_completion "$APP_A_TOKEN" "$doc_id"; then
        return 1
    fi

    local response chunk_count
    response=$(retrieve_chunks "$APP_A_TOKEN" "presentation slide" "$doc_id")
    chunk_count=$(count_chunks "$response")

    if [[ "$chunk_count" -ge 1 ]]; then
        log_success "PPTX retrieve ok (chunks=$chunk_count)"
    else
        log_error "PPTX retrieve failed: $response"
        return 1
    fi

    echo "$doc_id"
}

# ============================================================================
# Tenant Isolation Test
# ============================================================================

test_tenant_isolation() {
    log_section "Testing Tenant Isolation (Cross-App Security)"

    if [[ "$BYPASS_MODE" == "true" ]]; then
        log_info "Skipping tenant isolation test - bypass mode enabled"
        return 0
    fi

    # Ingest document as App A
    local app_a_doc_id
    app_a_doc_id=$(ingest_content "$APP_A_TOKEN" \
        "Secret document for App A only. Keywords: app_a_secret_data_xyz" \
        "app_a_secret.txt" \
        "/v2/app_a" \
        '{"owner":"app_a","secret":"true"}')

    if [[ "$app_a_doc_id" == ERROR:* ]]; then
        log_error "App A ingest failed: ${app_a_doc_id#ERROR:}"
        return 1
    fi
    log_success "App A document ingested (doc_id=$app_a_doc_id)"
    if ! wait_for_completion "$APP_A_TOKEN" "$app_a_doc_id"; then
        return 1
    fi

    # Ingest document as App B
    local app_b_doc_id
    app_b_doc_id=$(ingest_content "$APP_B_TOKEN" \
        "Secret document for App B only. Keywords: app_b_secret_data_xyz" \
        "app_b_secret.txt" \
        "/v2/app_b" \
        '{"owner":"app_b","secret":"true"}')

    if [[ "$app_b_doc_id" == ERROR:* ]]; then
        log_error "App B ingest failed: ${app_b_doc_id#ERROR:}"
        return 1
    fi
    log_success "App B document ingested (doc_id=$app_b_doc_id)"
    if ! wait_for_completion "$APP_B_TOKEN" "$app_b_doc_id"; then
        return 1
    fi

    # Test 1: App A should find its own document
    local response_a_own chunk_count_a_own
    response_a_own=$(retrieve_chunks "$APP_A_TOKEN" "app_a_secret_data_xyz" "$app_a_doc_id")
    chunk_count_a_own=$(count_chunks "$response_a_own")

    if [[ "$chunk_count_a_own" -ge 1 ]]; then
        log_success "App A can retrieve its own document"
    else
        log_error "App A cannot retrieve its own document!"
        return 1
    fi

    # Test 2: App B should find its own document
    local response_b_own chunk_count_b_own
    response_b_own=$(retrieve_chunks "$APP_B_TOKEN" "app_b_secret_data_xyz" "$app_b_doc_id")
    chunk_count_b_own=$(count_chunks "$response_b_own")

    if [[ "$chunk_count_b_own" -ge 1 ]]; then
        log_success "App B can retrieve its own document"
    else
        log_error "App B cannot retrieve its own document!"
        return 1
    fi

    # Test 3: App A should NOT find App B's document (even with doc_id)
    local response_a_cross chunk_count_a_cross
    response_a_cross=$(retrieve_chunks "$APP_A_TOKEN" "app_b_secret_data_xyz" "$app_b_doc_id")
    chunk_count_a_cross=$(count_chunks "$response_a_cross")

    if [[ "$chunk_count_a_cross" -eq 0 ]]; then
        log_success "App A cannot access App B's document (isolation verified)"
    else
        log_error "SECURITY VIOLATION: App A retrieved App B's document!"
        log_error "Response: $response_a_cross"
        return 1
    fi

    # Test 4: App B should NOT find App A's document
    local response_b_cross chunk_count_b_cross
    response_b_cross=$(retrieve_chunks "$APP_B_TOKEN" "app_a_secret_data_xyz" "$app_a_doc_id")
    chunk_count_b_cross=$(count_chunks "$response_b_cross")

    if [[ "$chunk_count_b_cross" -eq 0 ]]; then
        log_success "App B cannot access App A's document (isolation verified)"
    else
        log_error "SECURITY VIOLATION: App B retrieved App A's document!"
        log_error "Response: $response_b_cross"
        return 1
    fi

    # Test 5: Open query should only return own app's results
    local response_a_open
    response_a_open=$(curl -sS -X POST "$BASE_URL/v2/retrieve/chunks" \
        -H "Authorization: Bearer $APP_A_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"query":"secret_data_xyz","top_k":10}')

    local has_b_content
    has_b_content=$(echo "$response_a_open" | grep -c "app_b_secret" 2>/dev/null || true)
    has_b_content="${has_b_content:-0}"
    # Ensure we have a clean integer
    has_b_content=$(echo "$has_b_content" | tr -d '[:space:]' | head -c 10)
    if [[ -z "$has_b_content" || "$has_b_content" == "0" ]]; then
        log_success "Open query from App A does not leak App B's data"
    else
        log_error "SECURITY VIOLATION: Open query leaked App B's data!"
        return 1
    fi

    log_success "All tenant isolation tests passed"
}

# ============================================================================
# Main
# ============================================================================

main() {
    log_section "V2 API Sanity Tests"
    log_info "Server: $BASE_URL"
    log_info "Test Run: $TEST_RUN_ID"

    # Check server
    if ! curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
        log_error "Server not responding at $BASE_URL"
        exit 1
    fi
    log_success "Server is running"

    # Setup
    setup_auth
    create_test_files

    # Format tests
    test_txt_format || true
    test_md_format || true
    test_pdf_format || true
    test_docx_format || true
    test_pptx_format || true

    # Security tests
    test_tenant_isolation || true

    # Cleanup
    cleanup_apps

    # Summary
    log_section "Test Summary"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"

    if [[ "$TESTS_FAILED" -gt 0 ]]; then
        exit 1
    fi

    log_success "All V2 API sanity tests completed successfully"
}

main "$@"

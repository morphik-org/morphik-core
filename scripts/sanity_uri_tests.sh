#!/bin/bash
# URI-related tests for the sanity suite. Source from sanity_test.sh.

test_app_token_revocation() {
    log_section "Testing App Token Revocation"

    local auth_check_code
    auth_check_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -d "{\"skip\":0,\"limit\":1}" || echo "000")
    if [[ "$auth_check_code" != "401" ]]; then
        log_warn "Auth bypass detected (status $auth_check_code); skipping token revocation test"
        return
    fi

    if [[ -z "$ADMIN_SERVICE_SECRET" && -z "$JWT_SECRET_KEY" ]]; then
        log_warn "Skipping token revocation test (set ADMIN_SERVICE_SECRET or JWT_SECRET_KEY)"
        return
    fi

    local app_name="sanity_auth_${TEST_RUN_ID}"
    local user_id
    user_id=$(python3 -c "import uuid; print(uuid.uuid4())" 2>/dev/null)
    if [[ -z "$user_id" ]]; then
        log_error "Failed to generate user_id for token revocation test"
        return
    fi

    local response
    if [[ -n "$ADMIN_SERVICE_SECRET" ]]; then
        response=$(curl -sf -X POST "$BASE_URL/cloud/generate_uri" \
            -H "Content-Type: application/json" \
            -H "X-Morphik-Admin-Secret: $ADMIN_SERVICE_SECRET" \
            -d "{\"name\":\"$app_name\",\"user_id\":\"$user_id\"}" 2>&1) || {
            log_error "App creation (admin secret) failed: $response"
            return
        }
    else
        local bootstrap_token
        bootstrap_token=$(python3 - <<PYEOF 2>/dev/null || true
import jwt
import time
payload = {
    "user_id": "$user_id",
    "entity_id": "$user_id",
    "exp": int(time.time()) + 3600,
}
print(jwt.encode(payload, "$JWT_SECRET_KEY", algorithm="HS256"))
PYEOF
        )
        if [[ -z "$bootstrap_token" ]]; then
            log_warn "Skipping token revocation test (failed to generate bootstrap token)"
            return
        fi
        response=$(curl -sf -X POST "$BASE_URL/cloud/generate_uri" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $bootstrap_token" \
            -d "{\"name\":\"$app_name\",\"user_id\":\"$user_id\"}" 2>&1) || {
            log_error "App creation (bootstrap token) failed: $response"
            return
        }
    fi

    local app_id
    app_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('app_id',''))" 2>/dev/null)
    local uri
    uri=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uri',''))" 2>/dev/null)
    if [[ -z "$app_id" || -z "$uri" ]]; then
        log_error "App creation response missing app_id or uri"
        return
    fi

    local token
    token=$(echo "$uri" | sed -n 's/morphik:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    if [[ -z "$token" ]]; then
        log_error "Failed to extract token from uri"
        return
    fi
    log_success "Created app for revocation test: $app_id"

    response=$(curl -sf -X POST "$BASE_URL/ingest/text" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $token" \
        -d "{\"content\":\"Sanity auth revocation test\",\"filename\":\"auth_revocation.txt\",\"metadata\":{\"test_run_id\":\"$TEST_RUN_ID\",\"auth_revocation\":true}}" 2>&1) || {
        log_error "Authorized ingest failed: $response"
        return
    }
    local auth_doc_id
    auth_doc_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('external_id',''))" 2>/dev/null)
    if [[ -n "$auth_doc_id" ]]; then
        log_success "Authorized ingest queued document $auth_doc_id"
    else
        log_warn "Authorized ingest response missing external_id"
    fi

    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $token" \
        -d "{\"query\":\"Sanity auth revocation test\",\"k\":3,\"filters\":{\"test_run_id\":\"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Authorized retrieve failed: $response"
        return
    }
    local chunk_count
    chunk_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || chunk_count=0
    if [[ "$chunk_count" -gt 0 ]]; then
        log_success "Authorized retrieve returned $chunk_count chunk(s)"
    else
        log_warn "Authorized retrieve returned $chunk_count chunk(s)"
    fi

    response=$(curl -sf -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $token" \
        -d "{\"skip\":0,\"limit\":5,\"document_filters\":{\"test_run_id\":\"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Authorized list_docs failed: $response"
        return
    }
    local doc_count
    doc_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('documents',[])))" 2>/dev/null) || doc_count=0
    log_success "Authorized list_docs returned $doc_count document(s)"

    local delete_code
    if [[ -n "$ADMIN_SERVICE_SECRET" ]]; then
        delete_code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/apps?app_name=${app_name}" \
            -H "X-Morphik-Admin-Secret: $ADMIN_SERVICE_SECRET")
    else
        delete_code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/apps?app_name=${app_name}" \
            -H "Authorization: Bearer $token")
    fi
    if [[ "$delete_code" == "200" ]]; then
        log_success "Deleted app for revocation test"
    else
        log_error "Delete app expected 200, got $delete_code"
        return
    fi

    local revoked_code
    revoked_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $token" \
        -d "{\"skip\":0,\"limit\":1}")
    if [[ "$revoked_code" == "401" ]]; then
        log_success "Token rejected after app deletion (HTTP 401)"
    else
        log_error "Expected 401 after app delete, got $revoked_code"
    fi
}

test_app_token_rotation() {
    log_section "Testing App Token Rotation"

    local auth_check_code
    auth_check_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -d "{\"skip\":0,\"limit\":1}" || echo "000")
    if [[ "$auth_check_code" != "401" ]]; then
        log_warn "Auth bypass detected (status $auth_check_code); skipping token rotation test"
        return
    fi

    if [[ -z "$ADMIN_SERVICE_SECRET" && -z "$JWT_SECRET_KEY" ]]; then
        log_warn "Skipping token rotation test (set ADMIN_SERVICE_SECRET or JWT_SECRET_KEY)"
        return
    fi

    local app_name="sanity_rotate_${TEST_RUN_ID}"
    local user_id
    user_id=$(python3 -c "import uuid; print(uuid.uuid4())" 2>/dev/null)
    if [[ -z "$user_id" ]]; then
        log_error "Failed to generate user_id for token rotation test"
        return
    fi

    local response
    if [[ -n "$ADMIN_SERVICE_SECRET" ]]; then
        response=$(curl -sf -X POST "$BASE_URL/cloud/generate_uri" \
            -H "Content-Type: application/json" \
            -H "X-Morphik-Admin-Secret: $ADMIN_SERVICE_SECRET" \
            -d "{\"name\":\"$app_name\",\"user_id\":\"$user_id\"}" 2>&1) || {
            log_error "App creation (admin secret) failed: $response"
            return
        }
    else
        local bootstrap_token
        bootstrap_token=$(python3 - <<PYEOF 2>/dev/null || true
import jwt
import time
payload = {
    "user_id": "$user_id",
    "entity_id": "$user_id",
    "exp": int(time.time()) + 3600,
}
print(jwt.encode(payload, "$JWT_SECRET_KEY", algorithm="HS256"))
PYEOF
        )
        if [[ -z "$bootstrap_token" ]]; then
            log_warn "Skipping token rotation test (failed to generate bootstrap token)"
            return
        fi
        response=$(curl -sf -X POST "$BASE_URL/cloud/generate_uri" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $bootstrap_token" \
            -d "{\"name\":\"$app_name\",\"user_id\":\"$user_id\"}" 2>&1) || {
            log_error "App creation (bootstrap token) failed: $response"
            return
        }
    fi

    local app_id
    app_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('app_id',''))" 2>/dev/null)
    local uri
    uri=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uri',''))" 2>/dev/null)
    if [[ -z "$app_id" || -z "$uri" ]]; then
        log_error "App creation response missing app_id or uri"
        return
    fi

    local token
    token=$(echo "$uri" | sed -n 's/morphik:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    if [[ -z "$token" ]]; then
        log_error "Failed to extract token from uri"
        return
    fi
    log_success "Created app for rotation test: $app_id"

    response=$(curl -sf -X POST "$BASE_URL/ingest/text" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $token" \
        -d "{\"content\":\"Sanity auth rotation test\",\"filename\":\"auth_rotation.txt\",\"metadata\":{\"test_run_id\":\"$TEST_RUN_ID\",\"auth_rotation\":true}}" 2>&1) || {
        log_error "Authorized ingest failed: $response"
        return
    }
    local auth_doc_id
    auth_doc_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('external_id',''))" 2>/dev/null)
    if [[ -n "$auth_doc_id" ]]; then
        log_success "Authorized ingest queued document $auth_doc_id"
    else
        log_warn "Authorized ingest response missing external_id"
    fi

    response=$(curl -sf -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $token" \
        -d "{\"skip\":0,\"limit\":5,\"document_filters\":{\"test_run_id\":\"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Authorized list_docs failed: $response"
        return
    }
    local doc_count
    doc_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('documents',[])))" 2>/dev/null) || doc_count=0
    log_success "Authorized list_docs returned $doc_count document(s)"

    local rotate_response
    if [[ -n "$ADMIN_SERVICE_SECRET" ]]; then
        rotate_response=$(curl -sf -X POST "$BASE_URL/apps/rotate_token?app_id=${app_id}" \
            -H "X-Morphik-Admin-Secret: $ADMIN_SERVICE_SECRET" 2>&1) || {
            log_error "Token rotation (admin secret) failed: $rotate_response"
            return
        }
    else
        rotate_response=$(curl -sf -X POST "$BASE_URL/apps/rotate_token?app_id=${app_id}" \
            -H "Authorization: Bearer $token" 2>&1) || {
            log_error "Token rotation failed: $rotate_response"
            return
        }
    fi

    local new_uri
    new_uri=$(echo "$rotate_response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uri',''))" 2>/dev/null)
    if [[ -z "$new_uri" ]]; then
        log_error "Token rotation response missing uri"
        return
    fi
    local new_token
    new_token=$(echo "$new_uri" | sed -n 's/morphik:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    if [[ -z "$new_token" ]]; then
        log_error "Failed to extract new token from rotated uri"
        return
    fi

    local old_code
    old_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $token" \
        -d "{\"skip\":0,\"limit\":1}")
    if [[ "$old_code" == "401" ]]; then
        log_success "Old token rejected after rotation (HTTP 401)"
    else
        log_error "Expected 401 for old token after rotation, got $old_code"
    fi

    if [[ -n "$auth_doc_id" ]]; then
        local new_code
        new_code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $new_token" \
            "$BASE_URL/documents/$auth_doc_id")
        if [[ "$new_code" == "200" ]]; then
            log_success "New token can access existing document"
        else
            log_error "New token expected 200 for existing document, got $new_code"
        fi
    else
        response=$(curl -sf -X POST "$BASE_URL/documents" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $new_token" \
            -d "{\"skip\":0,\"limit\":5,\"document_filters\":{\"test_run_id\":\"$TEST_RUN_ID\"}}" 2>&1) || {
            log_error "New token list_docs failed: $response"
            return
        }
        doc_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('documents',[])))" 2>/dev/null) || doc_count=0
        if [[ "$doc_count" -gt 0 ]]; then
            log_success "New token list_docs returned $doc_count document(s)"
        else
            log_warn "New token list_docs returned $doc_count document(s)"
        fi
    fi
}

run_uri_tests() {
    test_app_token_revocation
    test_app_token_rotation
}

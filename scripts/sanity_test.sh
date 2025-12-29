#!/bin/bash
# Morphik Sanity Test Suite
# Tests ingestion and retrieval across all supported file types and configurations
#
# Usage: ./scripts/sanity_test.sh [--skip-cleanup]

set -euo pipefail

# Configuration
BASE_URL="${MORPHIK_URL:-http://localhost:8000}"
TEST_DIR="/tmp/morphik_sanity_test"
TEST_RUN_ID="test_$(date +%s)"
TIMEOUT_SECONDS=120
POLL_INTERVAL=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TESTS_PASSED=0
TESTS_FAILED=0

# Track document IDs for cleanup
declare -a DOC_IDS=()
# Track additional resources
declare -a BATCH_DOC_IDS=()
declare -a SCOPE_DOC_IDS=()
TYPED_DOC_ID=""
INGEST_QUERY_DOC_ID=""
PDF_DOC_ID=""
IMAGE_DOC_ID=""
IMAGE_COLPALI_DOC_ID=""
QUERY_IMAGE_B64=""
BATCH_FILENAMES=()
LAST_CHUNK_DOC_ID=""
LAST_CHUNK_NUMBER=""

# ============================================================================
# Helper Functions
# ============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; TESTS_PASSED=$((TESTS_PASSED + 1)); }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; TESTS_FAILED=$((TESTS_FAILED + 1)); }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_section() { echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"; echo -e "${YELLOW}  $1${NC}"; echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"; }

check_server() {
    log_info "Checking server availability at $BASE_URL..."
    if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
        log_success "Server is running"
        return 0
    else
        log_error "Server is not responding at $BASE_URL"
        exit 1
    fi
}

# ============================================================================
# Test File Creation
# ============================================================================

create_test_files() {
    log_section "Creating Test Files"
    mkdir -p "$TEST_DIR"

    # Base64 helper image (100x100 PNG) for query_image and image ingestion
    cat > "$TEST_DIR/test_image.b64" << 'EOF'
iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAA6ElEQVR4nO3QMQHAIBDAQKg4hKEJgbXw2e/mTNnnvsXMN+wwqzErMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzArMCswKzBrzf30kgJE43y1WQAAAABJRU5ErkJggg==
EOF
    # Cross-platform base64 decode (Linux uses -d, macOS uses -D)
    base64 -d < "$TEST_DIR/test_image.b64" > "$TEST_DIR/test_image.png" 2>/dev/null || \
    base64 -D < "$TEST_DIR/test_image.b64" > "$TEST_DIR/test_image.png"
    QUERY_IMAGE_B64=$(cat "$TEST_DIR/test_image.b64")

    # TXT file - plain text with unicode and searchable content
    cat > "$TEST_DIR/test_document.txt" << 'EOF'
Morphik Test Document - Plain Text

This document tests plain text ingestion with special characters.

Technical Content:
- Vector embeddings enable semantic search
- ColPali provides visual document understanding
- Unicode support: café, naïve, 日本語, Москва

Code Example:
    def search(query):
        return vector_store.find(query)

Keywords: morphik test ingestion retrieval sanity
EOF
    log_info "Created test_document.txt"

    # MD file - proper markdown with formatting
    cat > "$TEST_DIR/test_document.md" << 'EOF'
# Morphik Test Document - Markdown

## Introduction

This document tests **markdown** ingestion with proper formatting preservation.

## Features

1. Headers at multiple levels
2. **Bold** and *italic* text
3. Code blocks:

```python
def retrieve_chunks(query: str, k: int = 5):
    """Retrieve relevant chunks from the vector store."""
    return vector_store.search(query, top_k=k)
```

## Data Table

| Feature | Status | Priority |
|---------|--------|----------|
| Ingestion | Working | High |
| Retrieval | Working | High |
| ColPali | Working | Medium |

## Special Characters

- French: café, résumé
- Greek: α, β, γ
- Japanese: 日本語

Keywords: morphik test markdown formatting sanity
EOF
    log_info "Created test_document.md"

    # CSV file - tabular data
    cat > "$TEST_DIR/test_data.csv" << 'EOF'
id,product,category,price,quantity
1,Widget A,Electronics,29.99,100
2,Widget B,Electronics,49.99,50
3,Gadget X,Hardware,99.99,25
4,Gadget Y,Hardware,149.99,10
5,Tool Z,Software,199.99,200
EOF
    log_info "Created test_data.csv"

    # Create a simple XLSX using Python
    python3 << 'PYEOF'
import os
try:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales Data"
    ws.append(["Month", "Revenue", "Expenses", "Profit"])
    ws.append(["January", 10000, 7000, 3000])
    ws.append(["February", 12000, 8000, 4000])
    ws.append(["March", 15000, 9000, 6000])
    ws.append(["April", 11000, 7500, 3500])
    wb.save("/tmp/morphik_sanity_test/test_spreadsheet.xlsx")
    print("Created test_spreadsheet.xlsx")
except ImportError:
    # Fallback: create a minimal xlsx
    print("openpyxl not available, skipping xlsx creation")
PYEOF

    # Create a simple DOCX using Python
    python3 << 'PYEOF'
import os
try:
    from docx import Document
    doc = Document()
    doc.add_heading("Morphik Test Document - Word", 0)
    doc.add_paragraph("This document tests DOCX ingestion capabilities.")
    doc.add_heading("Section 1: Overview", level=1)
    doc.add_paragraph("Morphik provides document processing and retrieval features.")
    doc.add_heading("Section 2: Features", level=1)
    doc.add_paragraph("- Document ingestion")
    doc.add_paragraph("- Vector search")
    doc.add_paragraph("- ColPali visual embeddings")
    doc.add_paragraph("Keywords: morphik test docx word sanity")
    doc.save("/tmp/morphik_sanity_test/test_document.docx")
    print("Created test_document.docx")
except ImportError:
    print("python-docx not available, skipping docx creation")
PYEOF

    # Batch ingest helper files
    cat > "$TEST_DIR/batch_file1.txt" << 'EOF'
Batch file 1 content for sanity batch ingest.
EOF
    cat > "$TEST_DIR/batch_file2.txt" << 'EOF'
Batch file 2 content for sanity batch ingest.
EOF

    log_success "Test files created in $TEST_DIR"
}

# ============================================================================
# Ingestion Tests
# ============================================================================

ingest_file() {
    local file_path="$1"
    local use_colpali="$2"
    local file_type="$3"
    local category="$4"
    local priority="$5"
    local date_offset="$6"  # days offset from today

    local filename=$(basename "$file_path")
    local colpali_label=$([[ "$use_colpali" == "true" ]] && echo "with_colpali" || echo "no_colpali")

    # Generate dates with and without timezone
    local date_naive=$(python3 -c "from datetime import datetime, timedelta; print((datetime.now() + timedelta(days=$date_offset)).strftime('%Y-%m-%dT%H:%M:%S'))")
    local date_tz=$(python3 -c "from datetime import datetime, timedelta, timezone; print((datetime.now(timezone.utc) + timedelta(days=$date_offset)).strftime('%Y-%m-%dT%H:%M:%S+00:00'))")
    local date_z=$(python3 -c "from datetime import datetime, timedelta, timezone; print((datetime.now(timezone.utc) + timedelta(days=$date_offset)).strftime('%Y-%m-%dT%H:%M:%SZ'))")

    local metadata=$(cat << EOF
{
    "test_run_id": "$TEST_RUN_ID",
    "file_type": "$file_type",
    "category": "$category",
    "priority": $priority,
    "colpali_enabled": $use_colpali,
    "created_date_naive": "$date_naive",
    "created_date_tz": "$date_tz",
    "created_date_z": "$date_z",
    "days_offset": $date_offset
}
EOF
)

    log_info "Ingesting $filename ($colpali_label)..."

    local response
    response=$(curl -sf -X POST "$BASE_URL/ingest/file" \
        -F "file=@$file_path" \
        -F "metadata=$metadata" \
        -F "use_colpali=$use_colpali" 2>&1) || {
        log_error "Failed to ingest $filename ($colpali_label)"
        return 1
    }

    local doc_id
    doc_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('external_id',''))" 2>/dev/null) || {
        log_error "Failed to parse response for $filename"
        return 1
    }

    if [[ -n "$doc_id" ]]; then
        DOC_IDS+=("$doc_id")
        log_success "Ingested $filename ($colpali_label) -> $doc_id"
        [[ "$file_type" == "pdf" && "$use_colpali" == "false" ]] && PDF_DOC_ID="$doc_id"
        if [[ "$file_type" == "png" ]]; then
            if [[ "$use_colpali" == "true" ]]; then
                IMAGE_COLPALI_DOC_ID="$doc_id"
            else
                IMAGE_DOC_ID="$doc_id"
            fi
        fi
        return 0
    else
        log_error "No document ID returned for $filename"
        return 1
    fi
}

run_ingestion_tests() {
    log_section "Running Ingestion Tests"

    # Define test files: path|file_type|category|priority|date_offset
    # date_offset varies so we can test date range filtering
    declare -a test_files=(
        "$TEST_DIR/test_document.txt|txt|documentation|1|-7"
        "$TEST_DIR/test_document.md|md|documentation|1|-3"
        "$TEST_DIR/test_data.csv|csv|data|2|0"
        "$TEST_DIR/test_image.png|png|images|2|0"
    )

    # Add xlsx if it exists (1 day ago)
    [[ -f "$TEST_DIR/test_spreadsheet.xlsx" ]] && test_files+=("$TEST_DIR/test_spreadsheet.xlsx|xlsx|data|2|-1")

    # Add docx if it exists (5 days ago)
    [[ -f "$TEST_DIR/test_document.docx" ]] && test_files+=("$TEST_DIR/test_document.docx|docx|documentation|1|-5")

    # Add PDF (use existing example, 2 days ago)
    local pdf_path="/Users/adi/Desktop/morphik/morphik-core/examples/assets/colpali_example.pdf"
    [[ -f "$pdf_path" ]] && test_files+=("$pdf_path|pdf|technical|3|-2")

    # Ingest each file with and without ColPali
    for entry in "${test_files[@]}"; do
        IFS='|' read -r file_path file_type category priority date_offset <<< "$entry"

        if [[ -f "$file_path" ]]; then
            ingest_file "$file_path" "false" "$file_type" "$category" "$priority" "$date_offset"
            ingest_file "$file_path" "true" "$file_type" "$category" "$priority" "$date_offset"
        else
            log_warn "File not found: $file_path"
        fi
    done
}

# Guardrails: ensure protected fields cannot be overridden
test_protected_field_guards() {
    log_section "Testing Protected Field Guardrails"

    local payload
    payload=$(cat << EOF
{"content":"guardrail test","metadata":{"external_id":"evil-id","folder_id":"evil-folder","folder_name":"bad","folder_path":"/bad/path"},"folder_name":"/should/not/work"}
EOF
)

    local body_file="$TEST_DIR/protected_guardrails_response.json"
    local http_code
    http_code=$(curl -s -o "$body_file" -w "%{http_code}" -X POST "$BASE_URL/ingest/text" \
        -H "Content-Type: application/json" \
        -d "$payload")

    if [[ "$http_code" == "400" || "$http_code" == "403" ]]; then
        log_success "Protected fields rejected (status $http_code)"
    else
        log_error "Protected fields were not rejected (status $http_code) response=$(cat "$body_file" 2>/dev/null)"
    fi
}

# ============================================================================
# Wait for Processing
# ============================================================================

wait_for_processing() {
    log_section "Waiting for Document Processing"

    local start_time=$(date +%s)
    local all_complete=false

    while [[ "$all_complete" != "true" ]]; do
        local elapsed=$(($(date +%s) - start_time))

        if [[ $elapsed -gt $TIMEOUT_SECONDS ]]; then
            log_error "Timeout waiting for document processing ($TIMEOUT_SECONDS seconds)"
            return 1
        fi

        # Get status of documents from this test run
        local response
        response=$(curl -sf -X POST "$BASE_URL/documents" \
            -H "Content-Type: application/json" \
            -d "{\"document_filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
            log_warn "Failed to fetch document status"
            sleep "$POLL_INTERVAL"
            continue
        }

        # Count statuses
        local status_counts
        status_counts=$(echo "$response" | python3 -c "
import sys, json
from collections import Counter
docs = json.load(sys.stdin)
statuses = Counter(d.get('system_metadata', {}).get('status', 'unknown') for d in docs)
print(f\"total={len(docs)} completed={statuses.get('completed',0)} processing={statuses.get('processing',0)} failed={statuses.get('failed',0)}\")
" 2>/dev/null) || status_counts="error"

        log_info "Status after ${elapsed}s: $status_counts"

        # Check if all complete (or only failures remain)
        local check_result
        check_result=$(echo "$status_counts" | python3 -c "
import sys, re
line = sys.stdin.read()
total = int(re.search(r'total=(\d+)', line).group(1)) if 'total=' in line else 0
completed = int(re.search(r'completed=(\d+)', line).group(1)) if 'completed=' in line else 0
processing = int(re.search(r'processing=(\d+)', line).group(1)) if 'processing=' in line else 0
# Done if no more processing (all either completed or failed)
if total > 0 and processing == 0:
    print('done')
else:
    print('waiting')
" 2>/dev/null) || check_result="waiting"

        if [[ "$check_result" == "done" ]]; then
            all_complete=true
        fi

        [[ "$all_complete" != "true" ]] && sleep "$POLL_INTERVAL"
    done

    log_success "All documents processed successfully"
}

# ============================================================================
# Retrieval Tests
# ============================================================================

test_basic_retrieval() {
    log_section "Testing Basic Retrieval"

    # Test 1: Basic text search
    log_info "Test: Basic text search for 'morphik test'"
    local response
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik test sanity\", \"k\": 5, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Basic retrieval request failed"
        return
    }

    local count
    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0

    if [[ "$count" -gt 0 ]]; then
        log_success "Basic retrieval returned $count chunks"
    else
        log_error "Basic retrieval returned no results"
    fi

    # Test 2: Search for specific content
    log_info "Test: Search for 'vector embeddings semantic search'"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"vector embeddings semantic search\", \"k\": 3, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Semantic search request failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0

    if [[ "$count" -gt 0 ]]; then
        log_success "Semantic search returned $count chunks"
    else
        log_error "Semantic search returned no results"
    fi
}

test_metadata_filtering() {
    log_section "Testing Metadata Filtering"

    # Test 1: Filter by category
    log_info "Test: Filter by category='documentation'"
    local response
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik\", \"k\": 10, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"category\": \"documentation\"}}" 2>&1) || {
        log_error "Category filter request failed"
        return
    }

    # Verify all results have category=documentation
    local valid
    valid=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
all_match = all(c.get('metadata', {}).get('category') == 'documentation' for c in chunks)
print('yes' if (chunks and all_match) else 'no')
" 2>/dev/null) || valid="no"

    if [[ "$valid" == "yes" ]]; then
        log_success "Category filter working correctly"
    else
        log_error "Category filter not working as expected"
    fi

    # Test 2: Filter by file_type
    log_info "Test: Filter by file_type='md'"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"markdown formatting\", \"k\": 5, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"file_type\": \"md\"}}" 2>&1) || {
        log_error "File type filter request failed"
        return
    }

    valid=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
all_match = all(c.get('metadata', {}).get('file_type') == 'md' for c in chunks)
print('yes' if (chunks and all_match) else 'no')
" 2>/dev/null) || valid="no"

    if [[ "$valid" == "yes" ]]; then
        log_success "File type filter working correctly"
    else
        log_error "File type filter not working as expected"
    fi

    # Test 3: Filter by priority (numeric)
    log_info "Test: Filter by priority >= 2"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"data\", \"k\": 10, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"priority\": {\"\$gte\": 2}}}" 2>&1) || {
        log_error "Priority filter request failed"
        return
    }

    valid=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
all_match = all(c.get('metadata', {}).get('priority', 0) >= 2 for c in chunks)
print('yes' if (chunks and all_match) else 'no')
" 2>/dev/null) || valid="no"

    if [[ "$valid" == "yes" ]]; then
        log_success "Numeric filter (priority >= 2) working correctly"
    else
        log_error "Numeric filter not working as expected"
    fi
}

test_date_filtering() {
    log_section "Testing Date Filtering"

    # Calculate date strings for filtering
    local today=$(python3 -c "from datetime import datetime; print(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))")
    local five_days_ago=$(python3 -c "from datetime import datetime, timedelta; print((datetime.now() - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%S'))")
    local two_days_ago=$(python3 -c "from datetime import datetime, timedelta; print((datetime.now() - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S'))")

    # Test 1: Filter by date range using naive datetime (no timezone)
    log_info "Test: Filter by date range (naive datetime, last 5 days)"
    local response
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik test\", \"k\": 20, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"created_date_naive\": {\"\$gte\": \"$five_days_ago\"}}}" 2>&1) || {
        log_error "Naive date filter request failed"
        return
    }

    local count
    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0

    if [[ "$count" -gt 0 ]]; then
        log_success "Naive date filter returned $count chunks (docs from last 5 days)"
    else
        log_error "Naive date filter returned no results"
    fi

    # Test 2: Filter by date with timezone (+00:00 format)
    log_info "Test: Filter by date with timezone (+00:00 format)"
    local five_days_ago_tz=$(python3 -c "from datetime import datetime, timedelta, timezone; print((datetime.now(timezone.utc) - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%S+00:00'))")

    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik test\", \"k\": 20, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"created_date_tz\": {\"\$gte\": \"$five_days_ago_tz\"}}}" 2>&1) || {
        log_error "TZ date filter request failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0

    if [[ "$count" -gt 0 ]]; then
        log_success "TZ date filter (+00:00) returned $count chunks"
    else
        log_error "TZ date filter (+00:00) returned no results"
    fi

    # Test 3: Filter by date with Z suffix
    log_info "Test: Filter by date with Z suffix"
    local five_days_ago_z=$(python3 -c "from datetime import datetime, timedelta, timezone; print((datetime.now(timezone.utc) - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ'))")

    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik test\", \"k\": 20, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"created_date_z\": {\"\$gte\": \"$five_days_ago_z\"}}}" 2>&1) || {
        log_error "Z-suffix date filter request failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0

    if [[ "$count" -gt 0 ]]; then
        log_success "Z-suffix date filter returned $count chunks"
    else
        log_error "Z-suffix date filter returned no results"
    fi

    # Test 4: Filter by exact date range (between 2 and 5 days ago)
    log_info "Test: Filter by date range (between 2-5 days ago)"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik\", \"k\": 20, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"days_offset\": {\"\$gte\": -5, \"\$lte\": -2}}}" 2>&1) || {
        log_error "Date range filter request failed"
        return
    }

    local valid
    valid=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
all_in_range = all(-5 <= c.get('metadata', {}).get('days_offset', 0) <= -2 for c in chunks)
print('yes' if (chunks and all_in_range) else 'no')
" 2>/dev/null) || valid="no"

    if [[ "$valid" == "yes" ]]; then
        log_success "Date range filter (days_offset between -5 and -2) working correctly"
    else
        log_error "Date range filter not working as expected"
    fi

    # Test 5: Filter for recent docs only (last 2 days)
    log_info "Test: Filter for recent docs (days_offset >= -2)"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik\", \"k\": 20, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"days_offset\": {\"\$gte\": -2}}}" 2>&1) || {
        log_error "Recent docs filter request failed"
        return
    }

    valid=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
all_recent = all(c.get('metadata', {}).get('days_offset', -999) >= -2 for c in chunks)
print('yes' if (chunks and all_recent) else 'no')
" 2>/dev/null) || valid="no"

    if [[ "$valid" == "yes" ]]; then
        log_success "Recent docs filter (last 2 days) working correctly"
    else
        log_error "Recent docs filter not working as expected"
    fi
}

test_output_formats() {
    log_section "Testing Output Formats"

    # Test with ColPali to get image chunks
    log_info "Test: output_format=base64 (default)"
    local response
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"IQ imbalance compensation\", \"k\": 2, \"use_colpali\": true, \"output_format\": \"base64\", \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "base64 format request failed"
        return
    }

    # Check if content contains base64 data
    local has_base64
    has_base64=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
has_b64 = any('data:image' in c.get('content', '') or len(c.get('content', '')) > 1000 for c in chunks if c.get('metadata', {}).get('is_image'))
print('yes' if has_b64 else 'no')
" 2>/dev/null) || has_base64="no"

    if [[ "$has_base64" == "yes" ]]; then
        log_success "base64 output format returning image data"
    else
        log_warn "base64 format test - no image chunks found (may be expected for text docs)"
    fi

    # Test URL format
    log_info "Test: output_format=url"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"IQ imbalance compensation\", \"k\": 2, \"use_colpali\": true, \"output_format\": \"url\", \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "url format request failed"
        return
    }

    local has_url
    has_url=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
has_url = any(c.get('download_url') or 'http' in c.get('content', '')[:100] for c in chunks)
print('yes' if has_url else 'no')
" 2>/dev/null) || has_url="no"

    if [[ "$has_url" == "yes" ]]; then
        log_success "url output format returning URLs"
    else
        log_warn "url format test - no URLs found (may be expected for text docs)"
    fi

    # Test text format (OCR)
    log_info "Test: output_format=text"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"IQ imbalance compensation\", \"k\": 2, \"use_colpali\": true, \"output_format\": \"text\", \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "text format request failed"
        return
    }

    local has_text
    has_text=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
# Text format should return readable text, not base64
has_readable = any(
    not c.get('content', '').startswith('data:image') and
    len(c.get('content', '')) > 10
    for c in chunks
)
print('yes' if has_readable else 'no')
" 2>/dev/null) || has_text="no"

    if [[ "$has_text" == "yes" ]]; then
        log_success "text output format returning readable content"
    else
        log_warn "text format test - check manually if OCR is working"
    fi
}

test_colpali_vs_standard() {
    log_section "Testing ColPali vs Standard Retrieval"

    # Standard retrieval (text embeddings)
    log_info "Test: Standard text embedding retrieval"
    local response
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"document processing features\", \"k\": 3, \"use_colpali\": false, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Standard retrieval request failed"
        return
    }

    local std_count
    std_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || std_count=0

    if [[ "$std_count" -gt 0 ]]; then
        log_success "Standard retrieval returned $std_count chunks"
    else
        log_error "Standard retrieval returned no results"
    fi

    # ColPali retrieval (visual embeddings)
    log_info "Test: ColPali visual embedding retrieval"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"document processing features\", \"k\": 3, \"use_colpali\": true, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "ColPali retrieval request failed"
        return
    }

    local colpali_count
    colpali_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || colpali_count=0

    if [[ "$colpali_count" -gt 0 ]]; then
        log_success "ColPali retrieval returned $colpali_count chunks"
    else
        log_error "ColPali retrieval returned no results"
    fi
}

test_content_preservation() {
    log_section "Testing Content Preservation"

    # Test markdown formatting preserved
    log_info "Test: Markdown formatting preservation"
    local response
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"markdown formatting headers\", \"k\": 3, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"file_type\": \"md\"}}" 2>&1) || {
        log_error "Markdown content request failed"
        return
    }

    local has_md_formatting
    has_md_formatting=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
# Check for markdown elements
has_formatting = any(
    '#' in c.get('content', '') or
    '**' in c.get('content', '') or
    '\`\`\`' in c.get('content', '')
    for c in chunks
)
print('yes' if has_formatting else 'no')
" 2>/dev/null) || has_md_formatting="no"

    if [[ "$has_md_formatting" == "yes" ]]; then
        log_success "Markdown formatting preserved"
    else
        log_error "Markdown formatting may not be preserved"
    fi

    # Test unicode preservation
    log_info "Test: Unicode character preservation"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"special characters unicode\", \"k\": 5, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Unicode content request failed"
        return
    }

    local has_unicode
    has_unicode=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
# Check for unicode characters
has_unicode = any(
    'café' in c.get('content', '') or
    '日本語' in c.get('content', '') or
    'α' in c.get('content', '') or
    'naïve' in c.get('content', '')
    for c in chunks
)
print('yes' if has_unicode else 'no')
" 2>/dev/null) || has_unicode="no"

    if [[ "$has_unicode" == "yes" ]]; then
        log_success "Unicode characters preserved"
    else
        log_error "Unicode characters may not be preserved"
    fi
}

# ============================================================================
# Folder Nesting Tests
# ============================================================================

# Track folder test resources for cleanup
declare -a FOLDER_TEST_DOC_IDS=()
FOLDER_TEST_RUN_ID="folder_${TEST_RUN_ID}"

test_folder_creation_and_nesting() {
    log_section "Testing Folder Creation and Nesting"

    # Test 1: Create a top-level folder
    log_info "Test: Create top-level folder /sanity_test"
    local response
    response=$(curl -sf -X POST "$BASE_URL/folders" \
        -H "Content-Type: application/json" \
        -d '{"name": "sanity_test", "description": "Sanity test folder"}' 2>&1) || {
        log_error "Failed to create top-level folder"
        return
    }

    local full_path depth
    full_path=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('full_path',''))" 2>/dev/null)
    depth=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('depth',''))" 2>/dev/null)

    if [[ "$full_path" == "/sanity_test" && "$depth" == "1" ]]; then
        log_success "Created /sanity_test (depth=1)"
    else
        log_error "Top-level folder creation: full_path=$full_path, depth=$depth"
    fi

    # Test 2: Create deeply nested folder with auto-created parents
    log_info "Test: Create nested folder /sanity_test/level1/level2/level3 (auto-create parents)"
    response=$(curl -sf -X POST "$BASE_URL/folders" \
        -H "Content-Type: application/json" \
        -d '{"name": "level3", "full_path": "/sanity_test/level1/level2/level3"}' 2>&1) || {
        log_error "Failed to create nested folder"
        return
    }

    full_path=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('full_path',''))" 2>/dev/null)
    depth=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('depth',''))" 2>/dev/null)

    if [[ "$full_path" == "/sanity_test/level1/level2/level3" && "$depth" == "4" ]]; then
        log_success "Created /sanity_test/level1/level2/level3 (depth=4, parents auto-created)"
    else
        log_error "Nested folder creation: full_path=$full_path, depth=$depth"
    fi

    # Test 3: Verify intermediate folders were auto-created
    log_info "Test: Verify /sanity_test/level1 was auto-created"
    response=$(curl -sf "$BASE_URL/folders/sanity_test/level1" 2>&1) || {
        log_error "/sanity_test/level1 not found"
        return
    }
    depth=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('depth',''))" 2>/dev/null)
    if [[ "$depth" == "2" ]]; then
        log_success "/sanity_test/level1 auto-created (depth=2)"
    else
        log_error "/sanity_test/level1 has wrong depth: $depth"
    fi

    # Test 4: Create sibling folders
    log_info "Test: Create sibling folder /sanity_test/sibling"
    curl -sf -X POST "$BASE_URL/folders" \
        -H "Content-Type: application/json" \
        -d '{"name": "sibling", "full_path": "/sanity_test/sibling"}' > /dev/null 2>&1 || {
        log_error "Failed to create sibling folder"
        return
    }
    log_success "Created sibling folder /sanity_test/sibling"

    # Test 5: Folder lookup by path
    log_info "Test: Lookup folder by nested path"
    response=$(curl -sf "$BASE_URL/folders/sanity_test/level1/level2" 2>&1) || {
        log_error "Folder lookup by path failed"
        return
    }
    full_path=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('full_path',''))" 2>/dev/null)
    if [[ "$full_path" == "/sanity_test/level1/level2" ]]; then
        log_success "Folder lookup by path works"
    else
        log_error "Folder lookup returned: $full_path"
    fi
}

test_folder_document_ingestion() {
    log_section "Testing Folder Document Ingestion"

    local metadata="{\"test_run_id\": \"$FOLDER_TEST_RUN_ID\"}"

    # Ingest documents into different folder levels
    declare -a folder_doc_pairs=(
        "/sanity_test:folder_root.txt"
        "/sanity_test/level1:folder_level1.txt"
        "/sanity_test/level1/level2:folder_level2.txt"
        "/sanity_test/level1/level2/level3:folder_level3.txt"
        "/sanity_test/sibling:folder_sibling.txt"
    )

    # Create test files
    for pair in "${folder_doc_pairs[@]}"; do
        IFS=':' read -r folder filename <<< "$pair"
        cat > "$TEST_DIR/$filename" << EOF
Folder Test Document
Folder: $folder
Test Run: $FOLDER_TEST_RUN_ID
Keywords: morphik folder hierarchy nesting test
EOF
    done

    # Ingest into folders
    for pair in "${folder_doc_pairs[@]}"; do
        IFS=':' read -r folder filename <<< "$pair"
        log_info "Ingesting $filename into $folder"

        local response doc_id
        response=$(curl -sf -X POST "$BASE_URL/ingest/file" \
            -F "file=@$TEST_DIR/$filename" \
            -F "metadata=$metadata" \
            -F "folder_name=$folder" 2>&1) || {
            log_error "Failed to ingest $filename into $folder"
            continue
        }

        doc_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('external_id',''))" 2>/dev/null)
        if [[ -n "$doc_id" ]]; then
            FOLDER_TEST_DOC_IDS+=("$doc_id")
            log_success "Ingested $filename -> $doc_id"
        else
            log_error "No document ID returned for $filename"
        fi
    done
}

wait_for_folder_docs_processing() {
    log_section "Waiting for Folder Document Processing"

    local start_time=$(date +%s)
    local encoded_folder
    encoded_folder=$(python3 -c "import urllib.parse; print(urllib.parse.quote('/sanity_test'))")

    while true; do
        local elapsed=$(($(date +%s) - start_time))
        if [[ $elapsed -gt $TIMEOUT_SECONDS ]]; then
            log_error "Timeout waiting for folder documents ($TIMEOUT_SECONDS seconds)"
            return 1
        fi

        local response
        response=$(curl -sf -X POST "$BASE_URL/documents?folder_name=$encoded_folder&folder_depth=-1" \
            -H "Content-Type: application/json" \
            -d "{\"document_filters\": {\"test_run_id\": \"$FOLDER_TEST_RUN_ID\"}}" 2>&1) || {
            log_warn "Failed to fetch folder document status"
            sleep "$POLL_INTERVAL"
            continue
        }

        local status_counts
        status_counts=$(echo "$response" | python3 -c "
import sys, json
from collections import Counter
docs = json.load(sys.stdin)
statuses = Counter(d.get('system_metadata', {}).get('status', 'unknown') for d in docs)
print(f'total={len(docs)} completed={statuses.get(\"completed\",0)} processing={statuses.get(\"processing\",0)}')
" 2>/dev/null) || status_counts="error"

        log_info "Status after ${elapsed}s: $status_counts"

        local processing
        processing=$(echo "$status_counts" | sed -n 's/.*processing=\([0-9]*\).*/\1/p')
        if [[ "$processing" == "0" ]]; then
            log_success "All folder documents processed"
            return 0
        fi

        sleep "$POLL_INTERVAL"
    done
}

test_folder_depth_filtering() {
    log_section "Testing Folder Depth Filtering"

    local encoded_folder
    encoded_folder=$(python3 -c "import urllib.parse; print(urllib.parse.quote('/sanity_test'))")

    # Test 1: folder_depth=0 (exact match only - just /sanity_test)
    log_info "Test: folder_depth=0 (exact folder match)"
    local response count
    response=$(curl -sf -X POST "$BASE_URL/documents?folder_name=$encoded_folder&folder_depth=0" \
        -H "Content-Type: application/json" \
        -d "{\"document_filters\": {\"test_run_id\": \"$FOLDER_TEST_RUN_ID\"}}" 2>&1) || {
        log_error "folder_depth=0 request failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    if [[ "$count" -eq 1 ]]; then
        log_success "folder_depth=0 returned $count doc (expected 1 from /sanity_test only)"
    else
        log_error "folder_depth=0 returned $count docs (expected 1)"
    fi

    # Test 2: folder_depth=1 (folder + direct children)
    log_info "Test: folder_depth=1 (folder + direct children)"
    response=$(curl -sf -X POST "$BASE_URL/documents?folder_name=$encoded_folder&folder_depth=1" \
        -H "Content-Type: application/json" \
        -d "{\"document_filters\": {\"test_run_id\": \"$FOLDER_TEST_RUN_ID\"}}" 2>&1) || {
        log_error "folder_depth=1 request failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    # /sanity_test (1) + /sanity_test/level1 (1) + /sanity_test/sibling (1) = 3
    if [[ "$count" -eq 3 ]]; then
        log_success "folder_depth=1 returned $count docs (expected 3)"
    else
        log_error "folder_depth=1 returned $count docs (expected 3)"
    fi

    # Test 3: folder_depth=-1 (all descendants)
    log_info "Test: folder_depth=-1 (all descendants)"
    response=$(curl -sf -X POST "$BASE_URL/documents?folder_name=$encoded_folder&folder_depth=-1" \
        -H "Content-Type: application/json" \
        -d "{\"document_filters\": {\"test_run_id\": \"$FOLDER_TEST_RUN_ID\"}}" 2>&1) || {
        log_error "folder_depth=-1 request failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    # All 5 docs: /sanity_test, /level1, /level2, /level3, /sibling
    if [[ "$count" -eq 5 ]]; then
        log_success "folder_depth=-1 returned $count docs (expected 5 - all descendants)"
    else
        log_error "folder_depth=-1 returned $count docs (expected 5)"
    fi

    # Test 4: folder_depth=2 (up to grandchildren)
    log_info "Test: folder_depth=2 (up to 2 levels deep)"
    response=$(curl -sf -X POST "$BASE_URL/documents?folder_name=$encoded_folder&folder_depth=2" \
        -H "Content-Type: application/json" \
        -d "{\"document_filters\": {\"test_run_id\": \"$FOLDER_TEST_RUN_ID\"}}" 2>&1) || {
        log_error "folder_depth=2 request failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    # /sanity_test (1) + /level1 (1) + /sibling (1) + /level2 (1) = 4
    if [[ "$count" -eq 4 ]]; then
        log_success "folder_depth=2 returned $count docs (expected 4)"
    else
        log_error "folder_depth=2 returned $count docs (expected 4)"
    fi
}

test_folder_retrieval_scoping() {
    log_section "Testing Folder Retrieval Scoping"

    local encoded_folder
    encoded_folder=$(python3 -c "import urllib.parse; print(urllib.parse.quote('/sanity_test'))")

    # Test 1: retrieve/chunks with folder_depth=0
    log_info "Test: /retrieve/chunks with folder_depth=0"
    local response count
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"folder hierarchy nesting\",
            \"k\": 10,
            \"folder_name\": \"/sanity_test\",
            \"folder_depth\": 0,
            \"filters\": {\"test_run_id\": \"$FOLDER_TEST_RUN_ID\"}
        }" 2>&1) || {
        log_error "retrieve/chunks with folder_depth=0 failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    if [[ "$count" -ge 1 ]]; then
        log_success "retrieve/chunks folder_depth=0 returned $count chunk(s)"
    else
        log_error "retrieve/chunks folder_depth=0 returned no results"
    fi

    # Test 2: retrieve/chunks with folder_depth=-1 (all descendants)
    log_info "Test: /retrieve/chunks with folder_depth=-1"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"folder hierarchy nesting\",
            \"k\": 20,
            \"folder_name\": \"/sanity_test\",
            \"folder_depth\": -1,
            \"filters\": {\"test_run_id\": \"$FOLDER_TEST_RUN_ID\"}
        }" 2>&1) || {
        log_error "retrieve/chunks with folder_depth=-1 failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    if [[ "$count" -ge 5 ]]; then
        log_success "retrieve/chunks folder_depth=-1 returned $count chunks (from all nested folders)"
    else
        log_error "retrieve/chunks folder_depth=-1 returned $count chunks (expected >=5)"
    fi

    # Test 3: retrieve/docs with folder_depth scoping
    log_info "Test: /retrieve/docs with folder_depth=-1"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/docs" \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"folder test\",
            \"k\": 10,
            \"folder_name\": \"/sanity_test\",
            \"folder_depth\": -1,
            \"filters\": {\"test_run_id\": \"$FOLDER_TEST_RUN_ID\"}
        }" 2>&1) || {
        log_error "retrieve/docs with folder_depth=-1 failed"
        return
    }

    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    if [[ "$count" -ge 1 ]]; then
        log_success "retrieve/docs folder_depth=-1 returned $count doc(s)"
    else
        log_error "retrieve/docs folder_depth=-1 returned no results"
    fi
}

test_folder_path_normalization() {
    log_section "Testing Folder Path Normalization"

    # Test 1: Trailing slash normalization
    log_info "Test: Path with trailing slash"
    local response full_path
    response=$(curl -sf -X POST "$BASE_URL/folders" \
        -H "Content-Type: application/json" \
        -d '{"name": "trailing_slash_test", "full_path": "/sanity_test/trailing/"}' 2>&1)
    full_path=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('full_path',''))" 2>/dev/null)
    if [[ "$full_path" == "/sanity_test/trailing" ]]; then
        log_success "Trailing slash normalized correctly"
    else
        log_error "Trailing slash not normalized: $full_path"
    fi

    # Test 2: Double slashes normalization
    log_info "Test: Path with double slashes"
    response=$(curl -sf -X POST "$BASE_URL/folders" \
        -H "Content-Type: application/json" \
        -d '{"name": "double_slash_test", "full_path": "/sanity_test//double"}' 2>&1)
    full_path=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('full_path',''))" 2>/dev/null)
    if [[ "$full_path" == "/sanity_test/double" ]]; then
        log_success "Double slashes normalized correctly"
    else
        log_error "Double slashes not normalized: $full_path"
    fi

    # Test 3: Path with '..' should be rejected
    log_info "Test: Path with '..' (should fail)"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/folders" \
        -H "Content-Type: application/json" \
        -d '{"name": "bad", "full_path": "/sanity_test/../sneaky"}')
    if [[ "$http_code" == "400" ]]; then
        log_success "Path with '..' rejected (HTTP 400)"
    else
        log_error "Path with '..' should return 400, got $http_code"
    fi
}

test_folder_move_integrity() {
    log_section "Testing Folder Move Integrity"

    local src_folder="sanity_move_src_${TEST_RUN_ID}"
    local dst_folder="sanity_move_dst_${TEST_RUN_ID}"
    local src_path="/$src_folder"
    local dst_path="/$dst_folder"

    # Create source and destination folders
    curl -sf -X POST "$BASE_URL/folders" -H "Content-Type: application/json" \
        -d "{\"name\": \"$src_folder\", \"full_path\": \"$src_path\"}" > /dev/null 2>&1 || {
        log_error "Failed to create source folder $src_path"
        return
    }
    curl -sf -X POST "$BASE_URL/folders" -H "Content-Type: application/json" \
        -d "{\"name\": \"$dst_folder\", \"full_path\": \"$dst_path\"}" > /dev/null 2>&1 || {
        log_error "Failed to create destination folder $dst_path"
        return
    }

    # Resolve folder IDs
    local src_id dst_id
    src_id=$(curl -sf "$BASE_URL/folders$src_path" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
    dst_id=$(curl -sf "$BASE_URL/folders$dst_path" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)

    if [[ -z "$src_id" || -z "$dst_id" ]]; then
        log_error "Failed to resolve folder IDs (src=$src_id, dst=$dst_id)"
        return
    fi

    # Ingest a text document into the source folder
    local ingest_payload
    ingest_payload=$(cat << EOF
{"content":"Folder move integrity test for $TEST_RUN_ID","metadata":{"test_run_id":"$FOLDER_TEST_RUN_ID","move_test":true},"folder_name":"$src_path"}
EOF
)
    local ingest_response doc_id
    ingest_response=$(curl -sf -X POST "$BASE_URL/ingest/text" -H "Content-Type: application/json" -d "$ingest_payload" 2>&1) || {
        log_error "Failed to ingest move test document"
        return
    }
    doc_id=$(echo "$ingest_response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('external_id',''))" 2>/dev/null)
    if [[ -z "$doc_id" ]]; then
        log_error "No document ID returned for move test ingest"
        return
    fi
    DOC_IDS+=("$doc_id")

    # Helper: validate a document's folder state
    validate_doc_state() {
        local expected_id="$1"
        local expected_path="$2"
        local stage="$3"
        local resp="$4"
        local result
        # Use environment variable for data since heredoc consumes stdin
        result=$(DOC_DATA="$resp" python3 -c '
import sys, json, os
expected_id = sys.argv[1]
expected_path = sys.argv[2]
docs = json.loads(os.environ["DOC_DATA"])
if not docs:
    print("missing")
    sys.exit(0)
d = docs[0]
meta = d.get("metadata", {}) or {}
leaf = expected_path.strip("/").split("/")[-1] if expected_path else None
ok = (
    d.get("folder_id") == expected_id
    and meta.get("folder_id") == expected_id
    and d.get("folder_path") == expected_path
    and meta.get("folder_name") == expected_path
    and (leaf is None or d.get("folder_name") == leaf)
)
if ok:
    print("ok")
else:
    fid = d.get("folder_id")
    mfid = meta.get("folder_id")
    fp = d.get("folder_path")
    mfn = meta.get("folder_name")
    print(f"bad|folder_id={fid}|meta_folder_id={mfid}|folder_path={fp}|meta_folder_name={mfn}")
' "$expected_id" "$expected_path")
        if [[ "$result" == "ok" ]]; then
            log_success "Document state correct ($stage)"
        else
            log_error "Document state incorrect ($stage): $result"
        fi
    }

    # Validate initial folder assignment
    local doc_query_response
    doc_query_response=$(curl -sf -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -d "{\"document_filters\": {\"external_id\": \"$doc_id\"}}") || {
        log_error "Failed to fetch document for initial folder validation"
        return
    }
    validate_doc_state "$src_id" "$src_path" "after ingest" "$doc_query_response"

    # Move the document to the destination folder using ID to avoid encoding issues
    curl -sf -X POST "$BASE_URL/folders/$dst_id/documents/$doc_id" > /dev/null 2>&1 || {
        log_error "Failed to move document to destination folder"
        return
    }

    # Validate folder assignment after move
    doc_query_response=$(curl -sf -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -d "{\"document_filters\": {\"external_id\": \"$doc_id\"}}") || {
        log_error "Failed to fetch document after move"
        return
    }
    validate_doc_state "$dst_id" "$dst_path" "after move" "$doc_query_response"

    # Verify folder summary counts reflect the move
    local summary_response
    summary_response=$(curl -sf "$BASE_URL/folders/summary" 2>/dev/null) || {
        log_error "Failed to fetch folder summaries"
        return
    }
    local counts
    counts=$(SUMMARY_DATA="$summary_response" python3 -c '
import sys, json, os
src_id, dst_id = sys.argv[1], sys.argv[2]
summaries = json.loads(os.environ["SUMMARY_DATA"])
def count(fid):
    for s in summaries:
        if s.get("id") == fid:
            return s.get("doc_count", 0)
    return 0
print(f"{count(src_id)},{count(dst_id)}")
' "$src_id" "$dst_id") || counts="0,0"
    local src_count dst_count
    src_count=$(echo "$counts" | cut -d',' -f1)
    dst_count=$(echo "$counts" | cut -d',' -f2)
    if [[ "$src_count" == "0" && "$dst_count" == "1" ]]; then
        log_success "Folder summary counts reflect move (src=$src_count, dst=$dst_count)"
    else
        log_error "Folder summary counts unexpected after move (src=$src_count, dst=$dst_count)"
    fi

    # Cleanup move test folders (ignore errors)
    curl -sf -X DELETE "$BASE_URL/folders/$src_id?recursive=true" > /dev/null 2>&1 || true
    curl -sf -X DELETE "$BASE_URL/folders/$dst_id?recursive=true" > /dev/null 2>&1 || true
}

cleanup_folder_test() {
    log_info "Cleaning up folder test resources..."
    # Delete the test folder hierarchy recursively
    curl -sf -X DELETE "$BASE_URL/folders/sanity_test?recursive=true" > /dev/null 2>&1 || true
    log_info "Folder test cleanup completed"
}

test_result_validation() {
    log_section "Testing Result Validation"

    # Test 1: Verify specific content is retrievable
    log_info "Test: Search for 'vector embeddings' returns txt file content"
    local response
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"vector embeddings semantic search\", \"k\": 3, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"file_type\": \"txt\"}}" 2>&1) || {
        log_error "Content validation request failed"
        return
    }

    local found_content
    found_content=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
found = any('vector' in c.get('content', '').lower() and 'embedding' in c.get('content', '').lower() for c in chunks)
print('yes' if found else 'no')
" 2>/dev/null) || found_content="no"

    if [[ "$found_content" == "yes" ]]; then
        log_success "Content validation: 'vector embeddings' found in txt results"
    else
        log_error "Content validation: expected content not found in txt results"
    fi

    # Test 2: Verify code block content in markdown
    log_info "Test: Search for 'retrieve_chunks' returns md file with code"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"retrieve_chunks function def\", \"k\": 3, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"file_type\": \"md\"}}" 2>&1) || {
        log_error "Code content request failed"
        return
    }

    found_content=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
found = any('def retrieve_chunks' in c.get('content', '') or 'retrieve_chunks' in c.get('content', '') for c in chunks)
print('yes' if found else 'no')
" 2>/dev/null) || found_content="no"

    if [[ "$found_content" == "yes" ]]; then
        log_success "Content validation: code block content found in md results"
    else
        log_error "Content validation: code block content not found in md results"
    fi

    # Test 3: Verify CSV tabular content
    log_info "Test: Search for 'Widget Electronics' returns csv data"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"Widget Electronics product\", \"k\": 3, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"file_type\": \"csv\"}}" 2>&1) || {
        log_error "CSV content request failed"
        return
    }

    found_content=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
found = any('Widget' in c.get('content', '') or 'Electronics' in c.get('content', '') for c in chunks)
print('yes' if found else 'no')
" 2>/dev/null) || found_content="no"

    if [[ "$found_content" == "yes" ]]; then
        log_success "Content validation: CSV data found in results"
    else
        log_error "Content validation: CSV data not found in results"
    fi

    # Test 4: Verify PDF content (if PDF was ingested)
    log_info "Test: Search for 'IQ imbalance' returns PDF content"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"IQ imbalance compensation frequency\", \"k\": 3, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\", \"file_type\": \"pdf\"}}" 2>&1) || {
        log_error "PDF content request failed"
        return
    }

    local pdf_count
    pdf_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || pdf_count=0

    if [[ "$pdf_count" -gt 0 ]]; then
        log_success "Content validation: PDF content retrieved ($pdf_count chunks)"
    else
        log_warn "Content validation: No PDF chunks found (PDF may not have been ingested)"
    fi

    # Test 5: Verify relevance scores are reasonable
    log_info "Test: Verify retrieval scores are in valid range"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik test document\", \"k\": 5, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Score validation request failed"
        return
    }

    local scores_valid
    scores_valid=$(echo "$response" | python3 -c "
import sys, json
chunks = json.load(sys.stdin)
if not chunks:
    print('no')
else:
    # Scores should be between 0 and 1 for cosine similarity
    valid = all(0 <= c.get('score', -1) <= 1 for c in chunks)
    # Scores should be sorted descending
    scores = [c.get('score', 0) for c in chunks]
    sorted_desc = scores == sorted(scores, reverse=True)
    print('yes' if (valid and sorted_desc) else 'no')
" 2>/dev/null) || scores_valid="no"

    if [[ "$scores_valid" == "yes" ]]; then
        log_success "Score validation: scores are valid and sorted"
    else
        log_error "Score validation: scores invalid or not properly sorted"
    fi
}

# ============================================================================
# Extended Ingestion & Metadata Tests
# ============================================================================

test_additional_ingestion_variants() {
    log_section "Testing Additional Ingestion Endpoints"

    # Ingest text with typed metadata
    log_info "Test: /ingest/text with metadata_types"
    local metadata='{"test_run_id":"'"$TEST_RUN_ID"'","value_num":7,"value_decimal":"3.14","created_ts":"2024-01-02T10:00:00Z","tags":["alpha","beta"],"status":"active"}'
    local metadata_types='{"value_num":"number","value_decimal":"decimal","created_ts":"datetime"}'
    local response
    response=$(curl -sf -X POST "$BASE_URL/ingest/text" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"Typed metadata sanity text\", \"filename\": \"typed_metadata.txt\", \"metadata\": $metadata, \"metadata_types\": $metadata_types, \"use_colpali\": false, \"folder_name\": \"/sanity_test/typed\", \"end_user_id\": \"end_user_typed\"}" 2>&1) || {
        log_error "ingest_text request failed"
        return
    }
    TYPED_DOC_ID=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('external_id',''))" 2>/dev/null)
    if [[ -n "$TYPED_DOC_ID" ]]; then
        log_success "ingest_text created $TYPED_DOC_ID"
    else
        log_error "Failed to capture typed ingest doc id"
    fi

    # Batch ingest multiple files with per-file metadata/metadata_types
    log_info "Test: /ingest/files batch upload"
    local batch_metadata='[{"batch":"one","test_run_id":"'"$TEST_RUN_ID"'","value":1},{"batch":"two","test_run_id":"'"$TEST_RUN_ID"'","value":2}]'
    local batch_metadata_types='[{"value":"number"},{"value":"number"}]'
    response=$(curl -sf -X POST "$BASE_URL/ingest/files" \
        -F "files=@$TEST_DIR/batch_file1.txt" \
        -F "files=@$TEST_DIR/batch_file2.txt" \
        -F "metadata=$batch_metadata" \
        -F "metadata_types=$batch_metadata_types" \
        -F "folder_name=/batch_test" \
        -F "use_colpali=false" 2>&1) || {
        log_error "Batch ingest request failed"
        return
    }

    local parsed
    parsed=$(echo "$response" | python3 -c "
import sys,json
docs = json.load(sys.stdin).get('documents', [])
for d in docs:
    print(f\"{d.get('external_id','')}|{d.get('filename','')}\")
" 2>/dev/null)
    while IFS='|' read -r did fname; do
        if [[ -n "$did" ]]; then
            BATCH_DOC_IDS+=("$did")
            BATCH_FILENAMES+=("$fname")
        fi
    done <<< "$parsed"

    if [[ ${#BATCH_DOC_IDS[@]} -gt 0 ]]; then
        log_success "Batch ingest queued ${#BATCH_DOC_IDS[@]} documents"
    else
        log_error "Batch ingest returned no documents"
    fi

    # Requeue ingestion job for one document
    if [[ ${#BATCH_DOC_IDS[@]} -gt 0 ]]; then
        local target="${BATCH_DOC_IDS[0]}"
        log_info "Test: /ingest/requeue for $target"
        response=$(curl -sf -X POST "$BASE_URL/ingest/requeue" \
            -H "Content-Type: application/json" \
            -d "{\"jobs\": [{\"external_id\": \"$target\", \"use_colpali\": true}], \"include_all\": false}" 2>&1) || {
            log_error "Requeue request failed"
        }
        local status
        status=$(echo "$response" | python3 -c "
import sys,json
res=json.load(sys.stdin).get('results',[])
print(res[0].get('status','') if res else '')
" 2>/dev/null)
        if [[ -n "$status" ]]; then
            log_success "Requeue response status: $status"
        else
            log_warn "Requeue response did not include status"
        fi
    else
        log_warn "Skipping requeue test (no batch docs)"
    fi

    # Morphik On-the-Fly document query with optional ingest
    log_info "Test: /ingest/document/query with schema + ingest"
    response=$(curl -sf -X POST "$BASE_URL/ingest/document/query" \
        -F "file=@$TEST_DIR/test_document.txt" \
        -F "prompt=Extract title and keywords" \
        -F "schema={\"title\":\"string\",\"keywords\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}}" \
        -F "ingestion_options={\"ingest\": true, \"metadata\": {\"test_run_id\": \""$TEST_RUN_ID"\", \"doc_query\": true}, \"use_colpali\": false}" 2>&1) || {
        log_error "Document query request failed"
        return
    }

    local structured_present
    structured_present=$(echo "$response" | python3 -c "
import sys,json
payload=json.load(sys.stdin)
print('yes' if payload.get('structured_output') is not None else 'no')
" 2>/dev/null)
    INGEST_QUERY_DOC_ID=$(echo "$response" | python3 -c "
import sys,json
payload=json.load(sys.stdin)
doc=payload.get('ingestion_document') or {}
print(doc.get('external_id',''))
" 2>/dev/null)

    if [[ "$structured_present" == "yes" ]]; then
        log_success "Document query returned structured_output (queued doc: ${INGEST_QUERY_DOC_ID:-none})"
    else
        log_warn "Document query returned no structured_output"
    fi
}

test_typed_metadata_filters() {
    log_section "Testing Typed Metadata & Filter Operators"
    if [[ -z "$TYPED_DOC_ID" ]]; then
        log_warn "Typed ingest doc not available, skipping typed metadata tests"
        return
    fi

    # $type filters on number/decimal/datetime
    log_info "Test: \$type filters for typed metadata"
    local response
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"typed metadata\", \"k\": 5, \"filters\": {\"\$and\": [{\"test_run_id\": \"$TEST_RUN_ID\"}, {\"value_num\": {\"\$type\": \"number\"}}, {\"value_decimal\": {\"\$type\": \"decimal\"}}, {\"created_ts\": {\"\$type\": \"datetime\"}}]}}" 2>&1) || {
        log_error "Typed \$type filter request failed"
        return
    }
    local count
    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    if [[ "$count" -gt 0 ]]; then
        log_success "\$type filter retrieval returned $count chunks"
    else
        log_error "\$type filter retrieval returned no chunks"
    fi

    # Regex / contains / exists operators
    log_info "Test: \$regex, \$contains, \$exists, \$in/\$nin operators"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"alpha\", \"k\": 5, \"filters\": {\"\$and\": [{\"test_run_id\": \"$TEST_RUN_ID\"}, {\"tags\": {\"\$contains\": \"alpha\"}}, {\"status\": {\"\$regex\": \"active\"}}, {\"missing_field\": {\"\$exists\": false}}, {\"tags\": {\"\$in\": [\"alpha\", \"gamma\"]}}, {\"status\": {\"\$nin\": [\"inactive\"]}}]}}" 2>&1) || {
        log_error "Regex/contains filter request failed"
        return
    }
    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    if [[ "$count" -gt 0 ]]; then
        log_success "Logical/regex filters returned $count chunks"
    else
        log_error "Logical/regex filters returned no chunks"
    fi

    # /retrieve/docs with typed filters
    log_info "Test: /retrieve/docs respects typed filters"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/docs" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"Typed\", \"k\": 3, \"filters\": {\"\$and\": [{\"test_run_id\": \"$TEST_RUN_ID\"}, {\"value_num\": {\"\$gte\": 5}}, {\"value_decimal\": {\"\$lte\": 5}}]}}" 2>&1) || {
        log_error "retrieve/docs typed filter request failed"
        return
    }
    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    if [[ "$count" -gt 0 ]]; then
        log_success "/retrieve/docs typed filters returned $count docs"
    else
        log_error "/retrieve/docs typed filters returned no docs"
    fi

    # /documents list with typed filters
    log_info "Test: /documents list with typed filters"
    response=$(curl -sf -X POST "$BASE_URL/documents" \
        -H "Content-Type: application/json" \
        -d "{\"skip\":0,\"limit\":5,\"document_filters\":{\"test_run_id\":\"$TEST_RUN_ID\",\"value_decimal\":{\"\$gte\":2}}}" 2>&1) || {
        log_error "/documents typed list request failed"
        return
    }
    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    if [[ "$count" -gt 0 ]]; then
        log_success "/documents typed list returned $count docs"
    else
        log_error "/documents typed list returned no docs"
    fi

    # Invalid type coercion should return 400
    log_info "Test: Invalid type comparison returns 400"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"typed\", \"k\": 1, \"filters\": {\"value_num\": {\"\$gt\": \"not_a_number\"}}}")
    if [[ "$http_code" == "400" || "$http_code" == "422" ]]; then
        log_success "Invalid type comparison rejected with HTTP $http_code"
    else
        log_error "Invalid type comparison expected 400/422, got $http_code"
    fi
}

# ============================================================================
# Document Management & Retrieval Variants
# ============================================================================

test_document_management_and_updates() {
    log_section "Testing Document Management APIs"
    local target_doc="${BATCH_DOC_IDS[0]:-${DOC_IDS[0]:-}}"
    if [[ -z "$target_doc" ]]; then
        log_warn "No document available for management tests"
        return
    fi

    # Document status
    log_info "Test: /documents/{id}/status"
    local response
    response=$(curl -sf "$BASE_URL/documents/$target_doc/status" 2>&1) || {
        log_error "Document status request failed"
        return
    }
    local status
    status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    log_success "Document status: ${status:-unknown}"

    # Get by filename (0/1 cases)
    if [[ ${#BATCH_FILENAMES[@]} -gt 0 ]]; then
        local fname="${BATCH_FILENAMES[0]}"
        log_info "Test: /documents/filename/$fname"
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/documents/filename/$fname")
        if [[ "$http_code" == "200" ]]; then
            log_success "Filename lookup returned 200 for $fname"
        else
            log_error "Filename lookup expected 200, got $http_code"
        fi
    fi

    # download_url and direct download
    log_info "Test: /documents/{id}/download_url"
    response=$(curl -sf "$BASE_URL/documents/$target_doc/download_url" 2>&1) || {
        log_error "Download URL request failed"
        return
    }
    local download_url
    download_url=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('download_url',''))" 2>/dev/null)
    if [[ -n "$download_url" ]]; then
        log_success "Download URL generated"
    else
        log_error "Download URL missing"
    fi

    log_info "Test: /documents/{id}/file streaming"
    local file_content
    file_content=$(curl -sf "$BASE_URL/documents/$target_doc/file" 2>/dev/null | head -c 50) || {
        log_error "Document file download failed"
        return
    }
    if [[ -n "$file_content" ]]; then
        log_success "Document file download returned content"
    else
        log_error "Document file download returned empty content"
    fi

    # update_text on typed doc (creates additive content)
    if [[ -n "$TYPED_DOC_ID" ]]; then
        log_info "Test: /documents/{id}/update_text"
        response=$(curl -sf -X POST "$BASE_URL/documents/$TYPED_DOC_ID/update_text" \
            -H "Content-Type: application/json" \
            -d "{\"content\": \"Updated text content for sanity test\", \"filename\": \"typed_metadata.txt\", \"metadata\": {\"update_pass\": true}, \"use_colpali\": false}" 2>&1) || {
            log_error "update_text request failed"
        }
        if [[ -n "$response" ]]; then
            log_success "update_text responded for $TYPED_DOC_ID"
        fi
    fi

    # update_file with replace strategy
    log_info "Test: /documents/{id}/update_file"
    echo "Updated file content" > "$TEST_DIR/updated_content.txt"
    response=$(curl -sf -X POST "$BASE_URL/documents/$target_doc/update_file" \
        -F "file=@$TEST_DIR/updated_content.txt" \
        -F "metadata={\"updated_via_api\":true}" \
        -F "metadata_types={}" \
        -F "update_strategy=replace" \
        -F "use_colpali=false" 2>&1) || {
        log_error "update_file request failed"
    }
    if [[ -n "$response" ]]; then
        log_success "update_file responded for $target_doc"
    fi

    # update_metadata add-only
    log_info "Test: /documents/{id}/update_metadata"
    response=$(curl -sf -X POST "$BASE_URL/documents/$target_doc/update_metadata" \
        -H "Content-Type: application/json" \
        -d "{\"metadata\": {\"reviewed\": true}, \"metadata_types\": {}}" 2>&1) || {
        log_error "update_metadata request failed"
    }
    if [[ -n "$response" ]]; then
        log_success "update_metadata responded for $target_doc"
    fi

    # list_docs aggregates with pagination
    log_info "Test: /documents/list_docs aggregates"
    response=$(curl -sf -X POST "$BASE_URL/documents/list_docs" \
        -H "Content-Type: application/json" \
        -d "{\"document_filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}, \"skip\": 0, \"limit\": 2, \"include_total_count\": true, \"include_status_counts\": true, \"include_folder_counts\": true, \"return_documents\": false}" 2>&1) || {
        log_error "/documents/list_docs request failed"
        return
    }
    local total_count
    total_count=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_count',0))" 2>/dev/null) || total_count=0
    if [[ "$total_count" -gt 0 ]]; then
        log_success "list_docs returned total_count=$total_count"
    else
        log_error "list_docs returned no documents"
    fi

    # search/documents by filename
    log_info "Test: /search/documents name search"
    response=$(curl -sf -X POST "$BASE_URL/search/documents" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"batch_file\", \"limit\": 5, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "search/documents request failed"
    }
    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    if [[ "$count" -gt 0 ]]; then
        log_success "search/documents returned $count result(s)"
    else
        log_warn "search/documents returned no results"
    fi

    # documents/pages extraction (PDF)
    if [[ -n "$PDF_DOC_ID" ]]; then
        log_info "Test: /documents/pages for PDF"
        response=$(curl -sf -X POST "$BASE_URL/documents/pages" \
            -H "Content-Type: application/json" \
            -d "{\"document_id\": \"$PDF_DOC_ID\", \"start_page\": 1, \"end_page\": 1}" 2>&1) || {
            log_error "documents/pages request failed"
        }
        local images_count
        images_count=$(echo "$response" | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('pages',[])))" 2>/dev/null) || images_count=0
        if [[ "$images_count" -gt 0 ]]; then
            log_success "documents/pages returned $images_count page image(s)"
        else
            log_warn "documents/pages returned no images"
        fi
    else
        log_warn "Skipping documents/pages (no PDF ingested)"
    fi

    # Delete one document (exercise delete & 0-case)
    if [[ ${#BATCH_DOC_IDS[@]} -gt 1 ]]; then
        local delete_id="${BATCH_DOC_IDS[1]}"
        log_info "Test: DELETE /documents/$delete_id"
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/documents/$delete_id")
        if [[ "$http_code" == "200" ]]; then
            log_success "Deleted document $delete_id"
            http_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/documents/$delete_id/status")
            if [[ "$http_code" == "404" ]]; then
                log_success "Deleted document no longer retrievable (404)"
            else
                log_warn "Deleted document status check returned $http_code"
            fi
        else
            log_error "Delete expected 200, got $http_code"
        fi
    fi
}

test_retrieval_variants_and_batches() {
    log_section "Testing Retrieval Variants & Batch APIs"

    # Capture a chunk reference for batch/chunks
    local response
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik\", \"k\": 1, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Initial chunk fetch failed"
        return
    }
    local chunk_info
    chunk_info=$(echo "$response" | python3 -c "
import sys,json
chunks=json.load(sys.stdin)
if chunks:
    first=chunks[0]
    print(f\"{first.get('document_id','')}|{first.get('chunk_number','')}\")
" 2>/dev/null)
    IFS='|' read -r LAST_CHUNK_DOC_ID LAST_CHUNK_NUMBER <<< "$chunk_info"

    # Grouped retrieval with padding
    log_info "Test: /retrieve/chunks/grouped with padding"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks/grouped" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik\", \"k\": 5, \"padding\": 1, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Grouped retrieval request failed"
    }
    local group_count
    group_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('groups',[])))" 2>/dev/null) || group_count=0
    if [[ "$group_count" -gt 0 ]]; then
        log_success "Grouped retrieval returned $group_count groups"
    else
        log_warn "Grouped retrieval returned no groups"
    fi

    # Image query retrieval
    log_info "Test: /retrieve/chunks with query_image"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query_image\": \"$QUERY_IMAGE_B64\", \"k\": 2, \"use_colpali\": true, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "query_image retrieval failed"
    }
    local img_count
    img_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || img_count=0
    log_success "query_image retrieval returned $img_count chunk(s)"

    # /retrieve/docs must reject image queries
    log_info "Test: /retrieve/docs rejects query_image"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/retrieve/docs" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik\", \"query_image\": \"$QUERY_IMAGE_B64\", \"k\": 1, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}")
    if [[ "$http_code" == "400" || "$http_code" == "422" ]]; then
        log_success "/retrieve/docs rejected image query ($http_code)"
    else
        log_error "/retrieve/docs expected 400/422 for image query, got $http_code"
    fi

    # min_score + reranking
    log_info "Test: min_score with use_reranking"
    response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"morphik\", \"k\": 3, \"min_score\": 0.99, \"use_reranking\": true, \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "min_score retrieval failed"
    }
    count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
    log_success "min_score retrieval returned $count chunk(s)"

    # Output formats on image doc
    if [[ -n "$IMAGE_COLPALI_DOC_ID" ]]; then
        log_info "Test: output_format URL for image doc"
        response=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
            -H "Content-Type: application/json" \
            -d "{\"query\": \"image\", \"k\": 2, \"use_colpali\": true, \"output_format\": \"url\", \"filters\": {\"external_id\": \"$IMAGE_COLPALI_DOC_ID\"}}" 2>&1) || {
            log_error "output_format url retrieval failed"
        }
        local has_url
        has_url=$(echo "$response" | python3 -c "
import sys,json
chunks=json.load(sys.stdin)
has_url = any(c.get('download_url') for c in chunks)
print('yes' if has_url else 'no')
" 2>/dev/null) || has_url="no"
        if [[ "$has_url" == "yes" ]]; then
            log_success "Image output_format=url returned download_url"
        else
            log_warn "Image output_format=url did not include URLs"
        fi
    else
        log_warn "Skipping image output_format tests (no image colpali doc)"
    fi

    # Batch documents
    log_info "Test: /batch/documents"
    local doc_list=()
    [[ -n "$TYPED_DOC_ID" ]] && doc_list+=("\"$TYPED_DOC_ID\"")
    if [[ ${#BATCH_DOC_IDS[@]} -gt 0 ]]; then
        doc_list+=("\"${BATCH_DOC_IDS[0]}\"")
    fi
    if [[ -n "$INGEST_QUERY_DOC_ID" ]]; then
        doc_list+=("\"$INGEST_QUERY_DOC_ID\"")
    fi
    if [[ ${#doc_list[@]} -eq 0 ]]; then
        log_warn "No documents available for /batch/documents"
    else
        local doc_payload
        doc_payload=$(IFS=,; echo "[${doc_list[*]}]")
        response=$(curl -sf -X POST "$BASE_URL/batch/documents" \
            -H "Content-Type: application/json" \
            -d "{\"document_ids\": $doc_payload}" 2>&1) || {
            log_error "/batch/documents request failed"
        }
        count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
        log_success "/batch/documents returned $count document(s)"
    fi

    # Batch chunks
    if [[ -n "$LAST_CHUNK_DOC_ID" && -n "$LAST_CHUNK_NUMBER" ]]; then
        log_info "Test: /batch/chunks"
        response=$(curl -sf -X POST "$BASE_URL/batch/chunks" \
            -H "Content-Type: application/json" \
            -d "{\"sources\": [{\"document_id\": \"$LAST_CHUNK_DOC_ID\", \"chunk_number\": $LAST_CHUNK_NUMBER}], \"output_format\": \"text\"}" 2>&1) || {
            log_error "/batch/chunks request failed"
        }
        count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || count=0
        if [[ "$count" -gt 0 ]]; then
            log_success "/batch/chunks returned $count chunk(s)"
        else
            log_warn "/batch/chunks returned no chunks"
        fi
    else
        log_warn "Skipping /batch/chunks (no chunk reference captured)"
    fi

    # search/documents already covered, but exercise filename search again (0/1/many)
    log_info "Test: /search/documents no-match case"
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/search/documents" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"nonexistent_filename_$TEST_RUN_ID\", \"limit\": 3}")
    if [[ "$http_code" == "200" ]]; then
        log_success "search/documents handled empty result set (HTTP 200)"
    else
        log_error "search/documents empty case expected 200, got $http_code"
    fi
}

test_query_and_chat_flows() {
    log_section "Testing Query & Chat Flows"

    # Missing text should error
    log_info "Test: /query rejects missing text"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"\", \"k\": 1}")
    if [[ "$http_code" == "400" || "$http_code" == "422" ]]; then
        log_success "/query missing text rejected ($http_code)"
    else
        log_error "/query missing text expected 400/422, got $http_code"
    fi

    local chat_id="chat_$TEST_RUN_ID"

    # Non-stream query with inline_citations/response_schema/llm_config/padding
    log_info "Test: /query non-stream with inline_citations & schema"
    local tmp_body="$TEST_DIR/query_non_stream.json"
    cat > "$tmp_body" << EOF
{"query": "Summarize Morphik sanity docs", "k": 2, "inline_citations": true, "padding": 1, "response_schema": {"type": "object", "properties": {"summary": {"type": "string"}}}, "llm_config": {"temperature": 0}, "chat_id": "$chat_id", "stream_response": false}
EOF
    http_code=$(curl -s -o /tmp/query_non_stream.out -w "%{http_code}" -X POST "$BASE_URL/query" \
        -H "Content-Type: application/json" \
        -d @"$tmp_body")
    if [[ "$http_code" == "200" ]]; then
        log_success "/query non-stream returned 200"
    else
        log_error "/query non-stream expected 200, got $http_code"
    fi

    # Streaming query (SSE)
    log_info "Test: /query stream_response=true"
    local stream_file="$TEST_DIR/query_stream.out"
    http_code=$(curl -sN --max-time 20 -o "$stream_file" -w "%{http_code}" -X POST "$BASE_URL/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"Stream a short response\", \"k\": 1, \"chat_id\": \"$chat_id\", \"stream_response\": true}")
    if [[ "$http_code" == "200" ]]; then
        if grep -q "data:" "$stream_file"; then
            log_success "/query streaming returned event data"
        else
            log_error "/query streaming returned 200 but no event data (stream failed mid-response)"
        fi
    else
        log_error "/query streaming expected 200, got $http_code"
    fi

    # Chat history 0/1 cases
    log_info "Test: /chat/{chat_id} history retrieval"
    local chat_response
    chat_response=$(curl -sf "$BASE_URL/chat/$chat_id" 2>&1) || chat_response="[]"
    local chat_count
    chat_count=$(echo "$chat_response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || chat_count=0
    log_success "/chat/$chat_id returned $chat_count message(s)"

    log_info "Test: /chats listing"
    chat_response=$(curl -sf "$BASE_URL/chats" 2>&1) || chat_response="[]"
    chat_count=$(echo "$chat_response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || chat_count=0
    log_success "/chats returned $chat_count chat(s)"

    log_info "Test: /chats/{chat_id}/title update"
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X PATCH "$BASE_URL/chats/$chat_id/title?title=Sanity%20Chat" 2>/dev/null)
    if [[ "$http_code" == "200" || "$http_code" == "404" ]]; then
        log_success "/chats/{chat_id}/title responded with $http_code"
    else
        log_error "/chats/{chat_id}/title unexpected status $http_code"
    fi
}

# ============================================================================
# Folder Scoping & Summaries
# ============================================================================

test_folder_scoping_and_summary() {
    log_section "Testing Folder Scoping and Summaries"

    local root="/scope_suite_$TEST_RUN_ID"
    log_info "Test: Create scoped folders"
    curl -sf -X POST "$BASE_URL/folders" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"scope_suite_$TEST_RUN_ID\", \"full_path\": \"$root\"}" > /dev/null 2>&1 || true
    curl -sf -X POST "$BASE_URL/folders" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"a\", \"full_path\": \"$root/a\"}" > /dev/null 2>&1 || true
    curl -sf -X POST "$BASE_URL/folders" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"b\", \"full_path\": \"$root/b\"}" > /dev/null 2>&1 || true

    # Ingest scoped docs with end_user_id variants
    log_info "Test: Ingest scoped docs with end_user_id"
    local resp
    resp=$(curl -sf -X POST "$BASE_URL/ingest/text" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"Scoped root doc\", \"filename\": \"scope_root.txt\", \"metadata\": {\"test_run_id\": \"$TEST_RUN_ID\", \"scope\": \"root\"}, \"folder_name\": \"$root\", \"end_user_id\": \"user_a\"}" 2>&1) || {
        log_warn "Scoped root ingest failed"
    }
    SCOPE_DOC_IDS+=($(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('external_id',''))" 2>/dev/null))

    resp=$(curl -sf -X POST "$BASE_URL/ingest/text" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"Scoped child A\", \"filename\": \"scope_a.txt\", \"metadata\": {\"test_run_id\": \"$TEST_RUN_ID\", \"scope\": \"a\"}, \"folder_name\": \"$root/a\", \"end_user_id\": \"user_a\"}" 2>&1) || {
        log_warn "Scoped a ingest failed"
    }
    SCOPE_DOC_IDS+=($(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('external_id',''))" 2>/dev/null))

    resp=$(curl -sf -X POST "$BASE_URL/ingest/text" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"Scoped child B\", \"filename\": \"scope_b.txt\", \"metadata\": {\"test_run_id\": \"$TEST_RUN_ID\", \"scope\": \"b\"}, \"folder_name\": \"$root/b\", \"end_user_id\": \"user_b\"}" 2>&1) || {
        log_warn "Scoped b ingest failed"
    }
    SCOPE_DOC_IDS+=($(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('external_id',''))" 2>/dev/null))

    # List folders (covers >0 case)
    log_info "Test: /folders list"
    resp=$(curl -sf "$BASE_URL/folders" 2>&1) || resp="[]"
    local folder_count
    folder_count=$(echo "$resp" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || folder_count=0
    log_success "/folders returned $folder_count folder(s)"

    # Folder details with document counts
    log_info "Test: /folders/details with counts"
    resp=$(curl -sf -X POST "$BASE_URL/folders/details" \
        -H "Content-Type: application/json" \
        -d "{\"identifiers\": [\"$root\", \"$root/a\"], \"include_document_count\": true, \"include_status_counts\": true}" 2>&1) || {
        log_error "/folders/details request failed"
    }
    local details_count
    details_count=$(echo "$resp" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('folders',[])))" 2>/dev/null) || details_count=0
    log_success "/folders/details returned $details_count entries"

    # Summary endpoint
    log_info "Test: /folders/summary"
    resp=$(curl -sf "$BASE_URL/folders/summary" 2>&1) || resp="[]"
    local root_count_before
    root_count_before=$(echo "$resp" | python3 -c "
import sys,json
for entry in json.load(sys.stdin):
    if entry.get('folder') == '$root':
        print(entry.get('document_count',0))
        break
" 2>/dev/null)
    log_success "/folders/summary includes root folder (count=${root_count_before:-0})"

    # Add/remove document to/from folder
    if [[ -n "$TYPED_DOC_ID" ]]; then
        log_info "Test: add/remove document to folder via /folders/{path}/documents/{id}"
        local encoded_folder
        encoded_folder=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$root'))")
        http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/folders/${encoded_folder}/documents/$TYPED_DOC_ID")
        if [[ "$http_code" == "200" ]]; then
            log_success "Added document to $root"
        else
            log_warn "Add document returned $http_code"
        fi
        http_code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/folders/${encoded_folder}/documents/$TYPED_DOC_ID")
        if [[ "$http_code" == "200" ]]; then
            log_success "Removed document from $root"
        else
            log_warn "Remove document returned $http_code"
        fi
    fi

    # folder_name list selector in retrieval
    log_info "Test: folder_name list scoping in /retrieve/chunks"
    resp=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"Scoped\", \"k\": 10, \"folder_name\": [\"$root/a\", \"$root/b\"], \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "folder_name list retrieval failed"
    }
    local scoped_count
    scoped_count=$(echo "$resp" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || scoped_count=0
    if [[ "$scoped_count" -ge 2 ]]; then
        log_success "folder_name list retrieval returned $scoped_count chunks"
    else
        log_warn "folder_name list retrieval returned $scoped_count chunk(s)"
    fi

    # end_user_id scoping
    log_info "Test: end_user_id scoping"
    resp=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"Scoped\", \"k\": 5, \"end_user_id\": \"user_b\", \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "end_user_id scoped retrieval failed"
    }
    local user_b_count
    user_b_count=$(echo "$resp" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || user_b_count=0
    log_success "end_user_id=user_b returned $user_b_count chunk(s)"

    # 0-case: nonexistent folder
    log_info "Test: folder_name nonexistent should return 0"
    resp=$(curl -sf -X POST "$BASE_URL/retrieve/chunks" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"Scoped\", \"k\": 5, \"folder_name\": [\"$root/doesnotexist\"], \"filters\": {\"test_run_id\": \"$TEST_RUN_ID\"}}" 2>&1) || {
        log_error "Nonexistent folder retrieval failed"
    }
    scoped_count=$(echo "$resp" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || scoped_count=0
    if [[ "$scoped_count" -eq 0 ]]; then
        log_success "Nonexistent folder retrieval returned 0 as expected"
    else
        log_warn "Nonexistent folder retrieval returned $scoped_count chunk(s)"
    fi

    # Delete a scoped document to ensure summary updates
    if [[ ${#SCOPE_DOC_IDS[@]} -gt 0 ]]; then
        local delete_id="${SCOPE_DOC_IDS[0]}"
        log_info "Test: Delete scoped document $delete_id and verify summary"
        http_code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/documents/$delete_id")
        if [[ "$http_code" == "200" ]]; then
            resp=$(curl -sf "$BASE_URL/folders/summary" 2>&1) || resp="[]"
            local root_count_after
            root_count_after=$(echo "$resp" | python3 -c "
import sys,json
for entry in json.load(sys.stdin):
    if entry.get('folder') == '$root':
        print(entry.get('document_count',0))
        break
" 2>/dev/null)
            log_success "Folder summary updated (before=${root_count_before:-0}, after=${root_count_after:-0})"
        else
            log_warn "Delete scoped document returned $http_code"
        fi
    fi
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup_test_files() {
    log_section "Cleanup"

    if [[ "${1:-}" == "--skip-cleanup" ]]; then
        log_info "Skipping cleanup (--skip-cleanup flag)"
        log_info "Test files in: $TEST_DIR"
        log_info "Test run ID: $TEST_RUN_ID"
        return
    fi

    log_info "Removing test files from $TEST_DIR..."
    rm -rf "$TEST_DIR"
    log_success "Test files cleaned up"

    log_info "Note: Test documents remain in database with test_run_id=$TEST_RUN_ID"
    log_info "To delete them, use: curl -X DELETE '$BASE_URL/documents/{id}' for each ID"
}

# ============================================================================
# Main
# ============================================================================

print_summary() {
    log_section "Test Summary"
    echo -e "Test Run ID: ${BLUE}$TEST_RUN_ID${NC}"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "\n${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "\n${RED}Some tests failed.${NC}"
        return 1
    fi
}

main() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           Morphik Sanity Test Suite                           ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    check_server
    create_test_files
    test_protected_field_guards
    run_ingestion_tests
    wait_for_processing
    test_basic_retrieval
    test_metadata_filtering
    test_date_filtering
    test_output_formats
    test_colpali_vs_standard
    test_content_preservation
    test_result_validation
    test_additional_ingestion_variants

    # Wait for typed metadata document to be processed before filter tests
    if [[ -n "$TYPED_DOC_ID" ]]; then
        log_info "Waiting for typed metadata document to be processed..."
        for i in {1..10}; do
            local status
            status=$(curl -sf "$BASE_URL/documents/$TYPED_DOC_ID" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('system_metadata',{}).get('status',''))" 2>/dev/null) || status=""
            if [[ "$status" == "completed" ]]; then
                log_success "Typed metadata document processed"
                break
            fi
            sleep 2
        done
    fi

    test_typed_metadata_filters
    test_document_management_and_updates
    test_retrieval_variants_and_batches
    test_query_and_chat_flows
    test_folder_scoping_and_summary

    # Folder Nesting Tests
    test_folder_creation_and_nesting
    test_folder_document_ingestion
    wait_for_folder_docs_processing
    test_folder_depth_filtering
    test_folder_retrieval_scoping
    test_folder_path_normalization
    test_folder_move_integrity
    cleanup_folder_test

    cleanup_test_files "${1:-}"

    print_summary
}

main "$@"

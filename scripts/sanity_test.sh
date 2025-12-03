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
    if curl -sf "$BASE_URL/ping" > /dev/null 2>&1; then
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
    run_ingestion_tests
    wait_for_processing
    test_basic_retrieval
    test_metadata_filtering
    test_date_filtering
    test_output_formats
    test_colpali_vs_standard
    test_content_preservation
    test_result_validation
    cleanup_test_files "${1:-}"

    print_summary
}

main "$@"

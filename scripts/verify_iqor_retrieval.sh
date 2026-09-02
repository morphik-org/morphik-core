#!/usr/bin/env bash
set -euo pipefail

: "${IQOR_QUERY:?Set IQOR_QUERY to the retrieval text for work item 47490}"

MORPHIK_BASE_URL="${MORPHIK_BASE_URL:-http://localhost:8000}"
IQOR_PROJECT="${IQOR_PROJECT:-QA}"
IQOR_BACKLOG_TYPE="${IQOR_BACKLOG_TYPE:-Product Backlog Item}"
IQOR_WORK_ITEM_ID_JSON="${IQOR_WORK_ITEM_ID_JSON:-47490}"
IQOR_PROJECT_FIELD="${IQOR_PROJECT_FIELD:-project}"
IQOR_TYPE_FIELD="${IQOR_TYPE_FIELD:-work_item_type}"
IQOR_ID_FIELD="${IQOR_ID_FIELD:-work_item_id}"
IQOR_CANDIDATE_K="${IQOR_CANDIDATE_K:-50}"
IQOR_USE_COLPALI_JSON="${IQOR_USE_COLPALI_JSON:-false}"
IQOR_RESPONSE_FILE="${IQOR_RESPONSE_FILE:-iqor-retrieval-response.json}"

for command in curl jq; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "$command is required." >&2
        exit 1
    fi
done

if ! jq -en --argjson value "$IQOR_WORK_ITEM_ID_JSON" '$value' >/dev/null; then
    echo "IQOR_WORK_ITEM_ID_JSON must be valid JSON, for example 47490 or \"47490\"." >&2
    exit 2
fi
if ! jq -en --argjson value "$IQOR_USE_COLPALI_JSON" '$value | type == "boolean"' | grep -q true; then
    echo "IQOR_USE_COLPALI_JSON must be true or false." >&2
    exit 2
fi

curl_args=(
    --silent
    --show-error
    --output "$IQOR_RESPONSE_FILE"
    --write-out "%{http_code}"
    --header "Content-Type: application/json"
)
if [ -n "${MORPHIK_AUTH_TOKEN:-}" ]; then
    curl_args+=(--header "Authorization: Bearer ${MORPHIK_AUTH_TOKEN}")
fi

http_status=$(
    jq -n \
        --arg query "$IQOR_QUERY" \
        --arg project "$IQOR_PROJECT" \
        --arg backlog_type "$IQOR_BACKLOG_TYPE" \
        --arg project_field "$IQOR_PROJECT_FIELD" \
        --arg type_field "$IQOR_TYPE_FIELD" \
        --arg id_field "$IQOR_ID_FIELD" \
        --argjson current_id "$IQOR_WORK_ITEM_ID_JSON" \
        --argjson candidate_k "$IQOR_CANDIDATE_K" \
        --argjson use_colpali "$IQOR_USE_COLPALI_JSON" \
        '{
            query: $query,
            k: $candidate_k,
            min_score: 0.0,
            use_colpali: $use_colpali,
            output_format: "text",
            filters: {
                "$and": [
                    {($project_field): {"$eq": $project}},
                    {($type_field): {"$eq": $backlog_type}},
                    {($id_field): {"$ne": $current_id}}
                ]
            }
        }' |
        curl "${curl_args[@]}" --data-binary @- "${MORPHIK_BASE_URL%/}/retrieve/chunks"
)

case "$http_status" in
    2??) ;;
    *)
        echo "Morphik returned HTTP $http_status. Response saved to $IQOR_RESPONSE_FILE." >&2
        jq . "$IQOR_RESPONSE_FILE" >&2 2>/dev/null || sed -n '1,120p' "$IQOR_RESPONSE_FILE" >&2
        exit 1
        ;;
esac

if ! jq -e \
    --arg project "$IQOR_PROJECT" \
    --arg backlog_type "$IQOR_BACKLOG_TYPE" \
    --arg project_field "$IQOR_PROJECT_FIELD" \
    --arg type_field "$IQOR_TYPE_FIELD" \
    --arg id_field "$IQOR_ID_FIELD" \
    --argjson current_id "$IQOR_WORK_ITEM_ID_JSON" \
    'type == "array" and all(.[];
        (.metadata[$project_field] == $project) and
        (.metadata[$type_field] == $backlog_type) and
        (.metadata[$id_field] != $current_id) and
        (.document_id | type == "string") and
        (.chunk_number | type == "number") and
        (.score | type == "number"))' \
    "$IQOR_RESPONSE_FILE" >/dev/null; then
    echo "Response violated the metadata, exclusion, score, or source-ID contract." >&2
    exit 1
fi

distinct_count=$(jq \
    --arg id_field "$IQOR_ID_FIELD" \
    '[.[].metadata[$id_field]] | unique | length' \
    "$IQOR_RESPONSE_FILE")

if (( distinct_count < 5 )); then
    echo "Expected at least five distinct backlog items, got $distinct_count. Raw response: $IQOR_RESPONSE_FILE" >&2
    exit 1
fi

jq \
    --arg id_field "$IQOR_ID_FIELD" \
    'group_by(.metadata[$id_field])
    | map(max_by(.score))
    | sort_by(-.score)
    | .[:5]
    | map(. + {source_id: (.document_id + ":" + (.chunk_number | tostring))})' \
    "$IQOR_RESPONSE_FILE"

echo "PASS: response contains at least five distinct QA backlog items and excludes work item 47490." >&2
echo "Raw response saved to $IQOR_RESPONSE_FILE." >&2

#!/usr/bin/env bash
set -euo pipefail

# Purpose: Production startup script for Morphik (created by install_docker.sh)
# This script reads the port from morphik.toml and passes it to Docker Compose.
# It is safe to run repeatedly and never rewrites the compose file.
# Usage: ./start-morphik.sh [--version <tag>]

# Color output functions
print_info() {
    echo -e "\033[34mℹ️  $1\033[0m"
}

print_success() {
    echo -e "\033[32m✅ $1\033[0m"
}

print_error() {
    echo -e "\033[31m❌ $1\033[0m" >&2
}

# Parse --version flag (overrides .env)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            if [ "$#" -lt 2 ]; then
                print_error "--version requires a tag"
                exit 2
            fi
            export MORPHIK_VERSION="$2"
            shift 2
            ;;
        --version=*)
            export MORPHIK_VERSION="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Load MORPHIK_VERSION from .env if not set via flag
if [ -z "${MORPHIK_VERSION:-}" ] && [ -f ".env" ]; then
    MORPHIK_VERSION=$(sed -n 's/^MORPHIK_VERSION=//p' .env | tail -n1)
fi
export MORPHIK_VERSION="${MORPHIK_VERSION:-latest}"

print_info "Using Morphik version: ${MORPHIK_VERSION}"

COMPOSE_FILE="docker-compose.run.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    print_error "docker-compose.run.yml not found. Please run the install script first."
    exit 1
fi

# Read the API port and use 8000 when the setting is absent.
API_PORT=$(awk '/^\[api\]/{flag=1; next} /^\[/{flag=0} flag && /^port[[:space:]]*=/ {gsub(/^port[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || true)
export MORPHIK_API_PORT="${API_PORT:-8000}"

UI_PROFILE=""
if [ -f ".env" ] && grep -q "^UI_INSTALLED=true" .env; then
    UI_PROFILE="--profile ui"
fi

# Read one value from the [backup] section of morphik.toml.
backup_setting() {
    awk -v key="$1" '
        /^\[/ { in_backup = ($0 ~ /^\[backup\][[:space:]]*(#.*)?$/); next }
        in_backup && $0 ~ ("^" key "[[:space:]]*=") {
            sub(/^[^=]*=[[:space:]]*/, ""); sub(/[[:space:]]*#.*$/, ""); gsub(/"/, ""); print; exit
        }' morphik.toml 2>/dev/null || true
}

# [backup] enabled = true starts the scheduled backup service. A non-empty s3_uri also starts
# backup-s3, which copies every backup off this host.
BACKUP_PROFILES=""
if [ "$(backup_setting enabled)" = "true" ]; then
    if [ ! -f morphik-backup.sh ]; then
        print_error "[backup] enabled = true, but morphik-backup.sh is missing. Download it next to this script."
        exit 1
    fi
    BACKUP_DIR=$(backup_setting directory)
    export MORPHIK_BACKUP_DIR="${BACKUP_DIR:-./backups}"
    mkdir -p "$MORPHIK_BACKUP_DIR"
    chmod 700 "$MORPHIK_BACKUP_DIR"
    export MORPHIK_BACKUP_OWNER="$(id -u):$(id -g)"
    BACKUP_PROFILES="--profile backup"
    if [ -n "$(backup_setting s3_uri)" ]; then
        BACKUP_PROFILES="$BACKUP_PROFILES --profile backup-s3"
    fi
fi

print_info "Starting Morphik with port ${MORPHIK_API_PORT}..."
docker compose -f "$COMPOSE_FILE" $UI_PROFILE $BACKUP_PROFILES up -d --remove-orphans

print_success "🚀 Morphik is running!"
print_info "🌐 API endpoints:"
print_info "   Health check: http://localhost:${MORPHIK_API_PORT}/health"
print_info "   API docs:     http://localhost:${MORPHIK_API_PORT}/docs"
print_info "   Main API:     http://localhost:${MORPHIK_API_PORT}"
if [ -n "$BACKUP_PROFILES" ]; then
    print_info "💾 Scheduled backups are on. Files go to ${MORPHIK_BACKUP_DIR}."
else
    print_info "💾 Back up before upgrades or experiments: ./morphik-backup.sh backup"
fi

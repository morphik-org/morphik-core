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
    MORPHIK_VERSION=$(grep "^MORPHIK_VERSION=" .env 2>/dev/null | tail -n1 | cut -d= -f2-)
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

PROFILE_FLAGS=()
if [ -f ".env" ] && grep -q "^COMPOSE_PROFILES=" .env; then
    PROFILES=$(grep "^COMPOSE_PROFILES=" .env | tail -n1 | cut -d= -f2-)
    IFS=',' read -r -a PROFILE_ARRAY <<< "$PROFILES"
    for profile in "${PROFILE_ARRAY[@]}"; do
        profile=$(echo "$profile" | xargs)
        if [ -n "$profile" ]; then
            PROFILE_FLAGS+=("--profile" "$profile")
        fi
    done
elif [ -f ".env" ] && grep -q "^UI_INSTALLED=true" .env; then
    PROFILE_FLAGS+=("--profile" "ui")
fi

print_info "Starting Morphik with port ${MORPHIK_API_PORT}..."
docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" up -d --remove-orphans

print_success "🚀 Morphik is running!"
print_info "🌐 API endpoints:"
print_info "   Health check: http://localhost:${MORPHIK_API_PORT}/health"
print_info "   API docs:     http://localhost:${MORPHIK_API_PORT}/docs"
print_info "   Main API:     http://localhost:${MORPHIK_API_PORT}"

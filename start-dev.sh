#!/bin/bash
set -e

# Purpose: Development startup script for Morphik
# This script reads the port from morphik.toml and dynamically updates docker-compose.yml
# port mapping before starting services. This ensures developers can change ports in
# morphik.toml without manually editing docker-compose.yml
# Usage: ./start-dev.sh [docker-compose options]

# Color output functions
print_info() {
    echo -e "\033[34mℹ️  $1\033[0m"
}

print_success() {
    echo -e "\033[32m✅ $1\033[0m"
}

print_error() {
    echo -e "\033[31m❌ $1\033[0m"
}

# Read port from morphik.toml
API_PORT=$(awk '/^\[api\]/{flag=1; next} /^\[/{flag=0} flag && /^port[[:space:]]*=/ {gsub(/^port[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || echo "8000")
print_info "Detected port $API_PORT from morphik.toml"

# Create a temporary docker-compose with the correct port mapping
cp docker-compose.yml docker-compose.yml.tmp

# Update port mapping in the temporary file
sed -i.bak "s|\"8000:8000\"|\"${API_PORT}:${API_PORT}\"|g" docker-compose.yml.tmp
rm -f docker-compose.yml.tmp.bak

print_success "Updated port mapping to ${API_PORT}:${API_PORT}"

# Start the services using the temporary compose file
print_info "Starting Morphik development environment..."
docker compose -f docker-compose.yml.tmp up "$@"

# Cleanup temporary file on exit
trap "rm -f docker-compose.yml.tmp" EXIT

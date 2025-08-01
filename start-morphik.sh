#!/bin/bash
set -e

# Purpose: Production startup script for Morphik (created by install_docker.sh)
# This script reads the port from morphik.toml and dynamically updates docker-compose.run.yml
# port mapping if it has changed. This allows users to change ports in morphik.toml
# without manually editing docker-compose.run.yml after installation.
# Usage: ./start-morphik.sh

# Color output functions
print_info() {
    echo -e "\033[34mℹ️  $1\033[0m"
}

print_success() {
    echo -e "\033[32m✅ $1\033[0m"
}

# Read port from morphik.toml
API_PORT=$(awk '/^\[api\]/{flag=1; next} /^\[/{flag=0} flag && /^port[[:space:]]*=/ {gsub(/^port[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || echo "8000")

# Check if docker-compose.run.yml exists
if [ ! -f "docker-compose.run.yml" ]; then
    print_error "docker-compose.run.yml not found. Please run the install script first."
    exit 1
fi

# Create temporary compose file with updated port
cp docker-compose.run.yml docker-compose.run.yml.tmp
sed -i.bak "s|\"8000:8000\"|\"${API_PORT}:${API_PORT}\"|g" docker-compose.run.yml.tmp
rm -f docker-compose.run.yml.tmp.bak

print_info "Starting Morphik with port ${API_PORT}..."
docker compose -f docker-compose.run.yml.tmp up -d

print_success "🚀 Morphik is running!"
print_info "🌐 API endpoints:"
print_info "   Health check: http://localhost:${API_PORT}/health"
print_info "   API docs:     http://localhost:${API_PORT}/docs"
print_info "   Main API:     http://localhost:${API_PORT}"

# Cleanup temp file
rm -f docker-compose.run.yml.tmp

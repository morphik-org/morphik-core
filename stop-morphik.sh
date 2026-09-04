#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker-compose.run.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "docker-compose.run.yml not found. Run this script from the Morphik install directory." >&2
    exit 1
fi

# Activate every profile so `down` also removes optional UI and Ollama containers.
# Do not add --volumes here. PostgreSQL, Redis, model, and UI data must survive a
# routine stop/start cycle.
docker compose -f "$COMPOSE_FILE" --profile "*" down --remove-orphans
echo "Morphik services stopped. Persistent named volumes were preserved."

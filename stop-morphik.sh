#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ ! -f "$SCRIPT_DIR/morphik-compose-project.sh" ]]; then
    echo "morphik-compose-project.sh not found. Restore it from the Morphik release before stopping." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/morphik-compose-project.sh"

COMPOSE_FILE="docker-compose.run.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "docker-compose.run.yml not found. Run this script from the Morphik install directory." >&2
    exit 1
fi

# Activate every profile so `down` also removes optional UI and Ollama containers.
# Do not add --volumes here. PostgreSQL, Redis, model, and UI data must survive a
# routine stop/start cycle.
morphik_compose_resolve_existing_project
docker compose -f "$COMPOSE_FILE" --profile "*" down --remove-orphans
echo "Morphik services stopped. Persistent named volumes were preserved."

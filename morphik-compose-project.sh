#!/bin/bash

# Recover the Compose project attached to an existing Morphik deployment.
# Compose derives its default project from the install directory, so moving the
# directory can otherwise strand the old containers and Postgres volume.
morphik_compose_resolve_existing_project() {
    if [[ -n "${COMPOSE_PROJECT_NAME:-}" ]]; then
        return
    fi

    local configured_project=""
    if [[ -f .env ]]; then
        configured_project=$(sed -n 's/^COMPOSE_PROJECT_NAME=//p' .env 2>/dev/null | tail -n1)
    fi

    local container_project
    container_project=$(docker inspect morphik-postgres \
        --format '{{ index .Config.Labels "com.docker.compose.project" }}' 2>/dev/null || true)
    if [[ -n "$container_project" && "$container_project" != "<no value>" ]]; then
        export COMPOSE_PROJECT_NAME="$container_project"
        echo "[INFO] Using existing Docker Compose project '$COMPOSE_PROJECT_NAME'."
        return
    fi

    local current_project
    current_project=$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_-]+//g; s/^[^a-z0-9]+//')

    local candidate_project=""
    local multiple_projects=false
    local volume
    while IFS= read -r volume; do
        [[ -z "$volume" ]] && continue

        local volume_project
        volume_project=$(docker volume inspect "$volume" \
            --format '{{ index .Labels "com.docker.compose.project" }}' 2>/dev/null || true)
        if [[ -z "$volume_project" || "$volume_project" == "<no value>" ]]; then
            continue
        fi

        if [[ -n "$configured_project" && "$volume_project" == "$configured_project" ]]; then
            export COMPOSE_PROJECT_NAME="$volume_project"
            return
        fi

        if [[ "$volume_project" == "$current_project" ]]; then
            export COMPOSE_PROJECT_NAME="$volume_project"
            echo "[INFO] Using existing Docker Compose project '$COMPOSE_PROJECT_NAME'."
            return
        fi

        if [[ -z "$candidate_project" ]]; then
            candidate_project="$volume_project"
        elif [[ "$candidate_project" != "$volume_project" ]]; then
            multiple_projects=true
        fi
    done < <(docker volume ls -q --filter label=com.docker.compose.volume=postgres_data 2>/dev/null)

    if [[ "$multiple_projects" == true ]]; then
        echo "Multiple Morphik Postgres volumes were found. Set COMPOSE_PROJECT_NAME in .env to the project you want to use; no containers or volumes were changed." >&2
        return 1
    fi

    if [[ -n "$candidate_project" ]]; then
        export COMPOSE_PROJECT_NAME="$candidate_project"
        echo "[INFO] Using existing Docker Compose project '$COMPOSE_PROJECT_NAME'."
    elif [[ -n "$configured_project" ]]; then
        export COMPOSE_PROJECT_NAME="$configured_project"
    fi
}

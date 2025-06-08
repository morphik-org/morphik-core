#!/bin/bash
set -e

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Globals ---
SETUP_MARKER=".morphik_setup"
PLATFORM="unknown"
SETUP_TYPE="cloud" # Default to cloud
NO_UI=false
SKIP_VECTOR_CHECK=false

# --- Utility Functions ---
print_info() { echo -e "${BLUE}ℹ ${1}${NC}"; }
print_success() { echo -e "${GREEN}✓ ${1}${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ ${1}${NC}"; }
print_error() { echo -e "${RED}✗ ${1}${NC}"; }

# --- Prerequisite Checks ---
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

check_docker_compose() {
    if ! docker compose version > /dev/null 2>&1; then
        print_error "Docker Compose is not installed or not in your PATH."
        exit 1
    fi
}

detect_platform() {
    case "$(uname -s)" in
        Darwin*)    PLATFORM="mac";;
        Linux*)     PLATFORM="linux";;
        *)          PLATFORM="other";;
    esac
}

# --- First-Time Setup ---
run_first_time_setup() {
    echo ""
    print_info "🚀 Welcome to Morphik Setup!"
    echo "This one-time setup will configure your environment."
    echo ""
    echo "🤖 How would you like to run AI models?"
    echo "1) Cloud (OpenAI) - Recommended, requires an API key"
    echo "2) Local (Ollama) - Runs on your computer, requires Ollama"
    echo ""

    local choice
    while true; do
        read -p "Choose option (1 or 2): " choice
        case $choice in
            1)
                SETUP_TYPE="cloud"
                break
                ;;
            2)
                SETUP_TYPE="local"
                break
                ;;
            *)
                echo "Please enter 1 or 2"
                ;;
        esac
    done

    # Create .env file
    print_info "Creating .env file..."
    local JWT_SECRET=$(openssl rand -hex 32 2>/dev/null || date +%s)
    cat > .env << EOL
JWT_SECRET_KEY=${JWT_SECRET}
POSTGRES_URI=postgresql+asyncpg://morphik:morphik@postgres:5432/morphik
PGPASSWORD=morphik
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
REDIS_HOST=redis
REDIS_PORT=6379
EOL

    if [[ "$SETUP_TYPE" == "cloud" ]]; then
        print_info "To use cloud models, you need an OpenAI API key."
        print_info "You can get one here: https://platform.openai.com/api-keys"
        local OPENAI_KEY
        read -p "Enter your OpenAI API key: " OPENAI_KEY
        if [[ -n "$OPENAI_KEY" ]]; then
            echo "OPENAI_API_KEY=${OPENAI_KEY}" >> .env
            print_success "OpenAI API key saved to .env file."
        else
            print_error "No API key provided. Setup cannot continue."
            print_info "Please run ./start.sh again and provide a key."
            rm .env # Clean up incomplete .env
            exit 1
        fi
    fi

    # Create the setup marker file
    echo "$SETUP_TYPE" > "$SETUP_MARKER"
    print_success "Setup complete! Your choice ($SETUP_TYPE) has been saved."
    echo ""
}

# --- Configuration Management ---
update_toml_config() {
    # Backup the original config file
    cp morphik.toml morphik.toml.bak

    # Use a simple python script for robust toml updates
    python3 -c "
import tomli
import tomli_w
import sys
import os

config_path = 'morphik.toml'
setup_type = os.getenv('SETUP_TYPE', 'cloud')
platform = os.getenv('PLATFORM', 'linux')

with open(config_path, 'rb') as f:
    config = tomli.load(f)

# Set redis host based on docker usage
if platform == 'mac' and setup_type == 'local':
    # Native mode on mac uses localhost
    config['redis']['host'] = 'localhost'
else:
    # All docker-based setups use the 'redis' service name
    config['redis']['host'] = 'redis'

if setup_type == 'local':
    print('Configuring for LOCAL (Ollama) models...')
    config['agent']['model'] = 'ollama_qwen_vision'
    config['completion']['model'] = 'ollama_qwen_vision'
    config['embedding']['model'] = 'ollama_embedding'
    config['embedding']['dimensions'] = 768 # nomic-embed-text
    config['parser']['contextual_chunking_model'] = 'ollama_qwen_vision'
    config['document_analysis']['model'] = 'ollama_qwen_vision'
    config['parser']['vision']['model'] = 'ollama_qwen_vision'
    config['rules']['model'] = 'ollama_qwen_vision'
    config['graph']['model'] = 'ollama_qwen_vision'

    ollama_base_url = 'http://ollama:11434'
    if platform == 'mac':
        # On Mac, morphik runs in docker and needs to talk to host's ollama
        ollama_base_url = 'http://host.docker.internal:11434'

    for model, settings in config.get('registered_models', {}).items():
        if 'ollama' in settings.get('model_name', ''):
            settings['api_base'] = ollama_base_url

else: # cloud
    print('Configuring for CLOUD (OpenAI) models...')
    config['agent']['model'] = 'openai_gpt4-1-mini'
    config['completion']['model'] = 'openai_gpt4-1-mini'
    config['embedding']['model'] = 'openai_embedding'
    config['embedding']['dimensions'] = 1536
    config['parser']['contextual_chunking_model'] = 'openai_gpt4-1-mini'
    config['document_analysis']['model'] = 'openai_gpt4-1-mini'
    config['parser']['vision']['model'] = 'openai_gpt4-1-mini'
    config['rules']['model'] = 'openai_gpt4-1-mini'
    config['graph']['model'] = 'openai_gpt4-1-mini'

with open(config_path, 'wb') as f:
    tomli_w.dump(config, f)

print('Configuration updated successfully.')
"
}


# --- Vector DB Management ---
get_configured_dimensions() {
    python3 -c "
import tomli
try:
    with open('morphik.toml', 'rb') as f:
        config = tomli.load(f)
    print(config.get('embedding', {}).get('dimensions', 0))
except Exception:
    print(0)
" 2>/dev/null || echo "0"
}

check_database_dimensions() {
    # This check is now more robust, using psql inside the container.
    # This avoids any host python dependency issues.
    local check_exists_cmd="docker compose exec -T postgres psql -U morphik -d morphik -t -c \"SELECT to_regclass('public.chunks');\" 2>/dev/null"

    # The tr command removes leading/trailing whitespace which psql can add.
    local table_exists=$(eval $check_exists_cmd | tr -d '[:space:]')

    if [[ -z "$table_exists" ]]; then
        echo "0:no-table"
        return
    fi

    local query="SELECT vector_dims(embedding) FROM chunks LIMIT 1;"
    local result=$(docker compose exec -T postgres psql -U morphik -d morphik -t -c "$query" 2>/dev/null | tr -d '[:space:]')

    if [[ -n "$result" ]]; then
        echo "$result:existing"
    else
        # If the query returns nothing, the table is empty.
        echo "0:empty"
    fi
}

handle_vector_mismatch() {
    print_info "Checking vector dimensions..."
    local config_dims=$(get_configured_dimensions)

    # Wait for postgres to be ready
    print_info "Waiting for PostgreSQL container..."
    retries=0
    until docker compose ps postgres | grep -q "healthy"; do
        sleep 3
        ((retries++))
        if [ $retries -gt 15 ]; then
            print_error "PostgreSQL container did not become healthy in time."
            exit 1
        fi
    done
    print_success "PostgreSQL is ready."

    local db_result=$(check_database_dimensions)
    local db_dims=$(echo "$db_result" | cut -d':' -f1)
    local db_status=$(echo "$db_result" | cut -d':' -f2)

    if [[ "$db_status" == "existing" && "$db_dims" != "$config_dims" ]]; then
        print_error "FATAL: Vector dimension mismatch detected!"
        echo "  - Database vectors: $db_dims dimensions"
        echo "  - Configured vectors: $config_dims dimensions"
        echo ""
        print_warning "This requires deleting all existing vector data."
        print_info "To resolve this, run: ./start.sh --reset-vectors"
        exit 1
    elif [[ "$db_status" == "error" || "$db_status" == "connection-error" ]]; then
        print_warning "Could not verify vector dimensions in the database."
    else
        print_success "Vector dimensions match ($config_dims)."
    fi
}

clear_vector_tables() {
    print_warning "This will DELETE all vector data (chunks and multi-vector embeddings)."
    read -p "Are you sure you want to continue? (y/n): " choice
    if [[ "$choice" != "y" ]]; then
        print_info "Aborted by user."
        exit 0
    fi

    print_info "Connecting to database to clear vector tables..."
    docker compose exec -T postgres psql -U morphik -d morphik -c "DROP TABLE IF EXISTS chunks, multivector_embeddings, vector_embeddings CASCADE;"
    print_success "Vector tables cleared. You can now start Morphik normally."
    exit 0
}


# --- Service Management ---
start_dependencies() {
    print_info "Starting background services (Postgres, Redis)..."
    docker compose up -d postgres redis

    print_info "Waiting for PostgreSQL container..."
    retries=0
    until docker compose ps postgres | grep -q "healthy"; do
        sleep 3
        ((retries++))
        if [ $retries -gt 15 ]; then
            print_error "PostgreSQL container did not become healthy in time."
            exit 1
        fi
    done
    print_success "PostgreSQL is ready."
}

start_application() {
    print_info "Starting Morphik application services..."
    local compose_profiles=()
    if [[ "$SETUP_TYPE" == "local" ]]; then
        compose_profiles+=(--profile ollama)
        print_info "Ollama profile enabled."
    fi

    # The --no-deps flag prevents it from touching postgres/redis again
    # The --build flag is here to catch any new changes
    docker compose "${compose_profiles[@]}" build morphik worker
    docker compose "${compose_profiles[@]}" up -d --no-deps morphik worker
}

handle_mac_local_setup() {
    print_warning "macOS Local Setup Detected!"
    echo "For the best performance (with GPU acceleration), Morphik should be run natively."
    print_info "This script will start the required Docker containers (Postgres, Redis),"
    print_info "but you will need to start the Python server manually in a separate terminal."
    echo ""
    print_info "Instructions:"
    echo "1. Ensure you have installed Ollama.app and it is running."
    echo "2. Ensure you have pulled the required models:"
    echo "   - ollama pull nomic-embed-text"
    echo "   - ollama pull qwen2.5vl:latest"
    echo "3. In a new terminal, run: python3 start_server.py"
    echo ""
    read -p "Press Enter to start the background services (Postgres, Redis)..."

    # Bring down everything first to be safe
    docker compose down --remove-orphans >/dev/null 2>&1

    # Update config for native run
    update_toml_config

    # Start only core db/cache services, no morphik app container
    docker compose up -d postgres redis
    print_success "Background services started. Please start the server manually."
    exit 0
}


# --- UI Management ---
start_ui() {
    # Placeholder for UI start logic if it's a separate process
    return 0
}
stop_ui() {
    # Placeholder for UI stop logic
    return 0
}

# --- Main Execution ---
show_status() {
    echo ""
    print_success "🎉 Morphik is running!"
    print_info "API available at: http://localhost:8000"
    print_info "API docs at: http://localhost:8000/docs"
    echo ""
    print_info "View logs: docker compose logs -f"
    print_info "Stop services: docker compose down"
    echo ""
}

show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "A smart script to set up and run Morphik."
    echo ""
    echo "Commands:"
    echo "  up (default)         Starts Morphik. Runs first-time setup if needed."
    echo "  down                 Stops all Morphik services."
    echo "  logs                 Follows the logs of running services."
    echo "  --reset-vectors      Deletes all vector data. Use with caution."
    echo "  --help               Shows this help message."
    echo ""
}

main() {
    # Command parsing
    if [[ "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    if [[ "$1" == "--reset-vectors" ]]; then
        check_docker && check_docker_compose
        clear_vector_tables
    fi

    local cmd="up"
    if [[ -n "$1" ]]; then
        cmd="$1"
    fi

    # Execute command
    case "$cmd" in
        up)
            check_docker && check_docker_compose
            create_directories

            if [ ! -f "$SETUP_MARKER" ]; then
                run_first_time_setup
            fi

            SETUP_TYPE=$(cat "$SETUP_MARKER")
            print_info "Starting Morphik in '$SETUP_TYPE' mode."

            detect_platform
            export SETUP_TYPE PLATFORM

            if [[ "$SETUP_TYPE" == "local" && "$PLATFORM" == "mac" ]]; then
                handle_mac_local_setup
                exit 0
            fi

            # For all standard docker-based setups
            docker compose down --remove-orphans >/dev/null 2>&1
            update_toml_config
            start_dependencies

            if [[ "$SKIP_VECTOR_CHECK" != true ]]; then
                handle_vector_mismatch
            fi

            start_application
            [[ "$NO_UI" == false ]] && start_ui
            show_status
            ;;
        down)
            check_docker && check_docker_compose
            stop_ui
            docker compose down --remove-orphans
            print_success "Morphik stopped."
            ;;
        logs)
            check_docker && check_docker_compose
            docker compose logs -f
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

create_directories() {
    mkdir -p storage logs
}

# Run main
main "$@"

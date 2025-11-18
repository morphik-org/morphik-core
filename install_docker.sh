#!/bin/bash
set -e

# Purpose: End-user installation script for Morphik
# This script downloads docker-compose.run.yml, creates .env file, downloads morphik.toml,
# and starts all services. Creates start-morphik.sh for easy restarts with automatic
# port detection from morphik.toml configuration.
# Usage: curl -sSL https://install.morphik.ai | bash

# --- Configuration ---
REPO_URL="https://raw.githubusercontent.com/morphik-org/morphik-core/main"
REPO_ARCHIVE_URL="https://codeload.github.com/morphik-org/morphik-core/tar.gz/refs/heads/main"
COMPOSE_FILE="docker-compose.run.yml"
DIRECT_INSTALL_URL="https://www.morphik.ai/docs/getting-started#self-host-direct-installation-advanced"

EMBEDDING_PROVIDER=""
EMBEDDING_PROVIDER_LABEL=""

# --- Helper Functions ---
print_info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

print_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1" >&2
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "'$1' is not installed. Please install it to continue."
    fi
}

set_env_value() {
    local key="$1"
    local value="$2"
    local env_file=".env"

    if [ -f "$env_file" ] && grep -q "^${key}=" "$env_file"; then
        local tmp_file
        tmp_file=$(mktemp "${env_file}.XXXXXX") || {
            print_error "Failed to create temporary file while updating ${key}"
            return 1
        }
        grep -v "^${key}=" "$env_file" > "$tmp_file" || true
        mv "$tmp_file" "$env_file" || {
            rm -f "$tmp_file"
            print_error "Failed to update ${env_file} while setting ${key}"
            return 1
        }
    elif [ ! -f "$env_file" ]; then
        touch "$env_file"
    fi

    echo "${key}=${value}" >> "$env_file"
}

ensure_compose_profile() {
    local profile="$1"

    if grep -q "^COMPOSE_PROFILES=" .env 2>/dev/null; then
        local current
        current=$(grep "^COMPOSE_PROFILES=" .env | tail -n1 | cut -d= -f2-)
        if [[ -z "$current" ]]; then
            set_env_value "COMPOSE_PROFILES" "$profile"
            return
        fi

        if [[ ",${current}," != *",${profile},"* ]]; then
            set_env_value "COMPOSE_PROFILES" "${current},${profile}"
        fi
    else
        set_env_value "COMPOSE_PROFILES" "$profile"
    fi
}

copy_ui_from_image() {
    local tmp_container
    tmp_container=$(docker create ghcr.io/morphik-org/morphik-core:latest 2>/dev/null) || return 1

    mkdir -p ee
    rm -rf ee/ui-component

    if docker cp "$tmp_container:/app/ee/ui-component" ee/ui-component >/dev/null 2>&1; then
        docker rm "$tmp_container" >/dev/null 2>&1 || true
        return 0
    fi

    docker rm "$tmp_container" >/dev/null 2>&1 || true
    rm -rf ee/ui-component
    return 1
}

download_ui_from_repo() {
    local tmpdir
    tmpdir=$(mktemp -d 2>/dev/null) || return 1

    if curl -fsSL "$REPO_ARCHIVE_URL" | tar -xz -C "$tmpdir"; then
        local extracted_dir
        extracted_dir=$(find "$tmpdir" -maxdepth 1 -mindepth 1 -type d -name "morphik-org-morphik-core*" | head -1)
        if [[ -n "$extracted_dir" && -d "$extracted_dir/ee/ui-component" ]]; then
            mkdir -p ee
            rm -rf ee/ui-component
            cp -R "$extracted_dir/ee/ui-component" ee/
            rm -rf "$tmpdir"
            return 0
        fi
    fi

    rm -rf "$tmpdir"
    return 1
}

# --- Main Script ---

# 1. Check for prerequisites
print_info "Checking for Docker and Docker Compose..."
check_command "docker"
if ! docker compose version &> /dev/null; then
    print_error "Docker Compose V2 is required. Please ensure it's installed and accessible."
fi
print_success "Prerequisites are satisfied."

# 2. Apple Silicon Warning
if [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    print_warning "You are on an Apple Silicon Mac (arm64)."
    print_warning "For best performance (including GPU access), we strongly recommend the Direct Installation method."
    print_warning "You can find the guide here: $DIRECT_INSTALL_URL"
    read -p "Do you want to continue with the Docker installation anyway? (y/N): " choice < /dev/tty
    if [[ "$choice" != "y" && "$choice" != "Y" ]]; then
        echo "Installation aborted by user."
        exit 0
    fi
fi

# 3. Download necessary files
print_info "Downloading the Docker Compose configuration file..."
if curl -fsSL -o "$COMPOSE_FILE" "$REPO_URL/$COMPOSE_FILE"; then
    print_success "Downloaded '$COMPOSE_FILE'."
else
    print_error "Failed to download '$COMPOSE_FILE'. Please check your internet connection and the repository URL."
fi

# 4. Create .env and get User Input for API Key
print_info "Creating '.env' file for your secrets..."
cat > .env <<EOF
# Your OpenAI API key (optional - you can configure other providers in morphik.toml)
OPENAI_API_KEY=

# A secret key for signing JWTs. A random one is generated for you.
JWT_SECRET_KEY=your-super-secret-key-that-is-long-and-random-$(openssl rand -hex 16)

# Local URI token for secure URI generation (required for creating connection URIs)
LOCAL_URI_TOKEN=
EOF

print_info "Morphik supports 100s of models including OpenAI, Anthropic (Claude), Google Gemini, local models, and even custom models!"
read -p "Please enter your OpenAI API Key (or press Enter to skip and configure later): " openai_api_key < /dev/tty
if [[ -z "$openai_api_key" ]]; then
    print_warning "No OpenAI API key provided. You can add it later to .env or configure other providers in morphik.toml"
    print_info "Embeddings power ingestion, search, and querying in Morphik. Please choose an alternative provider to continue."
    while true; do
        echo ""
        echo "Select an embeddings provider:"
        echo "  1) Lemonade (download at https://lemonade-server.ai/)"
        echo "  2) Ollama (download at https://ollama.com/)"
        read -p "Enter 1 or 2 [2]: " embedding_choice < /dev/tty
        embedding_choice=${embedding_choice:-2}
        case "$embedding_choice" in
            1)
                EMBEDDING_PROVIDER="lemonade_embedding"
                EMBEDDING_PROVIDER_LABEL="Lemonade embeddings"
                break
                ;;
            2)
                EMBEDDING_PROVIDER="ollama_embedding"
                EMBEDDING_PROVIDER_LABEL="Ollama embeddings"
                break
                ;;
            *)
                print_warning "Please enter 1 or 2."
                ;;
        esac
    done
    print_info "Embeddings will be configured to use $EMBEDDING_PROVIDER_LABEL with 768 dimensions."
else
    # Use sed to safely replace the key in the .env file.
    sed -i.bak "s|OPENAI_API_KEY=|OPENAI_API_KEY=$openai_api_key|" .env
    rm -f .env.bak
    print_success "'.env' file has been configured with your API key."
fi

# 5. Download and setup configuration FIRST (before trying to modify it)
print_info "Setting up configuration file..."

# Pull the Docker image first if needed
print_info "Pulling Docker image if not already available..."
if ! docker pull ghcr.io/morphik-org/morphik-core:latest; then
    print_error "Failed to pull Docker image 'ghcr.io/morphik-org/morphik-core:latest'"
    print_info "Possible reasons:"
    print_info "  - The image hasn't been published to GitHub Container Registry yet"
    print_info "  - Network/firewall is blocking access to ghcr.io"
    print_info "  - Docker daemon is not running properly"
    print_info ""
    print_info "Attempting to download configuration from repository instead..."

    # Try to download morphik.docker.toml (Docker-specific config) first
    if curl -fsSL -o morphik.toml "$REPO_URL/morphik.docker.toml" 2>/dev/null; then
        print_success "Downloaded Docker-specific configuration from repository."
    elif curl -fsSL -o morphik.toml "$REPO_URL/morphik.toml" 2>/dev/null; then
        print_warning "Downloaded standard morphik.toml (may need adjustments for Docker)."
    else
        print_error "Could not download configuration file. Installation cannot continue."
        exit 1
    fi
else
    print_success "Docker image is available."
    print_info "Extracting default 'morphik.toml' for you to customize..."

    # Method 1: Try using docker run with output capture (more reliable on Windows)
    CONFIG_CONTENT=$(docker run --rm ghcr.io/morphik-org/morphik-core:latest cat /app/morphik.toml.default 2>/dev/null)
    if [ -n "$CONFIG_CONTENT" ]; then
        echo "$CONFIG_CONTENT" > morphik.toml
        if [ -f morphik.toml ] && [ -s morphik.toml ]; then
            print_success "Extracted configuration from Docker image."
        else
            print_warning "Failed to write configuration file."
            CONFIG_EXTRACTED=false
        fi
    else
        CONFIG_EXTRACTED=false
    fi

    # Method 2: If Method 1 failed, try docker cp approach
    if [ "$CONFIG_EXTRACTED" = "false" ] 2>/dev/null || [ ! -f morphik.toml ]; then
        print_info "Trying alternative extraction method..."
        TEMP_CONTAINER=$(docker create ghcr.io/morphik-org/morphik-core:latest)
        if docker cp "$TEMP_CONTAINER:/app/morphik.toml.default" morphik.toml 2>/dev/null; then
            docker rm "$TEMP_CONTAINER" >/dev/null 2>&1
            print_success "Extracted configuration using docker cp."
        else
            docker rm "$TEMP_CONTAINER" >/dev/null 2>&1
            print_warning "Could not extract morphik.toml from Docker image."
        fi
    fi

    # Method 3: If still no file, download from repository
    if [ ! -f morphik.toml ] || [ ! -s morphik.toml ]; then
        print_info "Downloading from repository instead..."

        # Try with curl first
        print_info "Attempting to download: $REPO_URL/morphik.docker.toml"
        if curl -fsSL "$REPO_URL/morphik.docker.toml" -o morphik.toml; then
            if [ -f morphik.toml ] && [ -s morphik.toml ]; then
                print_success "Downloaded Docker-specific configuration from repository."
            else
                rm -f morphik.toml
                print_warning "Downloaded file was empty."
            fi
        fi

        # If still no file, try the standard morphik.toml
        if [ ! -f morphik.toml ] || [ ! -s morphik.toml ]; then
            print_info "Attempting to download: $REPO_URL/morphik.toml"
            if curl -fsSL "$REPO_URL/morphik.toml" -o morphik.toml; then
                if [ -f morphik.toml ] && [ -s morphik.toml ]; then
                    print_warning "Downloaded standard morphik.toml (may need Docker adjustments)."
                else
                    rm -f morphik.toml
                    print_error "Could not obtain a valid configuration file."
                    exit 1
                fi
            else
                print_error "Could not download configuration file from repository."
                print_info "Please check your internet connection and that the repository is accessible."
                exit 1
            fi
        fi
    fi
fi

# 5.0 Configure embeddings when OpenAI key is not provided
if [[ -n "$EMBEDDING_PROVIDER" ]]; then
    if [ -f morphik.toml ]; then
        print_info "Configuring morphik.toml to use $EMBEDDING_PROVIDER_LABEL (768 dimensions)..."
        sed -i.bak \
            -e "/^\\[embedding\\]/,/^\\[/ s/^[[:space:]]*model[[:space:]]*=.*/model = \"$EMBEDDING_PROVIDER\"  # Reference to registered model/" \
            -e "/^\\[embedding\\]/,/^\\[/ s/^[[:space:]]*dimensions[[:space:]]*=.*/dimensions = 768/" \
            morphik.toml
        rm -f morphik.toml.bak
    else
        print_warning "morphik.toml not found. Skipping embedding configuration update."
    fi
fi

if [[ -n "$EMBEDDING_PROVIDER" && "$EMBEDDING_PROVIDER" == "lemonade_embedding" ]]; then
    print_warning "Ensure the Lemonade SDK is installed and running (see Lemonade installation prompt later in this script)."
fi

# 5.0.5 Now that morphik.toml exists, handle LOCAL_URI_TOKEN configuration
echo ""
print_info "🔐 Setting up authentication for your Morphik deployment:"
print_info "   • If you plan to access Morphik from outside this server, setting a LOCAL_URI_TOKEN will secure your deployment"
print_info "   • For local-only access, you can skip this step (bypass_auth_mode will be enabled)"
print_info "   • With a LOCAL_URI_TOKEN set, you'll need to use /generate_local_uri endpoint for authorization tokens"
echo ""
read -p "Please enter a secure LOCAL_URI_TOKEN (or press Enter to skip for local-only access): " local_uri_token < /dev/tty
if [[ -z "$local_uri_token" ]]; then
    print_info "No LOCAL_URI_TOKEN provided - enabling authentication bypass (bypass_auth_mode=true) for local access"
    print_info "This is suitable for local development and testing"
    # Enable bypass_auth_mode in morphik.toml (now that the file exists!)
    if [ -f morphik.toml ]; then
        sed -i.bak 's/bypass_auth_mode = false/bypass_auth_mode = true/' morphik.toml
        rm -f morphik.toml.bak
    else
        print_warning "morphik.toml not found, cannot set bypass_auth_mode"
    fi
else
    print_success "LOCAL_URI_TOKEN set - keeping production mode (bypass_auth_mode=false) with authentication enabled"
    print_info "Use the /generate_local_uri endpoint with this token to create authorized connection URIs"
fi

# Only update .env if a token was provided
if [[ -n "$local_uri_token" ]]; then
    # Use sed to safely replace the token in the .env file.
    sed -i.bak "s|LOCAL_URI_TOKEN=|LOCAL_URI_TOKEN=$local_uri_token|" .env
    rm -f .env.bak
    print_success "'.env' file has been configured with your LOCAL_URI_TOKEN."
fi

# 5.1 Ask about local inference with Lemonade (Windows only)
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null || [ -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
    echo ""
    print_info "🍋 Detected WSL environment. Morphik supports local inference with Lemonade SDK (Windows only)."
    print_info "   Lemonade provides high-performance local LLM inference with AMD GPU/NPU optimization."
    print_info "   This allows you to run both embeddings and completions completely locally."
    echo ""
    read -p "Would you like to install Lemonade SDK for local inference? (y/N): " install_lemonade < /dev/tty

    if [[ "$install_lemonade" == "y" || "$install_lemonade" == "Y" ]]; then
        print_info "Downloading Lemonade installer..."

        # Download and run the Lemonade installer
        if curl -fsSL -o lemonade-installer.sh "$REPO_URL/lemonade-installer.sh"; then
            chmod +x lemonade-installer.sh
            if ./lemonade-installer.sh; then
                LEMONADE_INSTALLED=true
                print_success "Lemonade installation completed!"
            else
                print_warning "Lemonade installation failed. You can retry later by running: ./lemonade-installer.sh"
            fi
        else
            print_warning "Failed to download Lemonade installer"
            print_info "You can manually install Lemonade later with:"
            print_info "  curl -sSL https://raw.githubusercontent.com/morphik-org/morphik-core/main/lemonade-installer.sh | bash"
        fi
    fi
fi

# 5.2 Ask about GPU availability for multimodal embeddings
echo ""
print_info "🚀 Morphik achieves ultra-accurate document understanding through advanced multimodal embeddings."
print_info "   These embeddings excel at processing images, PDFs, and complex layouts."
print_info "   While Morphik will work without a GPU, for best results we recommend using a GPU-enabled machine."
echo ""
read -p "Do you have a GPU available for Morphik to use? (y/N): " has_gpu < /dev/tty

if [[ "$has_gpu" != "y" && "$has_gpu" != "Y" ]]; then
    print_warning "Disabling multimodal embeddings and reranking since no GPU is available."
    print_info "Morphik will still work great with text-based embeddings!"
    print_info "You can enable multimodal embeddings and reranking later if you add GPU support."
    # Disable ColPali in morphik.toml
    sed -i.bak 's/enable_colpali = true/enable_colpali = false/' morphik.toml
    rm -f morphik.toml.bak
    # Ensure reranking is disabled in morphik.toml
    sed -i.bak 's/use_reranker = .*/use_reranker = false/' morphik.toml
    rm -f morphik.toml.bak
    print_success "Configuration updated for CPU-only operation."
else
    print_success "Excellent! Multimodal embeddings will be enabled for maximum accuracy."
    print_info "Make sure your Docker setup has GPU passthrough configured if using NVIDIA GPUs."
fi

print_info "Enabling configuration mounting in '$COMPOSE_FILE'..."
# Use sed to uncomment the volume mount lines for both services
sed -i.bak 's|# - ./morphik.toml:/app/morphik.toml:ro|- ./morphik.toml:/app/morphik.toml:ro|g' "$COMPOSE_FILE"
rm -f ${COMPOSE_FILE}.bak

print_success "Configuration has been set up at 'morphik.toml'."
print_info "You can edit this file to customize models, ports, or other settings."
read -p "Press [Enter] to continue with the current configuration or edit 'morphik.toml' in another terminal first..." < /dev/tty

# Update port mapping in docker-compose.run.yml to match morphik.toml
API_PORT=$(awk '/^\[api\]/{flag=1; next} /^\[/{flag=0} flag && /^port[[:space:]]*=/ {gsub(/^port[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || echo "8000")
sed -i.bak "s|\"8000:8000\"|\"${API_PORT}:${API_PORT}\"|g" "$COMPOSE_FILE"
rm -f ${COMPOSE_FILE}.bak

# 5.5. Ask about UI installation
echo ""
print_info "Morphik includes an admin UI for easier interaction."
read -p "Would you like to install the Admin UI? (y/N): " install_ui < /dev/tty

UI_PROFILE=""
if [[ "$install_ui" == "y" || "$install_ui" == "Y" ]]; then
    print_info "Extracting UI component files from Docker image..."

    if copy_ui_from_image; then
        print_success "UI component copied from Docker image."
    else
        print_warning "Failed to copy UI component from Docker image. Attempting to download from repository..."
        if download_ui_from_repo; then
            print_success "UI component downloaded from repository."
        fi
    fi

    if [ -d "ee/ui-component" ]; then
        UI_PROFILE="--profile ui"
        ensure_compose_profile "ui"
        set_env_value "UI_INSTALLED" "true"

        # Update NEXT_PUBLIC_API_URL to use the correct port
        sed -i.bak "s|NEXT_PUBLIC_API_URL=http://localhost:8000|NEXT_PUBLIC_API_URL=http://localhost:${API_PORT}|g" "$COMPOSE_FILE"
        rm -f ${COMPOSE_FILE}.bak
    else
        print_warning "Failed to download UI component. Continuing without UI."
    fi
fi

# 6. Start the application
print_info "Starting the Morphik stack... This may take a few minutes for the first run."
docker compose -f "$COMPOSE_FILE" $UI_PROFILE up -d

print_success "🚀 Morphik has been started!"
print_info "📝 Check the logs for status - it can take a few minutes to fully load"
print_info "🔄 The URL will show 'unavailable' until the service is ready"

# Read port from morphik.toml to display correct URL
API_PORT=$(awk '/^\[api\]/{flag=1; next} /^\[/{flag=0} flag && /^port[[:space:]]*=/ {gsub(/^port[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || echo "8000")

echo ""
print_info "🌐 API endpoints:"
print_info "   Health check: http://localhost:${API_PORT}/health"
print_info "   API docs:     http://localhost:${API_PORT}/docs"
print_info "   Main API:     http://localhost:${API_PORT}"

if [[ -n "$UI_PROFILE" ]]; then
    echo ""
    print_info "🎨 Admin UI:"
    print_info "   Interface:    http://localhost:3003"
    print_info "   Note: The UI may take a few minutes to build on first run"
fi

echo ""
print_info "📋 Management commands:"
print_info "   View logs:    docker compose -f $COMPOSE_FILE $UI_PROFILE logs -f"
print_info "   Stop services: ./stop-morphik.sh   # runs docker compose down --volumes --remove-orphans"
print_info "   Restart:      ./start-morphik.sh"

# Create convenience startup script
cat > start-morphik.sh << 'EOF'
#!/bin/bash
set -e

# Purpose: Production startup script for Morphik
# Automatically updates port mapping from morphik.toml and includes UI if installed

# Color functions
print_info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

API_PORT=$(awk '/^\[api\]/{flag=1; next} /^\[/{flag=0} flag && /^port[[:space:]]*=/ {gsub(/^port[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || echo "8000")
CURRENT_PORT=$(grep -oE '"[0-9]+:[0-9]+"' docker-compose.run.yml | head -1 | cut -d: -f1 | tr -d '"')

if [ "$CURRENT_PORT" != "$API_PORT" ]; then
    echo "Updating port mapping from $CURRENT_PORT to $API_PORT..."
    sed -i.bak "s|\"${CURRENT_PORT}:${CURRENT_PORT}\"|\"${API_PORT}:${API_PORT}\"|g" docker-compose.run.yml
    rm -f docker-compose.run.yml.bak
fi

# Check multimodal embeddings configuration
COLPALI_ENABLED=$(awk '/^\[morphik\]/{flag=1; next} /^\[/{flag=0} flag && /^enable_colpali[[:space:]]*=/ {gsub(/^enable_colpali[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || echo "true")

if [ "$COLPALI_ENABLED" = "false" ]; then
    print_warning "Multimodal embeddings are disabled. For best results with images/PDFs, enable them in morphik.toml if you have a GPU."
fi

# Check if UI is installed
UI_PROFILE=""
if [ -f ".env" ] && grep -q "UI_INSTALLED=true" .env; then
    UI_PROFILE="--profile ui"
fi

docker compose -f docker-compose.run.yml $UI_PROFILE up -d
echo "🚀 Morphik is running on http://localhost:${API_PORT}"
echo "   Health: http://localhost:${API_PORT}/health"
echo "   Docs:   http://localhost:${API_PORT}/docs"
if [ -n "$UI_PROFILE" ]; then
    echo ""
    echo "🎨 Admin UI: http://localhost:3003"
fi
EOF
chmod +x start-morphik.sh

cat > stop-morphik.sh << 'EOF'
#!/bin/bash
set -e

COMPOSE_FILE="docker-compose.run.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "docker-compose.run.yml not found. Run this script from the Morphik install directory."
    exit 1
fi

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
elif [ -f ".env" ] && grep -q "UI_INSTALLED=true" .env; then
    PROFILE_FLAGS+=("--profile" "ui")
fi

docker compose -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" down --volumes --remove-orphans
echo "🛑 Morphik services stopped. Containers, networks, and anonymous volumes removed."
EOF
chmod +x stop-morphik.sh

# Remind about Lemonade if installed
if [ "$LEMONADE_INSTALLED" = true ]; then
    echo ""
    print_info "🍋 Lemonade SDK has been installed! To use local inference:"
    print_info "   1. Start Lemonade Server by double-clicking start_lemonade.bat on your Desktop"
    print_info "   2. Select Lemonade models in the Morphik UI settings"
    print_info "   3. Enjoy fully local embeddings and completions!"
fi

echo ""
print_success "🎉 Enjoy using Morphik!"

#!/bin/bash
set -e

# Purpose: End-user installation script for Morphik
# This script downloads docker-compose.run.yml, creates .env file, downloads morphik.toml,
# and starts all services. Creates start-morphik.sh for easy restarts with automatic
# port detection from morphik.toml configuration.
# Usage: curl -sSL https://install.morphik.ai | bash

# --- Configuration ---
REPO_URL="https://raw.githubusercontent.com/morphik-org/morphik-core/main"
COMPOSE_FILE="docker-compose.run.yml"
DIRECT_INSTALL_URL="https://www.morphik.ai/docs/getting-started#self-host-direct-installation-advanced"

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
else
    # Use sed to safely replace the key in the .env file.
    sed -i.bak "s|OPENAI_API_KEY=|OPENAI_API_KEY=$openai_api_key|" .env
    rm -f .env.bak
    print_success "'.env' file has been configured with your API key."
fi

echo ""
print_info "🔐 Setting up authentication for your Morphik deployment:"
print_info "   • If you plan to access Morphik from outside this server, setting a LOCAL_URI_TOKEN will secure your deployment"
print_info "   • For local-only access, you can skip this step (dev_mode will be enabled)"
print_info "   • With a LOCAL_URI_TOKEN set, you'll need to use /generate_local_uri endpoint for authorization tokens"
echo ""
read -p "Please enter a secure LOCAL_URI_TOKEN (or press Enter to skip for local-only access): " local_uri_token < /dev/tty
if [[ -z "$local_uri_token" ]]; then
    print_info "No LOCAL_URI_TOKEN provided - enabling development mode (dev_mode=true) for local access"
    print_info "This is suitable for local development and testing"
    # Enable dev_mode in morphik.toml
    sed -i.bak 's/dev_mode = false/dev_mode = true/' morphik.toml
    rm -f morphik.toml.bak
else
    print_success "LOCAL_URI_TOKEN set - keeping production mode (dev_mode=false) with authentication enabled"
    print_info "Use the /generate_local_uri endpoint with this token to create authorized connection URIs"
fi

# Only update .env if a token was provided
if [[ -n "$local_uri_token" ]]; then
    # Use sed to safely replace the token in the .env file.
    sed -i.bak "s|LOCAL_URI_TOKEN=|LOCAL_URI_TOKEN=$local_uri_token|" .env
    rm -f .env.bak
    print_success "'.env' file has been configured with your LOCAL_URI_TOKEN."
fi

# 5. Download and setup configuration
print_info "Setting up configuration file..."
print_info "Extracting default 'morphik.toml' for you to customize..."
docker run --rm ghcr.io/morphik-org/morphik-core:latest \
       cat /app/morphik.toml.default > morphik.toml

# 5.1 Ask about GPU availability for multimodal embeddings
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
    # Extract the UI component from the Docker image (now included in the image)
    docker run --rm ghcr.io/morphik-org/morphik-core:latest \
           tar -czf - -C /app ee/ui-component | tar -xzf -

    if [ -d "ee/ui-component" ]; then
        print_success "UI component downloaded successfully."
        UI_PROFILE="--profile ui"

        # Update NEXT_PUBLIC_API_URL to use the correct port
        sed -i.bak "s|NEXT_PUBLIC_API_URL=http://localhost:8000|NEXT_PUBLIC_API_URL=http://localhost:${API_PORT}|g" "$COMPOSE_FILE"
        rm -f ${COMPOSE_FILE}.bak

        # Save UI installation flag for start-morphik.sh
        echo "UI_INSTALLED=true" >> .env
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
print_info "   Stop services: docker compose -f $COMPOSE_FILE $UI_PROFILE down"
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

echo ""
print_success "🎉 Enjoy using Morphik!"

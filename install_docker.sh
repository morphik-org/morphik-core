#!/bin/bash
set -e

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
# Your OpenAI API key is required
OPENAI_API_KEY=

# A secret key for signing JWTs. A random one is generated for you.
JWT_SECRET_KEY=your-super-secret-key-that-is-long-and-random-$(openssl rand -hex 16)
EOF

read -p "Please enter your OpenAI API Key (it will be saved to the .env file): " openai_api_key < /dev/tty
if [[ -z "$openai_api_key" ]]; then
    print_error "OpenAI API Key cannot be empty."
fi

# Use sed to safely replace the key in the .env file.
sed -i.bak "s|OPENAI_API_KEY=|OPENAI_API_KEY=$openai_api_key|" .env
rm -f .env.bak
print_success "'.env' file has been configured with your API key."

# 5. Ask about custom configuration
print_info "The default setup uses the standard OpenAI models."
read -p "Do you want to customize the configuration (e.g., change models) before starting? (y/N): " customize_choice < /dev/tty
if [[ "$customize_choice" == "y" || "$customize_choice" == "Y" ]]; then
    print_info "Extracting default 'morphik.toml' for you to edit..."
    docker run --rm ghcr.io/morphik-org/morphik-core:latest \
           cat /app/morphik.toml.default > morphik.toml

    print_info "The configuration has been saved to 'morphik.toml'. Please edit this file in another terminal window now."
    read -p "Press [Enter] once you have finished editing..." < /dev/tty

    print_info "Enabling custom configuration in '$COMPOSE_FILE'..."
    # Use sed to uncomment the volume mount lines for both services
    sed -i.bak 's|# - ./morphik.toml:/app/morphik.toml:ro|- ./morphik.toml:/app/morphik.toml:ro|g' "$COMPOSE_FILE"
    rm -f ${COMPOSE_FILE}.bak
    print_success "Custom configuration has been enabled."
fi

# 6. Start the application
print_info "Starting the Morphik stack... This may take a few minutes for the first run."
docker compose -f "$COMPOSE_FILE" up -d

print_success "Morphik is now running!"
print_info "API is available at http://localhost:8000"
print_info "To view logs, run: docker compose -f $COMPOSE_FILE logs -f"
print_info "To stop the services, run: docker compose -f $COMPOSE_FILE down"

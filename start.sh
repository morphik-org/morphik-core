#!/bin/bash
set -e

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if Docker Compose is installed
check_docker_compose() {
    if ! docker compose version > /dev/null 2>&1; then
        echo "Error: Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
}

# Function to check if required files exist
check_files() {
    if [ ! -f "docker-compose.yml" ]; then
        echo "Error: docker-compose.yml not found!"
        exit 1
    fi
    if [ ! -f "morphik.toml" ]; then
        echo "Error: morphik.toml not found!"
        exit 1
    fi
}

# Function to create .env file if it doesn't exist
setup_env() {
    if [ ! -f ".env" ]; then
        echo "Creating .env file with default values..."
        cat > .env << EOL
JWT_SECRET_KEY=your-secure-key-here
HOST=0.0.0.0
PORT=8000
EOL
        echo "Created .env file. Please edit it to set your JWT_SECRET_KEY and other settings."
    fi
}

# Function to create morphik.toml if it doesn't exist
create_morphik_toml() {
    if [ ! -f "morphik.docker.toml" ]; then
        echo "Error: morphik.docker.toml not found!"
        exit 1
    fi
    echo "Syncing morphik.toml with morphik.docker.toml..."
    cp -f morphik.docker.toml morphik.toml
}

# Function to create necessary directories
create_directories() {
    mkdir -p storage logs
}

# Function to configure model settings
configure_models() {
    echo "Which model provider would you like to use?"
    echo "1) Ollama (Local models - requires more resources)"
    echo "2) OpenAI/Anthropic (Cloud models - requires API keys)"
    read -p "Enter your choice (1 or 2): " model_choice

    case $model_choice in
        1)
            echo "Setting up Ollama models..."
            # Update morphik.toml for Ollama
            sed -i.bak 's/provider = "openai"/provider = "ollama"/g' morphik.toml
            sed -i.bak 's/model = "openai_gpt4-1"/model = "ollama_qwen_vision"/g' morphik.toml
            # Start with Ollama profile
            docker compose --profile ollama up --build -d
            ;;
        2)
            echo "Setting up OpenAI/Anthropic models..."
            echo "You will need API keys for OpenAI and/or Anthropic to use cloud models."
            echo "You can enter them now or add them later in the .env file."
            echo ""
            
            # Always ensure .env file exists
            if [ ! -f ".env" ]; then
                touch .env
            fi
            
            # Always ask for OpenAI API key
            read -p "Enter your OpenAI API key (press Enter to skip and add later): " openai_key
            if [ ! -z "$openai_key" ]; then
                # Remove existing OPENAI_API_KEY if it exists
                sed -i.bak '/OPENAI_API_KEY/d' .env
                echo "OPENAI_API_KEY=$openai_key" >> .env
                echo "OpenAI API key added to .env file"
            else
                # Remove existing OPENAI_API_KEY if it exists
                sed -i.bak '/OPENAI_API_KEY/d' .env
                echo "OPENAI_API_KEY=" >> .env
                echo "You can add your OpenAI API key later by editing the .env file"
            fi
            
            # Always ask for Anthropic API key
            read -p "Enter your Anthropic API key (press Enter to skip and add later): " anthropic_key
            if [ ! -z "$anthropic_key" ]; then
                # Remove existing ANTHROPIC_API_KEY if it exists
                sed -i.bak '/ANTHROPIC_API_KEY/d' .env
                echo "ANTHROPIC_API_KEY=$anthropic_key" >> .env
                echo "Anthropic API key added to .env file"
            else
                # Remove existing ANTHROPIC_API_KEY if it exists
                sed -i.bak '/ANTHROPIC_API_KEY/d' .env
                echo "ANTHROPIC_API_KEY=" >> .env
                echo "You can add your Anthropic API key later by editing the .env file"
            fi
            
            echo ""
            echo "Proceeding with Docker setup..."
            echo ""
            
            # Update morphik.toml for OpenAI
            sed -i.bak 's/provider = "ollama"/provider = "openai"/g' morphik.toml
            sed -i.bak 's/model = "ollama_qwen_vision"/model = "openai_gpt4-1"/g' morphik.toml
            # Start without Ollama profile
            docker compose up --build -d
            ;;
        *)
            echo "Invalid choice. Exiting..."
            exit 1
            ;;
    esac
}

# Main execution
echo "Starting Morphik setup..."

# Check prerequisites
check_docker
check_docker_compose
check_files
setup_env

# Create necessary files and directories
create_morphik_toml
create_directories

# Configure and start services
configure_models

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 5

# Check if services are running
if docker compose ps | grep -q "morphik.*Up"; then
    echo "✅ Morphik is running!"
    echo "You can access:"
    echo "- API Documentation: http://localhost:8000/docs"
    echo "- Health Check: http://localhost:8000/health"
else
    echo "❌ Morphik failed to start. Check the logs with:"
    echo "docker compose logs morphik"
fi
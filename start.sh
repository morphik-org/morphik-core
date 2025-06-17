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

# Function to create .env file if it doesn't exist
create_env_file() {
    if [ ! -f .env ]; then
        echo "Creating .env file..."
        cat > .env << EOL
JWT_SECRET_KEY=your-secret-key-here
POSTGRES_URI=postgresql+asyncpg://morphik:morphik@postgres:5432/morphik
PGPASSWORD=morphik
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=DEBUG
REDIS_HOST=redis
REDIS_PORT=6379
OPENAI_API_KEY=
EOL
        echo ".env file created successfully."
    fi
}

# Function to create morphik.toml if it doesn't exist
create_morphik_toml() {
    echo "Syncing morphik.toml with morphik.docker.toml..."
    cp -f morphik.docker.toml morphik.toml
}

# Function to create necessary directories
create_directories() {
    mkdir -p storage logs
}

# Main execution
echo "Starting Morphik setup..."

# Check prerequisites
check_docker
check_docker_compose

# Create necessary files and directories
create_env_file
create_morphik_toml
create_directories

# Build and start the containers
echo "Building and starting containers..."
docker compose build
docker compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check if services are running
if ! docker compose ps | grep -q "Up"; then
    echo "Error: Some services failed to start. Check the logs with 'docker compose logs'"
    exit 1
fi

echo "Morphik is now running!"
echo "API is available at http://localhost:8000"
echo "To view logs, run: docker compose logs -f"
echo "To stop the services, run: docker compose down"

# Docker Setup Guide for Morphik Core

Morphik Core provides a streamlined Docker-based setup that includes all necessary components: the core API, PostgreSQL with pgvector, and Ollama for AI models.

## Prerequisites

- Docker and Docker Compose 2.24.0 or newer installed on your system
- At least 10GB of free disk space (for models and data)
- 8GB+ RAM recommended

## Quick Start

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/morphik-org/morphik-core.git
cd morphik-core
```

2. Create a `.env` file for Docker secrets:
```bash
umask 077
cat > .env <<EOF
JWT_SECRET_KEY=$(openssl rand -hex 32)
SESSION_SECRET_KEY=$(openssl rand -hex 32)
LOCAL_URI_PASSWORD=
EOF
```

If `openssl` is not available, set `JWT_SECRET_KEY` and `SESSION_SECRET_KEY` to separate non-placeholder random hex values with at least 32 characters. Leave `LOCAL_URI_PASSWORD` blank unless you need `/local/generate_uri`. Shell-exported values for these same variables are also supported for CI or scripted deployments.

3. First-time setup:
```bash
docker compose up --build
```

This command will:
- Build all required containers
- Download necessary AI models (nomic-embed-text and llama3.2)
- Initialize the PostgreSQL database with pgvector
- Start all services

The initial setup may take 5-10 minutes depending on your internet speed, as it needs to download the AI models.

4. For subsequent runs:
```bash
docker compose up    # Start all services
docker compose down  # Stop all services
```

5. To completely reset (will delete all data and models):
```bash
docker compose down -v
```

> **Note:** If you enabled the optional UI profile (or any other compose profile), make sure to include `--profile ui` when stopping services (`docker compose --profile ui down --volumes --remove-orphans`). The hosted installer generates a `stop-morphik` script that does this for you automatically.

## Configuration

### 1. Default Setup

The default configuration works out of the box and includes:
- PostgreSQL with pgvector for document storage
- Ollama for AI models (embeddings and completions)
- Local file storage
- Basic authentication

### 2. Configuration File (morphik.toml)

The default `morphik.toml` is configured for Docker and includes:

```toml
[api]
host = "0.0.0.0"  # Important: Use 0.0.0.0 for Docker
port = 8000

[completion]
provider = "ollama"
model_name = "llama3.2"
base_url = "http://ollama:11434"  # Use Docker service name

[embedding]
provider = "ollama"
model_name = "nomic-embed-text"
base_url = "http://ollama:11434"  # Use Docker service name

[database]
provider = "postgres"

[vector_store]
provider = "pgvector"

[storage]
provider = "local"
storage_path = "/app/storage"
```

### 3. Environment Variables

Create a `.env` file before starting Docker. Docker Compose loads this file for both the API and worker services:

```bash
JWT_SECRET_KEY=<32+-character-random-hex-secret>      # Important: Change in production
SESSION_SECRET_KEY=<32+-character-random-hex-secret>  # Important: Change in production
LOCAL_URI_PASSWORD=<32+-character-random-hex-secret>  # Only needed for /local/generate_uri
OPENAI_API_KEY=sk-...                # Only if using OpenAI
HOST=0.0.0.0                         # Leave as is for Docker
PORT=8000                            # Change if needed
```

When `bypass_auth_mode = false`, `JWT_SECRET_KEY` and `SESSION_SECRET_KEY` must be non-empty, non-placeholder values with at least 32 characters. If `LOCAL_URI_PASSWORD` is unset or blank, `/local/generate_uri` is disabled; if you set it, use a non-placeholder value with at least 32 characters. When writing secrets to `.env`, use hex values such as `openssl rand -hex 32` so Docker Compose does not treat characters like `$`, quotes, or `#` as env-file syntax.

Upgrade note: existing authenticated Docker deployments must verify that `JWT_SECRET_KEY` and `SESSION_SECRET_KEY` are both non-placeholder random values with at least 32 characters before pulling an image with this validation. Use the same values for both the `morphik` API service and the `worker` service through `.env` or shell-exported environment variables. If `LOCAL_URI_PASSWORD` is set, replace weak or placeholder values with a non-placeholder 32+ character value, or clear it to disable `/local/generate_uri`.

### 4. Custom Configuration

To use your own configuration:
1. Create a custom `morphik.toml`
2. Mount it in `docker-compose.yml`:
```yaml
services:
  morphik:
    volumes:
      - ./my-custom-morphik.toml:/app/morphik.toml
```

## Accessing Services

- Morphik API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Storage and Data

- Database data: Stored in the `postgres_data` Docker volume
- AI Models: Stored in the `ollama_data` Docker volume
- Documents: Stored in `./storage` directory (mounted to container)
- Logs: Available in `./logs` directory

## Troubleshooting

1. **Service Won't Start**
   ```bash
   # View all logs
   docker compose logs

   # View specific service logs
   docker compose logs morphik
   docker compose logs postgres
   docker compose logs ollama
   ```

2. **Database Issues**
   - Check PostgreSQL is healthy: `docker compose ps`
   - Verify database connection: `docker compose exec postgres psql -U morphik -d morphik`

3. **Auth Secret Issues**
   - If startup fails with `JWT_SECRET_KEY` or `SESSION_SECRET_KEY` validation errors, set both values in `.env` to non-placeholder random strings with at least 32 characters and restart
   - If startup fails with `LOCAL_URI_PASSWORD` validation errors, replace it with a non-placeholder value with at least 32 characters, or clear it to disable `/local/generate_uri`
   - If `/local/generate_uri` returns HTTP `503` with `LOCAL_URI_PASSWORD is not configured; /local/generate_uri is disabled`, set `LOCAL_URI_PASSWORD` in `.env` to a non-placeholder value with at least 32 characters before using that endpoint

4. **Model Download Issues**
   - Check Ollama logs: `docker compose logs ollama`
   - Ensure enough disk space for models
   - Try restarting Ollama: `docker compose restart ollama`

5. **Performance Issues**
   - Monitor resources: `docker stats`
   - Ensure sufficient RAM (8GB+ recommended)
   - Check disk space: `df -h`

## Production Deployment

For production environments:

1. **Security**:
   - Use randomly generated `JWT_SECRET_KEY` and `SESSION_SECRET_KEY` values of at least 32 characters; do not rely on example or development defaults
   - Set a randomly generated `LOCAL_URI_PASSWORD` of at least 32 characters before using `/local/generate_uri`
   - Use proper network security groups
   - Enable HTTPS (recommended: use a reverse proxy)
   - Regularly update containers and dependencies

2. **Persistence**:
   - Use named volumes for all data
   - Set up regular backups of PostgreSQL
   - Back up the storage directory

3. **Monitoring**:
   - Set up container monitoring
   - Configure proper logging
   - Use health checks

## Support

For issues and feature requests:
- GitHub Issues: [https://github.com/morphik-org/morphik-core/issues](https://github.com/morphik-org/morphik-core/issues)
- Documentation: [https://docs.morphik.ai](https://docs.morphik.ai)

## Repository Information

- License: MIT

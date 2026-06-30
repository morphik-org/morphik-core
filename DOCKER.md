# Docker Setup Guide for Morphik Core

Morphik Core provides a streamlined Docker-based setup that includes the core API, PostgreSQL with pgvector, Redis, and optional profiles for Ollama and the web UI.

This guide covers local development from a cloned repository with `docker-compose.yml` and `./start-dev.sh`. The hosted installer and pre-built image path use `docker-compose.run.yml` and start from the Docker-specific `morphik.docker.toml` template.

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 10GB of free disk space (for models and data)
- 8GB+ RAM recommended

## Quick Start

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/morphik-org/morphik-core.git
cd morphik-core
```

2. Create the required Docker Compose `.env` file before first start:

   ```bash
   cp .env.example .env
   ```

   Telemetry is enabled by default for self-hosted deployments. To opt out before any service starts, add this line to `.env` before continuing:

   ```dotenv
   TELEMETRY=false
   ```

   You can add other environment overrides to the same file; see [Environment Variables](#3-environment-variables).

3. Choose a Docker-reachable model configuration before first start.

   The repository's checked-in `morphik.toml` is a local development config. It enables auth bypass and selects Ollama models at `http://localhost:11434`, which is useful for direct local runs but not reachable from inside Docker containers. Pick one Docker path before starting services:

   - For an OpenAI-backed Docker setup, copy the Docker template into `morphik.toml` if you do not have local edits, then set `OPENAI_API_KEY` in `.env`. The Docker template selects OpenAI models for model-backed ingestion, parsing, and query operations.

     ```bash
     cp morphik.docker.toml morphik.toml
     ```

   - For Ollama inside Docker Compose, keep or select Ollama-backed models in `morphik.toml`, update the selected Ollama `api_base` values to `http://ollama:11434`, and start with the `ollama` profile. For the checked-in `morphik.toml`, this updates the Docker-facing Ollama endpoints:

     ```bash
     python - <<'PY'
     from pathlib import Path

     path = Path("morphik.toml")
     text = path.read_text()
     text = text.replace('api_base = "http://localhost:11434"', 'api_base = "http://ollama:11434"')
     path.write_text(text)
     PY
     ```

     The included Ollama entrypoint pulls `nomic-embed-text` and `llama3.2`. The checked-in config also selects `qwen2.5vl:latest` for vision-capable completion and parsing, so pull that model after the Ollama service starts, or change those `morphik.toml` selections to a model that is already pulled:

     ```bash
     docker compose exec ollama ollama pull qwen2.5vl:latest
     ```

4. First-time setup:
```bash
./start-dev.sh --build
```

This command will:
- Build all required containers
- Initialize the PostgreSQL database with pgvector
- Start the API, worker, Redis, and PostgreSQL services

If you chose the Ollama-in-Compose path, include the optional Ollama profile:

```bash
COMPOSE_PROFILES=ollama ./start-dev.sh --build
```

When the Ollama profile is enabled in the development compose file, the Ollama entrypoint pulls `nomic-embed-text` and `llama3.2`; this can add several minutes to the first startup depending on your internet speed.

5. For subsequent runs:
```bash
./start-dev.sh       # Start services with the port from morphik.toml
docker compose down  # Stop services
```

6. To completely reset (will delete all data and models):
```bash
docker compose down -v
```

> **Note:** If you enabled compose profiles, include the same profiles when stopping services, for example `docker compose --profile ollama down` or `docker compose --profile ui --profile ollama down --volumes --remove-orphans`. The hosted installer generates a `stop-morphik` script that does this for you automatically.

## Configuration

### 1. Default Setup

The development compose stack includes:
- PostgreSQL with pgvector for document storage
- Model selection through `registered_models` in `morphik.toml`
- Local file storage
- Development auth bypass when using the repository's checked-in `morphik.toml`

Model-backed operations require the selected provider to be reachable from the containers. For the Docker template's OpenAI selections, set `OPENAI_API_KEY` in `.env`. For local models, use a Docker-reachable endpoint such as `http://ollama:11434` when running Ollama as a compose profile.

### 2. Configuration File (`morphik.toml`)

Docker Compose mounts the host `./morphik.toml` file into the API and worker containers. The installer and published Docker image use the Docker-specific `morphik.docker.toml` template as the starting point, then expose it as `morphik.toml` for local edits. Model providers are configured under `registered_models`, then selected by name:

```toml
[api]
host = "0.0.0.0"  # Important: Use 0.0.0.0 for Docker
port = 8000

[registered_models]
openai_gpt4-1-mini = { model_name = "gpt-4.1-mini" }
openai_embedding = { model_name = "text-embedding-3-small" }

[completion]
model = "openai_gpt4-1-mini"  # Reference to a key in registered_models

[embedding]
model = "openai_embedding"  # Reference to a key in registered_models
dimensions = 1536
similarity_metric = "cosine"

[database]
provider = "postgres"

[vector_store]
provider = "pgvector"

[storage]
provider = "local"
storage_path = "/app/storage"
```

### 3. Environment Variables

Docker Compose reads settings from the required `.env` file. If you followed the Quick Start, it was created from `.env.example`; update it to customize these settings:

```bash
JWT_SECRET_KEY=your-secure-key-here      # Important: change in production
SESSION_SECRET_KEY=your-session-key-here # Important: change in production
OPENAI_API_KEY=sk-...                    # Only if using OpenAI
ANTHROPIC_API_KEY=                       # Only if using Anthropic
GEMINI_API_KEY=                          # Only if using Gemini
LOCAL_URI_PASSWORD=                      # Optional: enables local URI generation
# TELEMETRY=false                        # Optional: disable telemetry before first start
```

Telemetry is enabled by default for self-hosted deployments. Set `TELEMETRY=false` in `.env` before starting services if your deployment should opt out; see [Morphik telemetry](docs/telemetry.md). Change API host, port, model, storage, and provider-selection settings in `morphik.toml`, not in `.env`. Docker Compose injects the container `POSTGRES_URI` for the bundled PostgreSQL service. The current image startup check still expects that bundled service at host `postgres` with user/database `morphik`, so rotated database users, renamed databases, or external database endpoints require updating the compose settings and image startup check together.

### 4. Security and Local-Only Defaults

The `./start-dev.sh` path is intended for local development. The repository's checked-in `morphik.toml` sets `bypass_auth_mode = true`, so JWT and session secrets do not protect the API until auth bypass is disabled in `morphik.toml`.

The bundled PostgreSQL service also uses the default `morphik/morphik` credentials and publishes port `5432` to the host. Keep this stack bound to a trusted local machine. Before any non-local deployment, remove or restrict host database port publishing; rotating the bundled database credentials or replacing the database service also requires updating the compose files and the image startup check that currently expects the bundled defaults.

### 5. Custom Configuration

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

3. **Model Download Issues**
   - Check Ollama logs: `docker compose logs ollama`
   - Ensure enough disk space for models
   - Try restarting Ollama: `docker compose restart ollama`

4. **Performance Issues**
   - Monitor resources: `docker stats`
   - Ensure sufficient RAM (8GB+ recommended)
   - Check disk space: `df -h`

## Non-local Deployment Checklist

For production or other non-local environments, prefer the hosted installer or pre-built image path that uses `docker-compose.run.yml`; treat this repository-clone compose file as a development starting point.

1. **Security**:
   - Change the default `JWT_SECRET_KEY` and `SESSION_SECRET_KEY`
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

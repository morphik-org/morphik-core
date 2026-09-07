# Docker setup guide for Morphik Core

This guide uses the production Compose deployment in `docker-compose.run.yml`. It runs the published Morphik Core
image with PostgreSQL and Redis. Ollama and the admin UI are optional profiles.

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 10GB of free disk space (for models and data)
- 8GB+ RAM recommended

## Quick start

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/morphik-org/morphik-core.git
cd morphik-core
```

2. Run the production installer:

```bash
./install_docker.sh
```

The installer pulls the published image, creates `.env`, writes the Docker configuration, asks which model provider to
use, and starts `docker-compose.run.yml`.

3. For subsequent runs:

```bash
./start-morphik.sh
./stop-morphik.sh
```

Both commands are idempotent. The stop script removes containers and the Compose network, including services in
optional profiles, but preserves the named volumes that hold PostgreSQL, Redis, and model data.

4. To completely reset all Compose-managed data, run this destructive command:

```bash
docker compose -f docker-compose.run.yml --profile "*" down --volumes --remove-orphans
```

It removes PostgreSQL and every other named volume. Back up the database and `./storage` before an intentional reset.

## Configuration

### 1. Default setup

The installed stack includes:

- PostgreSQL with pgvector for document storage
- Redis for ingestion jobs
- Local file storage
- Configurable local or external model providers
- Token authentication or explicit local-only bypass mode

### 2. Configuration file

Edit the generated `morphik.toml`. For example, an Ollama container on the same Compose network uses:

```toml
[api]
host = "0.0.0.0"
port = 8000

[registered_models]
ollama_chat = { model_name = "ollama_chat/llama3.2", api_base = "http://ollama:11434" }
ollama_embedding = { model_name = "ollama/nomic-embed-text", api_base = "http://ollama:11434" }

[completion]
model = "ollama_chat"

[embedding]
model = "ollama_embedding"
dimensions = 768
similarity_metric = "cosine"

[database]
provider = "postgres"

[vector_store]
provider = "pgvector"

[storage]
provider = "local"
storage_path = "/app/storage"

[morphik]
mode = "self_hosted"
enable_colpali = false
colpali_mode = "off"
```

Set `COMPOSE_PROFILES=ollama` in `.env` before starting this example. Pull the required models into Ollama before using
the API.

### 3. Environment variables

The installer creates `.env`. Review these values before exposing the service:

```bash
JWT_SECRET_KEY=your-secure-key-here  # Important: Change in production
OPENAI_API_KEY=sk-...                # Only if using OpenAI
TELEMETRY=false                      # Recommended for a no-egress deployment
COMPOSE_PROJECT_NAME=morphik         # Set once before the first start
```

### 4. Custom configuration

`docker-compose.run.yml` mounts `./morphik.toml` read-only into both the API and worker containers. Edit that file and
run `./start-morphik.sh` again to apply changes.

## Accessing services

- Morphik API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Storage and data

- Database data: Stored in the `postgres_data` Docker volume outside the PostgreSQL container
- AI Models: Stored in the `ollama_data` Docker volume
- Documents: Stored in `./storage` directory (mounted to container)
- Logs: Available in `./logs` directory

## Troubleshooting

1. **Service will not start**

   ```bash
   # View all logs
   docker compose -f docker-compose.run.yml --profile "*" logs

   # View specific service logs
   docker compose -f docker-compose.run.yml logs morphik
   docker compose -f docker-compose.run.yml logs postgres
   docker compose -f docker-compose.run.yml --profile ollama logs ollama
   ```

2. **Database issues**

   - Check PostgreSQL health: `docker compose -f docker-compose.run.yml ps`
   - Verify the database: `docker compose -f docker-compose.run.yml exec postgres psql -U morphik -d morphik`

3. **Model download issues**

   - Check Ollama logs: `docker compose -f docker-compose.run.yml --profile ollama logs ollama`
   - Ensure enough disk space for models
   - Restart Ollama: `docker compose -f docker-compose.run.yml --profile ollama restart ollama`

4. **Performance issues**

   - Monitor resources: `docker stats`
   - Ensure sufficient RAM (8GB+ recommended)
   - Check disk space: `df -h`

## Production deployment

For production environments:

1. **Security**

   - Change the default `JWT_SECRET_KEY`
   - Use proper network security groups
   - Enable HTTPS (recommended: use a reverse proxy)
   - Regularly update containers and dependencies

2. **Persistence**

   - Use named volumes for all data
   - Set up regular backups of PostgreSQL
   - Back up the storage directory

3. **Monitoring**

   - Set up container monitoring
   - Configure proper logging
   - Use health checks

## Support

For issues and feature requests:

- GitHub Issues: [https://github.com/morphik-org/morphik-core/issues](https://github.com/morphik-org/morphik-core/issues)
- Documentation: [https://docs.morphik.ai](https://docs.morphik.ai)

## Repository information

- License: The Docker image bundles components under multiple licenses. Review [LICENSE](./LICENSE), [ee/LICENSE](./ee/LICENSE), and bundled package metadata before production use.

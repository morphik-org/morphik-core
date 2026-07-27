# Docker Setup Guide for Morphik Core

Morphik Core provides a streamlined Docker-based setup that includes all necessary components: the core API, PostgreSQL with pgvector, and Ollama for AI models.

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

2. First-time setup:
```bash
docker compose up --build
```

This command will:
- Build all required containers
- Download necessary AI models (nomic-embed-text and llama3.2)
- Initialize the PostgreSQL database with pgvector
- Start all services

The initial setup may take 5-10 minutes depending on your internet speed, as it needs to download the AI models.

3. For subsequent runs:
```bash
docker compose up    # Start all services
docker compose down  # Stop all services
```

4. To completely reset (will delete all data and models):
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

A Docker `morphik.toml` can configure local Ollama models with registered model keys:

```toml
[api]
host = "0.0.0.0"  # Important: Use 0.0.0.0 for Docker
port = 8000

[registered_models]
ollama_chat = { model_name = "ollama_chat/llama3.2", api_base = "http://ollama:11434" }
ollama_embedding = { model_name = "ollama/nomic-embed-text", api_base = "http://ollama:11434" }

[completion]
model = "ollama_chat"  # Reference to a key in registered_models

[embedding]
model = "ollama_embedding"  # Reference to a key in registered_models
dimensions = 768
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

Create a `.env` file to customize these settings:

```bash
JWT_SECRET_KEY=your-secure-key-here  # Important: Change in production
# OPENAI_API_KEY=your-openai-or-proxy-key # Optional: OpenAI/OpenAI-compatible proxy key
HOST=0.0.0.0                         # Leave as is for Docker
PORT=8000                            # Change if needed
```

Keep `.env` local and out of version control when it contains real secrets.

### 4. Custom Configuration

To use your own configuration:
1. Create a custom `morphik.toml`
2. Mount the same file into each service that reads Morphik configuration, including both `morphik` and `worker`:
```yaml
services:
  morphik:
    volumes:
      - ./my-custom-morphik.toml:/app/morphik.toml
  worker:
    volumes:
      - ./my-custom-morphik.toml:/app/morphik.toml
```

### 5. LiteLLM Proxy or OpenAI-Compatible Endpoints

To use a LiteLLM proxy or another OpenAI-compatible endpoint:

1. Define proxy-backed models under `[registered_models]`.
2. Point `[completion].model` and `[embedding].model` at those registered model keys.
3. Recreate the proxy-calling services after updating `morphik.toml` or `.env`:

```bash
docker compose up -d --force-recreate morphik worker
```

After running one query and one small ingestion, you can use the logs as a smoke check for the selected model keys. This check is safe only when secrets are not embedded in registered-model config:

```bash
docker compose logs morphik worker | grep -E "litellm_proxy_(chat|embedding)"
```

```toml
[registered_models]
litellm_proxy_chat = { model_name = "openai/gpt-4o-mini", api_base = "http://litellm:4000" }
litellm_proxy_embedding = { model_name = "openai/text-embedding-3-small", api_base = "http://litellm:4000" }

[completion]
model = "litellm_proxy_chat"

[embedding]
model = "litellm_proxy_embedding"
dimensions = 1536
similarity_metric = "cosine"
```

If you change `[embedding].model` or `[embedding].dimensions` on an existing deployment, keep the new embedding dimensions compatible with existing vectors or plan a re-ingestion, new collection/vector table, migration, or database reset. Recreating `morphik` and `worker` reloads config, but it does not rewrite existing pgvector data.

Use the Docker Compose service name, such as `http://litellm:4000`, when the proxy runs in the same Compose project. Use `http://host.docker.internal:4000` only when the proxy runs on the Docker host; Linux Docker setups may require adding `extra_hosts: ["host.docker.internal:host-gateway"]` to every service that calls the proxy, at least `morphik` and `worker`:

```yaml
services:
  morphik:
    extra_hosts:
      - "host.docker.internal:host-gateway"
  worker:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

For proxy-backed LiteLLM/OpenAI-compatible `[registered_models]` entries, put the endpoint in `api_base`. The API-key surface is separate: `/api-keys` accepts `base_url` in requests and stores/returns it as `baseUrl`; those fields do not configure backend registered models.

Proxy-backed entries may be classified as `custom` in Morphik metadata/provider fields even when the upstream proxy exposes OpenAI-compatible models. Runtime calls still use the configured `model_name` and `api_base`.

Credential and secret handling:

- Set `OPENAI_API_KEY` in the shared `.env` or on every proxy-calling service, including `morphik` and `worker`, to the proxy key for general LiteLLM/OpenAI-compatible calls.
- If an authenticated embedding proxy uses a local-looking URL such as `localhost`, `127.0.0.1`, or `host.docker.internal`, also set `LITELLM_DUMMY_API_KEY` to the proxy key because Morphik forwards that value for local embedding providers.
- If your local proxy has authentication disabled, LiteLLM's OpenAI-compatible path may still require harmless placeholders such as `OPENAI_API_KEY=dummy` and, for local embeddings, `LITELLM_DUMMY_API_KEY=dummy`. Those dummy values are only local client-compatibility placeholders; they do not secure the proxy.
- Keep real secrets out of `registered_models.*.api_key`; registered-model config can appear in application or worker logs and model metadata responses.
- Prefer environment or secret-manager handling for proxy credentials, keep `.env` out of version control, do not commit real proxy credentials in shared config files, and do not embed credentials or bearer tokens in `api_base`, `base_url`, or `baseUrl` URLs.

This recipe also covers query and agent traffic that use `[completion].model`. If contextual chunking or parser vision traffic should use the same proxy, update `parser.contextual_chunking_model` and `[parser.vision].model` to proxy-backed registered model keys as well.

These examples use HTTP for local Docker networking. Use HTTPS and authentication for remote or shared-network proxy endpoints.

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

## Production Deployment

For production environments:

1. **Security**:
   - Change the default `JWT_SECRET_KEY`
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

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

Create a `.env` file to customize these settings:

```bash
JWT_SECRET_KEY=your-secure-key-here  # Important: Change in production
OPENAI_API_KEY=sk-...                # Only if using OpenAI
HOST=0.0.0.0                         # Leave as is for Docker
PORT=8000                            # Change if needed
```

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

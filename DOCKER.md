# Docker Setup Guide for DataBridge Core

DataBridge Core is designed to run with Docker Compose, making it easy to set up all required services locally.

## Quick Start with Docker Compose

1. Start the services:
```bash
docker compose up -d
```

This will start:
- DataBridge Core API
- PostgreSQL with pgvector for vector storage
- Ollama for embeddings and completions

The container includes a startup script that:
- Creates a default `databridge.toml` if none exists
- Waits for PostgreSQL to be ready before starting
- Starts the API server with the configured settings

All necessary environment variables have default values in the docker-compose.yml file. You can optionally override them by creating a `.env` file:

```bash
# Optional .env file
JWT_SECRET_KEY=your-custom-secret-key  # Defaults to "your-secret-key-here" if not set
OPENAI_API_KEY=sk-...                  # Only needed if using OpenAI
```

## Docker Hub Images

DataBridge Core is available on Docker Hub at `adityava369/databridge-core`. For simplicity, we currently use continuous deployment:

- Every commit to the main branch automatically updates the `latest` tag
- Pull the latest version: `docker pull adityava369/databridge-core:latest`

### Publishing to Docker Hub

The repository automatically publishes to Docker Hub on every commit to the main branch. For manual publishing:

```bash
# Build the image
docker build -t adityava369/databridge-core:latest .

# Push to Docker Hub (requires login)
docker push adityava369/databridge-core:latest
```

## Configuration

### Environment Variables

Most environment variables have sensible defaults in docker-compose.yml. You only need to set variables if you want to override the defaults:

- `JWT_SECRET_KEY` - Secret key for JWT token generation (default: "your-secret-key-here")
- `POSTGRES_URI` - PostgreSQL connection string (default: postgresql+asyncpg://databridge:databridge@postgres:5432/databridge)
- `OPENAI_API_KEY` - OpenAI API key (only if using OpenAI)
- `MONGODB_URI` - MongoDB connection string (only if using MongoDB)
- `AWS_ACCESS_KEY` - AWS access key (only if using S3)
- `AWS_SECRET_ACCESS_KEY` - AWS secret key (only if using S3)
- `HOST` - API host (default: 0.0.0.0)
- `PORT` - API port (default: 8000)

### Default Configuration

The container comes with a default `databridge.toml` that configures:
- Ollama for embeddings (nomic-embed-text) and completions (llama2)
- PostgreSQL with pgvector for vector storage
- Local file storage
- Unstructured parser without API

You can override this by mounting your own `databridge.toml`:

```yaml
# In docker-compose.yml
services:
  databridge:
    volumes:
      - ./databridge.toml:/app/databridge.toml
```

## Health Checks

The API exposes health check endpoints:
- `GET /health` - Basic health check
- `GET /health/ready` - Readiness check with component status

## Production Deployment Tips

1. **Security**:
   - Use a strong `JWT_SECRET_KEY`
   - Store sensitive environment variables in a secure environment
   - Enable HTTPS in production
   - Use proper network isolation

2. **Performance**:
   - Use volume mounts for persistent storage
   - Configure appropriate resource limits
   - Monitor container health and logs

3. **Backups**:
   - Regularly backup PostgreSQL data
   - Backup any file storage
   - Test restore procedures

## Troubleshooting

1. **Services won't start**:
   ```bash
   # Check logs
   docker compose logs
   
   # Check specific service
   docker compose logs databridge
   ```

2. **Database Connection Issues**:
   - Verify PostgreSQL is running: `docker compose ps`
   - Check database logs: `docker compose logs postgres`
   - Ensure database migrations are applied

3. **Performance Issues**:
   - Monitor resource usage: `docker stats`
   - Check container limits in docker-compose.yml
   - Review logging levels in databridge.toml

## Support

For issues and feature requests, please visit our GitHub repository. 

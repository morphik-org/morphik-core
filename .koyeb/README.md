# Morphik Koyeb Deployment

This directory contains configuration files for deploying Morphik to Koyeb.

## Prerequisites

Before deploying to Koyeb, you need to set up:

1. External PostgreSQL database with pgvector extension
2. External Redis service

## Environment Variables

Configure the following environment variables in Koyeb:

- `JWT_SECRET_KEY`: Generate a secure random string
- `POSTGRES_URI`: Set to `postgresql+asyncpg://username:password@your-postgres-host:5432/morphik`
- `PGPASSWORD`: Set to your PostgreSQL password
- `HOST`: Set to `0.0.0.0`
- `PORT`: Set to `8000`
- `LOG_LEVEL`: Set to `INFO` for production or `DEBUG` for troubleshooting
- `REDIS_HOST`: Set to your Redis host address
- `REDIS_PORT`: Set to your Redis port (typically `6379`)

## Database Initialization

Ensure your PostgreSQL database has the pgvector extension enabled and the required schema:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector_embeddings table
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255),
    chunk_number INTEGER,
    content TEXT,
    chunk_metadata TEXT,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## Deployment

Deploy both the API and worker services:

1. API Service: `koyeb service create --name morphik-api --port 8000 --docker-image <your-docker-image> --env-file .env`
2. Worker Service: `koyeb service create --name morphik-worker --docker-image <your-docker-image> --command "arq core.workers.ingestion_worker.WorkerSettings" --env-file .env`

## Troubleshooting

If the deployment fails, check:

1. Container logs for specific error messages
2. Connectivity to PostgreSQL and Redis services
3. Environment variable configuration
4. Proper initialization of the database schema

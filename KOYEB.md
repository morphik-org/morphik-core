# Morphik Koyeb Deployment Guide

This guide outlines the steps to deploy Morphik to Koyeb, a cloud platform that simplifies containerized application deployment.

## Prerequisites

Before deploying to Koyeb, you need to set up:

1. External PostgreSQL database with pgvector extension
2. External Redis service

## Database Setup

Ensure your PostgreSQL database has the pgvector extension enabled:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Initialize the database schema using the SQL statements in `init.sql`. The most critical table is:

```sql
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255),
    chunk_number INTEGER,
    content TEXT,
    chunk_metadata TEXT,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

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

## Deployment Options

### Option 1: Deploy using Koyeb Web UI

1. Create a new service in Koyeb
2. Select Docker Registry as the deployment method
3. Enter your Docker image URL
4. Configure environment variables
5. Set the port to 8000
6. Deploy

### Option 2: Deploy using Koyeb CLI

Deploy both the API and worker services:

```bash
# Deploy API service
koyeb service create --name morphik-api --port 8000 --docker-image <your-docker-image> --env-file .env

# Deploy Worker service
koyeb service create --name morphik-worker --docker-image <your-docker-image> --command "arq core.workers.ingestion_worker.WorkerSettings" --env-file .env
```

## Configuration Files

The `.koyeb` directory contains configuration files for deployment:

- `koyeb.yaml`: Configuration for the API service
- `worker.yaml`: Configuration for the worker service

## Troubleshooting

If the deployment fails, check:

1. Container logs for specific error messages
2. Connectivity to PostgreSQL and Redis services
3. Environment variable configuration
4. Proper initialization of the database schema

Remember that the application is designed to exit if it cannot connect to Redis or PostgreSQL, so ensure these services are properly configured and accessible.

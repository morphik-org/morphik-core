# syntax=docker/dockerfile:1

# Build stage
FROM python:3.12.5-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt averaged_perceptron_tagger

# Production stage
FROM python:3.12.5-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libmagic1 \
    tesseract-ocr \
    postgresql-client \
    poppler-utils \
    gcc \
    g++ \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /root/.local/bin /usr/local/bin
# Copy NLTK data from builder
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

# Create necessary directories
RUN mkdir -p storage logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PATH="/usr/local/bin:$PATH"

# Create default configuration
RUN echo '[api]\n\
host = "0.0.0.0"\n\
port = 8000\n\
reload = false\n\
\n\
[auth]\n\
jwt_algorithm = "HS256"\n\
dev_mode = true\n\
dev_entity_id = "dev_user"\n\
dev_entity_type = "developer"\n\
dev_permissions = ["read", "write", "admin"]\n\
\n\
[registered_models]\n\
# Ollama models for docker default configuration\n\
ollama_llama = { model_name = "ollama_chat/llama3.2", api_base = "http://ollama:11434" }\n\
ollama_embedding = { model_name = "ollama/nomic-embed-text", api_base = "http://ollama:11434" }\n\
\n\
[completion]\n\
model = "ollama_llama"\n\
default_max_tokens = "1000"\n\
default_temperature = 0.5\n\
\n\
[database]\n\
provider = "postgres"\n\
\n\
[embedding]\n\
model = "ollama_embedding"\n\
dimensions = 768\n\
similarity_metric = "cosine"\n\
\n\
[parser]\n\
chunk_size = 1000\n\
chunk_overlap = 200\n\
use_unstructured_api = false\n\
use_contextual_chunking = false\n\
\n\
[reranker]\n\
use_reranker = false\n\
\n\
[storage]\n\
provider = "local"\n\
storage_path = "/app/storage"\n\
\n\
[vector_store]\n\
provider = "pgvector"\n\
\n\
[morphik]\n\
enable_colpali = true\n\
mode = "self_hosted"\n\
' > /app/morphik.toml.default

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Copy default config if none exists\n\
if [ ! -f /app/morphik.toml ]; then\n\
    cp /app/morphik.toml.default /app/morphik.toml\n\
fi\n\
\n\
# Function to check PostgreSQL\n\
check_postgres() {\n\
    if [ -n "$POSTGRES_URI" ]; then\n\
        echo "Waiting for PostgreSQL..."\n\
        max_retries=30\n\
        retries=0\n\
        until PGPASSWORD=$PGPASSWORD pg_isready -h postgres -U morphik -d morphik; do\n\
            retries=$((retries + 1))\n\
            if [ $retries -eq $max_retries ]; then\n\
                echo "Error: PostgreSQL did not become ready in time"\n\
                exit 1\n\
            fi\n\
            echo "Waiting for PostgreSQL... (Attempt $retries/$max_retries)"\n\
            sleep 2\n\
        done\n\
        echo "PostgreSQL is ready!"\n\
        \n\
        # Verify database connection\n\
        if ! PGPASSWORD=$PGPASSWORD psql -h postgres -U morphik -d morphik -c "SELECT 1" > /dev/null 2>&1; then\n\
            echo "Error: Could not connect to PostgreSQL database"\n\
            exit 1\n\
        fi\n\
        echo "PostgreSQL connection verified!"\n\
    fi\n\
}\n\
\n\
# Check PostgreSQL\n\
check_postgres\n\
\n\
# Start the application with standard asyncio event loop\n\
exec uvicorn core.api:app --host $HOST --port $PORT --loop asyncio --http auto --ws auto --lifespan auto\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Copy application code
COPY core ./core
COPY README.md LICENSE ./

# Labels for the image
LABEL org.opencontainers.image.title="Morphik Core"
LABEL org.opencontainers.image.description="Morphik Core - A powerful document processing and retrieval system"
LABEL org.opencontainers.image.source="https://github.com/morphiklabs/morphik"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"

# Expose port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

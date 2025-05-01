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

# Install Rust using the simpler method
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

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

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
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
echo "[Entrypoint] Using PostgreSQL connection: $POSTGRES_URI"\n\
\n\
# Check if command arguments were passed ($# is the number of arguments)\n\
if [ $# -gt 0 ]; then\n\
    # If arguments exist, execute them (e.g., execute "arq core.workers...")\n\
    exec "$@"\n\
else\n\
    # Otherwise, execute the default command (Uvicorn for the API)\n\
    exec uvicorn core.api:app --host $HOST --port $PORT --loop asyncio --http auto --ws auto --lifespan auto\n\
fi\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# COPY docker-entrypoint.sh /app/docker-entrypoint.sh
# RUN chmod +x /app/docker-entrypoint.sh

# Copy application code
COPY core ./core
COPY README.md LICENSE ./

# Labels for the image
LABEL org.opencontainers.image.title="Morphik Core"
LABEL org.opencontainers.image.description="Morphik Core - A powerful document processing and retrieval system"
LABEL org.opencontainers.image.source="https://github.com/yourusername/morphik"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"

# Expose port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]
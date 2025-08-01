# syntax=docker/dockerfile:1

# Purpose: Production Docker image for Morphik Core
# This file builds the official Morphik image that gets published to ghcr.io
# It includes all dependencies and uses start_server.py which reads config from morphik.toml
# Used by: GitHub Actions (docker-publish.yml) and developers building locally

# Build stage
FROM python:3.11.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using the simpler method
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
# Activating cargo env for this RUN instruction and subsequent ones in this stage.
ENV PATH="/root/.cargo/bin:${PATH}"

# Set uv environment variables
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy project definition and lock file
COPY pyproject.toml uv.lock ./
COPY fde ./fde

# Create venv and install dependencies from lockfile (excluding the project itself initially for better caching)
# This also creates the /app/.venv directory
# Cache buster: 1 - verbose flag added
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --verbose --locked --no-install-project

# Copy the rest of the application code
# Assuming start_server.py is at the root or handled by pyproject.toml structure.
COPY . .

# Copy the UI component (including it in the image for optional use)
# This ensures the UI is available when users want to enable it
COPY ee/ui-component /app/ee/ui-component

# Install the project itself into the venv in non-editable mode
# Cache buster: 1 - verbose flag added
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --verbose --locked --no-editable

# Install additional packages as requested
# Cache buster: 1 - verbose flag added
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv pip install --verbose 'colpali-engine@git+https://github.com/illuin-tech/colpali@80fb72c9b827ecdb5687a3a8197077d0d01791b3'

# Enable backports and install GCC 11+ for Debian Bookworm
RUN echo "deb http://deb.debian.org/debian bookworm-backports main" > /etc/apt/sources.list.d/backports.list && \
    apt-get update && \
    apt-get install -y -t bookworm-backports gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Cache buster: 1 - verbose flag already present
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python==0.3.5

# Download NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt averaged_perceptron_tagger

# Production stage
FROM python:3.11.12-slim

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
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv
# Copy uv binaries from the builder stage
COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /bin/uvx /bin/uvx

# Copy NLTK data from builder
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

## copy fde package to avoid error at server startup
COPY --from=builder /app/fde ./fde

# Create necessary directories
RUN mkdir -p storage logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:/usr/local/bin:${PATH}"

# Create default configuration
COPY morphik.docker.toml /app/morphik.toml.default

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
# Check if command arguments were passed ($# is the number of arguments)\n\
if [ $# -gt 0 ]; then\n\
    # If arguments exist, execute them (e.g., execute "arq core.workers...")\n\
    exec "$@"\n\
else\n\
    # Otherwise, execute the default command (uv run start_server.py)\n\
    exec uv run start_server.py --skip-redis-check\n\
fi\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Copy application code
# pyproject.toml is needed for uv to identify the project context for `uv run`
COPY pyproject.toml uv.lock ./

## copy the fde package also to fix distribution not found error
COPY fde ./fde

COPY core ./core
COPY ee ./ee
COPY README.md LICENSE ./
# Assuming start_server.py is at the root of your project
COPY start_server.py ./

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

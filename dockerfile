# syntax=docker/dockerfile:1

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

# Create venv and install dependencies from lockfile (excluding the project itself initially for better caching)
# This also creates the /app/.venv directory
# Cache buster: 1 - verbose flag added
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --verbose --locked --no-install-project

# Copy the rest of the application code
# Assuming start_server.py is at the root or handled by pyproject.toml structure.
COPY . .

# Install the project itself into the venv in non-editable mode
# Cache buster: 1 - verbose flag added
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --verbose --locked --no-editable

# Install additional packages as requested
# Cache buster: 1 - verbose flag added
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv pip install --verbose 'colpali-engine@git+https://github.com/illuin-tech/colpali@80fb72c9b827ecdb5687a3a8197077d0d01791b3'

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

# Create necessary directories
RUN mkdir -p storage logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:/usr/local/bin:${PATH}"

# Create default configuration
COPY docker/morphik.toml.default /app/morphik.toml.default

# Create startup script
COPY docker/parse-postgres-uri.sh /app/parse-postgres-uri.sh
COPY docker/docker-entrypoint.sh /app/docker-entrypoint.sh

# Copy application code
# pyproject.toml is needed for uv to identify the project context for `uv run`
COPY pyproject.toml ./
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

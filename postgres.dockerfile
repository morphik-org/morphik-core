FROM postgres:15-alpine

# Install build dependencies
RUN apk add --no-cache \
    git \
    build-base \
    clang19 \
    llvm19 \
    postgresql-dev

# Clone and build pgvector
RUN git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make OPTFLAGS="" \
    && make install

# Cleanup
RUN apk del git build-base clang19 llvm19 postgresql-dev \
    && rm -rf /pgvector

# Copy data dump
COPY dump.sql /tmp/dump.sql

# (No database init script necessary — schema is created by the application at runtime)

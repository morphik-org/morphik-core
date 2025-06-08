FROM postgres:16-alpine

# Install build dependencies
RUN apk add --no-cache \
    git \
    build-base \
    clang19 \
    llvm19 \
    postgresql-dev

# Clone and build pgvector
RUN git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make OPTFLAGS="" \
    && make install

# Cleanup
RUN apk del git build-base clang19 llvm19 postgresql-dev \
    && rm -rf /pgvector

# Copy data dump
COPY dump.sql /tmp/dump.sql

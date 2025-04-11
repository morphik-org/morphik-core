FROM postgres:15-alpine

# Install build dependencies
RUN apk add --no-cache \
    git \
    build-base \
    clang \
    llvm \
    postgresql-dev

# Clone and build pgvector
RUN git clone --branch v0.6.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make OPTFLAGS="" \
    && make install

# Cleanup
RUN apk del git build-base clang llvm postgresql-dev \
    && rm -rf /pgvector 

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/10-init.sql

# Add script to create user explicitly - will run before init.sql due to ordering
COPY <<EOF /docker-entrypoint-initdb.d/01-create-user.sh
#!/bin/bash
set -e

# This script is only needed for extra safety
# The official postgres image should already handle user creation
# from environment variables, but we'll make it explicit
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER morphik WITH PASSWORD 'morphik' SUPERUSER;
    GRANT ALL PRIVILEGES ON DATABASE morphik TO morphik;
EOSQL
EOF
RUN chmod +x /docker-entrypoint-initdb.d/01-create-user.sh

# Add script to handle optional dump file
RUN echo '#!/bin/sh\nif [ -f /tmp/dump.sql ]; then psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/dump.sql; fi' > /docker-entrypoint-initdb.d/20-restore-dump.sh \
    && chmod +x /docker-entrypoint-initdb.d/20-restore-dump.sh
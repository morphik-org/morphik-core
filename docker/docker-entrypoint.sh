#!/bin/bash
set -e

# Copy default config if none exists
if [ ! -f /app/morphik.toml ]; then
    cp /app/morphik.toml.default /app/morphik.toml
fi

# Function to check PostgreSQL
check_postgres() {
    if [ -n "$POSTGRES_URI" ]; then
        # Extract connection details from POSTGRES_URI, which can be
        # postgresql:// or postgresql+asyncpg://
        URI=${POSTGRES_URI#postgresql*://}
        USER_PASS=${URI%%@*}
        USER=${USER_PASS%:*}
        PASS=${USER_PASS#*:}
        HOST_PORT_DB=${URI#*@}
        HOST_PORT=${HOST_PORT_DB%/*}
        HOST=${HOST_PORT%:*}
        PORT=${HOST_PORT#*:}
        DB=${HOST_PORT_DB#*/}

        echo "POSTGRES_URI: $POSTGRES_URI"
        echo "USER: $USER"
        echo "PASS: $PASS"
        echo "HOST: $HOST"
        echo "PORT: $PORT"
        echo "DB: $DB"

        if [ -z "$PASS" ]; then
            echo "Error: POSTGRES_URI does not contain a password"
            exit 1
        fi

        echo "Waiting for PostgreSQL..."
        max_retries=30
        retries=0
        until PGPASSWORD=$PASS pg_isready -h $HOST -p $PORT -U $USER -d $DB; do
            retries=$((retries + 1))
            if [ $retries -eq $max_retries ]; then
                echo "Error: PostgreSQL did not become ready in time"
                exit 1
            fi
            echo "Waiting for PostgreSQL... (Attempt $retries/$max_retries)"
            sleep 2
        done
        echo "PostgreSQL is ready!"
        
        # Verify database connection
        if ! PGPASSWORD=$PASS psql -h $HOST -p $PORT -U $USER -d $DB -c "SELECT 1" > /dev/null 2>&1; then
            echo "Error: Could not connect to PostgreSQL database"
            exit 1
        fi
        echo "PostgreSQL connection verified!"
    fi
}

# Check PostgreSQL
check_postgres

# Check if command arguments were passed ($# is the number of arguments)
if [ $# -gt 0 ]; then
    # If arguments exist, execute them (e.g., execute "arq core.workers...")
    exec "$@"
else
    # Otherwise, execute the default command (uv run start_server.py)
    exec uv run uvicorn core.api:app --host $HOST --port $PORT --loop asyncio --http auto --ws auto --lifespan auto
fi
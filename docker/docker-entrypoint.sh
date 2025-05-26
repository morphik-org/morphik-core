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

        # Using awk for more robust URI parsing that handles special characters
        eval $(echo "$POSTGRES_URI" | awk -F'postgresql' '{print $2}' | awk '{
            # Remove the +asyncpg if present and get the URI part after ://
            sub(/^[+a-z]*:\/\//, "");
            uri = $0;
            
            # Split into user:pass@host:port/db
            if (match(uri, /([^@]+)@([^\/]+)(\/(.*))?/, m)) {
                # Handle user:password
                user_pass = m[1];
                if (split(user_pass, up, ":") == 2) {
                    printf "USER=\"%s\"\n", up[1];
                    printf "PASS=\"%s\"\n", up[2];
                } else {
                    printf "USER=\"%s\"\n", user_pass;
                    printf "PASS=\"\"\n";
                }
                
                # Handle host:port/db
                host_port_db = m[2] m[3];
                if (split(host_port_db, hpd, "\/") > 1) {
                    host_port = hpd[1];
                    printf "DB=\"%s\"\n", hpd[2];
                } else {
                    host_port = host_port_db;
                    printf "DB=\"postgres\"\n";  # Default database
                }
                
                # Handle host:port
                if (split(host_port, hp, ":") == 2) {
                    printf "HOST=\"%s\"\n", hp[1];
                    printf "PORT=\"%s\"\n", hp[2];
                } else {
                    printf "HOST=\"%s\"\n", host_port;
                    printf "PORT=\"5432\"\n";  # Default port
                }
            }
        }')

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
        # NOTE: preserve stderr for debugging
        if ! PGPASSWORD=$PASS psql -h $HOST -p $PORT -U $USER -d $DB -c "SELECT 1"; then
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
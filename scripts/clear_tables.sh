#!/bin/bash

# Check if environment argument is provided
if [ "$#" -ne 1 ] || { [ "$1" != "docker" ] && [ "$1" != "local" ]; }; then
    echo "Error: Please specify environment (docker or local)"
    echo "Usage: $0 <docker|local>"
    exit 1
fi

ENV=$1
echo "Clearing all tables in the morphik database for $ENV environment..."

# SQL command to truncate all tables
TRUNCATE_CMD="
DO \$\$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
        EXECUTE 'TRUNCATE TABLE ' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END \$\$;
"

if [ "$ENV" = "docker" ]; then
    # Get the Postgres container ID
    POSTGRES_CONTAINER=$(docker ps | grep postgres | awk '{print $1}')

    if [ -z "$POSTGRES_CONTAINER" ]; then
        echo "Error: Postgres container not found!"
        exit 1
    fi

    # Execute the command in docker container
    if ! docker exec -it $POSTGRES_CONTAINER psql -U morphik -d morphik -c "$TRUNCATE_CMD"; then
        echo "Error: Failed to clear tables in docker environment"
        exit 1
    fi
else
    # Load environment variables from .env
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo "Error: .env file not found"
        exit 1
    fi

    # Extract connection details from POSTGRES_URI
    if [ -z "$POSTGRES_URI" ]; then
        echo "Error: POSTGRES_URI not found in .env file"
        exit 1
    fi

    # Parse SQLAlchemy URI to get connection details
    # Remove postgresql+asyncpg:// prefix
    URI=${POSTGRES_URI#postgresql+asyncpg://}
    # Extract username and password
    USER_PASS=${URI%%@*}
    USER=${USER_PASS%:*}
    PASS=${USER_PASS#*:}
    # Extract host, port and database
    HOST_PORT_DB=${URI#*@}
    HOST_PORT=${HOST_PORT_DB%/*}
    HOST=${HOST_PORT%:*}
    PORT=${HOST_PORT#*:}
    DB=${HOST_PORT_DB#*/}

    # Execute the command using parsed connection details
    PGPASSWORD=$PASS psql -h $HOST -p $PORT -U $USER -d $DB -c "$TRUNCATE_CMD"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clear tables in local environment"
        exit 1
    fi
fi

echo "All tables have been cleared successfully."

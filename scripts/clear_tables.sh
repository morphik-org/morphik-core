#!/bin/bash

echo "Clearing all tables in the databridge database..."

# Get the Postgres container ID
POSTGRES_CONTAINER=$(docker ps | grep postgres | awk '{print $1}')

if [ -z "$POSTGRES_CONTAINER" ]; then
    echo "Error: Postgres container not found!"
    exit 1
fi

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

# Execute the command
docker exec -it $POSTGRES_CONTAINER psql -U databridge -d databridge -c "$TRUNCATE_CMD"

echo "All tables have been cleared successfully!" 
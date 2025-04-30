set -e

if [ ! -f /app/morphik.toml ]; then
    cp /app/morphik.toml.default /app/morphik.toml
fi

check_postgres() {
    if [ -n "$POSTGRES_URI" ]; then
        echo "Waiting for PostgreSQL..."
        max_retries=30
        retries=0
        
        if [[ "$POSTGRES_URI" =~ .*@([^:]+):[0-9]+/([^?]+).* ]]; then
            PG_HOST="${BASH_REMATCH[1]}"
            PG_DB="${BASH_REMATCH[2]}"
        else
            PG_HOST="postgres"
            PG_DB="morphik"
        fi
        
        PG_USER="morphik"
        
        if [[ "$PG_HOST" != "postgres" ]]; then
            echo "Using external PostgreSQL at $PG_HOST"
            until PGPASSWORD=$PGPASSWORD psql "$POSTGRES_URI" -c "SELECT 1" > /dev/null 2>&1; do
                retries=$((retries + 1))
                if [ $retries -eq $max_retries ]; then
                    echo "Error: PostgreSQL did not become ready in time"
                    exit 1
                fi
                echo "Waiting for PostgreSQL... (Attempt $retries/$max_retries)"
                sleep 2
            done
        else
            until PGPASSWORD=$PGPASSWORD pg_isready -h $PG_HOST -U $PG_USER -d $PG_DB; do
                retries=$((retries + 1))
                if [ $retries -eq $max_retries ]; then
                    echo "Error: PostgreSQL did not become ready in time"
                    exit 1
                fi
                echo "Waiting for PostgreSQL... (Attempt $retries/$max_retries)"
                sleep 2
            done
            
            if ! PGPASSWORD=$PGPASSWORD psql -h $PG_HOST -U $PG_USER -d $PG_DB -c "SELECT 1" > /dev/null 2>&1; then
                echo "Error: Could not connect to PostgreSQL database"
                exit 1
            fi
        fi
        echo "PostgreSQL connection verified!"
    fi
}

check_postgres

if [ $# -gt 0 ]; then
    exec "$@"
else
    exec uvicorn core.api:app --host $HOST --port $PORT --loop asyncio --http auto --ws auto --lifespan auto
fi

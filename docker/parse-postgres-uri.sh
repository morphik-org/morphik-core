#!/bin/bash
set -euo pipefail

# Usage: eval $(parse-postgres-uri.sh "postgresql://user:pass@host:port/db")

# Make sure all variables are ultimately set to avoid downstream failures in
# scripts that require defined variables.
echo 'USER=""'
echo 'PASS=""'
echo 'HOST=""'
echo 'PORT=""'
echo 'DB=""'

echo "$1" | awk -F'postgresql' '{print $2}' | awk '{
    # Remove the +asyncpg if present and get the URI part after ://
    sub(/^[+a-z]*:\/\//, "");
    uri = $0;

    # Remove query parameters
    sub(/\?.*$/, "", uri);

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
        if (split(host_port_db, hpd, "/") > 1) {
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
}' | perl -pe 's/%([0-9a-fA-F]{2})/chr(hex($1))/ge' || echo "POSTGRES_URI_PARSE_FAILURE=1"